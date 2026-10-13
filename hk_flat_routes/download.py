"""Data acquisition with local caching.

Everything is cached under ``data/raw`` and re-downloaded only when missing
(or when ``force=True``), so re-running the pipeline never re-fetches the
14.1 GB of lidar or re-scans the Overture release.  Nothing reaches the cache
that is not the kind of file that was asked for: the government portals
answer some failures with HTTP 200 and a page of HTML, and one of them with
HTTP 200 and an error in JSON, so every download is checked before it is
kept (``_download_file``), and a cached file that fails the same check is
fetched again.

The Overture read is the interesting part: the segments of the
transportation theme are 72 GB spread over 128 Parquet files.  Each file
carries per-row-group statistics on the ``bbox`` struct column, and Overture
writes rows in spatial order, so we read the 128 footers (cheap, a couple of
range requests each), keep only the row groups whose bounding box intersects
Hong Kong, and read just those.  In practice 19 row groups in a single file
cover the territory.  Overture hosts only its latest few releases, so the
extracts are the one part of ``data/raw`` that cannot be fetched again for
ever (see ``sources.OVERTURE_RELEASE``); each records the release it was
read from, and ``download`` says when a cached one is another release's.
"""
from __future__ import annotations

import os
import re
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np
import requests

from . import sources
from .config import CRS_PROJECTED, RAW_DIR, STUDY_BBOX
from .utils import configure_gdal_for_proxy, get_logger, human_bytes, progress, step

log = get_logger("hk_flat_routes.download")

_S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

DEM_DIR = RAW_DIR / "dem"
#: Grid index of the DEM tiles (WGS84 polygons with a download URL each).
DEM_INDEX_GEOJSON = RAW_DIR / "dem_index_2020.geojson"
SEGMENTS_PARQUET = RAW_DIR / "overture_segments_hk.parquet"
CONNECTORS_PARQUET = RAW_DIR / "overture_connectors_hk.parquet"
PLACES_PARQUET = RAW_DIR / "overture_places_hk.parquet"
#: Lands Department gazetteer: the points, and the names that go with them.
PLACE_NAME_POINTS_GEOJSON = RAW_DIR / "landsd_place_name_points.geojson"
PLACE_NAME_NAMES_GEOJSON = RAW_DIR / "landsd_place_name_names.geojson"
#: Lands Department iGeoCom points of interest (one CSV inside a zip).
IGEOCOM_ZIP = RAW_DIR / "landsd_igeocom_csv.zip"
#: Lands Department building names (an attribute table, no geometry).
BUILDING_NAMES_GEOJSON = RAW_DIR / "landsd_building_names.geojson"
#: Address Lookup Service bulk file: one GeoJSON per district in a zip.
ALS_ZIP = RAW_DIR / "als_addresses_geojson.zip"
#: The address source under its upstream name.  Upstream read Overture's
#: addresses theme into a Parquet file; Hong Kong's addresses come from the
#: government's own file, and modules that import this name get that file.
ADDRESSES_PARQUET = ALS_ZIP
#: Transport Department stop lists for trams, the Peak Tram and ferries.
TD_STOPS_JSON = {mode: RAW_DIR / f"td_stops_{mode}.json"
                 for mode in ("tram", "peak_tram", "ferry")}
#: Overture base theme (OpenStreetMap): mapped parks, schools, stations...
BASE_PARQUETS = {typ: RAW_DIR / f"overture_{typ}_hk.parquet"
                 for typ in ("land_use", "infrastructure", "land")}
#: Every statutory planning scheme area, as published; the ones the analysis
#: uses are chosen in ``neighborhoods.load_neighborhoods``.
NEIGHBORHOODS_GEOJSON = RAW_DIR / "ozp_plan_scheme_areas.geojson"
#: Lands Department 5 m terrain model, for the cross-check in ``validate``.
DTM_5M_ZIP = RAW_DIR / "Whole_HK_DTM_5m.zip"
#: Overture division areas: the SAR boundary and the eighteen districts.
DIVISIONS_PARQUET = RAW_DIR / "overture_division_area_hk.parquet"

#: Columns pulled from the Overture segment table. Everything unused is left
#: on the server -- the nested route/destination columns are large.
SEGMENT_COLUMNS = [
    "id", "names", "subtype", "class", "subclass", "connectors",
    "road_flags", "access_restrictions", "road_surface", "speed_limits",
    "level_rules", "geometry", "bbox", "sources",
]
CONNECTOR_COLUMNS = ["id", "geometry", "bbox"]
#: The places schema of this release has no ``categories`` column (upstream's
#: pinned release did): the category is ``taxonomy.primary``, with
#: ``basic_category`` as the coarser fallback.
PLACE_COLUMNS = ["id", "names", "taxonomy", "basic_category", "operating_status",
                 "sources", "confidence", "geometry", "bbox"]
BASE_COLUMNS = ["id", "names", "subtype", "class", "geometry", "bbox"]
DIVISION_COLUMNS = ["id", "names", "subtype", "class", "country", "geometry", "bbox"]


# --------------------------------------------------------------------------
# generic helpers
# --------------------------------------------------------------------------
def _list_s3_keys(bucket: str, prefix: str, suffix: str = ".parquet") -> list[str]:
    """List keys under a public S3 prefix via the REST list-objects-v2 API."""
    keys: list[str] = []
    token = None
    while True:
        params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        r = requests.get(bucket + "/", params=params, timeout=120)
        r.raise_for_status()
        root = ET.fromstring(r.text)
        for c in root.findall("s3:Contents", _S3_NS):
            key = c.find("s3:Key", _S3_NS).text
            if key.endswith(suffix):
                keys.append(key)
        if root.findtext("s3:IsTruncated", default="false", namespaces=_S3_NS) == "true":
            token = root.findtext("s3:NextContinuationToken", namespaces=_S3_NS)
        else:
            break
    return keys


# What a download must be before it is kept.  Each check raises ``IOError``
# with the reason; ``_download_file`` runs it on the temporary file, so a
# body that fails is retried like a dropped connection and is never cached.
def _head(path: Path) -> str:
    """The first bytes of a file, for an error message."""
    with open(path, "rb") as fh:
        return f"it begins {fh.read(40)!r}"


def _is_zip(path: Path) -> None:
    import zipfile

    if not zipfile.is_zipfile(path):
        raise IOError(f"not a zip file: {_head(path)}")


def _is_geojson(path: Path) -> None:
    """A feature collection with something in it.

    Parsed whole, not judged by its first byte: the planning-areas server
    reports a failed query as HTTP 200 with ``{"error": {"code": 400, ...}}``,
    which begins like the data.  Some of the files carry a byte-order mark.
    """
    import json

    try:
        doc = json.loads(path.read_bytes().decode("utf-8-sig"))
    except ValueError:                  # not UTF-8, or not JSON
        raise IOError(f"not JSON: {_head(path)}") from None
    if not isinstance(doc, dict):
        raise IOError(f"not a feature collection: {_head(path)}")
    if "error" in doc:
        raise IOError(f"the server answered with an error: {str(doc['error'])[:160]}")
    if not isinstance(doc.get("features"), list) or not doc["features"]:
        raise IOError(f"a feature collection with no features: {_head(path)}")


def _is_xml(path: Path) -> None:
    """Well-formed XML that is not a web page."""
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        raise IOError(f"not XML: {_head(path)}") from None
    if root.tag.rsplit("}", 1)[-1].lower() == "html":
        raise IOError(f"a web page, not the data: {_head(path)}")


def _rejects(validate, path: Path) -> str | None:
    """Why ``path`` fails ``validate``; None when it passes or nothing is asked."""
    if validate is None:
        return None
    try:
        validate(path)
    except OSError as exc:
        return str(exc)
    return None


def _download_file(url: str, dest: Path, force: bool = False,
                   retries: int = 5, quiet: bool = False, validate=None) -> Path:
    """Stream a URL to disk with retries and an atomic rename.

    ``validate`` is one of the checks above.  It runs before the temporary
    file takes the destination's place, so a maintenance page sent with
    HTTP 200 is one more failed attempt and not a cached tile that
    ``build-network`` trips over an hour later; a cached file is held to the
    same check and fetched again when it fails.  ``quiet`` drops the
    per-file progress bar and log lines, for the DEM, which arrives as a
    few thousand small tiles.
    """
    if dest.exists() and not force:
        why = _rejects(validate, dest)
        if why is None:
            if not quiet:
                log.info("cached %s (%s)", dest.name, human_bytes(dest.stat().st_size))
            return dest
        log.warning("cached %s is not what it should be (%s); fetching it "
                    "again", dest.name, why)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    delay = 2.0
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=(30, 300)) as r:
                r.raise_for_status()
                total = int(r.headers.get("Content-Length") or 0)
                written = 0
                with open(tmp, "wb") as fh:
                    chunks = r.iter_content(chunk_size=1 << 20)
                    bar = chunks if quiet else progress(
                        chunks, desc=f"  {dest.name}",
                        total=(total >> 20) + 1 if total else None, unit="MB")
                    for chunk in bar:
                        fh.write(chunk)
                        written += len(chunk)
            if total and written < total:
                raise IOError(f"short read: {written} of {total} bytes")
            if validate is not None:
                validate(tmp)
            tmp.replace(dest)
            if not quiet:
                log.info("downloaded %s (%s)", dest.name,
                         human_bytes(dest.stat().st_size))
            return dest
        except Exception as exc:  # network flakiness is expected
            tmp.unlink(missing_ok=True)
            if attempt == retries:
                raise
            log.warning("attempt %d/%d for %s failed (%s); retrying in %.0fs",
                        attempt, retries, dest.name, exc, delay)
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


# --------------------------------------------------------------------------
# Overture
# --------------------------------------------------------------------------
def _bbox_stat_columns(metadata) -> dict[str, int]:
    cols = {c["path_in_schema"]: i
            for i, c in enumerate(metadata.row_group(0).to_dict()["columns"])}
    return {k: cols[k] for k in ("bbox.xmin", "bbox.xmax", "bbox.ymin", "bbox.ymax")}


def _matching_row_groups(metadata, bbox) -> list[int]:
    lon_min, lon_max, lat_min, lat_max = bbox
    ix = _bbox_stat_columns(metadata)
    out = []
    for rg in range(metadata.num_row_groups):
        g = metadata.row_group(rg)
        xmax = g.column(ix["bbox.xmax"]).statistics.max
        xmin = g.column(ix["bbox.xmin"]).statistics.min
        ymax = g.column(ix["bbox.ymax"]).statistics.max
        ymin = g.column(ix["bbox.ymin"]).statistics.min
        if xmax >= lon_min and xmin <= lon_max and ymax >= lat_min and ymin <= lat_max:
            out.append(rg)
    return out


def _overture_releases() -> list[str]:
    """The releases the Overture bucket hosts now (it keeps the latest few)."""
    r = requests.get(sources.OVERTURE_BUCKET + "/", timeout=120,
                     params={"list-type": "2", "prefix": "release/", "delimiter": "/"})
    r.raise_for_status()
    prefixes = ET.fromstring(r.text).findall("s3:CommonPrefixes/s3:Prefix", _S3_NS)
    return sorted(p.text.split("/")[1] for p in prefixes)


def _release_in(prefix: str) -> str:
    """The release an Overture prefix belongs to ("release/<name>/theme=...")."""
    m = re.match(r"release/([^/]+)/", prefix)
    return m.group(1) if m else sources.OVERTURE_RELEASE


#: Where an extract records the release it was read from: a key of the
#: Parquet file's own schema metadata, beside the "geo" key Overture writes.
_RELEASE_KEY = b"hk_flat_routes:overture_release"


def _extract_release(path: Path) -> str | None:
    """The release an extract on disk says it was read from, or None.

    None for a file that does not say: one written before the release was
    recorded, which every extract of the first published build is.
    """
    import pyarrow.parquet as pq

    try:
        said = (pq.read_schema(path).metadata or {}).get(_RELEASE_KEY)
    except Exception:                   # not a Parquet file: nothing to read
        return None
    return said.decode("utf-8") if said else None


def _no_overture_files(prefix: str, overture_type: str) -> str:
    """The error for an Overture prefix that lists no files, with the remedy.

    Overture hosts only its latest few releases, and one that has been
    retired lists as empty, not as missing.  Left to run on, the read used
    to end in "no Overture segment row groups intersect (113.82, ...)",
    which says nothing of what happened or what to do.
    """
    release = _release_in(prefix)
    try:
        hosted = _overture_releases()
    except Exception:                   # the listing is a courtesy; do without
        hosted = []
    if release in hosted:
        return (f"Overture release {release} is hosted but lists no "
                f"{overture_type} files under {prefix}: the layout of the "
                f"release differs from the one this code reads")
    now = f" Hosted now: {', '.join(hosted)}." if hosted else ""
    return (f"Overture release {release} is no longer hosted, so its "
            f"{overture_type} files cannot be downloaded.{now} Set "
            f"HKFR_OVERTURE_RELEASE to a hosted release to build from that "
            f"one (it is different data: the published figures come from "
            f"{sources.OVERTURE_RELEASE_PINNED}), or put a kept copy of the "
            f"overture_*_hk.parquet extracts into {RAW_DIR}")


def _cached_extract(dest: Path) -> bool:
    """Whether an Overture extract is on disk already, saying so in the log.

    A cached extract is kept whatever ``HKFR_OVERTURE_RELEASE`` says: the
    release it was read from may be gone.  So that a run cannot pass in
    silence for a build from the release that was asked for, the log says
    when the two differ.  An extract records its release
    (``_extract_release``), and one that names another is reported whether
    the release in force is the variable's or the pinned one: an extract
    fetched under the variable is still there, and still that release's,
    when the variable is unset again.  An extract that names none is taken
    for the pinned release's, as those of the published build are, and is
    reported only when the variable asks for another.
    """
    if not dest.exists():
        return False
    log.info("cached %s (%s)", dest.name, human_bytes(dest.stat().st_size))
    have, want = _extract_release(dest), sources.OVERTURE_RELEASE
    pinned = want == sources.OVERTURE_RELEASE_PINNED
    asked = (f"the pinned release {want}" if pinned
             else f"release {want} (HKFR_OVERTURE_RELEASE)")
    if have is None and not pinned:
        log.warning("%s is used as cached, whatever release it was read "
                    "from; to read it from %s move the file away or pass "
                    "--force", dest.name, asked)
    elif have is not None and have != want:
        log.warning("%s is used as cached, and it was read from release %s; "
                    "to read it from %s move the file away or pass --force",
                    dest.name, have, asked)
    return True


def _read_overture_type(overture_type: str, columns: list[str], dest: Path,
                        bbox=STUDY_BBOX, force: bool = False,
                        prefix: str = sources.OVERTURE_PREFIX) -> Path:
    """Row-group-pruned read of one Overture type (any theme)."""
    import fsspec
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not force and _cached_extract(dest):
        return dest

    keys = _list_s3_keys(sources.OVERTURE_BUCKET, f"{prefix}/type={overture_type}/")
    if not keys:
        raise RuntimeError(_no_overture_files(prefix, overture_type))
    log.info("overture %s: %d parquet files in release %s",
             overture_type, len(keys), _release_in(prefix))

    fs = fsspec.filesystem("https")

    def scan(key: str):
        url = f"{sources.OVERTURE_BUCKET}/{key}"
        with fs.open(url, block_size=8 << 20) as fh:
            md = pq.ParquetFile(fh).metadata
            return key, _matching_row_groups(md, bbox)

    with step(f"scanning {len(keys)} {overture_type} footers for the study bbox", log):
        with ThreadPoolExecutor(max_workers=16) as pool:
            scanned = list(pool.map(scan, keys))
    hits = [(k, rgs) for k, rgs in scanned if rgs]
    n_rg = sum(len(r) for _, r in hits)
    log.info("overture %s: %d row group(s) in %d file(s) intersect the study area",
             overture_type, n_rg, len(hits))
    if not hits:
        raise RuntimeError(f"no Overture {overture_type} row groups intersect {bbox}")

    tables = []
    for key, rgs in progress(hits, desc=f"  reading {overture_type}", unit="file"):
        url = f"{sources.OVERTURE_BUCKET}/{key}"
        with fs.open(url, block_size=16 << 20) as fh:
            t = pq.ParquetFile(fh).read_row_groups(rgs, columns=columns)
        tables.append(t)
    table = pa.concat_tables(tables)

    # Row groups are coarse; clip precisely to the study bbox.
    bb = table.column("bbox").combine_chunks()
    xmin = np.asarray(bb.field("xmin")); xmax = np.asarray(bb.field("xmax"))
    ymin = np.asarray(bb.field("ymin")); ymax = np.asarray(bb.field("ymax"))
    lon_min, lon_max, lat_min, lat_max = bbox
    keep = ((xmax >= lon_min) & (xmin <= lon_max)
            & (ymax >= lat_min) & (ymin <= lat_max))
    table = table.filter(pa.array(keep))
    log.info("overture %s: %d features inside the study bbox", overture_type,
             table.num_rows)
    # the file says which release it is, for ``_cached_extract``
    table = table.replace_schema_metadata({
        **(table.schema.metadata or {}), _RELEASE_KEY: _release_in(prefix).encode("utf-8")})

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(dest)
    log.info("wrote %s (%s)", dest.name, human_bytes(dest.stat().st_size))
    return dest


def download_street_network(force: bool = False) -> tuple[Path, Path]:
    seg = _read_overture_type("segment", SEGMENT_COLUMNS, SEGMENTS_PARQUET, force=force)
    con = _read_overture_type("connector", CONNECTOR_COLUMNS, CONNECTORS_PARQUET,
                              force=force)
    return seg, con


def download_land_use(force: bool = False) -> Path:
    """Overture's land-use polygons, which the street network is built with.

    ``network.closed_land`` takes the container terminals, the power
    stations and military land from them and closes the service roads
    inside, so this extract decides which streets are routable and is
    fetched with the network, where a failure stops the run.  The place
    search reads the same file for its parks and estates, which is why it
    is kept with the base theme's other two (``BASE_PARQUETS``).
    """
    return _read_overture_type("land_use", BASE_COLUMNS, BASE_PARQUETS["land_use"],
                               force=force, prefix=sources.OVERTURE_BASE_PREFIX)


#: The certificate the iGeoCom host leaves out of its chain; where it came
#: from and how it was checked is in ``vendor/README.md``.
IGEOCOM_INTERMEDIATE_PEM = (Path(__file__).resolve().parent / "vendor"
                            / "hongkongpost-ecert-ssl-ca-3-17.pem")


def _ca_bundle_with(extra_pem: Path, directory: Path) -> Path:
    """The CA bundle ``requests`` verifies against, with one certificate added.

    ``requests`` takes its roots from ``REQUESTS_CA_BUNDLE`` or
    ``CURL_CA_BUNDLE`` when one is set and from certifi otherwise; the same
    file is extended here, so a proxy's own root goes on working.
    """
    base = os.environ.get("REQUESTS_CA_BUNDLE") or os.environ.get("CURL_CA_BUNDLE")
    if not base or not Path(base).is_file():
        base = requests.certs.where()
    out = directory / "ca-bundle.pem"
    out.write_bytes(Path(base).read_bytes().rstrip() + b"\n\n" + extra_pem.read_bytes())
    return out


def _download_igeocom(dest: Path, force: bool = False, retries: int = 6) -> Path:
    """The iGeoCom zip, from whichever of the host's two servers answers.

    The download host balances between two back ends and, when this was
    written, one of them answered every request with a redirect to an
    internal error page (``hkms.landsd.hksarg``, a name that does not
    resolve outside the government network).  Which server answers is
    fixed by a cookie, so odd attempts ask for the one that worked and even
    attempts let the balancer choose, in case the fault has since moved.
    The redirect is not followed, and whatever arrives must be a zip before
    it is kept.  This is why the file does not go through
    ``_download_file``, which cannot send a cookie.

    The host has a second fault: it presents its certificate and the root
    but not the intermediate between them (Hongkong Post e-Cert SSL CA 3 -
    17), so Python, which checks against a fixed bundle of roots, cannot
    build the chain.  Browsers and some operating systems' TLS libraries
    fetch the missing link themselves.  Verification is never switched off
    here.  The intermediate is shipped in ``vendor/``, taken from the
    address the host's own certificate gives for its issuer and checked
    against the Hongkong Post root that certifi carries; when the chain as
    sent does not verify, the request is made again against a bundle with
    that one certificate added.  Should the host have moved to another
    intermediate by then, the request is handed to the system's ``curl``,
    which on macOS verifies against the system trust store, and where that
    fails too the source is reported as unavailable.
    """
    import shutil
    import subprocess
    import tempfile

    def by_python(cookies: dict | None, verify) -> None:
        r = requests.get(sources.IGEOCOM_URL, cookies=cookies, verify=verify,
                         allow_redirects=False, timeout=(30, 300))
        if r.status_code != 200:
            raise IOError(f"HTTP {r.status_code}")
        tmp.write_bytes(r.content)

    def by_curl(cookies: dict | None) -> None:
        curl = shutil.which("curl")
        if not curl:
            raise IOError("certificate chain incomplete and no curl to fall back on")
        cmd = [curl, "--fail", "--silent", "--show-error", "--max-time", "300",
               "--output", str(tmp)]
        for k, v in (cookies or {}).items():
            cmd += ["--cookie", f"{k}={v}"]
        done = subprocess.run(cmd + [sources.IGEOCOM_URL], capture_output=True,
                              text=True, timeout=330)
        if done.returncode:
            raise IOError(f"curl: {done.stderr.strip() or done.returncode}")

    if dest.exists() and not force and _rejects(_is_zip, dest) is None:
        log.info("cached %s (%s)", dest.name, human_bytes(dest.stat().st_size))
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    delay = 2.0
    with tempfile.TemporaryDirectory() as scratch:
        for attempt in range(1, retries + 1):
            cookies = sources.IGEOCOM_COOKIE if attempt % 2 else None
            try:
                try:
                    by_python(cookies, True)
                except requests.exceptions.SSLError:
                    log.info("%s: the host's certificate chain is incomplete; "
                             "adding the intermediate it leaves out", dest.name)
                    try:
                        by_python(cookies, str(_ca_bundle_with(
                            IGEOCOM_INTERMEDIATE_PEM, Path(scratch))))
                    except (requests.exceptions.SSLError, FileNotFoundError):
                        log.info("%s: the chain still does not verify; asking "
                                 "curl", dest.name)
                        by_curl(cookies)
                _is_zip(tmp)
                tmp.replace(dest)
                log.info("downloaded %s (%s)", dest.name,
                         human_bytes(dest.stat().st_size))
                return dest
            except Exception as exc:
                tmp.unlink(missing_ok=True)
                if attempt == retries:
                    raise
                log.warning("attempt %d/%d for %s failed (%s); retrying in %.0fs",
                            attempt, retries, dest.name, exc, delay)
                time.sleep(delay)
                delay = min(delay * 2, 30.0)
    raise RuntimeError("unreachable")


#: What the route page's search loses without each of its sources, for the
#: warnings of ``download``.
PLACE_SOURCES = {
    "overture_base": "mapped parks, estates, schools and piers",
    "overture_places": "shops, restaurants and hotels",
    "place_names": "the names of districts, towns, villages and hills",
    "igeocom": "MTR stations and all their exits, Light Rail stops, malls, "
               "markets and scenic spots",
    "building_names": "building names",
    "addresses": "street addresses and estate names",
    "transit_stops": "tram stops and ferry piers",
}


def download_places(force: bool = False) -> dict[str, list[Path]]:
    """Everything behind the route page's offline search.

    None of it is needed by the analysis, the land-use polygons apart
    (``download_land_use``).  Upstream read two Overture themes; Hong Kong
    needs more, because Overture's place names there are in one language
    and its addresses are an English-only copy of a government file that
    is bilingual at the source.  So the names come from the government's
    own lists as well:

    * Overture base theme (OpenStreetMap): mapped parks, estates, stations;
    * Overture places: shops, restaurants and hotels only;
    * the Lands Department gazetteer, iGeoCom points of interest and
      building names;
    * the Address Lookup Service bulk file, for street addresses and for
      estate names;
    * the Transport Department's tram, Peak Tram and ferry stops.

    Each source is fetched on its own: one that fails is logged and
    skipped, and ``places.build_places`` works with whatever arrived (the
    route page falls back to junction search when nothing did).  Returns
    the files now on disk, by source (upstream returned a pair of paths,
    which cannot say which of seven sources arrived).
    """
    def get(url: str, dest: Path, check) -> Path:
        return _download_file(url, dest, force=force, validate=check)

    fetchers = {
        # the land-use extract is the street network's as well, and is
        # forced with it (``download_land_use``), not a second time here
        "overture_base": lambda: [
            _read_overture_type(typ, BASE_COLUMNS, dest,
                                force=force and typ != "land_use",
                                prefix=sources.OVERTURE_BASE_PREFIX)
            for typ, dest in BASE_PARQUETS.items()],
        "overture_places": lambda: [_read_overture_type(
            "place", PLACE_COLUMNS, PLACES_PARQUET, force=force,
            prefix=sources.OVERTURE_PLACES_PREFIX)],
        "place_names": lambda: [
            get(sources.PLACE_NAME_POINTS_URL, PLACE_NAME_POINTS_GEOJSON, _is_geojson),
            get(sources.PLACE_NAME_NAMES_URL, PLACE_NAME_NAMES_GEOJSON, _is_geojson)],
        "igeocom": lambda: [_download_igeocom(IGEOCOM_ZIP, force=force)],
        "building_names": lambda: [
            get(sources.BUILDING_NAME_URL, BUILDING_NAMES_GEOJSON, _is_geojson)],
        "addresses": lambda: [get(sources.ALS_URL, ALS_ZIP, _is_zip)],
        "transit_stops": lambda: [
            get(url, TD_STOPS_JSON[mode], _is_geojson)
            for mode, url in sources.TD_STOPS_URLS.items()],
    }
    out: dict[str, list[Path]] = {}
    for key, fetch in fetchers.items():
        try:
            out[key] = fetch()
        except Exception as exc:        # every source is optional on its own
            log.warning("place source %s unavailable (%s); the search index "
                        "will be built without %s", key, exc, PLACE_SOURCES[key])
    return out


def download_boundary(force: bool = False) -> Path:
    """Division areas of the study bbox, kept for Hong Kong only.

    The SAR polygon (land and territorial waters) is what separates Hong
    Kong's streets from Shenzhen's; the districts come along for labelling.
    """
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    if not force and _cached_extract(DIVISIONS_PARQUET):
        return DIVISIONS_PARQUET
    tmp = DIVISIONS_PARQUET.with_name("overture_division_area_bbox.parquet")
    _read_overture_type("division_area", DIVISION_COLUMNS, tmp, force=True,
                        prefix=sources.OVERTURE_DIVISIONS_PREFIX)
    table = pq.read_table(tmp)
    table = table.filter(pc.equal(table.column("country"), "HK"))
    # written beside the file and moved into place, like every other
    # download: a write cut short would otherwise be the cached boundary
    part = DIVISIONS_PARQUET.with_suffix(".part")
    pq.write_table(table, part, compression="zstd")
    part.replace(DIVISIONS_PARQUET)
    tmp.unlink(missing_ok=True)
    log.info("wrote %s (%d Hong Kong division areas)", DIVISIONS_PARQUET.name,
             table.num_rows)
    return DIVISIONS_PARQUET


# --------------------------------------------------------------------------
# Elevation
# --------------------------------------------------------------------------
def download_dem_index(force: bool = False) -> Path:
    """The grid index of the 2020 LiDAR DTM: one polygon and URL per tile."""
    return _download_file(sources.DTM_INDEX_URL, DEM_INDEX_GEOJSON, force=force,
                          validate=_is_geojson)


def _tile_filename(name) -> str:
    """A tile's file name as the index gives it, or a ``ValueError``.

    The name is joined to ``data/raw/dem`` and the index is whatever the
    server sent, so one with a path in it ("../../x.zip") would be written
    outside that directory.  The 3,253 names of the real index are all
    plain file names ending in ".zip".
    """
    name = str(name)
    if (PurePosixPath(name).name != name or PureWindowsPath(name).name != name
            or name.startswith(".") or not name.lower().endswith(".zip")):
        raise ValueError(f"DEM tile index: {name!r} is not a plain .zip file name")
    return name


def dem_tile_index(path: Path | None = None) -> list[dict]:
    """Tiles of the 0.5 m DTM: name, file name, URL and HK1980 Grid bounds.

    The index is published in WGS84; the tiles themselves are 750 m x 600 m
    sheets of the 1:1000 map series, aligned to whole metres of the HK1980
    Grid, so projecting the corners back and rounding recovers the exact
    sheet bounds (checked against the GeoTIFFs' own transforms).
    """
    import json

    from pyproj import Transformer

    tr = Transformer.from_crs("EPSG:4326", CRS_PROJECTED, always_xy=True)
    out = []
    text = Path(path or DEM_INDEX_GEOJSON).read_bytes().decode("utf-8-sig")
    for feat in json.loads(text)["features"]:
        props = feat["properties"]
        ring = feat["geometry"]["coordinates"][0]
        xs, ys = tr.transform([c[0] for c in ring], [c[1] for c in ring])
        out.append({
            "name": props["TILE_NAME"],
            "filename": _tile_filename(props["Filename"]),
            "url": props["URL"],
            "bounds": (float(round(min(xs))), float(round(min(ys))),
                       float(round(max(xs))), float(round(max(ys)))),
        })
    return out


def download_dem(force: bool = False, workers: int = 8) -> list[Path]:
    """Fetch every tile of the CEDD 2020 LiDAR DTM (14.1 GB in 3,253 zips).

    The whole territory is taken, not only the tiles under a street: the
    hillshade and the lowland analysis want the hills between the streets
    too, and the set then does not depend on the street extract. Tiles are
    fetched nearest Central first, so an interrupted run leaves the urban
    core usable. The server caps the transfer rate at a few MB/s whatever
    the concurrency, so a cold run takes about an hour; it is resumable.
    """
    download_dem_index(force=force)
    tiles = dem_tile_index()
    cx, cy = 834000.0, 816000.0                      # Central, HK1980 Grid
    tiles.sort(key=lambda t: ((t["bounds"][0] + t["bounds"][2]) / 2 - cx) ** 2
               + ((t["bounds"][1] + t["bounds"][3]) / 2 - cy) ** 2)
    DEM_DIR.mkdir(parents=True, exist_ok=True)

    def cached(t: dict) -> bool:
        # a tile on disk that is not a zip was never the tile
        path = DEM_DIR / t["filename"]
        return path.exists() and _rejects(_is_zip, path) is None

    todo = [t for t in tiles if force or not cached(t)]
    log.info("DEM: %d tiles in the index, %d cached, %d to fetch",
             len(tiles), len(tiles) - len(todo), len(todo))

    def fetch(t: dict) -> Path:
        return _download_file(t["url"], DEM_DIR / t["filename"], force=force,
                              quiet=True, validate=_is_zip)

    if todo:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for _ in progress(pool.map(fetch, todo), desc="  DEM tiles",
                              total=len(todo), unit="tile"):
                pass
    paths = [DEM_DIR / t["filename"] for t in tiles]
    log.info("DEM: %d tiles, %s", len(paths),
             human_bytes(sum(p.stat().st_size for p in paths)))
    return paths


# --------------------------------------------------------------------------
# Neighborhoods
# --------------------------------------------------------------------------
def download_neighborhoods(force: bool = False) -> Path:
    return _download_file(sources.NEIGHBORHOOD_URL, NEIGHBORHOODS_GEOJSON,
                          force=force, validate=_is_geojson)


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------
def download_dtm_5m(force: bool = False) -> Path:
    """The independent 5 m terrain model used only to cross-check the lidar."""
    return _download_file(sources.DTM_5M_URL, DTM_5M_ZIP, force=force,
                          validate=_is_zip)


#: Survey and Mapping Office bench marks: the levelled heights the elevation
#: model is checked against in ``validate``.
BENCHMARKS_XML = RAW_DIR / "smo_bench_marks.xml"


def download_benchmarks(force: bool = False) -> Path:
    """The official bench marks (one small XML file, about 700 kB)."""
    return _download_file(sources.BENCHMARKS_URL, BENCHMARKS_XML, force=force,
                          validate=_is_xml)


def download_all(force: bool = False) -> dict[str, object]:
    """Fetch every source.  The street network with the land use it is
    built with, the boundary, the planning areas and the terrain are
    required and a failure stops the run; the rest is optional, and what
    could not be fetched is listed again at the end, after the hour of
    terrain tiles has scrolled the first warning away (``out["skipped"]``).
    """
    configure_gdal_for_proxy()
    out: dict[str, object] = {}
    skipped: list[str] = []
    with step("downloading street network (Overture)", log):
        out["segments"], out["connectors"] = download_street_network(force=force)
        out["land_use"] = download_land_use(force=force)
    with step("downloading the SAR boundary (Overture divisions)", log):
        out["boundary"] = download_boundary(force=force)
    with step("downloading neighborhood boundaries", log):
        out["neighborhoods"] = download_neighborhoods(force=force)
    with step("downloading places for the search index", log):
        try:
            out["places"] = download_places(force=force)
        except Exception as exc:  # optional: the route page can do without
            log.warning("place data unavailable (%s); the route page "
                        "will offer junction search only", exc)
            out["places"] = {}
        skipped += [f"{key}: the route page's search will lack {what}"
                    for key, what in PLACE_SOURCES.items()
                    if key not in out["places"]]
    with step("downloading the cross-check terrain model", log):
        try:
            out["dtm_5m"] = download_dtm_5m(force=force)
        except Exception as exc:  # optional: validation reports it as not run
            log.warning("5 m terrain model unavailable (%s)", exc)
            skipped.append("dtm_5m: `validate` will not run its cross-check "
                           "against the 5 m terrain model")
    with step("downloading the official bench marks", log):
        try:
            out["benchmarks"] = download_benchmarks(force=force)
        except Exception as exc:  # optional: validation reports it as not run
            log.warning("bench marks unavailable (%s)", exc)
            skipped.append("bench_marks: `validate` will not run its checks "
                           "against the bench marks")
    with step("downloading the LiDAR terrain model (CEDD 2020, 0.5 m)", log):
        out["dem"] = download_dem(force=force)
    if skipped:
        log.warning("download finished without %d optional source(s); the "
                    "reasons are logged above, and running `download` again "
                    "retries them:", len(skipped))
        for line in skipped:
            log.warning("  %s", line)
    out["skipped"] = skipped
    return out
