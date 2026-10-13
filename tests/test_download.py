"""Tests for the downloads: nothing is cached that is not the data.

The servers are stand-ins on the loopback interface (or a patched
``requests.get`` where TLS matters); nothing here touches the network or
the project's ``data/raw``.
"""
import hashlib
import http.server
import io
import json
import ssl
import threading
import zipfile
from pathlib import Path

import pytest
import requests

from hk_flat_routes import download as D
from hk_flat_routes import sources

HTML = (b"<html><head><title>System maintenance</title></head>"
        b"<body>Please try again later.</body></html>")
#: What the planning-areas server sends for a query it cannot run: HTTP 200.
ARCGIS_ERROR = (b'{"error":{"code":400,"message":"Unable to complete operation.",'
                b'"details":["Unable to perform query operation."]}}')
#: The same page as browsers are sent it: not well-formed, so not XML at all.
HTML5 = b"<!DOCTYPE html><html><head><meta charset=utf-8><title>Busy</title><body><p>Later"
XML = b'<?xml version="1.0" encoding="UTF-8"?>\n<tables><row STN_NO="1"/></tables>'
XHTML = (b'<?xml version="1.0" encoding="UTF-8"?>\n'
         b'<html xmlns="http://www.w3.org/1999/xhtml"><body>Maintenance</body></html>')


def _zip_bytes(name: str = "tile.tif", body: bytes = b"0" * 64) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(name, body)
    return buf.getvalue()


def _collection(features) -> bytes:
    return json.dumps({"type": "FeatureCollection", "features": features}).encode()


@pytest.fixture
def server():
    """A loopback server whose answers a test sets: path -> list of bodies.

    Each request for a path takes the next body in its list and the last
    one for ever after, so "a page first, then the file" can be played.
    """
    routes: dict[str, list[bytes]] = {}
    hits: dict[str, int] = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            bodies = routes.get(self.path)
            if bodies is None:
                self.send_error(404)
                return
            n = hits.get(self.path, 0)
            hits[self.path] = n + 1
            body = bodies[min(n, len(bodies) - 1)]
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    srv.routes, srv.hits = routes, hits
    srv.url = f"http://127.0.0.1:{srv.server_address[1]}"
    yield srv
    srv.shutdown()


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    """The retries back off for 2, 4, 8 and 16 seconds; not in a test."""
    monkeypatch.setattr("time.sleep", lambda s: None)


# ------------------------------------------------- the checks themselves
@pytest.mark.parametrize("kind, good, bad", [
    ("zip", _zip_bytes(), HTML),
    ("geojson", _collection([{"type": "Feature"}]), HTML),
    ("geojson", b"\xef\xbb\xbf" + _collection([{"type": "Feature"}]), ARCGIS_ERROR),
    ("geojson", _collection([{"type": "Feature"}]), _collection([])),
    ("geojson", _collection([{"type": "Feature"}]), b'[{"type": "Feature"}]'),
    ("xml", XML, HTML),
    ("xml", XML, HTML5),
    ("xml", XML, XHTML),
])
def test_each_check_tells_the_data_from_what_a_server_sends_instead(tmp_path, kind, good, bad):
    check = getattr(D, f"_is_{kind}")
    (tmp_path / "good").write_bytes(good)
    (tmp_path / "bad").write_bytes(bad)
    check(tmp_path / "good")
    with pytest.raises(IOError):
        check(tmp_path / "bad")


def test_the_error_says_what_arrived(tmp_path):
    (tmp_path / "x.zip").write_bytes(HTML)
    assert "System maintenance" in D._rejects(D._is_zip, tmp_path / "x.zip")
    (tmp_path / "x.json").write_bytes(ARCGIS_ERROR)
    assert "Unable to complete operation" in D._rejects(D._is_geojson, tmp_path / "x.json")


# ------------------------------------------------------- _download_file
def test_a_page_sent_with_http_200_is_never_cached(server, tmp_path):
    server.routes["/T1.zip"] = [HTML]
    dest = tmp_path / "T1.zip"
    with pytest.raises(IOError, match="not a zip file"):
        D._download_file(server.url + "/T1.zip", dest, validate=D._is_zip, quiet=True)
    assert server.hits["/T1.zip"] == 5              # retried like any failure
    assert list(tmp_path.iterdir()) == []           # no file, no .part


def test_without_a_check_the_page_would_have_been_kept(server, tmp_path):
    """The behaviour the hook exists to stop, pinned so the test above means something."""
    server.routes["/T1.zip"] = [HTML]
    dest = D._download_file(server.url + "/T1.zip", tmp_path / "T1.zip", quiet=True)
    assert dest.read_bytes() == HTML


def test_a_page_first_and_the_file_on_the_retry(server, tmp_path):
    server.routes["/T1.zip"] = [HTML, _zip_bytes()]
    dest = D._download_file(server.url + "/T1.zip", tmp_path / "T1.zip",
                            validate=D._is_zip, quiet=True)
    assert zipfile.is_zipfile(dest) and server.hits["/T1.zip"] == 2


def test_a_cached_file_that_fails_the_check_is_fetched_again(server, tmp_path, caplog):
    dest = tmp_path / "areas.geojson"
    dest.write_bytes(ARCGIS_ERROR)                  # cached by an earlier version
    server.routes["/areas"] = [_collection([{"type": "Feature"}])]
    with caplog.at_level("WARNING"):
        D._download_file(server.url + "/areas", dest, validate=D._is_geojson)
    assert json.loads(dest.read_text())["features"]
    assert "is not what it should be" in caplog.text

    # and a good cached file is not fetched at all
    D._download_file(server.url + "/areas", dest, validate=D._is_geojson)
    assert server.hits["/areas"] == 1


def test_a_failed_refetch_leaves_the_cached_file_alone(server, tmp_path):
    dest = tmp_path / "areas.geojson"
    good = _collection([{"type": "Feature"}])
    dest.write_bytes(good)
    server.routes["/areas"] = [HTML]
    with pytest.raises(IOError):
        D._download_file(server.url + "/areas", dest, force=True, validate=D._is_geojson)
    assert dest.read_bytes() == good


# ------------------------------------------- the sources that must arrive
def test_the_planning_areas_servers_own_error_is_refused(server, tmp_path, monkeypatch):
    server.routes["/query"] = [ARCGIS_ERROR]
    monkeypatch.setattr(sources, "NEIGHBORHOOD_URL", server.url + "/query")
    monkeypatch.setattr(D, "NEIGHBORHOODS_GEOJSON", tmp_path / "ozp.geojson")
    with pytest.raises(IOError, match="answered with an error"):
        D.download_neighborhoods()
    assert not (tmp_path / "ozp.geojson").exists()


def test_bench_marks_must_be_xml_and_the_5m_model_a_zip(server, tmp_path, monkeypatch):
    server.routes["/BM.xml"] = [HTML]
    server.routes["/dtm.zip"] = [HTML]
    monkeypatch.setattr(sources, "BENCHMARKS_URL", server.url + "/BM.xml")
    monkeypatch.setattr(sources, "DTM_5M_URL", server.url + "/dtm.zip")
    monkeypatch.setattr(D, "BENCHMARKS_XML", tmp_path / "bm.xml")
    monkeypatch.setattr(D, "DTM_5M_ZIP", tmp_path / "dtm.zip")
    with pytest.raises(IOError, match="a web page, not the data"):
        D.download_benchmarks()
    with pytest.raises(IOError, match="not a zip file"):
        D.download_dtm_5m()
    assert list(tmp_path.iterdir()) == []
    server.routes["/BM.xml"] = [HTML5]
    with pytest.raises(IOError, match="not XML"):
        D.download_benchmarks()

    server.routes["/BM.xml"] = [XML]
    assert D.download_benchmarks().read_bytes() == XML


def _tile_feature(i: int, filename: str, url: str) -> dict:
    """One 750 m x 600 m sheet of the index, in WGS84 as it is published."""
    from pyproj import Transformer

    to_ll = Transformer.from_crs("EPSG:2326", "EPSG:4326", always_xy=True)
    left, bottom = 830000 + 750 * i, 815000
    ring = [(left, bottom), (left, bottom + 600), (left + 750, bottom + 600),
            (left + 750, bottom), (left, bottom)]
    return {"type": "Feature",
            "geometry": {"type": "Polygon",
                         "coordinates": [[list(to_ll.transform(*p)) for p in ring]]},
            "properties": {"TILE_NAME": f"T{i}", "Filename": filename, "URL": url}}


@pytest.fixture
def dem(server, tmp_path, monkeypatch):
    """``download_dem`` pointed at the stand-in server and a scratch directory."""
    monkeypatch.setattr(sources, "DTM_INDEX_URL", server.url + "/index")
    monkeypatch.setattr(D, "DEM_INDEX_GEOJSON", tmp_path / "index.geojson")
    monkeypatch.setattr(D, "DEM_DIR", tmp_path / "dem")
    server.routes["/index"] = [_collection([
        _tile_feature(i, f"T{i}.zip", f"{server.url}/T{i}.zip") for i in range(2)])]
    return server


def test_a_tile_that_is_a_web_page_stops_the_download_and_is_not_kept(dem, tmp_path):
    dem.routes["/T0.zip"] = [_zip_bytes()]
    dem.routes["/T1.zip"] = [HTML]
    with pytest.raises(IOError, match="not a zip file"):
        D.download_dem(workers=1)
    assert sorted(p.name for p in (tmp_path / "dem").iterdir()) == ["T0.zip"]

    # the server recovers: the second run fetches the one tile that is missing
    dem.routes["/T1.zip"] = [_zip_bytes()]
    paths = D.download_dem(workers=1)
    assert sorted(p.name for p in paths) == ["T0.zip", "T1.zip"]
    assert all(zipfile.is_zipfile(p) for p in paths)
    assert dem.hits["/T0.zip"] == 1


def test_a_page_cached_as_a_tile_by_an_earlier_run_is_replaced(dem, tmp_path):
    (tmp_path / "dem").mkdir()
    (tmp_path / "dem" / "T0.zip").write_bytes(HTML)
    dem.routes["/T0.zip"] = [_zip_bytes()]
    dem.routes["/T1.zip"] = [_zip_bytes()]
    paths = D.download_dem(workers=1)
    assert all(zipfile.is_zipfile(p) for p in paths)


def test_an_index_that_is_a_web_page_is_refused_and_not_cached(dem, tmp_path):
    dem.routes["/index"] = [HTML]
    with pytest.raises(IOError, match="not JSON"):
        D.download_dem(workers=1)
    assert not (tmp_path / "index.geojson").exists()


@pytest.mark.parametrize("name", [
    "../../../escaped_from_dem_dir.zip", "/tmp/absolute.zip", "sub/dir.zip",
    "back\\slash.zip", "C:drive.zip", ".zip", "..", "", "tile.tif", "tile.zip.exe",
])
def test_tile_names_with_a_path_in_them_are_refused(name):
    with pytest.raises(ValueError, match="not a plain .zip file name"):
        D._tile_filename(name)


def test_real_tile_names_pass():
    # the shape of the 3,253 names in the published index
    for name in ("11NE14D(e842n820,e843n821).zip", "6NW11C(e819n830,e819n831).ZIP"):
        assert D._tile_filename(name) == name


def test_an_index_cannot_write_outside_the_tile_directory(dem, tmp_path):
    dem.routes["/index"] = [_collection([
        _tile_feature(0, "../../escaped.zip", dem.url + "/T0.zip")])]
    dem.routes["/T0.zip"] = [_zip_bytes()]
    with pytest.raises(ValueError, match="not a plain .zip file name"):
        D.download_dem(workers=1)
    assert not (tmp_path.parent / "escaped.zip").exists()
    assert "/T0.zip" not in dem.hits


# ------------------------------------------------- the Overture release
class _Listing:
    """A stand-in for the bucket's list-objects answers."""
    status_code = 200

    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        pass


def _bucket(hosted, files=()):
    ns = 'xmlns="http://s3.amazonaws.com/doc/2006-03-01/"'

    def get(url, params=None, **kwargs):
        if params.get("delimiter") == "/":
            body = "".join(f"<CommonPrefixes><Prefix>release/{r}/</Prefix></CommonPrefixes>"
                           for r in hosted)
        else:
            body = "".join(f"<Contents><Key>{k}</Key></Contents>" for k in files
                           if k.startswith(params["prefix"]))
        return _Listing(f'<?xml version="1.0"?><ListBucketResult {ns}>{body}'
                        "<IsTruncated>false</IsTruncated></ListBucketResult>")
    return get


def test_a_retired_release_is_named_with_the_way_out(tmp_path, monkeypatch):
    hosted = ["2026-08-19.0", "2026-09-23.0", "2026-09-23.1"]
    monkeypatch.setattr(D.requests, "get", _bucket(hosted))
    with pytest.raises(RuntimeError) as err:
        D._read_overture_type("segment", D.SEGMENT_COLUMNS, tmp_path / "x.parquet",
                              prefix="release/2025-01-22.0/theme=transportation")
    msg = str(err.value)
    assert "release 2025-01-22.0 is no longer hosted" in msg
    assert "Hosted now: 2026-08-19.0, 2026-09-23.0, 2026-09-23.1." in msg
    assert "HKFR_OVERTURE_RELEASE" in msg and sources.OVERTURE_RELEASE_PINNED in msg
    assert "row groups intersect" not in msg         # what it used to say
    assert not (tmp_path / "x.parquet").exists()


def test_a_hosted_release_with_no_such_files_is_a_different_error(tmp_path, monkeypatch):
    monkeypatch.setattr(D.requests, "get", _bucket(["2026-09-23.1"]))
    with pytest.raises(RuntimeError, match="is hosted but lists no segment files"):
        D._read_overture_type("segment", D.SEGMENT_COLUMNS, tmp_path / "x.parquet",
                              prefix="release/2026-09-23.1/theme=transportation")


def test_the_error_survives_a_listing_that_fails(tmp_path, monkeypatch):
    def get(url, params=None, **kwargs):
        if params.get("delimiter") == "/":
            raise requests.ConnectionError("offline")
        return _bucket([])(url, params)
    monkeypatch.setattr(D.requests, "get", get)
    with pytest.raises(RuntimeError, match="no longer hosted.*HKFR_OVERTURE_RELEASE"):
        D._read_overture_type("segment", D.SEGMENT_COLUMNS, tmp_path / "x.parquet",
                              prefix="release/2025-01-22.0/theme=transportation")


def test_a_cached_extract_never_asks_the_bucket(tmp_path, monkeypatch):
    def get(*a, **k):
        raise AssertionError("the bucket was asked")
    monkeypatch.setattr(D.requests, "get", get)
    (tmp_path / "x.parquet").write_bytes(b"kept from the release the figures were built on")
    assert D._read_overture_type("segment", D.SEGMENT_COLUMNS, tmp_path / "x.parquet",
                                 prefix="release/2025-01-22.0/theme=transportation")


def test_forcing_a_retired_release_fails_and_leaves_the_kept_extract_alone(tmp_path, monkeypatch):
    """``download --force`` reads the extracts again; when it cannot, the
    copy that was kept for exactly this day must still be there."""
    monkeypatch.setattr(D.requests, "get", _bucket(["2026-11-18.0"]))
    kept = tmp_path / "x.parquet"
    kept.write_bytes(b"kept from the release the figures were built on")
    with pytest.raises(RuntimeError, match="no longer hosted"):
        D._read_overture_type("segment", D.SEGMENT_COLUMNS, kept, force=True,
                              prefix="release/2026-09-23.1/theme=transportation")
    assert kept.read_bytes() == b"kept from the release the figures were built on"
    assert [p.name for p in tmp_path.iterdir()] == ["x.parquet"]


def test_a_cached_extract_under_another_release_is_not_used_in_silence(tmp_path, monkeypatch, caplog):
    """The variable applies only to what is not yet on disk, and the file
    does not say which release it is from: the log has to."""
    def get(*a, **k):
        raise AssertionError("the bucket was asked")
    monkeypatch.setattr(D.requests, "get", get)
    monkeypatch.setattr(D, "DIVISIONS_PARQUET", tmp_path / "overture_division_area_hk.parquet")
    kept = [tmp_path / "overture_segments_hk.parquet", D.DIVISIONS_PARQUET]
    for path in kept:
        path.write_bytes(b"kept")

    def fetch():
        del caplog.records[:]
        with caplog.at_level("INFO"):
            assert D._read_overture_type("segment", D.SEGMENT_COLUMNS, kept[0]) == kept[0]
            assert D.download_boundary() == kept[1]
        return [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]

    assert fetch() == []                            # the pinned release: nothing to say
    monkeypatch.setattr(sources, "OVERTURE_RELEASE", "2026-08-19.0")
    said = fetch()
    assert len(said) == 2
    for path, line in zip(kept, said):
        assert path.name in line and "used as cached" in line
        assert "release 2026-08-19.0 (HKFR_OVERTURE_RELEASE)" in line and "--force" in line
        assert path.read_bytes() == b"kept"


@pytest.fixture
def bucket(tmp_path, monkeypatch):
    """A stand-in for the Overture bucket: Parquet files in a directory,
    listed and opened as the hosted ones are, without the network.

    Returns ``put(key, rows)``, which writes one file of two-row row groups;
    a row is (id, lon, lat) with any further columns as keywords.
    """
    import fsspec
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = tmp_path / "bucket"
    local = fsspec.filesystem("file")
    monkeypatch.setattr(sources, "OVERTURE_BUCKET", str(root))
    monkeypatch.setattr(fsspec, "filesystem", lambda protocol: local)
    monkeypatch.setattr(D, "_list_s3_keys", lambda bucket, prefix, suffix=".parquet": sorted(
        key for key in (p.relative_to(root).as_posix() for p in root.rglob("*" + suffix))
        if key.startswith(prefix)))

    def put(key, rows, **columns):
        table = pa.table({
            "id": [r[0] for r in rows],
            "bbox": [{"xmin": r[1], "xmax": r[1], "ymin": r[2], "ymax": r[2]} for r in rows],
            **columns}).replace_schema_metadata({"geo": '{"version": "1.1.0"}'})
        (root / key).parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, root / key, row_group_size=2)
    return put


#: Statue Square; a point in the first row group's box and west of the study
#: area; and two in Zurich, whose row group is never read.
_HK, _WEST, _FAR = ("hk", 114.16, 22.28), ("west", 113.50, 22.30), ("far", 8.54, 47.37)


def test_an_extract_is_the_study_area_and_says_which_release_it_is(bucket, tmp_path):
    import pyarrow.parquet as pq

    bucket("release/2026-08-19.0/theme=transportation/type=segment/part-0.parquet",
           [_HK, _WEST, _FAR, _FAR])
    bucket("release/2026-08-19.0/theme=transportation/type=segment/part-1.parquet",
           [_FAR, _FAR])
    dest = tmp_path / "raw" / "overture_segments_hk.parquet"
    D._read_overture_type("segment", ["id", "bbox"], dest,
                          prefix="release/2026-08-19.0/theme=transportation")
    table = pq.read_table(dest)
    assert table.column("id").to_pylist() == ["hk"]  # pruned, then clipped
    assert table.schema.metadata[b"geo"] == b'{"version": "1.1.0"}'
    assert D._extract_release(dest) == "2026-08-19.0"
    assert [p.name for p in dest.parent.iterdir()] == [dest.name]   # no .part


def test_the_boundary_keeps_hong_kong_and_the_release_it_was_read_from(bucket, tmp_path, monkeypatch):
    import pyarrow.parquet as pq

    monkeypatch.setattr(D, "DIVISIONS_PARQUET", tmp_path / "raw" / "overture_division_area_hk.parquet")
    bucket(f"{sources.OVERTURE_DIVISIONS_PREFIX}/type=division_area/part-0.parquet",
           [_HK, ("shenzhen", 114.06, 22.54)], names=[None, None],
           subtype=["country", "locality"], country=["HK", "CN"],
           geometry=[b"", b""], **{"class": ["land", "land"]})
    out = D.download_boundary()
    assert pq.read_table(out).column("id").to_pylist() == ["hk"]
    assert D._extract_release(out) == sources.OVERTURE_RELEASE
    assert [p.name for p in out.parent.iterdir()] == [out.name]


def test_a_boundary_write_cut_short_is_not_the_cached_boundary(bucket, tmp_path, monkeypatch):
    import pyarrow.parquet as pq

    monkeypatch.setattr(D, "DIVISIONS_PARQUET", tmp_path / "raw" / "overture_division_area_hk.parquet")
    bucket(f"{sources.OVERTURE_DIVISIONS_PREFIX}/type=division_area/part-0.parquet",
           [_HK, ("shenzhen", 114.06, 22.54)], names=[None, None],
           subtype=["country", "locality"], country=["HK", "CN"],
           geometry=[b"", b""], **{"class": ["land", "land"]})
    write = pq.write_table

    def interrupted(table, where, **kwargs):
        if Path(where).stem == D.DIVISIONS_PARQUET.stem:    # the file, or its .part
            Path(where).write_bytes(b"PAR1 and no more")
            raise KeyboardInterrupt
        return write(table, where, **kwargs)
    monkeypatch.setattr(pq, "write_table", interrupted)
    with pytest.raises(KeyboardInterrupt):
        D.download_boundary()
    assert not D.DIVISIONS_PARQUET.exists()

    monkeypatch.setattr(pq, "write_table", write)           # the rerun starts afresh
    assert pq.read_table(D.download_boundary()).column("id").to_pylist() == ["hk"]


def test_an_extract_of_another_release_is_reported_whichever_release_is_in_force(bucket, tmp_path, monkeypatch, caplog):
    """Fetched under ``HKFR_OVERTURE_RELEASE`` and then run without it, the
    extract is still the other release's; the file says so, and so does
    the log.  Under the variable that fetched it there is nothing to say."""
    bucket("release/2026-08-19.0/theme=transportation/type=segment/part-0.parquet",
           [_HK, _WEST])
    dest = tmp_path / "overture_segments_hk.parquet"
    D._read_overture_type("segment", ["id", "bbox"], dest,
                          prefix="release/2026-08-19.0/theme=transportation")

    def cached():
        del caplog.records[:]
        with caplog.at_level("INFO"):
            assert D._cached_extract(dest)
        return [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]

    assert sources.OVERTURE_RELEASE == sources.OVERTURE_RELEASE_PINNED
    said, = cached()
    assert "it was read from release 2026-08-19.0" in said
    assert f"the pinned release {sources.OVERTURE_RELEASE_PINNED}" in said and "--force" in said
    monkeypatch.setattr(sources, "OVERTURE_RELEASE", "2026-08-19.0")
    assert cached() == []
    monkeypatch.setattr(sources, "OVERTURE_RELEASE", "2026-09-23.0")
    said, = cached()
    assert "it was read from release 2026-08-19.0" in said
    assert "release 2026-09-23.0 (HKFR_OVERTURE_RELEASE)" in said


def test_the_release_can_be_named_from_the_environment():
    import os
    import subprocess
    import sys

    root = Path(__file__).resolve().parent.parent
    code = ("from hk_flat_routes import sources as s; "
            "print(s.OVERTURE_RELEASE, s.OVERTURE_PREFIX, s.OVERTURE_DIVISIONS_PREFIX, "
            "s.DATASETS_BY_KEY['overture_segments'].title)")

    def run(extra):
        env = {k: v for k, v in os.environ.items() if k != "HKFR_OVERTURE_RELEASE"}
        env.update(extra, PYTHONPATH=str(root))
        return subprocess.run([sys.executable, "-c", code], env=env, cwd=root,
                              capture_output=True, text=True, check=True).stdout

    assert run({}).startswith("2026-09-23.1 release/2026-09-23.1/theme=transportation ")
    out = run({"HKFR_OVERTURE_RELEASE": "2026-08-19.0"})
    assert out.startswith("2026-08-19.0 release/2026-08-19.0/theme=transportation "
                          "release/2026-08-19.0/theme=divisions ")
    assert "(release 2026-08-19.0)" in out
    assert run({"HKFR_OVERTURE_RELEASE": ""}).startswith("2026-09-23.1 ")


# ------------------------------------- the land use the network is built with
def test_the_land_use_is_fetched_with_the_network_and_a_failure_stops_the_run(monkeypatch):
    """``network.closed_land`` closes the gated sites from it: without the
    file the container terminals are open, so it is not one of the optional
    place sources, whose failures are logged and passed over."""
    ran = []

    def stage(name, result=None):
        def run(force=False):
            ran.append(name)
            return result
        return run

    def retired(force=False):
        ran.append("land_use")
        raise RuntimeError("Overture release 2026-09-23.1 is no longer hosted")
    monkeypatch.setattr(D, "download_street_network", stage("network", ("seg", "con")))
    monkeypatch.setattr(D, "download_land_use", retired)
    for name in ("download_boundary", "download_neighborhoods", "download_places",
                 "download_dtm_5m", "download_benchmarks", "download_dem"):
        monkeypatch.setattr(D, name, stage(name))
    with pytest.raises(RuntimeError, match="no longer hosted"):
        D.download_all()
    assert ran == ["network", "land_use"]           # before the hour of tiles

    monkeypatch.setattr(D, "download_land_use", stage("land_use", "land use"))
    monkeypatch.setattr(D, "download_places", lambda force=False: {})
    assert D.download_all()["land_use"] == "land use"


def test_land_use_is_read_from_the_base_theme_and_forced_once(tmp_path, monkeypatch):
    calls = []

    def read(overture_type, columns, dest, bbox=None, force=False, prefix=None):
        calls.append((overture_type, dest.name, force, prefix))
        return dest
    monkeypatch.setattr(D, "_read_overture_type", read)
    monkeypatch.setattr(D, "_download_file", lambda *a, **k: tmp_path / "x")
    monkeypatch.setattr(D, "_download_igeocom", lambda *a, **k: tmp_path / "x")

    assert D.download_land_use(force=True) == D.BASE_PARQUETS["land_use"]
    assert calls == [("land_use", "overture_land_use_hk.parquet", True,
                      sources.OVERTURE_BASE_PREFIX)]

    # the place search reads the same file and leaves forcing it to the network
    del calls[:]
    D.download_places(force=True)
    forced = {typ: force for typ, _dest, force, _prefix in calls}
    assert forced == {"land_use": False, "infrastructure": True, "land": True,
                      "place": True}


# ----------------------------------------------- iGeoCom's missing link
class _Answer:
    def __init__(self, status_code=200, content=b""):
        self.status_code, self.content = status_code, content


def _no_curl(monkeypatch):
    import shutil
    import subprocess

    def refuse(*a, **k):
        raise AssertionError("curl was run")
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setattr(subprocess, "run", refuse)


def test_the_vendored_intermediate_is_the_certificate_the_readme_describes():
    pem = D.IGEOCOM_INTERMEDIATE_PEM.read_text(encoding="ascii")
    digest = hashlib.sha256(ssl.PEM_cert_to_DER_cert(pem)).hexdigest()
    readme = (D.IGEOCOM_INTERMEDIATE_PEM.parent / "README.md").read_text(encoding="utf-8")
    assert D.IGEOCOM_INTERMEDIATE_PEM.name in readme
    assert digest in readme.replace(":", "").lower()
    # one certificate, and one OpenSSL accepts as a CA to build chains with
    assert pem.count("BEGIN CERTIFICATE") == 1
    ssl.create_default_context(cadata=pem)


def test_igeocom_is_verified_with_the_intermediate_and_without_curl(tmp_path, monkeypatch):
    """Python cannot build the chain as sent; with the intermediate it can."""
    _no_curl(monkeypatch)
    intermediate = D.IGEOCOM_INTERMEDIATE_PEM.read_bytes()
    seen = []

    def get(url, cookies=None, verify=True, **kwargs):
        seen.append(verify)
        assert verify is not False                  # never switched off
        if verify is True:
            raise requests.exceptions.SSLError("unable to get local issuer certificate")
        bundle = Path(verify).read_bytes()
        assert intermediate in bundle
        assert len(bundle) > 50 * len(intermediate)  # added to the roots, not instead
        return _Answer(content=_zip_bytes("iGeoCom.csv"))
    monkeypatch.setattr(D.requests, "get", get)

    dest = D._download_igeocom(tmp_path / "igeocom.zip")
    assert zipfile.is_zipfile(dest)
    assert seen[0] is True and isinstance(seen[1], str) and len(seen) == 2
    assert not Path(seen[1]).exists()               # the bundle was a scratch file


def test_the_bundle_extends_the_one_the_environment_names(tmp_path, monkeypatch):
    own = tmp_path / "proxy-roots.pem"
    own.write_text("-----BEGIN CERTIFICATE-----\nPROXY\n-----END CERTIFICATE-----\n")
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(own))
    out = D._ca_bundle_with(D.IGEOCOM_INTERMEDIATE_PEM, tmp_path).read_text()
    assert out.startswith("-----BEGIN CERTIFICATE-----\nPROXY")
    assert out.count("BEGIN CERTIFICATE") == 2


def test_igeocom_falls_back_to_curl_only_when_the_intermediate_does_not_help(tmp_path, monkeypatch):
    import shutil
    import subprocess

    def get(url, **kwargs):
        raise requests.exceptions.SSLError("unable to get local issuer certificate")
    monkeypatch.setattr(D.requests, "get", get)
    ran = []

    def curl(cmd, **kwargs):
        ran.append(cmd)
        assert "--insecure" not in cmd and "-k" not in cmd
        Path(cmd[cmd.index("--output") + 1]).write_bytes(_zip_bytes("iGeoCom.csv"))
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/curl")
    monkeypatch.setattr(subprocess, "run", curl)
    assert zipfile.is_zipfile(D._download_igeocom(tmp_path / "igeocom.zip"))
    assert len(ran) == 1

    # and with no curl either, the source fails with the reason
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(IOError, match="no curl to fall back on"):
        D._download_igeocom(tmp_path / "again.zip")
    assert not (tmp_path / "again.zip").exists()


def test_an_http_error_from_igeocom_is_not_mistaken_for_a_certificate_fault(tmp_path, monkeypatch):
    _no_curl(monkeypatch)
    monkeypatch.setattr(D.requests, "get", lambda url, **k: _Answer(status_code=302))
    with pytest.raises(IOError, match="HTTP 302"):
        D._download_igeocom(tmp_path / "igeocom.zip")


# ------------------------------------------------- the closing summary
def test_download_ends_by_naming_the_optional_sources_it_went_without(tmp_path, monkeypatch, caplog):
    def fail(force=False):
        raise IOError("HTTP 503")
    monkeypatch.setattr(D, "download_street_network", lambda force=False: ("seg", "con"))
    monkeypatch.setattr(D, "download_land_use", lambda force=False: "land use")
    monkeypatch.setattr(D, "download_boundary", lambda force=False: "boundary")
    monkeypatch.setattr(D, "download_neighborhoods", lambda force=False: "areas")
    monkeypatch.setattr(D, "download_places", lambda force=False: {
        k: [] for k in D.PLACE_SOURCES if k != "igeocom"})
    monkeypatch.setattr(D, "download_dtm_5m", fail)
    monkeypatch.setattr(D, "download_benchmarks", lambda force=False: "bm")
    monkeypatch.setattr(D, "download_dem", lambda force=False: [])
    with caplog.at_level("WARNING"):
        out = D.download_all()
    assert [line.split(":")[0] for line in out["skipped"]] == ["igeocom", "dtm_5m"]
    last = caplog.text.split("download finished without 2 optional source(s)")[1]
    assert "MTR stations and all their exits" in last
    assert "cross-check against the 5 m terrain model" in last


def test_download_with_everything_says_nothing_more(monkeypatch, caplog):
    monkeypatch.setattr(D, "download_street_network", lambda force=False: ("seg", "con"))
    for name in ("download_land_use", "download_boundary", "download_neighborhoods",
                 "download_dtm_5m", "download_benchmarks", "download_dem"):
        monkeypatch.setattr(D, name, lambda force=False: "ok")
    monkeypatch.setattr(D, "download_places",
                        lambda force=False: {k: [] for k in D.PLACE_SOURCES})
    with caplog.at_level("WARNING"):
        assert D.download_all()["skipped"] == []
    assert "download finished without" not in caplog.text


def test_a_place_source_that_answers_with_a_page_is_skipped_and_not_cached(server, tmp_path, monkeypatch, caplog):
    """The place sources are optional, not unchecked: the gazetteer's two
    layers come from the portal that sends its maintenance page with HTTP
    200, and a page kept as a layer would break ``places.build_places``."""
    def elsewhere(*a, **k):
        raise IOError("not in this test")
    monkeypatch.setattr(D, "_read_overture_type", elsewhere)
    monkeypatch.setattr(D, "_download_igeocom", elsewhere)
    # every other source: a file of its own in the scratch directory, and
    # an address the stand-in server answers with 404
    for name in ("BUILDING_NAME_URL", "ALS_URL"):
        monkeypatch.setattr(sources, name, server.url + "/missing")
    monkeypatch.setattr(sources, "TD_STOPS_URLS", {"tram": server.url + "/missing"})
    monkeypatch.setattr(D, "BUILDING_NAMES_GEOJSON", tmp_path / "buildings.geojson")
    monkeypatch.setattr(D, "ALS_ZIP", tmp_path / "als.zip")
    monkeypatch.setattr(D, "TD_STOPS_JSON", {"tram": tmp_path / "tram.json"})
    monkeypatch.setattr(sources, "PLACE_NAME_POINTS_URL", server.url + "/points")
    monkeypatch.setattr(sources, "PLACE_NAME_NAMES_URL", server.url + "/names")
    monkeypatch.setattr(D, "PLACE_NAME_POINTS_GEOJSON", tmp_path / "points.geojson")
    monkeypatch.setattr(D, "PLACE_NAME_NAMES_GEOJSON", tmp_path / "names.geojson")
    server.routes["/points"] = [_collection([{"type": "Feature"}])]
    server.routes["/names"] = [HTML]
    with caplog.at_level("WARNING"):
        assert D.download_places() == {}
    assert "place source place_names unavailable (not JSON" in caplog.text
    assert sorted(p.name for p in tmp_path.iterdir()) == ["points.geojson"]

    server.routes["/names"] = [_collection([{"type": "Feature"}])]
    assert [p.name for p in D.download_places()["place_names"]] == [
        "points.geojson", "names.geojson"]
    assert server.hits["/points"] == 1               # the good layer was kept


def test_every_place_source_has_its_consequence_spelt_out(tmp_path, monkeypatch, caplog):
    """``download_places`` and the summary agree on what the sources are."""
    def gone(*a, **k):
        raise IOError("offline")
    monkeypatch.setattr(D, "_read_overture_type", gone)
    monkeypatch.setattr(D, "_download_file", gone)
    monkeypatch.setattr(D, "_download_igeocom", gone)
    with caplog.at_level("WARNING"):
        assert D.download_places() == {}
    for key, what in D.PLACE_SOURCES.items():
        assert f"place source {key} unavailable" in caplog.text
        assert what in caplog.text
