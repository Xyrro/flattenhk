"""The two web maps: the analysis explorer and the simple route page.

Both are single self-contained HTML files with Leaflet vendored and every
byte of data embedded, so they can be moved around and opened directly, and
both route in the browser over the packed graph (``webgraph.py``) with the
same cost model Python uses.

* **Explorer** (``outputs/hk_flat_routes_map.html``): every analysis layer
  (gradient-coloured network, corridors, passes, barriers, basins, bike
  facilities), the four objectives with live weight sliders, Pareto readout
  and the cost-warped city. Dense by design; this is the working view.
* **Route page**: one card with origin, destination and a shortest-to-
  flattest slider over a quiet hillshade. Place search is offline
  (junctions from the graph, places packed into the page). Written twice: as ``outputs/hk_flat_route_finder.html``, one
  file that opens from disk, and as the static site in ``site/`` (HTML, CSS,
  JS, the gzipped graph and the hillshade as separate cacheable files),
  which GitHub Pages serves as the demo.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .config import (CITY_NAME, CODE_ROOT, DEFAULT_TRIP, OUTPUT_DIR,
                     PRODUCT_NAME, REPO_URL, SITE_DIR, SITE_DOMAIN, SITE_URL,
                     UPSTREAM_AUTHOR, UPSTREAM_NAME, UPSTREAM_REPO,
                     UPSTREAM_SITE)
from .utils import get_logger, human_bytes, step

log = get_logger("hk_flat_routes.viz_interactive")

INTERACTIVE_HTML = OUTPUT_DIR / "hk_flat_routes_map.html"
#: The route finder as one self-contained file, and as a static site.
SIMPLE_HTML = OUTPUT_DIR / "hk_flat_route_finder.html"
SITE_INDEX = SITE_DIR / "index.html"
WEB_DIR = Path(__file__).resolve().parent / "web"
VENDOR_DIR = Path(__file__).resolve().parent / "vendor"

# --------------------------------------------------------------------------
# layer preparation
# --------------------------------------------------------------------------
def _round_geometry(geo: dict, ndigits: int = 5) -> dict:
    """Round every coordinate in a __geo_interface__ mapping.

    Full-precision floats serialise to ~17 characters each; five decimal
    places is about 1 m at this latitude, which is finer than the source
    data's own accuracy, and shrinks the embedded GeoJSON several-fold.
    """
    def walk(c):
        if isinstance(c, (list, tuple)):
            if c and isinstance(c[0], (int, float)):
                return [round(float(v), ndigits) for v in c[:2]]
            return [walk(x) for x in c]
        return c
    return {"type": geo["type"], "coordinates": walk(geo["coordinates"])}


def _geojson(gdf, props: list[str], simplify: float = 4.0) -> dict:
    """GeoDataFrame -> GeoJSON dict in WGS84, geometry simplified in metres."""
    g = gdf.copy()
    if simplify:
        g["geometry"] = g.geometry.simplify(simplify, preserve_topology=False)
    g = g[g.geometry.notna() & ~g.geometry.is_empty]
    g = g.to_crs("EPSG:4326")
    keep = [c for c in props if c in g.columns]
    feats = []
    for _, r in g.iterrows():
        p = {}
        for c in keep:
            v = r[c]
            if isinstance(v, (np.floating, float)):
                v = None if not np.isfinite(v) else round(float(v), 5)
            elif isinstance(v, (np.integer,)):
                v = int(v)
            elif isinstance(v, (np.bool_, bool)):
                v = bool(v)
            elif v is not None and not isinstance(v, str):
                v = str(v)
            p[c] = v
        feats.append({"type": "Feature", "properties": p,
                      "geometry": _round_geometry(r.geometry.__geo_interface__)})
    return {"type": "FeatureCollection", "features": feats}


def _first_vertices(geoms) -> np.ndarray:
    """The first (x, y) of every geometry; NaN for an empty one."""
    import shapely

    xy, idx = shapely.get_coordinates(np.asarray(geoms), return_index=True)
    out = np.full((len(geoms), 2), np.nan)
    if len(idx):
        first = np.r_[True, idx[1:] != idx[:-1]]
        out[idx[first]] = xy[first]
    return out


def _basin_membership(edges, basins):
    """Which lowland basin each street edge is in, without its geometry.

    Upstream ships the basins to the page as GeoJSON.  In San Francisco
    that is a few thousand blocks.  In Hong Kong the reclaimed shore and the
    valley floors are where the city *is*: 65,000 of the urban core's
    109,000 edges lie in a basin, and a second copy of their geometry would
    outweigh every other overlay put together and leave Leaflet hit-testing
    tens of thousands of line parts on every mouse move.  The page already
    holds every street in the packed graph, so all it needs is one small
    number per edge, and it paints the basins itself.

    A basin row is a street edge that starts at a basin node
    (``passes.run_pass_analysis``), so rows are matched to edges by where
    they start: by ``edge_id`` when the table carries one, and otherwise by
    first vertex to within half a metre (the table comes back from GeoJSON
    with seven decimal places and no index).

    Returns ``(member, rows)``: per edge, 0 for no basin and otherwise one
    plus the position of its basin in ``rows``, and ``rows`` as
    ``{"basin", "basin_label", "length_km"}`` dictionaries.
    """
    import pandas as pd

    member = np.zeros(len(edges), dtype="<u2")
    if basins is None or not len(basins) or "basin" not in basins.columns:
        return member.astype("<u1"), []
    b = basins[basins["basin"].notna() & basins.geometry.notna()]
    if b.crs is not None and b.crs != edges.crs:
        b = b.to_crs(edges.crs)
    ids = pd.to_numeric(b["basin"], errors="coerce").to_numpy()
    b, ids = b[np.isfinite(ids)], ids[np.isfinite(ids)].astype(np.int64)
    order = np.unique(ids)
    if not len(order):
        return member.astype("<u1"), []
    slot = np.searchsorted(order, ids) + 1          # 1-based; 0 means no basin

    # ``passes.read_basins`` hands back an ``edge_id`` column whether or not
    # the file on disk has one (a run made before the column was written
    # reads as all missing), so the column counts only when it is filled in.
    eid = (pd.to_numeric(b["edge_id"], errors="coerce").to_numpy(dtype=float)
           if "edge_id" in b.columns else None)
    if eid is not None and np.isfinite(eid).all():
        at = pd.Series(slot, index=eid.astype(np.int64))
        hit = edges["edge_id"].map(at[~at.index.duplicated()])
        member = hit.fillna(0).to_numpy().astype("<u2")
    else:
        from scipy.spatial import cKDTree

        start = _first_vertices(b.geometry.values)
        ok = np.isfinite(start).all(axis=1)
        tree = cKDTree(start[ok])
        dist, j = tree.query(_first_vertices(edges.geometry.values),
                             distance_upper_bound=0.5)
        found = np.isfinite(dist)
        member[found] = slot[ok][j[found]]
    matched = int((member > 0).sum())
    if abs(matched - len(b)) > 0.01 * len(b):
        log.warning("basins: %d street edges matched for %d basin rows; the "
                    "basin layer may be incomplete", matched, len(b))

    labels = (b["basin_label"].fillna("").astype(str).to_numpy()
              if "basin_label" in b.columns else np.full(len(b), ""))
    lengths = (pd.to_numeric(b["length_m"], errors="coerce").fillna(0.0).to_numpy()
               if "length_m" in b.columns else np.zeros(len(b)))
    # the length shown is the analysis's own (``basin_km``: street with both
    # ends in the basin), where the table carries it; summing the layer's
    # rows, as upstream's dissolve did, also counts the edges that leave the
    # basin and reads a few per cent longer than lowland_basins.csv
    measured = (pd.to_numeric(b["basin_km"], errors="coerce").to_numpy()
                if "basin_km" in b.columns else np.full(len(b), np.nan))
    rows = []
    for k, basin in enumerate(order):
        sel = slot == k + 1
        km = measured[sel][0]
        if not np.isfinite(km):
            km = float(lengths[sel].sum()) / 1000.0
        rows.append({"basin": int(basin), "basin_label": str(labels[sel][0]),
                     "length_km": round(float(km), 3)})
    if len(rows) < 255:
        member = member.astype("<u1")
    return member, rows


def build_layers(ctx, corridors, passes, barriers, basins,
                 arrays: dict | None = None):
    """The small vector overlays. The street network is *not* here.

    The network is served from the packed graph instead, so its geometry
    exists exactly once in the file and the gradient display can never
    disagree with what the router uses.  The lowland basins go the same
    way in Hong Kong (see ``_basin_membership``): the layer here is only
    their table, and the per-edge membership that goes with it is put into
    ``arrays`` (as ``edge_basin``) when a dictionary is passed for it.

    Any of the analysis layers may be missing or empty, as when a build
    covers one flat district that has no pass; the page shows what it gets.
    """
    edges = ctx.edges

    def local(gdf):
        """``gdf`` in the network's CRS, or None when there is nothing in it."""
        if gdf is None or not len(gdf):
            return None
        if gdf.crs is not None and gdf.crs != edges.crs:
            return gdf.to_crs(edges.crs)
        return gdf

    barriers = local(barriers)
    layers = {
        "neighborhoods": _geojson(ctx.neighborhoods,
                                  ["neighborhood", "neighborhood_zh", "area_km2"],
                                  simplify=12.0),
    }
    # Heights are metres above Principal Datum throughout; upstream's feet
    # column is gone from the pass table and is not asked for.
    for key, gdf, props, simplify in (
        ("corridors", local(corridors),
         ["corridor_id", "corridor_name", "street_names", "mode",
          "length_km", "mean_abs_grade", "max_grade", "gain_per_km",
          "pair_count_max", "neighborhood_span", "climb_saved_m",
          "elev_min_m", "elev_max_m", "neighborhoods", "total_score"], 5.0),
        ("passes", local(passes),
         ["edge_id", "name", "name_zh", "neighborhood", "nearest_neighborhood",
          "pass_elev_m", "pairs_served", "max_abs_grade",
          "neighborhoods_separated"], 2.0),
        ("barriers", None if barriers is None else barriers.head(350),
         ["edge_id", "name", "name_zh", "neighborhood", "nearest_neighborhood",
          "max_abs_grade", "length_m", "shortest_use",
          "flat_use_per_objective", "unavoidability"], 2.0),
    ):
        if gdf is not None:
            layers[key] = _geojson(gdf, props, simplify=simplify)

    # Upstream draws every edge that carries a facility label.  Here the
    # label "shared_path" sits on every footway and path, some 60,000 of
    # them in the urban core alone, and cycling on a footway is an offence
    # unless the path is signed as shared
    # (``config.BIKE_PERMIT_REQUIRED_CLASSES``); so only the facilities a
    # bicycle may actually use are drawn.
    facility = edges["bike_facility"].fillna("").astype(str)
    bike = edges[(facility != "") & edges["bike_ok"].fillna(False).astype(bool)]
    layers["bike_network"] = _geojson(
        bike, ["name", "name_zh", "cls", "bike_facility", "length_m",
               "max_abs_grade"], simplify=5.0)
    carfree = edges[edges["cls"].isin(["pedestrian", "living_street"])
                    | (facility == "car_free_street")]
    layers["low_stress"] = _geojson(
        carfree, ["name", "name_zh", "cls", "bike_facility", "length_m"],
        simplify=5.0)
    member, rows = _basin_membership(edges, basins)
    if rows:
        layers["basins"] = {"rows": rows}
        if arrays is not None:
            arrays["edge_basin"] = member
    return layers


#: Guided examples, so that opening the map demonstrates the findings
#: without anyone having to know which neighbourhoods to pick.  Both ends
#: of a pair are on one landmass: nothing crosses the harbour on foot or by
#: bicycle, so a Central to Tsim Sha Tsui example would have no route to
#: show.  The words say only what the map itself shows; the figures are
#: filled in from the analysis outputs at build time.  The first pair that
#: survives is also the trip the page opens on.
_EXAMPLE_PAIRS = (
    ("Pok Fu Lam", "Shau Kei Wan", "walk", "min_climb",
     "the length of Hong Kong Island, west to east"),
    ("Aberdeen & Ap Lei Chau", "Central District", "walk", "min_climb",
     "south side to north shore: over the hills or round them"),
    ("Central District", "The Peak Area", "walk", "grade_averse",
     "up to the Peak: the climb cannot be avoided, only made gentler"),
    ("Tsim Sha Tsui", "Kowloon Tong", "walk", "min_climb",
     "up the Kowloon peninsula"),
    ("Cha Kwo Ling, Yau Tong, Lei Yue Mun", "Wang Tau Hom & Tung Tau",
     "bike", "balanced", "by bicycle across east Kowloon"),
)


def _examples(points: dict) -> list[dict]:
    """Attach measured savings, in metres, to each guided example.

    An example is dropped when either end has no access point in this
    build, and also when the pair table exists but holds no row for it:
    the table carries every pair that shares a landmass, so a pair missing
    from it has no route.
    """
    from .pairs import PAIRS_PARQUET
    import pandas as pd

    out = []
    table = None
    if PAIRS_PARQUET.exists():
        table = pd.read_parquet(PAIRS_PARQUET)
    for o, d, mode, prof, blurb in _EXAMPLE_PAIRS:
        if o not in points.get(mode, {}) or d not in points.get(mode, {}):
            continue
        note = blurb
        if table is not None:
            sel = table[(table["mode"] == mode) & (table["profile"] == prof)
                        & (table["origin"] == o) & (table["destination"] == d)]
            if not len(sel):
                continue
            r = sel.iloc[0]
            extra_km = (r["distance_m"] - r["shortest_distance_m"]) / 1000.0
            if prof == "grade_averse":
                # this objective trades distance for gentleness, not for
                # less climbing, so the steepest pitch is the figure to show
                change = (f"a steepest pitch of {r['shortest_max_grade']*100:.0f}% "
                          f"becomes {r['max_grade']*100:.0f}%")
            else:
                change = (f"{r['shortest_gain_m']:.0f} m of climbing "
                          f"becomes {r['elev_gain_m']:.0f} m")
            note = f"{blurb} — {change}, for {extra_km:.1f} km more"
        out.append({"o": o, "d": d, "mode": mode, "profile": prof,
                    "note": note})
    return out


def _asset(name: str) -> str:
    return (WEB_DIR / name).read_text(encoding="utf-8")


def _vendor(name: str) -> str:
    """Read a vendored asset for inlining (see ``vendor/README.md``)."""
    return (VENDOR_DIR / name).read_text(encoding="utf-8")


def _json(obj) -> str:
    """Compact JSON for inlining in a <script>.

    Chinese is written as itself, not as six-byte escapes (the name table
    is a third smaller for it), and "</" is broken up so that no name can
    ever close the script element.
    """
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")


def _render(payload: dict) -> str:
    """The explorer page.

    The page's name and the credit to the project this one is a port of
    are written into the shell and the script from ``config``, so a fork
    builds with its own links and upstream's stay where they belong.  The
    data goes in last, so that nothing in it can be mistaken for a token.
    """
    html = _asset("index.html")
    html = html.replace("/*__LEAFLET_CSS__*/", _vendor("leaflet-1.9.4.css"))
    html = html.replace("/*__APP_CSS__*/", _asset("app.css"))
    html = html.replace("/*__LEAFLET_JS__*/", _vendor("leaflet-1.9.4.min.js"))
    html = html.replace("/*__APP_JS__*/", _asset("engine.js") + "\n" + _asset("app.js")
                        + "\n" + _asset("warp.js"))
    for token, value in (("PRODUCT_NAME", PRODUCT_NAME), ("CITY_NAME", CITY_NAME),
                         ("REPO_URL", REPO_URL),
                         ("UPSTREAM_NAME", UPSTREAM_NAME),
                         ("UPSTREAM_SITE", UPSTREAM_SITE),
                         ("UPSTREAM_REPO", UPSTREAM_REPO),
                         ("UPSTREAM_AUTHOR", UPSTREAM_AUTHOR)):
        html = html.replace(f"/*__{token}__*/", value)
    return html.replace("/*__DATA__*/", _json(payload))


_DESCRIPTION = (f"The flattest walking or cycling route between any two places "
                f"in {CITY_NAME}, and every route between it and the shortest one.")


def _route_page_html(payload: dict, linked: bool, assets: dict | None = None) -> str:
    """The route page: assets inlined (one file) or linked (the site).

    ``assets`` maps each plain asset name to its content-hashed file name.
    """
    html = _asset("simple.html").replace("/*__REPO_URL__*/", REPO_URL)
    # segments, as the README counts them: ``n_arcs`` is the directed arcs,
    # two to a two-way segment, and under the words "street and path
    # segments" it stated double the network (520,486 for 260,615)
    html = html.replace("/*__N_EDGES__*/", f"{payload['meta']['n_edges']:,}")
    if linked:
        a = assets or {}
        html = html.replace('<style>/*__LEAFLET_CSS__*/</style>',
                            f'<link rel="stylesheet" href="{a.get("leaflet.css", "leaflet.css")}">')
        html = html.replace('<style>/*__APP_CSS__*/</style>',
                            f'<link rel="stylesheet" href="{a.get("app.css", "app.css")}">')
        html = html.replace('<script>/*__LEAFLET_JS__*/</script>',
                            f'<script src="{a.get("leaflet.js", "leaflet.js")}"></script>')
        html = html.replace('<script>/*__APP_JS__*/</script>',
                            f'<script src="{a.get("app.js", "app.js")}"></script>')
        trip = payload.get("default") or []
        alt = (f"A map of {CITY_NAME} with a fan of walking routes between "
               f"{trip[0]['label']} and {trip[1]['label']}" if len(trip) == 2
               else f"A map of {CITY_NAME} with a fan of walking routes")
        extra = "\n".join([
            f'<link rel="canonical" href="{SITE_URL}">',
            '<link rel="icon" href="favicon.svg" type="image/svg+xml">',
            '<meta property="og:type" content="website">',
            f'<meta property="og:title" content="{PRODUCT_NAME}">',
            f'<meta property="og:site_name" content="{PRODUCT_NAME}">',
            f'<meta property="og:description" content="{_DESCRIPTION}">',
            f'<meta property="og:url" content="{SITE_URL}">',
            f'<meta property="og:image" content="{SITE_URL}preview.jpg">',
            f'<meta property="og:image:secure_url" content="{SITE_URL}preview.jpg">',
            '<meta property="og:image:type" content="image/jpeg">',
            '<meta property="og:image:width" content="1200">',
            '<meta property="og:image:height" content="630">',
            f'<meta property="og:image:alt" content="{alt}">',
            '<meta name="twitter:card" content="summary_large_image">',
            f'<meta name="twitter:title" content="{PRODUCT_NAME}">',
            f'<meta name="twitter:description" content="{_DESCRIPTION}">',
            f'<meta name="twitter:image" content="{SITE_URL}preview.jpg">',
            f'<link rel="preload" href="{payload["bundle_url"]}" as="fetch" crossorigin>',
        ])
        html = html.replace("<!--__HEAD_EXTRA__-->", extra)
    else:
        html = html.replace("<!--__HEAD_EXTRA__-->\n", "")
        html = html.replace("/*__LEAFLET_CSS__*/", _vendor("leaflet-1.9.4.css"))
        html = html.replace("/*__APP_CSS__*/", _asset("simple.css"))
        html = html.replace("/*__LEAFLET_JS__*/", _vendor("leaflet-1.9.4.min.js"))
        html = html.replace("/*__APP_JS__*/", _asset("engine.js") + "\n" + _asset("simple.js"))
    return html.replace("/*__DATA__*/", _json(payload))


_FAVICON = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">
<rect width="32" height="32" rx="7" fill="#0f8f6f"/>
<path d="M5 22 C10 22 11 12 16 12 S22 20 27 10" fill="none" stroke="#fff" stroke-width="3.2" stroke-linecap="round"/>
</svg>
"""


def _copy_licences(site: Path) -> None:
    """Put the notices the served files are used under beside them.

    Only ``site/`` is published.  Leaflet is BSD-2-Clause, and the page's
    own script is upstream's code and this port's, each under an MIT
    notice that asks to travel with copies: the copyright lines at the top
    of ``engine.js`` point at ``LICENSE.txt``, which is the project's
    ``LICENSE`` whole (its three MIT notices: this port's, upstream's and
    the Zurich port's, whose code is not in the page; and the note on the
    data's own terms).

    The file is read from beside the code (``CODE_ROOT``), not from the
    data root, which ``HKFR_ROOT`` may put anywhere.  A copy installed
    from a wheel has no ``LICENSE`` beside the package; the site is then
    written without it, with a warning.
    """
    import shutil

    shutil.copyfile(VENDOR_DIR / "LICENSE-leaflet.txt", site / "LICENSE-leaflet.txt")
    licence = CODE_ROOT / "LICENSE"
    if licence.exists():
        shutil.copyfile(licence, site / "LICENSE.txt")
    else:
        log.warning("no LICENSE at %s; the site is written without the MIT "
                    "notices its script refers to", licence)


def _default_trip(places: dict | None, points: dict) -> list[dict]:
    """Where the route page opens before anyone types (``config.DEFAULT_TRIP``).

    Each end is (label, place names to try in order, fallback), where the
    fallback is a neighborhood access point or a (lon, lat) pair for a spot
    the index does not carry.
    """
    out = []
    for label, names, fallback in DEFAULT_TRIP:
        hit = None
        if places:
            for want in names:
                for i, n in enumerate(places["names"]):
                    if n == want:
                        hit = {"label": label, "lon": places["lon"][i], "lat": places["lat"][i]}
                        break
                if hit:
                    break
        if hit is None and isinstance(fallback, tuple):
            hit = {"label": label, "lon": fallback[0], "lat": fallback[1]}
        if hit is None and fallback in points.get("walk", {}):
            lon, lat = points["walk"][fallback]
            hit = {"label": label, "lon": lon, "lat": lat}
        if hit is None:
            return []
        out.append(hit)
    return out


def _labels(ctx) -> list[dict]:
    """Sparse neighborhood labels for the route page's base map."""
    nb = ctx.neighborhoods.to_crs("EPSG:4326")
    out = []
    for _, r in nb.iterrows():
        p = r.geometry.representative_point()
        out.append({"n": r["neighborhood"], "lon": round(p.x, 5), "lat": round(p.y, 5)})
    return out


def _write_route_page(ctx, graph: dict, pts: dict) -> Path:
    """Pack the place index and hillshade with the graph; write both forms."""
    import hashlib
    import shutil

    from .download import ADDRESSES_PARQUET
    from .places import build_addresses, build_hillshade, build_places
    from .webgraph import bundle

    from .webgraph import EXPLORER_ONLY_ARRAYS

    strings = {"geom": graph["geom"]}
    # the route page trades distance against climbing and nothing else, so
    # it carries none of the arrays behind the explorer's other objectives
    arrays = {k: v for k, v in graph["arrays"].items()
              if k not in EXPLORER_ONLY_ARRAYS}
    places = None
    try:
        places = build_places()
    except FileNotFoundError as exc:
        log.warning("no place data (%s); the route page will search junctions only", exc)
    except Exception as exc:   # the search index is optional; the router is not
        log.warning("place index failed (%s: %s); the route page will search "
                    "junctions only", type(exc).__name__, exc)
    if places:
        strings["places"] = json.dumps(places, separators=(",", ":"), ensure_ascii=False)
    addr_meta = None
    try:
        addr = build_addresses() if ADDRESSES_PARQUET.exists() else None
    except Exception as exc:
        log.warning("no address index (%s: %s)", type(exc).__name__, exc)
        addr = None
    if addr:
        strings["addr_streets"] = json.dumps(addr["streets"], separators=(",", ":"),
                                             ensure_ascii=False)
        if addr.get("streets_zh"):
            strings["addr_streets_zh"] = json.dumps(
                addr["streets_zh"], separators=(",", ":"), ensure_ascii=False)
        for k in ("street", "number", "lon", "lat"):
            arrays["addr_" + k] = addr[k]
        addr_meta = {"origin": addr["origin"], "step": 1e-5}
    try:
        hillshade = build_hillshade()
    except Exception as exc:  # the page works without it, on streets alone
        log.warning("hillshade unavailable (%s)", exc)
        hillshade = None

    with step("bundling the route page payload", log):
        packed = bundle(dict(graph, arrays=arrays), strings)
    common = {
        "manifest": packed["manifest"], "meta": packed["meta"], "addr": addr_meta,
        "labels": _labels(ctx), "default": _default_trip(places, pts),
        "site": {"name": PRODUCT_NAME, "url": SITE_URL},
    }

    # one file that opens from disk: everything inline
    inline = dict(common, bundle=packed["b64"],
                  hillshade=hillshade and {"bounds": hillshade["bounds"],
                                           "data_uri": hillshade["data_uri"]})
    SIMPLE_HTML.write_text(_route_page_html(inline, linked=False), encoding="utf-8")

    # the static site: fetchable, cacheable files. Everything the page
    # references carries a content hash in its name, so a browser that cached
    # yesterday's app.js cannot run it against today's index.html.
    data_dir = SITE_DIR / "data"
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True)
    for stale in SITE_DIR.glob("*-*.??*"):
        if re.match(r"^(app|leaflet)-[0-9a-f]{10}\.(js|css)$", stale.name):
            stale.unlink()

    def hashed(stem: str, ext: str, text: str) -> str:
        name = f"{stem}-{hashlib.sha1(text.encode('utf-8')).hexdigest()[:10]}.{ext}"
        (SITE_DIR / name).write_text(text, encoding="utf-8")
        return name

    assets = {
        "app.css": hashed("app", "css", _asset("simple.css")),
        "app.js": hashed("app", "js", _asset("engine.js") + "\n" + _asset("simple.js")),
        "leaflet.css": hashed("leaflet", "css", _vendor("leaflet-1.9.4.css")),
        "leaflet.js": hashed("leaflet", "js", _vendor("leaflet-1.9.4.min.js")),
    }
    gz = packed["gz"]
    gz_name = f"graph-{hashlib.sha1(gz).hexdigest()[:10]}.bin.gz"
    (data_dir / gz_name).write_bytes(gz)
    linked = dict(common, bundle_url="data/" + gz_name, bundle_bytes=len(gz))
    if hillshade:
        png = hillshade["png"]
        png_name = f"hillshade-{hashlib.sha1(png).hexdigest()[:10]}.png"
        (data_dir / png_name).write_bytes(png)
        linked["hillshade"] = {"bounds": hillshade["bounds"], "url": "data/" + png_name}
    else:
        linked["hillshade"] = None
    SITE_INDEX.write_text(_route_page_html(linked, linked=True, assets=assets),
                          encoding="utf-8")
    for plain in ("app.css", "app.js", "leaflet.css", "leaflet.js"):
        (SITE_DIR / plain).unlink(missing_ok=True)
    (SITE_DIR / "favicon.svg").write_text(_FAVICON, encoding="utf-8")
    (SITE_DIR / ".nojekyll").write_text("", encoding="utf-8")
    (SITE_DIR / "robots.txt").write_text("User-agent: *\nAllow: /\n", encoding="utf-8")
    _copy_licences(SITE_DIR)
    # GitHub Pages reads a custom domain from here on branch deploys and
    # from Settings -> Pages on Actions deploys; shipping it covers both.
    # A project site (<user>.github.io/<repo>) must not carry one.
    if SITE_DOMAIN:
        (SITE_DIR / "CNAME").write_text(SITE_DOMAIN + "\n", encoding="utf-8")
    else:
        (SITE_DIR / "CNAME").unlink(missing_ok=True)
    site_bytes = sum(f.stat().st_size for f in SITE_DIR.rglob("*") if f.is_file())
    log.info("wrote %s (%s) and the site in %s (%s)", SIMPLE_HTML.name,
             human_bytes(SIMPLE_HTML.stat().st_size), SITE_DIR.name, human_bytes(site_bytes))
    return SIMPLE_HTML


def make_interactive_map(ctx, corridors, passes, barriers, pairs_df=None,
                         arc_store=None, basins=None) -> Path:
    """Assemble and write the interactive map.

    ``pairs_df`` and ``arc_store`` are accepted for call-site compatibility
    but are no longer needed: the map routes for itself rather than looking
    up precomputed answers.
    """
    from .webgraph import build_payload, bundle

    with step("preparing interactive map layers", log):
        # per-edge arrays that belong to the explorer's layers alone
        explorer_arrays: dict = {}
        layers = build_layers(ctx, corridors, passes, barriers, basins,
                              arrays=explorer_arrays)

    graph = build_payload(ctx.edges, ctx.directed)

    pts = {}
    for mode, gdf in ctx.points.items():
        g = gdf.to_crs("EPSG:4326")
        pts[mode] = {r["neighborhood"]: [round(r.geometry.x, 6),
                                         round(r.geometry.y, 6)]
                     for _, r in g.iterrows()}

    with step("bundling and compressing the map payload", log):
        # a copy of the array table: the route page packs ``graph`` too,
        # and has no use for the explorer's extras
        packed = bundle(dict(graph, arrays={**graph["arrays"], **explorer_arrays}), {
            "geom": graph["geom"],
            "layers": json.dumps(layers, separators=(",", ":"), ensure_ascii=False),
        })
    # The menus offer a neighbourhood only where the network reaches it.  A
    # planning area with no access point (outside a partial build's window,
    # or with no street inside it) still has a centroid, and upstream would
    # list it and snap that to whatever junction is nearest: in Hong Kong,
    # as likely as not, one across the water.
    shown = _points(ctx)
    names = sorted(set(shown.get("walk", {})) | set(shown.get("bike", {})))
    nb = ctx.neighborhoods
    zh = (dict(zip(nb["neighborhood"], nb["neighborhood_zh"].fillna("")))
          if "neighborhood_zh" in nb.columns else {})
    from .passes import BASIN_ELEV_M
    payload = {
        "manifest": packed["manifest"], "bundle": packed["b64"],
        "meta": packed["meta"], "points": shown, "neighborhood_names": names,
        "neighborhood_zh": {n: zh[n] for n in names if zh.get(n)},
        "examples": _examples(shown),
        # the height (mPD) below which a street counts as lowland, for the
        # basin layer's label
        "basin_elev_m": BASIN_ELEV_M,
    }

    with step("writing the explorer HTML", log):
        INTERACTIVE_HTML.parent.mkdir(parents=True, exist_ok=True)
        INTERACTIVE_HTML.write_text(_render(payload), encoding="utf-8")
    log.info("wrote %s (%s)", INTERACTIVE_HTML.name,
             human_bytes(INTERACTIVE_HTML.stat().st_size))
    with step("writing the route page", log):
        _write_route_page(ctx, graph, pts)
    return INTERACTIVE_HTML


def _points(ctx) -> dict:
    """Neighborhood access points per mode, as {name: [lon, lat]}."""
    pts = {}
    for mode, gdf in ctx.points.items():
        g = gdf[gdf["node"].notna()] if "node" in gdf.columns else gdf
        g = g.to_crs("EPSG:4326")
        pts[mode] = {r["neighborhood"]: [round(r.geometry.x, 6),
                                         round(r.geometry.y, 6)]
                     for _, r in g.iterrows()}
    return pts


def make_route_page(ctx) -> Path:
    """Build only the route finder: the one-file page and the static site.

    Nothing here needs the pair analysis, the corridors or the passes, so
    the site can be rebuilt straight after ``build-network`` (the ``site``
    command) without waiting for the rest.
    """
    from .webgraph import build_payload

    graph = build_payload(ctx.edges, ctx.directed)
    with step("writing the route page", log):
        return _write_route_page(ctx, graph, _points(ctx))
