"""Tests for the browser payload: quantisation, bundling, landmasses, parity."""
import base64
import gzip
import math

import numpy as np
import pandas as pd
import pytest

from hk_flat_routes.bikeways import stress as bike_stress
from hk_flat_routes.config import (BIKE, BIKE_PUSH_MULTIPLIER, CRS_PROJECTED,
                                   ELEVATION, GRADE_THRESHOLDS,
                                   MAX_EDGE_LENGTH_M, MIN_COMPONENT_NODES,
                                   MIN_RELIABLE_GRADE_LENGTH_M, PROCESSED_DIR)
from hk_flat_routes.webgraph import (CM, COORD_Q, DM, GRADE_Q, LANDMASSES,
                                     STRESS_Q, _b64, _i16, _u16,
                                     build_payload, bundle, encode_polyline)

_TH = [int(t * 100) for t in GRADE_THRESHOLDS]


def decode_polyline(s, precision=5):
    """Reference decoder, mirroring the JavaScript in the map."""
    factor = 10 ** precision
    index = lat = lon = 0
    out = []
    while index < len(s):
        for which in ("lat", "lon"):
            shift = result = 0
            while True:
                b = ord(s[index]) - 63
                index += 1
                result |= (b & 0x1f) << shift
                shift += 5
                if b < 0x20:
                    break
            d = ~(result >> 1) if (result & 1) else (result >> 1)
            if which == "lat":
                lat += d
            else:
                lon += d
        out.append((lon / factor, lat / factor))
    return out


# ------------------------------------------------------------- quantisation
def test_length_quantisation_covers_the_longest_edge():
    """No edge is left longer than MAX_EDGE_LENGTH_M (a trail or a
    catchwater path can run for kilometres between junctions, and the
    network builder cuts them); uint16 at 5 cm must reach that."""
    assert MAX_EDGE_LENGTH_M * DM < 65535
    q = _u16([MAX_EDGE_LENGTH_M - 0.1], DM, "arc length (m)")
    assert abs(q[0] / DM - (MAX_EDGE_LENGTH_M - 0.1)) <= 1.0 / DM


def test_length_quantisation_error_is_centimetre_scale():
    rng = np.random.default_rng(0)
    vals = rng.uniform(0, MAX_EDGE_LENGTH_M, 5000)
    back = _u16(vals, DM) / DM
    assert np.max(np.abs(back - vals)) <= 0.5 / DM + 1e-9


def test_climb_quantisation_covers_a_long_steep_edge():
    """A whole hillside in one edge: 400 m of climb is still centimetres."""
    assert 600 * CM < 65535
    back = _u16([412.34], CM, "arc climb (m)")[0] / CM
    assert abs(back - 412.34) < 0.01


def test_gradient_quantisation_is_signed_and_fine():
    for g in (-0.6, -0.315, 0.0, 0.0123, 0.315, 0.6):
        back = _i16([g], GRADE_Q)[0] / GRADE_Q
        assert abs(back - g) < 1e-4


def test_node_elevation_reaches_the_highest_ground():
    """Tai Mo Shan is 957 m; int16 at 5 cm goes to 1,638 m."""
    back = _i16([957.0, -3.2], DM, "node elevation (m)") / DM
    assert back.tolist() == pytest.approx([957.0, -3.2], abs=0.03)


def test_unnamed_quantisation_clamps_instead_of_wrapping():
    """A wrapped uint16 would silently invent a tiny value for a huge one.
    Display-only values (a gradient past the plausibility clip) clamp."""
    assert _u16([1e9], DM)[0] == 65535
    assert _u16([-5.0], DM)[0] == 0
    assert _i16([1e9], GRADE_Q)[0] == 32767
    assert _i16([-1e9], GRADE_Q)[0] == -32768


def test_an_over_long_edge_raises_instead_of_clamping():
    """Lengths and climbs feed the router's totals.  Clamped, a 4 km trail
    would count as 3.3 km in every route over it and nobody would know; so
    a value that does not fit is an error to be fixed where edges are cut."""
    with pytest.raises(OverflowError, match="arc length"):
        _u16([120.0, 4000.0], DM, "arc length (m)")
    with pytest.raises(OverflowError, match="arc climb"):
        _u16([700.0], CM, "arc climb (m)")
    with pytest.raises(OverflowError):
        _u16([-1.0], DM, "arc length (m)")
    with pytest.raises(OverflowError, match="node elevation"):
        _i16([2000.0], DM, "node elevation (m)")
    # just inside the range is fine
    assert _u16([65535 / DM], DM, "arc length (m)")[0] == 65535


def test_b64_round_trips_a_typed_array():
    arr = np.arange(1000, dtype="<i4")
    raw = base64.b64decode(_b64(arr))
    assert np.array_equal(np.frombuffer(raw, dtype="<i4"), arr)


# ------------------------------------------------------------------ polyline
def test_polyline_round_trips_within_precision():
    # Statue Square, Admiralty, Wan Chai, and back west to Sheung Wan
    coords = [(114.15995, 22.28105), (114.16540, 22.27912),
              (114.17281, 22.27760), (114.15012, 22.28688)]
    back = decode_polyline(encode_polyline(coords))
    assert len(back) == len(coords)
    for (x0, y0), (x1, y1) in zip(coords, back):
        assert abs(x0 - x1) < 2e-5 and abs(y0 - y1) < 2e-5


def test_polyline_handles_repeated_and_negative_deltas():
    coords = [(114.2, 22.3), (114.2, 22.3), (114.1, 22.2)]
    back = decode_polyline(encode_polyline(coords))
    assert len(back) == 3
    assert back[1] == pytest.approx(back[0])


# -------------------------------------------------------------------- bundle
def test_bundle_round_trips_arrays_and_strings():
    arrays = {
        "a": np.arange(50, dtype="<i4"),
        "b": np.full(30, 7, dtype="<u2"),
        "c": np.array([-3, 0, 9], dtype="<i2"),
    }
    # street names go in as UTF-8, Chinese as itself
    out = bundle({"arrays": arrays, "meta": {"hello": "world"}},
                 {"txt": "some text", "more": "皇后大道中 Queen's Road Central"})
    raw = gzip.decompress(base64.b64decode(out["b64"]))
    assert len(raw) == out["manifest"]["bytes"]
    for name, arr in arrays.items():
        m = out["manifest"]["arrays"][name]
        view = np.frombuffer(raw, dtype="<" + m["t"], count=m["n"],
                             offset=m["o"])
        assert np.array_equal(view, arr)
    for name, text in (("txt", "some text"),
                       ("more", "皇后大道中 Queen's Road Central")):
        m = out["manifest"]["strings"][name]
        assert raw[m["o"]:m["o"] + m["b"]].decode("utf-8") == text
    assert out["meta"] == {"hello": "world"}


def test_bundle_offsets_are_contiguous_and_ordered():
    arrays = {f"a{i}": np.arange(10 + i, dtype="<i4") for i in range(5)}
    out = bundle({"arrays": arrays, "meta": {}}, {"s": "x" * 17})
    entries = sorted(
        [(m["o"], m["n"] * 4) for m in out["manifest"]["arrays"].values()]
        + [(m["o"], m["b"]) for m in out["manifest"]["strings"].values()])
    pos = 0
    for off, nbytes in entries:
        assert off == pos
        pos += nbytes
    assert pos == out["manifest"]["bytes"]


def test_bundle_rejects_an_unsupported_dtype():
    with pytest.raises(TypeError):
        bundle({"arrays": {"bad": np.array(["x"], dtype=object)}, "meta": {}}, {})


def test_bundle_actually_compresses():
    """The arrays are mostly small integers with long zero runs."""
    arrays = {"z": np.zeros(200_000, dtype="<u2")}
    out = bundle({"arrays": arrays, "meta": {}}, {})
    assert len(base64.b64decode(out["b64"])) < 0.05 * 400_000


# ---------------------------------------------------------------- landmasses
# A small Hong Kong: a ring of streets on the island, a larger ring across
# the harbour, and a stray walkway joined to neither.  Rings are laid round
# the points webgraph.LANDMASSES names its components by.
_ISLAND = next(p for p in LANDMASSES if p[0] == "Hong Kong Island")
_KOWLOON = next(p for p in LANDMASSES if p[0].startswith("Kowloon"))
_N_KOWLOON, _N_ISLAND, _N_SCRAP = MIN_COMPONENT_NODES + 10, MIN_COMPONENT_NODES, 6
#: a district of the island that bicycles can enter and not leave
_N_DISTRICT = MIN_COMPONENT_NODES + 1


def _mini_network(long_edge_m: float | None = None, district: bool = False):
    """(edges GeoDataFrame, directed table) for the three rings.

    The island ring is walkable but closed to bicycles on one edge in one
    direction, so the two modes' graphs differ.  ``long_edge_m`` overrides
    the stored length of the first edge.

    With ``district``, a fourth ring W stands beside the island ring, 2 km
    from its reference point, joined to it by a street H0 - T - W0 that
    bicycles may ride towards W only (Wan Chai North by bicycle), and the
    first edge of that street is a pushed link.
    """
    import geopandas as gpd
    from pyproj import Transformer
    from shapely.geometry import LineString

    to_grid = Transformer.from_crs("EPSG:4326", CRS_PROJECTED, always_xy=True)
    edge_rows, arc_rows = [], []

    def ring(prefix, n, lon0, lat0, name, name_zh, cls="residential"):
        pts = []
        for k in range(n):
            a = 2 * math.pi * k / n
            lon = lon0 + 0.0030 * math.cos(a)
            lat = lat0 + 0.0028 * math.sin(a)
            pts.append(to_grid.transform(lon, lat))
        for k in range(n):
            eid = len(edge_rows)
            u, v = f"{prefix}{k}", f"{prefix}{(k + 1) % n}"
            geom = LineString([pts[k], pts[(k + 1) % n]])
            length = geom.length
            if eid == 0 and long_edge_m is not None:
                length = long_edge_m
            rise = 0.6 * math.sin(2 * math.pi * k / n)      # a gentle roll
            edge_rows.append({
                "edge_id": eid, "u": u, "v": v, "geometry": geom, "cls": cls,
                "name": name, "name_zh": name_zh, "length_m": length,
                "max_abs_grade": abs(rise) / geom.length,
                "avg_grade_fwd": rise / geom.length,
                "cum_gain_fwd": max(rise, 0.0), "low_stress": False,
            })
            for direction, a, b, dz in (("fwd", u, v, rise), ("rev", v, u, -rise)):
                row = {
                    "edge_id": eid, "from_node": a, "to_node": b,
                    "direction": direction, "length_m": length, "cls": cls,
                    "cum_gain": max(dz, 0.0), "cum_loss": max(-dz, 0.0),
                    "max_grade": dz / geom.length,
                    "mean_abs_grade": abs(dz) / geom.length,
                    "start_elev": 5.0, "end_elev": 5.0 + dz,
                    "walk_traversable": True, "bike_traversable": True,
                }
                for t in _TH:
                    row[f"d_above_{t}"] = 0.0
                arc_rows.append(row)

    ring("K", _N_KOWLOON, _KOWLOON[1], _KOWLOON[2], "Nathan Road", "彌敦道")
    ring("H", _N_ISLAND, _ISLAND[1], _ISLAND[2], "Queen's Road Central", "皇后大道中")
    # a walkway out at sea, 4 km from either reference point; no Chinese name
    ring("X", _N_SCRAP, 114.2150, 22.3000, "Podium Walk", None, cls="footway")

    if district:
        ring("W", _N_DISTRICT, _ISLAND[1] + 0.0200, _ISLAND[2], "Expo Drive", "博覽道")
        where = {r["u"]: r["geometry"].coords[0] for r in edge_rows}
        mid = ((where["H0"][0] + where["W0"][0]) / 2, where["H0"][1] + 40.0)
        for u, v, a, b in (("H0", "T", where["H0"], mid), ("T", "W0", mid, where["W0"])):
            eid = len(edge_rows)
            geom = LineString([a, b])
            edge_rows.append({
                "edge_id": eid, "u": u, "v": v, "geometry": geom,
                "cls": "footway" if u == "H0" else "residential",
                "name": "Fleming Road", "name_zh": "菲林明道",
                "length_m": geom.length, "max_abs_grade": 0.0,
                "avg_grade_fwd": 0.0, "cum_gain_fwd": 0.0, "low_stress": False,
                "bike_push": u == "H0"})
            for direction, p, q in (("fwd", u, v), ("rev", v, u)):
                row = {"edge_id": eid, "from_node": p, "to_node": q,
                       "direction": direction, "length_m": geom.length,
                       "cls": edge_rows[-1]["cls"], "cum_gain": 0.0,
                       "cum_loss": 0.0, "max_grade": 0.0, "mean_abs_grade": 0.0,
                       "start_elev": 5.0, "end_elev": 5.0,
                       "walk_traversable": True,
                       "bike_traversable": direction == "fwd",
                       "bike_push": u == "H0"}
                for t in _TH:
                    row[f"d_above_{t}"] = 0.0
                arc_rows.append(row)

    edges = gpd.GeoDataFrame(edge_rows, geometry="geometry", crs=CRS_PROJECTED)
    directed = pd.DataFrame(arc_rows)
    if district:
        edges["bike_push"] = edges["bike_push"].fillna(False).astype(bool)
        directed["bike_push"] = directed["bike_push"].fillna(False).astype(bool)
    # one island edge is one-way for bicycles: the ring stays connected for
    # them (the long way round), but the modes no longer have the same arcs
    first_island = _N_KOWLOON
    shut = (directed["edge_id"] == first_island) & (directed["direction"] == "rev")
    directed.loc[shut, "bike_traversable"] = False
    return edges, directed


@pytest.fixture(scope="module")
def mini():
    edges, directed = _mini_network()
    return edges, directed, build_payload(edges, directed)


def _node_index(payload, edges) -> dict:
    """Node id -> row in the payload's node arrays, found by position."""
    ll = edges.to_crs("EPSG:4326")
    where = {}
    for u, v, geom in zip(edges["u"], edges["v"], ll.geometry):
        where[u] = geom.coords[0]
        where[v] = geom.coords[-1]
    lon, lat = payload["arrays"]["node_lon"], payload["arrays"]["node_lat"]
    rows = {(int(x), int(y)): i for i, (x, y) in enumerate(zip(lon, lat))}
    return {n: rows[(round(x * COORD_Q), round(y * COORD_Q))]
            for n, (x, y) in where.items()}


def test_two_landmasses_are_both_kept_and_a_fragment_is_dropped(mini):
    edges, _directed, payload = mini
    comp = payload["arrays"]["node_comp_walk"]
    idx = _node_index(payload, edges)
    by_prefix = {p: {int(comp[i]) for n, i in idx.items() if n.startswith(p)}
                 for p in "KHX"}
    # rank + 1, largest first; 0 for a node of no kept component
    assert by_prefix == {"K": {1}, "H": {2}, "X": {0}}
    assert np.bincount(comp).tolist() == [_N_SCRAP, _N_KOWLOON, _N_ISLAND]


def test_node_flags_say_which_modes_can_start_at_a_node(mini):
    edges, _directed, payload = mini
    a = payload["arrays"]
    for bit, mode in enumerate(("walk", "bike")):
        usable = (a["node_flags"] >> bit) & 1
        assert np.array_equal(usable, (a[f"node_comp_{mode}"] > 0).astype(usable.dtype))


def test_landmasses_are_named_for_the_page(mini):
    """The page says "the origin is on Hong Kong Island": the names come
    from here, in rank order per mode."""
    _edges, _directed, payload = mini
    names = payload["meta"]["components"]
    assert names["walk"] == [_KOWLOON[0], _ISLAND[0]]
    assert names["bike"] == [_KOWLOON[0], _ISLAND[0]]


def test_no_arc_joins_two_landmasses_and_fragment_arcs_are_gone(mini):
    edges, directed, payload = mini
    a = payload["arrays"]
    n = payload["meta"]["n_nodes"]
    tail = np.repeat(np.arange(n), np.diff(a["indptr"]))
    head = a["arc_head"]
    for bit, mode in enumerate(("walk", "bike")):
        on = ((a["arc_flags"] >> bit) & 1).astype(bool)
        comp = a[f"node_comp_{mode}"]
        assert (comp[tail[on]] > 0).all()
        assert np.array_equal(comp[tail[on]], comp[head[on]])
    # the stray walkway's arcs are in neither mode, so they are not packed
    kept_edges = set(a["arc_edge"].tolist())
    scrap_edges = set(edges.index[edges["u"].str.startswith("X")])
    assert not kept_edges & scrap_edges
    assert payload["meta"]["n_arcs"] == 2 * (_N_KOWLOON + _N_ISLAND)
    # ... but its geometry is still there to be drawn
    assert payload["meta"]["n_edges"] == len(edges)


def test_per_mode_components_agree_with_the_python_router(mini):
    """Anything the browser can route, Python can route identically: both
    take their components from routing.component_ranks."""
    from hk_flat_routes.routing import build_route_graph

    edges, directed, payload = mini
    idx = _node_index(payload, edges)
    a = payload["arrays"]
    for bit, mode in enumerate(("walk", "bike")):
        g = build_route_graph(directed, mode)
        comp = a[f"node_comp_{mode}"]
        for node, i in idx.items():
            j = g.index_of(node)
            assert int(comp[i]) == (0 if j is None else int(g.node_comp[j]) + 1), node
        # and the same arcs: every arc Python keeps is flagged, and no more
        assert int(((a["arc_flags"] >> bit) & 1).sum()) == g.n_arcs
    # the one-way island edge is the modes' only difference
    walk_only = ((a["arc_flags"] & 1) == 1) & ((a["arc_flags"] & 2) == 0)
    assert int(walk_only.sum()) == 1


def test_landmass_sizes_are_shipped_beside_their_names(mini):
    _edges, _directed, payload = mini
    meta = payload["meta"]
    assert meta["component_nodes"] == {"walk": [_N_KOWLOON, _N_ISLAND],
                                       "bike": [_N_KOWLOON, _N_ISLAND]}
    for mode in ("walk", "bike"):
        assert len(meta["component_nodes"][mode]) == len(meta["components"][mode])


def test_a_bundle_without_pushed_links_says_so(mini):
    """The directed table of an older build has no ``bike_push`` column:
    the array is there, all zeros, and the multiplier beside it."""
    _edges, directed, payload = mini
    assert "bike_push" not in directed.columns
    push = payload["arrays"]["arc_push"]
    assert push.dtype == np.uint8 and len(push) == payload["meta"]["n_arcs"]
    assert not push.any()
    assert payload["meta"]["push_multiplier"] == BIKE_PUSH_MULTIPLIER


def test_the_page_is_told_what_a_steepest_pitch_is_made_of(mini):
    """engine.js builds a route's steepest pitch as routing.arc_pitch does:
    an arc's stored maximum from 15 m up, its net climb over its length,
    held to the ceiling, below that.  The length and the ceiling travel in
    the bundle, and the stored maximum is each arc's own, untouched."""
    _edges, directed, payload = mini
    meta = payload["meta"]
    assert meta["min_reliable_grade_length_m"] == MIN_RELIABLE_GRADE_LENGTH_M
    assert meta["max_plausible_grade"] == ELEVATION.max_plausible_grade
    a = payload["arrays"]
    back = ((a["arc_flags"] >> 2) & 1).astype(bool)
    want = directed.set_index(["edge_id", directed["direction"] == "rev"])["max_grade"]
    for e, r, g in zip(a["arc_edge"], back, a["arc_maxgrade"] / GRADE_Q):
        assert g == pytest.approx(want[(int(e), bool(r))], abs=0.5 / GRADE_Q)


def test_an_arc_just_under_the_reliable_length_does_not_round_up_to_it():
    """The page knows an arc's length in 5 cm units and reads its steepest
    pitch from the sampled maximum once that says 15 m.  An arc of 14.98 m
    rounds to 15.00, and would be judged on a maximum the analysis sets
    aside; it is packed one unit short instead."""
    edges, directed = _mini_network()
    lengths = {3: MIN_RELIABLE_GRADE_LENGTH_M - 0.02,     # rounds up to 15.00
               4: MIN_RELIABLE_GRADE_LENGTH_M - 0.03,     # rounds to 14.95
               5: MIN_RELIABLE_GRADE_LENGTH_M}
    for eid, length in lengths.items():
        directed.loc[directed["edge_id"] == eid, "length_m"] = length
    a = build_payload(edges, directed)["arrays"]
    packed = {eid: set((a["arc_len"][a["arc_edge"] == eid] / DM).tolist())
              for eid in lengths}
    below = MIN_RELIABLE_GRADE_LENGTH_M - 1.0 / DM
    assert packed == {3: {below}, 4: {below}, 5: {MIN_RELIABLE_GRADE_LENGTH_M}}
    # every arc falls on the side of the threshold its true length is on
    true = directed.set_index(["edge_id", directed["direction"] == "rev"])["length_m"]
    back = ((a["arc_flags"] >> 2) & 1).astype(bool)
    for e, r, q in zip(a["arc_edge"], back, a["arc_len"] / DM):
        assert (q >= MIN_RELIABLE_GRADE_LENGTH_M) == \
            (true[(int(e), bool(r))] >= MIN_RELIABLE_GRADE_LENGTH_M)


# --------------------------------------------- a district attached one way
@pytest.fixture(scope="module")
def attached():
    edges, directed = _mini_network(district=True)
    return edges, directed, build_payload(edges, directed)


def test_a_one_way_district_is_on_its_neighbours_landmass(attached):
    """By bicycle the district can be entered from the island ring and not
    left.  It is the island's landmass all the same, link included, and the
    page must not call it another island."""
    edges, _directed, payload = attached
    a = payload["arrays"]
    idx = _node_index(payload, edges)
    for mode in ("walk", "bike"):
        comp = a[f"node_comp_{mode}"]
        by_prefix = {p: {int(comp[i]) for n, i in idx.items() if n.startswith(p)}
                     for p in "KHWTX"}
        # the island with its district outgrows the Kowloon ring
        assert by_prefix == {"K": {2}, "H": {1}, "W": {1}, "T": {1}, "X": {0}}, mode
    size = _N_ISLAND + _N_DISTRICT + 1
    assert payload["meta"]["component_nodes"]["bike"] == [size, _N_KOWLOON]
    assert payload["meta"]["components"]["bike"] == [_ISLAND[0], _KOWLOON[0]]
    # the link's two arcs are flagged for bicycles in the one direction
    tail = np.repeat(np.arange(payload["meta"]["n_nodes"]), np.diff(a["indptr"]))
    t = idx["T"]
    bike = ((a["arc_flags"] >> 1) & 1).astype(bool)
    assert sorted(zip(tail[bike & ((tail == t) | (a["arc_head"] == t))].tolist(),
                      a["arc_head"][bike & ((tail == t) | (a["arc_head"] == t))].tolist())) \
        == sorted([(idx["H0"], t), (t, idx["W0"])])


def test_the_one_way_district_agrees_with_the_python_router(attached):
    from hk_flat_routes.routing import build_route_graph

    edges, directed, payload = attached
    idx = _node_index(payload, edges)
    a = payload["arrays"]
    for bit, mode in enumerate(("walk", "bike")):
        g = build_route_graph(directed, mode)
        comp = a[f"node_comp_{mode}"]
        for node, i in idx.items():
            j = g.index_of(node)
            assert int(comp[i]) == (0 if j is None else int(g.node_comp[j]) + 1), node
        assert int(((a["arc_flags"] >> bit) & 1).sum()) == g.n_arcs


def test_pushed_links_are_marked_arc_by_arc(attached):
    edges, directed, payload = attached
    a = payload["arrays"]
    pushed = set(a["arc_edge"][a["arc_push"] == 1].tolist())
    assert pushed == set(edges.index[edges["bike_push"]])
    # the mark is on the arc a bicycle can take; the other way along the
    # same footway is open to walkers alone, and is nobody's pushed link
    bike = ((a["arc_flags"] >> 1) & 1).astype(bool)
    assert int(a["arc_push"].sum()) == 1 and bike[a["arc_push"] == 1].all()
    assert int((a["arc_edge"] == next(iter(pushed))).sum()) == 2
    assert payload["meta"]["push_multiplier"] == BIKE_PUSH_MULTIPLIER
    # the comfort weight shipped for the edge is its class's alone, a byte:
    # the engine multiplies the push factor in from ``arc_push``, and a
    # weight that already held it would be charged nine times over
    on_link = edges.index[edges["bike_push"]][0]
    stress = a["edge_stress"] / STRESS_Q
    assert a["edge_stress"].dtype == np.uint8
    assert stress[on_link] == pytest.approx(1.3)
    assert np.allclose(stress, bike_stress(edges, push=False))


@pytest.fixture(scope="module")
def engine_on_attached(attached):
    """``web/engine.js`` in headless Chromium, holding the district payload."""
    from pathlib import Path

    from hk_flat_routes import webgraph
    sync_api = pytest.importorskip("playwright.sync_api",
                                   reason="playwright is not installed")
    payload = attached[2]
    packed = bundle(payload, {"geom": payload["geom"]})
    source = Path(webgraph.__file__).with_name("web") / "engine.js"
    errors: list = []
    with sync_api.sync_playwright() as pw:
        try:
            browser = pw.chromium.launch(args=["--no-sandbox", "--disable-gpu"])
        except Exception as exc:                            # pragma: no cover
            pytest.skip(f"no usable Chromium: {exc}")
        page = browser.new_page()
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.set_content("<!DOCTYPE html><html><body></body></html>")
        page.add_script_tag(content=source.read_text(encoding="utf-8"))
        page.evaluate("""async ([b64, manifest, meta]) => {
            window.g = new Graph(new Bundle(await inflate(b64), manifest), meta); }""",
                      [packed["b64"], packed["manifest"], packed["meta"]])
        assert not errors, errors
        yield page
        browser.close()


def test_the_engine_prices_every_arc_as_python_does(attached, engine_on_attached):
    """The payload built here, read by the real engine: each bicycle arc
    costs what ``routing.edge_costs`` says under the class multipliers and
    under pure distance, and its comfort-weighted length is its length
    times ``bikeways.stress``.  The pushed link is a kilometre long, so a
    push multiplier left out by one side, or put in by both, is off by
    kilometres: the bundle carries the class weight and ``arc_push``, and
    the engine alone multiplies them."""
    from hk_flat_routes.config import ROUTING_PROFILES
    from hk_flat_routes.routing import build_route_graph, edge_costs

    edges, directed, _payload = attached
    t = build_route_graph(directed, "bike").table
    rows = [[int(e), d == "rev"] for e, d in zip(t["edge_id"], t["direction"])]
    got = engine_on_attached.evaluate("""(rows) => {
        const p = g.meta.profiles;
        const calm = Object.assign({}, p.shortest, { stress: true });
        return rows.map(([e, rev]) => { const a = g.arcOf(e, rev); return a < 0 ? null : {
            push: g.arcPush[a], balanced: g.pathCost([a], 'bike', p.balanced),
            shortest: g.pathCost([a], 'bike', p.shortest),
            calm: g.pathCost([a], 'bike', calm), calmLen: g.arcLenStress[a] / g.DM }; }); }""",
                                      rows)
    assert None not in got
    length = t["length_m"].to_numpy()
    want = {"balanced": edge_costs(t, ROUTING_PROFILES["balanced"], "bike"),
            "shortest": edge_costs(t, ROUTING_PROFILES["shortest"], "bike"),
            "calm": length * bike_stress(edges)[t["edge_id"].to_numpy()]}
    pushed = t["bike_push"].to_numpy()
    assert pushed.sum() == 1 and length[pushed][0] > 900.0
    # the link itself, spelt out: three times over with either kind of
    # comfort weight, its own length without
    link = int(np.flatnonzero(pushed)[0])
    footway = BIKE.class_multiplier.get("footway", 1.0)
    assert want["balanced"][link] == pytest.approx(
        length[link] * footway * BIKE_PUSH_MULTIPLIER)
    assert want["calm"][link] == pytest.approx(length[link] * 1.3 * BIKE_PUSH_MULTIPLIER)
    assert want["shortest"][link] == pytest.approx(length[link])
    for i, o in enumerate(got):
        assert o["push"] == int(pushed[i])
        for key in ("balanced", "shortest", "calm"):
            # lengths travel in 5 cm units and climbs in centimetres
            assert o[key] == pytest.approx(want[key][i], abs=0.5), (rows[i], key)
        assert o["calmLen"] == pytest.approx(o["calm"])


def test_a_pushed_link_on_a_fragment_bicycles_cannot_reach_is_not_marked():
    """Cycle-track stubs joined by a pushed footway can still add up to a
    piece too small to keep.  Walkers reach it from the street; no bicycle
    route can use it, and the page counts on every marked arc being one a
    bicycle can take."""
    from shapely.geometry import LineString

    edges, directed = _mini_network()
    where = {r["u"]: r["geometry"].coords[0] for _, r in edges.iterrows()}
    x, y = where["K0"]
    stub = [("K0", "P0", "footway", False, False), ("P0", "P1", "footway", True, True),
            ("P1", "P2", "cycleway", True, False)]
    at = {"K0": (x, y), "P0": (x + 30, y), "P1": (x + 60, y), "P2": (x + 90, y)}
    edge_rows, arc_rows = [], []
    for k, (u, v, cls, by_bike, pushed) in enumerate(stub):
        eid = len(edges) + k
        edge_rows.append({**edges.iloc[0].to_dict(), "edge_id": eid, "u": u, "v": v,
                          "geometry": LineString([at[u], at[v]]), "cls": cls,
                          "length_m": 30.0, "bike_push": pushed})
        for direction, a, b in (("fwd", u, v), ("rev", v, u)):
            arc_rows.append({**directed.iloc[0].to_dict(), "edge_id": eid,
                             "from_node": a, "to_node": b, "direction": direction,
                             "length_m": 30.0, "cls": cls, "walk_traversable": True,
                             "bike_traversable": by_bike, "bike_push": pushed})
    edges = pd.concat([edges.assign(bike_push=False),
                       type(edges)(edge_rows, geometry="geometry", crs=edges.crs)],
                      ignore_index=True)
    directed = pd.concat([directed.assign(bike_push=False), pd.DataFrame(arc_rows)],
                         ignore_index=True)
    a = build_payload(edges, directed)["arrays"]
    link = a["arc_edge"] == edges.index[edges["bike_push"]][0]
    # both arcs of the link are there for walkers, and neither for bicycles
    assert int(link.sum()) == 2 and (a["arc_flags"][link] & 3 == 1).all()
    assert int(a["arc_push"].sum()) == 0


def test_an_unnamed_bicycle_landmass_takes_the_walkers_name():
    """Bicycles shut out of the link altogether: the district is a landmass
    of its own for them, 2 km from any reference point, but it is a piece of
    the walker's Hong Kong Island and is called that.  A walker's landmass
    with no reference point near it stays unnamed."""
    edges, directed = _mini_network(district=True)
    link = directed["from_node"].isin(["T"]) | directed["to_node"].isin(["T"])
    directed.loc[link, "bike_traversable"] = False
    meta = build_payload(edges, directed)["meta"]
    assert meta["components"]["walk"] == [_ISLAND[0], _KOWLOON[0]]
    assert meta["components"]["bike"] == [_KOWLOON[0], _ISLAND[0], _ISLAND[0]]
    assert meta["component_nodes"]["bike"] == [_N_KOWLOON, _N_DISTRICT, _N_ISLAND]


def test_a_blocks_climb_per_km_does_not_depend_on_which_way_it_is_drawn(mini):
    """The explorer's "climb per km" of a block is the mean of the two
    directions of travel.  Upstream ships the forward climb, which is zero
    on every block that happens to be drawn downhill."""
    edges, directed, _payload = mini
    rise = edges["avg_grade_fwd"] * edges.geometry.length
    both = edges.assign(cum_loss_fwd=(-rise).clip(lower=0.0))
    km = edges["length_m"] / 1000.0
    got = build_payload(both, directed)["arrays"]["edge_gainkm"] / DM
    assert np.allclose(got, rise.abs() / 2.0 / km, atol=0.5 / DM)
    downhill = (rise < -0.3).to_numpy()
    assert downhill.any() and (got[downhill] > 0.5).all()
    # an edge table that carries no descent is read as upstream reads it
    old = _payload["arrays"]["edge_gainkm"] / DM
    assert np.allclose(old, edges["cum_gain_fwd"] / km, atol=0.5 / DM)
    assert (old[downhill] == 0.0).all()


def test_an_over_long_edge_fails_the_build():
    """End to end: the packer refuses a network it would misreport."""
    edges, directed = _mini_network(long_edge_m=4000.0)
    with pytest.raises(OverflowError, match="length"):
        build_payload(edges, directed)


def test_chinese_names_align_with_the_name_table(mini):
    edges, _directed, payload = mini
    meta = payload["meta"]
    assert len(meta["names_zh"]) == len(meta["names"])
    zh = dict(zip(meta["names"], meta["names_zh"]))
    assert zh == {"Nathan Road": "彌敦道", "Queen's Road Central": "皇后大道中",
                  "Podium Walk": ""}
    # every edge's name index points at its own name (0 = unnamed)
    idx = payload["arrays"]["edge_name"]
    assert [meta["names"][i - 1] for i in idx] == edges["name"].tolist()


# ------------------------------------------------------------ lowland basins
def test_basin_membership_is_the_same_however_the_table_is_read():
    """The explorer marks basin streets in its own copy of the network (one
    number per edge) instead of drawing them a second time.  The basin table
    reaches it three ways: with an ``edge_id`` per row, with no such column
    (a run made before it was written), and, through ``passes.read_basins``,
    with the column there but empty for such a run.  The last once matched
    nothing, and the layer vanished from the page without an error."""
    from hk_flat_routes.viz_interactive import _basin_membership

    edges, _directed = _mini_network()
    # two "basins": the first six Kowloon edges and four island edges
    rows = edges.iloc[list(range(6)) + list(range(_N_KOWLOON, _N_KOWLOON + 4))]
    basins = rows[["edge_id", "name", "length_m", "geometry"]].copy()
    basins["basin"] = [0] * 6 + [1] * 4
    basins["basin_label"] = ["Kowloon flats"] * 6 + ["North shore"] * 4

    want = np.zeros(len(edges), dtype=int)
    want[:6] = 1
    want[_N_KOWLOON:_N_KOWLOON + 4] = 2
    empty = basins.copy()
    empty["edge_id"] = pd.Series([None] * len(empty), index=empty.index, dtype="object")
    for table in (basins, basins.drop(columns=["edge_id"]), empty):
        member, out = _basin_membership(edges, table)
        assert member.astype(int).tolist() == want.tolist()
        assert [r["basin_label"] for r in out] == ["Kowloon flats", "North shore"]
        assert out[0]["length_km"] == pytest.approx(rows["length_m"].iloc[:6].sum() / 1000, abs=1e-3)
    # nothing to mark: no rows, and a byte per edge of zeros
    member, out = _basin_membership(edges, basins.iloc[0:0])
    assert out == [] and not member.any() and len(member) == len(edges)
    member, out = _basin_membership(edges, None)
    assert out == [] and not member.any()


# ------------------------------------------------------ parity on real data
_real = pytest.mark.skipif(
    not (PROCESSED_DIR / "edges_directed.parquet").exists()
    or not (PROCESSED_DIR / "edges_metrics.parquet").exists(),
    reason="processed data not built")


@pytest.fixture(scope="module")
def real():
    import geopandas as gpd

    edges = gpd.read_parquet(PROCESSED_DIR / "edges_metrics.parquet")
    directed = pd.read_parquet(PROCESSED_DIR / "edges_directed.parquet")
    return edges, directed, build_payload(edges, directed)


@_real
def test_real_components_match_the_python_router(real):
    """On the built network: the same landmasses, node for node, per mode."""
    from hk_flat_routes.routing import build_route_graph

    _edges, directed, payload = real
    nodes = pd.unique(pd.concat([directed["from_node"], directed["to_node"]],
                                ignore_index=True))
    a = payload["arrays"]
    assert len(nodes) == payload["meta"]["n_nodes"]
    for bit, mode in enumerate(("walk", "bike")):
        g = build_route_graph(directed, mode)
        rank = pd.Series(g.node_comp + 1, index=g.node_ids)
        want = rank.reindex(nodes).fillna(0).to_numpy().astype(int)
        assert np.array_equal(a[f"node_comp_{mode}"].astype(int), want)
        assert int(((a["arc_flags"] >> bit) & 1).sum()) == g.n_arcs
        assert len(payload["meta"]["components"][mode]) == int(want.max())


@_real
def test_real_network_has_more_than_one_landmass(real):
    """The reason for all of the above.  Kowloon and Hong Kong Island are in
    every build that covers the harbour; a build of one side has one."""
    _edges, _directed, payload = real
    names = payload["meta"]["components"]["walk"]
    comp = payload["arrays"]["node_comp_walk"]
    sizes = np.bincount(comp)[1:]
    assert (sizes >= MIN_COMPONENT_NODES).all() or len(sizes) == 1
    assert sizes.tolist() == sorted(sizes.tolist(), reverse=True)
    if {"Hong Kong Island", "Kowloon and the New Territories"} <= set(names):
        assert names.index("Hong Kong Island") != \
            names.index("Kowloon and the New Territories")


@_real
def test_real_lengths_and_climbs_are_not_clamped(real):
    """No edge sits at the top of the uint16 range, and the packed totals
    are the table's totals to within the quantum."""
    edges, _directed, payload = real
    a = payload["arrays"]
    assert int(a["edge_len"].max()) < 65535
    assert int(a["arc_len"].max()) < 65535 and int(a["arc_gain"].max()) < 65535
    assert edges["length_m"].max() <= MAX_EDGE_LENGTH_M + 1.0
    total = a["edge_len"].astype(np.float64).sum() / DM
    assert total == pytest.approx(edges["length_m"].sum(), abs=len(edges) * 0.5 / DM)


@_real
def test_real_chinese_names_align(real):
    edges, _directed, payload = real
    meta = payload["meta"]
    assert len(meta["names_zh"]) == len(meta["names"])
    assert meta["names"] == sorted(meta["names"])
    if "name_zh" in edges.columns:
        both = edges[["name", "name_zh"]].dropna()
        pairs = set(zip(both["name"], both["name_zh"]))
        zh = dict(zip(meta["names"], meta["names_zh"]))
        # each Chinese name shipped is one the data pairs with that street
        assert all((n, z) in pairs for n, z in zh.items() if z)
        # and most named streets have one
        assert sum(1 for z in meta["names_zh"] if z) > 0.5 * len(meta["names"])
