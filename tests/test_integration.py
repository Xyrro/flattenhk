"""Integration checks against the real processed data.

These are skipped automatically when the pipeline has not been run, so the
unit suite stays runnable on a fresh clone.  They read whatever build the
configuration points at, so setting ``HKFR_RUN_DIR`` runs them against a run
directory instead of the main build:

    HKFR_RUN_DIR=/path/to/run python -m pytest tests/test_integration.py

A run need not cover the whole territory.  A reference street that lies
outside the run skips its test; it never fails it.

The last section needs no build at all: it puts the tools the validation
measures real streets with through a street small enough to check by hand.
"""
import numpy as np
import pandas as pd
import pytest

from hk_flat_routes.config import ELEVATION, PROCESSED_DIR

EDGES = PROCESSED_DIR / "edges_metrics.parquet"
DIRECTED = PROCESSED_DIR / "edges_directed.parquet"
PROFILES = PROCESSED_DIR / "edge_profiles.npz"
PAIRS = PROCESSED_DIR / "neighborhood_pairs.parquet"

_NO_BUILD = (f"no processed data in {PROCESSED_DIR}; run "
             "`python -m hk_flat_routes all`")


@pytest.fixture(scope="module")
def edges():
    import geopandas as gpd
    if not (EDGES.exists() and DIRECTED.exists()):
        pytest.skip(_NO_BUILD)
    return gpd.read_parquet(EDGES)


@pytest.fixture(scope="module")
def directed():
    if not (EDGES.exists() and DIRECTED.exists()):
        pytest.skip(_NO_BUILD)
    return pd.read_parquet(DIRECTED)


@pytest.fixture(scope="module")
def profiles():
    """Edge id -> (distance, elevation), as the pipeline caches them."""
    if not PROFILES.exists():
        pytest.skip("elevation profiles not cached")
    npz = np.load(PROFILES)
    off, dist, elev = npz["offsets"], npz["dist"], npz["elev"]
    return {int(e): (dist[off[k]:off[k + 1]], elev[off[k]:off[k + 1]])
            for k, e in enumerate(npz["edge_ids"])}


@pytest.fixture(scope="module")
def marks():
    """The official bench marks, if they have been downloaded."""
    from hk_flat_routes.validate import load_benchmarks
    m = load_benchmarks()
    if m.empty:
        pytest.skip("bench marks not downloaded")
    return m


def _street(edges, name):
    """Edges of a street, or skip the test when the run does not hold it."""
    from hk_flat_routes.validate import street_mask
    sub = edges[street_mask(edges, [name])]
    if sub.empty:
        pytest.skip(f"{name} is not in this run")
    return sub


def _climb_per_km(sub) -> float:
    """Climbing per kilometre, averaged over the two directions."""
    km = sub["length_m"].sum() / 1000
    return (sub["cum_gain_fwd"].sum() + sub["cum_loss_fwd"].sum()) / 2 / km


# ------------------------------------------------------------- invariants
def test_every_edge_has_two_directions(edges, directed):
    assert len(directed) == 2 * len(edges)
    counts = directed.groupby("edge_id").size()
    assert counts.min() == 2 and counts.max() == 2


def test_gain_and_loss_mirror_between_directions(directed):
    piv = directed.pivot_table(index="edge_id", columns="direction",
                               values=["cum_gain", "cum_loss"])
    assert np.allclose(piv[("cum_gain", "fwd")], piv[("cum_loss", "rev")],
                       atol=1e-6)
    assert np.allclose(piv[("cum_loss", "fwd")], piv[("cum_gain", "rev")],
                       atol=1e-6)


def test_net_change_is_antisymmetric(directed):
    piv = directed.pivot_table(index="edge_id", columns="direction",
                               values="net_change")
    assert np.allclose(piv["fwd"], -piv["rev"], atol=1e-6)


def test_gain_minus_loss_equals_net_change(directed):
    """Exact per-edge identity, guaranteed by protecting edge boundaries
    from the dead-band pruning."""
    d = directed
    assert np.allclose(d["cum_gain"] - d["cum_loss"], d["net_change"], atol=1e-3)


def test_every_node_has_one_elevation(directed):
    """Edges meeting at a junction agree on its height to the millimetre,
    or a route would step up or down for nothing each time it turned."""
    a = directed[["from_node", "start_elev"]].rename(
        columns={"from_node": "node", "start_elev": "z"})
    b = directed[["to_node", "end_elev"]].rename(
        columns={"to_node": "node", "end_elev": "z"})
    g = pd.concat([a, b], ignore_index=True).groupby("node")["z"]
    assert (g.max() - g.min()).max() < 1e-3


def test_no_edge_exceeds_the_plausible_grade_clip(edges):
    assert edges["max_abs_grade"].max() <= ELEVATION.max_plausible_grade + 1e-9


def test_elevations_are_in_a_sane_range_for_hong_kong(edges):
    # Tai Mo Shan, the highest point of the territory, is 957 m above
    # Principal Datum; and ground more than a metre below the datum is
    # treated as an excavation and bridged, so nothing should sit far
    # below it
    from hk_flat_routes.validate import ELEVATION_RANGE_M
    lo, hi = ELEVATION_RANGE_M
    assert lo == -5 and hi == 960
    assert edges["elev_max"].max() < hi
    assert edges["elev_min"].min() > lo
    assert edges["elev_min"].notna().all() and edges["elev_max"].notna().all()


def test_distance_above_thresholds_are_nested_and_bounded(directed):
    ths = [3, 5, 8, 10, 15]
    for a, b in zip(ths[:-1], ths[1:]):
        assert (directed[f"d_above_{a}"] >= directed[f"d_above_{b}"] - 1e-6).all()
    # tolerance covers float accumulation: interval lengths are summed,
    # while length_m comes from the geometry
    assert (directed["d_above_3"] <= directed["length_m"] * (1 + 1e-6) + 1e-3).all()


def test_stairways_are_never_bicycle_traversable(directed):
    steps = directed[directed["cls"] == "steps"]
    assert len(steps) > 0
    assert not steps["bike_traversable"].any()


def test_the_validation_invariants_agree(edges, directed):
    """The report's own table of invariants holds on this build."""
    from types import SimpleNamespace
    from hk_flat_routes.validate import check_invariants
    inv = check_invariants(SimpleNamespace(edges=edges, directed=directed))
    assert len(inv) >= 8
    assert (inv["verdict"] == "holds").all(), inv[inv["verdict"] != "holds"]


# ------------------------------------------------------ the street network
def _shortest_m(edges, mode, origin, destination, snap_m=300.0):
    """Metres from one (lon, lat) to another on the mode's network, or inf.

    Bicycles keep to one-way streets.  Each end is taken at the nearest
    node of a piece of the network of at least ``MIN_COMPONENT_NODES``
    nodes; a point with none within ``snap_m`` is outside the run, and the
    test is skipped.
    """
    import scipy.sparse as sp
    import shapely
    from pyproj import Transformer
    from scipy.sparse.csgraph import connected_components, dijkstra
    from scipy.spatial import cKDTree
    from hk_flat_routes.config import MIN_COMPONENT_NODES

    e = edges[edges[f"{mode}_ok"].to_numpy()]
    ids, inv = np.unique(np.concatenate([e["u"], e["v"]]), return_inverse=True)
    a, b = inv[:len(e)], inv[len(e):]
    length = e["length_m"].to_numpy()
    two_way = ~e["bike_oneway"].to_numpy() if mode == "bike" else np.ones(len(e), bool)
    graph = sp.coo_matrix(
        (np.r_[length, length[two_way]], (np.r_[a, b[two_way]], np.r_[b, a[two_way]])),
        shape=(len(ids), len(ids))).tocsr()
    xy = np.zeros((len(ids), 2))
    xy[a] = shapely.get_coordinates(shapely.get_point(e.geometry.values, 0))
    xy[b] = shapely.get_coordinates(shapely.get_point(e.geometry.values, -1))
    _, label = connected_components(graph, directed=True, connection="strong")
    solid = np.flatnonzero(np.bincount(label)[label] >= MIN_COMPONENT_NODES)
    tree = cKDTree(xy[solid])
    to_grid = Transformer.from_crs("EPSG:4326", "EPSG:2326", always_xy=True)
    ends = []
    for lon, lat in (origin, destination):
        d, k = tree.query(to_grid.transform(lon, lat))
        if d > snap_m:
            pytest.skip(f"({lon}, {lat}) is not in this run")
        ends.append(int(solid[k]))
    return float(dijkstra(graph, directed=True, indices=ends[0])[ends[1]])


def test_edges_meet_at_their_nodes(edges):
    """Every edge that ends at a node ends at the same point.  Segments are
    cut at their connectors by length in metres; cut by length in degrees,
    the ends at one node were up to 42 m apart, and the node's one
    elevation was forced onto all of them."""
    import shapely
    g = edges.geometry.values
    ends = pd.DataFrame(
        np.vstack([shapely.get_coordinates(shapely.get_point(g, 0)),
                   shapely.get_coordinates(shapely.get_point(g, -1))]),
        columns=["x", "y"])
    ends["node"] = np.r_[edges["u"].to_numpy(), edges["v"].to_numpy()]
    spread = ends.groupby("node").agg(["min", "max"])
    gap = np.hypot(spread[("x", "max")] - spread[("x", "min")],
                   spread[("y", "max")] - spread[("y", "min")])
    assert gap.max() < 0.02, gap.sort_values().tail(3).to_dict()


def test_no_street_of_unknown_class(edges):
    # roads under construction and segments from TomTom with nothing to
    # judge them by; the short links kept between two streets are service
    # roads
    assert not (edges["cls"] == "unknown").any()


def test_tsing_tsuen_bridge_is_walkable(edges):
    # Tsuen Wan West Station to Tsing Yi Station: 2.2 km over the bridge's
    # pavements, 5.3 km round by Kwai Chung without them
    d = _shortest_m(edges, "walk", (114.1097, 22.3684), (114.1075, 22.3585))
    assert d < 3000.0, d


def test_north_lantau_is_one_piece_by_bicycle(edges):
    # Tung Chung Station to Sunny Bay Station along Cheung Tung Road, which
    # passes under a flyover in what the data calls a tunnel
    d = _shortest_m(edges, "bike", (113.9416, 22.2890), (114.0290, 22.3320))
    assert d < 15000.0, d


def test_walkers_are_kept_off_trunk_road_viaducts(edges):
    # the walk from Tsing Yi to Mong Kok used to run over the viaducts of
    # Kwai Chung Interchange; no viaduct of a trunk road is open to walkers
    # unless the data says so by name
    deck = edges[(edges["cls"] == "trunk") & edges["is_bridge"] & edges["walk_ok"]]
    assert deck["length_m"].sum() < 1500.0, deck["length_m"].sum()
    kwai = edges[(edges["name"] == "Kwai Chung Interchange") & edges["is_bridge"]
                 & (edges["cls"] == "trunk")]
    assert len(kwai) and not kwai["walk_ok"].any()


def test_a_pin_on_a_trunk_road_is_not_stranded(edges):
    # Diamond Hill Station is on Lung Cheung Road, a trunk road at grade
    # with pavements; its slip roads at grade are the way off it.  Lions
    # Rise Mall is 460 m away, and was a walk of 1.6 km with them closed.
    d = _shortest_m(edges, "walk", (114.19723, 22.34072), (114.20165, 22.34003), snap_m=400.0)
    assert d < 800.0, d
    # Hung Kiu Bus Terminus to Tuen Mun Hospital, by San Tak Street, which
    # the map classes as a slip road of Tuen Mun Road
    d = _shortest_m(edges, "walk", (113.97747, 22.40200), (113.97621, 22.40708), snap_m=400.0)
    assert d < 1500.0, d


def test_a_street_is_not_cut_where_a_slip_road_crosses_it(edges):
    # Tuen Hing Road in Tuen Mun: 12 m of it is classed with the slip road
    # of Tuen Mun Road that joins it there
    d = _shortest_m(edges, "walk", (113.97825, 22.39030), (113.97822, 22.39040))
    assert d < 100.0, d


def test_gated_sites_are_closed(edges):
    """The service roads of the container port, the power stations and
    military land are open to nobody, and Lamma, whose power station they
    cross, is still one place on foot."""
    from hk_flat_routes.network import _inside, closed_land
    land = closed_land()
    if not len(land):
        pytest.skip("land-use extract not downloaded")
    service = edges[edges["cls"] == "service"]
    inside = _inside(service.geometry.values, land) >= 0
    used = inside & (service["walk_ok"] | service["bike_ok"]).to_numpy()
    # 76 km without the rule; what is left is a kilometre beside Shek Kong
    # airfield that the data opens to walkers and bicycles by name
    assert service.loc[used, "length_m"].sum() < 2000.0
    power_station = service[service["name"].isin(["Stacker Road", "Precipitator Road"])]
    assert not power_station["walk_ok"].any() and not power_station["bike_ok"].any()
    # Yung Shue Wan ferry pier to Sok Kwu Wan pier
    d = _shortest_m(edges, "walk", (114.1108, 22.2262), (114.1313, 22.2053))
    assert d < 6000.0, d


def test_cycle_tracks_are_joined_to_the_cycling_network(edges):
    """Short footway links are opened to pushed bicycles, without which 76
    of the 269 km of cycle track is in pieces no route can reach."""
    from hk_flat_routes.config import (BIKE_PERMIT_REQUIRED_CLASSES,
                                       BIKE_PUSH_MAX_M, MIN_COMPONENT_NODES)
    from hk_flat_routes.network import _component_sizes
    size = _component_sizes(edges["u"].to_numpy(), edges["v"].to_numpy(),
                            edges["bike_ok"].to_numpy())
    track = (edges["cls"] == "cycleway") & edges["bike_ok"]
    if track.sum() < 100:
        pytest.skip("too little cycle track in this run")
    km = edges["length_m"] / 1000.0
    assert km[track & (size >= MIN_COMPONENT_NODES)].sum() > 0.95 * km[track].sum()
    push = edges[edges["bike_push"]]
    assert len(push) and push["bike_ok"].all() and push["walk_ok"].all()
    assert push["cls"].isin(BIKE_PERMIT_REQUIRED_CLASSES).all()
    assert not push["is_structure"].any() and not push["bike_oneway"].any()
    assert push["length_m"].max() <= BIKE_PUSH_MAX_M
    assert (push["bike_facility"] == "").all()
    # a facility label means a bicycle may ride there
    labelled = edges[edges["bike_facility"] == "shared_path"]
    assert labelled["bike_ok"].all()


# ------------------------------------------------------ reference streets
def test_the_tram_line_is_level_and_near_sea_level(edges):
    # Hennessy Road carries the tram across Wan Chai on reclaimed land;
    # its two bench marks stand at 4.1 and 4.6 m
    sub = _street(edges, "Hennessy Road")
    assert _climb_per_km(sub) < 4.0, "Hennessy Road should be level"
    assert 2.0 < sub["elev_min"].min() and sub["elev_max"].max() < 9.0


def test_known_flat_and_steep_streets_are_correctly_separated(edges):
    # Hatton Road climbs from the Mid-Levels to the Peak at 13.5% between
    # its bench marks, so about 135 m per kilometre in the uphill direction
    # and half that averaged over both
    flat = _climb_per_km(_street(edges, "Hennessy Road"))
    steep = _climb_per_km(_street(edges, "Hatton Road"))
    assert flat < 4.0, f"Hennessy Road should be level, got {flat:.1f} m/km"
    assert steep > 40.0, f"Hatton Road should be steep, got {steep:.1f} m/km"
    assert steep > 8 * flat


def test_a_contour_road_is_level_but_high(edges):
    # Bowen Road follows the 120 m contour for three and a half kilometres
    # (eight bench marks between 119.9 and 124.2 m): Hong Kong's kind of
    # flat.  The bounds allow for the gullies the bare-earth survey dips
    # into where the road crosses them on a culvert.
    sub = _street(edges, "Bowen Road")
    assert _climb_per_km(sub) < 15.0
    assert sub["elev_min"].min() > 100.0 and sub["elev_max"].max() < 140.0


def test_hatton_road_matches_the_gradient_between_its_bench_marks(edges, profiles, marks):
    from hk_flat_routes.validate import check_steep_streets
    steep = check_steep_streets(edges, profiles, marks)
    row = steep[(steep["group"] == "benchmark")
                & (steep["street"] == "Hatton Road")].iloc[0]
    if row["verdict"] == "not in this run":
        pytest.skip("Hatton Road is not in this run")
    assert row["verdict"] == "ok", row.to_dict()
    assert abs(row["computed"] - row["published"]) < 0.01
    assert abs(row["model_rise_m"] - row["mark_rise_m"]) < 2.0
    # the model's own distance along the road is the researched one
    assert abs(row["length_m"] - row["stated_length_m"]) < 0.03 * row["stated_length_m"]


def test_a_ladder_street_comes_out_as_steep_as_its_sign(edges, profiles, marks):
    # Peel Street is signed 1:4 above Staunton Street
    from hk_flat_routes.validate import check_steep_streets
    steep = check_steep_streets(edges, profiles, marks)
    row = steep[(steep["group"] == "ratio")
                & (steep["street"] == "Peel Street")].iloc[0]
    if row["verdict"] == "not in this run":
        pytest.skip("Peel Street is not in this run")
    assert row["published"] == 0.25
    assert abs(row["computed"] - 0.25) < 0.05, row.to_dict()


def test_street_elevations_agree_with_the_official_bench_marks(edges, profiles, marks):
    from hk_flat_routes.validate import check_benchmarks
    bench = check_benchmarks(edges, profiles, marks)
    if len(bench) < 50:
        pytest.skip("too few bench marks beside a road in this run")
    d = bench["diff"]
    # marks stand a few decimetres above the carriageway, never below it
    assert -1.0 < d.median() < 0.0
    assert (d.abs() <= 1.0).mean() > 0.75
    assert (d.abs() <= 2.0).mean() > 0.90


def test_gaps_are_judged_at_the_gap(edges, directed, profiles, marks):
    """A gap's height is read where the gap is.  Two gaps can share a
    circle (Wan Chai Gap is under 500 m from Magazine Gap, and lower), and
    the lowest crossing between the marks is then over the wrong one; the
    shortest route's high point stands in for it."""
    from types import SimpleNamespace
    from hk_flat_routes.routing import build_route_graph
    from hk_flat_routes.validate import (GAP_NEAR_M, GAP_TOLERANCE_M,
                                         check_passes)
    ctx = SimpleNamespace(edges=edges, profiles=profiles,
                          graphs={"walk": build_route_graph(directed, "walk")})
    gaps = check_passes(ctx, marks)
    seen = gaps[gaps["pass_m"].notna()]
    if seen.empty:
        pytest.skip("none of the reference gaps is in this run")
    for _, r in seen.iterrows():
        if r["verdict"] == "lower way round":
            continue
        offset = r["route_offset_m"] if r["judged"] == "route" else r["pass_offset_m"]
        assert offset <= GAP_NEAR_M, r.to_dict()
    # Magazine Gap is the one with a bench mark standing at it
    mg = seen[(seen["gap"] == "Magazine Gap") & seen["mark_m"].notna()]
    if len(mg):
        r = mg.iloc[0]
        height = r["route_high_m"] if r["judged"] == "route" else r["pass_m"]
        assert abs(height - r["mark_m"]) <= GAP_TOLERANCE_M, r.to_dict()
        assert r["verdict"] == "agrees"


# ----------------------------------------------------------------- routes
@pytest.mark.skipif(not PAIRS.exists(), reason="pair analysis not run")
def test_no_objective_beats_the_shortest_path_on_distance():
    p = pd.read_parquet(PAIRS)
    assert (p["detour_ratio"] >= 0.999).all()


@pytest.mark.skipif(not PAIRS.exists(), reason="pair analysis not run")
def test_climb_averse_objectives_actually_reduce_climbing():
    p = pd.read_parquet(PAIRS)
    means = p.groupby("profile")["elev_gain_m"].mean()
    assert means["min_climb"] < means["shortest"]
    assert means["balanced"] < means["shortest"]


@pytest.mark.skipif(not PAIRS.exists(), reason="pair analysis not run")
def test_grade_averse_lowers_maximum_gradient_most(p_=None):
    p = pd.read_parquet(PAIRS)
    means = p.groupby("profile")["max_grade"].mean()
    assert means["grade_averse"] < means["min_climb"]
    assert means["grade_averse"] < means["shortest"]


@pytest.mark.skipif(not PAIRS.exists(), reason="pair analysis not run")
def test_route_metrics_are_internally_consistent():
    p = pd.read_parquet(PAIRS)
    assert (p["elev_gain_m"] >= 0).all() and (p["elev_loss_m"] >= 0).all()
    net = p["elev_gain_m"] - p["elev_loss_m"]
    endpoint_net = p["end_elev_m"] - p["start_elev_m"]
    assert np.allclose(net, endpoint_net, atol=0.05)


# ------------------------------------- the measuring tools, on a toy street
def _toy_street():
    """A street 400 m long with a hole in its name and a junction box.

    West to east: 100 m level at 10 m, 100 m climbing to 30 m (20%), a 20 m
    junction mapped under no name, then 180 m level at 30 m.  The second
    piece is drawn east to west, as half of any real street is; the third
    carries only the Chinese name; a sidewalk shadows the first; and a long
    way round under another name joins the two ends.
    """
    import geopandas as gpd
    from shapely.geometry import LineString

    def edge(eid, u, v, pts, name, zh, cls="tertiary", sub=None):
        return {"edge_id": eid, "u": u, "v": v, "name": name, "name_zh": zh,
                "cls": cls, "subclass": sub, "geometry": LineString(pts),
                "is_structure": False, "walk_ok": True}
    rows = [
        edge(0, "a", "b", [(0, 0), (100, 0)], "Hill Street", "山街"),
        edge(1, "c", "b", [(200, 0), (100, 0)], "Hill Street", "山街"),
        edge(2, "c", "d", [(200, 0), (220, 0)], None, None, cls="unclassified"),
        edge(3, "d", "e", [(220, 0), (400, 0)], "山街", None),
        edge(4, "a", "b", [(0, 3), (100, 3)], "Hill Street", "山街",
             cls="footway", sub="sidewalk"),
        edge(5, "a", "e", [(0, 0), (0, -300), (400, -300), (400, 0)],
             "Round Road", "環路"),
    ]
    e = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:2326")
    e["length_m"] = e.geometry.length

    def prof(length, z0, z1):
        d = np.linspace(0.0, length, int(length / 5) + 1)
        return d, z0 + (z1 - z0) * d / length
    profiles = {0: prof(100, 10, 10), 1: prof(100, 30, 10), 2: prof(20, 30, 30),
                3: prof(180, 30, 30), 4: prof(100, 10, 10), 5: prof(1000, 10, 30)}
    z = {k: v[1] for k, v in profiles.items()}
    e["elev_min"] = [float(z[i].min()) for i in e["edge_id"]]
    e["elev_max"] = [float(z[i].max()) for i in e["edge_id"]]
    e["cum_gain_fwd"] = [max(float(z[i][-1] - z[i][0]), 0.0) for i in e["edge_id"]]
    e["cum_loss_fwd"] = [max(float(z[i][0] - z[i][-1]), 0.0) for i in e["edge_id"]]
    e["avg_grade_fwd"] = [float(z[i][-1] - z[i][0]) for i in e["edge_id"]] / e["length_m"]
    return e, profiles


def test_steepest_stretch_is_a_sustained_gradient_not_a_spike():
    from hk_flat_routes.validate import steepest_stretch
    d = np.arange(0.0, 301.0, 5.0)
    z = np.where(d < 100, 10.0, np.where(d < 200, 10.0 + 0.2 * (d - 100), 30.0))
    assert steepest_stretch(d, z, 50.0) == pytest.approx(0.20)
    assert steepest_stretch(d, z, 100.0) == pytest.approx(0.20)
    assert steepest_stretch(d, z, 200.0) == pytest.approx(0.10)
    # a profile shorter than the window can only vouch for its own mean
    assert steepest_stretch(d[:11], z[:11], 100.0) == pytest.approx(0.0)
    assert steepest_stretch(d, z, 600.0) == pytest.approx(20.0 / 300.0)
    # one bad sample is diluted by the window instead of setting the answer
    z2 = z.copy(); z2[5] += 3.0
    assert steepest_stretch(d, z2, 50.0) < 0.21


def test_a_street_is_matched_by_its_chinese_name_too_and_not_its_sidewalk():
    from hk_flat_routes.validate import street_mask
    e, _ = _toy_street()
    assert list(e["edge_id"][street_mask(e, ["Hill Street"])]) == [0, 1, 3]
    assert not street_mask(e, ["No Such Street"]).any()


def test_a_section_is_followed_along_the_street_and_across_its_gaps():
    from hk_flat_routes.validate import StreetNet, profile_stats
    e, profiles = _toy_street()
    net = StreetNet(e, profiles)
    street = net.street(["Hill Street"])
    start = net.snap((20.0, 4.0), street, 30.0)       # nearer the sidewalk
    finish = net.snap((390.0, -2.0), street, 30.0)
    assert start[0] == 0 and start[1] == pytest.approx(20.0)
    assert net.snap((20.0, 100.0), street, 30.0) is None
    pieces = net.between(start, finish, street)
    # it stays on the street, crossing the unnamed junction, and does not
    # take the way round although that would also arrive
    assert [int(net.eid[r]) for r, _a, _b in pieces] == [0, 1, 2, 3]
    assert net.share(pieces, street) == pytest.approx(350.0 / 370.0)
    d, z = net.profile(pieces)
    assert d[0] == 0.0 and d[-1] == pytest.approx(370.0)
    assert (np.diff(d) > 0).all() and (np.diff(z) >= -1e-9).all()
    st = profile_stats(d, z)
    assert st["rise_m"] == pytest.approx(20.0)
    assert st["steepest_50"] == pytest.approx(0.20)
    assert st["mean_grade"] == pytest.approx(20.0 / 370.0)
    # and back again is the same street downhill
    back = net.between(finish, start, street)
    d2, z2 = net.profile(back)
    assert d2[-1] == pytest.approx(370.0)
    assert np.allclose(z2, np.interp(370.0 - d2, d, z))
    # two points on one edge need no search at all
    assert net.between(start, net.snap((80.0, 0.0), street, 30.0), street) == [(0, 20.0, 80.0)]


def test_a_section_given_by_one_end_and_a_length_runs_the_right_way():
    from hk_flat_routes.validate import StreetNet
    e, profiles = _toy_street()
    net = StreetNet(e, profiles)
    street = net.street(["Hill Street"])
    bottom = net.snap((50.0, 0.0), street, 30.0)
    up = net.along(bottom, street, 150.0, uphill=True)
    d, z = net.profile(up)
    assert d[-1] == pytest.approx(150.0) and z[-1] - z[0] == pytest.approx(20.0)
    # nowhere on the street is lower than its bottom
    assert net.along(bottom, street, 150.0, uphill=False) == []


def test_sections_report_why_they_could_not_be_measured():
    from hk_flat_routes.validate import StreetNet, _measure_section
    e, profiles = _toy_street()
    net = StreetNet(e, profiles)
    ok = _measure_section(net, ["Hill Street"], None, (200.0, 0.0), (100.0, 0.0), 100)
    assert ok["status"] == "measured" and ok["mean_grade"] == pytest.approx(-0.20)
    assert _measure_section(net, ["Elsewhere Road"], None, (0, 0), (9, 9))["status"] \
        == "not in this run"
    # the toy network is 400 m across, so every point is near the run's edge
    # and an end that misses the street is the run's doing, not the model's
    lost = _measure_section(net, ["Hill Street"], None, (200.0, 250.0), (100.0, 0.0))
    assert lost["status"] == "not in this run"
    # a stated length the street cannot supply is not judged either
    short = _measure_section(net, ["Hill Street"], None, (200.0, 0.0), (100.0, 0.0), 900)
    assert short["status"] != "measured" and "900 m is stated" in short["why"]


def test_level_means_little_climbing_within_a_narrow_band():
    from hk_flat_routes import validate as V
    e, _ = _toy_street()
    marks = pd.DataFrame({"station": ["1", "2", "3"],
                          "locality": ["HILL STREET", "HILL STREET, EAST", "ROUND ROAD"],
                          "type": "STAPLE", "x": [10.0, 150.0, 5.0],
                          "y": [5.0, -4.0, -20.0], "h": [10.2, 20.3, 12.0]})
    spec = {"Hill Street": {"streets": ["Hill Street"], "expect": "not level"},
            "Round Road": {"streets": ["Round Road"], "expect": "level"},
            "Far Road": {"streets": ["Far Road"], "expect": "level"}}
    old, V.KNOWN_FLAT = V.KNOWN_FLAT, spec
    try:
        flat = V.check_flat_corridors(e, None, marks).set_index("corridor")
    finally:
        V.KNOWN_FLAT = old
    # 20 m of climbing in 380 m is 26 m/km averaged over the two directions
    hill = flat.loc["Hill Street"]
    assert hill["gain_per_km"] == pytest.approx(10.0 / 0.38)
    assert hill["verdict"] == "not level" and hill["as_expected"]
    # only the marks whose locality names the street count as its evidence
    assert hill["n_marks"] == 2 and (hill["mark_min"], hill["mark_max"]) == (10.2, 20.3)
    # 20 m in a kilometre is under the per-kilometre limit and inside the band
    assert flat.loc["Round Road", "verdict"] == "level"
    assert flat.loc["Far Road", "verdict"] == "not in this run"


def test_a_one_ended_section_crosses_a_staggered_junction():
    """Shatin Pass Road in small: the street stops at a side street and
    begins again 70 m on, across a junction mapped under another name."""
    import geopandas as gpd
    from shapely.geometry import LineString
    from hk_flat_routes.validate import StreetNet, _measure_section

    def edge(eid, u, v, pts, name):
        return {"edge_id": eid, "u": u, "v": v, "name": name, "name_zh": None,
                "cls": "tertiary", "subclass": None, "geometry": LineString(pts),
                "is_structure": False, "walk_ok": True}
    e = gpd.GeoDataFrame([
        edge(0, "a", "b", [(0, 0), (100, 0)], "Pass Road"),
        edge(1, "b", "c", [(100, 0), (100, 70)], "Side Street"),
        edge(2, "c", "d", [(100, 70), (350, 70)], "Pass Road"),
    ], geometry="geometry", crs="EPSG:2326")
    e["length_m"] = e.geometry.length
    profiles = {0: (np.array([0.0, 100.0]), np.array([0.0, 10.0])),
                1: (np.array([0.0, 70.0]), np.array([10.0, 18.0])),
                2: (np.array([0.0, 250.0]), np.array([18.0, 58.0]))}
    net = StreetNet(e, profiles)
    street = net.street(["Pass Road"])
    bottom = net.snap((0.0, 0.0), street, 30.0)
    assert 60.0 < 70.0 <= net.link_max_m
    d, z = net.profile(net.along(bottom, street, 400.0, uphill=True))
    assert d[-1] == pytest.approx(420.0) and z[-1] == pytest.approx(58.0)
    # the whole section is judged only because four fifths of it is the street
    m = _measure_section(net, ["Pass Road"], None, None, (0.0, 0.0), 400)
    assert m["on_street"] == pytest.approx(350.0 / 420.0)
    # a link longer than the allowance leaves only the run below the junction
    old, StreetNet.link_max_m = StreetNet.link_max_m, 60.0
    try:
        d, z = net.profile(net.along(bottom, street, 400.0, uphill=True))
    finally:
        StreetNet.link_max_m = old
    assert d[-1] == pytest.approx(100.0) and z[-1] == pytest.approx(10.0)


def test_bench_mark_localities_are_written_as_names():
    from hk_flat_routes.validate import _proper
    assert _proper("HIRAM'S HIGHWAY") == "Hiram's Highway"
    assert _proper("CAPE D'AGUILAR ROAD") == "Cape D'Aguilar Road"
    assert _proper("CASTLE PEAK ROAD - TAI LAM, TAI LAM KOK") == \
        "Castle Peak Road - Tai Lam, Tai Lam Kok"
