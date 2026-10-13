"""Tests for the minimax (bottleneck) pass analysis.

The toy networks are in metres above Principal Datum on the HK1980 Grid
(EPSG:2326), with street names in the two forms Hong Kong's data carries.
"""
import json

import numpy as np
import pandas as pd
import pytest

import hk_flat_routes.passes as P

CRS = "EPSG:2326"


def toy_edges():
    """Two low valleys joined by two ridges: a low col and a high one.

    A(4m) -- B(8m) -- C(40m) -- D(10m) -- E(5m)      via the 40 m col
    A      --------- F(95m) --------- E              via the 95 m ridge

    The minimax pass between A and E must be the 40 m col, never the ridge,
    and the responsible edge must be the one that reaches 40 m.
    """
    rows = [
        # (u, v, elev_max, elev_min, name, name_zh)
        ("A", "B", 8.0, 4.0, "Praya Road", "海旁道"),
        ("B", "C", 40.0, 8.0, "Gap Road", "峽道"),
        ("C", "D", 40.0, 10.0, "Gap Road", "峽道"),
        ("D", "E", 10.0, 5.0, "Bay Road", "灣道"),
        ("A", "F", 95.0, 4.0, "Peak Path", "山頂徑"),
        ("F", "E", 95.0, 5.0, "Peak Path", "山頂徑"),
    ]
    recs = []
    for i, (u, v, emax, emin, name, name_zh) in enumerate(rows):
        recs.append({"edge_id": i, "u": u, "v": v, "elev_max": emax,
                     "elev_min": emin, "name": name, "name_zh": name_zh,
                     "cls": "residential", "length_m": 100.0,
                     "max_abs_grade": 0.1, "walk_ok": True})
    return pd.DataFrame(recs)


def two_landmasses():
    """The toy valley, plus an island with no link to it.

    X(3m) -- Y(60m) -- Z(6m) stands for the far side of the harbour: a
    network of its own, reachable from A..F only by ferry.
    """
    e = toy_edges()
    extra = pd.DataFrame([
        {"edge_id": 6, "u": "X", "v": "Y", "elev_max": 60.0, "elev_min": 3.0,
         "name": "Island Road", "name_zh": "島道", "cls": "residential",
         "length_m": 100.0, "max_abs_grade": 0.1, "walk_ok": True},
        {"edge_id": 7, "u": "Y", "v": "Z", "elev_max": 60.0, "elev_min": 6.0,
         "name": "Island Road", "name_zh": "島道", "cls": "residential",
         "length_m": 100.0, "max_abs_grade": 0.1, "walk_ok": True},
    ])
    return pd.concat([e, extra], ignore_index=True)


def test_bottleneck_tree_finds_the_low_col_not_the_high_ridge():
    e = toy_edges()
    tree = P.build_bottleneck_tree(e, mode="walk")
    h, eid = P.pass_height(tree, "A", "E")
    assert h == pytest.approx(40.0)
    assert e.loc[e["edge_id"] == eid, "name"].iloc[0] == "Gap Road"


def test_pass_height_is_symmetric():
    tree = P.build_bottleneck_tree(toy_edges(), mode="walk")
    assert P.pass_height(tree, "A", "E")[0] == pytest.approx(
        P.pass_height(tree, "E", "A")[0])


def test_pass_height_within_a_valley_is_low():
    tree = P.build_bottleneck_tree(toy_edges(), mode="walk")
    h, _ = P.pass_height(tree, "A", "B")
    assert h == pytest.approx(8.0)


def test_pass_height_is_at_least_the_endpoint_elevations():
    """The crossing can never be lower than where you start or finish."""
    e = toy_edges()
    tree = P.build_bottleneck_tree(e, mode="walk")
    for a, b in [("A", "E"), ("B", "D"), ("A", "F")]:
        h, _ = P.pass_height(tree, a, b)
        za = e[(e["u"] == a) | (e["v"] == a)]["elev_min"].min()
        zb = e[(e["u"] == b) | (e["v"] == b)]["elev_min"].min()
        assert h >= max(za, zb) - 1e-9


def test_unknown_node_returns_none():
    tree = P.build_bottleneck_tree(toy_edges(), mode="walk")
    assert P.pass_height(tree, "A", "nowhere") is None
    assert P.pass_height(tree, "A", None) is None


def test_a_node_has_no_pass_to_itself():
    """Not minus infinity, which is what the tree's leaf label would give."""
    tree = P.build_bottleneck_tree(toy_edges(), mode="walk")
    assert P.pass_height(tree, "A", "A") is None


def test_neighborhood_pass_matrix_covers_every_unordered_pair():
    e = toy_edges()
    tree = P.build_bottleneck_tree(e, mode="walk")
    pts = pd.DataFrame({"neighborhood": ["Sai Wan", "Tung Wan", "The Gap"],
                        "node": ["A", "E", "C"]})
    m = P.neighborhood_pass_matrix(tree, pts)
    assert len(m) == 3                       # 3 choose 2
    ae = m[(m["neighborhood_a"] == "Sai Wan") & (m["neighborhood_b"] == "Tung Wan")]
    assert ae["pass_elev_m"].iloc[0] == pytest.approx(40.0)


def test_pass_matrix_is_metric_only():
    """Upstream's feet column is gone; heights are metres above Principal Datum."""
    tree = P.build_bottleneck_tree(toy_edges(), mode="walk")
    pts = pd.DataFrame({"neighborhood": ["Sai Wan", "Tung Wan"], "node": ["A", "E"]})
    m = P.neighborhood_pass_matrix(tree, pts)
    assert list(m.columns) == list(P.PASS_MATRIX_COLUMNS)
    assert not [c for c in m.columns if c.endswith("_ft")]


# --------------------------------------------------------------------------
# how far the lowest way round is
# --------------------------------------------------------------------------
def long_way_round():
    """The toy valley, with a level road right round the hill.

    A to E is 400 m over the 40 m col, and 3 km round by G and H without
    rising above 12 mPD.  The lowest crossing is then 12 mPD, and the route
    that keeps to it is the long one.  A second, longer edge between G and
    H, no higher, must not be added to the first.
    """
    e = toy_edges()
    rows = [("A", "G", 9.0, 1000.0), ("G", "H", 12.0, 1000.0),
            ("H", "E", 10.0, 1000.0), ("G", "H", 12.0, 2500.0)]
    extra = pd.DataFrame([
        {"edge_id": 6 + i, "u": u, "v": v, "elev_max": crest, "elev_min": 4.0,
         "name": "Coast Road", "name_zh": "海岸道", "cls": "residential",
         "length_m": length, "max_abs_grade": 0.01, "walk_ok": True}
        for i, (u, v, crest, length) in enumerate(rows)])
    return pd.concat([e, extra], ignore_index=True)


def test_the_lowest_route_can_be_much_longer_than_the_shortest():
    """A pass height is a statement about height.  The report says how far
    the way that keeps to it is, so that a divide at 32 mPD is not read as
    a route somebody would walk."""
    e = long_way_round()
    tree = P.build_bottleneck_tree(e, mode="walk")
    pts = pd.DataFrame({"neighborhood": ["Sai Wan", "Tung Wan", "The Gap"],
                        "node": ["A", "E", "C"]})
    m = P.neighborhood_pass_matrix(tree, pts)
    low = P.low_route_lengths(m, e, pts)
    assert low.name == P.LOW_ROUTE_COLUMN and low.index.equals(m.index)
    by_pair = {(a, b): (h, d) for a, b, h, d in
               zip(m["neighborhood_a"], m["neighborhood_b"], m["pass_elev_m"], low)}
    # the way round by the shorter of the two edges between G and H
    assert by_pair[("Sai Wan", "Tung Wan")] == (12.0, pytest.approx(3000.0))
    # the col itself can only be reached by climbing to it: 200 m either way
    assert by_pair[("Sai Wan", "The Gap")] == (40.0, pytest.approx(200.0))
    assert by_pair[("Tung Wan", "The Gap")] == (40.0, pytest.approx(200.0))


def test_lowest_routes_of_nothing_and_of_a_missing_point():
    e = long_way_round()
    tree = P.build_bottleneck_tree(e, mode="walk")
    pts = pd.DataFrame({"neighborhood": ["Sai Wan", "Tung Wan"], "node": ["A", "E"]})
    m = P.neighborhood_pass_matrix(tree, pts)
    assert P.low_route_lengths(m.iloc[0:0], e, pts).empty
    # the pair is in the matrix, but its access point has since gone
    gone = pts.assign(node=["A", None])
    assert P.low_route_lengths(m, e, gone).isna().all()


# --------------------------------------------------------------------------
# several landmasses
# --------------------------------------------------------------------------
def test_no_pass_between_two_landmasses():
    tree = P.build_bottleneck_tree(two_landmasses(), mode="walk")
    assert P.pass_height(tree, "A", "X") is None
    assert P.pass_height(tree, "Z", "E") is None
    # and each landmass still answers for itself
    assert P.pass_height(tree, "X", "Z")[0] == pytest.approx(60.0)
    assert P.pass_height(tree, "A", "E")[0] == pytest.approx(40.0)


def test_pass_matrix_skips_cross_harbour_pairs_cleanly():
    """Pairs across the water are absent: not zero, not infinite, not NaN."""
    tree = P.build_bottleneck_tree(two_landmasses(), mode="walk")
    pts = pd.DataFrame({
        "neighborhood": ["Central", "Wan Chai", "Tsim Sha Tsui", "Mong Kok"],
        "node": ["A", "E", "X", "Z"],
        "component": [1, 1, 0, 0]})
    m = P.neighborhood_pass_matrix(tree, pts)
    pairs = {frozenset(t) for t in zip(m["neighborhood_a"], m["neighborhood_b"])}
    assert pairs == {frozenset({"Central", "Wan Chai"}),
                     frozenset({"Tsim Sha Tsui", "Mong Kok"})}
    assert np.isfinite(m["pass_elev_m"]).all()
    assert (m["pass_elev_m"] > 0).all()


def test_pass_matrix_skips_them_without_a_component_column_too():
    """The tree alone is enough: the two sides have no common ancestor."""
    tree = P.build_bottleneck_tree(two_landmasses(), mode="walk")
    pts = pd.DataFrame({"neighborhood": ["Central", "Tsim Sha Tsui", "Wan Chai"],
                        "node": ["A", "X", "E"]})
    m = P.neighborhood_pass_matrix(tree, pts)
    assert len(m) == 1
    assert set(m.iloc[0][["neighborhood_a", "neighborhood_b"]]) == {"Central", "Wan Chai"}


def test_pass_matrix_ignores_neighborhoods_with_no_network():
    """A partial build leaves some planning areas without an access point."""
    tree = P.build_bottleneck_tree(toy_edges(), mode="walk")
    pts = pd.DataFrame({"neighborhood": ["Central", "Wan Chai", "Sha Tin", "Tai Po"],
                        "node": ["A", "E", None, float("nan")],
                        "component": [1.0, 1.0, float("nan"), float("nan")]})
    m = P.neighborhood_pass_matrix(tree, pts)
    assert len(m) == 1
    assert "Sha Tin" not in set(m["neighborhood_a"]) | set(m["neighborhood_b"])


def test_pass_matrix_of_nothing_keeps_its_columns():
    tree = P.build_bottleneck_tree(toy_edges(), mode="walk")
    pts = pd.DataFrame({"neighborhood": ["Sha Tin"], "node": [None]})
    m = P.neighborhood_pass_matrix(tree, pts)
    assert m.empty and list(m.columns) == list(P.PASS_MATRIX_COLUMNS)


def test_basin_passes_never_join_two_landmasses():
    e = two_landmasses()
    z = pd.Series({"A": 4.0, "B": 8.0, "C": 40.0, "D": 10.0, "E": 5.0,
                   "F": 95.0, "X": 3.0, "Y": 60.0, "Z": 6.0})
    labels, basins = P.find_basins(e, z, threshold=15.0, min_km=0.05)
    assert len(basins) == 2                  # A-B and D-E; X and Z are lone nodes
    passes = P.find_passes(e, z, labels)
    assert len(passes) == 1
    assert passes["pass_elev_m"].iloc[0] == pytest.approx(40.0)


def test_edges_without_elevation_do_not_become_passes():
    e = toy_edges()
    e.loc[e["name"] == "Gap Road", "elev_max"] = float("nan")
    tree = P.build_bottleneck_tree(e, mode="walk")
    h, eid = P.pass_height(tree, "A", "E")
    assert h == pytest.approx(95.0)          # only the ridge is left
    assert np.isfinite(h)


# --------------------------------------------------------------------------
def test_disjoint_union_find_merges_correctly():
    dsu = P._DSU(5)
    assert dsu.find(0) == 0
    assert dsu.union(0, 1) is not None
    assert dsu.find(0) == dsu.find(1)
    assert dsu.union(0, 1) is None           # already joined
    dsu.union(2, 3)
    assert dsu.find(2) == dsu.find(3)
    assert dsu.find(0) != dsu.find(2)


def test_node_elevations_are_the_median_of_incident_endpoints():
    directed = pd.DataFrame({
        "from_node": ["A", "A", "B"],
        "to_node": ["B", "B", "A"],
        "start_elev": [10.0, 12.0, 20.0],
        "end_elev": [20.0, 20.0, 11.0],
    })
    z = P.node_elevations(directed)
    assert z["A"] == pytest.approx(11.0)     # median of 10, 12, 11
    assert z["B"] == pytest.approx(20.0)


def test_basins_require_a_minimum_size():
    e = toy_edges()
    z = pd.Series({"A": 4.0, "B": 8.0, "C": 40.0, "D": 10.0, "E": 5.0, "F": 95.0})
    labels, basins = P.find_basins(e, z, threshold=15.0, min_km=10.0)
    assert basins == {}                      # nothing is 10 km long
    labels, basins = P.find_basins(e, z, threshold=15.0, min_km=0.05)
    assert len(basins) == 2                  # A-B and D-E, split by the col


def test_basin_threshold_is_above_the_reclamation_level():
    """Reclaimed ground is formed at 4-6 mPD; the threshold must clear it."""
    assert 8.0 <= P.BASIN_ELEV_M <= 20.0


# --------------------------------------------------------------------------
# where a pass is: planning areas do not tile the territory
# --------------------------------------------------------------------------
def planning_areas():
    import geopandas as gpd
    from shapely.geometry import box
    return gpd.GeoDataFrame(
        {"neighborhood": ["Central District", "Wan Chai"]},
        geometry=[box(833000, 815000, 834000, 816000),
                  box(835000, 815000, 836000, 816000)], crs=CRS)


def test_locate_names_the_area_or_the_nearest_one():
    from shapely.geometry import LineString
    lines = [LineString([(833400, 815500), (833600, 815500)]),     # inside Central
             LineString([(834400, 815500), (834500, 815500)]),     # between the two
             LineString([(840000, 815500), (840100, 815500)])]     # far to the east
    w = P.locate(lines, planning_areas(), CRS)
    assert w["neighborhood"].iloc[0] == "Central District"
    assert w["nearest_distance_m"].iloc[0] == 0.0
    assert pd.isna(w["neighborhood"].iloc[1])
    assert w["nearest_neighborhood"].iloc[1] == "Central District"
    assert w["nearest_distance_m"].iloc[1] == pytest.approx(450.0)
    assert pd.isna(w["neighborhood"].iloc[2])
    assert w["nearest_neighborhood"].iloc[2] == "Wan Chai"
    assert w["nearest_distance_m"].iloc[2] == pytest.approx(4050.0)


def test_locate_of_nothing():
    w = P.locate([], planning_areas(), CRS)
    assert list(w.columns) == ["neighborhood", "nearest_neighborhood",
                               "nearest_distance_m"]
    assert w.empty


def test_place_label_says_how_far_off_a_place_is():
    assert P.place_label("Stanley", "Stanley", 0.0) == "Stanley"
    assert P.place_label(None, "Stanley", 300.0) == "near Stanley"
    assert P.place_label(float("nan"), "Pok Fu Lam", 4200.0) == "4 km from Pok Fu Lam"
    assert P.place_label(None, None, None) == "outside the planning areas"


def test_ranked_passes_carry_both_names_and_a_location():
    import geopandas as gpd
    from shapely.geometry import LineString
    e = toy_edges()
    geoms = [LineString([(833000 + 200 * i, 815500), (833200 + 200 * i, 815500)])
             for i in range(len(e))]
    e = gpd.GeoDataFrame(e, geometry=geoms, crs=CRS)
    tree = P.build_bottleneck_tree(e, mode="walk")
    pts = pd.DataFrame({"neighborhood": ["Central District", "Wan Chai", "The Gap"],
                        "node": ["A", "E", "C"]})
    ranked = P.rank_critical_passes(P.neighborhood_pass_matrix(tree, pts), e,
                                    planning_areas())
    for col in ("name", "name_zh", "pass_elev_m", "pairs_served",
                "neighborhood", "nearest_neighborhood", "nearest_distance_m"):
        assert col in ranked.columns
    assert "pass_elev_ft" not in ranked.columns
    assert ranked["pairs_served"].is_monotonic_decreasing
    assert set(ranked["name_zh"]) == {"峽道"}


# --------------------------------------------------------------------------
# output files exist even when there is nothing in them
# --------------------------------------------------------------------------
def test_empty_layers_are_written_and_read_back_with_their_columns(tmp_path):
    import geopandas as gpd
    empty = gpd.GeoDataFrame(columns=["edge_id", "geometry"], geometry="geometry",
                             crs=CRS)
    gj, csv = tmp_path / "passes.geojson", tmp_path / "passes.csv"
    P._write_layer(empty, gj, csv, P.PASS_COLUMNS)
    assert json.loads(gj.read_text()) == {"type": "FeatureCollection", "features": []}
    assert list(pd.read_csv(csv).columns) == list(P.PASS_COLUMNS)
    back = P.read_layer(gj, P.PASS_COLUMNS)
    assert len(back) == 0
    for col in P.PASS_COLUMNS:
        assert col in back.columns
    assert back.head(12).empty               # what the map and report do with it


def test_a_missing_layer_reads_as_an_empty_one(tmp_path):
    back = P.read_layer(tmp_path / "barriers.geojson", P.BARRIER_COLUMNS)
    assert len(back) == 0 and "unavoidability" in back.columns


def test_a_layer_with_rows_round_trips(tmp_path):
    import geopandas as gpd
    from shapely.geometry import LineString
    gdf = gpd.GeoDataFrame(
        {"edge_id": [7], "name": ["Gap Road"], "name_zh": ["峽道"],
         "pass_elev_m": [40.0], "pairs_served": [2]},
        geometry=[LineString([(833000, 815000), (833100, 815000)])], crs=CRS)
    gj, csv = tmp_path / "passes.geojson", tmp_path / "passes.csv"
    P._write_layer(gdf, gj, csv, P.PASS_COLUMNS)
    back = P.read_layer(gj, P.PASS_COLUMNS)
    assert back["name_zh"].iloc[0] == "峽道"
    assert back["pass_elev_m"].iloc[0] == pytest.approx(40.0)
    assert back.crs.to_epsg() == 4326        # GeoJSON is longitude and latitude
    assert "nearest_neighborhood" in back.columns   # absent in the file, added


# --------------------------------------------------------------------------
# the written analysis: landmasses and what a pass separates
# --------------------------------------------------------------------------
def test_landmasses_are_named_by_the_neighborhoods_on_them():
    from hk_flat_routes.report import name_landmasses
    members = {
        0: ["Tsim Sha Tsui", "Mong Kok", "Kwai Chung"],
        1: ["Central District", "Wan Chai", "Stanley"],
        2: ["Tung Chung Town Centre Area", "Discovery Bay"],
        3: ["Tsing Yi"],
    }
    names = name_landmasses(members, area={"Tsing Yi": 10.7})
    assert names[0] == "Kowloon and the New Territories"
    assert names[1] == "Hong Kong Island"
    assert names[2] == "Lantau"
    assert names[3] == "Tsing Yi"            # no listed name: its own


def test_a_landmass_name_is_given_only_once():
    """A test window can cut one landmass in two; only one piece keeps the name."""
    from hk_flat_routes.report import name_landmasses
    members = {0: ["Tsim Sha Tsui", "Mong Kok", "Hung Hom"],
               5: ["Sha Tin", "Tai Po"]}
    names = name_landmasses(members, area={"Sha Tin": 27.8, "Tai Po": 24.4})
    assert names[0] == "Kowloon and the New Territories"
    assert names[5] == "Sha Tin and 1 more"
    assert len(set(names.values())) == 2


def test_featured_pairs_stay_on_one_landmass():
    """Every featured pair must be routable: both ends on the same landmass.

    The grouping is the one the neighbourhood set was built with: plans on
    Hong Kong Island, plans on Lantau, and everything else on the mainland.
    """
    from hk_flat_routes.config import FEATURED_PAIRS
    island = {"Central District", "Mid-Levels West", "Sai Ying Pun & Sheung Wan",
              "The Peak Area", "Kennedy Town & Mount Davis", "Causeway Bay",
              "Wan Chai", "Aberdeen & Ap Lei Chau", "Stanley", "North Point",
              "Chai Wan", "Pok Fu Lam", "Wong Nai Chung",
              "Jardine's Lookout & Wong Nai Chung Gap"}
    lantau = {"Tung Chung Town Centre Area", "Discovery Bay", "Mui Wo Fringe",
              "South Lantau Coast", "Tai O Town Centre"}

    def landmass(name):
        return "island" if name in island else "lantau" if name in lantau else "mainland"

    assert len(FEATURED_PAIRS) == len(set(FEATURED_PAIRS))
    seen = {landmass(o) for o, _ in FEATURED_PAIRS}
    assert seen == {"island", "mainland", "lantau"}
    for o, d in FEATURED_PAIRS:
        assert o != d
        assert landmass(o) == landmass(d), (o, d)


def test_the_two_sides_of_a_pass():
    from hk_flat_routes.report import _separates, _sides
    pm = pd.DataFrame({
        "neighborhood_a": ["Stanley", "Stanley", "Chai Wan", "Central District"],
        "neighborhood_b": ["Wan Chai", "Central District", "Stanley", "Wan Chai"],
        "pass_elev_m": [107.0, 107.0, 107.0, 5.0],
        "pass_edge_id": [9, 9, 9, 4]})
    small, big = _sides(pm, 9)
    assert small == ["Stanley"]
    assert big == ["Central District", "Chai Wan", "Wan Chai"]
    assert _separates(small, big, {"Stanley": 6.0}) == \
        "Stanley (at 6 mPD) from the other 3"
    assert _sides(pm, 123) == ([], [])
    # a landmass of two neighbourhoods: the other side is named, not counted
    assert _separates(["Tai O Town Centre"], ["Tung Chung Town Centre Area"],
                      {"Tai O Town Centre": 2.0}) == \
        "Tai O Town Centre (at 2 mPD) from Tung Chung Town Centre Area"
    assert _separates(["Mui Wo Fringe", "South Lantau Coast"],
                      ["Tai O Town Centre", "Tung Chung Town Centre Area"], {}) == \
        "Mui Wo Fringe and South Lantau Coast from 2 others"
    assert _separates(["A", "B", "C"], list("DEFGH"), {}) == \
        "3 neighbourhoods from 5 others"


def test_neighbourhoods_are_found_by_either_name():
    from hk_flat_routes.__main__ import resolve_neighborhood
    names = ["Central District", "Kwun Tong (North)", "Kwun Tong (South)",
             "Tsz Wan Shan, Diamond Hill & San Po Kong", "Mong Kok"]
    zh = ["中區", "觀塘北部", "觀塘南部", "慈雲山、鑽石山及新蒲崗", "旺角"]
    assert resolve_neighborhood("central district", names, zh)[0] == "Central District"
    assert resolve_neighborhood("旺角", names, zh)[0] == "Mong Kok"
    assert resolve_neighborhood("diamond hill", names, zh)[0] == \
        "Tsz Wan Shan, Diamond Hill & San Po Kong"
    name, cand = resolve_neighborhood("Kwun Tong", names, zh)
    assert name is None and cand == ["Kwun Tong (North)", "Kwun Tong (South)"]
    name, cand = resolve_neighborhood("Atlantis", names, zh)
    assert name is None and cand == names
