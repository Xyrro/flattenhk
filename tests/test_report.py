"""Tests for the written analysis: what its sentences and tables say.

The report is generated from the outputs of a run, so each test builds the
few tables a section reads -- four neighbourhoods on Hong Kong Island, with
figures chosen so that every sentence under test has something to say --
and checks the words against them.
"""
import numpy as np
import pandas as pd

import hk_flat_routes.report as R
from hk_flat_routes.config import PARETO_LAMBDA_SWEEP, ROUTING_PROFILES
from hk_flat_routes.pairs import add_comparisons

CRS = "EPSG:2326"
PLACES = ["Central District", "Mid-Levels West", "Wan Chai", "Causeway Bay"]
HEIGHT = {"Central District": 4.0, "Mid-Levels West": 119.0, "Wan Chai": 4.0,
          "Causeway Bay": 5.0}
ISLAND = "Hong Kong Island"


def pair_table(with_paths: bool = True) -> pd.DataFrame:
    """Twelve ordered pairs under four objectives, on foot and by bicycle.

    On foot the shortest route climbs 100 m, 40 m of it on paths; the
    flattest climbs 40 m, except between Wan Chai and Causeway Bay, where
    shortest and flattest both climb 20 m.  By bicycle the flattest route
    from Central District climbs 3 m more than the shortest.
    """
    rows = []
    for mode in ("walk", "bike"):
        for o in PLACES:
            for d in PLACES:
                if o == d:
                    continue
                level = {o, d} == {"Wan Chai", "Causeway Bay"}
                for profile, dist, gain, grade, path_gain, path_m in (
                        ("shortest", 2000.0, 100.0, 0.60, 40.0, 200.0),
                        ("balanced", 2300.0, 50.0, 0.12, 0.0, 0.0),
                        ("min_climb", 2400.0, 40.0, 0.20, 2.0, 24.0),
                        ("grade_averse", 2800.0, 55.0, 0.08, 0.0, 0.0)):
                    if level:
                        gain, path_gain = 20.0, 0.0
                    if mode == "bike" and profile == "min_climb" and o == PLACES[0]:
                        gain = 103.0
                    rows.append({
                        "mode": mode, "profile": profile, "origin": o,
                        "destination": d, "distance_m": dist,
                        "elev_gain_m": gain, "max_grade": grade,
                        "start_elev_m": HEIGHT[o], "end_elev_m": HEIGHT[d],
                        "path_gain_m": path_gain, "path_distance_m": path_m})
    df = add_comparisons(pd.DataFrame(rows))
    if not with_paths:
        df = df.drop(columns=["path_gain_m", "path_distance_m"])
    return df.assign(landmass=ISLAND, landmass_to=ISLAND)


def sweep_table() -> pd.DataFrame:
    """The frontier file: per walking pair, the shortest route, the flattest
    one, and from Central District a route with 9 m less climbing still."""
    rows = []
    for o in PLACES:
        for d in PLACES:
            if o == d:
                continue
            level = {o, d} == {"Wan Chai", "Causeway Bay"}
            pts = [(2000.0, 20.0 if level else 100.0, 0.60),
                   (2400.0, 20.0 if level else 40.0, 0.20)]
            if o == PLACES[0]:
                pts.append((3100.0, 31.0, 0.05))
            for dist, gain, grade in pts:
                rows.append({"mode": "walk", "origin": o, "destination": d,
                             "distance_m": dist, "elev_gain_m": gain,
                             "max_grade": grade, "pareto_optimal": True})
    return pd.DataFrame(rows)


def run(**tables) -> dict:
    """The dictionary the report's sections read, for one landmass."""
    import geopandas as gpd
    from shapely.geometry import LineString

    edges = gpd.GeoDataFrame(
        {"edge_id": [0, 1, 2], "cls": ["residential", "steps", "path"],
         "length_m": [9000.0, 500.0, 500.0], "cum_gain_fwd": [90.0, 60.0, 50.0],
         "walk_ok": [True, True, True], "bike_ok": [True, False, False]},
        geometry=[LineString([(833000, 816000), (833100, 816000)])] * 3, crs=CRS)
    d = {"edges": edges, "pairs": pair_table(),
         "lm": [{"key": 1, "name": ISLAND, "neighborhoods": list(PLACES),
                 "km": 10.0, "possible": 12, "routed": 12}],
         "lm_of": {p: ISLAND for p in PLACES}, "lm_by_component": True,
         "lm_names": [ISLAND], "no_network": [], "n_pieces": 1,
         "other_pieces_km": [], "area": {}}
    d.update(tables)
    return d


def text(lines) -> str:
    return "\n".join(lines)


# --------------------------------------------------------------------------
# the headline: what "flattest" is, and what the baseline is
# --------------------------------------------------------------------------
def test_the_flattest_objective_is_not_called_the_minimum():
    """Its climbing weight is 120 and the sweep goes to 210: the table and
    the sentence under it say "strongly climb-averse" and give the rate."""
    out = text(R._headline(run(pareto=sweep_table())))
    assert "| Flattest (strongly climb-averse) | 2.40 km |" in out
    assert "minimum climbing" not in out and "minimum-climbing" not in out
    alpha = ROUTING_PROFILES["min_climb"].alpha
    assert f"accepts {alpha:.0f} m of extra walking to save one metre of climb" in out


def test_the_headline_says_how_far_the_flattest_is_from_the_least_climbing():
    """Three pairs out of Central District have a route in the sweep with
    9 m less climbing: counted, and the climbing avoided recomputed with
    the least found for every pair (52.2% against 50.0%: ten pairs avoid
    60% and two none, and three of the ten would avoid 69%)."""
    out = text(R._headline(run(pareto=sweep_table())))
    assert "It is not the least climbing possible." in out
    assert "for 3 of the 12 pairs (3 by more than 5 m, 9 m at most)" in out
    assert "at 52.2% where the flattest route gives 50.0%" in out
    # a run with no sweep makes no such comparison
    assert "least climbing possible" not in text(R._headline(run()))


def test_the_headline_says_how_much_of_the_baseline_is_on_paths():
    """Ten of the twelve shortest routes climb 100 m, 40 m of it on paths,
    and all twelve have 200 m of path in their 2 km: 400 m of 1,040 m of
    climbing (38%) on 10% of the length.  The flattest: 20 m of 440 m."""
    out = text(R._headline(run()))
    assert ("38% of the climbing on the shortest routes is done on hill paths, "
            "tracks and steps, which make up 10% of their length") in out
    assert "on the flattest routes it is 5% of the climbing and 1% of the length" in out
    # a pair table made before the path columns existed says nothing
    old = text(R._headline(run(pairs=pair_table(with_paths=False))))
    assert "hill paths" not in old


def test_where_flat_routing_does_not_pay_is_checked_against_the_sweep():
    """Wan Chai to Causeway Bay and back: 20 m either way, and the flattest
    route saves nothing.  "No route gets round it" is a claim about every
    route, so it is made only of the sweep, and only where the sweep agrees."""
    out = text(R._featured(run(pareto=sweep_table())))
    assert "in the other 2 the climb is there and the flattest route does not get round it" in out
    assert "Nor, for 2 of them, does any route of the wider sweep" in out
    assert "no route gets round it" not in out
    # give the sweep a route that does save 8 m one way
    better = pd.concat([sweep_table(), pd.DataFrame([{
        "mode": "walk", "origin": "Wan Chai", "destination": "Causeway Bay",
        "distance_m": 5000.0, "elev_gain_m": 12.0, "max_grade": 0.03,
        "pareto_optimal": True}])], ignore_index=True)
    out = text(R._featured(run(pareto=better)))
    assert ("Nor, for 1 of them, does any route of the wider sweep of weights "
            "described further down; for the other 1 one of its routes does "
            "avoid 5 m or more (8 m at most).") in out


# --------------------------------------------------------------------------
# corridors
# --------------------------------------------------------------------------
def corridor_table(**columns):
    import geopandas as gpd
    from shapely.geometry import LineString
    rows = [
        # name, score, grade, climb per km, lowest, highest
        ("King's Road", 90.0, 0.005, 2.6, 4.0, 8.0),
        ("Shek O Road", 40.0, 0.005, 5.1, 136.0, 151.0),
        ("Repulse Bay Road", 30.0, 0.051, 25.5, 97.0, 184.0),
        ("Victoria Road", 20.0, 0.022, 9.2, 28.0, 82.0),
    ]
    c = gpd.GeoDataFrame(
        [{"corridor_name": n, "mode": "walk", "total_score": s, "length_km": 2.0,
          "mean_abs_grade": g, "gain_per_km": k, "pair_count_max": 9,
          "neighborhood_span": 4, "elev_min_m": lo, "elev_max_m": hi,
          "climb_saved_m": 5400.0, "flat_routes": 7,
          "flat_routes_saved_m": 19100.0, "neighborhoods": "Wan Chai",
          "landmass": 1} for n, s, g, k, lo, hi in rows],
        geometry=[LineString([(833000, 816000), (835000, 816000)])] * len(rows),
        crs=CRS)
    for name, value in columns.items():
        c[name] = value
    return c


def test_corridor_ground_is_one_of_four_kinds():
    kinds = R._corridor_kinds(corridor_table())
    assert kinds.tolist() == ["low", "contour", "over", ""]


def test_the_corridor_section_is_not_headed_low_elevation():
    """Repulse Bay Road at 97-184 mPD and a mean gradient of 5.1% is in the
    list for being the least steep way over; the section says so."""
    out = text(R._corridors(run(corridors=corridor_table())))
    assert out.startswith("## The corridors flat routes use")
    assert "low-elevation corridors" not in out
    assert "| 97-184 mPD, a way over high ground |" in out
    assert "| 136-151 mPD, a contour road |" in out
    assert "| 4-8 mPD | Wan Chai |" in out
    assert "### Low ground, contour roads and ways over" in out
    assert "1 of the 4 lie wholly below 15 mPD" in out
    assert "2 never drop below 50 mPD. 1 of those hold a mean gradient of 2% or less" in out
    assert ("The other 1 is neither level nor low. Its mean gradient is 5.1%, "
            "and it is in the list because") in out
    assert "Climb per km is the mean of the two directions of travel." in out


def test_what_the_routes_along_a_corridor_avoid_is_the_computed_figure():
    """The score (5.4 km here) shares each route's saving out by length;
    the sentence is about the routes, 7 of them and 19.1 km."""
    out = text(R._corridors(run(corridors=corridor_table())))
    assert ("The flattest routes of 7 ordered pairs run along some part of "
            "it, and between them those routes climb 19.1 km less than the "
            "shortest routes for the same pairs.") in out
    assert "avoid 5.4 km" not in out and "climbing between them" not in out


def test_an_older_corridor_table_gets_the_score_called_what_it_is():
    old = corridor_table().drop(columns=["flat_routes", "flat_routes_saved_m"])
    out = text(R._corridors(run(corridors=old)))
    assert ("its share of the climbing those routes avoid, apportioned by "
            "length over the three climb-averse objectives, comes to 5.4 km.") in out
    assert "between them" not in out


# --------------------------------------------------------------------------
# passes: access points, kinds and the way round
# --------------------------------------------------------------------------
def pass_tables(divide: bool = False):
    """The pass matrix and the passes of the four places.

    Mid-Levels West stands at 119 mPD behind Castle Road at 125; Wan Chai
    and Causeway Bay are parted by a 6 m rise.  As a rule Central District
    then lies behind an unnamed hill path at 30 mPD from the two of them.
    With ``divide`` the hill path is at 130 mPD instead and parts Central
    District and Mid-Levels West, together, from the other two.
    """
    import geopandas as gpd
    from shapely.geometry import LineString
    top, mlw = (130.0, (130.0, 30)) if divide else (30.0, (125.0, 20))
    pm = pd.DataFrame(
        [("Mid-Levels West", "Central District", 125.0, 20, 900.0),
         ("Wan Chai", "Causeway Bay", 6.0, 21, 2000.0),
         ("Central District", "Wan Chai", top, 30, 9000.0),
         ("Central District", "Causeway Bay", top, 30, 11000.0),
         ("Mid-Levels West", "Wan Chai", *mlw, 13000.0),
         ("Mid-Levels West", "Causeway Bay", *mlw, 15000.0)],
        columns=["neighborhood_a", "neighborhood_b", "pass_elev_m",
                 "pass_edge_id", "low_route_m"])
    served = pm.groupby("pass_edge_id").size()
    pz = gpd.GeoDataFrame(
        [{"edge_id": 20, "pass_elev_m": 125.0, "name": "Castle Road",
          "cls": "residential", "max_abs_grade": 0.127,
          "neighborhood": "Mid-Levels West"},
         {"edge_id": 30, "pass_elev_m": top, "name": None, "cls": "path",
          "max_abs_grade": 0.21, "neighborhood": "Wan Chai"},
         {"edge_id": 21, "pass_elev_m": 6.0, "name": "Hennessy Road",
          "cls": "primary", "max_abs_grade": 0.01, "neighborhood": "Wan Chai"}],
        geometry=[LineString([(833000, 816000), (833100, 816000)])] * 3, crs=CRS)
    pz["pairs_served"] = pz["edge_id"].map(served)
    pz["nearest_neighborhood"] = pz["neighborhood"]
    pz["nearest_distance_m"] = 0.0
    pz = pz.sort_values("pairs_served", ascending=False, kind="stable")
    return pm, pz.reset_index(drop=True)


def points(median_mlw: float) -> pd.DataFrame:
    return pd.DataFrame({
        "neighborhood": PLACES, "node": ["a", "b", "c", "d"],
        "elev_m": [HEIGHT[p] for p in PLACES],
        "street_median_m": [4.0, median_mlw, 4.0, 5.0], "offset_m": 100.0})


def test_report_and_access_rule_use_the_same_band():
    """The report singles out the points the height rule could not bring
    down; it can only do that if the two agree on what "down" is."""
    import hk_flat_routes.neighborhoods as N
    assert R.ACCESS_ABOVE_STREETS_M == N.ACCESS_HEIGHT_BAND_M


def test_points_above_their_streets_are_found_from_the_points_file():
    d = run(points_walk=points(60.0))
    assert R._points_above_streets(d) == {"Mid-Levels West": (119.0, 60.0)}
    assert R._points_above_streets(run(points_walk=points(112.0))) == {}
    # a points file from before the heights were recorded says nothing
    old = points(60.0).drop(columns=["elev_m", "street_median_m"])
    assert R._points_above_streets(run(points_walk=old)) == {}
    assert R._points_above_streets(run()) == {}


def test_a_high_access_point_is_not_reported_as_a_hill_neighbourhood():
    """Mid-Levels West's point at 119 mPD with its streets at 60: the pass
    at its doorstep is the height of the point, and is called that."""
    pm, pz = pass_tables()
    out = text(R._passes(run(pass_matrix=pm, passes=pz, barriers=pd.DataFrame(),
                             points_walk=points(60.0))))
    assert ("Mid-Levels West (its access point at 119 mPD, its streets at "
            "60 mPD) from the other 3") in out
    assert ("Its access point is at 119 mPD, 59 m above the median height of "
            "the neighbourhood's own streets (60 mPD), so this is the height "
            "of the point chosen for it rather than of the neighbourhood") in out
    assert ("The other 1 is the doorstep of an access point and not of a "
            "neighbourhood: Mid-Levels West, whose point stands more than "
            "10 m above the median height of its own streets.") in out
    assert "1 could not be and stands higher (Mid-Levels West)." in out
    assert "height of the neighbourhood: whoever goes there" not in out


def test_a_point_at_the_height_of_its_streets_is_the_neighbourhoods_height():
    pm, pz = pass_tables()
    out = text(R._passes(run(pass_matrix=pm, passes=pz, barriers=pd.DataFrame(),
                             points_walk=points(112.0))))
    assert "Mid-Levels West (at 119 mPD) from the other 3" in out
    assert "less a saddle than the height of the neighbourhood" in out
    assert "doorstep of an access point" not in out
    assert "so an end that high is where the neighbourhood's streets are." in out


def test_the_divide_sentence_gives_the_length_of_the_way_round():
    """Four pairs across the divide: 2 km each by the shortest route, and
    9, 11, 13 and 15 km without rising above 130 mPD."""
    pm, pz = pass_tables(divide=True)
    d = run(pass_matrix=pm, passes=pz, barriers=pd.DataFrame())
    out = text(R._passes(d))
    assert ("The divide that binds most pairs is (unnamed hill path), Wan "
            "Chai, at 130 mPD: Central District and Mid-Levels West lie on "
            "one side of it and Causeway Bay and Wan Chai on the other.") in out
    assert ("For the 4 pairs it divides, the shortest route that never rises "
            "above 130 mPD is 12 km at the median, where their shortest route "
            "is 2 km; from Mid-Levels West to Causeway Bay it is 15 km "
            "against 2 km.") in out
    # a pass matrix from before the lengths were worked out
    out = text(R._passes(run(pass_matrix=pm.drop(columns=["low_route_m"]),
                             passes=pz, barriers=pd.DataFrame())))
    assert "never rises above" not in out and "The divide that binds" in out


def test_most_of_the_height_is_claimed_only_of_most_pairs():
    """Here 1 pair in 6 has its lowest crossing at one end of the trip."""
    pm, pz = pass_tables()
    out = text(R._passes(run(pass_matrix=pm, passes=pz, barriers=pd.DataFrame())))
    assert "Much of that height is not a ridge in the way" in out
    assert "Most of that height" not in out


def test_low_route_sentence_needs_both_figures():
    pm, _pz = pass_tables()
    rows = pm[pm["pass_edge_id"] == 30]
    assert R._low_route_sentence(rows, {}, 30.0) == ""
    assert R._low_route_sentence(rows.drop(columns=["low_route_m"]),
                                 {("Central District", "Wan Chai"): 2000.0}, 30.0) == ""


# --------------------------------------------------------------------------
# barriers
# --------------------------------------------------------------------------
def test_a_way_that_is_not_a_street_is_labelled():
    assert R._way_label({"name": "牛乸", "cls": "path"}) == "牛乸 (hill path)"
    assert R._way_label({"name": np.nan, "cls": "path"}) == "(unnamed hill path)"
    assert R._way_label({"name": "獅紅古道", "cls": "footway"}) == "獅紅古道 (footpath)"
    assert R._way_label({"name": None, "cls": "steps"}) == "(unnamed steps)"
    assert R._way_label({"name": "On Kin Road", "cls": "residential"}) == "On Kin Road"
    assert R._way_label({"name": None, "cls": "service"}) == "(unnamed service)"


def test_the_barrier_table_calls_hill_paths_hill_paths():
    b = pd.DataFrame({
        "name": ["牛乸", None, "On Kin Road"], "cls": ["path", "footway", "residential"],
        "max_abs_grade": [0.5, 0.37, 0.14], "shortest_use": [44, 46, 56],
        "flat_use_per_objective": [0.0, 0.0, 50.0],
        "unavoidability": [0.0, 0.0, 0.9], "barrier_score": [9.0, 8.0, 7.0],
        "neighborhood": [None, None, "Wan Chai"],
        "nearest_neighborhood": ["Wan Chai"] * 3,
        "nearest_distance_m": [3000.0, 500.0, 0.0]})
    out = text(R._barriers({"barriers": b}))
    assert "| Street or path | Where |" in out
    assert "| 牛乸 (hill path) | 3 km from Wan Chai |" in out
    assert "| (unnamed footpath) | near Wan Chai |" in out
    assert "| On Kin Road | Wan Chai |" in out
    assert ("2 of the 3 are not streets but footpaths, hill paths, tracks and "
            "flights of steps") in out and "2 of those are avoidable" in out
    assert "On Kin Road, Wan Chai (14%, 56 pairs)" in out


# --------------------------------------------------------------------------
# the trade-off section
# --------------------------------------------------------------------------
def test_frontier_rows_fall_in_climb_as_they_rise_in_distance():
    """The frontier file is non-dominated over distance, climbing *and*
    steepest gradient, so by distance its climbing can go up: the real rows
    for Tsim Sha Tsui to Kowloon Tong ran 4.97 km / 35 m, 5.06 km / 37 m."""
    g = pd.DataFrame(
        [(4550.0, 65.0, 0.23), (4750.0, 47.0, 0.05), (4970.0, 35.0, 0.16),
         (5060.0, 37.0, 0.05), (5290.0, 35.4, 0.05), (5750.0, 32.0, 0.05)],
        columns=["distance_m", "elev_gain_m", "max_grade"])
    rows = R._frontier_rows(g.sample(frac=1.0, random_state=0))
    assert rows == ["| 4.55 km | 65 m | 23% |", "| 4.75 km | 47 m | 5% |",
                    "| 4.97 km | 35 m | 16% |", "| 5.75 km | 32 m | 5% |"]


def test_frontier_rows_that_print_alike_are_given_once():
    """Two routes 3 m and 8 cm of climb apart are one row to a reader; so
    are two of one printed length, of which the lower climb is the row."""
    g = pd.DataFrame(
        [(1828.9, 133.85, 0.49), (1832.3, 133.77, 0.49), (4530.0, 131.0, 0.17),
         (4532.0, 128.2, 0.17)],
        columns=["distance_m", "elev_gain_m", "max_grade"])
    assert R._frontier_rows(g) == ["| 1.83 km | 134 m | 49% |",
                                   "| 4.53 km | 128 m | 17% |"]


def test_the_sweep_is_described_as_running_past_the_flattest_objective():
    out = text(R._pareto(run(pareto=sweep_table())))
    top = max(PARETO_LAMBDA_SWEEP)
    assert f"from zero (pure distance) to {top:g} times their size" in out
    assert ("The sweep runs past that objective: at its far end a metre of "
            f"climb is priced at {top * ROUTING_PROFILES['balanced'].alpha:.0f} "
            f"m of walking, against {ROUTING_PROFILES['min_climb'].alpha:.0f} m") in out
    assert "up to the minimum-climbing objective" not in out
    assert "flattest possible route" not in out
    assert "Taken to the least-climbing route, the median pair sheds" in out
    # the featured pair's table: shortest, flattest, and the 31 m route
    assert "**Central District to Mid-Levels West** (Hong Kong Island)" in out
    for row in ("| 2.00 km | 100 m | 60% |", "| 2.40 km | 40 m | 20% |",
                "| 3.10 km | 31 m | 5% |"):
        assert row in out


def test_sweep_least_is_the_least_climbing_per_pair():
    least = R._sweep_least(run(pareto=sweep_table()))
    assert least[("Central District", "Wan Chai")] == 31.0
    assert least[("Wan Chai", "Central District")] == 40.0
    assert R._sweep_least(run()) is None
    assert R._sweep_least(run(pareto=sweep_table().iloc[0:0])) is None


# --------------------------------------------------------------------------
# walking and cycling
# --------------------------------------------------------------------------
def test_a_bicycles_flattest_route_can_climb_more_than_its_shortest():
    """Road class is in the bicycle's objective; the three routes out of
    Central District climb 103 m against the shortest's 100."""
    out = text(R._modes(run()))
    assert ("for 3 of the 12 pairs it climbs more than the shortest route "
            "does, by 3 m at most") in out
    level = pair_table()
    level.loc[level["mode"] == "bike", "gain_saved_m"] = 0.0
    assert "climbs more than the shortest" not in text(R._modes(run(pairs=level)))


def test_pushed_links_are_not_counted_as_footways_open_to_bicycles():
    """A pushed link is a footway with ``bike_ok`` set, so that a route can
    use it.  It is not a footway the data opens to bicycles by name, and the
    kilometres a bicycle may be ridden on do not include it."""
    import geopandas as gpd
    from shapely.geometry import LineString

    from hk_flat_routes.config import BIKE_PUSH_MAX_M, BIKE_PUSH_MULTIPLIER
    edges = gpd.GeoDataFrame(
        {"edge_id": range(5),
         "cls": ["residential", "steps", "footway", "footway", "footway"],
         "length_m": [9000.0, 500.0, 3000.0, 1000.0, 2000.0],
         "cum_gain_fwd": 0.0, "walk_ok": True,
         "bike_ok": [True, False, False, True, True],
         "bike_push": [False, False, False, False, True]},
        geometry=[LineString([(833000, 816000), (833100, 816000)])] * 5, crs=CRS)
    out = text(R._modes(run(edges=edges)))
    assert "a bicycle may be ridden on 10 km" in out
    assert "that keeps 1 km of the 6 km a walker has of them" in out
    assert (f"A further 2 km of them, in runs of at most {BIKE_PUSH_MAX_M:.0f} m "
            f"that join two parts of the cycling network, is open to a bicycle "
            f"pushed on foot, at {BIKE_PUSH_MULTIPLIER:g} times its length") in out
    # an edge table from before there were pushed links says nothing of them
    before = text(R._modes(run(edges=edges[~edges["bike_push"]]
                               .drop(columns=["bike_push"]))))
    assert "pushed" not in before and "a bicycle may be ridden on 10 km" in before


# --------------------------------------------------------------------------
# limits
# --------------------------------------------------------------------------
def test_the_limits_say_how_access_points_are_placed(monkeypatch):
    import geopandas as gpd
    from shapely.geometry import box
    monkeypatch.setattr(R, "_outside_share", lambda d: (0.0, 0.0))
    nb = gpd.GeoDataFrame({"neighborhood": PLACES, "area_km2": [1.0, 2.0, 3.0, 4.0]},
                          geometry=[box(0, 0, 1, 1)] * 4, crs=CRS)
    out = text(R._limits(run(neighborhoods=nb, points_walk=points(60.0))))
    assert ("It is taken within 10 m of the median height of the area's "
            "ordinary streets") in out
    assert "1 of the points could not be and stands higher." in out
    out = text(R._limits(run(neighborhoods=nb, points_walk=points(112.0))))
    assert "one point cannot stand for both. A trip to the upper streets" in out
