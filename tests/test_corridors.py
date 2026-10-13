"""Tests for corridor scoring and merging.

Synthetic streets on the HK1980 Grid (EPSG:2326), named the way Hong Kong's
are: an English name and a Chinese one, or sometimes only the Chinese.
"""
import pandas as pd
import pytest

from hk_flat_routes.config import ANALYSIS
from hk_flat_routes.corridors import (_canonical_name, _chinese_only,
                                      _clean_names, _display_names,
                                      english_names, merge_corridors,
                                      score_edges)

#: A corner of the grid near Central, so coordinates look like real ones.
X0, Y0 = 833000.0, 816000.0


def edge_table(rows):
    """Edges laid end to end along one street, node ``i`` to node ``i + 1``."""
    import geopandas as gpd
    from shapely.geometry import LineString
    recs = []
    for i, r in enumerate(rows):
        d = {"edge_id": i, "u": i, "v": i + 1, "name": "Queen's Road Central",
             "name_zh": "皇后大道中", "cls": "residential", "length_m": 100.0,
             "max_abs_grade": 0.02, "avg_grade_fwd": 0.01, "cum_gain_fwd": 1.0,
             "elev_min": 4.0, "elev_max": 6.0, "low_stress": True,
             "bike_facility": "", "grade_reliable": True,
             "geometry": LineString([(X0 + i * 100, Y0), (X0 + i * 100 + 100, Y0)])}
        d.update(r)
        recs.append(d)
    return gpd.GeoDataFrame(recs, geometry="geometry", crs="EPSG:2326")


def usage_table(rows):
    recs = []
    for i, r in enumerate(rows):
        d = {"edge_id": i, "mode": "walk", "pair_count": 10,
             "neighborhood_span": 5, "climb_saved_m": 100.0,
             "detour_efficiency": 0.1, "pareto_count": 0}
        d.update(r)
        recs.append(d)
    return pd.DataFrame(recs)


def test_steep_edges_score_zero_however_heavily_used():
    """A busy steep street is a barrier, not a flat corridor."""
    e = edge_table([{"max_abs_grade": 0.02, "avg_grade_fwd": 0.01},
                    {"max_abs_grade": 0.22, "avg_grade_fwd": 0.19}])
    u = usage_table([{"pair_count": 300, "neighborhood_span": 30},
                     {"pair_count": 300, "neighborhood_span": 30}])
    s = score_edges(u, e)
    assert s.loc[s.edge_id == 0, "score"].iloc[0] > 0
    assert s.loc[s.edge_id == 1, "score"].iloc[0] == 0


def test_score_rises_with_usage_and_with_breadth():
    e = edge_table([{}, {}, {}])
    u = usage_table([
        {"pair_count": 5, "neighborhood_span": 3},
        {"pair_count": 50, "neighborhood_span": 3},
        {"pair_count": 50, "neighborhood_span": 30},
    ])
    s = score_edges(u, e).set_index("edge_id")["score"]
    assert s[0] < s[1] < s[2]


def test_a_short_stub_is_judged_on_average_grade_not_its_maximum():
    """A 5 m DEM artefact must not disqualify an otherwise flat block.

    Regression test carried over from upstream, where a 5 m connector
    reported a 41% gradient from a 1.3 m artefact and knocked flat blocks
    out of corridors.  Hong Kong's network is full of such stubs: the few
    metres between a footbridge landing and the pavement.
    """
    e = edge_table([{"length_m": 5.0, "max_abs_grade": 0.41,
                     "avg_grade_fwd": 0.01, "grade_reliable": False}])
    s = score_edges(usage_table([{}]), e)
    assert bool(s["flat_enough"].iloc[0])
    assert s["score"].iloc[0] > 0


def test_a_long_steep_edge_is_still_excluded():
    e = edge_table([{"length_m": 90.0, "max_abs_grade": 0.41,
                     "avg_grade_fwd": 0.01, "grade_reliable": True}])
    s = score_edges(usage_table([{}]), e)
    assert not bool(s["flat_enough"].iloc[0])


def test_climb_saved_increases_the_score():
    e = edge_table([{}, {}])
    u = usage_table([{"climb_saved_m": 0.0}, {"climb_saved_m": 5000.0}])
    s = score_edges(u, e).set_index("edge_id")["score"]
    assert s[1] > s[0]


def test_average_grade_limit_comes_from_configuration():
    lim = ANALYSIS.corridor_max_avg_grade
    e = edge_table([{"avg_grade_fwd": lim - 0.001},
                    {"avg_grade_fwd": lim + 0.001}])
    s = score_edges(usage_table([{}, {}]), e).set_index("edge_id")
    assert bool(s.loc[0, "flat_enough"])
    assert not bool(s.loc[1, "flat_enough"])


def test_clean_names_drops_nan_and_blanks():
    assert _clean_names(["A", None, float("nan"), "", "  ", "B"]) == ["A", "B"]


def test_canonical_name_uses_the_most_common_streets():
    names = ["King's Road"] * 5 + ["Java Road"] * 2 + [None]
    label = _canonical_name(names)
    assert label.startswith("King's Road")
    assert "Java Road" in label


def test_canonical_name_of_all_unnamed():
    assert _canonical_name([None, float("nan")]) == "unnamed corridor"
    assert _canonical_name([None], unnamed="") == ""


# --------------------------------------------------------------------------
# two names per street
# --------------------------------------------------------------------------
def test_chinese_only_names_are_recognised():
    assert _chinese_only("吉席街")
    assert not _chinese_only("Catchick Street")
    assert not _chinese_only("堅彌地城海旁 Praya, Kennedy Town")
    assert not _chinese_only(None) and not _chinese_only(float("nan"))
    assert not _chinese_only("  ")


def test_a_block_mapped_only_in_chinese_joins_its_street():
    """One street, three blocks, the middle one without its English name.

    Counted as mapped it would be two streets and the corridor would be
    labelled "Catchick Street - 吉席街".
    """
    names = ["Catchick Street", "吉席街", "Catchick Street"]
    names_zh = ["吉席街", None, "吉席街"]
    en, zh = _display_names(names, names_zh)
    assert en == ["Catchick Street"] * 3
    assert zh == ["吉席街"] * 3
    assert _canonical_name(en) == "Catchick Street"
    assert _canonical_name(zh, unnamed="") == "吉席街"


def test_the_rest_of_the_network_can_supply_the_english_name():
    en, zh = _display_names(["德輔道中", None], [None, None],
                            english={"德輔道中": "Des Voeux Road Central"})
    assert en == ["Des Voeux Road Central", None]
    assert zh == ["德輔道中", None]


def test_an_unresolved_chinese_name_is_kept_not_dropped():
    en, zh = _display_names(["無名里"], [None])
    assert en == ["無名里"] and zh == ["無名里"]


def test_english_names_skips_ambiguous_chinese_names():
    """Two villages can each have a 大街; neither English name may be guessed."""
    e = edge_table([
        {"name": "Hennessy Road", "name_zh": "軒尼詩道"},
        {"name": "Hennessy Road", "name_zh": "軒尼詩道"},
        {"name": "Main Street, Ap Lei Chau", "name_zh": "大街"},
        {"name": "Tai O Main Street", "name_zh": "大街"},
        {"name": "吉席街", "name_zh": None},
    ])
    table = english_names(e)
    assert table == {"軒尼詩道": "Hennessy Road"}
    assert english_names(e.drop(columns=["name_zh"])) == {}


def test_merged_corridor_has_one_english_and_one_chinese_label():
    e = edge_table([
        {"name": "Catchick Street", "name_zh": "吉席街"},
        {"name": "吉席街", "name_zh": None},
        {"name": "Catchick Street", "name_zh": "吉席街"},
        {"name": "Catchick Street", "name_zh": "吉席街"},
    ])
    scored = score_edges(usage_table([{}, {}, {}, {}]), e)
    cor = merge_corridors(scored, e, quantile=0.0)
    assert len(cor) == 1
    assert cor["corridor_name"].iloc[0] == "Catchick Street"
    assert cor["corridor_name_zh"].iloc[0] == "吉席街"
    assert cor["street_names"].iloc[0] == "Catchick Street"
    assert cor["length_m"].iloc[0] == 400.0
    assert cor.crs.to_epsg() == 2326


def test_corridors_work_without_chinese_names():
    """An edge table with no name_zh column (upstream's shape) still merges."""
    e = edge_table([{}, {}, {}, {}]).drop(columns=["name_zh"])
    scored = score_edges(usage_table([{}, {}, {}, {}]), e)
    cor = merge_corridors(scored, e, quantile=0.0)
    assert len(cor) == 1
    assert cor["corridor_name"].iloc[0] == "Queen's Road Central"
    assert cor["corridor_name_zh"].iloc[0] == ""


# --------------------------------------------------------------------------
# one threshold per landmass
# --------------------------------------------------------------------------
def two_shores():
    """Two streets of ten blocks with no link between them: two landmasses.

    The first is on a landmass with many neighbourhoods and its blocks are
    each used by hundreds of pairs; the second is on a small island where
    a handful of pairs is all there can be.
    """
    from types import SimpleNamespace
    rows = [{"name": "Nathan Road", "name_zh": "彌敦道"} for _ in range(10)]
    rows += [{"name": "South Lantau Road", "name_zh": "嶼南道"} for _ in range(10)]
    e = edge_table(rows)
    # break the chain between the two streets: the island gets its own nodes
    e.loc[10:, "u"] = range(100, 110)
    e.loc[10:, "v"] = range(101, 111)
    nodes = list(range(0, 11)) + list(range(100, 111))
    graph = SimpleNamespace(mode="walk", node_ids=nodes,
                            node_comp=[0] * 11 + [1] * 11)
    busy = [{"pair_count": 300, "neighborhood_span": 30,
             "climb_saved_m": 4000.0} for _ in range(10)]
    quiet = [{"pair_count": 3, "neighborhood_span": 4,
              "climb_saved_m": 40.0} for _ in range(10)]
    return e, graph, usage_table(busy + quiet)


def test_edges_know_their_landmass():
    from hk_flat_routes.corridors import edge_landmass
    e, graph, _ = two_shores()
    lm = edge_landmass(e, graph)
    assert list(lm.loc[:9]) == [0.0] * 10 and list(lm.loc[10:]) == [1.0] * 10
    from types import SimpleNamespace
    assert edge_landmass(e, SimpleNamespace(node_ids=[], node_comp=None)).isna().all()


def test_a_small_landmass_keeps_its_corridors():
    """One threshold for everything lets the big landmass set the bar.

    Every island block scores below every mainland block, so a quantile
    taken over both leaves the island with nothing; taken landmass by
    landmass, each keeps its own best street.
    """
    from hk_flat_routes.corridors import detect_corridors
    e, graph, usage = two_shores()
    _, together = detect_corridors(usage, e, graph, per_landmass=False)
    assert set(together["corridor_name"]) == {"Nathan Road"}
    scored, apart = detect_corridors(usage, e, graph, per_landmass=True)
    assert set(apart["corridor_name"]) == {"Nathan Road", "South Lantau Road"}
    assert dict(zip(apart["corridor_name"], apart["landmass"])) == {
        "Nathan Road": 0, "South Lantau Road": 1}
    assert set(scored["landmass"]) == {0, 1}
    # ranked by score across landmasses, and numbered in that order
    assert list(apart["corridor_id"]) == [0, 1]
    assert apart["total_score"].is_monotonic_decreasing


def test_one_landmass_is_scored_as_upstream_scores_it():
    """Where the network is one piece the two ways are the same thing."""
    from hk_flat_routes.corridors import detect_corridors
    e, graph, usage = two_shores()
    mainland, usage = e.iloc[:10], usage.iloc[:10]
    graph.node_comp = [0] * len(graph.node_ids)
    _, a = detect_corridors(usage, mainland, graph, per_landmass=True)
    _, b = detect_corridors(usage, mainland, None, per_landmass=False)
    assert list(a["corridor_name"]) == list(b["corridor_name"])
    assert a["length_m"].tolist() == b["length_m"].tolist()
    assert a["total_score"].tolist() == pytest.approx(b["total_score"].tolist())


def test_pieces_too_short_to_be_corridors_give_an_empty_table():
    """A landmass whose best street is one block long has no corridor.

    Upstream built a frame from the empty list and failed; with a threshold
    per landmass, a two-neighbourhood island makes this the ordinary case.
    """
    e = edge_table([{}, {}])                      # 200 m in all, under the minimum
    scored = score_edges(usage_table([{}, {}]), e)
    cor = merge_corridors(scored, e, quantile=0.0)
    assert cor.empty and list(cor.columns) == ["corridor_id", "geometry"]


def test_no_usage_gives_no_corridors():
    from hk_flat_routes.corridors import detect_corridors
    e, graph, usage = two_shores()
    scored, cor = detect_corridors(usage.iloc[0:0], e, graph)
    assert scored.empty and cor.empty and "corridor_id" in cor.columns


# --------------------------------------------------------------------------
# what the routes along a corridor avoid
# --------------------------------------------------------------------------
def one_street_two_corridors():
    """Six blocks in a row; blocks 0-2 and 3-5 are two corridors.

    Three flattest routes: A -> B over blocks 0-4 (both corridors), saving
    30 m; C -> D over blocks 0-1, saving 12 m; E -> F over block 5 alone,
    which climbs 2 m *more* than its shortest route.  The same pairs under
    another objective must not be counted, nor a route with no pair row.
    """
    from types import SimpleNamespace
    import geopandas as gpd
    from shapely.geometry import LineString
    table = pd.DataFrame({"edge_id": [0, 1, 2, 3, 4, 5, 9]})
    graph = SimpleNamespace(mode="walk", table=table)
    arc_store = {("min_climb", "A", "B"): [0, 1, 2, 3, 4],
                 ("min_climb", "C", "D"): [0, 1],
                 ("min_climb", "E", "F"): [5],
                 ("min_climb", "G", "H"): [6],            # off every corridor
                 ("min_climb", "X", "Y"): [0, 5],         # no row in the table
                 ("balanced", "A", "B"): [0, 1, 2, 3, 4, 5],
                 ("min_climb", "N", "N"): []}
    rows = [("min_climb", "A", "B", 30.0), ("min_climb", "C", "D", 12.0),
            ("min_climb", "E", "F", -2.0), ("min_climb", "G", "H", 50.0),
            ("balanced", "A", "B", 25.0)]
    pairs = pd.DataFrame(rows, columns=["profile", "origin", "destination",
                                        "gain_saved_m"]).assign(mode="walk")
    bike = pairs.assign(mode="bike", gain_saved_m=1000.0)
    line = LineString([(X0, Y0), (X0 + 300, Y0)])
    corridors = gpd.GeoDataFrame(
        {"corridor_id": [0, 1], "edge_ids": [[0, 1, 2], [3, 4, 5]],
         "climb_saved_m": [7.0, 3.0]},
        geometry=[line, line], crs="EPSG:2326")
    return corridors, graph, arc_store, pd.concat([pairs, bike], ignore_index=True)


def test_each_route_along_a_corridor_counts_once_and_in_full():
    """``climb_saved_m`` shares a route's saving out by length over three
    objectives; a sentence that says "the routes that use it avoid X km of
    climbing between them" needs each route's whole saving, once."""
    from hk_flat_routes.corridors import route_savings
    corridors, graph, arc_store, pairs = one_street_two_corridors()
    out = route_savings(corridors, graph, arc_store, pairs)
    assert out["flat_routes"].tolist() == [2, 2]
    # A->B counts in full on both corridors, though it uses three blocks of
    # one and two of the other; E->F's extra climb comes off the second
    assert out["flat_routes_saved_m"].tolist() == [42.0, 28.0]
    assert out["climb_saved_m"].tolist() == [7.0, 3.0]        # the score stays
    assert "flat_routes" not in corridors.columns             # a copy


def test_route_savings_of_no_corridors_is_an_empty_table_with_the_columns():
    from hk_flat_routes.corridors import route_savings
    corridors, graph, arc_store, pairs = one_street_two_corridors()
    out = route_savings(corridors.iloc[0:0], graph, arc_store, pairs)
    assert out.empty and {"flat_routes", "flat_routes_saved_m"} <= set(out.columns)


def test_merged_corridors_remember_their_edges():
    """Same-street pieces fused into one corridor bring their edges along."""
    rows = [{} for _ in range(4)] + [{"name": "Other Street", "name_zh": None}] \
        + [{} for _ in range(4)]
    e = edge_table(rows)
    u = usage_table([{"pair_count": 50} for _ in range(4)] + [{"pair_count": 0}]
                    + [{"pair_count": 50} for _ in range(4)])
    scored = score_edges(u, e)
    scored.loc[scored["edge_id"] == 4, "score"] = 0.0
    cor = merge_corridors(scored, e, quantile=0.0)
    assert len(cor) == 1 and int(cor["n_pieces"].iloc[0]) == 2
    assert sorted(cor["edge_ids"].iloc[0]) == [0, 1, 2, 3, 5, 6, 7, 8]


def test_a_corridors_climb_per_km_does_not_depend_on_how_it_is_drawn():
    """Four blocks rising 5 m each.  Drawn uphill the forward climb is 50 m
    per kilometre and drawn downhill it is nothing; the corridor's figure is
    the mean of the two directions of travel either way."""
    up = edge_table([{"cum_gain_fwd": 5.0, "cum_loss_fwd": 0.0} for _ in range(4)])
    down = edge_table([{"cum_gain_fwd": 0.0, "cum_loss_fwd": 5.0} for _ in range(4)])
    mixed = edge_table([{"cum_gain_fwd": 5.0 * (k % 2), "cum_loss_fwd": 5.0 * (1 - k % 2)}
                        for k in range(4)])
    for e in (up, down, mixed):
        cor = merge_corridors(score_edges(usage_table([{}] * 4), e), e, quantile=0.0)
        assert cor["gain_per_km"].tolist() == pytest.approx([25.0])
    # an edge table that carries no descent is read as upstream reads it
    old = up.drop(columns=["cum_loss_fwd"])
    cor = merge_corridors(score_edges(usage_table([{}] * 4), old), old, quantile=0.0)
    assert cor["gain_per_km"].tolist() == pytest.approx([50.0])


# --------------------------------------------------------------------------
# closing the gaps between pieces of a corridor
# --------------------------------------------------------------------------
def gapped_streets(n: int, short_m: float = 30.0, long_m: float = 80.0):
    """``n`` places where two high-scoring blocks face each other across a gap.

    At place ``k`` the blocks are ``a1 - a2`` and ``b1 - b2``, and two chains
    of two eligible blocks each could close the gap: ``a1 - x - b1`` and
    ``a2 - y - b2``.  The first is the shorter one at even places and the
    longer one at odd places, so no fixed order of the nodes finds the
    shorter chain first everywhere.  Node ids are strings, as the build's
    are.  Returns ``(core edge ids, eligible edge ids, edges)``; the edges of
    place ``k`` are numbered ``10 k`` (the two blocks: 0, 1; first chain: 2,
    3; second chain: 4, 5).
    """
    rows, core, eligible = [], set(), set()
    for k in range(n):
        first, second = (short_m, long_m) if k % 2 == 0 else (long_m, short_m)
        for j, (u, v, length) in enumerate((
                ("a1", "a2", 100.0), ("b1", "b2", 100.0),
                ("a1", "x", first), ("x", "b1", first),
                ("a2", "y", second), ("y", "b2", second))):
            eid = 10 * k + j
            rows.append({"edge_id": eid, "u": f"{u}-{k}", "v": f"{v}-{k}",
                         "length_m": length})
            (core if j < 2 else eligible).add(eid)
    return core, core | eligible, pd.DataFrame(rows)


def test_a_gap_is_closed_by_the_shortest_chain_whatever_the_order_of_a_set():
    """Upstream adopts the first chain its loop over the core nodes meets.
    The nodes are held in sets and this port's ids are strings, so which
    came first changed from one Python process to the next, and with it the
    corridor table.  Twelve gaps, each with a 60 m chain and a 160 m one:
    the first-met rule gets all twelve right one time in four thousand."""
    from hk_flat_routes.corridors import _bridge_gaps
    n = 12
    core, candidates, edges = gapped_streets(n)
    shorter = {10 * k + j for k in range(n) for j in ((2, 3) if k % 2 == 0 else (4, 5))}
    assert _bridge_gaps(core, candidates, edges) == shorter
    # the edge table in another order, and the sets built the other way round
    again = _bridge_gaps(set(sorted(core, reverse=True)),
                         set(sorted(candidates, reverse=True)),
                         edges.iloc[::-1].reset_index(drop=True))
    assert again == shorter


def test_of_two_chains_of_one_length_the_lower_edge_ids_close_the_gap():
    from hk_flat_routes.corridors import _bridge_gaps
    n = 12
    core, candidates, edges = gapped_streets(n, short_m=50.0, long_m=50.0)
    assert _bridge_gaps(core, candidates, edges) == {
        10 * k + j for k in range(n) for j in (2, 3)}


def test_a_chain_too_long_closes_no_gap():
    """Three blocks at most, and 220 m: a longer way round is not a gap in
    one street."""
    from hk_flat_routes.corridors import _bridge_gaps
    core, candidates, edges = gapped_streets(1, short_m=120.0, long_m=150.0)
    assert _bridge_gaps(core, candidates, edges) == set()
    # one piece has no gap to close
    assert _bridge_gaps({0}, candidates, edges) == set()
