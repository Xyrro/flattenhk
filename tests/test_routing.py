"""Tests for the cost model and shortest-path machinery."""
import pandas as pd
import pytest

import numpy as np

from hk_flat_routes.config import (BIKE, BIKE_PUSH_MULTIPLIER, ELEVATION,
                                   GRADE_THRESHOLDS, MIN_COMPONENT_NODES,
                                   MIN_RELIABLE_GRADE_LENGTH_M,
                                   ROUTING_PROFILES, CostWeights, profile,
                                   with_alpha)
from hk_flat_routes.routing import (arc_pitch, build_route_graph,
                                    component_ranks, edge_costs, route,
                                    summarise_route)

_TH = [int(t * 100) for t in GRADE_THRESHOLDS]


def make_directed(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal directed-edge table with all required columns."""
    base = {
        "length_m": 100.0, "cum_gain": 0.0, "cum_loss": 0.0, "net_change": 0.0,
        "avg_grade": 0.0, "max_grade": 0.0, "p95_grade": 0.0,
        "mean_abs_grade": 0.0, "start_elev": 0.0, "end_elev": 0.0,
        "cls": "residential", "subclass": None, "name": "Test Street",
        "is_structure": False, "walk_ok": True, "walk_oneway": False,
        "bike_ok": True, "bike_oneway": False, "bike_facility": "",
        "low_stress": True, "walk_traversable": True, "bike_traversable": True,
        "direction": "fwd",
    }
    for t in _TH:
        base[f"d_above_{t}"] = 0.0
    out = []
    for i, r in enumerate(rows):
        d = dict(base)
        d.update(r)
        d.setdefault("edge_id", i)
        out.append(d)
    return pd.DataFrame(out)


# ------------------------------------------------------------ cost model
def test_shortest_cost_is_pure_distance():
    """The baseline objective must minimise distance only.

    Regression test: the bicycle comfort multipliers used to apply to the
    shortest objective too, which made the "shortest" bike route longer in
    real metres than the flat route and broke every distance comparison.
    """
    d = make_directed([{"length_m": 250.0, "cls": "trunk", "cum_gain": 40.0,
                        "d_above_8": 120.0}])
    c = edge_costs(d, profile("shortest"), mode="bike")
    assert c[0] == pytest.approx(250.0)
    c_walk = edge_costs(d, profile("shortest"), mode="walk")
    assert c_walk[0] == pytest.approx(250.0)


def test_climbing_weight_prices_gain_in_equivalent_metres():
    d = make_directed([{"length_m": 100.0, "cum_gain": 10.0}])
    w = CostWeights(alpha=8.0, beta=0.0, gamma=0.0,
                    threshold_penalties=(0, 0, 0, 0, 0), extreme_extra=0.0)
    assert edge_costs(d, w, "walk")[0] == pytest.approx(100.0 + 8.0 * 10.0)


def test_grade_penalty_is_cumulative_across_thresholds():
    """A 12% grade pays the 3, 5, 8 and 10% penalties simultaneously."""
    w = CostWeights(alpha=0.0, beta=1.0, gamma=0.0,
                    threshold_penalties=(0.25, 0.75, 2.0, 4.0, 10.0),
                    extreme_extra=0.0)
    gentle = make_directed([{"length_m": 100.0, "d_above_3": 100.0}])
    steep = make_directed([{"length_m": 100.0, "d_above_3": 100.0,
                            "d_above_5": 100.0, "d_above_8": 100.0,
                            "d_above_10": 100.0}])
    c_gentle = edge_costs(gentle, w, "walk")[0]
    c_steep = edge_costs(steep, w, "walk")[0]
    assert c_gentle == pytest.approx(100.0 + 25.0)
    assert c_steep == pytest.approx(100.0 + 100.0 * (0.25 + 0.75 + 2.0 + 4.0))
    # the marginal cost of steepness must rise, not stay linear
    assert (c_steep - 100.0) / (c_gentle - 100.0) > 4.0


def test_extreme_grade_incurs_the_gamma_term():
    w = CostWeights(alpha=0.0, beta=0.0, gamma=2.0,
                    threshold_penalties=(0, 0, 0, 0, 0), extreme_extra=10.0)
    d = make_directed([{"length_m": 50.0, f"d_above_{_TH[-1]}": 50.0}])
    assert edge_costs(d, w, "walk")[0] == pytest.approx(50.0 + 2.0 * 10.0 * 50.0)


def test_class_multiplier_applies_only_to_the_named_class():
    d = make_directed([{"cls": "steps"}, {"cls": "residential"}])
    w = profile("balanced")
    c = edge_costs(d, w, "walk")
    assert c[0] > c[1]                       # stairs cost more on foot
    assert c[1] == pytest.approx(100.0)


def test_a_pushed_link_costs_a_bicycle_three_times_its_length():
    """A few metres of footway between a cycle track and the road: the rider
    gets off and pushes.  The factor goes on top of the class multiplier,
    for bicycles only, and under the same switch, so that the shortest
    objective still measures distance."""
    d = make_directed([{"length_m": 40.0, "cls": "footway", "bike_push": True},
                       {"length_m": 40.0, "cls": "footway", "bike_push": False}])
    footway = BIKE.class_multiplier["footway"]
    c = edge_costs(d, profile("min_climb"), "bike")
    assert c[0] == pytest.approx(40.0 * footway * BIKE_PUSH_MULTIPLIER)
    assert c[1] == pytest.approx(40.0 * footway)
    assert edge_costs(d, profile("shortest"), "bike").tolist() == [40.0, 40.0]
    walk = edge_costs(d, profile("min_climb"), "walk")
    assert walk[0] == walk[1]


def test_a_table_without_the_push_column_has_no_pushed_links():
    """The directed table of a build made before pushed links existed."""
    d = make_directed([{"length_m": 40.0, "cls": "footway"}])
    assert "bike_push" not in d.columns
    assert edge_costs(d, profile("min_climb"), "bike")[0] == pytest.approx(
        40.0 * BIKE.class_multiplier["footway"])
    # and a column with gaps in it reads as "not pushed" where it is empty
    d["bike_push"] = pd.Series([None], dtype="object")
    assert edge_costs(d, profile("min_climb"), "bike")[0] == pytest.approx(
        40.0 * BIKE.class_multiplier["footway"])


def test_costs_are_strictly_positive():
    d = make_directed([{"length_m": 0.0}])
    assert edge_costs(d, profile("balanced"), "walk")[0] > 0


def test_directional_costs_differ_for_the_same_street():
    """Uphill and downhill traversals of one edge must not cost the same."""
    up = make_directed([{"cum_gain": 20.0, "cum_loss": 0.0, "d_above_8": 80.0,
                         "direction": "fwd"}])
    down = make_directed([{"cum_gain": 0.0, "cum_loss": 20.0, "d_above_8": 0.0,
                           "direction": "rev"}])
    w = profile("balanced")
    assert edge_costs(up, w, "walk")[0] > edge_costs(down, w, "walk")[0]


def test_min_climb_prefers_climbing_over_distance_more_than_balanced():
    assert ROUTING_PROFILES["min_climb"].alpha > ROUTING_PROFILES["balanced"].alpha
    assert ROUTING_PROFILES["balanced"].alpha > ROUTING_PROFILES["shortest"].alpha


def test_with_alpha_leaves_other_weights_untouched():
    base = profile("balanced")
    w = with_alpha(base, 42.0)
    assert w.alpha == 42.0
    assert w.beta == base.beta
    assert w.threshold_penalties == base.threshold_penalties


# ------------------------------------------------------------ graph + paths
def _diamond():
    """Two routes from A to D: short+steep via B, long+flat via C."""
    rows = []
    def pair(eid, u, v, length, gain, steep):
        rows.append({"edge_id": eid, "from_node": u, "to_node": v,
                     "length_m": length, "cum_gain": gain, "cum_loss": 0.0,
                     "d_above_8": steep, "max_grade": 0.15 if steep else 0.01,
                     "mean_abs_grade": 0.15 if steep else 0.01,
                     "direction": "fwd"})
        rows.append({"edge_id": eid, "from_node": v, "to_node": u,
                     "length_m": length, "cum_gain": 0.0, "cum_loss": gain,
                     "d_above_8": 0.0, "max_grade": 0.01,
                     "mean_abs_grade": 0.15 if steep else 0.01,
                     "direction": "rev"})
    pair(0, "A", "B", 300.0, 45.0, 300.0)
    pair(1, "B", "D", 300.0, 0.0, 0.0)
    pair(2, "A", "C", 600.0, 3.0, 0.0)
    pair(3, "C", "D", 600.0, 0.0, 0.0)
    return make_directed(rows)


def test_shortest_takes_the_short_steep_route():
    g = build_route_graph(_diamond(), "walk")
    arcs, s = route(g, "A", "D", profile("shortest"))
    assert s["distance_m"] == pytest.approx(600.0)
    assert s["elev_gain_m"] == pytest.approx(45.0)


def test_flat_objective_takes_the_long_gentle_route():
    g = build_route_graph(_diamond(), "walk")
    arcs, s = route(g, "A", "D", profile("min_climb"))
    assert s["distance_m"] == pytest.approx(1200.0)
    assert s["elev_gain_m"] == pytest.approx(3.0)


def test_route_summary_sums_edge_metrics():
    g = build_route_graph(_diamond(), "walk")
    arcs, s = route(g, "A", "D", profile("min_climb"))
    t = g.table.iloc[arcs]
    assert s["distance_m"] == pytest.approx(t["length_m"].sum())
    assert s["elev_gain_m"] == pytest.approx(t["cum_gain"].sum())
    assert s["n_edges"] == len(arcs)


def test_one_way_blocks_bicycles_but_not_pedestrians():
    rows = [
        {"edge_id": 0, "from_node": "A", "to_node": "B", "direction": "fwd",
         "bike_oneway": True, "bike_traversable": True},
        {"edge_id": 0, "from_node": "B", "to_node": "A", "direction": "rev",
         "bike_oneway": True, "bike_traversable": False},
        {"edge_id": 1, "from_node": "B", "to_node": "A", "direction": "fwd"},
        {"edge_id": 1, "from_node": "A", "to_node": "B", "direction": "rev"},
    ]
    d = make_directed(rows)
    gb = build_route_graph(d, "bike")
    assert not any((gb.table["edge_id"] == 0) & (gb.table["direction"] == "rev"))
    gw = build_route_graph(d, "walk")
    assert any((gw.table["edge_id"] == 0) & (gw.table["direction"] == "rev"))


def test_parallel_arcs_take_the_cheaper_one_not_their_sum():
    """Two streets between the same intersections must not be summed.

    A COO->CSR conversion adds duplicate entries, which would invent a more
    expensive street than either real one.
    """
    rows = [
        {"edge_id": 0, "from_node": "A", "to_node": "B", "length_m": 100.0,
         "direction": "fwd"},
        {"edge_id": 1, "from_node": "A", "to_node": "B", "length_m": 400.0,
         "direction": "fwd"},
        {"edge_id": 0, "from_node": "B", "to_node": "A", "length_m": 100.0,
         "direction": "rev"},
        {"edge_id": 1, "from_node": "B", "to_node": "A", "length_m": 400.0,
         "direction": "rev"},
    ]
    g = build_route_graph(make_directed(rows), "walk")
    arcs, s = route(g, "A", "B", profile("shortest"))
    assert s["distance_m"] == pytest.approx(100.0)


def test_unreachable_target_raises():
    rows = [{"edge_id": 0, "from_node": "A", "to_node": "B", "direction": "fwd"},
            {"edge_id": 0, "from_node": "B", "to_node": "A", "direction": "rev"}]
    g = build_route_graph(make_directed(rows), "walk")
    with pytest.raises(KeyError):
        route(g, "A", "ZZZ", profile("shortest"))


def test_empty_route_summary_is_all_zero():
    g = build_route_graph(_diamond(), "walk")
    s = summarise_route(g, [])
    assert s["distance_m"] == 0.0 and s["elev_gain_m"] == 0.0
    assert s["path_distance_m"] == 0.0 and s["path_gain_m"] == 0.0


# ------------------------------------------------------- the steepest pitch
def _street(pieces) -> pd.DataFrame:
    """A street of pieces laid end to end, node ``n0`` to ``n1`` and so on.

    The arcs in the direction given come first, in order, then the same
    pieces walked back down.
    """
    there = [{"edge_id": i, "from_node": f"n{i}", "to_node": f"n{i + 1}",
              "direction": "fwd", **p} for i, p in enumerate(pieces)]
    back = [{"edge_id": i, "from_node": f"n{i + 1}", "to_node": f"n{i}",
             "direction": "rev", **p, "max_grade": 0.0, "cum_gain": 0.0,
             "cum_loss": p.get("cum_gain", 0.0)}
            for i, p in enumerate(pieces)]
    return make_directed(there + back)


def test_a_short_arc_is_judged_on_its_average_gradient():
    """Two in five of Hong Kong's arcs are shorter than 15 m, and a sampled
    maximum over a few metres is terrain-model noise: a 5 m stub that reads
    50% on a street that climbs 8% must not be the route's steepest pitch."""
    short = MIN_RELIABLE_GRADE_LENGTH_M - 10.0
    d = _street([
        {"length_m": 100.0, "max_grade": 0.08, "cum_gain": 6.0},
        {"length_m": short, "max_grade": 0.50, "cum_gain": 0.1},
        {"length_m": 100.0, "max_grade": 0.03, "cum_gain": 1.0},
    ])
    assert arc_pitch(d)[:3].tolist() == pytest.approx([0.08, 0.1 / short, 0.03])
    g = build_route_graph(d, "bike")
    arcs, s = route(g, "n0", "n3", profile("shortest"))
    assert len(arcs) == 3
    assert s["max_grade"] == pytest.approx(0.08)
    # the unfiltered figure is still there for whoever wants it
    assert s["max_abs_grade_any"] == pytest.approx(0.50)


def test_an_arc_of_exactly_the_reliable_length_keeps_its_maximum():
    d = _street([{"length_m": MIN_RELIABLE_GRADE_LENGTH_M, "max_grade": 0.31,
                  "cum_gain": 1.5}])
    assert arc_pitch(d)[0] == pytest.approx(0.31)


def test_a_short_arcs_average_is_held_to_the_plausibility_ceiling():
    """2.5 m of rise on a 2.3 m stub of a ladder street is a step in the
    data, not a 110% slope; every sampled gradient is held to the ceiling,
    and so is this one.  Downhill the same stub is no climb at all."""
    cap = ELEVATION.max_plausible_grade
    stub = _street([{"length_m": 2.3, "max_grade": cap, "cum_gain": 2.52}])
    assert arc_pitch(stub).tolist() == pytest.approx([cap, -cap])
    # a short arc that dips and comes back up is judged on its net climb
    dip = _street([{"length_m": 10.0, "max_grade": 0.4, "cum_gain": 1.5,
                    "cum_loss": 1.0}])
    assert arc_pitch(dip)[0] == pytest.approx(0.05)
    # an arc of no length at all divides by nothing
    assert np.isfinite(arc_pitch(_street([{"length_m": 0.0}]))).all()


def test_a_route_that_only_descends_has_no_steepest_pitch():
    """Arc by arc the pitch is signed, and a descent's is below zero.  A
    route's steepest pitch is the steepest it *climbs*: nil on the way
    down, as the route page shows it, and never a negative gradient."""
    d = _street([{"length_m": 100.0, "max_grade": 0.08, "cum_gain": 6.0},
                 {"length_m": 5.0, "max_grade": 0.30, "cum_gain": 0.5}])
    back = d[d["direction"] == "rev"].assign(max_grade=[-0.02, -0.01])
    d = pd.concat([d[d["direction"] == "fwd"], back], ignore_index=True)
    assert arc_pitch(d).tolist() == pytest.approx([0.08, 0.1, -0.02, -0.1])
    g = build_route_graph(d, "walk")
    up = route(g, "n0", "n2", profile("shortest"))[1]
    down = route(g, "n2", "n0", profile("shortest"))[1]
    assert up["max_grade"] == pytest.approx(0.1)
    assert down["max_grade"] == 0.0
    assert down["max_abs_grade_any"] == pytest.approx(0.02)


def test_route_summary_counts_what_is_on_hill_paths_and_steps():
    """A shortest route in Hong Kong does much of its climbing off the
    street; the summary says how much, so the report can."""
    d = _street([
        {"length_m": 300.0, "cls": "residential", "cum_gain": 5.0},
        {"length_m": 200.0, "cls": "path", "cum_gain": 60.0},
        {"length_m": 50.0, "cls": "steps", "cum_gain": 20.0},
        {"length_m": 100.0, "cls": "track", "cum_gain": 4.0},
        {"length_m": 80.0, "cls": "footway", "cum_gain": 2.0},
    ])
    g = build_route_graph(d, "walk")
    arcs = list(range(5))
    s = summarise_route(g, arcs)
    assert s["path_distance_m"] == pytest.approx(350.0)
    assert s["path_gain_m"] == pytest.approx(84.0)
    assert s["distance_m"] == pytest.approx(730.0)


# ------------------------------------------------------------- landmasses
# Hong Kong is not one graph on foot or by bike: nothing walkable crosses
# Victoria Harbour, and Lantau, Lamma and Cheung Chau stand apart too.  The
# router keeps every strongly connected component of a useful size, with
# whatever joins two of them one way (further down), ranks the landmasses
# these make by size, and routes only within one.
def _ring(prefix: str, n: int, first_edge: int, **extra) -> list[dict]:
    """A two-way ring of ``n`` nodes named ``prefix0`` .. ``prefix{n-1}``."""
    rows = []
    for k in range(n):
        u, v = f"{prefix}{k}", f"{prefix}{(k + 1) % n}"
        rows.append({"edge_id": first_edge + k, "from_node": u, "to_node": v,
                     "direction": "fwd", **extra})
        rows.append({"edge_id": first_edge + k, "from_node": v, "to_node": u,
                     "direction": "rev", **extra})
    return rows


#: Sizes for the synthetic landmasses: two worth keeping, one fragment.
_BIG, _SMALL, _SCRAP = MIN_COMPONENT_NODES + 12, MIN_COMPONENT_NODES, 5


def _archipelago() -> pd.DataFrame:
    """An "island" ring, a larger "mainland" ring and a stray walkway."""
    rows = (_ring("K", _BIG, 0) + _ring("H", _SMALL, 1000)
            + _ring("X", _SCRAP, 2000))
    return make_directed(rows)


def test_every_landmass_of_a_useful_size_is_kept():
    """Upstream kept the largest component only, which here would delete
    Hong Kong Island (or Kowloon) from the router without a word."""
    g = build_route_graph(_archipelago(), "walk")
    assert g.n_nodes == _BIG + _SMALL
    kept = {str(n)[0] for n in g.node_ids}
    assert kept == {"K", "H"}


def test_landmasses_are_ranked_largest_first():
    g = build_route_graph(_archipelago(), "walk")
    rank = {str(n)[0]: int(g.node_comp[i]) for i, n in enumerate(g.node_ids)}
    assert rank == {"K": 0, "H": 1}
    assert len(g.node_comp) == g.n_nodes
    assert np.bincount(g.node_comp).tolist() == [_BIG, _SMALL]


def test_a_small_fragment_is_dropped():
    """A podium walkway nobody joined to the street is not a place to route
    from: its nodes are gone, and asking for one is an error, not a
    silent snap to something else."""
    g = build_route_graph(_archipelago(), "walk")
    assert g.index_of("X0") is None
    assert not g.table["from_node"].astype(str).str.startswith("X").any()
    with pytest.raises(KeyError):
        route(g, "K0", "X0", profile("shortest"))


def test_connected_is_true_within_a_landmass_and_false_across():
    g = build_route_graph(_archipelago(), "walk")
    k0, k9, h0 = g.index_of("K0"), g.index_of("K9"), g.index_of("H0")
    assert g.connected(k0, k9)
    assert g.connected(h0, h0)
    assert not g.connected(k0, h0)
    assert not g.connected(h0, k9)


def test_a_route_across_the_water_is_empty_not_an_error():
    """Both ends exist, on different landmasses: there is no route, and the
    summary says zero rather than raising."""
    g = build_route_graph(_archipelago(), "walk")
    arcs, s = route(g, "K0", "H0", profile("shortest"))
    assert arcs == []
    assert s["distance_m"] == 0.0 and s["n_edges"] == 0


def test_a_route_within_the_smaller_landmass_is_found():
    """The second landmass is routable in its own right, not merely kept."""
    g = build_route_graph(_archipelago(), "walk")
    half = _SMALL // 2
    arcs, s = route(g, "H0", f"H{half}", profile("shortest"))
    assert s["distance_m"] == pytest.approx(100.0 * half)
    assert all(str(n).startswith("H") for n in g.table.iloc[arcs]["from_node"])


def test_the_largest_landmass_is_kept_however_small():
    """A four-node test network must still route."""
    g = build_route_graph(_diamond(), "walk")
    assert g.n_nodes == 4
    assert set(g.node_comp.tolist()) == {0}


def test_the_component_floor_is_a_parameter():
    g = build_route_graph(_archipelago(), "walk", min_component_nodes=_SCRAP)
    assert g.n_nodes == _BIG + _SMALL + _SCRAP
    assert sorted(set(g.node_comp.tolist())) == [0, 1, 2]


def test_component_ranks_orders_by_size_and_marks_fragments():
    # nodes 0-5 in a two-way path, 6-8 in another, 9 alone
    tail = np.array([0, 1, 2, 3, 4, 6, 7] + [1, 2, 3, 4, 5, 7, 8])
    head = np.array([1, 2, 3, 4, 5, 7, 8] + [0, 1, 2, 3, 4, 6, 7])
    rank, core = component_ranks(tail, head, 10, min_nodes=3)
    assert rank[:6].tolist() == [0] * 6
    assert rank[6:9].tolist() == [1] * 3
    assert rank[9] == -1
    # every node of a landmass that is one component is in its core
    assert core.tolist() == [True] * 9 + [False]
    # raise the floor and the second one goes too; the first never does
    rank, core = component_ranks(tail, head, 10, min_nodes=50)
    assert rank[:6].tolist() == [0] * 6 and (rank[6:] == -1).all()
    assert core[:6].all() and not core[6:].any()
    # a cap on the number of landmasses drops the smallest first
    rank, _ = component_ranks(tail, head, 10, min_nodes=1, max_components=2)
    assert rank[:6].tolist() == [0] * 6 and rank[6:9].tolist() == [1] * 3
    assert rank[9] == -1


def test_a_one_way_dead_end_is_on_no_landmass():
    """A one-way street out to a headland is a place a cyclist cannot come
    back from, and it leads to nothing kept, so it is not part of the
    bicycle's landmass; a walker, who ignores one-way rules, keeps it."""
    rows = _ring("K", _BIG, 0)
    spur = _BIG + 1
    for k in range(_SCRAP):                       # K0 -> S0 -> S1 ... one way
        u = "K0" if k == 0 else f"S{k - 1}"
        rows.append({"edge_id": spur + k, "from_node": u, "to_node": f"S{k}",
                     "direction": "fwd", "bike_oneway": True})
        rows.append({"edge_id": spur + k, "from_node": f"S{k}", "to_node": u,
                     "direction": "rev", "bike_oneway": True,
                     "bike_traversable": False})
    d = make_directed(rows)
    walk = build_route_graph(d, "walk")
    bike = build_route_graph(d, "bike")
    assert walk.index_of(f"S{_SCRAP - 1}") is not None
    assert bike.index_of("S0") is None
    assert bike.n_nodes == _BIG and walk.n_nodes == _BIG + _SCRAP


def test_with_scale_zero_is_pure_distance():
    from hk_flat_routes.config import with_scale
    w = with_scale(profile("balanced"), 0.0)
    assert w.alpha == 0 and w.beta == 0 and w.gamma == 0
    assert w.use_class_multiplier is False
    d = make_directed([{"length_m": 250.0, "cls": "steps", "cum_gain": 40.0,
                        "d_above_8": 120.0}])
    assert edge_costs(d, w, "walk")[0] == pytest.approx(250.0)
    # ... and therefore coincides with the shortest objective exactly
    assert edge_costs(d, w, "walk")[0] == edge_costs(d, profile("shortest"), "walk")[0]


def test_with_scale_one_is_the_profile_itself():
    from hk_flat_routes.config import with_scale
    base = profile("balanced")
    w = with_scale(base, 1.0)
    assert (w.alpha, w.beta, w.gamma) == (base.alpha, base.beta, base.gamma)
    assert w.use_class_multiplier == base.use_class_multiplier


def test_pareto_sweep_starts_at_zero_and_is_increasing():
    from hk_flat_routes.config import PARETO_LAMBDA_SWEEP
    assert PARETO_LAMBDA_SWEEP[0] == 0.0
    assert list(PARETO_LAMBDA_SWEEP) == sorted(PARETO_LAMBDA_SWEEP)


# --------------------------------------------- a district attached one way
# By bicycle Wan Chai North can be ridden into from the rest of Hong Kong
# Island and not out of: the one-way streets that leave it end where the
# data has a gap.  Each is strongly connected in itself.  Taken as two
# landmasses, the router refused the ride in that its own graph holds; they
# are one landmass, with a route one way.
_DISTRICT = MIN_COMPONENT_NODES + 3


def _one_way(rows: list, eid: int, u: str, v: str) -> None:
    """A street bicycles may ride from ``u`` to ``v`` only."""
    rows.append({"edge_id": eid, "from_node": u, "to_node": v,
                 "direction": "fwd", "bike_oneway": True})
    rows.append({"edge_id": eid, "from_node": v, "to_node": u,
                 "direction": "rev", "bike_oneway": True,
                 "bike_traversable": False})


def _attached_one_way() -> pd.DataFrame:
    """A main ring K, a district ring W, and K0 -> T0 -> T1 -> W0 one way.

    Also a dead end off the link (T0 -> D0) and a stray ring X that is
    joined to nothing.
    """
    rows = _ring("K", _BIG, 0) + _ring("W", _DISTRICT, 1000) + _ring("X", _SCRAP, 2000)
    _one_way(rows, 3000, "K0", "T0")
    _one_way(rows, 3001, "T0", "T1")
    _one_way(rows, 3002, "T1", "W0")
    _one_way(rows, 3003, "T0", "D0")
    return make_directed(rows)


def test_a_district_attached_one_way_is_part_of_the_landmass():
    g = build_route_graph(_attached_one_way(), "bike")
    k, w, t = g.index_of("K5"), g.index_of("W5"), g.index_of("T1")
    assert None not in (k, w, t)
    # one label for the two rings and for the nodes of the link between them
    assert set(g.node_comp.tolist()) == {0}
    assert g.n_nodes == _BIG + _DISTRICT + 2
    assert g.connected(k, w) and g.connected(w, k) and g.connected(k, t)
    # the dead end off the link and the stray ring are on no landmass
    assert g.index_of("D0") is None and g.index_of("X0") is None
    # the three arcs of the link are kept, in the one direction they have
    link = g.table[g.table["edge_id"].isin([3000, 3001, 3002])]
    assert sorted(link["edge_id"]) == [3000, 3001, 3002]
    assert set(link["direction"]) == {"fwd"}


def test_within_such_a_landmass_a_route_can_exist_one_way_only():
    g = build_route_graph(_attached_one_way(), "bike")
    arcs, s = route(g, "K5", "W5", profile("shortest"))
    assert arcs and s["distance_m"] == pytest.approx(100.0 * (5 + 3 + 5))
    assert [str(n)[0] for n in g.table.iloc[arcs]["from_node"]].count("T") == 2
    # and back there is none: an empty route, not an error
    arcs, s = route(g, "W5", "K5", profile("shortest"))
    assert arcs == [] and s["distance_m"] == 0.0


def test_the_core_of_a_landmass_is_its_largest_strongly_connected_part():
    """Access points are taken in the core, where every node reaches every
    other; the district and the link to it are outside it."""
    g = build_route_graph(_attached_one_way(), "bike")
    core = {str(n) for n, c in zip(g.node_ids, g.node_core) if c}
    assert core == {f"K{i}" for i in range(_BIG)}
    # a walker ignores the one-way rule, so the lot is one component: all core
    walk = build_route_graph(_attached_one_way(), "walk")
    assert walk.node_core.all()
    assert walk.index_of("D0") is not None and walk.index_of("X0") is None


def test_a_landmass_of_several_parts_is_logged(caplog):
    with caplog.at_level("WARNING", logger="hk_flat_routes.routing"):
        build_route_graph(_attached_one_way(), "bike")
    said = [r.getMessage() for r in caplog.records]
    assert len(said) == 1 and said[0].startswith("mode bike: landmass 0 is 2 ")
    assert f"({_BIG}, {_DISTRICT} nodes)" in said[0] and "through 2 other nodes;" in said[0]
    # two parts need not have a route even one way (both may lead into a third)
    assert said[0].endswith("a route exists in one direction at most")
    caplog.clear()
    with caplog.at_level("WARNING", logger="hk_flat_routes.routing"):
        build_route_graph(_attached_one_way(), "walk")
        build_route_graph(_archipelago(), "bike")
    assert not caplog.records


def test_two_districts_that_lead_into_a_third_are_one_landmass_with_no_route_between_them(caplog):
    """K -> W <- M through one node each: W can be reached from both rings
    and neither from the other.  One landmass by the rule, whose core is
    its largest part; between K and M the search comes back empty both ways."""
    middle = _BIG + 2
    rows = _ring("K", _BIG, 0) + _ring("W", _DISTRICT, 1000) + _ring("M", middle, 5000)
    for eid, (u, v) in enumerate((("K0", "S"), ("S", "W0"), ("M0", "T"), ("T", "W3")), 3000):
        _one_way(rows, eid, u, v)
    with caplog.at_level("WARNING", logger="hk_flat_routes.routing"):
        g = build_route_graph(make_directed(rows), "bike")
    assert set(g.node_comp.tolist()) == {0} and g.n_nodes == _BIG + _DISTRICT + middle + 2
    assert {str(n)[0] for n, c in zip(g.node_ids, g.node_core) if c} == {"M"}
    assert g.connected(g.index_of("K5"), g.index_of("M5"))
    for a, b in (("K5", "M5"), ("M5", "K5"), ("W5", "K5")):
        assert route(g, a, b, profile("shortest"))[0] == []
    assert route(g, "M5", "W5", profile("shortest"))[0]
    said = [r.getMessage() for r in caplog.records]
    assert len(said) == 1 and "is 3 strongly connected parts" in said[0]
    assert f"({middle}, {_BIG}, {_DISTRICT} nodes)" in said[0]
    assert "through 2 other nodes;" in said[0]
    # one node on the way is one node
    rows = _ring("K", _BIG, 0) + _ring("W", _DISTRICT, 1000)
    _one_way(rows, 3000, "K0", "S")
    _one_way(rows, 3001, "S", "W0")
    caplog.clear()
    with caplog.at_level("WARNING", logger="hk_flat_routes.routing"):
        build_route_graph(make_directed(rows), "bike")
    assert "through 1 other node;" in caplog.records[0].getMessage()


def test_landmasses_are_ranked_by_all_their_nodes():
    """The ring M is larger than K and smaller than K with its district:
    the landmass is what is ranked, not its largest part."""
    middle = _BIG + 2
    d = pd.concat([_attached_one_way(),
                   make_directed(_ring("M", middle, 5000))], ignore_index=True)
    g = build_route_graph(d, "bike")
    rank = {p: {int(g.node_comp[i]) for i, n in enumerate(g.node_ids)
                if str(n).startswith(p)} for p in "KWTM"}
    assert rank == {"K": {0}, "W": {0}, "T": {0}, "M": {1}}
    assert np.bincount(g.node_comp).tolist() == [_BIG + _DISTRICT + 2, middle]


def test_two_landmasses_that_only_drain_to_the_same_dead_end_stay_apart():
    """K0 -> S <- H0: something can be reached from both, but neither from
    the other.  A weak connection is not a landmass."""
    rows = _ring("K", _BIG, 0) + _ring("H", _SMALL, 1000)
    _one_way(rows, 3000, "K0", "S")
    _one_way(rows, 3001, "H0", "S")
    g = build_route_graph(make_directed(rows), "bike")
    assert g.index_of("S") is None
    assert not g.connected(g.index_of("K0"), g.index_of("H0"))
    assert np.bincount(g.node_comp).tolist() == [_BIG, _SMALL]


def test_equal_landmasses_are_ordered_by_the_arc_list_not_the_numbering():
    """The Python router and the browser payload number the nodes
    differently and pass the same arcs in the same order; two landmasses of
    one size must get the same ranks from both."""
    n = 6
    ring = np.arange(n)
    tail = np.concatenate([ring, (ring + 1) % n, n + ring, n + (ring + 1) % n])
    head = np.concatenate([(ring + 1) % n, ring, n + (ring + 1) % n, n + ring])
    first, _ = component_ranks(tail, head, 2 * n, min_nodes=3)
    assert first[:n].tolist() == [0] * n and first[n:].tolist() == [1] * n
    # the same arcs in the same order, the two rings' numbers exchanged
    swap = np.concatenate([n + ring, ring])
    second, _ = component_ranks(swap[tail], swap[head], 2 * n, min_nodes=3)
    assert second[swap[:n]].tolist() == [0] * n
    assert second[swap[n:]].tolist() == [1] * n
