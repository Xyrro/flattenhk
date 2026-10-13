"""The bike comfort multiplier."""
from __future__ import annotations

import pandas as pd

from hk_flat_routes import bikeways


def test_stress_prefers_cycle_tracks_and_penalises_main_roads():
    edges = pd.DataFrame({"cls": ["cycleway", "residential", "tertiary", "secondary",
                                  "primary", "trunk", None, "something_new"]})
    m = bikeways.stress(edges)
    assert m[0] < m[1] < m[2] < m[3] < m[4] < m[5]
    assert m[1] == 1.0                          # the unit: an ordinary block
    assert m[6] == bikeways.CLASS_STRESS["unknown"]
    assert m[7] == bikeways.DEFAULT_STRESS


def test_no_class_is_cheaper_than_the_loop_search_assumes():
    """engine.js keeps its A* exact with a floor of 0.78 on cost per metre."""
    assert min(bikeways.CLASS_STRESS.values()) >= 0.78


def test_a_pushed_link_is_three_times_as_stressful_as_its_class():
    """The "calm" weighting the route page uses in bike mode: walking a
    bicycle along a footway counts as the routing cost counts it."""
    from hk_flat_routes.config import BIKE_PUSH_MULTIPLIER
    edges = pd.DataFrame({"cls": ["footway", "footway", "cycleway"],
                          "bike_push": [True, False, None]})
    m = bikeways.stress(edges)
    assert m[0] == bikeways.CLASS_STRESS["footway"] * BIKE_PUSH_MULTIPLIER
    assert m[1] == bikeways.CLASS_STRESS["footway"]
    assert m[2] == bikeways.CLASS_STRESS["cycleway"]
    # an edge table from before pushed links existed
    assert bikeways.stress(edges.drop(columns=["bike_push"]))[0] == \
        bikeways.CLASS_STRESS["footway"]
    # the class weight alone, as the browser payload ships it
    assert bikeways.stress(edges, push=False).tolist() == [
        bikeways.CLASS_STRESS[c] for c in ("footway", "footway", "cycleway")]
