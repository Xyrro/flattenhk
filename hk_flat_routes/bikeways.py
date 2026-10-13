"""The bike "stress" multiplier the route page uses to prefer calm streets.

Upstream rates each San Francisco block by its bikeway, conflating the
SFMTA's facility layer (class I path to class IV separated lane) onto the
street graph.  Hong Kong publishes nothing comparable: the Transport
Department's cycle track dataset is 211 km of plain lines with an owner and
no facility class, and 97.5% of it lies within 8 m of an edge that
OpenStreetMap already classes as a cycleway.  The territory's cycling
network is also of one kind -- the off-road cycle tracks of the New
Territories new towns -- with almost no painted lanes on the roads.  So the
road class is the facility here: a cycle track rates as upstream's
protected path, and everything else by how much traffic its class carries.

The stress multiplier is in equivalent metres per metre: a block that feels
like 1.4 blocks.  It is a comfort scale, not a speed model, and it only ever
applies to the distance objective in bike mode when the rider asks for calm
streets, so the climbing axis of the frontier is untouched.

A pushed link (``bike_push``: a short piece of footway where the rider
dismounts) counts ``BIKE_PUSH_MULTIPLIER`` times over on top of its class,
the same factor the routing cost gives it wherever the class comfort
multipliers apply.  The page puts the two together itself: the bundle
carries the class weight per edge (``edge_stress``), the pushed arcs
(``arc_push``) and the multiplier, and ``engine.js`` multiplies them into
the comfort-weighted length it searches on.
"""
from __future__ import annotations

import numpy as np

from .config import BIKE_PUSH_MULTIPLIER

#: comfort multiplier by road class.  The smallest value must stay at or
#: above 0.78: the loop search in ``web/engine.js`` uses that as the floor
#: on cost per metre that keeps its A* exact.
CLASS_STRESS = {
    "cycleway": 0.8, "living_street": 0.9, "path": 1.0, "residential": 1.0,
    "unclassified": 1.0, "track": 1.1, "service": 1.1, "unknown": 1.1,
    "pedestrian": 1.2, "tertiary": 1.2, "footway": 1.3, "secondary": 1.4,
    "primary": 1.6, "trunk": 2.0,
}
#: and for a class the table does not know
DEFAULT_STRESS = 1.1


def stress(edges, push: bool = True) -> np.ndarray:
    """Bike comfort multiplier per edge (1.0 = an ordinary residential block).

    A pushed link counts ``BIKE_PUSH_MULTIPLIER`` times over; an edge table
    built before there were any has no ``bike_push`` column.  ``push=False``
    gives the class weight alone, which is what the browser payload ships:
    the engine applies the push factor arc by arc, and a weight that already
    held it would be charged nine times over instead of three.
    """
    cls = edges["cls"].fillna("unknown").to_numpy()
    out = np.array([CLASS_STRESS.get(c, DEFAULT_STRESS) for c in cls])
    if push and "bike_push" in edges.columns:
        pushed = edges["bike_push"].to_numpy(dtype=bool, na_value=False)
        out[pushed] *= BIKE_PUSH_MULTIPLIER
    return out
