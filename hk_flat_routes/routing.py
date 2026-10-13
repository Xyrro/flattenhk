"""Routing over the directed, elevation-attributed street graph.

Cost model
----------
The cost of a directed edge, in **equivalent metres** (the distance a
traveller would consider "as bad as" this edge), is

    cost = length * mode_multiplier(class)
         + alpha * cumulative_gain
         + beta  * SUM_k  penalty_k * distance_above_threshold_k
         + gamma * extreme_extra * distance_above_highest_threshold

The threshold terms are *cumulative*: 100 m at a 12% grade incurs the 3%, 5%,
8% and 10% penalties simultaneously, so the marginal cost of extra steepness
rises with grade rather than staying linear.  With the default balanced
weights, a metre travelled at 12% costs about 8 equivalent metres of grade
penalty on top of the climbing term, while a metre at 4% costs 0.25.

``alpha`` is the substitution rate between climbing and distance.  Naismith's
rule for walking implies roughly 8 m of flat walking per metre climbed; the
"min_climb" profile pushes it to 120, a strong aversion to climbing and not
a minimum of it.  The sweep in ``PARETO_LAMBDA_SWEEP`` scales every climbing
and gradient term of the balanced profile from nothing to fifteen times its
size, where a metre of climb costs 210 m, so it runs past "min_climb": on
the full build it holds a route with more than a metre less climbing than
the "min_climb" route for 277 of the 1,354 walking pairs.

The per-class comfort multipliers (a cycle track at 0.85, a trunk road
at 1.9) express stress rather than distance, so they are switched **off** for
the ``shortest`` objective, which minimises pure distance.  That keeps
``shortest`` a meaningful baseline: every distance-penalty and
elevation-saved figure in the results matrix is measured against a genuine
shortest path.

A bicycle's *pushed link* (``bike_push``: a few metres of footway between a
cycle track and the road, where the rider gets off and walks) is charged
``BIKE_PUSH_MULTIPLIER`` times over, on top of its class multiplier and
under the same switch, so the ``shortest`` objective still measures it at
its length.

Only ``cumulative_gain`` enters the cost, never ``net_change``: a route that
climbs 120 m and descends 120 m is not flat, and the analytical point of the
project is that net elevation difference is the wrong variable.

Descent is not free in reality (steep descents are hard on knees and brakes),
but it is charged only through the grade-threshold terms, which are computed
on the *climbing* grade in the direction of travel, plus the class
multipliers.  This keeps the cost function monotone in climbing, which the
Pareto analysis relies on.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import (BIKE_PUSH_MULTIPLIER, CostWeights, ELEVATION,
                     GRADE_THRESHOLDS, MAX_COMPONENTS, MIN_COMPONENT_NODES,
                     MIN_RELIABLE_GRADE_LENGTH_M, MODES)
from .utils import get_logger

log = get_logger("hk_flat_routes.routing")

_TH = [int(t * 100) for t in GRADE_THRESHOLDS]


# --------------------------------------------------------------------------
# cost computation
# --------------------------------------------------------------------------
def edge_costs(directed: pd.DataFrame, weights: CostWeights,
               mode: str = "walk") -> np.ndarray:
    """Vectorised traversal cost, in equivalent metres, for every directed edge."""
    cfg = MODES[mode]
    length = directed["length_m"].to_numpy(dtype="float64")

    mult = np.ones_like(length)
    if cfg.class_multiplier and getattr(weights, "use_class_multiplier", True):
        cls = directed["cls"].to_numpy()
        for klass, m in cfg.class_multiplier.items():
            mult[cls == klass] = m
        # a table built before pushed links existed has no such column
        if mode == "bike" and "bike_push" in directed.columns:
            push = directed["bike_push"].to_numpy(dtype=bool, na_value=False)
            mult[push] *= BIKE_PUSH_MULTIPLIER

    cost = length * mult
    cost = cost + weights.alpha * directed["cum_gain"].to_numpy(dtype="float64")

    if weights.beta:
        pen = np.zeros_like(length)
        for th, w in zip(_TH, weights.threshold_penalties):
            if w:
                pen += w * directed[f"d_above_{th}"].to_numpy(dtype="float64")
        cost = cost + weights.beta * pen

    if weights.gamma and weights.extreme_extra:
        cost = cost + (weights.gamma * weights.extreme_extra
                       * directed[f"d_above_{_TH[-1]}"].to_numpy(dtype="float64"))

    return np.maximum(cost, 1e-6)


# --------------------------------------------------------------------------
# graph container
# --------------------------------------------------------------------------
@dataclass
class RouteGraph:
    """Node/arc structure for one travel mode, backed by ``scipy.csgraph``.

    Shortest paths run through ``scipy.sparse.csgraph.dijkstra``, which needs
    a node-by-node sparse matrix.  Two nodes can be joined by more than one
    arc (two different streets between the same pair of intersections), and a
    COO->CSR conversion would *sum* such duplicates, silently inventing a
    more expensive street.  Arcs are therefore grouped by ordered node pair,
    and for each cost vector the cheapest arc in each group is selected, so
    the matrix holds a true minimum and the winning arc is recoverable.
    """
    mode: str
    node_ids: np.ndarray                  # index -> node id
    node_index: dict                      # node id -> index
    table: pd.DataFrame                   # directed arcs, positional order
    arc_pair: np.ndarray                  # arc -> pair index
    pair_from: np.ndarray                 # pair -> tail node index
    pair_to: np.ndarray                   # pair -> head node index
    pair_lookup: dict                     # (tail, head) -> pair index
    #: rank of each node's landmass (0 = the largest); see ``component_ranks``
    node_comp: np.ndarray | None = None
    #: whether each node is in the core of its landmass: its largest strongly
    #: connected part, within which every node reaches every other
    node_core: np.ndarray | None = None

    @property
    def n_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def n_arcs(self) -> int:
        return len(self.table)

    def build_costs(self, weights: CostWeights) -> np.ndarray:
        """Cost per arc, in equivalent metres."""
        return edge_costs(self.table, weights, self.mode)

    def pair_costs(self, arc_cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Collapse arc costs to one cost per node pair, keeping the winner."""
        n_pairs = len(self.pair_from)
        best = np.full(n_pairs, np.inf)
        np.minimum.at(best, self.arc_pair, arc_cost)
        # recover which arc achieved the minimum for each pair
        winner = np.full(n_pairs, -1, dtype=np.int64)
        is_min = arc_cost <= best[self.arc_pair] + 0.0
        idx = np.flatnonzero(is_min)
        winner[self.arc_pair[idx]] = idx
        return best, winner

    def matrix(self, arc_cost: np.ndarray):
        """CSR cost matrix plus the arc chosen for each retained pair."""
        import scipy.sparse as sp
        best, winner = self.pair_costs(arc_cost)
        m = sp.csr_matrix((best, (self.pair_from, self.pair_to)),
                          shape=(self.n_nodes, self.n_nodes))
        return m, winner

    def index_of(self, node_id):
        return self.node_index.get(node_id)

    def connected(self, a: int, b: int) -> bool:
        """Are node indices ``a`` and ``b`` on the same landmass?

        Nothing is routed between two landmasses.  Within one a route nearly
        always exists both ways, but not by definition: a landmass can hold
        a district that one-way streets lead into and not out of (see
        ``component_ranks``), and then a search still comes back empty in
        one direction.  Callers treat an empty route as "no route".
        """
        return self.node_comp is None or self.node_comp[a] == self.node_comp[b]


def component_ranks(tail: np.ndarray, head: np.ndarray, n_nodes: int,
                    min_nodes: int = MIN_COMPONENT_NODES,
                    max_components: int = MAX_COMPONENTS,
                    label: str = "") -> tuple[np.ndarray, np.ndarray]:
    """Label every node with the landmass it is on.

    The building block is the strongly connected component, within which
    every node can reach every other; one of at least ``min_nodes`` nodes is
    *kept* (the largest always is, however small).  A **landmass** is the
    set of kept components that directed paths join, in either direction,
    together with the nodes those paths pass through on the way.

    For a walker, who ignores one-way rules, that is the kept component
    itself.  For a bicycle it is not, and taking the component alone for the
    landmass cut a district off.  On the full build Wan Chai North (Hung
    Hing Road, Harbour Road, Lung Wo Road, Expo Drive: 371 nodes and 15.8 km
    of street) can be ridden into from the rest of Hong Kong Island through
    eight nodes and not out of, the one-way streets that leave it ending
    where their continuation is missing from the edge table.  As a
    component of its own it was an "island", and the 2.0 km ride from
    Central to Expo Drive that the graph holds was refused.  A dead-end
    spur, which leads from a kept component to none, still belongs to no
    landmass.

    Returns ``(rank, core)``, one value per node.  ``rank`` is 0 for the
    landmass with the most nodes, 1 for the next, and so on, and -1 for a
    node on none (or on one beyond ``max_components``).  ``core`` marks the
    nodes of the largest kept component of each landmass.  Two core nodes
    of one landmass have a route both ways; any two nodes of one landmass
    may have a route one way only (or none, where two districts each lead
    into a third and not to each other), and two nodes of different
    landmasses have none.  A landmass made of more than one kept component
    is logged as a warning, ``label`` saying whose graph it is.

    Shared by the Python router and the browser payload, so that both keep
    exactly the same landmasses.  Equal sizes are ordered by where each
    first appears in the arc list, which the two callers pass in the same
    order while numbering the nodes differently.
    """
    import scipy.sparse as sp
    from scipy.sparse.csgraph import breadth_first_order, connected_components

    tail = np.asarray(tail, dtype=np.int64)
    head = np.asarray(head, dtype=np.int64)
    ones = np.ones(len(tail), dtype=np.int8)
    adj = sp.coo_matrix((ones, (tail, head)), shape=(n_nodes, n_nodes)).tocsr()
    n_scc, scc = connected_components(adj, directed=True, connection="strong")
    size = np.bincount(scc, minlength=n_scc)
    first = np.full(n_scc, len(tail), dtype=np.int64)
    np.minimum.at(first, scc[tail], np.arange(len(tail)))

    kept = size >= min_nodes
    if not kept.any():
        kept[np.lexsort((first, -size))[0]] = True
    seeds = np.flatnonzero(kept)

    # The condensation: one node per component, an arc wherever a street
    # leads from one component into another.  A component lies on a path
    # between two kept ones exactly when some kept component reaches it and
    # it reaches some kept component; the graph has no cycles, so the two
    # are then different.  One search each way finds them all, started from
    # an extra node joined to every kept component.
    ct, ch = scc[tail], scc[head]
    cross = ct != ch
    ct, ch = ct[cross], ch[cross]

    def reached(src, dst):
        a = np.concatenate([src, np.full(len(seeds), n_scc)])
        b = np.concatenate([dst, seeds])
        g = sp.coo_matrix((np.ones(len(a), dtype=np.int8), (a, b)),
                          shape=(n_scc + 1, n_scc + 1)).tocsr()
        hit = np.zeros(n_scc + 1, dtype=bool)
        hit[breadth_first_order(g, n_scc, directed=True,
                                return_predecessors=False)] = True
        return hit[:n_scc]

    member = reached(ct, ch) & reached(ch, ct)

    # landmasses: the pieces the member components fall into once the
    # direction of the arcs between them is forgotten
    ok = member[ct] & member[ch]
    link = sp.coo_matrix((np.ones(int(ok.sum()), dtype=np.int8),
                          (ct[ok], ch[ok])), shape=(n_scc, n_scc))
    n_land, land = connected_components(link, directed=False)
    inside = np.flatnonzero(member)
    land_size = np.bincount(land[inside], weights=size[inside],
                            minlength=n_land).astype(np.int64)
    land_first = np.full(n_land, len(tail), dtype=np.int64)
    np.minimum.at(land_first, land[inside], first[inside])
    # largest first; a piece holding no member component has size zero
    order = np.lexsort((np.arange(n_land), land_first, -land_size))
    order = order[land_size[order] > 0][:max_components]
    rank_of_land = np.full(n_land, -1, dtype=np.int64)
    rank_of_land[order] = np.arange(len(order))
    rank = np.where(member, rank_of_land[land], -1)[scc]

    # the core of a landmass is its largest kept component
    by_size = seeds[np.lexsort((first[seeds], -size[seeds]))]
    core_of_land = np.full(n_land, -1, dtype=np.int64)
    core_of_land[land[by_size][::-1]] = by_size[::-1]
    core = (core_of_land[land[scc]] == scc) & (rank >= 0)

    pieces = np.bincount(land[seeds], minlength=n_land)
    for lm in order[pieces[order] > 1]:
        parts = by_size[land[by_size] == lm]
        between = int(land_size[lm] - size[parts].sum())
        log.warning(
            "%slandmass %d is %d strongly connected parts (%s nodes) joined "
            "one way through %d other node%s; between two of the parts a "
            "route exists in one direction at most", label, rank_of_land[lm],
            len(parts), ", ".join(f"{int(v):,}" for v in size[parts]),
            between, "" if between == 1 else "s")
    return rank, core


def build_route_graph(directed: pd.DataFrame, mode: str = "walk",
                      min_component_nodes: int = MIN_COMPONENT_NODES) -> RouteGraph:
    """Assemble the routable graph for a mode.

    The graph is the union of the mode's landmasses (Hong Kong Island,
    Kowloon with the New Territories, Lantau, the larger outlying islands
    ...): its strongly connected components of at least
    ``min_component_nodes`` nodes and the one-way links between them, as
    ``component_ranks`` defines them.  Arcs that lead out of a landmass and
    never come back, or lie in a fragment too small to keep, are dropped.
    Between landmasses nothing can be routed; within one every origin
    reaches every target, except into or out of a part that is joined to
    the rest in one direction only, which the build log names.
    """
    col = f"{mode}_traversable"
    sub = directed[directed[col]].reset_index(drop=True)
    if sub.empty:
        raise RuntimeError(f"no traversable directed edges for mode {mode!r}")

    def _index(frame):
        nodes = pd.unique(pd.concat([frame["from_node"], frame["to_node"]],
                                    ignore_index=True))
        return nodes, {n: i for i, n in enumerate(nodes)}

    nodes, node_index = _index(sub)
    fi = sub["from_node"].map(node_index).to_numpy(dtype=np.int64)
    ti = sub["to_node"].map(node_index).to_numpy(dtype=np.int64)
    comp, core = component_ranks(fi, ti, len(nodes),
                                 min_nodes=min_component_nodes,
                                 label=f"mode {mode}: ")
    # both ends on a landmass means both on the same one: a landmass takes
    # in every arc between its nodes
    mask = (comp[fi] >= 0) & (comp[ti] >= 0)
    if not mask.all() or (comp < 0).any():
        sizes = np.bincount(comp[comp >= 0])
        log.info("mode %s: %d landmass(es) kept (largest %s nodes); "
                 "%d/%d nodes and %d/%d arcs", mode, sizes.size,
                 ", ".join(f"{int(v):,}" for v in sizes[:6]),
                 int((comp >= 0).sum()), len(nodes), int(mask.sum()), len(fi))
        kept_comp = dict(zip(nodes[comp >= 0], comp[comp >= 0]))
        kept_core = dict(zip(nodes[comp >= 0], core[comp >= 0]))
        sub = sub[mask].reset_index(drop=True)
        nodes, node_index = _index(sub)
        fi = sub["from_node"].map(node_index).to_numpy(dtype=np.int64)
        ti = sub["to_node"].map(node_index).to_numpy(dtype=np.int64)
        node_comp = np.asarray([kept_comp[n] for n in nodes], dtype=np.int64)
        node_core = np.asarray([kept_core[n] for n in nodes], dtype=bool)
    else:
        node_comp, node_core = comp, core

    # group arcs by ordered node pair
    key = fi.astype(np.int64) * len(nodes) + ti.astype(np.int64)
    uniq, arc_pair = np.unique(key, return_inverse=True)
    pair_from = (uniq // len(nodes)).astype(np.int64)
    pair_to = (uniq % len(nodes)).astype(np.int64)
    pair_lookup = {(int(a), int(b)): i for i, (a, b) in
                   enumerate(zip(pair_from, pair_to))}

    g = RouteGraph(mode=mode, node_ids=np.asarray(nodes), node_index=node_index,
                   table=sub, arc_pair=arc_pair.astype(np.int64),
                   pair_from=pair_from, pair_to=pair_to,
                   pair_lookup=pair_lookup, node_comp=node_comp,
                   node_core=node_core)
    n_parallel = len(arc_pair) - len(uniq)
    log.info("mode %s graph: %d nodes, %d arcs (%d parallel), %.0f km",
             mode, g.n_nodes, g.n_arcs, n_parallel,
             sub["length_m"].sum() / 1000)
    return g


# --------------------------------------------------------------------------
# shortest paths
# --------------------------------------------------------------------------
def shortest_paths(graph: RouteGraph, arc_cost: np.ndarray, sources,
                   targets=None):
    """Multi-source Dijkstra. Returns (dist, predecessors, winner_arc).

    ``dist`` and ``predecessors`` are (n_sources, n_nodes) as returned by
    scipy; ``winner_arc`` maps a pair index to the arc that realises it.
    """
    from scipy.sparse.csgraph import dijkstra as sp_dijkstra

    m, winner = graph.matrix(arc_cost)
    src = np.atleast_1d(np.asarray(sources, dtype=np.int64))
    dist, pred = sp_dijkstra(m, directed=True, indices=src,
                             return_predecessors=True)
    if dist.ndim == 1:
        dist = dist[None, :]
        pred = pred[None, :]
    return dist, pred, winner


def arcs_from_predecessors(graph: RouteGraph, pred_row: np.ndarray,
                           winner: np.ndarray, source: int,
                           target: int) -> list[int]:
    """Rebuild the arc sequence of one path from a scipy predecessor row."""
    if source == target:
        return []
    path_nodes = [int(target)]
    v = int(target)
    guard = 0
    while v != source:
        p = int(pred_row[v])
        if p < 0:
            return []                       # unreachable
        path_nodes.append(p)
        v = p
        guard += 1
        if guard > graph.n_nodes + 5:       # pathological safety net
            return []
    path_nodes.reverse()

    arcs: list[int] = []
    for a, b in zip(path_nodes[:-1], path_nodes[1:]):
        pi = graph.pair_lookup.get((a, b))
        if pi is None or winner[pi] < 0:
            return []
        arcs.append(int(winner[pi]))
    return arcs


def route(graph: RouteGraph, source_node, target_node, weights: CostWeights,
          arc_cost: np.ndarray | None = None):
    """Route between two node ids under one cost profile. Returns (arcs, summary)."""
    si = graph.index_of(source_node)
    ti = graph.index_of(target_node)
    if si is None or ti is None:
        raise KeyError("source or target node is not in a routable component")
    c = arc_cost if arc_cost is not None else graph.build_costs(weights)
    dist, pred, winner = shortest_paths(graph, c, si)
    arcs = arcs_from_predecessors(graph, pred[0], winner, si, ti)
    return arcs, summarise_route(graph, arcs)


# --------------------------------------------------------------------------
# route summarisation
# --------------------------------------------------------------------------
#: Ways that are not streets: hill paths, tracks and flights of steps.  A
#: route's length and climb on them are reported beside its totals, because
#: a distance-minimising walker in Hong Kong goes straight over the hills on
#: them.
PATH_CLASSES = ("path", "track", "steps")


def arc_pitch(table: pd.DataFrame) -> np.ndarray:
    """The steepest climbing gradient each arc can be said to have.

    For an arc of at least ``MIN_RELIABLE_GRADE_LENGTH_M`` that is its
    ``max_grade``, the steepest sampled interval in the direction of travel.
    A shorter arc cannot support one (a decimetre of terrain-model noise
    over a few metres reads as a wall), so it is given its average gradient
    instead: net climb over length, held to the plausibility ceiling that
    every sampled gradient is held to, since two metres of rise on a
    two-metre stub is a step in the data and not a slope.

    The figure is signed: an arc that only descends has a pitch below
    zero.  The route page works out the same one from the same quantities
    (``Graph.summarise`` in ``web/engine.js``, which takes the length and
    the ceiling from the bundle's ``meta``).  They reach it rounded to 5 cm
    and 1 cm, so on a short arc the two agree to that rounding and no
    closer.
    """
    length = table["length_m"].to_numpy(dtype="float64")
    cap = ELEVATION.max_plausible_grade
    net = (table["cum_gain"].to_numpy(dtype="float64")
           - table["cum_loss"].to_numpy(dtype="float64"))
    mean = np.clip(net / np.maximum(length, 1e-6), -cap, cap)
    return np.where(length >= MIN_RELIABLE_GRADE_LENGTH_M,
                    table["max_grade"].to_numpy(dtype="float64"), mean)


def summarise_route(graph: RouteGraph, arcs: list[int]) -> dict:
    """Aggregate metrics for a route given as a list of arc indices.

    ``max_grade`` is the route's steepest pitch: the largest ``arc_pitch``
    along it, and zero for a route that climbs nowhere, which is how the
    route page counts it too.  It used to be the largest ``max_grade`` of
    any arc, and on the full build 41% of the walking arcs are shorter than
    15 m and 40% of the shortest routes had their steepest pitch on one.
    ``max_abs_grade_any`` keeps that figure, in absolute value.
    ``path_distance_m`` and ``path_gain_m`` are the length and the climb on
    ways of ``PATH_CLASSES``.
    """
    if not arcs:
        return {"n_edges": 0, "distance_m": 0.0, "elev_gain_m": 0.0,
                "elev_loss_m": 0.0, "max_grade": 0.0, "net_change_m": 0.0,
                "avg_abs_grade": 0.0, "path_distance_m": 0.0,
                "path_gain_m": 0.0,
                **{f"d_above_{t}": 0.0 for t in _TH}}
    t = graph.table.iloc[arcs]
    length = t["length_m"].to_numpy(dtype="float64")
    gain = t["cum_gain"].to_numpy(dtype="float64")
    total = float(length.sum())
    mean_abs = t["mean_abs_grade"].to_numpy(dtype="float64")
    on_path = t["cls"].isin(PATH_CLASSES).to_numpy()
    out = {
        "n_edges": len(arcs),
        "distance_m": total,
        "elev_gain_m": float(gain.sum()),
        "elev_loss_m": float(t["cum_loss"].sum()),
        "net_change_m": float(t["net_change"].sum()),
        "max_grade": max(float(arc_pitch(t).max()), 0.0),
        "max_abs_grade_any": float(t["max_grade"].abs().max()),
        "avg_abs_grade": float((mean_abs * length).sum() / total) if total else 0.0,
        "p95_grade": float(np.percentile(t["p95_grade"].to_numpy(), 95))
        if len(t) else 0.0,
        "start_elev_m": float(t["start_elev"].iloc[0]),
        "end_elev_m": float(t["end_elev"].iloc[-1]),
        "path_distance_m": float(length[on_path].sum()),
        "path_gain_m": float(gain[on_path].sum()),
    }
    for th in _TH:
        out[f"d_above_{th}"] = float(t[f"d_above_{th}"].sum())
    return out


def route_geometry(graph: RouteGraph, arcs: list[int], edges_gdf):
    """Merged LineString for a route, oriented in travel order."""
    from shapely.geometry import LineString

    if not arcs:
        return None
    geom_by_id = edges_gdf.set_index("edge_id").geometry
    coords: list[tuple] = []
    t = graph.table.iloc[arcs]
    for eid, direction in zip(t["edge_id"], t["direction"]):
        g = geom_by_id.loc[eid]
        cs = list(g.coords)
        if direction == "rev":
            cs = cs[::-1]
        if coords and coords[-1] == cs[0]:
            coords.extend(cs[1:])
        else:
            coords.extend(cs)
    if len(coords) < 2:
        return None
    return LineString(coords)


def route_profile(graph: RouteGraph, arcs: list[int], profiles: dict):
    """Concatenated (cumulative_distance, elevation) profile for a route."""
    if not arcs:
        return np.array([]), np.array([])
    t = graph.table.iloc[arcs]
    dists: list[np.ndarray] = []
    elevs: list[np.ndarray] = []
    base = 0.0
    for eid, direction in zip(t["edge_id"], t["direction"]):
        d, z = profiles[int(eid)]
        d = np.asarray(d, dtype="float64"); z = np.asarray(z, dtype="float64")
        if direction == "rev":
            d = d[-1] - d[::-1]
            z = z[::-1]
        if dists:
            d = d[1:]; z = z[1:]
        dists.append(base + d)
        elevs.append(z)
        base = dists[-1][-1] if dists[-1].size else base
    return np.concatenate(dists), np.concatenate(elevs)


