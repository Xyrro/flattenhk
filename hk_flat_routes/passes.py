"""Passes, saddles and barriers: where Hong Kong's geography forces a climb.

Method
------
The key question -- "if I want to get from this part of the city to that one,
what is the *lowest* summit I can possibly cross?" -- is a **minimax (bottleneck)
path** problem, not a shortest-path problem.  For two nodes *s* and *t* the
answer is

    pass_height(s, t) = min over all paths P from s to t of max elevation on P

and the edge realising that maximum is the pass itself.  This has an exact,
elegant solution: sort every edge by its crest (the highest elevation reached
anywhere along it), add edges to a union-find structure in increasing crest
order, and the crest of the edge that first connects *s* to *t* is
``pass_height(s, t)``.  The resulting structure is a minimum bottleneck
spanning tree, and its critical edges are precisely the city's passes.

Hong Kong is several landmasses with no walking link between them, so the
structure is a *forest*: one tree for Hong Kong Island, one for Kowloon and
the New Territories, one for Lantau, one for each outlying island.  Two
places on different trees have no pass between them at all, and such pairs
are left out of every output rather than reported with a height of zero or
infinity (``pass_height`` returns ``None`` for them).

Rather than asking about arbitrary node pairs, the analysis first identifies
the **lowland basins**: connected components of the street network lying
entirely below a low-elevation threshold.  These are the flat districts that
people actually travel between -- in the urban core, the Kowloon peninsula,
the reclaimed north shore of Hong Kong Island, and the separate floors of
Chai Wan, Aberdeen and Tseung Kwan O behind their headlands.  The pass tree
over those basins then answers, for every pair of flat districts, how much
climbing is geometrically unavoidable and exactly which block you must climb
it on.

Barriers are the complementary view: steep edges that carry a large share of
inter-neighborhood traffic because no flatter alternative exists.  A street
with a high grade *and* high usage is a genuine wall in the city's
topography; a steep street nobody routes over is merely steep.

Heights are metres above the Hong Kong Principal Datum (mPD), the datum of
the terrain model, which lies about 1.2 m below mean sea level.
"""
from __future__ import annotations

import collections
import json

import numpy as np
import pandas as pd

from .config import OUTPUT_DIR
from .utils import get_logger, step

log = get_logger("hk_flat_routes.passes")

PASSES_GEOJSON = OUTPUT_DIR / "passes.geojson"
PASSES_CSV = OUTPUT_DIR / "passes.csv"
BARRIERS_GEOJSON = OUTPUT_DIR / "barriers.geojson"
BARRIERS_CSV = OUTPUT_DIR / "barriers.csv"
BASINS_GEOJSON = OUTPUT_DIR / "lowland_basins.geojson"
#: One row per basin (label, neighborhoods, street length): the summary the
#: written analysis reads, so that it need not open the basin geometry.
BASINS_CSV = OUTPUT_DIR / "lowland_basins.csv"
PASS_MATRIX_CSV = OUTPUT_DIR / "pass_matrix.csv"
BASIN_PASSES_CSV = OUTPUT_DIR / "basin_passes.csv"

#: Elevation (mPD) below which street is considered "lowland" when
#: delineating basins.  San Francisco uses 15 m; the value was derived again
#: for Hong Kong on the urban-core run (Hong Kong Island and Kowloon, 4,154 km
#: of walkable street) and came out the same, for reasons of its own.
#: Hong Kong's flat ground, most of it reclaimed, lies within a metre or two
#: of one level: the median ground height of the Kai Tak, South West
#: Kowloon, Wan Chai, Mong Kok and Yuen Long planning areas is between 4.3
#: and 5.4 mPD.  So 20% of the run's street lies below 5 m and 32% below
#: 6 m, each of those two metres adding 500-600 km, and a threshold in that
#: range cuts through the flats themselves: at 5 m Kowloon falls into five
#: basins, the largest 244 km, along rises of a few decimetres, and at 6 m
#: it is one basin of 704 km.  From 8 m to 20 m the picture stops moving.
#: Each further metre adds only 30-80 km, and the basins of more than 20 km
#: are the same six throughout (the Kowloon peninsula, 971-1,274 km; Hong
#: Kong Island's north shore, 389-475 km; Tseung Kwan O; Chai Wan; Aberdeen;
#: Tsing Yi), each divided from the next by ground a walker really has to
#: climb: 87 m between the north shore and Chai Wan, 142 m between Kowloon
#: and Tseung Kwan O.  By 25 m the Aberdeen basin has started up the valleys
#: behind it (112 km, against 50 km at 15 m).  Within the stable range the
#: choice is settled by the one settled plain that is *not* at that level:
#: the ground of the Fanling / Sheung Shui planning area has a median height
#: of 12 mPD, and 34% of it lies below 10 m, 68% below 15 m and 84% below
#: 20 m.  15 m takes in most of that plain and stays clear of the slopes.
BASIN_ELEV_M = 15.0
#: A basin must contain at least this much street to count as a district.
BASIN_MIN_KM = 2.0

#: Columns of the pass and barrier layers.  Both files are always written,
#: with no rows if need be (a run confined to one reclaimed flat has no
#: passes), and a GeoJSON file with no features carries no column names; the
#: readers below put these back so that callers can rely on them.
PASS_COLUMNS = ("edge_id", "pairs_served", "pass_elev_m",
                "neighborhoods_separated", "name", "name_zh", "cls",
                "length_m", "max_abs_grade", "elev_min", "elev_max",
                "neighborhood", "nearest_neighborhood", "nearest_distance_m")
BARRIER_COLUMNS = ("edge_id", "name", "name_zh", "cls", "length_m",
                   "max_abs_grade", "shortest_use", "flat_use",
                   "flat_use_per_objective", "unavoidability",
                   "barrier_score", "neighborhood", "nearest_neighborhood",
                   "nearest_distance_m")
BASIN_COLUMNS = ("edge_id", "basin", "name", "length_m", "basin_label",
                 "basin_km")
PASS_MATRIX_COLUMNS = ("neighborhood_a", "neighborhood_b", "pass_elev_m",
                       "pass_edge_id")
#: Added to the pass matrix by ``low_route_lengths``: how far the lowest way
#: round is.
LOW_ROUTE_COLUMN = "low_route_m"


# --------------------------------------------------------------------------
# node elevations
# --------------------------------------------------------------------------
def node_elevations(directed: pd.DataFrame) -> pd.Series:
    """Elevation per graph node, from the structure-corrected edge profiles."""
    a = directed[["from_node", "start_elev"]].rename(
        columns={"from_node": "node", "start_elev": "z"})
    b = directed[["to_node", "end_elev"]].rename(
        columns={"to_node": "node", "end_elev": "z"})
    both = pd.concat([a, b], ignore_index=True)
    return both.groupby("node")["z"].median()


# --------------------------------------------------------------------------
# union-find
# --------------------------------------------------------------------------
class _DSU:
    __slots__ = ("parent", "rank")

    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int64)
        self.rank = np.zeros(n, dtype=np.int8)

    def find(self, x: int) -> int:
        p = self.parent
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return int(x)

    def union(self, a: int, b: int) -> tuple[int, int] | None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return None
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return ra, rb


# --------------------------------------------------------------------------
# where a pass or a barrier is
# --------------------------------------------------------------------------
def locate(geoms, neighborhoods, crs) -> pd.DataFrame:
    """The neighborhood each line's midpoint is in, and the nearest one.

    San Francisco's neighborhoods tile the city, so every street is in one.
    Hong Kong's planning areas do not: most of the countryside belongs to no plan,
    and the ridges the passes cross are mostly inside them.  ``neighborhood``
    is therefore empty for a good share of passes and barriers, and
    ``nearest_neighborhood`` says which planning area the place is closest
    to (the same one wherever ``neighborhood`` is set), with
    ``nearest_distance_m`` the distance to its edge (0 inside it).  The
    distance matters: the nearest planning area to a street on Lamma is
    across three kilometres of sea.

    Returns a frame with those three columns, in the order of ``geoms``.
    """
    import geopandas as gpd

    geoms = gpd.GeoSeries(geoms, crs=crs).reset_index(drop=True)
    n = len(geoms)
    out = pd.DataFrame({
        "neighborhood": pd.Series([None] * n, dtype="object"),
        "nearest_neighborhood": pd.Series([None] * n, dtype="object"),
        "nearest_distance_m": np.full(n, np.nan)})
    if not n or neighborhoods is None or not len(neighborhoods):
        return out
    mid = geoms.interpolate(0.5, normalized=True)
    pts = gpd.GeoDataFrame(geometry=mid.values, crs=crs)
    nb = neighborhoods[["neighborhood", "geometry"]]
    if nb.crs != pts.crs:
        nb = nb.to_crs(pts.crs)
    # two plans can overlap by a sliver, and a point in the sliver comes back
    # twice; keep one row per point so the columns stay aligned
    inside = gpd.sjoin(pts, nb, how="left", predicate="within")
    inside = inside[~inside.index.duplicated()]
    held = inside["neighborhood"].reindex(pts.index)
    out["neighborhood"] = held.to_numpy()
    out["nearest_neighborhood"] = held.to_numpy()
    out.loc[held.notna().to_numpy(), "nearest_distance_m"] = 0.0
    loose = np.flatnonzero(held.isna().to_numpy())
    if len(loose):
        near = gpd.sjoin_nearest(pts.iloc[loose], nb, how="left",
                                 distance_col="_d")
        near = near[~near.index.duplicated()].reindex(pts.index[loose])
        out.loc[loose, "nearest_neighborhood"] = near["neighborhood"].to_numpy()
        out.loc[loose, "nearest_distance_m"] = near["_d"].to_numpy()
    return out


#: Up to this distance (m) from a planning area a place is described as
#: "near" it; beyond, the distance is given.
NEAR_M = 1000.0


def place_label(neighborhood, nearest, distance_m) -> str:
    """How a place is named in a table: 'X', 'near X' or '3 km from X'."""
    if isinstance(neighborhood, str) and neighborhood:
        return neighborhood
    if not (isinstance(nearest, str) and nearest):
        return "outside the planning areas"
    if distance_m is None or not np.isfinite(distance_m) or distance_m <= NEAR_M:
        return f"near {nearest}"
    return f"{distance_m / 1000:.0f} km from {nearest}"


# --------------------------------------------------------------------------
# basins
# --------------------------------------------------------------------------
def find_basins(edges, node_z: pd.Series, threshold: float = BASIN_ELEV_M,
                min_km: float = BASIN_MIN_KM, mode: str = "walk"):
    """Connected lowland components of the network below ``threshold``."""
    import networkx as nx

    sub = edges[edges[f"{mode}_ok"]]
    zu = sub["u"].map(node_z)
    zv = sub["v"].map(node_z)
    low = sub[(zu <= threshold) & (zv <= threshold)]
    g = nx.Graph()
    for u, v, L in zip(low["u"], low["v"], low["length_m"]):
        if g.has_edge(u, v):
            g[u][v]["length"] = min(g[u][v]["length"], L)
        else:
            g.add_edge(u, v, length=L)

    basins = {}
    labels: dict[str, int] = {}
    bid = 0
    for comp in nx.connected_components(g):
        km = sum(d["length"] for _, _, d in g.subgraph(comp).edges(data=True)) / 1000
        if km < min_km:
            continue
        for n in comp:
            labels[n] = bid
        basins[bid] = {"n_nodes": len(comp), "length_km": km}
        bid += 1
    log.info("lowland basins below %.0f m: %d districts (>= %.1f km of street)",
             threshold, len(basins), min_km)
    return labels, basins


def name_basins(labels: dict, edges, neighborhoods, node_z: pd.Series):
    """Label each basin with the neighborhoods it covers, by street length.

    A basin with no street inside any planning area (a container port, an
    island that no plan in the set covers) is labelled by the planning area
    it lies nearest to, as "near ..." or, further off, "3 km from ...".
    """
    sub = edges[edges["u"].map(labels).notna() | edges["v"].map(labels).notna()].copy()
    sub["basin"] = sub["u"].map(labels)
    sub = sub[sub["basin"].notna()]
    where = locate(sub.geometry.values, neighborhoods, edges.crs)
    j = pd.DataFrame({"basin": sub["basin"].astype(int).to_numpy(),
                      "length_m": sub["length_m"].to_numpy(),
                      "neighborhood": where["neighborhood"].to_numpy(),
                      "nearest": where["nearest_neighborhood"].to_numpy(),
                      "distance": where["nearest_distance_m"].to_numpy()})
    out = {}
    for b, grp in j.groupby("basin"):
        top = (grp.groupby("neighborhood")["length_m"].sum()
               .sort_values(ascending=False))
        if len(top):
            label = " / ".join(top.head(3).index.tolist())
        else:
            near = (grp.groupby("nearest")["length_m"].sum()
                    .sort_values(ascending=False))
            label = place_label(None, near.index[0], float(
                grp.loc[grp["nearest"] == near.index[0], "distance"].median())
            ) if len(near) else ""
        out[int(b)] = {
            "label": label,
            "neighborhoods": "; ".join(sorted(top[top > 200].index.tolist())),
            "street_km": float(grp["length_m"].sum() / 1000),
        }
    return out


# --------------------------------------------------------------------------
# pass tree
# --------------------------------------------------------------------------
def find_passes(edges, node_z: pd.Series, basin_labels: dict,
                mode: str = "walk"):
    """Minimum bottleneck spanning tree over basins -> the city's passes.

    Returns one record per pass: the crest elevation at which two groups of
    basins first become connected, and the edge on which that happens.
    Basins on different landmasses never become connected and yield no
    record.
    """
    sub = edges[edges[f"{mode}_ok"]].copy()
    # crest of an edge = highest elevation reached anywhere along it
    sub["crest"] = sub[["elev_max"]].max(axis=1)
    sub = sub[np.isfinite(sub["crest"])].sort_values("crest", kind="stable")

    nodes = pd.unique(pd.concat([sub["u"], sub["v"]], ignore_index=True))
    nidx = {n: i for i, n in enumerate(nodes)}
    dsu = _DSU(len(nodes))
    # which basins each component currently contains
    comp_basins: dict[int, set] = collections.defaultdict(set)
    for n, b in basin_labels.items():
        i = nidx.get(n)
        if i is not None:
            comp_basins[i].add(b)

    passes = []
    u_idx = sub["u"].map(nidx).to_numpy()
    v_idx = sub["v"].map(nidx).to_numpy()
    crest = sub["crest"].to_numpy(dtype="float64")
    eid = sub["edge_id"].to_numpy()

    with step(f"building the minimum bottleneck spanning tree [{mode}]", log):
        for k in range(len(sub)):
            a, b = int(u_idx[k]), int(v_idx[k])
            ra, rb = dsu.find(a), dsu.find(b)
            if ra == rb:
                continue
            sa, sb = comp_basins.get(ra, set()), comp_basins.get(rb, set())
            merged = dsu.union(a, b)
            if merged is None:
                continue
            root, child = merged
            new = sa | sb
            comp_basins[root] = new
            comp_basins.pop(child, None)
            # a pass exists only where two *different* basin groups join
            if sa and sb and not (sa & sb):
                passes.append({
                    "edge_id": int(eid[k]),
                    "pass_elev_m": float(crest[k]),
                    "basins_a": sorted(sa),
                    "basins_b": sorted(sb),
                    "n_basins_joined": len(sa) + len(sb),
                })
    log.info("  %d passes found linking the lowland districts", len(passes))
    return pd.DataFrame(passes)


def _edge_columns(e) -> list[str]:
    """Edge attributes carried onto a pass record, where the table has them."""
    return [c for c in ["name", "name_zh", "cls", "length_m", "max_abs_grade",
                        "elev_min", "elev_max", "geometry"] if c in e.columns]


def describe_passes(passes: pd.DataFrame, edges, neighborhoods,
                    basin_names: dict):
    """Attach geometry, street name, neighborhood and basin labels."""
    import geopandas as gpd

    if passes.empty:
        return gpd.GeoDataFrame(columns=["edge_id", "geometry"],
                                geometry="geometry", crs=edges.crs)
    e = edges.set_index("edge_id")
    df = passes.join(e[_edge_columns(e)], on="edge_id")

    def label(bs):
        return " + ".join(basin_names.get(int(b), {}).get("label", f"basin {b}")
                          for b in bs)

    df["side_a"] = df["basins_a"].apply(label)
    df["side_b"] = df["basins_b"].apply(label)
    df["basins_a"] = df["basins_a"].apply(lambda x: ",".join(map(str, x)))
    df["basins_b"] = df["basins_b"].apply(lambda x: ",".join(map(str, x)))

    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=edges.crs)
    where = locate(gdf.geometry.values, neighborhoods, edges.crs)
    gdf["neighborhood"] = where["neighborhood"].to_numpy()
    gdf["nearest_neighborhood"] = where["nearest_neighborhood"].to_numpy()
    gdf["nearest_distance_m"] = where["nearest_distance_m"].to_numpy()
    return gdf.sort_values("pass_elev_m").reset_index(drop=True)




# --------------------------------------------------------------------------
# minimum bottleneck spanning tree + LCA: pass height for any pair of points
# --------------------------------------------------------------------------
def build_bottleneck_tree(edges, mode: str = "walk"):
    """Kruskal merge tree over crest-sorted edges (a Cartesian/merge tree).

    Each union creates an internal node labelled with the crest elevation at
    which the two sides became connected, and with the edge responsible.  The
    minimax pass height between any two graph nodes is then the label of their
    lowest common ancestor in this tree, and the pass itself is that
    ancestor's edge.  This is exact, and one construction answers every pair.

    On a network of several landmasses the result is one tree per landmass,
    held in the same arrays; nodes of different trees have no common
    ancestor.
    """
    sub = edges[edges[f"{mode}_ok"]].copy()
    sub["crest"] = sub["elev_max"]
    # an edge with no usable elevation would sort last and join whatever is
    # left with a crest of NaN; it is not a crossing anyone can be told about
    sub = sub[np.isfinite(sub["crest"])].sort_values("crest", kind="stable")

    nodes = pd.unique(pd.concat([sub["u"], sub["v"]], ignore_index=True))
    nidx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)

    # merge-tree arrays: leaves 0..n-1 are graph nodes, internals follow
    parent_tree = np.full(2 * n, -1, dtype=np.int64)
    label = np.full(2 * n, -np.inf)
    tree_edge = np.full(2 * n, -1, dtype=np.int64)
    comp_root = np.arange(n, dtype=np.int64)      # DSU root -> tree node
    dsu = _DSU(n)
    next_node = n

    u_idx = sub["u"].map(nidx).to_numpy()
    v_idx = sub["v"].map(nidx).to_numpy()
    crest = sub["crest"].to_numpy(dtype="float64")
    eid = sub["edge_id"].to_numpy()

    with step(f"building the minimum bottleneck spanning tree [{mode}]", log):
        for k in range(len(sub)):
            a, b = int(u_idx[k]), int(v_idx[k])
            ra, rb = dsu.find(a), dsu.find(b)
            if ra == rb:
                continue
            ta, tb = int(comp_root[ra]), int(comp_root[rb])
            t = next_node; next_node += 1
            parent_tree[ta] = t
            parent_tree[tb] = t
            label[t] = float(crest[k])
            tree_edge[t] = int(eid[k])
            merged = dsu.union(a, b)
            root, child = merged
            comp_root[root] = t
    log.info("  merge tree: %d leaves, %d internal nodes, %d separate trees",
             n, next_node - n, 2 * n - next_node)
    return {"nidx": nidx, "parent": parent_tree[:next_node],
            "label": label[:next_node], "edge": tree_edge[:next_node],
            "n_leaves": n}


def _ancestors(tree, leaf: int) -> dict[int, int]:
    """Map ancestor -> depth for one leaf (root last)."""
    out = {}
    v = leaf
    d = 0
    par = tree["parent"]
    while v != -1:
        out[v] = d
        v = int(par[v])
        d += 1
    return out


def pass_height(tree, node_a, node_b):
    """Minimax pass elevation and the responsible edge for one pair of nodes.

    ``None`` when either node is not on the network, when the two are the
    same node (there is nothing to cross), or when they are on different
    landmasses and so have no common ancestor.
    """
    ia = tree["nidx"].get(node_a)
    ib = tree["nidx"].get(node_b)
    if ia is None or ib is None or ia == ib:
        return None
    anc = _ancestors(tree, ia)
    v = ib
    par = tree["parent"]
    while v != -1:
        if v in anc:
            return float(tree["label"][v]), int(tree["edge"][v])
        v = int(par[v])
    return None


def neighborhood_pass_matrix(tree, points: pd.DataFrame):
    """Minimax pass elevation for every unordered pair of neighborhood points.

    This is the direct answer to "how much climbing is unavoidable between
    these two parts of the city, and where".

    Only pairs that can be walked between get a row.  A neighborhood with
    no access point (``node`` empty: no street network inside it in this
    build) is left out, and so is every pair whose points lie on different
    landmasses -- by the points' ``component`` column where there is one,
    and in any case by the tree, in which the two have no common ancestor.
    Nothing is ever reported as zero or infinite.
    """
    names = list(points["neighborhood"])
    nodes = list(points["node"])
    comps = (list(points["component"]) if "component" in points.columns
             else [None] * len(names))
    rows = []
    skipped = 0
    with step(f"minimax pass heights for {len(names)} neighborhoods", log):
        for i in range(len(names)):
            if pd.isna(nodes[i]):
                continue
            for j in range(i + 1, len(names)):
                if pd.isna(nodes[j]):
                    continue
                if (comps[i] is not None and comps[j] is not None
                        and not pd.isna(comps[i]) and not pd.isna(comps[j])
                        and comps[i] != comps[j]):
                    skipped += 1
                    continue
                r = pass_height(tree, nodes[i], nodes[j])
                if r is None or not np.isfinite(r[0]):
                    skipped += 1
                    continue
                h, e = r
                rows.append({"neighborhood_a": names[i], "neighborhood_b": names[j],
                             "pass_elev_m": h, "pass_edge_id": e})
    if skipped:
        log.info("  %d pairs lie on different landmasses and have no pass",
                 skipped)
    return pd.DataFrame(rows, columns=list(PASS_MATRIX_COLUMNS))


def low_route_lengths(pass_matrix: pd.DataFrame, edges, points: pd.DataFrame,
                      mode: str = "walk") -> pd.Series:
    """Length (m) of each pair's shortest route that keeps to its pass height.

    A pass height says how high a traveller must go and nothing about how
    far the way that stays so low is.  It can be very far: on the full
    build the 133 pairs divided by Castle Peak Road at Ting Kau (32 mPD)
    have a lowest route of 40 km at the median where their shortest is
    23 km, and from Kowloon Tong to Sha Tin it is 74 km against 7 km.  The
    length is that of the shortest path over the edges whose crest is no
    higher than the pair's pass, which by the minimax property join the two
    access points.

    One search per pass, from the side of it with fewer neighborhoods
    (every pair a pass binds has one end on each side).  Returns a series
    aligned with ``pass_matrix``, NaN where an access point is missing.
    """
    import scipy.sparse as sp
    from scipy.sparse.csgraph import dijkstra

    out = pd.Series(np.nan, index=pass_matrix.index, name=LOW_ROUTE_COLUMN)
    if pass_matrix.empty:
        return out
    sub = edges[edges[f"{mode}_ok"] & np.isfinite(edges["elev_max"])]
    # shortest first, so that of two edges between the same pair of nodes
    # the first met is the one to keep (a sparse matrix would add them up)
    sub = sub.sort_values("length_m", kind="stable")
    n = len(sub)
    codes, nodes = pd.factorize(pd.concat([sub["u"], sub["v"]], ignore_index=True))
    a, b = np.minimum(codes[:n], codes[n:]), np.maximum(codes[:n], codes[n:])
    key = a.astype(np.int64) * len(nodes) + b
    length = sub["length_m"].to_numpy(dtype="float64")
    crest = sub["elev_max"].to_numpy(dtype="float64")
    index = {node: i for i, node in enumerate(nodes)}
    at = {name: index[node] for name, node in
          zip(points["neighborhood"], points["node"]) if node in index}

    with step(f"lengths of the lowest routes for {len(pass_matrix)} pairs", log):
        for _edge, grp in pass_matrix.groupby("pass_edge_id"):
            grp = grp[grp["neighborhood_a"].isin(at) & grp["neighborhood_b"].isin(at)]
            if grp.empty:
                continue
            low = np.flatnonzero(crest <= float(grp["pass_elev_m"].iloc[0]))
            _, first = np.unique(key[low], return_index=True)
            low = low[first]
            m = sp.coo_matrix((length[low], (a[low], b[low])),
                              shape=(len(nodes), len(nodes))).tocsr()
            # the two sides of the pass: whatever is paired with one
            # neighborhood, and the rest
            one = grp["neighborhood_a"].iloc[0]
            far = set(grp.loc[grp["neighborhood_a"] == one, "neighborhood_b"]) \
                | set(grp.loc[grp["neighborhood_b"] == one, "neighborhood_a"])
            near = (set(grp["neighborhood_a"]) | set(grp["neighborhood_b"])) - far
            src = sorted(min(near, far, key=len))
            dist = dijkstra(m, directed=False, indices=[at[s] for s in src])
            row_of = {name: i for i, name in enumerate(src)}
            for i, x, y in zip(grp.index, grp["neighborhood_a"],
                               grp["neighborhood_b"]):
                s, t = (x, y) if x in row_of else (y, x)
                d = dist[row_of[s], at[t]]
                out.loc[i] = float(d) if np.isfinite(d) else np.nan
    return out


def rank_critical_passes(pass_matrix: pd.DataFrame, edges, neighborhoods):
    """Aggregate the pass matrix onto edges: the city's true saddles.

    An edge that is the binding constraint for many neighborhood pairs is a
    pass that Hong Kong's topography genuinely forces traffic over.
    """
    import geopandas as gpd

    if pass_matrix.empty:
        return gpd.GeoDataFrame(columns=["edge_id", "geometry"],
                                geometry="geometry", crs=edges.crs)
    agg = (pass_matrix.groupby("pass_edge_id")
           .agg(pairs_served=("pass_elev_m", "size"),
                pass_elev_m=("pass_elev_m", "first"))
           .reset_index().rename(columns={"pass_edge_id": "edge_id"}))
    # which neighborhoods sit on each side
    sides = (pass_matrix.groupby("pass_edge_id")
             .apply(lambda g: "; ".join(sorted(set(g["neighborhood_a"])
                                               | set(g["neighborhood_b"]))),
                    include_groups=False)
             .rename("neighborhoods_separated"))
    agg = agg.merge(sides, left_on="edge_id", right_index=True, how="left")

    e = edges.set_index("edge_id")
    agg = agg.join(e[_edge_columns(e)], on="edge_id")
    gdf = gpd.GeoDataFrame(agg, geometry="geometry", crs=edges.crs)
    where = locate(gdf.geometry.values, neighborhoods, edges.crs)
    gdf["neighborhood"] = where["neighborhood"].to_numpy()
    gdf["nearest_neighborhood"] = where["nearest_neighborhood"].to_numpy()
    gdf["nearest_distance_m"] = where["nearest_distance_m"].to_numpy()
    # ties in pairs served are broken by height, the higher pass first: it
    # is the harder one to do without
    return (gdf.sort_values(["pairs_served", "pass_elev_m"],
                            ascending=[False, False], kind="stable")
            .reset_index(drop=True))



# --------------------------------------------------------------------------
# barriers
# --------------------------------------------------------------------------
def find_barriers(graph, arc_store: dict, edges, neighborhoods,
                  min_grade: float = 0.08, profile: str = "shortest"):
    """Steep edges that inter-neighborhood traffic cannot avoid.

    Usage is counted on the **shortest**-path routes, because those describe
    what the street grid forces on someone who is not trying to avoid hills;
    a steep edge with high shortest-path usage is a true barrier in the city's
    fabric.  Each barrier is then annotated with how much of that traffic
    survives on the climb-averse routes -- a barrier that vanishes under the
    flat objective has an alternative, while one that persists does not.
    """
    import geopandas as gpd

    arc_edge = graph.table["edge_id"].to_numpy()
    short_use: collections.Counter = collections.Counter()
    flat_use: collections.Counter = collections.Counter()
    for (pname, o, d), arcs in arc_store.items():
        if not arcs:
            continue
        target = short_use if pname == profile else flat_use
        for eid in np.unique(arc_edge[arcs]):
            target[int(eid)] += 1

    steep = edges[edges["max_abs_grade"] >= min_grade].copy()
    steep["shortest_use"] = steep["edge_id"].map(short_use).fillna(0).astype(int)
    steep["flat_use"] = steep["edge_id"].map(flat_use).fillna(0).astype(int)
    steep = steep[steep["shortest_use"] > 0].copy()
    # ``flat_use`` is summed over the three climb-averse objectives, so it is
    # not directly comparable with ``shortest_use`` until it is averaged.
    n_flat_profiles = len({p for (p, _o, _d) in arc_store if p != profile}) or 1
    steep["flat_use_per_objective"] = steep["flat_use"] / n_flat_profiles
    # a barrier is unavoidable to the extent flat routes still have to use it
    steep["unavoidability"] = (steep["flat_use_per_objective"]
                               / steep["shortest_use"].clip(lower=1)).clip(0, 1)
    steep["barrier_score"] = (steep["shortest_use"]
                              * steep["max_abs_grade"]
                              * steep["length_m"] / 100.0)
    steep = steep.sort_values("barrier_score", ascending=False)

    gdf = gpd.GeoDataFrame(steep, geometry="geometry", crs=edges.crs)
    where = locate(gdf.geometry.values, neighborhoods, edges.crs)
    gdf["neighborhood"] = where["neighborhood"].to_numpy()
    gdf["nearest_neighborhood"] = where["nearest_neighborhood"].to_numpy()
    gdf["nearest_distance_m"] = where["nearest_distance_m"].to_numpy()
    log.info("barriers: %d steep edges (>=%.0f%%) carry shortest-path traffic",
             len(gdf), min_grade * 100)
    return gdf


# --------------------------------------------------------------------------
# output files
# --------------------------------------------------------------------------
def _write_layer(gdf, geojson_path, csv_path, columns) -> None:
    """Write a layer as GeoJSON and CSV, also when it has no rows.

    Upstream skipped empty layers, and the map and report stages, which
    open these files unconditionally, then failed on a missing file.  An
    empty layer is a legitimate result here: a build of one flat district,
    or of an island with a single neighborhood, has no pass at all.
    """
    if len(gdf):
        gdf.to_crs("EPSG:4326").to_file(geojson_path, driver="GeoJSON")
        if csv_path is not None:
            gdf.drop(columns=["geometry"]).to_csv(csv_path, index=False)
        return
    geojson_path.write_text(json.dumps({"type": "FeatureCollection",
                                        "features": []}))
    if csv_path is not None:
        pd.DataFrame(columns=list(columns)).to_csv(csv_path, index=False)


def read_layer(path, columns=()):
    """Read one of the GeoJSON outputs, with ``columns`` present even if empty.

    A missing file reads as an empty layer too, so the map and the report
    can be made from a run whose pass analysis found nothing.
    """
    import geopandas as gpd

    if path.exists():
        gdf = gpd.read_file(path)
    else:
        gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs="EPSG:4326"),
                               crs="EPSG:4326")
    for c in columns:
        if c not in gdf.columns:
            gdf[c] = pd.Series([None] * len(gdf), index=gdf.index, dtype="object")
    return gdf


def read_passes():
    """The critical passes, most pairs served first."""
    return read_layer(PASSES_GEOJSON, PASS_COLUMNS)


def read_barriers():
    """The barriers, highest barrier score first."""
    return read_layer(BARRIERS_GEOJSON, BARRIER_COLUMNS)


def read_basins():
    """The lowland basins' streets."""
    return read_layer(BASINS_GEOJSON, BASIN_COLUMNS)


def run_pass_analysis(graphs: dict, arc_store: dict, edges, directed,
                      neighborhoods, points=None, force: bool = False):
    """Full pass/barrier/basin pipeline (walking network)."""
    import geopandas as gpd

    if PASSES_GEOJSON.exists() and BARRIERS_GEOJSON.exists() and not force:
        log.info("cached passes and barriers")
        return read_passes(), read_barriers(), read_basins()

    node_z = node_elevations(directed)

    # --- lowland basins (descriptive geography) ---
    labels, basins = find_basins(edges, node_z, mode="walk")
    basin_names = name_basins(labels, edges, neighborhoods, node_z)
    for b, info in sorted(basins.items()):
        log.info("  basin %-2d %-46s %6.1f km of street", b,
                 basin_names.get(b, {}).get("label", "?"), info["length_km"])
    pd.DataFrame(
        [{"basin": b, "label": basin_names.get(b, {}).get("label", ""),
          "neighborhoods": basin_names.get(b, {}).get("neighborhoods", ""),
          "length_km": info["length_km"], "n_nodes": info["n_nodes"],
          "threshold_m": BASIN_ELEV_M}
         for b, info in sorted(basins.items())],
        columns=["basin", "label", "neighborhoods", "length_km", "n_nodes",
                 "threshold_m"]).to_csv(BASINS_CSV, index=False)

    # --- minimax passes between every neighborhood pair ---
    tree = build_bottleneck_tree(edges, mode="walk")
    pass_matrix = neighborhood_pass_matrix(tree, points) if points is not None \
        else pd.DataFrame(columns=list(PASS_MATRIX_COLUMNS))
    passes_gdf = rank_critical_passes(pass_matrix, edges, neighborhoods)
    if points is not None:
        pass_matrix[LOW_ROUTE_COLUMN] = low_route_lengths(pass_matrix, edges,
                                                          points)
    pass_matrix.to_csv(PASS_MATRIX_CSV, index=False)
    log.info("  %d distinct critical passes over %d neighborhood pairs",
             len(passes_gdf), len(pass_matrix))

    # --- basin-to-basin passes (kept as a coarser summary) ---
    basin_passes = find_passes(edges, node_z, labels, mode="walk")
    described = describe_passes(basin_passes, edges, neighborhoods, basin_names)
    described.drop(columns=["geometry"]).to_csv(BASIN_PASSES_CSV, index=False)

    barriers = find_barriers(graphs["walk"], arc_store["walk"], edges,
                             neighborhoods)

    sub = edges[edges["u"].map(labels).notna()].copy()
    sub["basin"] = sub["u"].map(labels).astype("Int64")
    # ``edge_id`` goes with each row so that the map can mark the basin
    # streets in its own copy of the network instead of drawing these again
    basins_gdf = gpd.GeoDataFrame(
        sub[["edge_id", "basin", "name", "length_m", "geometry"]],
        geometry="geometry", crs=edges.crs)
    basins_gdf["basin_label"] = basins_gdf["basin"].map(
        lambda b: basin_names.get(int(b), {}).get("label", "") if pd.notna(b) else "")
    # the basin's own length (lowland_basins.csv), for the map's inspector
    basins_gdf["basin_km"] = basins_gdf["basin"].map(
        lambda b: basins.get(int(b), {}).get("length_km") if pd.notna(b) else None)

    _write_layer(passes_gdf, PASSES_GEOJSON, PASSES_CSV, PASS_COLUMNS)
    _write_layer(barriers.head(400), BARRIERS_GEOJSON, BARRIERS_CSV,
                 BARRIER_COLUMNS)
    _write_layer(basins_gdf, BASINS_GEOJSON, None, BASIN_COLUMNS)
    log.info("wrote passes, barriers and basins")
    return passes_gdf, barriers, basins_gdf
