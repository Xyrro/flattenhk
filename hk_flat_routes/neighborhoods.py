"""Neighborhood boundaries, city boundary and representative access points.

Choosing an origin/destination point per neighborhood matters more than it
looks.  A polygon centroid can easily land in the middle of a park, on a
hillside with no street, in the water, or (for a concave planning area like
Tai Tam & Shek O) outside the neighborhood altogether.  Routing from
such a point produces garbage distances.

The representative point is therefore chosen as follows:

1.  Take every graph node inside the neighborhood that lies on the landmass
    carrying most of its street.
2.  Compute a **street-weighted centre**: the mean node position weighted by
    the length of street incident on each node.  Street length is a far
    better proxy for where people actually start journeys than polygon area,
    so parks, cliffs and water pull the centre much less than they pull a
    geometric centroid.
3.  Snap to the nearest *qualifying intersection* -- a node of degree >= 3 on
    a named street of an ordinary urban class, in the core of the landmass
    (so that a route exists to and from every other access point on it) and
    at the height most of the neighborhood's streets are at.  Falling back
    progressively to looser criteria guarantees a usable point for every
    neighborhood.

The height matters in Hong Kong as it does not in San Francisco.  A planning
area typically runs from a reclaimed waterfront up the hillside behind it,
so its street-weighted centre falls on the slope between the two, and the
nearest intersection to that is on a hillside road: North Point was given a
point at 110 mPD on Cloud View Road when the median of its ordinary streets
is at 5 mPD, and was then reported as a neighborhood on a hill.  Candidates
are therefore held to within ``ACCESS_HEIGHT_BAND_M`` of that median.

The offset between the geometric centroid and the chosen point is recorded so
the choice is auditable, and so are the point's height and the median it was
held to.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import (ANALYSIS, CRS_PROJECTED, EXCLUDED_NEIGHBORHOODS,
                     NEIGHBORHOOD_MERGE, NEIGHBORHOOD_PLAN_PREFIXES,
                     NEIGHBORHOOD_PLANS, PROCESSED_DIR)
from .download import NEIGHBORHOODS_GEOJSON
from .utils import get_logger, step

log = get_logger("hk_flat_routes.neighborhoods")

NEIGHBORHOODS_GPKG = PROCESSED_DIR / "neighborhoods.gpkg"
POINTS_GPKG = PROCESSED_DIR / "neighborhood_points.gpkg"

#: Classes that make a node a credible journey origin (not a park path,
#: alley, driveway or freeway ramp).
_GOOD_CLASSES = ("residential", "living_street", "tertiary", "secondary",
                 "primary", "unclassified", "trunk", "pedestrian")

#: An access point is taken within this height (m) of the median height of
#: its neighborhood's ordinary streets (those of ``_GOOD_CLASSES``, weighted
#: by length), wherever a qualifying intersection stands that close to it.
ACCESS_HEIGHT_BAND_M = 10.0


def _write_gpkg(gdf, path: Path) -> None:
    """Write a GeoPackage beside its target, then move it into place.

    A run directory (``HKFR_RUN_DIR``) can hold a link to the main build's
    copy of a shared file -- the sensitivity harness links
    ``neighborhoods.gpkg`` into every run -- and a plain write follows the
    link and overwrites the main build.  The move replaces the link itself.
    """
    part = path.with_name(path.stem + ".part.gpkg")
    part.unlink(missing_ok=True)
    gdf.to_file(part, layer=path.stem, driver="GPKG")
    part.replace(path)


def _plan_code(plan_no: str) -> str:
    """'S/H15/34' -> 'S/H15': the plan without its amendment number."""
    return str(plan_no).rsplit("/", 1)[0]


def load_neighborhoods(path: Path = NEIGHBORHOODS_GEOJSON, force: bool = False):
    """Load neighborhood polygons, projected, with cleaned attributes.

    The source is every statutory planning scheme area in Hong Kong; the
    ones kept are listed in ``config`` (``NEIGHBORHOOD_PLAN_PREFIXES``,
    ``NEIGHBORHOOD_PLANS``), with extensions dissolved into their parent
    plan (``NEIGHBORHOOD_MERGE``).  The planning areas do not tile the
    territory -- most of the countryside is in no plan -- so they name places and
    seed access points, and are *not* what the network is clipped to (see
    ``city_boundary``).
    """
    import geopandas as gpd

    if NEIGHBORHOODS_GPKG.exists() and not force:
        return gpd.read_file(NEIGHBORHOODS_GPKG)

    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    if "PLAN_NO" in gdf.columns:
        code = gdf["PLAN_NO"].map(_plan_code)
        keep = code.isin(NEIGHBORHOOD_PLANS) | code.isin(NEIGHBORHOOD_MERGE)
        for prefix in NEIGHBORHOOD_PLAN_PREFIXES:
            keep |= code.str.match(rf"^{prefix}\d")
        gdf = gdf[keep].copy()
        gdf["code"] = code[keep].replace(NEIGHBORHOOD_MERGE)
        # the parent plan names the merged unit
        parent = ~code[keep].isin(NEIGHBORHOOD_MERGE)
        names = gdf[parent.to_numpy()].set_index("code")[["NAME_ENG", "NAME_CHT"]]
        gdf["geometry"] = gdf.geometry.buffer(0)
        gdf = gdf.dissolve(by="code").reset_index()
        gdf["name"] = gdf["code"].map(names["NAME_ENG"])
        gdf["name_zh"] = gdf["code"].map(names["NAME_CHT"])
    gdf = gdf.rename(columns={"name": "neighborhood",
                              "name_zh": "neighborhood_zh"})
    if "neighborhood_zh" not in gdf.columns:
        gdf["neighborhood_zh"] = ""
    gdf = gdf[["neighborhood", "neighborhood_zh", "geometry"]].copy()
    gdf = gdf.to_crs(CRS_PROJECTED)
    gdf["geometry"] = gdf.geometry.buffer(0)          # repair any self-touching rings
    gdf["area_km2"] = gdf.geometry.area / 1e6
    gdf = gdf.sort_values("neighborhood").reset_index(drop=True)
    NEIGHBORHOODS_GPKG.parent.mkdir(parents=True, exist_ok=True)
    _write_gpkg(gdf, NEIGHBORHOODS_GPKG)
    log.info("loaded %d neighborhoods, %.1f km2 total",
             len(gdf), gdf["area_km2"].sum())
    return gdf


def city_boundary(neighborhoods=None, buffer_m: float = 0.0):
    """The Hong Kong SAR, land and waters, in the projected CRS.

    This is what the street network is clipped to.  It is the SAR's own
    boundary from Overture's divisions theme rather than the union of the
    neighborhood polygons: the border with Shenzhen runs down the middle of
    a river 50 m wide, so the 250 m margin upstream allows around San
    Francisco would take in a strip of Shenzhen's streets, while a polygon
    of land alone would cut off piers, breakwaters and the bridges out to
    the airport.  ``neighborhoods`` is accepted for call compatibility.
    """
    import geopandas as gpd
    import pyarrow.parquet as pq
    import shapely
    from shapely.ops import unary_union

    from .download import DIVISIONS_PARQUET

    t = pq.read_table(DIVISIONS_PARQUET,
                      columns=["subtype", "geometry"]).to_pandas()
    sar = t[t["subtype"] == "dependency"]
    if sar.empty:
        raise RuntimeError(f"no SAR polygon in {DIVISIONS_PARQUET}")
    geom = unary_union(list(shapely.from_wkb(sar["geometry"].values)))
    geom = gpd.GeoSeries([geom], crs="EPSG:4326").to_crs(CRS_PROJECTED).iloc[0]
    geom = geom.buffer(0)
    if buffer_m:
        geom = geom.buffer(buffer_m)
    return geom


def analysis_neighborhoods(neighborhoods=None):
    """Neighborhoods used for pair analysis (excludes unreachable islands)."""
    nb = load_neighborhoods() if neighborhoods is None else neighborhoods
    return nb[~nb["neighborhood"].isin(EXCLUDED_NEIGHBORHOODS)].reset_index(drop=True)


# --------------------------------------------------------------------------
# representative points
# --------------------------------------------------------------------------
def _node_table(edges) -> pd.DataFrame:
    """Node coordinates, incident street length and best incident class."""
    rows = []
    for u, v, geom, length, cls, name in zip(
            edges["u"], edges["v"], edges.geometry, edges["length_m"],
            edges["cls"], edges["name"]):
        c = geom.coords
        good = cls in _GOOD_CLASSES
        # a missing name arrives from the edge table as NaN, which is truthy
        named = isinstance(name, str) and bool(name.strip())
        rows.append((u, c[0][0], c[0][1], length, good, named))
        rows.append((v, c[-1][0], c[-1][1], length, good, named))
    df = pd.DataFrame(rows, columns=["node", "x", "y", "length_m", "good", "named"])
    df["good_len"] = df["length_m"].where(df["good"], 0.0)
    agg = df.groupby("node").agg(
        x=("x", "first"), y=("y", "first"),
        street_len=("length_m", "sum"), degree=("length_m", "size"),
        good=("good", "max"), named=("named", "max"),
        good_len=("good_len", "sum")).reset_index()
    return agg


def _street_median_height(sub: pd.DataFrame) -> float:
    """Height that half of a neighborhood's ordinary street lies below.

    ``sub`` is the neighborhood's rows of the node table with a height
    ``z``; each node stands for the ordinary-class street that meets at it
    (all street, where the neighborhood has none of an ordinary class).
    NaN when no node has a height.
    """
    known = sub[sub["z"].notna()]
    for col in ("good_len", "street_len"):
        w = known[col].to_numpy(dtype="float64")
        if w.sum() > 0:
            z = known["z"].to_numpy(dtype="float64")
            order = np.argsort(z, kind="stable")
            share = np.cumsum(w[order]) / w.sum()
            return float(z[order][min(int(np.searchsorted(share, 0.5)),
                                      len(z) - 1)])
    return float("nan")


def choose_representative_points(edges, neighborhoods=None, mode: str = "walk",
                                 valid_nodes=None, force: bool = False,
                                 node_comp: dict | None = None,
                                 core_nodes=None, node_z=None):
    """Pick one representative, network-snapped access point per neighborhood.

    ``valid_nodes`` restricts candidates to nodes that are actually routable
    for this mode (the landmasses of the mode's graph).  Without it a
    neighborhood can be handed a node that exists only on a footpath, which
    is unreachable by bicycle.

    ``node_comp`` maps each routable node to its landmass (its rank).  A
    neighborhood's point is taken on the landmass that carries most of its
    street length, so that a fenced compound or a theme park inside the
    polygon -- its own little landmass -- cannot capture the point and cut
    the neighborhood off from everything else.  The landmass is recorded in
    the ``component`` column.

    ``core_nodes`` holds the nodes in the core of their landmass
    (``RouteGraph.node_core``).  The point is taken among them, because two
    core nodes of a landmass have a route both ways, while a district that
    one-way streets lead into and not out of (Wan Chai North by bicycle)
    would give its neighborhood a point nothing can be reached from.

    ``node_z`` maps each node to its height (mPD).  The point is taken
    within ``ACCESS_HEIGHT_BAND_M`` of the median height of the
    neighborhood's ordinary streets where a qualifying intersection stands
    there; ``elev_m`` and ``street_median_m`` record the point's height and
    that median, so a point the rule could not bring down is visible.
    """
    import geopandas as gpd
    from shapely.geometry import Point
    from shapely.prepared import prep

    out_path = POINTS_GPKG.with_name(f"neighborhood_points_{mode}.gpkg")
    if out_path.exists() and not force:
        return gpd.read_file(out_path)

    nb = analysis_neighborhoods(neighborhoods)
    mode_edges = edges[edges[f"{mode}_ok"]] if f"{mode}_ok" in edges.columns else edges
    if mode == "bike" and "bike_push" in mode_edges.columns:
        # A pushed link is a footway the cyclist walks along.  It makes no
        # junction of the street it leaves, and nobody's ride starts on it.
        mode_edges = mode_edges[
            ~mode_edges["bike_push"].to_numpy(dtype=bool, na_value=False)]
    nodes = _node_table(mode_edges)
    if valid_nodes is not None:
        before = len(nodes)
        nodes = nodes[nodes["node"].isin(set(valid_nodes))].reset_index(drop=True)
        log.info("%s: %d of %d nodes are on a routable landmass",
                 mode, len(nodes), before)
    nodes["core"] = (nodes["node"].isin(set(core_nodes))
                     if core_nodes is not None else True)
    nodes["z"] = (nodes["node"].map(node_z).astype("float64")
                  if node_z is not None else np.nan)
    node_pts = gpd.GeoSeries(gpd.points_from_xy(nodes["x"], nodes["y"]),
                             crs=CRS_PROJECTED)
    sidx = node_pts.sindex

    records = []
    with step(f"choosing representative points for {len(nb)} neighborhoods ({mode})",
              log):
        for _, row in nb.iterrows():
            poly = row.geometry
            cand_idx = list(sidx.query(poly, predicate="intersects"))
            if not cand_idx:
                log.warning("%s: no network nodes inside polygon; using centroid",
                            row["neighborhood"])
                p = poly.representative_point()
                records.append({"neighborhood": row["neighborhood"], "node": None,
                                "geometry": p, "offset_m": 0.0,
                                "selection": "centroid_no_nodes"})
                continue
            pr = prep(poly)
            inside = [i for i in cand_idx if pr.contains(node_pts.iloc[i])]
            if not inside:
                inside = cand_idx
            sub = nodes.iloc[inside]
            if node_comp is not None:
                comp = sub["node"].map(node_comp)
                by_comp = sub.groupby(comp)["street_len"].sum()
                if len(by_comp):
                    sub = sub[(comp == by_comp.idxmax()).to_numpy()]

            # street-length-weighted centre of the neighborhood's network
            w = sub["street_len"].to_numpy()
            w = w / w.sum() if w.sum() > 0 else np.full(len(sub), 1 / len(sub))
            cx = float((sub["x"].to_numpy() * w).sum())
            cy = float((sub["y"].to_numpy() * w).sum())

            # the height its streets are at (all of them, not only those in
            # the core), and the nodes that are near it
            median_z = _street_median_height(sub)
            if sub["core"].any():
                sub = sub[sub["core"].to_numpy()]
            else:
                log.warning("%s: none of its %s nodes is in the core of its "
                            "landmass; a route to or from its access point "
                            "may exist one way only", row["neighborhood"], mode)
            level = ((sub["z"] - median_z).abs() <= ACCESS_HEIGHT_BAND_M).to_numpy()

            # snap to the nearest qualifying intersection, loosening criteria
            for crit, label in (
                ((sub["degree"] >= 3) & sub["good"] & sub["named"], "intersection"),
                ((sub["degree"] >= 3) & sub["good"], "intersection_unnamed"),
                (sub["good"], "street_node"),
                (sub["degree"] >= 3, "any_intersection"),
                (pd.Series(True, index=sub.index), "any_node"),
            ):
                crit = crit.to_numpy(dtype=bool)
                # at the neighborhood's own height if any candidate is
                pool = sub[crit & level] if (crit & level).any() else sub[crit]
                if len(pool):
                    d2 = ((pool["x"] - cx) ** 2 + (pool["y"] - cy) ** 2).to_numpy()
                    order = np.argsort(d2, kind="stable")
                    best = pool.iloc[int(order[min(ANALYSIS.point_rank, len(order) - 1)])]
                    centroid = poly.centroid
                    pt = Point(best["x"], best["y"])
                    records.append({
                        "neighborhood": row["neighborhood"],
                        "node": best["node"],
                        "geometry": pt,
                        "offset_m": float(pt.distance(centroid)),
                        "centroid_inside": bool(poly.contains(centroid)),
                        "selection": label,
                        "degree": int(best["degree"]),
                        "component": (int(node_comp[best["node"]])
                                      if node_comp is not None else 0),
                        "elev_m": float(best["z"]),
                        "street_median_m": median_z,
                    })
                    break

    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=CRS_PROJECTED)
    _write_gpkg(gdf, out_path)
    far = gdf[gdf["offset_m"] > 400]
    log.info("representative points: %d chosen; %d more than 400 m from the "
             "polygon centroid", len(gdf), len(far))
    for _, r in far.iterrows():
        log.info("   %-26s %6.0f m from centroid (%s)", r["neighborhood"],
                 r["offset_m"], r["selection"])
    if "elev_m" in gdf.columns:
        high = gdf[(gdf["elev_m"] - gdf["street_median_m"]).abs()
                   > ACCESS_HEIGHT_BAND_M]
        for _, r in high.iterrows():
            log.info("   %-26s at %.0f mPD, no qualifying intersection within "
                     "%.0f m of its median street (%.0f mPD)",
                     r["neighborhood"], r["elev_m"], ACCESS_HEIGHT_BAND_M,
                     r["street_median_m"])
    return gdf


def assign_edges_to_neighborhoods(edges, neighborhoods=None):
    """Label each edge with the neighborhood containing its midpoint."""
    import geopandas as gpd

    nb = load_neighborhoods() if neighborhoods is None else neighborhoods
    mid = edges.geometry.interpolate(0.5, normalized=True)
    pts = gpd.GeoDataFrame({"edge_id": edges["edge_id"].values},
                           geometry=mid.values, crs=CRS_PROJECTED)
    joined = gpd.sjoin(pts, nb[["neighborhood", "geometry"]], how="left",
                       predicate="within")
    joined = joined.drop_duplicates("edge_id")
    return joined.set_index("edge_id")["neighborhood"]
