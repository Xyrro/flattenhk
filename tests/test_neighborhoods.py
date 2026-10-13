"""Tests for the choice of a neighbourhood's access point.

One synthetic planning area on the HK1980 Grid (EPSG:2326), shaped like the
real ones: a waterfront street at 5 mPD, a road along the hillside at
110 mPD, and paths running on up the hill behind it.
"""
import numpy as np
import pandas as pd
import pytest

import hk_flat_routes.neighborhoods as N

CRS = "EPSG:2326"
X0, Y0 = 833000.0, 816000.0


def hillside_area(low_name="Java Road"):
    """(edges, neighbourhoods, node heights) for the area described above.

    Waterfront: a0 .. a4 along y = 100, each with a side street to b0 .. b4.
    Hillside:   h0 .. h2 along y = 600, each with a path up to p0 .. p2.
    Steps join a2 to h1.  The paths carry enough length to pull the
    street-weighted centre up to y = 423, nearer the hillside road than the
    waterfront.
    """
    import geopandas as gpd
    from shapely.geometry import LineString, box

    at = {}
    for k in range(5):
        at[f"a{k}"] = (100 + 200 * k, 100)
        at[f"b{k}"] = (100 + 200 * k, 20)
    for k in range(3):
        at[f"h{k}"] = (300 + 200 * k, 600)
        at[f"p{k}"] = (300 + 200 * k, 950)
    rows = []

    def edge(u, v, cls, name):
        (x1, y1), (x2, y2) = at[u], at[v]
        geom = LineString([(X0 + x1, Y0 + y1), (X0 + x2, Y0 + y2)])
        rows.append({"edge_id": len(rows), "u": u, "v": v, "cls": cls,
                     "name": name, "length_m": geom.length, "walk_ok": True,
                     "geometry": geom})

    for k in range(4):
        edge(f"a{k}", f"a{k + 1}", "residential", low_name)
    for k in range(5):
        edge(f"a{k}", f"b{k}", "residential", low_name)
    for k in range(2):
        edge(f"h{k}", f"h{k + 1}", "residential", "Cloud View Road")
    for k in range(3):
        edge(f"h{k}", f"p{k}", "path", None)
    edge("a2", "h1", "steps", None)
    edges = gpd.GeoDataFrame(rows, geometry="geometry", crs=CRS)
    area = gpd.GeoDataFrame(
        {"neighborhood": ["North Point"], "neighborhood_zh": ["北角"]},
        geometry=[box(X0, Y0, X0 + 1000, Y0 + 1000)], crs=CRS)
    z = {n: (5.0 if n[0] in "ab" else 110.0 if n[0] == "h" else 260.0) for n in at}
    return edges, area, pd.Series(z)


@pytest.fixture
def scratch(tmp_path, monkeypatch):
    """Send the points files to a temporary directory."""
    monkeypatch.setattr(N, "POINTS_GPKG", tmp_path / "neighborhood_points.gpkg")
    monkeypatch.setattr(N, "NEIGHBORHOODS_GPKG", tmp_path / "neighborhoods.gpkg")
    return tmp_path


def _chosen(**kwargs):
    edges, area, z = hillside_area(**{k: kwargs.pop(k) for k in ("low_name",)
                                      if k in kwargs})
    if kwargs.pop("heights", True):
        kwargs["node_z"] = z
    return N.choose_representative_points(edges, area, mode="walk", force=True,
                                          **kwargs).iloc[0]


def test_without_heights_the_point_lands_on_the_hillside(scratch):
    """What the rule is for: the street-weighted centre of an area that
    runs up a hill falls on the slope, and the nearest intersection to it is
    on the hillside road."""
    r = _chosen(heights=False)
    assert r["node"] == "h1" and r["selection"] == "intersection"
    assert np.isnan(r["elev_m"]) and np.isnan(r["street_median_m"])


def test_the_point_is_taken_at_the_height_of_the_areas_streets(scratch):
    """Three quarters of the ordinary street is on the waterfront at 5 mPD,
    so that is where the access point goes: the nearest waterfront
    intersection to the centre, not the hillside one 177 m from it."""
    r = _chosen()
    assert r["node"] == "a2" and r["selection"] == "intersection"
    assert r["elev_m"] == pytest.approx(5.0)
    assert r["street_median_m"] == pytest.approx(5.0)
    assert abs(r["elev_m"] - r["street_median_m"]) <= N.ACCESS_HEIGHT_BAND_M


def test_with_no_intersection_at_that_height_the_old_pool_stands(scratch, caplog):
    """An unnamed waterfront street leaves the hillside junction as the
    only named intersection: it is taken, as before, and the gap between
    the point and the area's streets is on record and in the log."""
    with caplog.at_level("INFO", logger="hk_flat_routes.neighborhoods"):
        r = _chosen(low_name=None)
    assert r["node"] == "h1" and r["selection"] == "intersection"
    assert r["elev_m"] == pytest.approx(110.0)
    assert r["street_median_m"] == pytest.approx(5.0)
    assert any("no qualifying intersection within" in rec.getMessage()
               and "North Point" in rec.getMessage() for rec in caplog.records)


def test_the_point_is_taken_in_the_core_of_the_landmass(scratch):
    """A node outside the core may have no route back; the nearest
    intersection is passed over for the nearest one in the core."""
    edges, _area, _z = hillside_area()
    everything = set(edges["u"]) | set(edges["v"])
    r = _chosen(core_nodes=everything - {"a1", "a2"})
    assert r["node"] == "a3"


def test_an_area_with_no_node_in_the_core_keeps_its_point_and_says_so(scratch, caplog):
    """Nothing to prefer, so the point is chosen as if there were no core;
    the pair matrix may then lack routes one way, and the log says why."""
    with caplog.at_level("WARNING", logger="hk_flat_routes.neighborhoods"):
        assert _chosen(core_nodes=set())["node"] == "a2"
    said = [r.getMessage() for r in caplog.records]
    assert len(said) == 1 and said[0].startswith(
        "North Point: none of its walk nodes is in the core of its landmass")
    caplog.clear()
    with caplog.at_level("WARNING", logger="hk_flat_routes.neighborhoods"):
        _chosen()
        _chosen(core_nodes={"a3"})
    assert not caplog.records


def test_the_height_of_an_areas_streets_counts_those_outside_the_core(scratch):
    """The waterfront outside the core (a district bicycles can enter and not
    leave): the point has to go to the hillside road, but the median it is
    measured against is still that of all the area's streets, so the gap is
    on record and not hidden by a median of the core alone."""
    edges, _area, _z = hillside_area()
    hillside = {n for n in set(edges["u"]) | set(edges["v"]) if n[0] in "hp"}
    r = _chosen(core_nodes=hillside)
    assert r["node"] == "h1"
    assert r["elev_m"] == pytest.approx(110.0)
    assert r["street_median_m"] == pytest.approx(5.0)


def test_the_landmass_with_most_street_still_decides_first(scratch):
    """The waterfront on a landmass of its own: the hillside, with its
    paths, has the more street (3.4 km against 2.9), so the point is taken
    there, at the height of *that* landmass's streets, and the landmass is
    recorded."""
    edges, _area, _z = hillside_area()
    nodes = set(edges["u"]) | set(edges["v"])
    comp = {n: (3 if n[0] in "ab" else 0) for n in nodes}
    r = _chosen(node_comp=comp, valid_nodes=nodes)
    assert r["node"] == "h1" and r["component"] == 0
    assert r["street_median_m"] == pytest.approx(110.0)


def test_an_unnamed_street_is_not_a_named_one():
    """A missing name arrives from the edge table as NaN, and NaN is truthy:
    read with ``bool`` every street counted as named, and the first rule of
    the choice (an intersection on a *named* street) never applied."""
    edges, _area, _z = hillside_area(low_name=None)
    assert edges["name"].isna().any()
    t = N._node_table(edges).set_index("node")
    assert not t.loc["a2", "named"] and not t.loc["p0", "named"]
    assert t.loc["h1", "named"] and t.loc["h0", "named"]
    blank = edges.assign(name=edges["name"].where(edges["name"].notna(), "  "))
    assert not N._node_table(blank).set_index("node").loc["a2", "named"]


def test_a_pushed_link_makes_no_junction_for_a_bicycle(scratch):
    """A pushed link is a footway the cyclist walks along.  Where one leaves
    a street mid-block the street has no junction there, and a pedestrian
    street that is only pushed along is not a street a ride starts on."""
    import geopandas as gpd
    from shapely.geometry import LineString, box

    at = {f"s{k}": (100 + 200 * k, 500) for k in range(5)}
    at.update(link=(500, 420), side=(700, 300), mall=(300, 420))
    rows = []
    for u, v, cls, name, push in (
            [(f"s{k}", f"s{k + 1}", "residential", "Lockhart Road", False)
             for k in range(4)]
            + [("s2", "link", "footway", "Lockhart Road", True),
               ("s1", "mall", "pedestrian", "Lee Tung Avenue", True),
               ("s3", "side", "residential", "Fleming Road", False)]):
        geom = LineString([(X0 + at[u][0], Y0 + at[u][1]),
                           (X0 + at[v][0], Y0 + at[v][1])])
        rows.append({"edge_id": len(rows), "u": u, "v": v, "cls": cls, "name": name,
                     "length_m": geom.length, "walk_ok": True, "bike_ok": True,
                     "bike_push": push, "geometry": geom})
    edges = gpd.GeoDataFrame(rows, geometry="geometry", crs=CRS)
    area = gpd.GeoDataFrame({"neighborhood": ["Wan Chai"]},
                            geometry=[box(X0, Y0, X0 + 1000, Y0 + 1000)], crs=CRS)

    def chosen(mode):
        return N.choose_representative_points(edges, area, mode=mode,
                                              force=True).iloc[0]
    # on foot the footway is a street like any other, and its corner, the
    # nearest to the centre of the area's streets, is the access point
    assert chosen("walk")["node"] == "s2"
    by_bike = chosen("bike")
    assert by_bike["node"] == "s3" and by_bike["selection"] == "intersection"
    # an edge table from before there were pushed links is read as it was
    old = edges.drop(columns=["bike_push"])
    assert N.choose_representative_points(old, area, mode="bike",
                                          force=True).iloc[0]["node"] == "s2"


def test_street_median_height_is_weighted_by_ordinary_street():
    sub = pd.DataFrame({"z": [5.0, 110.0, 260.0, np.nan],
                        "good_len": [300.0, 200.0, 0.0, 900.0],
                        "street_len": [300.0, 200.0, 5000.0, 900.0]})
    # 300 m at 5 mPD against 200 m at 110: paths and unknown heights apart
    assert N._street_median_height(sub) == pytest.approx(5.0)
    sub.loc[0, "good_len"] = 100.0
    assert N._street_median_height(sub) == pytest.approx(110.0)
    # an area with no ordinary street at all is judged on what it has
    sub["good_len"] = 0.0
    assert N._street_median_height(sub) == pytest.approx(260.0)
    assert np.isnan(N._street_median_height(sub.assign(z=np.nan)))


def test_node_table_counts_ordinary_street_apart():
    edges, _area, _z = hillside_area()
    t = N._node_table(edges).set_index("node")
    # a2: two blocks of the street and a side street, and the steps uphill
    assert t.loc["a2", "street_len"] == pytest.approx(200 + 200 + 80 + 500)
    assert t.loc["a2", "good_len"] == pytest.approx(200 + 200 + 80)
    assert t.loc["p0", "good_len"] == 0.0 and t.loc["a2", "degree"] == 4


# --------------------------------------------------------------------------
# files shared between run directories
# --------------------------------------------------------------------------
def test_points_are_written_beside_the_file_and_moved_into_place(scratch):
    """``build_context`` rewrites the points on every call.  In a run
    directory that links to another build's points file, that must replace
    the link, not the other build's file."""
    import geopandas as gpd
    edges, area, z = hillside_area()
    main = scratch / "main_points.gpkg"
    N.choose_representative_points(edges, area, mode="walk", force=True)
    (scratch / "neighborhood_points_walk.gpkg").rename(main)
    before = main.read_bytes()
    (scratch / "neighborhood_points_walk.gpkg").symlink_to(main)

    N.choose_representative_points(edges, area, mode="walk", force=True, node_z=z)
    out = scratch / "neighborhood_points_walk.gpkg"
    assert not out.is_symlink()
    assert main.read_bytes() == before
    assert gpd.read_file(out)["node"].tolist() == ["a2"]
    assert gpd.read_file(main)["node"].tolist() == ["h1"]
    assert not list(scratch.glob("*.part.gpkg"))


def test_neighbourhoods_are_rebuilt_without_touching_a_linked_copy(scratch):
    """The sensitivity harness links the main build's ``neighborhoods.gpkg``
    into each run directory; a forced reload there once overwrote it."""
    import geopandas as gpd
    _edges, area, _z = hillside_area()
    source = scratch / "plans.geojson"
    area.rename(columns={"neighborhood": "name", "neighborhood_zh": "name_zh"}) \
        .to_crs("EPSG:4326").to_file(source, driver="GeoJSON")
    main = scratch / "main_neighborhoods.gpkg"
    area.assign(neighborhood="The main build's").to_file(main, driver="GPKG")
    before = main.read_bytes()
    N.NEIGHBORHOODS_GPKG.symlink_to(main)

    got = N.load_neighborhoods(source, force=True)
    assert got["neighborhood"].tolist() == ["North Point"]
    assert main.read_bytes() == before
    assert gpd.read_file(main)["neighborhood"].tolist() == ["The main build's"]
    assert not N.NEIGHBORHOODS_GPKG.is_symlink()
    assert gpd.read_file(N.NEIGHBORHOODS_GPKG)["neighborhood"].tolist() == ["North Point"]
    # and without force the file just written is what comes back
    assert N.load_neighborhoods(scratch / "no_such_file.geojson")["neighborhood"] \
        .tolist() == ["North Point"]


def test_build_context_passes_force_and_heights_on(monkeypatch):
    """``build-network --force`` must refresh the neighbourhoods as it does
    everything else, and the access points need the node heights and the
    core of each landmass."""
    from types import SimpleNamespace

    from hk_flat_routes import (elevation, metrics, network, passes, pipeline,
                                routing)
    seen = {}
    graph = SimpleNamespace(node_ids=np.array(["a", "b", "c"]),
                            node_comp=np.array([0, 0, 0]),
                            node_core=np.array([True, False, True]))
    monkeypatch.setattr(network, "build_edges", lambda path, force=False: "raw")
    monkeypatch.setattr(elevation, "sample_edge_profiles",
                        lambda edges, dem, force=False: {})
    monkeypatch.setattr(metrics, "compute_edge_metrics",
                        lambda edges, prof, force=False: ("edges", "directed"))
    monkeypatch.setattr(passes, "node_elevations", lambda directed: {"a": 4.0})
    monkeypatch.setattr(routing, "build_route_graph", lambda directed, mode: graph)

    def load(force=False):
        seen["force"] = force
        return "areas"

    def choose(edges, neighborhoods, **kwargs):
        seen[kwargs["mode"]] = kwargs
        return "points"

    monkeypatch.setattr(N, "load_neighborhoods", load)
    monkeypatch.setattr(N, "choose_representative_points", choose)
    for forced in (False, True):
        ctx = pipeline.build_context(modes=("walk",), force=forced)
        assert seen["force"] is forced
    assert ctx.points == {"walk": "points"} and ctx.neighborhoods == "areas"
    kw = seen["walk"]
    assert kw["node_z"] == {"a": 4.0}
    assert list(kw["core_nodes"]) == ["a", "c"]
    assert kw["node_comp"] == {"a": 0, "b": 0, "c": 0}
