"""Tests for map-output helpers."""
import json

import numpy as np
import pytest

from hk_flat_routes.viz_interactive import _round_geometry
from hk_flat_routes.viz_static import hillshade

# Polyline encoding now lives in the browser payload packer; its tests are in
# tests/test_webgraph.py.

CRS = "EPSG:2326"


def test_round_geometry_shortens_coordinates():
    geo = {"type": "LineString",
           "coordinates": [(114.158201234567, 22.281931234567),
                           (114.172241234567, 22.295071234567)]}
    r = _round_geometry(geo, 5)
    assert r["coordinates"][0] == [114.1582, 22.28193]
    assert len(json.dumps(r)) < len(json.dumps(geo))


def test_round_geometry_handles_multilinestring():
    geo = {"type": "MultiLineString",
           "coordinates": [[(114.1234567, 22.1234567), (114.2, 22.2)],
                           [(114.3, 22.3)]]}
    r = _round_geometry(geo, 4)
    assert r["coordinates"][0][0] == [114.1235, 22.1235]
    assert r["type"] == "MultiLineString"


def test_round_geometry_keeps_only_two_dimensions():
    geo = {"type": "LineString", "coordinates": [(114.1, 22.3, 552.0)]}
    assert _round_geometry(geo, 5)["coordinates"][0] == [114.1, 22.3]


def test_hillshade_is_bounded_and_flat_ground_is_uniform():
    flat = np.zeros((20, 20))
    hs = hillshade(flat, res=1.0)
    assert hs.shape == flat.shape
    assert np.all((hs >= 0) & (hs <= 1))
    assert np.ptp(hs) == pytest.approx(0.0, abs=1e-9)


def test_hillshade_distinguishes_slope_direction():
    ramp = np.tile(np.arange(20.0), (20, 1))          # rises to the east
    hs = hillshade(ramp, res=1.0)
    other = hillshade(-ramp, res=1.0)
    assert not np.allclose(hs, other)


def test_hillshade_lights_the_north_west_face_of_a_hill():
    """The light comes from the north-west, so that hills read as hills.

    Rows run north to south.  A ridge running north-east to south-west has
    one face toward the light and one away from it; with the gradient
    arguments swapped (as they once were upstream) the two change places.
    """
    y, x = np.mgrid[0:60, 0:60].astype(float)       # y grows southward
    ridge = 30.0 - np.abs((x - 30.0) + (y - 30.0)) / np.sqrt(2.0)
    hs = hillshade(ridge, res=1.0)
    north_west = hs[15, 15]                          # row 15 is north, col 15 west
    south_east = hs[45, 45]
    assert north_west > south_east + 0.2


def test_hillshade_respects_cell_size():
    """The same heights on coarser cells are gentler slopes, so less contrast."""
    ramp = np.tile(np.arange(20.0), (20, 1))
    fine = hillshade(ramp, res=1.0)
    coarse = hillshade(ramp, res=10.0)
    flat = hillshade(np.zeros((20, 20)), res=1.0)
    assert abs(coarse.mean() - flat.mean()) < abs(fine.mean() - flat.mean())


# --------------------------------------------------------------------------
# the frame: fitted to whatever network a build has
# --------------------------------------------------------------------------
def street_grid(x0, y0, width, height, step=500.0):
    """A rectangular grid of streets on the HK1980 Grid, with edge metrics."""
    import geopandas as gpd
    from shapely.geometry import LineString
    lines = []
    xs = np.arange(x0, x0 + width + 1, step)
    ys = np.arange(y0, y0 + height + 1, step)
    for y in ys:
        lines += [LineString([(a, y), (b, y)]) for a, b in zip(xs[:-1], xs[1:])]
    for x in xs:
        lines += [LineString([(x, a), (x, b)]) for a, b in zip(ys[:-1], ys[1:])]
    n = len(lines)
    grade = np.linspace(0.0, 0.3, n)
    return gpd.GeoDataFrame({
        "edge_id": np.arange(n), "walk_ok": True, "bike_ok": np.arange(n) % 2 == 0,
        "max_abs_grade": grade, "grade_reliable": True,
        "name": ["Hennessy Road"] * n, "name_zh": ["軒尼詩道"] * n,
    }, geometry=lines, crs=CRS)


def planning_areas(x0, y0, width, height):
    import geopandas as gpd
    from shapely.geometry import box
    half = width / 2.0
    return gpd.GeoDataFrame({
        "neighborhood": ["Wan Chai", "Tsz Wan Shan, Diamond Hill & San Po Kong"],
        "neighborhood_zh": ["灣仔", "慈雲山、鑽石山及新蒲崗"],
        "area_km2": [half * height / 1e6] * 2,
    }, geometry=[box(x0, y0, x0 + half, y0 + height),
                 box(x0 + half, y0, x0 + width, y0 + height)], crs=CRS)


def test_frame_follows_the_network_present():
    from hk_flat_routes.viz_static import map_extent
    core = street_grid(830000, 812000, 12000, 9000)
    ext = map_extent(core)
    assert ext[0] < 830000 and ext[2] > 842000
    assert ext[1] < 812000 and ext[3] > 821000
    # the margin is a few percent, not a different map
    assert ext[2] - ext[0] < 12000 * 1.1
    # only streets of the mode count: nothing here is rideable east of ...
    half = core.copy()
    half["bike_ok"] = half.geometry.bounds["maxx"] <= 836000
    assert map_extent(half, mode="bike")[2] < map_extent(half, mode="walk")[2]


def test_page_is_landscape_or_portrait_as_the_frame_demands():
    from hk_flat_routes.viz_static import FIG_LONG_IN, FIG_MIN_ASPECT, figure_size
    w, h = figure_size((800000, 800000, 865000, 848000))    # the territory: 65 x 48 km
    assert w == FIG_LONG_IN and h < w
    assert h == pytest.approx(FIG_LONG_IN * 48 / 65)
    w, h = figure_size((830000, 808000, 842000, 826000))    # 12 x 18 km
    assert h == FIG_LONG_IN and w < h
    w, h = figure_size((0, 0, 100000, 1000))                # a thread of a network
    assert h == pytest.approx(FIG_LONG_IN * FIG_MIN_ASPECT)


def test_scale_bar_is_a_round_length():
    from hk_flat_routes.viz_static import _nice_length
    assert _nice_length(0.16 * 65000) == 10000.0
    assert _nice_length(0.16 * 18000) == 2000.0
    assert _nice_length(7400) == 5000.0
    assert _nice_length(999) == 500.0


def test_corridors_thin_out_on_a_sheet_of_the_whole_territory():
    from hk_flat_routes.viz_static import line_weight
    core = line_weight((830000, 808000, 846000, 824000))        # 16 km
    territory = line_weight((800000, 800000, 865000, 848000))   # 65 km
    assert core == 1.0
    assert 0.4 <= territory < core
    assert line_weight((0, 0, 1e7, 1e7)) == line_weight((0, 0, 1e6, 1e6))   # floored


# --------------------------------------------------------------------------
# lettering
# --------------------------------------------------------------------------
def test_cjk_font_is_a_font_that_really_has_the_glyphs():
    """Either nothing (English-only labels) or a face that can draw the names."""
    from matplotlib.ft2font import FT2Font
    from hk_flat_routes.viz_static import cjk_font
    sample = "鰂魚涌 香港仔及鴨脷洲 啓德"
    fp = cjk_font(sample)
    if fp is None:
        pytest.skip("no font with Chinese glyphs on this machine")
    face = FT2Font(fp.get_file())
    for ch in sample.replace(" ", ""):
        assert face.get_char_index(ord(ch)) != 0


def test_cjk_font_refuses_a_character_no_font_has():
    from hk_flat_routes.viz_static import cjk_font
    assert cjk_font("\U000F0000") is None        # a private-use code point


def test_long_names_are_wrapped():
    from hk_flat_routes.viz_static import _has_cjk, _wrap
    lines = _wrap("Tsz Wan Shan, Diamond Hill & San Po Kong").split("\n")
    assert len(lines) >= 2 and max(len(s) for s in lines) <= 18
    assert _wrap("Wan Chai") == "Wan Chai"
    assert _has_cjk("慈雲山") and not _has_cjk("Tsz Wan Shan") and not _has_cjk(None)


def test_overlapping_labels_are_refused():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from hk_flat_routes.viz_static import _Labels
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    ax.set_xlim(830000, 834000); ax.set_ylim(814000, 818000)
    labels = _Labels(ax)
    style = {"fontsize": 8.0}
    assert labels.add(832000, 816000, [("Central District", style)])
    assert not labels.add(832010, 816010, [("Sheung Wan", style)])   # on top of it
    assert labels.add(832000, 817500, [("Mid-Levels", style)])       # well clear
    assert not labels.add(900000, 816000, [("Sha Tin", style)])      # off the sheet
    assert len(labels.kept) == 2
    assert sorted(t.get_text() for t in ax.texts) == ["Central District", "Mid-Levels"]
    plt.close(fig)


# --------------------------------------------------------------------------
# the two figures, end to end on a synthetic district
# --------------------------------------------------------------------------
@pytest.fixture
def small_figures(tmp_path, monkeypatch):
    """Send the figures to a temporary directory and keep them small."""
    import hk_flat_routes.viz_static as V
    monkeypatch.setattr(V, "STATIC_PNG", tmp_path / "hk_flat_backbone.png")
    monkeypatch.setattr(V, "STATIC_PDF", tmp_path / "hk_flat_backbone.pdf")
    monkeypatch.setattr(V, "GRADE_PNG", tmp_path / "hk_street_grades.png")
    monkeypatch.setattr(V, "FIG_LONG_IN", 5.0)

    def relief(extent, max_px=0):
        """A hill in the middle of the frame, sea along its western edge."""
        y, x = np.mgrid[0:60, 0:80].astype(float)
        dem = 300.0 * np.exp(-((x - 40) ** 2 + (y - 30) ** 2) / 300.0)
        valid = x > 6
        return (hillshade(dem, res=50.0), np.where(valid, dem, np.nan), valid,
                (extent[0], extent[2], extent[1], extent[3]))

    monkeypatch.setattr(V, "_load_hillshade", relief)
    return V


def test_backbone_map_renders(small_figures):
    import geopandas as gpd
    from shapely.geometry import LineString
    V = small_figures
    edges = street_grid(833000, 815000, 4000, 3000)
    areas = planning_areas(833000, 815000, 4000, 3000)
    corridors = gpd.GeoDataFrame({
        "mode": ["walk", "walk", "bike"],
        "corridor_name": ["Hennessy Road - Johnston Road", "吉席街", "Gloucester Road"],
        "total_score": [9.0, 4.0, 5.0], "mean_abs_grade": [0.004, 0.02, 0.01],
    }, geometry=[LineString([(833200, 815500), (836800, 815500)]),
                 LineString([(833200, 817000), (836800, 817000)]),
                 LineString([(833200, 816000), (836800, 816000)])], crs=CRS)
    passes = gpd.GeoDataFrame({"pairs_served": [3]}, geometry=[
        LineString([(835000, 816500), (835100, 816500)])], crs=CRS)
    png, pdf = V.make_backbone_map(corridors, edges, areas, passes)
    assert png.exists() and png.stat().st_size > 20_000
    assert pdf.exists() and pdf.read_bytes()[:5] == b"%PDF-"
    from PIL import Image
    w, h = Image.open(png).size
    assert w > h                                   # a 4 x 3 km district: landscape


def test_backbone_map_renders_with_nothing_found(small_figures):
    """A build with no corridors and no passes still gets its map."""
    import geopandas as gpd
    V = small_figures
    edges = street_grid(833000, 815000, 2000, 4000)
    areas = planning_areas(833000, 815000, 2000, 4000)
    no_corridors = gpd.GeoDataFrame(columns=["corridor_id", "geometry"],
                                    geometry="geometry", crs=CRS)
    no_passes = gpd.GeoDataFrame(geometry=gpd.GeoSeries([], crs="EPSG:4326"))
    png, _ = V.make_backbone_map(no_corridors, edges, areas, no_passes)
    from PIL import Image
    w, h = Image.open(png).size
    assert h > w                                   # 2 x 4 km: portrait


def test_grade_map_renders(small_figures):
    V = small_figures
    edges = street_grid(833000, 815000, 4000, 3000)
    png = V.make_grade_map(edges, planning_areas(833000, 815000, 4000, 3000))
    assert png.exists() and png.stat().st_size > 20_000


def test_maps_render_without_a_terrain_model(small_figures, monkeypatch):
    """No overview raster (no tiles on disk) means no relief, not no map."""
    V = small_figures
    monkeypatch.setattr(V, "_load_hillshade", lambda extent, max_px=0: None)
    edges = street_grid(833000, 815000, 3000, 3000)
    assert V.make_grade_map(edges, planning_areas(833000, 815000, 3000, 3000)).exists()


def test_relief_is_read_from_the_overview_inside_the_frame(tmp_path, monkeypatch):
    """The relief comes from the 10 m overview: nodata is sea, the rest land."""
    import rasterio
    from rasterio.transform import from_origin
    import hk_flat_routes.elevation as E
    import hk_flat_routes.viz_static as V

    path = tmp_path / "dem_hk_10m.tif"
    y, x = np.mgrid[0:300, 0:400].astype("float32")
    dem = 2.0 * x                                   # rises eastward, 0.2 m per metre
    dem[:, :50] = -9999.0                           # the sea, in the west
    with rasterio.open(path, "w", driver="GTiff", height=300, width=400, count=1,
                       dtype="float32", crs=CRS, nodata=-9999.0,
                       transform=from_origin(830000, 818000, 10, 10)) as dst:
        dst.write(dem, 1)
    monkeypatch.setattr(E, "DEM_MOSAIC", path)

    hs, elev, valid, extent = V._load_hillshade((829000, 814000, 835000, 819000),
                                                max_px=100)
    # clipped to where the raster is, in matplotlib's (left, right, bottom, top)
    assert extent == (830000.0, 834000.0, 815000.0, 818000.0)
    assert max(hs.shape) <= 100 and hs.shape == elev.shape == valid.shape
    assert not valid[:, 0].any() and valid[:, -1].all()        # sea west, land east
    assert np.isnan(elev[~valid]).all()
    assert np.nanmin(elev) >= 0.0                   # the sea's nodata did not bleed in
    assert np.isfinite(hs).all()                    # shaded on past the shore
    # a frame that misses the raster altogether has no relief
    assert V._load_hillshade((700000, 700000, 701000, 701000)) is None


def test_corridor_width_is_relative_to_its_own_landmass():
    """The best corridor of a small landmass is drawn as boldly as any other's."""
    import pandas as pd
    from hk_flat_routes.viz_static import relative_score
    cor = pd.DataFrame({"total_score": [90.0, 45.0, 3.0, 1.5],
                        "landmass": [0, 0, 2, 2]})
    assert relative_score(cor).tolist() == [1.0, 0.5, 1.0, 0.5]
    # no landmass column (upstream's table): the best overall is the yardstick
    plain = relative_score(cor.drop(columns=["landmass"]))
    assert plain[0] == 1.0 and plain[2] == pytest.approx(3.0 / 90.0)
    assert relative_score(cor.iloc[0:0]).size == 0


def test_every_landmass_gets_its_leading_passes_marked():
    """The pass table is ranked by pairs bound, which favours the mainland.

    Two streets with no link between them: a mainland whose passes each bind
    dozens of pairs and an island whose passes bind a handful.  The head of
    the table is all mainland; taken landmass by landmass, the island's
    passes are marked too, each landmass in the table's order.
    """
    import pandas as pd
    from hk_flat_routes.viz_static import leading_passes
    edges = pd.DataFrame({
        "edge_id": range(12),
        "u": list(range(0, 6)) + list(range(100, 106)),
        "v": list(range(1, 7)) + list(range(101, 107)),
        "walk_ok": True})
    passes = pd.DataFrame({
        "edge_id": [0, 1, 2, 3, 4, 5, 6, 7, 8],
        "pairs_served": [90, 80, 70, 60, 50, 40, 4, 3, 2]})
    assert set(passes.head(4)["edge_id"]) <= set(range(6))       # mainland only
    marked = leading_passes(passes, edges, per_landmass=2)
    assert marked["edge_id"].tolist() == [0, 1, 6, 7]
    # one landmass: simply the head of the table
    one = leading_passes(passes.head(6), edges, per_landmass=3)
    assert one["edge_id"].tolist() == [0, 1, 2]
    # a pass table or an edge table that cannot say where a pass is
    assert len(leading_passes(passes.drop(columns=["edge_id"]), edges,
                              per_landmass=2)) == 4
    assert len(leading_passes(passes, edges.drop(columns=["u"]),
                              per_landmass=2)) == 4
    assert leading_passes(passes.iloc[0:0], edges).empty
