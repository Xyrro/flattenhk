"""Tests for the tiled DEM, its sampler and the Hong Kong profile corrections.

All but the last runs on synthetic tiles written to a temporary directory in
the same layout as the CEDD 2020 LiDAR model: zipped GeoTIFFs on a regular
grid of sheets, with an index of their bounds.  The last reads a real build,
and is skipped where there is none.
"""
import json
import zipfile

import numpy as np
import pytest

from hk_flat_routes import elevation
from hk_flat_routes.elevation import (DemSampler, TiledDem, flattest_offset,
                                      lateral_offsets)

RES = 0.5
COLS, ROWS = 200, 160            # a small sheet: 100 m x 80 m
X0, Y0 = 830000.0, 815000.0


def _surface(x, y):
    """A smooth hillside: 10% up to the east, 4% up to the north."""
    return 20.0 + 0.10 * (x - X0) + 0.04 * (y - Y0)


def _write_tiles(tmp_path, keys, surface=_surface, sea=None):
    """Write sheets at grid positions ``keys`` and return a TiledDem."""
    import rasterio
    from pyproj import Transformer
    from rasterio.io import MemoryFile
    from rasterio.transform import from_origin

    to_ll = Transformer.from_crs("EPSG:2326", "EPSG:4326", always_xy=True)
    w, h = COLS * RES, ROWS * RES
    feats = []
    for c, r in keys:
        left, bottom = X0 + c * w, Y0 + r * h
        xs = left + (np.arange(COLS) + 0.5) * RES
        ys = bottom + h - (np.arange(ROWS) + 0.5) * RES
        gx, gy = np.meshgrid(xs, ys)
        z = surface(gx, gy).astype("float32")
        if sea is not None:
            z[sea(gx, gy)] = -9999.0
        name = f"t{c}_{r}"
        with MemoryFile() as mem:
            with mem.open(driver="GTiff", height=ROWS, width=COLS, count=1,
                          dtype="float32", crs="EPSG:2326", nodata=-9999.0,
                          transform=from_origin(left, bottom + h, RES, RES)) as dst:
                dst.write(z, 1)
            raw = mem.read()
        with zipfile.ZipFile(tmp_path / f"{name}.zip", "w") as zf:
            zf.writestr(f"{name}.tif", raw)
        ring = [(left, bottom), (left, bottom + h), (left + w, bottom + h),
                (left + w, bottom), (left, bottom)]
        feats.append({"type": "Feature",
                      "geometry": {"type": "Polygon",
                                   "coordinates": [[list(to_ll.transform(*p)) for p in ring]]},
                      "properties": {"TILE_NAME": name, "Filename": f"{name}.zip",
                                     "URL": f"https://example.invalid/{name}.zip"}})
    index = tmp_path / "index.geojson"
    index.write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    return TiledDem(tile_dir=tmp_path, index=index)


def test_tiles_form_one_seamless_grid(tmp_path):
    dem = _write_tiles(tmp_path, [(0, 0), (1, 0), (0, 1), (1, 1)])
    assert (dem.cols, dem.rows, dem.res) == (COLS, ROWS, RES)
    assert (dem.n_cols, dem.n_rows) == (2, 2)
    pad = 6
    block = dem.window((0, 0), pad)
    assert block.shape == (ROWS + 2 * pad, COLS + 2 * pad)
    # the margins east and north come from the neighbours; south and west are empty
    assert np.isfinite(block[pad:-pad, -pad:]).all() and np.isfinite(block[:pad, pad:-pad]).all()
    assert np.isnan(block[-pad:, :]).all() and np.isnan(block[:, :pad]).all()


def test_a_tile_that_cannot_be_read_is_named(tmp_path):
    """A download cut short, bytes damaged where the file lies, and a zip with
    something else in it: each must say which of the tiles it is and what to
    do about it.  The first two said only "File is not a zip file" or "Bad
    CRC-32", and the third raised a StopIteration with no message at all.
    Nor is every damaged zip a BadZipFile to the library that reads it: one
    with bytes missing from its middle is an "Invalid argument" from the
    operating system, and one whose directory names a compression method
    that does not exist is "not supported"."""
    dem = _write_tiles(tmp_path, [(c, 0) for c in range(7)])
    whole = (tmp_path / "t1_0.zip").read_bytes()
    (tmp_path / "t1_0.zip").write_bytes(whole[:len(whole) // 2])
    damaged = bytearray((tmp_path / "t2_0.zip").read_bytes())
    damaged[2000:2064] = bytes(b ^ 0xFF for b in damaged[2000:2064])
    (tmp_path / "t2_0.zip").write_bytes(damaged)
    with zipfile.ZipFile(tmp_path / "t3_0.zip", "w") as zf:
        zf.writestr("readme.txt", "not a terrain tile")
    whole = (tmp_path / "t4_0.zip").read_bytes()
    (tmp_path / "t4_0.zip").write_bytes(whole[:1000] + whole[1100:])
    damaged = bytearray((tmp_path / "t5_0.zip").read_bytes())
    directory = damaged.rindex(b"PK\x01\x02")
    damaged[8:10] = damaged[directory + 10:directory + 12] = (99).to_bytes(2, "little")
    (tmp_path / "t5_0.zip").write_bytes(damaged)
    assert dem.read((0, 0)).shape == (ROWS, COLS)
    for key in ((1, 0), (2, 0), (3, 0), (4, 0), (5, 0)):
        with pytest.raises(zipfile.BadZipFile) as err:
            dem.read(key)
        assert str(tmp_path / f"t{key[0]}_0.zip") in str(err.value)
        assert "hk_flat_routes download" in str(err.value)
    # a tile that has gone since the index was read is no damaged zip, and
    # the system's own error already names it
    (tmp_path / "t6_0.zip").unlink()
    with pytest.raises(FileNotFoundError, match="t6_0.zip"):
        dem.read((6, 0))
    # and a sampler that comes to such a tile stops there, with the same words
    with pytest.raises(zipfile.BadZipFile, match="t1_0.zip"):
        DemSampler(dem).sample([X0 + 150.0], [Y0 + 40.0])


def test_sampling_is_exact_on_a_plane_and_seamless_across_tiles(tmp_path):
    dem = _write_tiles(tmp_path, [(0, 0), (1, 0), (0, 1), (1, 1)])
    w, h = COLS * RES, ROWS * RES
    rng = np.random.default_rng(3)
    # keep clear of the outer edge of the tile set, where there is no data beyond
    x = X0 + rng.uniform(10, 2 * w - 10, 500)
    y = Y0 + rng.uniform(10, 2 * h - 10, 500)
    for sampler in (DemSampler(dem, smooth=False), DemSampler(dem, sigma_m=1.5)):
        z = sampler.sample(x, y)
        # a Gaussian leaves a plane a plane, so both must reproduce it
        assert np.nanmax(np.abs(z - _surface(x, y))) < 1e-3
    # straight across the seam between the two lower tiles
    xs = X0 + w + np.linspace(-3, 3, 25)
    z = DemSampler(dem, sigma_m=1.5).sample(xs, np.full_like(xs, Y0 + 30.0))
    assert np.allclose(np.diff(z), 0.10 * np.diff(xs), atol=1e-3)


def test_points_off_the_tiles_or_over_the_sea_have_no_elevation(tmp_path):
    dem = _write_tiles(tmp_path, [(0, 0)], sea=lambda x, y: x > X0 + 60.0)
    s = DemSampler(dem, sigma_m=1.5)
    z = s.sample([X0 + 20.0, X0 + 80.0, X0 - 50.0, X0 + 500.0],
                 [Y0 + 40.0, Y0 + 40.0, Y0 + 40.0, Y0 + 40.0])
    assert np.isfinite(z[0]) and np.isnan(z[1:]).all()


def test_the_coast_is_not_dragged_toward_a_fill_value(tmp_path):
    """Level ground at 4 m beside the sea must still read 4 m at the seawall.

    Upstream fills nodata with the mean elevation before smoothing, which is
    harmless where the DEM carries values over water; here the sea is nodata
    and a fill would raise or lower every waterfront street."""
    dem = _write_tiles(tmp_path, [(0, 0)], surface=lambda x, y: np.full_like(x, 4.0),
                       sea=lambda x, y: x > X0 + 60.0)
    s = DemSampler(dem, sigma_m=3.0)
    xs = X0 + np.array([30.0, 55.0, 58.0, 59.5])
    z = s.sample(xs, np.full_like(xs, Y0 + 40.0))
    assert np.allclose(z, 4.0, atol=1e-4)


def test_the_unfiltered_coast_is_not_dragged_toward_zero(tmp_path):
    """Without the Gaussian nothing has spread the land over the cells beside
    it, and a sample within half a cell of the seawall once mixed the sea in
    as 0 m: a tenth of a metre from the edge, 4 m of ground read 2.8 m."""
    dem = _write_tiles(tmp_path, [(0, 0)], surface=lambda x, y: np.full_like(x, 4.0),
                       sea=lambda x, y: x > X0 + 60.0)
    xs = X0 + np.array([30.0, 59.0, 59.6, 59.9, 59.99])
    ys = np.full_like(xs, Y0 + 40.1)
    for s in (DemSampler(dem, smooth=False), DemSampler(dem, sigma_m=0.0),
              DemSampler(dem, sigma_m=1.5)):
        assert np.allclose(s.sample(xs, ys), 4.0, atol=1e-4)
    # on a slope, what is left of the stencil still reads the ground there
    dem = _write_tiles(tmp_path, [(0, 0)], sea=lambda x, y: x > X0 + 60.0)
    z = DemSampler(dem, smooth=False).sample(xs, ys)
    assert np.max(np.abs(z - _surface(xs, ys))) < 0.03


def test_the_overview_averages_the_tiles_down(tmp_path, monkeypatch):
    import rasterio
    dem = _write_tiles(tmp_path, [(0, 0), (1, 0)])
    monkeypatch.setattr(elevation, "DEM_OVERVIEW", tmp_path / "overview.tif")
    monkeypatch.setattr(elevation, "TiledDem", lambda cache_tiles=1: dem)
    path = elevation.build_dem_overview(force=True, res_m=10.0)
    with rasterio.open(path) as src:
        a = src.read(1)
        assert src.crs.to_epsg() == 2326 and src.res == (10.0, 10.0)
        assert a.shape == (8, 20)
        # the top-left cell is the mean of the plane over its 10 m square
        assert abs(a[0, 0] - _surface(X0 + 5.0, Y0 + 75.0)) < 1e-3


def test_the_overview_is_rebuilt_when_tiles_arrive(tmp_path, monkeypatch):
    """An overview built while the download was part done must not stay
    cropped to those tiles once the rest are on disk."""
    import os
    import time

    import rasterio
    tiles = tmp_path / "dem"; tiles.mkdir()
    index = tiles / "index.geojson"
    overview = tmp_path / "processed" / "overview.tif"
    opened = []

    def open_tiles(cache_tiles=1):
        opened.append(1)
        return TiledDem(tile_dir=tiles, index=index, cache_tiles=cache_tiles)

    monkeypatch.setattr(elevation, "DEM_OVERVIEW", overview)
    monkeypatch.setattr(elevation, "DEM_DIR", tiles)
    monkeypatch.setattr(elevation, "DEM_INDEX_GEOJSON", index)
    monkeypatch.setattr(elevation, "TiledDem", open_tiles)

    def width():
        with rasterio.open(elevation.build_dem_overview()) as src:
            return src.width

    def backdate(path, seconds):
        t = time.time() - seconds
        os.utime(path, (t, t))

    _write_tiles(tiles, [(0, 0)])
    assert width() == 10
    assert width() == 10 and len(opened) == 1        # cached: no tile was opened
    # the rest of the download lands a minute later, whatever the clock's grain
    backdate(tiles / "t0_0.zip", 120); backdate(overview, 60)
    assert width() == 10 and len(opened) == 1
    _write_tiles(tiles, [(0, 0), (1, 0)])
    assert width() == 20 and len(opened) == 2
    assert width() == 20 and len(opened) == 2        # and cached again
    assert elevation.build_dem_overview(force=True) == overview and len(opened) == 3

    # A tile deleted to win back the disk, and a file that is no tile left in
    # the directory: both change the directory and neither is a tile arriving.
    # Rebuilt now, the overview would be cropped to the one tile that is left.
    (tiles / "t1_0.zip").unlink()
    (tiles / ".DS_Store").write_bytes(b"\0")
    assert width() == 20 and len(opened) == 3
    # a tile that does arrive while the index is gone cannot be placed: the
    # overview on disk stands, where a forced rebuild fails
    index_text = index.read_text()
    index.unlink()
    backdate(tiles / "t0_0.zip", -60)
    assert width() == 20 and len(opened) == 4        # tried, and kept
    with pytest.raises(FileNotFoundError):
        elevation.build_dem_overview(force=True)
    index.write_text(index_text)

    # every tile gone: there is nothing to build from
    (tiles / "t0_0.zip").unlink()
    assert width() == 20
    with pytest.raises(FileNotFoundError):           # a forced rebuild says so
        elevation.build_dem_overview(force=True)
    overview.unlink()
    with pytest.raises(FileNotFoundError):           # and so does having neither
        elevation.build_dem_overview()


# ----------------------------------------------------------- terrace search
def test_the_search_finds_a_path_cut_into_a_hillside():
    """A level terrace 4 m wide, mapped 2 m too far downhill."""
    offs = lateral_offsets()
    # ground across the path at each offset from the *mapped* line:
    # true path centre is at +2 m, level at 100 m; 60% slope either side
    d = offs - 2.0
    cross = np.where(np.abs(d) <= 2.0, 100.0, 100.0 + 0.6 * (np.abs(d) - 2.0) * np.sign(d))
    Z = np.tile(cross, (12, 1))
    k = flattest_offset(Z, offs, penalty=0.04)
    assert np.all(np.abs(offs[k] - 2.0) <= 1.0)
    assert np.allclose(Z[np.arange(12), k], 100.0)


def test_the_search_stays_put_on_level_ground():
    offs = lateral_offsets()
    Z = np.full((9, offs.size), 5.0) + 0.01 * offs[None, :]      # a 1% crossfall
    k = flattest_offset(Z, offs, penalty=0.04)
    assert np.all(offs[k] == 0.0)


def test_the_search_keeps_to_the_path_at_the_ends_of_an_edge():
    """The same terrace, but at the edge's first and last samples the path
    has a 15% crossfall and a level shelf lies further down the slope.  The
    end samples are the ones shared with the next edge; a hold that only
    reaches the middle of the edge let them hop seven metres to the shelf."""
    offs = lateral_offsets()
    d = offs - 2.0
    cross = np.where(np.abs(d) <= 2.0, 100.0, 100.0 + 0.6 * (np.abs(d) - 2.0) * np.sign(d))
    Z = np.tile(cross, (12, 1))
    for row in (0, 11):
        Z[row] = cross + 0.15 * d * (np.abs(d) <= 2.0)
        Z[row, :4] = Z[row, 3]
    k = flattest_offset(Z, offs, penalty=0.04)
    assert np.all(offs[k] == 2.0)


def test_the_search_reads_a_short_edge_along_one_line():
    """Three samples on a 50% hillside with no ledge: the slope only pauses
    for a metre, at a different place across each sample.  Taken sample by
    sample the profile went five metres down the slope, one up and six up,
    and a level edge rose 5 m; an edge this short had no hold at all."""
    offs = lateral_offsets()
    Z = np.tile(100.0 + 0.5 * offs, (3, 1))
    for row, at in enumerate((-5.0, 1.0, 5.0)):
        Z[row, offs > at] -= 0.5
    k = flattest_offset(Z, offs, penalty=0.04)
    z = Z[np.arange(3), k]
    assert np.ptp(offs[k]) <= 1.0
    assert abs(z[-1] - z[0]) <= 0.5


def test_edges_end_to_end_are_searched_as_each_would_be_alone(monkeypatch):
    """All the edges of a build go through the search in one table, and are
    taken from it a block of edges at a time."""
    offs = lateral_offsets()
    rng = np.random.default_rng(5)
    sizes = np.array([3, 9, 3, 4, 21, 5, 3, 12])
    starts = np.r_[0, np.cumsum(sizes)[:-1]]
    Z = 50.0 + rng.normal(0.0, 1.0, (sizes.sum(), offs.size)).cumsum(axis=1)
    Z[rng.random(Z.shape) < 0.03] = np.nan
    Z[4] = np.nan                                    # a sample with no data at all
    together = flattest_offset(Z, offs, 0.04, starts)
    alone = np.concatenate([flattest_offset(Z[a:a + n], offs, 0.04)
                            for a, n in zip(starts, sizes)])
    assert np.array_equal(together, alone)
    monkeypatch.setattr(elevation, "SEARCH_BLOCK_EDGES", 3)      # blocks of 3, 3 and 2
    assert np.array_equal(flattest_offset(Z, offs, 0.04, starts), alone)
    picked = Z[np.arange(Z.shape[0]), together]
    assert np.isfinite(np.delete(picked, 4)).all()   # never a hole when there is ground


def test_edges_meeting_at_a_node_share_one_height():
    import pandas as pd
    edges = pd.DataFrame({"u": ["a", "b", "b", "b", "e"], "v": ["b", "c", "d", "f", "b"],
                          "is_structure": [False, False, False, True, False]})
    offsets = np.array([0, 3, 6, 9, 12, 15])
    cat_z = np.array([10.0, 11.0, 12.0,   13.5, 14.0, 15.0,   12.4, 9.0, 8.0,
                      30.0, 30.0, 30.0,   7.0, 7.5, np.nan])
    out = elevation._one_height_per_node(edges, cat_z, offsets)
    # at b: 12.0, 13.5 and 12.4 on the ground; the deck and the hole stand aside
    assert out[2] == out[3] == out[6] == 12.4
    assert out[9] == 30.0 and np.isnan(out[14])
    changed = np.flatnonzero(out != cat_z)
    assert set(changed) == {2, 3, 14}                # NaN != NaN; nothing else moved


# ---------------------------------------------------------------- abutments
def test_a_bridge_end_is_levelled_from_its_approach():
    """A level path at 122 m reaches a footbridge over a gully.  The model
    under the last ten metres is already falling into the gully."""
    import geopandas as gpd
    from shapely.geometry import LineString
    edges = gpd.GeoDataFrame({
        "u": ["a", "b"], "v": ["b", "c"],
        "is_structure": [False, True],
    }, geometry=[LineString([(0, 0), (60, 0)]), LineString([(60, 0), (80, 0)])],
        crs="EPSG:2326")
    dist = np.concatenate([np.arange(0, 61, 5.0), np.array([0.0, 10.0, 20.0])])
    z_path = np.full(13, 122.0)
    z_path[-2:] = [120.0, 117.0]                 # the dip at the abutment
    cat_z = np.concatenate([z_path, [117.0, 110.0, 117.0]])
    offsets = np.array([0, 13, 16])
    out, node_elev = elevation._correct_abutments(
        edges, dist, cat_z, offsets, {"a": 122.0, "b": 117.0, "c": 117.0})
    assert abs(node_elev["b"] - 122.0) < 1e-6
    assert np.allclose(out[:13], 122.0)          # the approach is level again
    assert node_elev["a"] == 122.0               # nothing else moved


def test_a_sound_bridge_end_is_left_alone():
    import geopandas as gpd
    from shapely.geometry import LineString
    edges = gpd.GeoDataFrame({"u": ["a", "b"], "v": ["b", "c"],
                              "is_structure": [False, True]},
                             geometry=[LineString([(0, 0), (60, 0)]),
                                       LineString([(60, 0), (80, 0)])], crs="EPSG:2326")
    dist = np.concatenate([np.arange(0, 61, 5.0), np.array([0.0, 10.0, 20.0])])
    z_path = 100.0 + 0.05 * np.arange(0, 61, 5.0)        # a steady 5% climb
    cat_z = np.concatenate([z_path, [103.0, 103.5, 104.0]])
    out, node_elev = elevation._correct_abutments(
        edges, dist, cat_z, np.array([0, 13, 16]), {"a": 100.0, "b": 103.0, "c": 104.0})
    assert node_elev["b"] == 103.0 and np.array_equal(out, cat_z)


def test_the_running_median_leaves_a_steady_slope_exactly_as_it_is():
    z = 50.0 + 0.08 * 5.0 * np.arange(40)        # an 8% climb sampled every 5 m
    assert np.array_equal(elevation._despike(z, 5.0), z)


def test_the_running_median_fills_most_of_a_gully():
    """An untagged culvert 15 m wide and 4 m deep on a level path vanishes;
    on an 8% climb what is left of it is about a metre."""
    level = np.full(40, 120.0)
    dipped = level.copy(); dipped[18:21] -= 4.0
    assert np.array_equal(elevation._despike(dipped, 5.0), level)
    z = 50.0 + 0.08 * 5.0 * np.arange(40)
    dipped = z.copy(); dipped[18:21] -= 4.0
    out = elevation._despike(dipped, 5.0)
    assert np.max(np.abs(out - z)) <= 1.3
    assert np.all(np.diff(out) >= 0)             # and it never runs downhill


# ------------------------------------------------- decks, piers and landings
def _chain(lengths, is_structure=True):
    """Edges n0-n1-...-nk of the given lengths, as the solver reads them."""
    import pandas as pd
    names = [f"n{i}" for i in range(len(lengths) + 1)]
    return names, pd.DataFrame({"u": names[:-1], "v": names[1:],
                                "length_m": np.asarray(lengths, dtype="float64"),
                                "is_structure": is_structure})


def test_a_long_viaduct_is_one_straight_ramp():
    """Decks of every length from one metre to 400 between ground at 10 m and
    at 40 m: 121 spans, and the 120 nodes between them have only the water
    under the deck to read.  Relaxed for 400 sweeps from that reading, this
    chain was left up to 35 m from the answer; solved, it is the ramp to the
    last digit."""
    rng = np.random.default_rng(7)
    lengths = rng.choice([1.0, 3.0, 25.0, 120.0, 400.0], size=121)
    names, edges = _chain(lengths)
    node_elev = dict.fromkeys(names, 2.0)            # the water under the deck
    node_elev[names[0]], node_elev[names[-1]] = 10.0, 40.0
    out = elevation._solve_structure_nodes(edges, node_elev, {names[0], names[-1]})
    along = np.concatenate([[0.0], np.cumsum(lengths)])
    ramp = 10.0 + 30.0 * along / along[-1]
    assert np.allclose([out[n] for n in names], ramp, atol=1e-8)


def test_a_structure_that_joins_no_measured_ground_gets_no_invented_height():
    """A footbridge b-c-d whose landings b and d are on nodata, in a town at
    5 m, while a bridge elsewhere stands at 100 m.  The structure pass has
    nothing to hang it from and must say so, whatever the terrain under the
    deck reads; the pass over the whole graph then finds the streets."""
    import pandas as pd
    nan = np.nan
    edges = pd.DataFrame({
        "u": ["a", "b", "c", "d", "p", "q", "r"],
        "v": ["b", "c", "d", "e", "q", "r", "s"],
        "length_m": [50.0, 20.0, 20.0, 50.0, 30.0, 30.0, 30.0],
        "is_structure": [False, True, True, False, False, True, True],
    })
    node_elev = {"a": 5.0, "b": nan, "c": 1.5, "d": nan, "e": 5.0,
                 "p": 100.0, "q": 100.0, "r": 60.0, "s": 60.0}
    reliable = {"a", "e", "p", "q"}                  # on the ground, with a value
    first = elevation._solve_structure_nodes(edges, node_elev, reliable)
    assert all(np.isnan(first[n]) for n in "bcd")
    # the bridge that does meet the ground is solved from where it meets it
    assert first["r"] == pytest.approx(100.0) and first["s"] == pytest.approx(100.0)
    reliable |= {n for n, z in first.items() if np.isfinite(z)}
    final = elevation._solve_structure_nodes(edges, first, reliable, subset=edges,
                                             label="DEM-void", level=node_elev)
    assert [final[n] for n in "bcd"] == pytest.approx([5.0, 5.0, 5.0])
    assert final["a"] == 5.0 and final["q"] == 100.0


def test_decks_joined_to_no_street_are_laid_level():
    """A walkway system mapped with no link to the ground has no height to
    take from anywhere.  It is laid level on the terrain under its nodes,
    not left to follow that terrain span by span."""
    names, edges = _chain([20.0, 5.0, 30.0])
    under = dict(zip(names, [4.0, 7.0, 6.0, 5.0]))
    first = elevation._solve_structure_nodes(edges, under, set())
    assert all(np.isnan(first[n]) for n in names)
    final = elevation._solve_structure_nodes(edges, first, set(), subset=edges,
                                             level=under)
    assert [final[n] for n in names] == pytest.approx([5.5] * 4)
    # and with nothing under it either, it stays without an elevation
    final = elevation._solve_structure_nodes(edges, first, set(), subset=edges,
                                             level=dict.fromkeys(names, np.nan))
    assert all(np.isnan(final[n]) for n in names)


def _edges(rows):
    """A small network: (u, v, [(x, y), ...] in metres from the sheet's
    corner, is_structure, segment) per edge; consecutive edges of one
    segment are its parts."""
    import geopandas as gpd
    from shapely.geometry import LineString
    geoms = [LineString([(X0 + x, Y0 + y) for x, y in r[2]]) for r in rows]
    seg = [r[4] if len(r) > 4 else f"s{i}" for i, r in enumerate(rows)]
    part = [sum(s == seg[i] for s in seg[:i]) for i in range(len(rows))]
    return gpd.GeoDataFrame({
        "edge_id": np.arange(len(rows)), "segment_id": seg, "part": part,
        "u": [r[0] for r in rows], "v": [r[1] for r in rows],
        "length_m": [g.length for g in geoms],
        "is_structure": [bool(r[3]) for r in rows]}, geometry=geoms, crs="EPSG:2326")


def _profiles(tmp_path, monkeypatch, dem, edges):
    monkeypatch.setattr(elevation, "PROFILES_NPZ", tmp_path / "profiles.npz")
    return elevation.sample_edge_profiles(edges, DemSampler(dem), force=True)


def test_a_pier_stands_at_the_height_of_its_waterfront(tmp_path, monkeypatch):
    """A street at 5 m runs out to the seawall and a pier carries on over the
    sea, which is nodata: neither the pier's landing nor anything on it has
    a terrain value.  On a hill behind, a bridge stands at 100 m.  The pier
    once took the mean height of every other bridge end, here 100 m."""
    dem = _write_tiles(
        tmp_path, [(0, 0)],
        surface=lambda x, y: np.where(y < Y0 + 50.0, 5.0, 100.0),
        sea=lambda x, y: (x > X0 + 60.0) & (y < Y0 + 50.0))
    edges = _edges([
        ("a0", "a", [(5, 20), (53, 20)], False),
        ("a", "b", [(53, 20), (60.5, 20)], False),   # too short to level b from
        ("b", "c", [(60.5, 20), (80, 20)], True),
        ("c", "d", [(80, 20), (95, 20)], True),
        ("p", "q", [(5, 65), (30, 65)], False),
        ("q", "r", [(30, 65), (50, 65)], True),
        ("r", "s", [(50, 65), (80, 65)], False),
    ])
    prof = _profiles(tmp_path, monkeypatch, dem, edges)
    for eid in (0, 1, 2, 3):
        assert np.allclose(prof[eid][1], 5.0, atol=0.01), eid
    for eid in (4, 5, 6):
        assert np.allclose(prof[eid][1], 100.0, atol=0.01), eid


def test_steps_beyond_the_shore_stand_at_the_shore(tmp_path, monkeypatch):
    """A 52 m street comes down a 10% slope from 11 m to the water, and its
    last node is over the sea, with a short flight of landing steps beyond.
    Neither node has terrain under it.  They once took the height of the
    street's far end, 11 m, and the steps were left hanging 5 m over a shore
    that the street's own samples put at 6.3 m."""
    dem = _write_tiles(tmp_path, [(0, 0)],
                       surface=lambda x, y: 12.0 - 0.10 * (x - X0),
                       sea=lambda x, y: x > X0 + 60.0)
    edges = _edges([
        ("a", "b", [(10, 40), (62, 40)], False),
        ("b", "c", [(62, 40), (70, 40)], False),
    ])
    prof = _profiles(tmp_path, monkeypatch, dem, edges)
    street, steps = prof[0][1], prof[1][1]
    assert street[0] == pytest.approx(11.0, abs=0.1)
    # the last sample with terrain under it is 4.7 m short of the node, at 6.27 m
    assert street[-1] == pytest.approx(6.27, abs=0.15)
    assert np.allclose(steps, 6.27, atol=0.15)


def test_a_landing_levelled_from_its_approach_keeps_that_level(tmp_path, monkeypatch):
    """A street at 5 m ends at the edge of a nullah, on a cell with no data,
    and a footbridge crosses to a bank at 20 m.  The landing's height comes
    from the street's last 35 m; the deck must start there, not be handed
    the height of its other end, which once pulled the street up to 12.5 m."""
    dem = _write_tiles(
        tmp_path, [(0, 0)],
        surface=lambda x, y: np.where(x < X0 + 62.0, 5.0, 20.0),
        sea=lambda x, y: (x > X0 + 60.0) & (x < X0 + 64.0))
    edges = _edges([
        ("a", "b", [(5, 40), (60.5, 40)], False),
        ("b", "c", [(60.5, 40), (75, 40)], True),
        ("c", "d", [(75, 40), (98, 40)], False),
    ])
    prof = _profiles(tmp_path, monkeypatch, dem, edges)
    assert np.allclose(prof[0][1], 5.0, atol=0.01)
    assert prof[1][1][0] == pytest.approx(5.0, abs=0.01)
    assert prof[1][1][-1] == pytest.approx(20.0, abs=0.01)
    assert np.all(np.diff(prof[1][1]) > 0)           # one ramp between the two
    assert np.allclose(prof[2][1], 20.0, atol=0.01)


# ---------------------------------------------------------- the profile cache
def test_cached_profiles_are_not_read_for_another_network(tmp_path, monkeypatch):
    """A forced build stopped while the terrain was being read leaves the new
    network beside the old profiles.  Edge ids are row numbers, so the old
    profiles answered to the new edges, id for id, and without ``--force``
    they were handed out as theirs."""
    dem = _write_tiles(tmp_path, [(0, 0)])
    old = _edges([("a", "b", [(10, 40), (50, 40)], False),
                  ("b", "c", [(50, 40), (90, 40)], False)])
    new = _edges([("a", "b", [(10, 40), (30, 40)], False),
                  ("b", "c", [(30, 40), (30, 70)], False)])
    first = _profiles(tmp_path, monkeypatch, dem, old)

    class NoTiles:
        def sample(self, x, y):
            raise AssertionError("the terrain was read again")

    # the same network is served from the cache, and nothing is sampled
    again = elevation.sample_edge_profiles(old, NoTiles())
    assert all(np.array_equal(again[k][1], first[k][1]) for k in first)
    # the same ids on other streets are not
    with pytest.raises(AssertionError, match="read again"):
        elevation.sample_edge_profiles(new, NoTiles())
    prof = elevation.sample_edge_profiles(new, DemSampler(dem))
    assert [prof[k][0][-1] for k in (0, 1)] == pytest.approx([20.0, 30.0])
    assert prof[1][1][-1] - prof[1][1][0] == pytest.approx(0.04 * 30.0, abs=0.01)


# ------------------------------------------------------------ profile filters
def test_a_short_piece_of_a_steady_street_keeps_the_street_gradient(tmp_path, monkeypatch):
    """A 10% street cut into pieces of 60 m, 1 m and 60 m.  Every piece has
    at least three samples, so the short one's are half a metre apart; taken
    for 5 m apart by the filters they gave it several times the gradient of
    the street it is a metre of."""
    dem = _write_tiles(tmp_path, [(0, 0), (1, 0)])
    cuts = [20.0, 80.0, 81.0, 141.0]
    edges = _edges([(f"n{i}", f"n{i + 1}", [(cuts[i], 40), (cuts[i + 1], 40)],
                     False, "street") for i in range(3)])
    prof = _profiles(tmp_path, monkeypatch, dem, edges)
    for eid in range(3):
        d, z = prof[eid]
        assert (z[-1] - z[0]) / d[-1] == pytest.approx(0.10, abs=0.005), eid


def test_filtering_reads_an_uneven_run_at_even_stations():
    """A gully 15 m wide on a level run is taken out by the running median
    wherever the run's cuts fall, and even samples are filtered as before."""
    even = np.arange(0.0, 200.1, 5.0)
    z = np.full(even.size, 120.0); z[18:21] -= 4.0
    assert np.allclose(elevation._filter_run(even, z, 5.0), 120.0)
    # to the last digit, in fact: the stations of an even run are its samples
    rolling = 120.0 + 3.0 * np.sin(even / 17.0) + 0.4 * np.cos(even / 3.0)
    as_before = elevation._smooth_profile(elevation._despike(rolling, 5.0), 5.0)
    assert not np.allclose(as_before, rolling, atol=0.1)
    assert np.array_equal(elevation._filter_run(even, rolling, 5.0), as_before)
    # the same ground, with three pieces a metre long cut into the run
    d = np.unique(np.concatenate([even, 40 + np.arange(7) * 0.5,
                                  93 + np.arange(7) * 0.5]))
    z = np.where((d >= 90) & (d <= 100), 116.0, 120.0)
    assert np.allclose(elevation._filter_run(d, z, 5.0), 120.0, atol=0.2)
    # a run shorter than the spacing has nothing to filter
    short = np.array([0.0, 1.5, 3.0]); zs = np.array([5.0, 5.4, 5.1])
    assert np.array_equal(elevation._filter_run(short, zs, 5.0), zs)


# ------------------------------------------------- the terrace search, whole
def test_a_side_path_does_not_pull_a_junction_off_the_terrace(tmp_path, monkeypatch):
    """A level path 4 m wide is cut into a 60% hillside at 100 m and mapped
    5 m down the slope, where the ground is at 98.2 m.  Both of its edges
    find it; the path that leaves the junction straight down the slope has
    no terrace to find and reads the junction where it is mapped.  Averaged
    in, that reading sank the junction, and the level path with it, by
    0.6 m.  A side path long enough to be smoothed still takes 0.14 m off
    the junction, where its own profile is rounded into the slope."""
    def hillside(x, y):
        v = y - Y0
        return 100.0 + 0.6 * (np.clip(v - 42.0, 0, None) - np.clip(38.0 - v, 0, None))

    dem = _write_tiles(tmp_path, [(0, 0)], surface=hillside)
    for down, slack in ((10.0, 0.05), (30.0, 0.2)):
        edges = _edges([
            ("a", "b", [(10, 35), (50, 35)], False),
            ("b", "d", [(50, 35), (90, 35)], False),
            ("b", "c", [(50, 35), (50, 35 - down)], False),
        ])
        prof = _profiles(tmp_path, monkeypatch, dem, edges)
        assert np.allclose(prof[0][1], 100.0, atol=slack), down
        assert np.allclose(prof[1][1], 100.0, atol=slack), down
        assert prof[2][1][0] == pytest.approx(100.0, abs=slack), down
        assert prof[2][1][-1] == pytest.approx(98.2 - 0.6 * down, abs=0.1 + slack), down


# ------------------------------------------------------------- the real build
def test_no_node_stands_clear_of_all_the_ground_around_it():
    """Against a real build (skipped without one; ``HKFR_RUN_DIR`` selects it).

    A street node is on the ground, so the terrain within 30 m of it cannot
    all be far below it; and a deck is the smoothest surface between its
    landings, so it is never higher than the highest ground near a landing
    of its own run of bridges and tunnels.  (The deck's own surroundings
    will not do: 27 sound nodes stand more than 8 m over everything within
    30 m of them, up to 23 m on the Kwai Tsing Road viaduct.)  A node out
    over the water is judged by the land nearest to it: 121 nodes, pier
    heads for the most part, have no land within 30 m.  Piers that were
    given the mean height of every bridge end in the territory stood 34 m
    over the sea front on this measure, and 102 nodes in all were more than
    5 m up; solved from the streets they join, none is 3 m up.
    """
    import geopandas as gpd
    import pandas as pd
    import rasterio
    from scipy.ndimage import distance_transform_edt, maximum_filter
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    from hk_flat_routes.config import PROCESSED_DIR

    edges_path = PROCESSED_DIR / "edges_raw.parquet"
    for path in (edges_path, elevation.PROFILES_NPZ, elevation.DEM_OVERVIEW):
        if not path.exists():
            pytest.skip(f"no {path.name} in {PROCESSED_DIR}")
    edges = gpd.read_parquet(edges_path, columns=["edge_id", "u", "v", "is_structure",
                                                  "geometry"])
    with np.load(elevation.PROFILES_NPZ, allow_pickle=False) as npz:
        ids, off, elev = npz["edge_ids"], npz["offsets"], npz["elev"]
    if not np.array_equal(ids, edges["edge_id"].to_numpy()):
        pytest.skip("the cached profiles are not those of the cached network")
    names, x, y, solid = elevation.node_points(edges)
    index = {n: i for i, n in enumerate(names)}
    iu = edges["u"].map(index).to_numpy(); iv = edges["v"].map(index).to_numpy()
    n = len(names)
    z = np.full(n, np.nan)
    z[iu] = elev[off[:-1]]; z[iv] = elev[off[1:] - 1]

    # The highest 10 m cell among the one a node is in and three either way.
    # A cell of sea stands for the land nearest to it, or a pier head with
    # nothing but water around it would have no ground to be judged by.
    with rasterio.open(elevation.DEM_OVERVIEW) as src:
        dem = src.read(1)
        nearest = distance_transform_edt(dem == src.nodata, return_distances=False,
                                         return_indices=True)
        rows, cols = rasterio.transform.rowcol(src.transform, x, y)
    top = maximum_filter(dem[tuple(nearest)], size=7)
    rows, cols = np.asarray(rows), np.asarray(cols)
    inside = (rows >= 0) & (rows < dem.shape[0]) & (cols >= 0) & (cols < dem.shape[1])
    ground = np.full(n, np.nan)
    ground[inside] = top[rows[inside], cols[inside]]

    # for a node on a structure, the highest ground near any landing of its
    # run (near any of its nodes, where a run is mapped with no landing)
    st = edges["is_structure"].to_numpy()
    _, run = connected_components(coo_matrix((np.ones(st.sum()), (iu[st], iv[st])),
                                             shape=(n, n)), directed=False)
    on_structure = np.zeros(n, dtype=bool)
    on_structure[iu[st]] = True; on_structure[iv[st]] = True
    landing = on_structure & np.fromiter((m in solid for m in names), dtype=bool, count=n)

    def highest(values):
        return pd.Series(values).groupby(run).transform("max").to_numpy()

    of_run = highest(np.where(landing, ground, np.nan))
    of_run = np.where(np.isfinite(of_run), of_run, highest(ground))
    ground = np.where(on_structure, of_run, ground)

    clear = z - ground
    judged = np.isfinite(clear)
    assert judged.sum() > 0.999 * n
    worst = np.argsort(-np.where(judged, clear, -np.inf))[:5]
    high = judged & (clear > 5.0)
    assert not high.any(), (
        f"{int(high.sum())} nodes stand more than 5 m above all the ground around "
        f"them; the worst: " + "; ".join(
            f"{z[i]:.1f} m over {ground[i]:.1f} m at ({x[i]:.0f}, {y[i]:.0f})"
            for i in worst))
