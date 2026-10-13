"""The static maps' relief and the terrain overview it is read from."""
import numpy as np
import pytest

CRS = "EPSG:2326"
FRAME = (829000, 814000, 835000, 819000)


@pytest.fixture
def overview(tmp_path, monkeypatch):
    """A small overview raster standing in for ``dem_hk_10m.tif``."""
    import rasterio
    from rasterio.transform import from_origin
    import hk_flat_routes.elevation as E

    path = tmp_path / "dem_hk_10m.tif"
    dem = np.tile(np.arange(400, dtype="float32"), (300, 1))
    with rasterio.open(path, "w", driver="GTiff", height=300, width=400, count=1,
                       dtype="float32", crs=CRS, nodata=-9999.0,
                       transform=from_origin(830000, 818000, 10, 10)) as dst:
        dst.write(dem, 1)
    monkeypatch.setattr(E, "DEM_MOSAIC", path)
    return path


def test_the_overview_is_asked_for_even_when_one_is_on_disk(overview, monkeypatch):
    """The relief used to be read from whatever overview existed.  One built
    while the tile download was part done stayed cropped to those tiles in
    both static maps, since nothing asked ``build_dem_overview`` whether the
    tiles had changed."""
    import hk_flat_routes.elevation as E
    import hk_flat_routes.viz_static as V

    calls = []
    monkeypatch.setattr(E, "build_dem_overview",
                        lambda *a, **k: calls.append((a, k)) or overview)
    assert V._load_hillshade(FRAME, max_px=100) is not None
    # once, and not forced: the overview decides whether it is stale
    assert calls == [((), {})]


def test_an_overview_that_cannot_be_rebuilt_is_still_drawn(overview, monkeypatch, caplog):
    """The processed data kept and the tiles deleted: there is nothing to
    rebuild from, and the overview on disk is the relief."""
    import hk_flat_routes.elevation as E
    import hk_flat_routes.viz_static as V

    def no_tiles(*a, **k):
        raise FileNotFoundError("no terrain tiles")
    monkeypatch.setattr(E, "build_dem_overview", no_tiles)
    with caplog.at_level("WARNING", logger="hk_flat_routes.viz_static"):
        got = V._load_hillshade(FRAME, max_px=100)
    assert got is not None and got[2].any()
    assert "drawn from the one on disk" in caplog.text


def test_no_overview_and_no_tiles_means_no_relief(overview, monkeypatch, caplog):
    import hk_flat_routes.elevation as E
    import hk_flat_routes.viz_static as V

    def no_tiles(*a, **k):
        raise FileNotFoundError("no terrain tiles")
    monkeypatch.setattr(E, "build_dem_overview", no_tiles)
    overview.unlink()
    with caplog.at_level("WARNING", logger="hk_flat_routes.viz_static"):
        assert V._load_hillshade(FRAME, max_px=100) is None
    assert "no terrain overview for the relief" in caplog.text
