"""Elevation sampling from the CEDD 2020 LiDAR terrain model (0.5 m).

Accuracy of the climbing metrics rests entirely on this module, so the
method is spelled out here and mirrored in the README.

1.  **Tiles, not a mosaic.**  The terrain model is 3,253 tiles of
    750 m x 600 m at 0.5 m, in HK1980 Grid (EPSG:2326), which is also the
    CRS of every length and slope computation, so no raster reprojection is
    ever performed and no resampling error is introduced.  The whole
    territory at 0.5 m is 12 billion cells, far more than fits in memory,
    so the tiles are read on demand (``TiledDem``) and every sample point
    is served from its own tile plus a margin of its neighbours.

2.  **Noise suppression (spatial).**  A Gaussian filter of sigma = 1.5 m
    (``ElevationConfig.dem_sigma_m``; upstream uses 3 m) is applied to the
    DEM before sampling.  A bare-earth lidar DEM still contains
    decimetre-scale artefacts from kerbs, parked vehicles, vegetation
    misclassification and interpolation over occlusions (in Hong Kong,
    under every building and flyover).  A 1.5 m sigma is three cells of
    the model: wide enough to damp those, narrow enough that a hillside
    road four to eight metres wide is not averaged with the wall above it
    and the drop below, and far narrower than the block scale on which
    real street grade varies.  The filter is a *normalised* convolution:
    cells with no data (the sea) carry no weight instead of a fill value,
    so a waterfront promenade is not dragged toward some territory-wide
    mean height.  The margin read from neighbouring tiles is wider than
    the filter's reach, so the result is seamless across tile edges.

3.  **Sampling.**  Each edge geometry is densified at 5 m spacing
    (endpoints always included, minimum 3 samples) and sampled with bilinear
    interpolation.  Where the ground falls away across an edge, the mapped
    centreline is not good enough to say which side of a retaining wall
    the road is on, and each sample is taken at the flattest spot within
    6 m of it instead, along the line of such spots that costs least to
    follow from one end of the edge to the other (``flattest_offset``).

4.  **Structures.**  Where an edge is flagged ``is_bridge`` or ``is_tunnel``
    the DEM describes the ground or water surface under the deck, or the
    hill over the tunnel, not the way itself.  Elevations on such edges are
    replaced by a linear ramp between their endpoints, and an endpoint on
    the ground is levelled from the approach to it where the model is
    already falling away under the deck (``_correct_abutments``).  Endpoint
    elevations that are themselves unreliable (nodes touched only by
    structure edges, e.g. mid-viaduct) are recovered by solving a discrete
    Laplace problem over the structure sub-graph with the reliable nodes
    as boundary conditions -- i.e. the deck is modelled as the smoothest
    ramp consistent with where it meets the ground.  A structure that meets no
    measured ground at all (a pier: the sea is nodata, and so is the
    seawall cell under its landing) is solved with the streets around it,
    over the whole graph, from the nearest terrain under each of them.

5.  **Noise suppression (profile).**  Each contiguous run of a street is
    read at stations 5 m apart, where a running median over 35 m and then
    a Savitzky-Golay filter (order 2) over a ~50 m window are applied; the
    samples of each edge take their values back from the filtered run.

6.  **Dead-band.**  Cumulative gain and loss ignore consecutive elevation
    changes smaller than 0.5 m.  Without this, residual DEM noise makes a
    genuinely flat street accumulate tens of metres of phantom climbing over
    a few kilometres.
"""
from __future__ import annotations

import os
import zipfile
import zlib
from collections import OrderedDict
from pathlib import Path

import numpy as np

from .config import CRS_PROJECTED, ELEVATION, PROCESSED_DIR
from .download import DEM_DIR, DEM_INDEX_GEOJSON, dem_tile_index
from .utils import get_logger, progress, step

log = get_logger("hk_flat_routes.elevation")

#: The terrain model averaged to 10 m, for the hillshade and static maps
#: (the analysis itself samples the 0.5 m tiles).
DEM_OVERVIEW = PROCESSED_DIR / "dem_hk_10m.tif"
#: Kept under its upstream name for the code that draws relief from it.
DEM_MOSAIC = DEM_OVERVIEW
OVERVIEW_RES_M = 10.0
PROFILES_NPZ = PROCESSED_DIR / "edge_profiles.npz"

#: Standard deviation (m) of the Gaussian pre-filter applied to the DEM
#: (configured in ``config.ElevationConfig.dem_sigma_m``).
DEM_SMOOTH_SIGMA_M = ELEVATION.dem_sigma_m


# --------------------------------------------------------------------------
# the tiled DEM
# --------------------------------------------------------------------------
class TiledDem:
    """The 0.5 m tiles as one seamless raster, read a tile at a time.

    Tiles sit on a regular grid (every sheet is the same size and the
    sheets abut), so a point's tile is found by integer division and a
    tile's neighbours by its grid key.  Decoded tiles are held in a small
    LRU cache; sweeping the grid row by row then reads each tile about
    three times, which costs less than the smoothing does.
    """

    def __init__(self, tile_dir: Path = DEM_DIR, index: Path = DEM_INDEX_GEOJSON,
                 cache_tiles: int = 36):
        tiles = dem_tile_index(index)
        if not tiles:
            raise FileNotFoundError(f"empty DEM tile index {index}")
        widths = {t["bounds"][2] - t["bounds"][0] for t in tiles}
        heights = {t["bounds"][3] - t["bounds"][1] for t in tiles}
        if len(widths) != 1 or len(heights) != 1:
            raise ValueError(f"DEM tiles are not one size: {widths} x {heights}")
        self.tile_w = widths.pop()
        self.tile_h = heights.pop()
        self.x0 = min(t["bounds"][0] for t in tiles)
        self.y0 = min(t["bounds"][1] for t in tiles)
        self.paths: dict[tuple[int, int], Path] = {}
        missing = 0
        for t in tiles:
            path = Path(tile_dir) / t["filename"]
            if not path.exists():
                missing += 1
                continue
            self.paths[self._key(t["bounds"][0], t["bounds"][1])] = path
        if not self.paths:
            raise FileNotFoundError(
                f"no DEM tiles in {tile_dir}; run `python -m hk_flat_routes download`")
        if missing:
            log.warning("DEM: %d of %d tiles are not on disk; streets over "
                        "them will have no elevation", missing, len(tiles))
        self.n_cols = max(k[0] for k in self.paths) + 1
        self.n_rows = max(k[1] for k in self.paths) + 1
        self.bounds = (self.x0, self.y0, self.x0 + self.n_cols * self.tile_w,
                       self.y0 + self.n_rows * self.tile_h)
        self._cache: OrderedDict = OrderedDict()
        self._cache_tiles = cache_tiles
        first = self.read(next(iter(self.paths)))
        self.rows, self.cols = first.shape
        self.res = self.tile_w / self.cols
        if abs(self.tile_h / self.rows - self.res) > 1e-9:
            raise ValueError("DEM cells are not square")
        self.crs = CRS_PROJECTED

    def _key(self, x: float, y: float) -> tuple[int, int]:
        return (int(round((x - self.x0) / self.tile_w)),
                int(round((y - self.y0) / self.tile_h)))

    def keys_of(self, x: np.ndarray, y: np.ndarray):
        """Grid column and row of the tile containing each point."""
        col = np.floor((np.asarray(x) - self.x0) / self.tile_w).astype(np.int64)
        row = np.floor((np.asarray(y) - self.y0) / self.tile_h).astype(np.int64)
        return col, row

    def origin(self, key: tuple[int, int]) -> tuple[float, float]:
        """Top-left corner (x, y) of a tile."""
        return (self.x0 + key[0] * self.tile_w,
                self.y0 + (key[1] + 1) * self.tile_h)

    def read(self, key: tuple[int, int]):
        """One tile as float32 with NaN for nodata, or None if absent."""
        path = self.paths.get(key)
        if path is None:
            return None
        hit = self._cache.get(key)
        if hit is not None:
            self._cache.move_to_end(key)
            return hit
        from rasterio.io import MemoryFile

        try:
            with zipfile.ZipFile(path) as zf:
                inner = next(n for n in zf.namelist() if n.lower().endswith(".tif"))
                raw = zf.read(inner)
        except (zipfile.BadZipFile, zlib.error, RuntimeError, StopIteration,
                OSError) as exc:
            if isinstance(exc, OSError) and exc.filename is not None:
                raise                   # it could not be opened, and says so
            # A download cut short, a file damaged on disk, or a zip with no
            # GeoTIFF in it.  None of these says which of 3,253 files it
            # was.  The last arrives as a bare StopIteration, a file with
            # bytes missing from its middle as "OSError: [Errno 22] Invalid
            # argument", and one damaged in its directory as a compression
            # method that is not supported or a password that is required.
            raise zipfile.BadZipFile(
                f"{path}: not a readable terrain tile; delete it and run "
                f"`python -m hk_flat_routes download` again") from exc
        with MemoryFile(raw) as mem, mem.open() as src:
            arr = src.read(1).astype("float32")
            nodata = src.nodata
            left, top = src.transform.c, src.transform.f
        want = self.origin(key)
        if abs(left - want[0]) > 1e-6 or abs(top - want[1]) > 1e-6:
            raise ValueError(f"{path.name}: tile origin {(left, top)} does not "
                             f"match the index {want}")
        if nodata is not None:
            arr[arr == nodata] = np.nan
        self._cache[key] = arr
        while len(self._cache) > self._cache_tiles:
            self._cache.popitem(last=False)
        return arr

    def window(self, key: tuple[int, int], pad: int) -> np.ndarray:
        """A tile with ``pad`` cells of its eight neighbours around it."""
        rows, cols = self.rows, self.cols
        out = np.full((rows + 2 * pad, cols + 2 * pad), np.nan, dtype="float32")
        for dr in (-1, 0, 1):               # dr = +1 is the tile to the north
            for dc in (-1, 0, 1):
                arr = self.read((key[0] + dc, key[1] + dr))
                if arr is None:
                    continue
                # destination slice in the padded array, source slice in the tile
                if dc == 0:
                    dx, sx = slice(pad, pad + cols), slice(0, cols)
                elif dc < 0:
                    dx, sx = slice(0, pad), slice(cols - pad, cols)
                else:
                    dx, sx = slice(pad + cols, pad + cols + pad), slice(0, pad)
                if dr == 0:
                    dy, sy = slice(pad, pad + rows), slice(0, rows)
                elif dr > 0:
                    dy, sy = slice(0, pad), slice(rows - pad, rows)
                else:
                    dy, sy = slice(pad + rows, pad + rows + pad), slice(0, pad)
                out[dy, dx] = arr[sy, sx]
        return out


def _tile_newer_than(path: Path) -> bool:
    """True if a terrain tile on disk was written after ``path`` was."""
    if not DEM_DIR.is_dir():
        return False
    written = path.stat().st_mtime
    with os.scandir(DEM_DIR) as entries:
        return any(e.name.lower().endswith(".zip") and e.is_file()
                   and e.stat().st_mtime > written for e in entries)


def build_dem_overview(force: bool = False, res_m: float = OVERVIEW_RES_M) -> Path:
    """Average the 0.5 m tiles down to one small raster (cached).

    This is what the hillshade and the static maps are drawn from; nothing
    that is measured uses it.  The cache stands until a tile arrives that
    is newer than it: an overview first built while the download was part
    done would otherwise stay cropped to those tiles for good.  It is the
    tiles that are asked, not their directory, which changes as well when
    tiles are deleted to win back their 14 GB: rebuilt then, a whole
    overview would be cropped to the tiles that are left, with nothing to
    make it whole again from.  ``force`` rebuilds from the tiles there are
    and fails when there are none; without it, an overview that cannot be
    rebuilt is kept.  A rebuild reads every tile and takes a minute and a
    half.
    """
    import rasterio
    from rasterio.transform import from_origin

    if not force and DEM_OVERVIEW.exists() and not _tile_newer_than(DEM_OVERVIEW):
        log.info("cached %s", DEM_OVERVIEW.name)
        return DEM_OVERVIEW

    try:
        dem = TiledDem(cache_tiles=1)
    except FileNotFoundError as exc:
        if force or not DEM_OVERVIEW.exists():
            raise
        log.warning("%s cannot be rebuilt (%s); keeping the one on disk",
                    DEM_OVERVIEW.name, exc)
        return DEM_OVERVIEW
    k = int(round(res_m / dem.res))
    if dem.rows % k or dem.cols % k:
        raise ValueError(f"{res_m} m does not divide the DEM tiles evenly")
    tr, tc = dem.rows // k, dem.cols // k
    out = np.full((dem.n_rows * tr, dem.n_cols * tc), np.nan, dtype="float32")
    with step(f"averaging {len(dem.paths)} DEM tiles to {res_m:g} m", log):
        for key in progress(sorted(dem.paths), desc="  overview", unit="tile"):
            arr = dem.read(key)
            blocks = arr.reshape(tr, k, tc, k)
            n = np.isfinite(blocks).sum(axis=(1, 3))
            total = np.nansum(blocks, axis=(1, 3))
            # a cell is land when at least half of it is
            mean = np.where(n >= k * k / 2, total / np.maximum(n, 1), np.nan)
            r0 = (dem.n_rows - 1 - key[1]) * tr
            c0 = key[0] * tc
            out[r0:r0 + tr, c0:c0 + tc] = mean
    nodata = -9999.0
    transform = from_origin(dem.x0, dem.y0 + dem.n_rows * dem.tile_h, res_m, res_m)
    DEM_OVERVIEW.parent.mkdir(parents=True, exist_ok=True)
    tmp = DEM_OVERVIEW.with_suffix(".part.tif")
    with rasterio.open(tmp, "w", driver="GTiff", height=out.shape[0],
                       width=out.shape[1], count=1, dtype="float32",
                       crs=CRS_PROJECTED, transform=transform, nodata=nodata,
                       compress="deflate", predictor=3, tiled=True,
                       blockxsize=512, blockysize=512) as dst:
        dst.write(np.where(np.isfinite(out), out, nodata).astype("float32"), 1)
    tmp.replace(DEM_OVERVIEW)
    log.info("wrote %s  shape=%s  %.0f%% land", DEM_OVERVIEW.name, out.shape,
             100.0 * np.isfinite(out).mean())
    return DEM_OVERVIEW


class DemSampler:
    """Pre-smoothed bilinear point sampling over the tiled DEM."""

    def __init__(self, dem: TiledDem | None = None,
                 sigma_m: float = DEM_SMOOTH_SIGMA_M, smooth: bool = True,
                 floor: float | None = ELEVATION.min_ground_elev_m):
        self.dem = dem or TiledDem()
        self.floor = floor
        self.res = self.dem.res
        self.crs = self.dem.crs
        self.sigma_m = sigma_m if smooth else 0.0
        self.sigma_px = self.sigma_m / self.res
        # the filter reaches 4 sigma; two more cells cover the bilinear stencil
        self.pad = int(4.0 * self.sigma_px + 0.5) + 2
        if self.pad > min(self.dem.rows, self.dem.cols):
            raise ValueError("DEM smoothing reaches beyond the neighbouring tile")
        log.info("DEM sampler ready: %d tiles of %dx%d cells at %.2f m, "
                 "Gaussian sigma=%g m", len(self.dem.paths), self.dem.cols,
                 self.dem.rows, self.res, self.sigma_m)

    def _smoothed(self, key: tuple[int, int]):
        """A padded tile, smoothed, and the mask of its cells with data."""
        from scipy.ndimage import gaussian_filter

        block = self.dem.window(key, self.pad)
        valid = np.isfinite(block)
        if self.floor is not None:
            # excavations open at the time of the survey (see config)
            with np.errstate(invalid="ignore"):
                valid &= ~(block < self.floor)
            block = np.where(valid, block, np.nan)
        if self.sigma_px <= 0:
            return np.where(valid, block, 0.0), valid
        if valid.all():
            return gaussian_filter(block, sigma=self.sigma_px, mode="nearest"), valid
        # normalised convolution: nodata carries no weight, so the coast is
        # averaged over land only instead of toward a fill value
        weight = gaussian_filter(valid.astype("float32"), sigma=self.sigma_px,
                                 mode="nearest")
        total = gaussian_filter(np.where(valid, block, 0.0).astype("float32"),
                                sigma=self.sigma_px, mode="nearest")
        return np.where(weight > 1e-3, total / np.maximum(weight, 1e-3), 0.0), valid

    def sample(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Bilinear sample; returns NaN outside the DEM or over nodata."""
        x = np.asarray(x, dtype="float64")
        y = np.asarray(y, dtype="float64")
        shape = x.shape
        x = x.ravel(); y = y.ravel()
        out = np.full(x.shape, np.nan)
        dem, pad, res = self.dem, self.pad, self.res
        col, row = dem.keys_of(x, y)
        inside = (col >= 0) & (row >= 0) & (col < dem.n_cols) & (row < dem.n_rows)
        idx = np.flatnonzero(inside)
        if idx.size == 0:
            return out.reshape(shape)
        flat = row[idx] * dem.n_cols + col[idx]
        order = np.argsort(flat, kind="stable")
        idx, flat = idx[order], flat[order]
        starts = np.flatnonzero(np.r_[True, flat[1:] != flat[:-1]])
        ends = np.r_[starts[1:], flat.size]
        bar = progress(zip(starts, ends), desc="  DEM tiles", total=starts.size,
                       unit="tile") if starts.size > 50 else zip(starts, ends)
        for a, b in bar:
            key = (int(flat[a] % dem.n_cols), int(flat[a] // dem.n_cols))
            if key not in dem.paths:
                continue
            data, valid = self._smoothed(key)
            left, top = dem.origin(key)
            pts = idx[a:b]
            # cell-centre coordinates in the padded array
            c = (x[pts] - left) / res + pad - 0.5
            r = (top - y[pts]) / res + pad - 0.5
            c0 = np.floor(c).astype(np.int64); r0 = np.floor(r).astype(np.int64)
            fc = c - c0; fr = r - r0
            v00 = data[r0, c0]; v01 = data[r0, c0 + 1]
            v10 = data[r0 + 1, c0]; v11 = data[r0 + 1, c0 + 1]
            z = ((v00 * (1 - fc) + v01 * fc) * (1 - fr)
                 + (v10 * (1 - fc) + v11 * fc) * fr)
            if self.sigma_px <= 0:
                # Unfiltered, a nodata cell holds 0 where the filter would
                # have spread the land beside it, and a point a tenth of a
                # metre inside a seawall at 4 m read 2.8 m.  The stencil is
                # weighed by its cells with data instead; the nearest cell
                # has data wherever the result is kept, and it carries at
                # least a quarter of the weight.
                m00 = valid[r0, c0]; m01 = valid[r0, c0 + 1]
                m10 = valid[r0 + 1, c0]; m11 = valid[r0 + 1, c0 + 1]
                w = ((m00 * (1 - fc) + m01 * fc) * (1 - fr)
                     + (m10 * (1 - fc) + m11 * fc) * fr)
                z = np.where(m00 & m01 & m10 & m11, z, z / np.maximum(w, 0.25))
            # invalidate points whose nearest cell was nodata
            ok = valid[np.round(r).astype(np.int64), np.round(c).astype(np.int64)]
            out[pts] = np.where(ok, z, np.nan)
        return out.reshape(shape)


# --------------------------------------------------------------------------
# densification
# --------------------------------------------------------------------------
def densify(geom, spacing: float, min_samples: int):
    """Return (distance_along, x, y) sample arrays for one LineString."""
    length = geom.length
    n = max(min_samples, int(np.ceil(length / spacing)) + 1)
    dist = np.linspace(0.0, length, n)
    pts = [geom.interpolate(float(d)) for d in dist]
    x = np.fromiter((p.x for p in pts), dtype="float64", count=n)
    y = np.fromiter((p.y for p in pts), dtype="float64", count=n)
    return dist, x, y


# --------------------------------------------------------------------------
# structure (bridge/tunnel) elevation recovery
# --------------------------------------------------------------------------
def _solve_structure_nodes(edges, node_elev: dict, reliable: set,
                           subset=None, label: str = "structure-only",
                           level: dict | None = None) -> dict:
    """Laplace-smooth elevations for nodes with no usable DEM value.

    Nodes on a viaduct have no DEM-derived elevation worth using, and a
    handful of nodes (piers, DEM holes) have none at all.  Treating the
    relevant sub-graph as a resistor network with the reliable nodes as
    Dirichlet boundary conditions produces the smoothest surface consistent
    with where it meets solid, measured ground.

    Two things differ from upstream, which relaxes the system for 400
    sweeps starting from the mean height of every bridge end in the city.

    The system is solved directly.  Relaxation needs sweeps in proportion
    to the square of a chain's length, and more where long spans meet
    short ones: 400 settle an even chain of twenty nodes and leave one of
    fifty nearly 3 m of a 10 m starting error.  The decks of the Tseung
    Kwan O Cross Bay Link, 3 km of cycleway and footway that land at 4 to
    5 m at every end, were left rising to 16 m over the bay.

    And a group of unknown nodes that joins no reliable node has nothing
    to be smoothed from, so it is handed back without an elevation (NaN)
    instead of being left at the starting value.  San Francisco's model
    has values over water; here the sea is nodata, every pier and every
    waterfront footbridge with its landings on a seawall is such a group,
    and the Star Ferry gangway at Tsim Sha Tsui stood at that mean, 35.69
    mPD.  The caller solves what comes back empty over the whole graph,
    from the streets around it.

    ``subset`` selects the edges spanned; the default is the structure
    sub-graph.  ``level`` is for the last pass, when no other will follow:
    it gives the terrain under every node, and a group that joins no
    reliable node is then laid level at the mean of the terrain under its
    own nodes, which is all that is known about where it stands.
    """
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse.linalg import spsolve

    struct = edges[edges["is_structure"]] if subset is None else subset
    us = struct["u"].to_numpy(); vs = struct["v"].to_numpy()
    weight = 1.0 / np.maximum(struct["length_m"].to_numpy(dtype="float64"), 1.0)
    unknown = [n for n in dict.fromkeys([*us, *vs]) if n not in reliable]
    if not unknown:
        return node_elev
    index = {n: i for i, n in enumerate(unknown)}
    n = len(unknown)
    iu = np.fromiter((index.get(m, -1) for m in us), dtype=np.int64, count=us.size)
    iv = np.fromiter((index.get(m, -1) for m in vs), dtype=np.int64, count=vs.size)
    zu = np.fromiter((node_elev.get(m, np.nan) for m in us), dtype="float64",
                     count=us.size)
    zv = np.fromiter((node_elev.get(m, np.nan) for m in vs), dtype="float64",
                     count=vs.size)

    # an edge to a reliable node holds an unknown one to that height ...
    diag = np.zeros(n); rhs = np.zeros(n)
    for a, b, zb in ((iu, iv, zv), (iv, iu, zu)):
        k = (a >= 0) & (b < 0) & np.isfinite(zb)
        np.add.at(diag, a[k], weight[k])
        np.add.at(rhs, a[k], weight[k] * zb[k])
    anchored = diag > 0
    # ... and an edge between two unknown nodes holds them to each other
    k = (iu >= 0) & (iv >= 0)
    np.add.at(diag, iu[k], weight[k])
    np.add.at(diag, iv[k], weight[k])
    own = np.arange(n)
    system = sp.coo_matrix(
        (np.concatenate([diag, -weight[k], -weight[k]]),
         (np.concatenate([own, iu[k], iv[k]]), np.concatenate([own, iv[k], iu[k]]))),
        shape=(n, n)).tocsr()

    # only a group with a reliable node beside it has a solution
    _, group = connected_components(system, directed=False)
    has_anchor = np.zeros(group.max() + 1, dtype=bool)
    has_anchor[group[anchored]] = True
    solved = has_anchor[group]
    z = np.full(n, np.nan)
    if solved.any():
        z[solved] = spsolve(system[solved][:, solved].tocsc(), rhs[solved])

    if level is not None:
        under = np.fromiter((level.get(m, np.nan) for m in unknown),
                            dtype="float64", count=n)
        for g in np.flatnonzero(~has_anchor):
            members = group == g
            if np.isfinite(under[members]).any():
                z[members] = np.nanmean(under[members])

    out = dict(node_elev)
    out.update(zip(unknown, z.tolist()))
    log.info("  recovered elevations for %d %s nodes", int(solved.sum()), label)
    if not solved.all():
        log.info("  %d %s nodes in %d groups join no measured ground%s",
                 int((~solved).sum()), label, int((~has_anchor).sum()),
                 "" if level is None else
                 "; each group is laid level on the terrain under it")
    return out


def _links_to_terrain(edges, cat_dist, cat_z, offsets_idx, missing: set):
    """The graph over which the nodes with no terrain value are solved.

    Such a node is usually the end of a street that has terrain under the
    rest of it: the last cell before a seawall, the head of a flight of
    landing steps.  Solved from the nodes at the far ends of its edges, it
    took the height of the street behind it: where a 72 m footway comes
    down from 10.5 m to the shore (22.3343 N 114.0558 E) and its own
    samples reach 2.0 m before the model ends, the steps beyond stood at
    10.5 m.  So an edge on the ground with terrain under any of its samples
    joins a missing end not to its other end but to the nearest of those
    samples, which stands in as a node of known height at its distance
    along the edge.  A bridge, a tunnel or an edge with no terrain under it
    at all joins its two ends, as in the structure solve.

    Returns the links as a table of ``u``, ``v`` and ``length_m``, and the
    elevations of the samples that stand in as nodes.
    """
    import pandas as pd

    us = edges["u"].to_numpy(); vs = edges["v"].to_numpy()
    is_struct = edges["is_structure"].to_numpy()
    length = edges["length_m"].to_numpy(dtype="float64")
    touching = (edges["u"].isin(missing) | edges["v"].isin(missing)).to_numpy()
    links, at_sample = [], {}
    for i in np.flatnonzero(touching):
        a, b = offsets_idx[i], offsets_idx[i + 1]
        good = np.flatnonzero(np.isfinite(cat_z[a:b]))
        if is_struct[i] or good.size == 0:
            links.append((us[i], vs[i], length[i]))
            continue
        d = cat_dist[a:b]
        for node, k, away in ((us[i], good[0], d[good[0]] - d[0]),
                              (vs[i], good[-1], d[-1] - d[good[-1]])):
            if node in missing:
                sample = ("sample", int(a + k))
                at_sample[sample] = float(cat_z[a + k])
                links.append((node, sample, float(away)))
    return pd.DataFrame(links, columns=["u", "v", "length_m"]), at_sample


def node_points(edges):
    """Every graph node with its position, and the nodes on solid ground.

    Returns ``(nodes, x, y, solid)``: ``solid`` holds the nodes touched by at
    least one edge that is not a bridge or a tunnel, where the DEM can be
    believed.
    """
    import collections
    xs: dict[str, list[float]] = collections.defaultdict(list)
    ys: dict[str, list[float]] = collections.defaultdict(list)
    solid: set[str] = set()

    for u, v, geom, is_struct in zip(edges["u"], edges["v"], edges.geometry,
                                     edges["is_structure"]):
        c = geom.coords
        xs[u].append(c[0][0]); ys[u].append(c[0][1])
        xs[v].append(c[-1][0]); ys[v].append(c[-1][1])
        if not is_struct:
            solid.add(u); solid.add(v)

    nodes = list(xs)
    px = np.array([np.mean(xs[n]) for n in nodes])
    py = np.array([np.mean(ys[n]) for n in nodes])
    return nodes, px, py, solid


def compute_node_elevations(edges, sampler: DemSampler, z=None) -> tuple[dict, set]:
    """DEM elevation at every graph node, plus the set of reliable nodes.

    ``z`` may carry elevations already sampled at ``node_points(edges)``,
    to save a pass over the DEM tiles.
    """
    nodes, px, py, solid = node_points(edges)
    if z is None:
        z = sampler.sample(px, py)
    node_elev = {n: float(zz) for n, zz in zip(nodes, z)}
    reliable = {n for n in solid if np.isfinite(node_elev[n])}
    log.info("node elevations: %d nodes, %d reliable (non-structure, valid DEM)",
             len(nodes), len(reliable))
    return node_elev, reliable


# --------------------------------------------------------------------------
# profile extraction
# --------------------------------------------------------------------------
def _despike(z: np.ndarray, spacing: float) -> np.ndarray:
    """Running median along the profile (``despike_window_m``)."""
    from scipy.ndimage import median_filter
    win = int(round(ELEVATION.despike_window_m / max(spacing, 1e-6)))
    if win % 2 == 0:
        win += 1
    if win < 3 or z.size < win:
        return z
    return median_filter(z, size=win, mode="nearest")


def _smooth_profile(z: np.ndarray, spacing: float) -> np.ndarray:
    """Savitzky-Golay along the profile where enough samples exist."""
    from scipy.signal import savgol_filter
    n = z.size
    if n < 5:
        return z
    win = int(round(2 * ELEVATION.smooth_window_m / max(spacing, 1e-6))) + 1
    win = min(win, n if n % 2 == 1 else n - 1)
    if win < 5:
        return z
    if win % 2 == 0:
        win -= 1
    try:
        return savgol_filter(z, win, ELEVATION.smooth_polyorder, mode="interp")
    except Exception:
        return z


def _filter_run(dist: np.ndarray, z: np.ndarray, spacing: float) -> np.ndarray:
    """The running median and Savitzky-Golay filters along one street run.

    Both filters count in samples, and the samples of a run are not evenly
    spaced: every edge has at least three, so a street piece one metre
    long carries samples half a metre apart between neighbours sampled
    every five.  Filtered as they stand, a quadratic fitted over eleven
    samples takes the crowded ones for fifty metres of street and moves
    elevation from the long pieces onto the short one: on a steady 20%
    street a 1 m piece between two of 60 m came out rising 1.27 m, not
    0.20 m.  So the run is read at stations ``spacing`` apart (a little
    less, to end on the last sample), filtered there, and each sample
    takes its value back from the result.  A run no longer than the
    spacing is left as it is.
    """
    length = float(dist[-1] - dist[0])
    n = int(np.ceil(length / spacing)) + 1
    if n < 3:
        return z
    stations = np.linspace(dist[0], dist[-1], n)
    step = length / (n - 1)
    at_stations = np.interp(stations, dist, z)
    at_stations = _smooth_profile(_despike(at_stations, step), step)
    return np.interp(dist, stations, at_stations)


#: What the terrace search pays to change its sideways offset between one
#: sample and the next, in cross-slope per metre of change (see
#: ``flattest_offset``).  The value is not critical: from 0.02 to 0.08
#: Bowen Road reads 10.2 to 10.9 m of climbing per km, against 11.9 with
#: the five-sample median this replaced, and no fewer bench marks are
#: matched within a metre.  At 0.15 the offset is too stiff to follow the
#: road where the mapped line wanders across it: Bowen Road reads 12.8.
LATERAL_CHANGE_COST = 0.05

#: Edges searched at a time (see ``_least_cost_offsets``).
SEARCH_BLOCK_EDGES = 1000


def lateral_offsets(cfg=ELEVATION) -> np.ndarray:
    """Sideways offsets (m) tried by the terrace search, centre included."""
    n = int(round(cfg.lateral_search_m / cfg.lateral_step_m))
    return np.arange(-n, n + 1, dtype="float64") * cfg.lateral_step_m


def flattest_offset(Z: np.ndarray, offsets: np.ndarray, penalty: float,
                    starts: np.ndarray | None = None) -> np.ndarray:
    """Index of the spot to sample across the way, for each sample.

    ``Z`` holds the DEM at each sample (rows) and each sideways offset
    (columns), for one edge or for several end to end; ``starts`` is then
    the first row of each.  The cross-slope at an offset is taken over
    three metres around it, and a spot scores that plus ``penalty`` per
    metre from the centreline.  Along each edge the offsets taken are the
    sequence with the lowest total score, where every metre of sideways
    change between one sample and the next costs ``LATERAL_CHANGE_COST``
    as well: the search follows a ledge and does not hop between ledges.

    The best spot used to be chosen sample by sample and held steady by a
    five-sample median afterwards.  That holds nothing at the two ends of
    an edge, the samples it shares with its neighbours, and nothing at all
    on an edge of fewer than five samples, which 24% of the edges searched
    are; and where a hillside has no ledge the choice is noise.  A 10 m
    path on Lantau whose three samples rise 3.2 m along the centreline was
    read 6 m to one side, then 1 m and 5 m to the other, and rose 18.0 m.
    """
    n = Z.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    starts = np.zeros(1, dtype=np.int64) if starts is None else np.asarray(starts)
    per = SEARCH_BLOCK_EDGES
    cuts = np.r_[starts[::per], n]
    return np.concatenate([
        _least_cost_offsets(Z[a:b], offsets, penalty, starts[i:i + per] - a)
        for i, a, b in zip(range(0, starts.size, per), cuts[:-1], cuts[1:])])


def _least_cost_offsets(Z: np.ndarray, offsets: np.ndarray, penalty: float,
                        starts: np.ndarray) -> np.ndarray:
    """``flattest_offset`` for one block of edges.

    The tables below come to several times the size of ``Z``.  For the
    1.55 million samples of a full build taken whole that is 850 MB at
    once; taken 1,000 edges at a time it is under 100 MB, and no slower.
    """
    from scipy.ndimage import uniform_filter1d

    n, m = Z.shape
    sizes = np.diff(np.r_[starts, n])
    step = offsets[1] - offsets[0]
    finite = np.isfinite(Z)
    # a cell with no data stands at its edge's mean for the slopes around it
    total = np.add.reduceat(np.where(finite, Z, 0.0).sum(axis=1), starts)
    count = np.add.reduceat(finite.sum(axis=1), starts)
    fill = np.repeat(total / np.maximum(count, 1), sizes)
    score = np.abs(np.gradient(np.where(finite, Z, fill[:, None]), step, axis=1))
    score = uniform_filter1d(score, max(int(round(3.0 / step)), 1), axis=1,
                             mode="nearest")
    score += penalty * np.abs(offsets)[None, :]
    score[~finite] = np.inf

    # The cheapest path across each edge's table of scores, all the edges
    # of the block at once: the longest come first, so those with a sample
    # at position i are always the leading rows of ``cost``.
    order = np.argsort(-sizes, kind="stable")
    first, sizes = starts[order], sizes[order]
    longest = int(sizes[0])
    active = np.searchsorted(-sizes, -np.arange(longest), side="left")
    change = LATERAL_CHANGE_COST * np.abs(offsets[:, None] - offsets[None, :])
    # a cell with no data is priced as a cross-slope of 1000%: out of reach,
    # yet finite, so that the paths through it can still be added up
    price = np.where(finite, score, 10.0)
    cost = price[first]
    back = np.zeros((n, m), dtype=np.int16)
    for i in range(1, longest):
        rows = first[:active[i]] + i
        via = cost[:active[i], :, None] + change     # from each offset to each
        back[rows] = via.argmin(axis=1)
        cost[:active[i]] = via.min(axis=1) + price[rows]
    k = np.zeros(n, dtype=np.int64)
    k[first + sizes - 1] = cost.argmin(axis=1)
    for i in range(longest - 1, 0, -1):
        rows = first[:active[i]] + i
        k[rows - 1] = back[rows, k[rows]]

    # a sample with no data at the chosen spot falls back to its own best
    bad = ~finite[np.arange(n), k]
    if bad.any():
        k[bad] = np.argmin(score[bad], axis=1)
    return k


def _edge_normals(cat_x, cat_y, cat_dist, offsets_idx):
    """Unit normals along every edge, from its own sample points."""
    nx = np.zeros_like(cat_x); ny = np.zeros_like(cat_y)
    for i in range(offsets_idx.size - 1):
        a, b = offsets_idx[i], offsets_idx[i + 1]
        if b - a < 2:
            continue
        tx = np.gradient(cat_x[a:b], cat_dist[a:b])
        ty = np.gradient(cat_y[a:b], cat_dist[a:b])
        norm = np.hypot(tx, ty)
        norm[norm == 0] = 1.0
        nx[a:b] = -ty / norm; ny[a:b] = tx / norm
    return nx, ny


def _terrace_search(sampler: "DemSampler", cat_x, cat_y, nx, ny, offsets_idx,
                    cat_z, z_right, z_left, skip_edge) -> tuple[np.ndarray, int]:
    """Re-sample edges that run across a slope at the flattest spot nearby.

    ``z_right`` and ``z_left`` are the DEM two metres either side of each
    sample (the normals point left), which give the cross-slope at the
    centreline.  Returns the corrected elevations and the number of edges
    searched.
    """
    cfg = ELEVATION
    n_edges = offsets_idx.size - 1
    cross0 = np.abs(z_right - z_left) / 4.0
    steep = np.isfinite(cross0) & (cross0 > cfg.lateral_min_cross_slope)

    # edges with enough of their length across a slope are searched whole
    counts = np.add.reduceat(steep.astype(np.int64), offsets_idx[:-1])
    sizes = np.diff(offsets_idx)
    todo = (counts >= np.maximum(2, 0.2 * sizes)) & ~np.asarray(skip_edge)
    pick = np.flatnonzero(np.repeat(todo, sizes))
    if pick.size == 0:
        return cat_z, 0
    offs = lateral_offsets(cfg)
    with step(f"searching {int(todo.sum()):,} hillside edges for the road "
              f"surface ({pick.size:,} points x {offs.size} offsets)", log):
        Z = sampler.sample(cat_x[pick][:, None] + offs[None, :] * nx[pick][:, None],
                           cat_y[pick][:, None] + offs[None, :] * ny[pick][:, None])
    out = cat_z.copy()
    # first row of each searched edge within ``pick``
    edge_of = np.repeat(np.arange(n_edges), sizes)[pick]
    starts = np.flatnonzero(np.r_[True, edge_of[1:] != edge_of[:-1]])
    k = flattest_offset(Z, offs, cfg.lateral_penalty, starts)
    z = Z[np.arange(pick.size), k]
    ok = np.isfinite(z)
    out[pick[ok]] = z[ok]
    log.info("  mean sideways shift on those edges: %.2f m",
             float(np.abs(offs[k]).mean()))
    return out, int(todo.sum())


def _one_height_per_node(edges, cat_z, offsets_idx) -> np.ndarray:
    """Give the edges that meet at a node one elevation for it.

    Every edge is searched along its own normals, so the edges meeting at
    a node on a hillside each take the node's sample from a different spot,
    up to twice the search distance apart.  Each end sample on the ground
    is replaced by the median of the ends that share its node; the ends of
    bridges and tunnels are set from their nodes later.

    The search needs it, now that it holds an edge to its own ledge right
    up to its ends: left as they are read, the ends of 943 edges have to
    be moved by more than 2 m when the nodes are reconciled at the last,
    and each such move tilts a profile; with this step 399 have.  (The
    bench marks do not tell the two apart: 1,165 and 1,166 of the 1,325
    beside roads are matched within a metre.)
    """
    import pandas as pd

    ground = ~edges["is_structure"].to_numpy()
    at = np.concatenate([offsets_idx[:-1][ground], offsets_idx[1:][ground] - 1])
    node = np.concatenate([edges["u"].to_numpy()[ground],
                           edges["v"].to_numpy()[ground]])
    keep = np.isfinite(cat_z[at])
    at, node = at[keep], node[keep]
    out = cat_z.copy()
    out[at] = pd.Series(cat_z[at]).groupby(node).transform("median").to_numpy()
    return out


def _correct_abutments(edges, cat_dist, cat_z, offsets_idx, node_elev: dict):
    """Elevations where a bridge or a tunnel meets the ground.

    See ``ElevationConfig.abutment_blank_m``.  Returns the corrected sample
    elevations and node elevations.
    """
    import collections

    cfg = ELEVATION
    if cfg.abutment_blank_m <= 0:
        return cat_z, node_elev
    blank, fit = cfg.abutment_blank_m, cfg.abutment_fit_m
    is_struct = edges["is_structure"].to_numpy()
    us = edges["u"].to_numpy(); vs = edges["v"].to_numpy()
    on_struct = set(us[is_struct]) | set(vs[is_struct])
    approach: dict = collections.defaultdict(list)     # node -> (edge row, at u end?)
    for i in np.flatnonzero(~is_struct):
        if us[i] in on_struct:
            approach[us[i]].append((i, True))
        if vs[i] in on_struct:
            approach[vs[i]].append((i, False))

    def from_node(i: int, at_u: bool):
        a, b = offsets_idx[i], offsets_idx[i + 1]
        d = cat_dist[a:b]
        return (d if at_u else d[-1] - d), cat_z[a:b]

    out = cat_z.copy()
    node_elev = dict(node_elev)
    n_fixed = 0
    shifts = []
    for node, ends in approach.items():
        est = []
        for i, at_u in ends:
            s, z = from_node(i, at_u)
            sel = (s >= blank) & (s <= blank + fit) & np.isfinite(z)
            if sel.sum() < 3:
                continue
            slope, intercept = np.polyfit(s[sel], z[sel], 1)
            if abs(slope) <= 0.30:
                est.append(float(intercept))
        if not est:
            continue
        z_new = float(np.median(est))
        z_old = node_elev.get(node, np.nan)
        if np.isfinite(z_old) and abs(z_new - z_old) <= cfg.abutment_min_shift_m:
            continue
        node_elev[node] = z_new
        n_fixed += 1
        if np.isfinite(z_old):
            shifts.append(z_new - z_old)
        # the approach itself: a straight line over the untrusted stretch
        for i, at_u in ends:
            a, b = offsets_idx[i], offsets_idx[i + 1]
            s, z = from_node(i, at_u)
            close = s < blank
            good = ~close & np.isfinite(z)
            if not close.any() or not good.any():
                continue
            k = int(np.flatnonzero(good)[np.argmin(s[good])])
            seg = out[a:b]
            seg[close] = z_new + (z[k] - z_new) * (s[close] / s[k])
    if shifts:
        sh = np.asarray(shifts)
        log.info("  %d structure ends re-levelled from their approaches "
                 "(median %+.1f m, %d raised, %d lowered)", n_fixed,
                 float(np.median(sh)), int((sh > 0).sum()), int((sh < 0).sum()))
    return out, node_elev


def _profiles_are_of(edges, ids, offsets, dist) -> bool:
    """True if cached profiles belong to this edge table.

    A forced build writes the network and then spends ten minutes sampling
    it.  Stopped in between and run again without ``--force``, it found the
    new network and the old profiles both cached, and an edge id is only a
    row number: the same ids name other streets.  So the profiles must
    also be as long as the edges they are read for.
    """
    return (ids.size == len(edges)
            and np.array_equal(ids, np.asarray(edges["edge_id"].values))
            and np.allclose(dist[offsets[1:] - 1], edges.geometry.length.to_numpy(),
                            atol=0.01))


def sample_edge_profiles(edges, sampler: DemSampler | None = None,
                         force: bool = False) -> dict[int, tuple]:
    """Elevation profile for every edge, cached to ``edge_profiles.npz``.

    The cache is used only for the network it was sampled on (see
    ``_profiles_are_of``).  Returns ``{edge_id: (distance_along_m,
    elevation_m)}``.
    """
    if PROFILES_NPZ.exists() and not force:
        with np.load(PROFILES_NPZ, allow_pickle=False) as npz:
            ids = npz["edge_ids"]
            offsets = npz["offsets"]
            dist = npz["dist"]
            elev = npz["elev"]
        if _profiles_are_of(edges, ids, offsets, dist):
            out = {int(e): (dist[offsets[i]:offsets[i + 1]],
                            elev[offsets[i]:offsets[i + 1]])
                   for i, e in enumerate(ids)}
            log.info("cached %s (%d profiles)", PROFILES_NPZ.name, len(out))
            return out
        log.warning("%s holds the profiles of another network (%d, for %d edges "
                    "here); sampling again", PROFILES_NPZ.name, ids.size, len(edges))

    sampler = sampler or DemSampler()
    cfg = ELEVATION

    # densify first, so that nodes, centrelines and the two side lines that
    # measure cross-slope are all read in one pass over the DEM tiles
    with step("densifying edge geometries", log):
        all_dist, all_x, all_y, counts = [], [], [], []
        for geom in progress(edges.geometry, desc="  densify", total=len(edges),
                             unit="edge"):
            d, x, y = densify(geom, cfg.sample_spacing_m, cfg.min_samples)
            all_dist.append(d); all_x.append(x); all_y.append(y)
            counts.append(d.size)
    counts = np.asarray(counts)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    cat_x = np.concatenate(all_x); cat_y = np.concatenate(all_y)
    cat_dist = np.concatenate(all_dist)
    nodes, node_x, node_y, solid = node_points(edges)
    search = cfg.lateral_search_m > 0
    parts_x = [node_x, cat_x]; parts_y = [node_y, cat_y]
    if search:
        nx, ny = _edge_normals(cat_x, cat_y, cat_dist, offsets)
        parts_x += [cat_x - 2.0 * nx, cat_x + 2.0 * nx]
        parts_y += [cat_y - 2.0 * ny, cat_y + 2.0 * ny]
    n_all = sum(a.size for a in parts_x)
    with step(f"sampling DEM at {n_all:,} points", log):
        z_all = sampler.sample(np.concatenate(parts_x), np.concatenate(parts_y))
    cuts = np.cumsum([a.size for a in parts_x])[:-1]
    z_parts = np.split(z_all, cuts)
    cat_z = z_parts[1]

    edge_ids = np.asarray(edges["edge_id"].values)
    is_struct = np.asarray(edges["is_structure"].values)
    if search:
        # hillside terraces: take each sample where the road actually is.
        # Structures are left alone (their profile is replaced by a ramp).
        cat_z, _ = _terrace_search(sampler, cat_x, cat_y, nx, ny, offsets, cat_z,
                                   z_parts[2], z_parts[3], skip_edge=is_struct)
        cat_z = _one_height_per_node(edges, cat_z, offsets)

    with step("computing node elevations", log):
        node_elev, reliable = compute_node_elevations(edges, sampler, z=z_parts[0])
    with step("correcting the ground at bridge and tunnel ends", log):
        cat_z, node_elev = _correct_abutments(edges, cat_dist, cat_z, offsets,
                                              node_elev)
        # an end with no terrain value of its own (on a seawall, at a
        # channel's edge) that was given one from its approach is measured
        # ground from here on, like any other end
        reliable = {n for n in solid if np.isfinite(node_elev.get(n, np.nan))}
    with step("recovering bridge/tunnel deck elevations", log):
        under = node_elev
        node_elev = _solve_structure_nodes(edges, node_elev, reliable)
        reliable = reliable | {n for n in node_elev
                               if np.isfinite(node_elev.get(n, np.nan))}
    missing = [n for n in node_elev if not np.isfinite(node_elev.get(n, np.nan))]
    if missing:
        # A few nodes sit where the DEM has no data at all (piers over water,
        # interpolation holes), and a structure whose every landing is such
        # a node comes back from the solve above without an elevation.
        # Assigning them 0 m would be catastrophic: it once put an 18 m
        # footway 65 m below the street it joins. Solve them over the whole
        # graph instead, from the nearest terrain on the streets they stand
        # on.  Decks mapped with no join to any street (42 nodes of walkway
        # at the airport) have nothing to be solved from, and are laid level.
        with step(f"recovering {len(missing)} nodes with no DEM coverage", log):
            links, at_sample = _links_to_terrain(edges, cat_dist, cat_z, offsets,
                                                 set(missing))
            solved = _solve_structure_nodes(
                links, {**node_elev, **at_sample}, reliable | set(at_sample),
                subset=links, label="DEM-void", level=under)
            node_elev.update((n, solved[n]) for n in missing)

    us = list(edges["u"]); vs = list(edges["v"])
    row_of = {int(e): i for i, e in enumerate(edge_ids)}

    # Structure ramps, gap filling and smoothing are all applied to the
    # *concatenated* profile of each contiguous run of a street segment, not
    # edge by edge. Smoothing an edge in isolation gives the two edges either
    # side of an intersection different elevations for the same corner, and
    # in a street grid that happens every 80 m; the discontinuities then
    # propagate into the climbing figures.
    from .metrics import contiguous_runs

    out_dist: list[np.ndarray] = [None] * len(edge_ids)
    out_elev: list[np.ndarray] = [None] * len(edge_ids)
    n_struct_fixed = n_gapfilled = n_unresolved = 0

    runs = contiguous_runs(edges)
    for eids in progress(runs, desc="  profiles", total=len(runs), unit="run"):
        rows = [row_of[int(e)] for e in eids]
        spans, segs, zs, base = [], [], [], 0.0
        for k, i in enumerate(rows):
            a, b = offsets[i], offsets[i + 1]
            d = cat_dist[a:b].copy()
            z = cat_z[a:b].copy()
            if k > 0:
                d = d[1:]; z = z[1:]
            segs.append(base + d); zs.append(z); spans.append(d.size)
            base = segs[-1][-1] if segs[-1].size else base
        run_d = np.concatenate(segs)
        run_z = np.concatenate(zs)

        # index range of each edge within the run, sharing joint stations
        ranges, pos = [], 0
        for k in range(len(rows)):
            lo = pos if k == 0 else pos - 1
            hi = pos + spans[k]
            ranges.append((lo, hi))
            pos = hi

        # structures: replace the DEM's view of the ground under the deck
        if cfg.interpolate_structures:
            for k, i in enumerate(rows):
                if not is_struct[i]:
                    continue
                lo, hi = ranges[k]
                z0 = node_elev.get(us[i], np.nan)
                z1 = node_elev.get(vs[i], np.nan)
                if np.isfinite(z0) and np.isfinite(z1):
                    dd = run_d[lo:hi]
                    spanlen = dd[-1] - dd[0]
                    frac = (dd - dd[0]) / (spanlen if spanlen > 0 else 1.0)
                    run_z[lo:hi] = z0 + (z1 - z0) * frac
                    n_struct_fixed += 1

        if not np.all(np.isfinite(run_z)):
            good = np.isfinite(run_z)
            if good.any():
                run_z = np.interp(run_d, run_d[good], run_z[good])
            else:
                ends = [node_elev.get(us[rows[0]], np.nan),
                        node_elev.get(vs[rows[-1]], np.nan)]
                if all(np.isfinite(ends)):
                    frac = ((run_d - run_d[0])
                            / max(run_d[-1] - run_d[0], 1e-9))
                    run_z = ends[0] + (ends[1] - ends[0]) * frac
                else:
                    n_unresolved += len(rows)
            n_gapfilled += 1

        # smooth the whole run, except where it is entirely a structure deck
        if np.all(np.isfinite(run_z)) and not all(is_struct[i] for i in rows):
            run_z = _filter_run(run_d, run_z, cfg.sample_spacing_m)

        for k, i in enumerate(rows):
            lo, hi = ranges[k]
            d = run_d[lo:hi]
            out_dist[i] = (d - d[0]).astype("float32")
            out_elev[i] = run_z[lo:hi].astype("float32")

    log.info("  linear ramp applied to %d structure edges; %d runs gap-filled",
             n_struct_fixed, n_gapfilled)
    if n_unresolved:
        log.warning("  %d edges have no recoverable elevation and carry NaN; "
                    "the metrics table keeps them, closed to routing",
                    n_unresolved)

    # ---- one elevation per intersection -------------------------------
    # Each street segment is smoothed independently, so two streets meeting
    # at a corner end up with slightly different elevations for that same
    # corner. Per-edge climbing is still exact, but summing it along a route
    # then drifts from the difference between the route's endpoints (a few
    # metres over a long route). Reconciling every node to a single
    # elevation and rubber-sheeting each profile onto it with a linear
    # correction removes the drift entirely while leaving the profile's
    # shape, and therefore its gradients, essentially untouched.
    with step("reconciling elevations at intersections", log):
        acc: dict[str, list[float]] = {}
        for i in range(len(edge_ids)):
            if out_elev[i] is None or not np.all(np.isfinite(out_elev[i])):
                continue
            acc.setdefault(us[i], []).append(float(out_elev[i][0]))
            acc.setdefault(vs[i], []).append(float(out_elev[i][-1]))
        canon = {n: float(np.mean(v)) for n, v in acc.items()}

        shifts = []
        for i in range(len(edge_ids)):
            z = out_elev[i]
            if z is None or z.size < 2 or not np.all(np.isfinite(z)):
                continue
            z0 = canon.get(us[i]); z1 = canon.get(vs[i])
            if z0 is None or z1 is None:
                continue
            d = out_dist[i].astype("float64")
            span = d[-1] - d[0]
            frac = (d - d[0]) / (span if span > 0 else 1.0)
            delta0 = z0 - float(z[0])
            delta1 = z1 - float(z[-1])
            out_elev[i] = (z.astype("float64")
                           + delta0 * (1.0 - frac) + delta1 * frac
                           ).astype("float32")
            shifts.append(max(abs(delta0), abs(delta1)))
        if shifts:
            sh = np.asarray(shifts)
            log.info("  node reconciliation shifted profile ends by "
                     "%.3f m on average, %.2f m at worst (%d edges)",
                     sh.mean(), sh.max(), sh.size)

    cat_d = np.concatenate(out_dist); cat_e = np.concatenate(out_elev)
    np.savez_compressed(PROFILES_NPZ, edge_ids=edge_ids, offsets=offsets,
                        dist=cat_d, elev=cat_e)
    log.info("wrote %s", PROFILES_NPZ.name)
    return {int(e): (out_dist[i], out_elev[i]) for i, e in enumerate(edge_ids)}
