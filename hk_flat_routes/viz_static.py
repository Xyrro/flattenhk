"""Publication-quality static map: Hong Kong's low-elevation backbone.

The map answers one question -- *which streets form Hong Kong's
low-elevation transportation backbone?* -- so the visual hierarchy is built
to answer it and nothing else:

* a hillshade computed from the same LiDAR terrain model the analysis uses,
  so the topography the corridors are threading through is visible;
* the full street network drawn very faintly, present for context but
  carrying almost no visual weight;
* the discovered corridors drawn boldly, with line width scaled by corridor
  importance and colour by mean gradient;
* the critical passes marked, because a backbone is defined as much by where
  it must cross a ridge as by where it runs level;
* labels for the major corridors and for the neighborhoods.

What differs from San Francisco is the sheet of paper.  San Francisco is a
compact square of land that fills a portrait page.  Hong Kong is 65 km by
48 km of which most is sea and country park, and a build may cover all of
it or only the urban core, so nothing about the layout is fixed here: the
frame is fitted to the street network that is actually present, the page is
landscape or portrait as that frame demands, and the scale bar, the label
spacing and the tint of the relief are all derived from it.  Land is drawn
wherever the terrain model has ground (it has none over the sea), not from
the neighborhood polygons, which in Hong Kong do not tile the land.
"""
from __future__ import annotations

import textwrap

import numpy as np

from .config import CITY_NAME, OUTPUT_DIR
from .utils import get_logger, step

log = get_logger("hk_flat_routes.viz_static")

STATIC_PNG = OUTPUT_DIR / "hk_flat_backbone.png"
STATIC_PDF = OUTPUT_DIR / "hk_flat_backbone.pdf"
GRADE_PNG = OUTPUT_DIR / "hk_street_grades.png"

#: Longer side of either figure, in inches, and the least the shorter side
#: may be as a share of it (a very long, thin network still needs room for
#: the title and the legend).
FIG_LONG_IN = 16.0
FIG_MIN_ASPECT = 0.6
#: Margin around the network, as a share of the frame's longer side.
FRAME_PAD = 0.035
#: The relief is read at no more than this many cells along the longer side
#: of the frame; finer than the page can show only costs memory.
RELIEF_MAX_PX = 2600
#: Vertical exaggeration of the hillshade.  San Francisco uses 2.1 for hills
#: of under 300 m; with Hong Kong's, several times as high and far steeper,
#: that turns every north-west slope white and every south-east one black.
RELIEF_Z_FACTOR = 0.9
#: The elevation tint runs from sea level to this percentile of the land's
#: height inside the frame, so the urban core and the whole territory each
#: use the full range of the ramp; the few summits above it saturate.
TINT_PERCENTILE = 98.0

#: How a mode reads in a subtitle (upstream's "{mode}ing" made "bikeing").
MODE_WORDS = {"walk": "walking", "bike": "cycling"}

#: The line under each figure.  The terrain model is used under the CSDI
#: Portal's terms, which ask that both the Government and the portal be
#: named as the source, so the portal is named here as it is in the README.
CREDITS = ("Streets \N{COPYRIGHT SIGN} OpenStreetMap contributors, via Overture "
           "Maps  \N{MIDDLE DOT}  Elevation: CEDD 2020 LiDAR DTM, HKSAR "
           "Government, via the CSDI Portal  \N{MIDDLE DOT}  Planning areas: "
           "Town Planning Board")

#: Passes marked on the backbone map, per landmass.
PASSES_PER_LANDMASS = 8

#: Fonts that can draw Traditional Chinese, in order of preference: the
#: system faces of macOS, Windows and the common Linux packages.
CJK_FAMILIES = (
    "PingFang HK", "PingFang TC", "Hiragino Sans TC", "Hiragino Sans CNS",
    "Heiti TC", "Noto Sans CJK HK", "Noto Sans CJK TC", "Noto Sans HK",
    "Noto Sans TC", "Noto Sans CJK JP", "Noto Sans CJK SC",
    "Source Han Sans HK", "Source Han Sans TC", "Source Han Sans",
    "Microsoft JhengHei", "WenQuanYi Zen Hei", "WenQuanYi Micro Hei",
    "AR PL UMing HK", "LiHei Pro", "STHeiti", "Arial Unicode MS",
)


# --------------------------------------------------------------------------
def hillshade(dem: np.ndarray, res: float = 1.0, azimuth: float = 315.0,
              altitude: float = 45.0, z_factor: float = 1.6) -> np.ndarray:
    """Standard Horn hillshade, returned in 0..1."""
    # explicit axes: np.gradient's scalar-spacing return shape varies by
    # numpy version, and the axis form is unambiguous. Rows run north to
    # south, so dy is the southward slope, which is what the ESRI/Horn
    # aspect below expects. (The arguments were swapped once; the light then
    # came from the south-east and every hill read as a hollow.)
    dy = np.gradient(dem, res, axis=0)
    dx = np.gradient(dem, res, axis=1)
    slope = np.arctan(z_factor * np.hypot(dx, dy))
    aspect = np.arctan2(dy, -dx)
    az = np.radians(360.0 - azimuth + 90.0)
    alt = np.radians(altitude)
    hs = (np.sin(alt) * np.cos(slope)
          + np.cos(alt) * np.sin(slope) * np.cos(az - aspect))
    return np.clip(hs, 0, 1)


# --------------------------------------------------------------------------
# the frame
# --------------------------------------------------------------------------
def map_extent(edges, mode: str = "walk", pad: float = FRAME_PAD) -> tuple:
    """Frame of the map: the network that is present, plus a margin.

    Returns ``(xmin, ymin, xmax, ymax)`` in the edges' CRS.  The network,
    not the neighborhood set or the study bounding box, decides the frame:
    a build of the urban core should fill the page with the urban core, and
    the neighborhoods of a place the build does not reach have nothing to
    show.
    """
    col = f"{mode}_ok"
    net = edges[edges[col]] if col in edges.columns else edges
    if not len(net):
        net = edges
    x0, y0, x1, y1 = (float(v) for v in net.total_bounds)
    m = max(pad * max(x1 - x0, y1 - y0), 250.0)
    return (x0 - m, y0 - m, x1 + m, y1 + m)


def figure_size(extent, long_side: float = FIG_LONG_IN) -> tuple[float, float]:
    """Page size (inches) for a frame: landscape or portrait as it demands."""
    w, h = extent[2] - extent[0], extent[3] - extent[1]
    aspect = w / h if h > 0 else 1.0
    if aspect >= 1.0:
        return (long_side, long_side * max(1.0 / aspect, FIG_MIN_ASPECT))
    return (long_side * max(aspect, FIG_MIN_ASPECT), long_side)


def line_weight(extent, reference_m: float = 18000.0, floor: float = 0.45) -> float:
    """Factor on corridor widths and markers for a frame of this size.

    Upstream's widths were chosen for San Francisco, a city with about
    the land area of Hong Kong's urban core (121 km2 against 128), and a
    frame of up to 18 km, which is the urban core, keeps them.  On a sheet
    showing the whole territory the same widths would turn the corridors
    of the north shore into one blot, so they thin in proportion to the
    frame, though never below ``floor``: a hairline corridor stops reading
    as the subject of the map.
    """
    long_side = max(extent[2] - extent[0], extent[3] - extent[1])
    return float(min(1.0, max(floor, reference_m / long_side)))


def _nice_length(target_m: float) -> float:
    """The round distance (1, 2 or 5 times a power of ten) nearest below."""
    if target_m <= 0:
        return 1000.0
    power = 10.0 ** np.floor(np.log10(target_m))
    for mult in (5.0, 2.0, 1.0):
        if mult * power <= target_m:
            return mult * power
    return power


# --------------------------------------------------------------------------
# relief
# --------------------------------------------------------------------------
def _load_hillshade(extent, max_px: int = RELIEF_MAX_PX):
    """Hillshade and elevation of the land inside ``extent``.

    Read from the 10 m overview of the terrain model, which has no data
    over the sea, so the coastline needs no mask from elsewhere.  (Upstream
    masked to the union of the neighborhoods; Hong Kong's planning areas
    leave out the country parks, which is most of the relief.)

    Returns ``(hillshade, elevation, valid, raster_extent)`` with the last
    in matplotlib's ``(left, right, bottom, top)`` order, or ``None`` when
    there is no terrain model to read.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import from_bounds

    from .elevation import DEM_MOSAIC, build_dem_overview

    # Asked for every time, not only when the file is missing: whether the
    # tiles have changed since the overview was built is the overview's own
    # business, and one built during an interrupted download otherwise
    # stayed cropped to those tiles in both static maps.  Where it cannot
    # be rebuilt (the processed data kept, the tiles gone) the one on disk
    # is still the best there is.
    try:
        build_dem_overview()
    except Exception as exc:                     # no tiles on disk, usually
        if not DEM_MOSAIC.exists():
            log.warning("no terrain overview for the relief (%s)", exc)
            return None
        log.warning("the terrain overview could not be refreshed (%s); the "
                    "relief is drawn from the one on disk", exc)

    with rasterio.open(DEM_MOSAIC) as src:
        b = src.bounds
        left, right = max(extent[0], b.left), min(extent[2], b.right)
        bottom, top = max(extent[1], b.bottom), min(extent[3], b.top)
        if right <= left or top <= bottom:
            log.warning("the terrain overview does not reach the map frame")
            return None
        win = from_bounds(left, bottom, right, top, src.transform)
        win = win.round_offsets().round_lengths()
        step_px = max(1, int(np.ceil(max(win.width, win.height) / max_px)))
        h = max(1, int(win.height // step_px))
        w = max(1, int(win.width // step_px))
        # a masked read, so that averaging down does not blend the nodata
        # value of the sea into the shore
        dem = src.read(1, window=win, out_shape=(h, w), masked=True,
                       resampling=Resampling.average)
        wb = src.window_bounds(win)
        res = src.res[0] * win.width / w

    valid = ~np.ma.getmaskarray(dem)
    dem = np.where(valid, dem.filled(np.nan).astype("float64"), np.nan)
    valid &= np.isfinite(dem)
    if not valid.any():
        return None
    # shade a surface that carries on past the coast at the height of the
    # nearest land, so the shoreline does not read as a cliff
    from scipy.ndimage import distance_transform_edt
    idx = distance_transform_edt(~valid, return_distances=False,
                                 return_indices=True)
    filled = dem[tuple(idx)]
    hs = hillshade(filled, res=res, z_factor=RELIEF_Z_FACTOR)
    return hs, dem, valid, (wb[0], wb[2], wb[1], wb[3])


def _draw_relief(ax, relief, tint: bool = True) -> None:
    """Land as tinted, shaded relief; a no-op without a terrain model."""
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    if relief is None:
        return
    hs, dem, valid, extent = relief
    land = np.where(valid, 1.0, np.nan)
    if tint:
        top = max(float(np.nanpercentile(dem, TINT_PERCENTILE)), 50.0)
        elev_cmap = LinearSegmentedColormap.from_list(
            "hk_terrain", ["#f2efe6", "#e8e1cf", "#ddd2b6", "#cfc09b",
                           "#bfa87f"])
        ax.imshow(dem, extent=extent, origin="upper", cmap=elev_cmap,
                  norm=Normalize(0.0, top), alpha=1.0,
                  interpolation="bilinear", zorder=1)
        ax.imshow(hs * land, extent=extent, origin="upper", cmap="gray",
                  alpha=0.46, interpolation="bilinear", vmin=0.15, vmax=0.95,
                  zorder=2)
    else:
        flat = LinearSegmentedColormap.from_list("hk_land", ["#f8f7f4", "#f8f7f4"])
        ax.imshow(land, extent=extent, origin="upper", cmap=flat,
                  interpolation="nearest", zorder=1)
        ax.imshow(hs * land, extent=extent, origin="upper", cmap="gray",
                  alpha=0.16, interpolation="bilinear", vmin=0.15, vmax=0.95,
                  zorder=2)


# --------------------------------------------------------------------------
# lettering
# --------------------------------------------------------------------------
def _has_cjk(text) -> bool:
    return isinstance(text, str) and any(
        "⺀" <= ch <= "鿿" or "豈" <= ch <= "﫿" for ch in text)


def cjk_font(sample: str = ""):
    """A font that can really draw the Chinese in ``sample``, or ``None``.

    matplotlib's default face has no Chinese glyphs and draws an empty box
    for each, with only a warning to show for it.  A name in a font list is
    no proof either (a Japanese face lacks 鰂 and 脷), so each candidate is
    opened and asked for every character that will be drawn.  When nothing
    qualifies the caller draws English only.
    """
    from matplotlib import font_manager as fm
    from matplotlib.ft2font import FT2Font

    need = {ch for ch in sample if ord(ch) > 0x2E7F} or set("香港")
    by_name: dict[str, list[str]] = {}
    for f in fm.fontManager.ttflist:
        by_name.setdefault(f.name, []).append(f.fname)
    for family in CJK_FAMILIES:
        for path in by_name.get(family, ()):
            try:
                face = FT2Font(path)
                if all(face.get_char_index(ord(ch)) for ch in need):
                    return fm.FontProperties(fname=path)
            except Exception:                    # an unreadable font file
                continue
    return None


def _outline(lw: float):
    import matplotlib.patheffects as pe
    return [pe.withStroke(linewidth=lw, foreground="white", alpha=0.9)]


class _Labels:
    """Places text on a map and refuses any that would cover another.

    San Francisco's 37 neighborhoods each fit a label; Hong Kong's urban
    planning areas are a kilometre across, and on a sheet that shows the
    whole territory their names would pile into one black knot.  Each label
    is drawn, measured as rendered, and removed again if its box overlaps
    one already kept, so callers add labels in order of importance.
    """

    def __init__(self, ax, margin_px: float = 6.0):
        self.ax = ax
        self.fig = ax.figure
        self.margin = margin_px
        self.kept: list = []
        self._renderer = None

    def _box(self, texts):
        from matplotlib.transforms import Bbox
        if self._renderer is None:
            self.fig.canvas.draw()
            self._renderer = self.fig.canvas.get_renderer()
        box = Bbox.union([t.get_window_extent(self._renderer) for t in texts])
        return box.expanded(1.0, 1.0).padded(self.margin)

    def add(self, x: float, y: float, lines: list[tuple[str, dict]],
            above: bool = False) -> bool:
        """Draw a label of one or more lines at (x, y), each its own style.

        Each line is ``(text, text_kwargs)``; a line may itself be wrapped.
        The block is centred on the point, or set just ``above`` it (for a
        label that names a line and should not sit on it).  Returns False,
        leaving nothing drawn, if the label falls outside the frame or
        collides with one already placed.
        """
        x0, x1 = sorted(self.ax.get_xlim())
        y0, y1 = sorted(self.ax.get_ylim())
        if not (x0 < x < x1 and y0 < y < y1):
            return False
        heights = [kw.get("fontsize", 7.0) * 1.25 * (text.count("\n") + 1)
                   for text, kw in lines]
        top = sum(heights) + 3.0 if above else sum(heights) / 2.0
        texts = []
        offset = 0.0
        for (text, kw), height in zip(lines, heights):
            t = self.ax.annotate(text, (x, y), xytext=(0, top - offset),
                                 textcoords="offset points", ha="center",
                                 va="top", annotation_clip=False,
                                 linespacing=1.05, **kw)
            texts.append(t)
            offset += height
        box = self._box(texts)
        frame = self.ax.get_window_extent(self._renderer)
        inside = (box.x0 >= frame.x0 and box.x1 <= frame.x1
                  and box.y0 >= frame.y0 and box.y1 <= frame.y1)
        if not inside or any(box.overlaps(k) for k in self.kept):
            for t in texts:
                t.remove()
            return False
        self.kept.append(box)
        return True


def _wrap(name: str, width: int = 18) -> str:
    """Break a long neighborhood name over lines: 'Tsz Wan Shan,\\nDiamond Hill ...'."""
    return "\n".join(textwrap.wrap(name, width=width, break_long_words=False)) \
        or name


def _label_neighborhoods(labels: _Labels, neighborhoods, font_zh, extent) -> int:
    """Name the neighborhoods, largest first, in English and Chinese.

    The Chinese name goes on a second line when a font that can draw it was
    found; where the two lines together do not fit, the English name is
    tried alone.
    """
    style = dict(fontsize=6.2, color="#33404b", zorder=10, alpha=0.95,
                 style="italic", path_effects=_outline(1.9))
    style_zh = dict(fontsize=6.0, color="#33404b", zorder=10, alpha=0.95,
                    path_effects=_outline(1.9))
    nb = neighborhoods
    if "area_km2" in nb.columns:
        nb = nb.sort_values("area_km2", ascending=False)
    placed = 0
    for _, r in nb.iterrows():
        c = r.geometry.representative_point()
        if not (extent[0] < c.x < extent[2] and extent[1] < c.y < extent[3]):
            continue
        en = [(_wrap(str(r["neighborhood"])), style)]
        zh = r.get("neighborhood_zh")
        if font_zh is not None and _has_cjk(zh):
            both = en + [(str(zh), dict(style_zh, fontproperties=font_zh))]
            if labels.add(c.x, c.y, both):
                placed += 1
                continue
        placed += bool(labels.add(c.x, c.y, en))
    return placed


def _scalebar(ax, extent):
    """A scale bar of a round length, bottom right, clear of the legend."""
    w, h = extent[2] - extent[0], extent[3] - extent[1]
    length_m = _nice_length(0.16 * w)
    x0 = extent[2] - length_m - 0.03 * w
    y0 = extent[1] + 0.035 * h
    ax.plot([x0, x0 + length_m], [y0, y0], color="#14202b", lw=2.4,
            solid_capstyle="butt", zorder=12)
    return ax.annotate(f"{length_m/1000:g} km", (x0 + length_m / 2, y0),
                       xytext=(0, 4), textcoords="offset points", fontsize=7.4,
                       ha="center", va="bottom", color="#14202b", zorder=12,
                       path_effects=_outline(2.0))


def _north_arrow(ax):
    """A small north arrow, top right: the grid's north is up the page."""
    # the letter sits at the foot (``xytext``) and the head at ``xy``, above
    return ax.annotate(
        "N", xy=(0.975, 0.972), xytext=(0.975, 0.925),
        xycoords="axes fraction", textcoords="axes fraction",
        ha="center", va="center", fontsize=8.5, fontweight="bold",
        color="#14202b", zorder=12, path_effects=_outline(2.0),
        arrowprops=dict(arrowstyle="-|>", color="#14202b", lw=1.2,
                        shrinkA=1.0, shrinkB=0.0))


def _frame(ax, extent) -> None:
    ax.set_xlim(extent[0], extent[2])
    ax.set_ylim(extent[1], extent[3])
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    # geopandas labels a projected CRS's axes "Easting [metre]" and so on
    ax.set_xlabel(""); ax.set_ylabel("")
    for sp in ax.spines.values():
        sp.set_visible(False)


def _plot_lines(ax, gdf, **kw) -> None:
    """Draw line geometries as one collection (fast for 100,000s of edges).

    The collection is rasterised, which only matters in the PDF: a quarter
    of a million street centrelines as vector paths make a file no viewer
    opens gladly, while as an image under the vector corridors and labels
    they look the same on the page.
    """
    from matplotlib.collections import LineCollection

    segs = []
    for g in gdf.geometry.values:
        if g is None or g.is_empty:
            continue
        for part in getattr(g, "geoms", [g]):
            segs.append(np.asarray(part.coords)[:, :2])
    if segs:
        ax.add_collection(LineCollection(segs, rasterized=True, **kw))


# --------------------------------------------------------------------------
def relative_score(cor) -> np.ndarray:
    """Each corridor's score as a share of the best on its own landmass.

    Corridors are found landmass by landmass (``corridors.detect_corridors``)
    and their scores count pairs of that landmass only, so the best corridor
    on Lantau, with five neighborhoods, scores a small fraction of the best
    in Kowloon.  Drawn on one scale it would be a hairline and would never
    be named.  Without a ``landmass`` column the best corridor overall is
    the yardstick, as upstream has it.
    """
    score = cor["total_score"].to_numpy("float64")
    if not len(cor):
        return score
    if "landmass" in cor.columns and cor["landmass"].notna().any():
        best = cor.groupby(cor["landmass"].fillna(-1))["total_score"].transform("max")
        best = best.to_numpy("float64")
    else:
        best = np.full(len(cor), score.max())
    return score / np.where(best > 0, best, 1.0)


def leading_passes(passes, edges, mode: str = "walk",
                   per_landmass: int = PASSES_PER_LANDMASS):
    """The passes to mark: the leading few of each landmass.

    The pass table is ranked by the number of neighborhood pairs a pass
    binds, and a landmass with few neighborhoods has few pairs to offer.
    Upstream marks the first sixteen of the table; on the whole territory
    twelve of those are on the mainland, four on Hong Kong Island and none
    on Lantau, whose four passes bind at most four pairs each.  So the
    passes are taken landmass by landmass, as the corridors are, each in
    the table's order.

    A pass is on the landmass of the street it lies on, found here as a
    connected piece of the mode's network.  Where that cannot be worked out
    (a pass table without ``edge_id``, edges without their end nodes) the
    head of the table is returned, twice ``per_landmass`` long, which is
    upstream's sixteen.
    """
    import pandas as pd

    if not len(passes) or "edge_id" not in passes.columns \
            or not {"edge_id", "u", "v"} <= set(edges.columns):
        return passes.head(2 * per_landmass)
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    col = f"{mode}_ok"
    net = edges[edges[col]] if col in edges.columns else edges
    n = len(net)
    if not n:
        return passes.head(2 * per_landmass)
    codes, nodes = pd.factorize(pd.concat([net["u"], net["v"]], ignore_index=True))
    graph = coo_matrix((np.ones(n, dtype="int8"), (codes[:n], codes[n:])),
                       shape=(len(nodes),) * 2)
    _, piece = connected_components(graph, directed=False)
    of_edge = pd.Series(piece[codes[:n]], index=net["edge_id"].to_numpy())
    of_edge = of_edge[~of_edge.index.duplicated()]
    # a pass on no street of this mode (none should be) is its own group
    landmass = passes["edge_id"].map(of_edge).fillna(-1).to_numpy()
    return passes.groupby(landmass, sort=False).head(per_landmass)


def make_backbone_map(corridors, edges, neighborhoods, passes=None,
                      mode: str = "walk", top_n: int | None = None,
                      label_n: int = 16, extent=None) -> tuple:
    """Draw the flagship static map.

    Every corridor of the mode is drawn (or the ``top_n`` most important,
    if given), its width set by its standing on its own landmass, and up to
    ``label_n`` are named, the leading corridor of each landmass first.
    (San Francisco draws 22 and names 11 on one landmass; here the same
    sheet has to carry Hong Kong Island, Kowloon and the New Territories
    and Lantau, each with corridors of its own.)  The passes marked are the
    leading few of each landmass too (``leading_passes``).  ``extent``
    overrides the frame, which otherwise fits the network.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D

    if len(corridors) and "mode" in corridors.columns:
        cor = corridors[corridors["mode"] == mode].copy()
        if cor.crs is not None and cor.crs != edges.crs:
            cor = cor.to_crs(edges.crs)
        cor = cor.assign(_rel=relative_score(cor))
        cor = cor.sort_values(["_rel", "total_score"], ascending=False)
        if top_n is not None:
            cor = cor.head(top_n)
    else:
        cor = corridors.iloc[0:0].assign(_rel=[])

    with step("rendering the static backbone map", log):
        extent = tuple(extent) if extent is not None else map_extent(edges, mode)
        relief = _load_hillshade(extent)

        fig, ax = plt.subplots(figsize=figure_size(extent), dpi=220)
        fig.patch.set_facecolor("#f7f5f0")
        ax.set_facecolor("#dfe8ef")                       # water

        # --- terrain: hillshade tinted by elevation -------------------
        _draw_relief(ax, relief)

        # --- neighborhoods -------------------------------------------
        neighborhoods.boundary.plot(ax=ax, color="#ffffff", linewidth=1.0,
                                    alpha=0.8, zorder=3)

        # --- full street network: present for context, low weight -----
        net = edges[edges[f"{mode}_ok"]]
        _plot_lines(ax, net, colors="#59646f", linewidths=0.20, alpha=0.42,
                    zorder=4)

        # --- steep street emphasis (the walls the backbone avoids) ----
        rel = net["grade_reliable"] if "grade_reliable" in net.columns else True
        steep = net[(net["max_abs_grade"] >= 0.12) & rel]
        _plot_lines(ax, steep, colors="#a8321f",
                    linewidths=0.5 * max(line_weight(extent), 0.6), alpha=0.5,
                    zorder=5)

        # --- the corridors -------------------------------------------
        weight = line_weight(extent)
        grade_cmap = LinearSegmentedColormap.from_list(
            "flatness", ["#08306b", "#1f78b4", "#41b6c4", "#7fcdbb"])
        for _, r in cor.iloc[::-1].iterrows():        # the strongest on top
            lw = weight * (1.9 + 5.4 * r["_rel"] ** 0.65)
            col = grade_cmap(min(r["mean_abs_grade"] / 0.035, 1.0))
            gs = getattr(r.geometry, "geoms", [r.geometry])
            for g in gs:
                xs, ys = g.xy
                ax.plot(xs, ys, color="white", linewidth=lw + 1.7 * weight,
                        solid_capstyle="round", alpha=0.85, zorder=6)
                ax.plot(xs, ys, color=col, linewidth=lw,
                        solid_capstyle="round", zorder=7)

        # --- critical passes -----------------------------------------
        if passes is not None and len(passes):
            pz = leading_passes(passes, edges, mode)
            if pz.crs is not None and pz.crs != edges.crs:
                pz = pz.to_crs(edges.crs)
            pts = pz.geometry.interpolate(0.5, normalized=True)
            ax.scatter([p.x for p in pts], [p.y for p in pts],
                       s=54 * max(weight, 0.6), marker="^",
                       facecolor="#ffd166", edgecolor="#4a3b10",
                       linewidth=0.8, zorder=9)

        # --- frame (before the labels: they are measured against it) --
        _frame(ax, extent)
        ax.set_title(f"{CITY_NAME}'s low-elevation backbone",
                     fontsize=21, fontweight="bold", color="#14202b",
                     loc="left", pad=26)
        ax.annotate(f"Streets that repeatedly carry low-gradient routes "
                    f"between neighbourhoods  \N{MIDDLE DOT}  "
                    f"{MODE_WORDS.get(mode, mode)} network",
                    (0.0, 1.0), xytext=(0, 6), xycoords="axes fraction",
                    textcoords="offset points", fontsize=10.2,
                    color="#4a5560", va="bottom")

        legend = [
            Line2D([], [], color="#1f78b4", lw=5.0,
                   label="Flat corridor (width = importance on its landmass)"),
            Line2D([], [], color="#08306b", lw=3.0,
                   label="Corridor mean grade < 1%"),
            Line2D([], [], color="#7fcdbb", lw=3.0,
                   label="Corridor mean grade ~3.5%"),
            Line2D([], [], color="#b3402f", lw=1.6,
                   label="Street steeper than 12%"),
            Line2D([], [], color="#5d6b78", lw=0.8, alpha=0.5,
                   label="All other streets"),
            Line2D([], [], color="none", marker="^", markersize=7,
                   markerfacecolor="#ffd166", markeredgecolor="#4a3b10",
                   label="Critical pass (lowest crossing)"),
        ]
        leg = ax.legend(handles=legend, loc="lower left", fontsize=7.6,
                        frameon=True, facecolor="white", framealpha=0.92,
                        edgecolor="#c9cdd2", borderpad=0.7)
        leg.set_zorder(13)

        ax.text(0.995, -0.012, CREDITS, transform=ax.transAxes, fontsize=6.4,
                color="#6a747d", ha="right", va="top")

        furniture = [_scalebar(ax, extent), _north_arrow(ax)]
        fig.tight_layout()

        # --- labels ---------------------------------------------------
        names = " ".join(str(v) for v in neighborhoods.get(
            "neighborhood_zh", [])) + " ".join(
            str(v) for v in cor.get("corridor_name", []))
        font_zh = cjk_font(names)
        if font_zh is None:
            log.info("no font with Chinese glyphs found; labels are English only")
        labels = _Labels(ax)
        # keep clear of the legend, the scale bar and the north arrow
        labels.kept += [labels._box([item]) for item in [leg] + furniture]

        # corridors first: they are what the map is about
        used_labels: set[str] = set()
        for _, r in cor.iterrows():
            if len(used_labels) >= label_n:
                break
            label = str(r["corridor_name"]).split(" - ")[0]
            if label in used_labels or label == "unnamed corridor":
                continue
            style = dict(fontsize=8.0, fontweight="bold", color="#062a5c",
                         zorder=11, path_effects=_outline(2.8))
            if _has_cjk(label):
                if font_zh is None:
                    continue
                style["fontproperties"] = font_zh
                style["fontsize"] = 8.0
            g = r.geometry
            gs = list(getattr(g, "geoms", [g]))
            longest = max(gs, key=lambda x: x.length)
            # try a few positions along the corridor and keep the first that
            # is not crowding an existing label
            for frac in (0.55, 0.3, 0.75, 0.45, 0.9, 0.15):
                p = longest.interpolate(frac, normalized=True)
                if labels.add(p.x, p.y, [(label, style)], above=True):
                    used_labels.add(label)
                    break
        n_nb = _label_neighborhoods(labels, neighborhoods, font_zh, extent)
        log.info("  labelled %d corridors and %d neighborhoods",
                 len(used_labels), n_nb)

        fig.savefig(STATIC_PNG, bbox_inches="tight", facecolor=fig.get_facecolor())
        # suppressing the PDF creation date keeps the output reproducible
        fig.savefig(STATIC_PDF, bbox_inches="tight", facecolor=fig.get_facecolor(),
                    metadata={"CreationDate": None})
        plt.close(fig)
    log.info("wrote %s and %s", STATIC_PNG.name, STATIC_PDF.name)
    return STATIC_PNG, STATIC_PDF


# --------------------------------------------------------------------------
def make_grade_map(edges, neighborhoods, mode: str = "walk", extent=None):
    """Secondary figure: the whole network coloured by gradient."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
    from matplotlib.cm import ScalarMappable

    with step("rendering the territory-wide grade map", log):
        net = edges[edges[f"{mode}_ok"]].copy()
        net["g"] = net["max_abs_grade"].fillna(0.0)
        bounds_g = [0, 0.03, 0.05, 0.08, 0.10, 0.15, 0.60]
        cmap = LinearSegmentedColormap.from_list(
            "grades", ["#2c7bb6", "#8fd0c0", "#ffffbf", "#fdae61", "#e8613c",
                       "#7f1d1d"], N=6)
        norm = BoundaryNorm(bounds_g, cmap.N)

        extent = tuple(extent) if extent is not None else map_extent(edges, mode)
        fig, ax = plt.subplots(figsize=figure_size(extent), dpi=200)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#eef3f6")
        # the land, faintly shaded; planning-area outlines over it
        _draw_relief(ax, _load_hillshade(extent), tint=False)
        neighborhoods.boundary.plot(ax=ax, color="#d8dde1", linewidth=0.6,
                                    zorder=2)
        for lo, hi in zip(bounds_g[:-1], bounds_g[1:]):
            sel = net[(net["g"] >= lo) & (net["g"] < hi)]
            if sel.empty:
                continue
            lw = 0.22 if hi <= 0.05 else (0.38 if hi <= 0.10 else 0.62)
            _plot_lines(ax, sel, colors=[cmap(norm(lo + 1e-9))], linewidths=lw,
                        alpha=0.95, zorder=3 + bounds_g.index(lo))

        _frame(ax, extent)
        ax.set_title(f"{CITY_NAME} street gradients", fontsize=19,
                     fontweight="bold", loc="left", pad=24)
        ax.annotate("Maximum sampled gradient per street segment, from the "
                    f"CEDD 2020 LiDAR terrain model  \N{MIDDLE DOT}  "
                    f"{MODE_WORDS.get(mode, mode)} network",
                    (0.0, 1.0), xytext=(0, 6), xycoords="axes fraction",
                    textcoords="offset points", fontsize=9.6,
                    color="#4a5560", va="bottom")
        ax.text(0.995, -0.012, CREDITS, transform=ax.transAxes, fontsize=6.4,
                color="#6a747d", ha="right", va="top")
        sm = ScalarMappable(norm=norm, cmap=cmap)
        cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.015,
                          boundaries=bounds_g, ticks=bounds_g)
        cb.ax.set_yticklabels([f"{v:.0%}" for v in bounds_g])
        cb.set_label("maximum gradient", fontsize=9)
        _scalebar(ax, extent)
        _north_arrow(ax)
        fig.tight_layout()
        fig.savefig(GRADE_PNG, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
    log.info("wrote %s", GRADE_PNG.name)
    return GRADE_PNG
