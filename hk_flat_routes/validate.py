"""Validation of the elevation and routing model against known ground truth.

Upstream (flattensf) checks San Francisco against an independent USGS DEM,
the published grades of its famously steep streets and the corridors local
knowledge calls flat.  Hong Kong has better ground truth than that, most of
it official, so the same questions are put to sharper references:

1.  **Absolute elevation** -- the model's street elevations against the
    Survey and Mapping Office's bench marks: 1,862 levelled points whose
    heights above the Principal Datum are known to the millimetre.  This
    replaces upstream's DEM-against-DEM comparison as the primary check,
    because it tests the thing the project actually uses (the elevation of
    a street) against something that was measured rather than modelled.

2.  **An independent terrain model** -- the LiDAR against the Lands
    Department's 5 m photogrammetric model, upstream's check in its
    original form.  The 5 m model is far too coarse to judge anything by;
    what it can show is that the LiDAR tiles are georeferenced and in the
    expected vertical datum everywhere, not only where a bench mark stands.

3.  **Steep streets** -- computed gradients against three kinds of
    documented figure: the Transport Department's maximum gradients for
    its long steep downhill roads, the ratios signed and published for the
    old stepped streets above Central and Sai Ying Pun, and the mean
    gradient between two bench marks on the same road.

4.  **Level corridors** -- the streets that must come out level (the tram
    line, Nathan Road, and the contour roads cut into the hillside that are
    Hong Kong's version of flat: level, but a hundred metres up), and the
    ones that must *not* (long roads that climb steadily and gently).

5.  **Round, not over** -- a trip where anyone who walks it knows to follow
    the contour road instead of crossing the hill.  The model has to find
    that for itself.

6.  **The gaps** -- the passes where roads cross the island's spine and the
    Kowloon ridge, against their published heights.

7.  **Internal invariants** -- identities the tables must satisfy whatever
    the terrain.

The model is *not* tuned to make these pass; where a target disagrees, the
disagreement is reported with its numbers.  A reference outside the area a
run covers is reported as "not in this run", never as a failure, so the same
report works for the whole territory and for a test window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (CRS_GEOGRAPHIC, ELEVATION, OUTPUT_DIR,
                     SIDEPATH_SUBCLASSES)
from .utils import get_logger, step

log = get_logger("hk_flat_routes.validate")

VALIDATION_MD = OUTPUT_DIR / "validation_report.md"

DRIVABLE = ("residential", "living_street", "tertiary", "secondary",
            "primary", "trunk", "unclassified")
#: Classes whose centreline a bench mark is compared with: the drivable
#: ones, plus service roads and the paved roads closed to traffic (Bowen
#: Road, Hatton Road, Mount Parker Road), which carry many of the hill marks.
ROAD_CLASSES = DRIVABLE + ("pedestrian", "service")

# --------------------------------------------------------------------------
# reference data
#
# Every figure below comes from the validation research for this port (see
# the README's provenance section); the source of each table is given with
# it.  Heights of bench marks are *not* typed in here: they are read from
# the official file at run time, by station number.
# --------------------------------------------------------------------------
#: A bench mark further than this (m) from every road centreline is not
#: compared: it is on a slope, a wall or a building, not on the road.
BENCHMARK_MAX_OFFSET_M = 12.0
#: How far (m) a mark may be from a *named* road to count as being on it.
#: Wider than the limit above because the question is only which road the
#: mark belongs to, and dual carriageways are mapped as two lines.
BENCHMARK_STREET_OFFSET_M = 30.0
#: How far (m) a documented junction may be from the street it is on.
JUNCTION_SNAP_M = 60.0
#: A measured section whose length differs from the stated one by more than
#: this fraction is not the documented section, and is not judged.
SECTION_LENGTH_TOLERANCE = 0.40
#: Streets are cut short at the edge of the area a run covers.  A reference
#: with an end this close (m) to that edge is "not in this run" when it
#: cannot be matched, not a failure of the model.
RUN_EDGE_MARGIN_M = 300.0

#: Long steep downhill roads reviewed by the Transport Department in 2008
#: (Legislative Council paper CB(1)1977/07-08(04), annex): street, stated
#: maximum gradient, stated length (m), from (the top), to (the bottom), and
#: the two ends as (lon, lat) where the research could place them on the
#: map.  The annex lists 29 sections; these are the ones with at least one
#: end located, less two.  Clear Water Bay Road from Fei Ngo Shan Road to
#: New Clear Water Bay Road is stated as 1,350 m and the two ends the
#: research found for it are under 500 m apart along the road.  And the
#: annex's second Chai Wan Road section (10% over 1,000 m, from Tai Tam Road
#: down to the Chai Wan roundabout) has only its top located, and that top
#: is the crest of the road: Chai Wan Road falls away from Tai Tam Road on
#: both sides, so "down from the top" does not say which side is meant
#: (the other side is the first Chai Wan Road section, which is here).  A
#: section with one end is measured for its stated length from that end.
#: Stated gradients are rounded; the research's own cross-checks against
#: bench marks put them good to two or three points.
TD_ANNEX_SECTIONS = 29
TD_STEEP_ROADS = (
    (("Old Peak Road",), 0.17, 560, "Tregunter Path", "Robinson Road",
     (114.15201, 22.27557), (114.15462, 22.27723)),
    (("Shatin Pass Road", "Sha Tin Pass Road"), 0.17, 500, "Fat Chong Temple",
     "Nga Chuk Street", None, (114.19565, 22.34498)),
    (("Aberdeen Reservoir Road",), 0.16, 500, "No. 38", "Aberdeen Main Road",
     None, (114.15572, 22.24933)),
    (("Blue Pool Road",), 0.14, 500, "Tai Hang Road", "No. 98",
     (114.18892, 22.26267), None),
    (("Garden Road",), 0.13, 800, "Robinson Road", "Queen's Road Central",
     (114.15487, 22.27570), (114.16089, 22.27987)),
    (("Lam Kam Road",), 0.13, 800, "Kadoorie Agricultural Research Centre",
     "Route Twisk", None, (114.10454, 22.43149)),
    (("Tai Mo Shan Road",), 0.125, 2000, "Sze Lok Yuen youth hostel",
     "Route Twisk", None, (114.10524, 22.40561)),
    (("Wan Wah Street",), 0.12, 500, "Tsz Wan Shan Road", "Shung Wah Street",
     (114.20023, 22.35042), (114.19892, 22.34629)),
    (("Chai Wan Road",), 0.11, 880, "Tai Tam Road", "Shau Kei Wan Road",
     (114.23336, 22.27097), (114.22819, 22.27697)),
    (("Smithfield",), 0.11, 1000, "Pok Fu Lam Road", "Forbes Street",
     (114.13059, 22.27737), (114.12829, 22.28158)),
    (("Sassoon Road",), 0.11, 520, "Pok Fu Lam Road", "Victoria Road",
     (114.13007, 22.26991), (114.12829, 22.26613)),
    (("Ede Road",), 0.11, 560, "Eastbourne Road", "Cornwall Street",
     (114.17622, 22.34320), (114.17803, 22.34008)),
    (("Magazine Gap Road",), 0.10, 1850, "Peak Road", "Garden Road",
     (114.16554, 22.26927), (114.15502, 22.27539)),
    (("Broadwood Road",), 0.10, 900, "Tai Hang Road", "Link Road",
     (114.18979, 22.26851), (114.18506, 22.27464)),
    (("Pak Fuk Road",), 0.10, 800, "Tin Hau Temple Road", "the Safety Town",
     (114.20270, 22.28842), None),
    (("Cloud View Road",), 0.10, 520, "No. 33", "Tin Hau Temple Road",
     None, (114.19545, 22.28663)),
)
#: The annex does not say over what length its maximum is taken.  The
#: research's cross-checks show it is a sustained figure, not a pitch:
#: Magazine Gap Road is listed at 10% where its bench marks give 10.1% over
#: a kilometre and a half and 11.6% over the steepest quarter kilometre
#: between two of them.  So a stated maximum is expected to lie between the
#: model's gradient over the whole section and its gradient over the
#: section's steepest stretch of this length (m), give or take the rounding.
TD_WINDOW_M = 100.0
TD_TOLERANCE = 0.02

#: Steep streets of Central, the Mid-Levels and Sai Ying Pun with a ratio
#: on a steep-hill sign or in print (1 in N): street, Chinese name, N,
#: section and its two ends.  Sources are the Chinese Wikipedia article on
#: long steep slopes and the street articles, mostly matching OpenStreetMap
#: ``incline`` tags; no primary source was found, so this table carries less
#: weight than the two official ones.  Several are ladder streets: part
#: stepped, part closed to traffic, class ``steps`` in the network.
STEEP_RATIO_STREETS = (
    ("Centre Street", "正街", 4, "Queen's Road West to High Street",
     (114.14229, 22.28692), (114.14218, 22.28516)),
    ("Peel Street", "卑利街", 4, "Staunton Street to Caine Road",
     (114.15234, 22.28238), (114.15167, 22.28171)),
    ("Eastern Street", "東邊街", 5, "Queen's Road West to Bonham Road",
     (114.14372, 22.28687), (114.14357, 22.28445)),
    ("Aberdeen Street", "鴨巴甸街", 5, "Hollywood Road to Caine Road",
     (114.15247, 22.28350), (114.15109, 22.28213)),
    ("Graham Street", "嘉咸街", 5, "Hollywood Road to Staunton Street",
     (114.15306, 22.28256), (114.15264, 22.28214)),
    ("Glenealy", "己連拿利", 5, "Robinson Road to Conduit Road",
     (114.15293, 22.27785), (114.15225, 22.27822)),
    ("Breezy Path", "卑利士道", 5, "Bonham Road to Park Road",
     (114.14655, 22.28363), (114.14535, 22.28382)),
    ("Honiton Road", "漢寧頓道", 5, "up from Bonham Road",
     (114.14066, 22.28420), (114.14138, 22.28391)),
    ("Castle Road", "衛城道", 5, "Seymour Road to Robinson Road",
     (114.14989, 22.28215), (114.14788, 22.28170)),
    ("Western Street", "西邊街", 6, "Queen's Road West to Bonham Road",
     (114.14068, 22.28697), (114.14071, 22.28423)),
    ("Old Bailey Street", "奧卑利街", 6, "Hollywood Road to Caine Road",
     (114.15387, 22.28202), (114.15305, 22.28059)),
    ("Fortress Hill Road", "炮台山道", 8, "King's Road to Tin Hau Temple Road",
     (114.19464, 22.28951), (114.19524, 22.28631)),
)
#: A sign gives the steepest pitch, so these are compared with the steepest
#: stretch of this length (m).  Shorter windows are no use: over 10 m the
#: raw LiDAR already shows artefacts above 30% on these streets.
RATIO_WINDOW_M = 50.0
RATIO_TOLERANCE = 0.04

#: Stretches of road between two bench marks: road names, the mark at the
#: bottom, the mark at the top, and the distance (m) between them along the
#: road as the research measured it on OpenStreetMap centrelines.  The
#: heights, and so the rise, come from the official file.
BENCHMARK_STRETCHES = (
    (("Wan Chai Gap Road",), "213026", "213037", 599),
    (("Hatton Road",), "213158", "213154", 1378),
    (("Tai Mo Shan Road",), "30922", "30925", 1532),
    (("Fei Ngo Shan Road",), "221261", "221263", 568),
    (("Peel Rise",), "210130", "213053", 2968),
    (("Jat's Incline",), "221271", "221265", 2949),
    (("Mount Parker Road",), "210064", "210060", 1953),
    (("Hiram's Highway", "New Hiram's Highway"), "220083", "220084", 400),
    (("Chai Wan Road",), "210009", "210010.1", 458),
    (("Magazine Gap Road",), "213229", "210106.1", 1564),
    (("Route Twisk",), "230174", "230188", 4504),      # from Shek Kong
    (("Route Twisk",), "220293", "230189", 5213),      # from Tsuen Wan
    (("Peak Road",), "213036", "210115", 2926),
    (("Stubbs Road",), "213145", "10622", 1325),
)
BENCHMARK_GRADE_TOLERANCE = 0.01

#: Streets whose levelness is known, and streets known not to be level.
#: Where a name alone is ambiguous a window isolates the corridor:
#: (lon_min, lon_max, lat_min, lat_max) in WGS84.  The evidence printed in
#: the report is computed from the bench marks along each street; the
#: research's own readings were the tram line at 3-7 m between Sai Ying Pun
#: and Admiralty, bench marks of 8.9 and 13.0 m on Nathan Road, eight marks
#: between 119.9 and 124.2 m along Bowen Road and three between 139.8 and
#: 143.7 m along Conduit Road, against Lung Cheung Road's 15.7 to 90.7 m,
#: Kennedy Road's 18.7 to 65.6 m, Victoria Road's 11.5 to 82.0 m and
#: Waterloo Road's 5.6 to 53.6 m.
KNOWN_FLAT = {
    "Tram line: Des Voeux Road West and Central": {
        "streets": ["Des Voeux Road West", "Des Voeux Road Central"],
        "expect": "level"},
    "Tram line: Hennessy Road": {"streets": ["Hennessy Road"], "expect": "level"},
    "Tram line east of Causeway Bay: King's Road": {
        "streets": ["King's Road"], "expect": "level"},
    "Queen's Road Central": {"streets": ["Queen's Road Central"], "expect": "level"},
    "Nathan Road": {"streets": ["Nathan Road"], "expect": "level"},
    "Kwun Tong Road": {"streets": ["Kwun Tong Road"], "expect": "level"},
    "Tai Chung Kiu Road (beside the Shing Mun River)": {
        "streets": ["Tai Chung Kiu Road"], "expect": "level"},
    "Bowen Road (contour road, Mid-Levels East)": {
        "streets": ["Bowen Road"], "expect": "level"},
    "Conduit Road (contour road, Mid-Levels West)": {
        "streets": ["Conduit Road"], "expect": "level"},
    "Lung Cheung Road": {"streets": ["Lung Cheung Road"], "expect": "not level"},
    "Kennedy Road": {"streets": ["Kennedy Road"], "expect": "not level"},
    "Victoria Road": {"streets": ["Victoria Road"], "expect": "not level"},
    "Waterloo Road": {"streets": ["Waterloo Road"], "expect": "not level"},
}
#: A street is called level when it climbs less than this per kilometre
#: (upstream's threshold, here the mean of the two directions) ...
LEVEL_MAX_CLIMB_PER_KM = 15.0
#: ... and stays within a band this tall (m) from end to end.  The second
#: condition is Hong Kong's: a road that rises 75 m in four kilometres,
#: as Lung Cheung Road does, climbs only gently per kilometre and is still
#: nobody's idea of level.
LEVEL_MAX_RANGE_M = 30.0

#: The trip for check 5.  From the University of Hong Kong to the
#: Zoological and Botanical Gardens both ends are about 70 m up the same
#: hillside, two and a half kilometres apart.  The research's rough router
#: sent the shortest route over the hill by Robinson Road and the flattest
#: along the Bonham Road - Caine Road contour; which streets the pipeline's
#: own router uses is measured, not assumed.
CONTOUR_TRIP = {
    "label": "HKU Main Building to the Zoological and Botanical Gardens",
    "from": ("the Main Building of the University of Hong Kong",
             (114.1379, 22.2839)),
    "to": ("the Zoological and Botanical Gardens at Albany Road",
           (114.1566, 22.2777)),
    "round": ("Bonham Road", "Caine Road", "Upper Albert Road"),
    "over": ("Robinson Road",),
}
#: A trip end further than this (m) from the network is outside the run.
TRIP_SNAP_M = 250.0

#: Road passes: name, Chinese name, position and height of the saddle as
#: mapped in OpenStreetMap, and a bench mark on the road down each side
#: (None where there is no mark on the far side).  OpenStreetMap heights
#: for saddles are map readings and differ from the bench marks standing at
#: the same gaps by one to seven metres, so both are shown.
KNOWN_GAPS = (
    ("Wong Nai Chung Gap", "黃泥涌峽", 114.19304, 22.25735, 191.0,
     ("210047", "210066.1")),
    ("Tai Tam Gap", "大潭峽", 114.22816, 22.25746, 152.0, ("213136", "10287")),
    ("Magazine Gap", "馬己仙峽", 114.16551, 22.26922, 285.0,
     ("210105.1", "213035")),
    ("Wan Chai Gap", "灣仔峽", 114.16990, 22.26761, 241.0, ("210108", "213038")),
    ("Victoria Gap", "爐峰峽", 114.14919, 22.27104, 395.0, ("210113", "213056")),
    ("Quarry Gap", "大風坳", 114.21354, 22.26709, 320.0, ("210060", "210057")),
    ("Sha Tin Pass", "沙田坳", 114.19945, 22.35610, 297.0, None),
    ("Tsuen Kam Au", "荃錦坳", 114.10507, 22.40880, 472.0, ("230189", "230187")),
)
#: The pass the model finds counts as "at the gap" within this distance (m).
GAP_NEAR_M = 250.0
#: A bench mark this close (m) to the mapped saddle is taken as standing at
#: the gap.  No further: on a road climbing at 8% a mark 100 m short of the
#: top is already 8 m below it.
GAP_MARK_M = 100.0
GAP_TOLERANCE_M = 5.0

#: Plausible elevations (m above Principal Datum).  Tai Mo Shan, the
#: highest point of the territory, is 957 m (Lands Department).  The floor
#: is a little below the datum, not at it: the elevation stage treats
#: ground more than a metre under the datum as an excavation caught by the
#: 2020 survey and bridges it (``ElevationConfig.min_ground_elev_m``), and
#: smoothing can carry a profile slightly lower than that.
ELEVATION_RANGE_M = (-5.0, 960.0)


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def _not_run(reason: str, columns=()) -> pd.DataFrame:
    """An empty result that remembers why it is empty."""
    df = pd.DataFrame(columns=list(columns))
    df.attrs["reason"] = reason
    return df


def _to_grid(edges, lon, lat) -> tuple[float, float]:
    from pyproj import Transformer
    tr = Transformer.from_crs(CRS_GEOGRAPHIC, edges.crs, always_xy=True)
    return tr.transform(lon, lat)


def _is_sidepath(edges) -> np.ndarray:
    return edges["subclass"].fillna("").isin(SIDEPATH_SUBCLASSES).to_numpy()


def street_mask(edges, names) -> np.ndarray:
    """Edges of a street, by its English names, mapped sidewalks apart.

    A stretch of a street whose English name is missing from the map
    carries the Chinese one in ``name`` and nothing in ``name_zh`` (see
    ``network.split_name``), so matching on the English name alone leaves
    holes in the middle of Sassoon Road and Smithfield.  The street's
    Chinese names are therefore read off the edges that do match, and every
    edge carrying one of them, in either column, is taken as well.
    """
    on = edges["name"].isin(list(names))
    zh = set(edges.loc[on, "name_zh"].dropna()) if "name_zh" in edges else set()
    if zh:
        on = on | edges["name_zh"].isin(zh) | edges["name"].isin(zh)
    return on.to_numpy() & ~_is_sidepath(edges)


def _mad(values) -> float:
    v = np.asarray(values, dtype="float64")
    return float(np.median(np.abs(v - np.median(v)))) if v.size else float("nan")


def steepest_stretch(dist: np.ndarray, elev: np.ndarray, window_m: float) -> float:
    """Steepest gradient sustained over ``window_m`` of a profile.

    The profile is resampled every metre and the rise over every window of
    that length compared.  A profile shorter than the window gives its own
    mean gradient, which is the steepest thing it can be said to sustain.
    """
    dist = np.asarray(dist, dtype="float64")
    elev = np.asarray(elev, dtype="float64")
    length = float(dist[-1] - dist[0]) if dist.size > 1 else 0.0
    if length <= 0:
        return float("nan")
    if length <= window_m:
        return abs(float(elev[-1] - elev[0])) / length
    s = np.arange(dist[0], dist[-1] + 1e-9, 1.0)
    z = np.interp(s, dist, elev)
    w = int(round(window_m))
    return float(np.max(np.abs(z[w:] - z[:-w])) / w)


def profile_stats(dist: np.ndarray, elev: np.ndarray) -> dict:
    """Length, rise, mean and steepest gradients and climbing of a profile."""
    from .metrics import cumulative_gain_loss

    dist = np.asarray(dist, dtype="float64")
    elev = np.asarray(elev, dtype="float64")
    length = float(dist[-1] - dist[0])
    gain, loss = cumulative_gain_loss(elev, ELEVATION.gain_deadband_m, dist)
    top = int(np.argmax(elev))
    return {
        "length_m": length,
        "rise_m": float(elev[-1] - elev[0]),
        "mean_grade": float(elev[-1] - elev[0]) / length if length else float("nan"),
        "steepest_50": steepest_stretch(dist, elev, 50.0),
        "steepest_100": steepest_stretch(dist, elev, 100.0),
        "gain_m": gain, "loss_m": loss,
        "elev_min": float(elev.min()), "elev_max": float(elev.max()),
        "top_at_m": float(dist[top] - dist[0]),
    }


# --------------------------------------------------------------------------
# bench marks
# --------------------------------------------------------------------------
_BM_NS = "{http://www.safe.com/xml/xmltables}"


def load_benchmarks(path=None) -> pd.DataFrame:
    """The Survey and Mapping Office bench marks, one row per mark.

    Columns: ``station`` (the station number, kept as text because some
    carry a suffix such as ``210106.1``), ``locality``, ``type``, ``x`` and
    ``y`` on the HK1980 Grid (rounded to 1 m by the publisher) and ``h``,
    the height of the top of the mark in metres above Principal Datum.
    Marks without a height are dropped.  Returns an empty table, with the
    reason, if the file has not been downloaded.
    """
    import xml.etree.ElementTree as ET

    from .download import BENCHMARKS_XML

    path = BENCHMARKS_XML if path is None else path
    cols = ["station", "locality", "type", "x", "y", "h"]
    if not path.exists():
        return _not_run(f"`{path.name}` is not cached; run "
                        "`python -m hk_flat_routes download`", cols)
    rows = []
    for el in ET.parse(path).getroot().iter(_BM_NS + "BM"):
        try:
            rows.append((el.findtext(_BM_NS + "STN_NO"),
                         el.findtext(_BM_NS + "LOCALITY"),
                         el.findtext(_BM_NS + "MONU_TYPE"),
                         float(el.findtext(_BM_NS + "Easting_m")),
                         float(el.findtext(_BM_NS + "Northing_m")),
                         float(el.findtext(_BM_NS + "HKPD_m"))))
        except (TypeError, ValueError):
            continue                      # a record with a field missing
    df = pd.DataFrame(rows, columns=cols)
    df.attrs["n_published"] = len(df)
    out = df[df["h"] > 0].reset_index(drop=True)
    out.attrs["n_published"] = len(df)
    return out


def check_benchmarks(edges, profiles: dict, marks=None) -> pd.DataFrame:
    """The model's street elevation at every bench mark beside a road.

    Each mark is matched to the nearest centreline of a road that stands on
    the ground (structures are left out: a mark on a flyover parapet is not
    a statement about the ramp the model draws), and the edge's elevation
    profile is read at the point of the centreline nearest the mark.  Marks
    more than ``BENCHMARK_MAX_OFFSET_M`` from any road are not compared.

    ``diff`` is model minus mark.  It is expected to be a little negative:
    the published height is that of the top of the mark, which stands about
    0.2 m above the ground, on the pavement or a wall, while the centreline
    runs along the carriageway beyond the kerb.
    """
    from shapely.geometry import Point
    from shapely.strtree import STRtree

    cols = ["station", "locality", "x", "y", "h_mark", "z_model", "diff",
            "offset_m", "cls", "name", "edge_id"]
    marks = load_benchmarks() if marks is None else marks
    if marks.empty:
        return _not_run(marks.attrs.get("reason", "no bench marks"), cols)
    cand = edges[edges["cls"].isin(ROAD_CLASSES) & ~edges["is_structure"]
                 & ~_is_sidepath(edges)]
    if cand.empty:
        return _not_run("the run has no roads", cols)
    geoms = np.asarray(cand.geometry.values)
    eids = cand["edge_id"].to_numpy()
    cls = cand["cls"].to_numpy(dtype=object)
    names = cand["name"].to_numpy(dtype=object)
    tree = STRtree(geoms)
    rows = []
    for m in marks.itertuples(index=False):
        p = Point(m.x, m.y)
        i = int(tree.nearest(p))
        off = float(geoms[i].distance(p))
        if off > BENCHMARK_MAX_OFFSET_M:
            continue
        prof = profiles.get(int(eids[i]))
        if prof is None:
            continue
        z = float(np.interp(geoms[i].project(p), prof[0], prof[1]))
        rows.append((m.station, m.locality, m.x, m.y, m.h, z, z - m.h, off,
                     cls[i], names[i] if isinstance(names[i], str) else None,
                     int(eids[i])))
    df = pd.DataFrame(rows, columns=cols)
    df.attrs["n_marks"] = len(marks)
    df.attrs["n_published"] = marks.attrs.get("n_published", len(marks))
    if len(df):
        a = df["diff"].abs()
        log.info("bench marks: %d of %d beside a road in this run; model minus "
                 "mark median %+.2f m, MAD %.2f m, %.0f%% within 1 m, %.0f%% "
                 "within 2 m", len(df), len(marks), df["diff"].median(),
                 _mad(df["diff"]), 100 * (a <= 1).mean(), 100 * (a <= 2).mean())
    else:
        df.attrs["reason"] = "no bench mark lies beside a road of this run"
    return df


# --------------------------------------------------------------------------
# the independent terrain model
# --------------------------------------------------------------------------
def _read_dtm_5m():
    """The Lands Department 5 m model: (array, left, top, cell size).

    The zip holds one ESRI ASCII grid of whole metres on the HK1980 Grid;
    the file itself states no CRS.
    """
    import zipfile

    import rasterio

    from .download import DTM_5M_ZIP

    with zipfile.ZipFile(DTM_5M_ZIP) as zf:
        inner = next(n for n in zf.namelist() if n.lower().endswith(".asc"))
    with rasterio.open(f"/vsizip/{DTM_5M_ZIP}/{inner}") as src:
        arr = src.read(1)
        tr = src.transform
    return arr, float(tr.c), float(tr.f), float(tr.a)


def _raw_sampler():
    """The LiDAR tiles without the Gaussian pre-filter, or None."""
    from .elevation import DemSampler
    try:
        return DemSampler(smooth=False)
    except (FileNotFoundError, ValueError) as exc:
        log.warning("LiDAR tiles unavailable (%s)", exc)
        return None


def check_dem_agreement(edges, n_points: int = 4000, seed: int = 0,
                        n_tiles: int = 80) -> pd.DataFrame:
    """Compare the LiDAR with the 5 m terrain model at random street points.

    Upstream draws points anywhere inside its mosaic.  Here they are drawn
    along the run's own streets, for two reasons.  The 5 m model is
    photogrammetric and, by its publisher's account, carries tree canopy
    and elevated roads, so over a wooded hillside it says nothing about the
    ground; and only the LiDAR tiles under the run's streets need be on
    disk.  Each point is flagged ``built_up`` when it has at least a dozen
    other street edges within 100 m: the dense grid where the ground is all
    there is to photograph.  The rest (hill roads under trees) are kept and
    reported separately, as the measure of what canopy does.

    The LiDAR is read unsmoothed, tile by tile, so the points are taken
    from ``n_tiles`` tiles chosen at random in proportion to the street
    they hold; scattering 4,000 points over the whole territory would open
    most of its 3,253 tiles for one sample each.

    ``diff`` is LiDAR minus the 5 m model, the primary source minus the
    reference as upstream has it.  Cells of the 5 m model at or below zero
    are sea in a model that has no nodata value, and are skipped.
    """
    import shapely
    from shapely.strtree import STRtree

    from .download import DTM_5M_ZIP

    cols = ["x", "y", "z_lidar", "z_dtm5", "diff", "built_up"]
    if not DTM_5M_ZIP.exists():
        log.warning("5 m terrain model not cached; skipping cross-check")
        return _not_run(f"`{DTM_5M_ZIP.name}` is not cached; run "
                        "`python -m hk_flat_routes download`", cols)
    sampler = _raw_sampler()
    if sampler is None:
        return _not_run("the LiDAR tiles could not be opened; run "
                        "`python -m hk_flat_routes download`", cols)

    streets = edges[edges["cls"].isin(DRIVABLE + ("pedestrian",))
                    & ~edges["is_structure"] & ~_is_sidepath(edges)
                    & (edges["length_m"] >= 15.0)]
    if streets.empty:
        return _not_run("the run has no streets", cols)
    geoms = np.asarray(streets.geometry.values)
    length = streets["length_m"].to_numpy(dtype="float64")
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(geoms), size=n_points * 12, p=length / length.sum())
    pts = shapely.line_interpolate_point(geoms[pick],
                                         rng.uniform(0.0, length[pick]))
    xs, ys = shapely.get_x(pts), shapely.get_y(pts)

    # keep the points of a random handful of tiles that are on disk
    dem = sampler.dem
    col, row = dem.keys_of(xs, ys)
    flat = row * dem.n_cols + col
    have = np.array([(int(c), int(r)) in dem.paths for c, r in zip(col, row)])
    tiles, counts = np.unique(flat[have], return_counts=True)
    if tiles.size == 0:
        return _not_run("no LiDAR tile under this run's streets is on disk", cols)
    chosen = rng.choice(tiles, size=min(n_tiles, tiles.size), replace=False,
                        p=counts / counts.sum())
    keep = np.flatnonzero(have & np.isin(flat, chosen))
    keep = rng.permutation(keep)[:n_points]
    xs, ys = xs[keep], ys[keep]
    z1 = sampler.sample(xs, ys)

    arr, left, top, cell = _read_dtm_5m()
    c = np.floor((xs - left) / cell).astype(np.int64)
    r = np.floor((top - ys) / cell).astype(np.int64)
    inside = (r >= 0) & (c >= 0) & (r < arr.shape[0]) & (c < arr.shape[1])
    z2 = np.full(xs.shape, np.nan)
    z2[inside] = arr[r[inside], c[inside]]
    good = np.isfinite(z1) & np.isfinite(z2) & (z2 > 0)

    # how built-up is each point: street edges within 100 m
    tree = STRtree(geoms)
    near = tree.query(shapely.points(xs, ys), predicate="dwithin", distance=100.0)
    n_near = np.bincount(near[0], minlength=len(xs))

    df = pd.DataFrame({"x": xs[good], "y": ys[good], "z_lidar": z1[good],
                       "z_dtm5": z2[good], "built_up": n_near[good] >= 12})
    df["diff"] = df["z_lidar"] - df["z_dtm5"]
    df.attrs["n_tiles"] = int(len(chosen))
    df.attrs["n_sea"] = int((np.isfinite(z1) & np.isfinite(z2) & (z2 <= 0)).sum())
    b = df[df["built_up"]]
    if len(b):
        log.info("DEM cross-check on %d street points in %d tiles: built-up "
                 "(%d) median diff %+.2f m, MAD %.2f m, |diff|<2 m for %.1f%%",
                 len(df), len(chosen), len(b), b["diff"].median(),
                 _mad(b["diff"]), 100.0 * (b["diff"].abs() < 2).mean())
    return df[cols]


# --------------------------------------------------------------------------
# measuring along a named street
# --------------------------------------------------------------------------
class StreetNet:
    """The edge table as a graph for following a named street.

    Upstream compares a published grade with the steepest edge of that name
    anywhere in the city.  That cannot work in Hong Kong: Stubbs Road is
    seven kilometres of hairpins with one documented stretch, a street name
    also covers its slip roads, and one edge under a flyover that reads 50%
    would stand for the whole road.  The references here are *sections*
    (from this junction to that one, or between two bench marks), so the
    comparison follows the street from one end of the section to the other
    and measures the concatenated elevation profile.

    The path is the cheapest one between the two ends where a metre on the
    named street costs one and a metre anywhere else costs
    ``off_street_penalty``.  It therefore stays on the street wherever the
    street is continuous, and can still cross the short pieces mapped under
    another name or none (a roundabout, a junction box) that would
    otherwise break it.  Mapped sidewalks are left out altogether.
    """

    off_street_penalty = 20.0
    #: how much of a section given by one end (m) may run under another
    #: name, joining two parts of the street.  The shortest such section is
    #: stated as 500 m, so this is the same fifth of it that
    #: ``_measure_section`` allows off the street in any case; and it is
    #: what the widest break met so far needs, the staggered crossing that
    #: splits Shatin Pass Road above Nga Chuk Street (96 m by the links
    #: between its two runs).
    link_max_m = 100.0

    def __init__(self, edges, profiles: dict):
        import shapely

        e = edges.reset_index(drop=True)
        self.edges = e
        self.crs = edges.crs
        self.profiles = profiles
        self.geom = np.asarray(e.geometry.values)
        self.bounds = shapely.bounds(self.geom)
        nodes = pd.unique(pd.concat([e["u"], e["v"]], ignore_index=True))
        index = {n: i for i, n in enumerate(nodes)}
        self.u = e["u"].map(index).to_numpy(dtype=np.int64)
        self.v = e["v"].map(index).to_numpy(dtype=np.int64)
        self.length = e["length_m"].to_numpy(dtype="float64")
        self.eid = e["edge_id"].to_numpy()
        self.cls = e["cls"].to_numpy(dtype=object)
        self.sidepath = _is_sidepath(e)
        # node elevations, from the ends of the reconciled profiles
        self.node_z = np.full(len(nodes), np.nan)
        for k, eid in enumerate(self.eid):
            prof = profiles.get(int(eid))
            if prof is not None and len(prof[1]):
                self.node_z[self.u[k]] = prof[1][0]
                self.node_z[self.v[k]] = prof[1][-1]
        self.total_bounds = (float(self.bounds[:, 0].min()), float(self.bounds[:, 1].min()),
                             float(self.bounds[:, 2].max()), float(self.bounds[:, 3].max()))

    # -- selecting and snapping -------------------------------------------
    def street(self, names, classes=None) -> np.ndarray:
        """Mask of the edges of a street: its names, optionally its classes."""
        m = street_mask(self.edges, names)
        if classes is not None:
            m &= self.edges["cls"].isin(list(classes)).to_numpy()
        return m

    def covers(self, xy, margin: float = 0.0) -> bool:
        """Is a point inside the area the run's network spans?"""
        x0, y0, x1, y1 = self.total_bounds
        return (x0 + margin <= xy[0] <= x1 - margin
                and y0 + margin <= xy[1] <= y1 - margin)

    def snap(self, xy, mask, max_m: float):
        """Nearest masked edge to a point: (row, offset along it, distance)."""
        import shapely

        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return None
        p = shapely.Point(float(xy[0]), float(xy[1]))
        d = shapely.distance(self.geom[idx], p)
        k = int(np.argmin(d))
        if d[k] > max_m:
            return None
        row = int(idx[k])
        return row, float(shapely.line_locate_point(self.geom[row], p)), float(d[k])

    def elevation_at(self, row: int, offset: float) -> float:
        d, z = self.profiles[int(self.eid[row])]
        return float(np.interp(offset, d, z))

    # -- shortest paths that prefer the street ----------------------------
    def _search(self, start, favoured, reach_m: float, street_only: bool):
        """Dijkstra from both ends of the start edge over the local graph."""
        import scipy.sparse as sp
        from scipy.sparse.csgraph import dijkstra

        row, off, _ = start
        p = self.geom[row].interpolate(off)
        b = self.bounds
        box = ((b[:, 0] <= p.x + reach_m) & (b[:, 2] >= p.x - reach_m)
               & (b[:, 1] <= p.y + reach_m) & (b[:, 3] >= p.y - reach_m))
        use = box & (favoured if street_only else ~self.sidepath)
        use[row] = True
        use = np.flatnonzero(use)
        factor = np.where(favoured[use], 1.0, self.off_street_penalty)
        w = self.length[use] * factor
        nodes, inv = np.unique(np.r_[self.u[use], self.v[use]], return_inverse=True)
        lu, lv = inv[:len(use)], inv[len(use):]
        lo, hi = np.minimum(lu, lv), np.maximum(lu, lv)
        # two streets can join the same pair of junctions: keep the cheaper,
        # since a sparse matrix would otherwise add them together
        order = np.lexsort((w, hi, lo))
        first = np.r_[True, (lo[order][1:] != lo[order][:-1])
                      | (hi[order][1:] != hi[order][:-1])]
        keep = order[first]
        keep = keep[lo[keep] != hi[keep]]            # loops lead nowhere
        m = sp.coo_matrix((w[keep], (lo[keep], hi[keep])),
                          shape=(len(nodes), len(nodes))).tocsr()
        ends = np.searchsorted(nodes, [self.u[row], self.v[row]])
        dist, pred = dijkstra(m, directed=False, indices=ends,
                              return_predecessors=True)
        f0 = 1.0 if favoured[row] else self.off_street_penalty
        return {
            "row": row, "off": off, "nodes": nodes, "dist": dist, "pred": pred,
            "ends": ends, "start_cost": (off * f0, (self.length[row] - off) * f0),
            "edge_of": {(int(lo[k]), int(hi[k])): int(use[k]) for k in keep},
        }

    def _pieces_to(self, s: dict, end: int, node: int) -> list:
        """Path from the start point, out by one end of its edge, to a node.

        Pieces are (edge row, offset from, offset to) in travel order.
        """
        row, off = s["row"], s["off"]
        pieces = [(row, off, 0.0 if end == 0 else float(self.length[row]))]
        chain = [int(node)]
        while chain[-1] != int(s["ends"][end]):
            prev = int(s["pred"][end][chain[-1]])
            if prev < 0:
                return []
            chain.append(prev)
        chain.reverse()
        for a, b in zip(chain[:-1], chain[1:]):
            k = s["edge_of"][(min(a, b), max(a, b))]
            fwd = self.u[k] == s["nodes"][a]
            pieces.append((k, 0.0, float(self.length[k])) if fwd
                          else (k, float(self.length[k]), 0.0))
        return [p for p in pieces if p[1] != p[2]]

    def between(self, start, finish, favoured) -> list:
        """Pieces of the path from one snapped point to another."""
        (ra, oa, _), (rb, ob, _) = start, finish
        if ra == rb:
            return [(ra, oa, ob)] if oa != ob else []
        pa = self.geom[ra].interpolate(oa)
        pb = self.geom[rb].interpolate(ob)
        s = self._search(start, favoured, max(500.0, 1.5 * pa.distance(pb)),
                         street_only=False)
        fb = 1.0 if favoured[rb] else self.off_street_penalty
        best, best_cost = [], np.inf
        for end in (0, 1):
            for node_id, tail, far in ((self.u[rb], ob, 0.0),
                                       (self.v[rb], self.length[rb] - ob,
                                        float(self.length[rb]))):
                j = int(np.searchsorted(s["nodes"], node_id))
                if j >= len(s["nodes"]) or s["nodes"][j] != node_id:
                    continue
                cost = s["start_cost"][end] + s["dist"][end][j] + tail * fb
                if cost < best_cost:
                    head = self._pieces_to(s, end, j)
                    if not head and s["ends"][end] != j:
                        continue
                    last = [(rb, far, ob)] if far != ob else []
                    best, best_cost = head + last, cost
        return best

    def along(self, start, favoured, length_m: float, uphill: bool) -> list:
        """Pieces of the path a stated length along the street from a point.

        For the sections documented by one end and a length.  The path ends
        at the junction on the street, no further than the stated length (a
        tenth over is allowed for where the junction falls), that is highest
        above the start for an uphill section or lowest below it for a
        downhill one.

        As in ``between``, the path may cross short pieces mapped under
        another name: Shatin Pass Road is separate runs of that name in the
        network, and the run that climbs from Nga Chuk Street stops at
        Lung Fung Street, with the next beginning across a staggered
        junction.  At most ``link_max_m`` of the way may be off the street.

        The path can only end at a junction, and hill roads have few, so
        the section measured may fall short of the stated length (Lam Kam
        Road above Route Twisk does, by more than a hundred metres);
        ``SECTION_LENGTH_TOLERANCE`` decides whether it still counts, and
        the report prints both lengths.
        """
        s = self._search(start, favoured, 1.2 * length_m + 200.0,
                         street_only=False)
        on_street = np.zeros(self.node_z.size, dtype=bool)
        on_street[self.u[favoured]] = True
        on_street[self.v[favoured]] = True
        z0 = self.elevation_at(start[0], start[1])
        gain = (self.node_z[s["nodes"]] - z0) * (1.0 if uphill else -1.0)
        # a path of true length t with g metres off the street costs
        # t + (penalty - 1) * g, so this bound admits every path wanted
        limit = 1.1 * length_m + (self.off_street_penalty - 1.0) * self.link_max_m
        cands = []
        for end in (0, 1):
            cost = s["start_cost"][end] + s["dist"][end]
            ok = (np.isfinite(cost) & (cost <= limit) & np.isfinite(gain)
                  & (gain > 0) & on_street[s["nodes"]])
            cands += [(float(gain[j]), end, int(j)) for j in np.flatnonzero(ok)]
        for _gain, end, j in sorted(cands, reverse=True):
            pieces = self._pieces_to(s, end, j)
            total = sum(abs(s1 - s0) for _r, s0, s1 in pieces)
            off = sum(abs(s1 - s0) for r, s0, s1 in pieces if not favoured[r])
            if pieces and total <= 1.1 * length_m and off <= self.link_max_m:
                return pieces
        return []

    # -- profiles ----------------------------------------------------------
    def profile(self, pieces: list) -> tuple[np.ndarray, np.ndarray]:
        """Concatenated (distance, elevation) along a list of pieces."""
        dists, elevs, base = [], [], 0.0
        for row, s0, s1 in pieces:
            d, z = self.profiles[int(self.eid[row])]
            d = np.asarray(d, dtype="float64")
            z = np.asarray(z, dtype="float64")
            lo, hi = min(s0, s1), max(s0, s1)
            at = np.r_[lo, d[(d > lo) & (d < hi)], hi]
            zz = np.interp(at, d, z)
            run = np.abs(at - s0)
            if s1 < s0:
                run, zz = run[::-1], zz[::-1]
            if dists:                       # the joint is already there
                run, zz = run[1:], zz[1:]
            dists.append(base + run)
            elevs.append(zz)
            base += hi - lo
        if not dists:
            return np.array([]), np.array([])
        return np.concatenate(dists), np.concatenate(elevs)

    def share(self, pieces: list, mask) -> float:
        """Share of a path's length that lies on masked edges."""
        total = sum(abs(s1 - s0) for _r, s0, s1 in pieces)
        on = sum(abs(s1 - s0) for r, s0, s1 in pieces if mask[r])
        return on / total if total else float("nan")

    def point_on(self, pieces: list, at_m: float):
        """The point a given distance along a path."""
        for row, s0, s1 in pieces:
            run = abs(s1 - s0)
            if at_m <= run:
                return self.geom[row].interpolate(s0 + np.sign(s1 - s0) * at_m)
            at_m -= run
        row, _s0, s1 = pieces[-1]
        return self.geom[row].interpolate(s1)


# --------------------------------------------------------------------------
# steep streets
# --------------------------------------------------------------------------
def _measure_section(net: StreetNet, names, classes, a, b, stated_m=None,
                     snap_m: float = JUNCTION_SNAP_M) -> dict:
    """Follow a street between two ends, or for a stated length from one.

    ``a`` is the top of the section and ``b`` the bottom, each an (x, y)
    on the grid or None.  Returns the profile statistics, or a ``status``
    saying why there are none: the street or the end is not in this run, or
    the section could not be matched to the network.
    """
    street = net.street(names, classes)
    if not street.any():
        return {"status": "not in this run"}
    # Is the whole section inside the area the run spans?  If not, whatever
    # goes wrong below is the run's edge cutting the street, not the model.
    reach = RUN_EDGE_MARGIN_M
    if a is None or b is None:              # the far end is a length away
        reach += stated_m or 0.0
    inside = all(net.covers(xy, margin=reach) for xy in (a, b) if xy is not None)
    lost = "not matched" if inside else "not in this run"
    ends = []
    for xy in (a, b):
        if xy is None:
            ends.append(None)
            continue
        hit = net.snap(xy, street, snap_m)
        if hit is None:
            return {"status": lost,
                    "why": "an end of the section is not on the street"}
        ends.append(hit)
    if ends[0] is not None and ends[1] is not None:
        pieces = net.between(ends[0], ends[1], street)
    elif ends[0] is not None:               # known top: go down
        pieces = net.along(ends[0], street, stated_m, uphill=False)
    else:                                   # known bottom: go up, then turn
        pieces = net.along(ends[1], street, stated_m, uphill=True)
        pieces = [(r, s1, s0) for r, s0, s1 in reversed(pieces)]
    if not pieces:
        return {"status": lost,
                "why": "the street cannot be followed between its ends"}
    dist, elev = net.profile(pieces)
    out = profile_stats(dist, elev)
    out["on_street"] = net.share(pieces, street)
    out["steps_share"] = net.share(pieces, net.cls == "steps")
    out["n_edges"] = len({r for r, _a, _b in pieces})
    out["status"] = "measured"
    if stated_m and abs(out["length_m"] - stated_m) > SECTION_LENGTH_TOLERANCE * stated_m:
        out["status"] = lost
        out["why"] = (f"{out['length_m']:.0f} m was measured where "
                      f"{stated_m:.0f} m is stated"
                      if ends[0] is not None and ends[1] is not None else
                      f"the street can be followed for only "
                      f"{out['length_m']:.0f} m from that end before it "
                      f"breaks off in the network, where {stated_m:.0f} m is stated")
    elif out["on_street"] < 0.8:
        out["status"] = lost
        out["why"] = (f"only {100 * out['on_street']:.0f}% of the path "
                      "between its ends is on the street")
    return out


def _grade_verdict(diff: float, tolerance: float) -> str:
    return ("ok" if abs(diff) <= tolerance else
            "under-reported" if diff < 0 else "over-reported")


def check_steep_streets(edges, profiles: dict | None = None, marks=None,
                        net: StreetNet | None = None) -> pd.DataFrame:
    """Computed against documented gradients, section by section.

    Three groups, told apart by the ``group`` column:

    ``td``
        The Transport Department's maximum gradients.  Drivable classes
        only.  The stated maximum is expected between the model's gradient
        over the whole section and over its steepest ``TD_WINDOW_M`` (see
        that constant for why); ``computed`` is the latter and ``diff`` is
        how far outside that bracket the stated figure falls, positive
        when the model is the steeper.
    ``ratio``
        The signed and published ratios.  Every class counts, because the
        steepest part of a ladder street is a flight of steps, and
        ``steps_share`` says how much of the section is one.  Judged on the
        steepest ``RATIO_WINDOW_M``.
    ``benchmark``
        The mean gradient between two bench marks on the same road, judged
        against the same quantity from the official heights.

    ``published`` and ``computed`` are the pair that is judged; ``verdict``
    is ``ok``, ``under-reported``, ``over-reported``, ``not matched`` or
    ``not in this run``.
    """
    if net is None:
        net = StreetNet(edges, profiles)
    rows = []

    def base(group, street, section, published, stated=None):
        return {"group": group, "street": street, "section": section,
                "published": published, "stated_length_m": stated,
                "computed": np.nan, "diff": np.nan, "length_m": np.nan,
                "mean_grade": np.nan, "steepest_50": np.nan,
                "steepest_100": np.nan, "steps_share": np.nan,
                "n_edges": 0, "street_km": np.nan}

    def fill(row, m, judged, tolerance):
        if m["status"] != "measured":
            row["verdict"] = m["status"]
            row["why"] = m.get("why", "")
            for k in ("length_m", "steepest_50", "steepest_100", "mean_grade"):
                row[k] = m.get(k, np.nan)
            return row
        for k in ("length_m", "steepest_50", "steepest_100", "steps_share",
                  "n_edges", "rise_m", "elev_min", "elev_max"):
            row[k] = m[k]
        row["mean_grade"] = abs(m["mean_grade"])
        row["street_km"] = m["length_m"] / 1000.0
        row["computed"] = abs(m[judged])
        row["diff"] = row["computed"] - row["published"]
        row["verdict"] = _grade_verdict(row["diff"], tolerance)
        return row

    for names, grade, stated, frm, to, a, b in TD_STEEP_ROADS:
        ends = [None if p is None else _to_grid(edges, *p) for p in (a, b)]
        section = (f"{frm} to {to}" if a and b else
                   f"{stated} m down from {frm}" if a else
                   f"{stated} m up from {to}")
        m = _measure_section(net, names, DRIVABLE, ends[0], ends[1], stated)
        row = fill(base("td", names[0], section, grade, stated), m,
                   "steepest_100", TD_TOLERANCE)
        if m["status"] == "measured":
            row["diff"] = (row["mean_grade"] - grade if grade < row["mean_grade"]
                           else row["computed"] - grade if grade > row["computed"]
                           else 0.0)
            row["verdict"] = _grade_verdict(row["diff"], TD_TOLERANCE)
        rows.append(row)

    for name, name_zh, n, section, a, b in STEEP_RATIO_STREETS:
        m = _measure_section(net, (name,), None, _to_grid(edges, *a),
                             _to_grid(edges, *b))
        row = base("ratio", name, section, 1.0 / n)
        row["name_zh"], row["ratio"] = name_zh, f"1:{n}"
        rows.append(fill(row, m, "steepest_50", RATIO_TOLERANCE))

    marks = load_benchmarks() if marks is None else marks
    by_station = (marks.set_index("station") if len(marks) else None)
    for names, low, high, chainage in BENCHMARK_STRETCHES:
        row = base("benchmark", names[0], f"bench marks {low} to {high}",
                   np.nan, chainage)
        if by_station is None:
            row["verdict"] = "not run"
            row["why"] = marks.attrs.get("reason", "no bench marks")
            rows.append(row)
            continue
        if low not in by_station.index or high not in by_station.index:
            row["verdict"] = "not matched"
            row["why"] = "a station is no longer in the published file"
            rows.append(row)
            continue
        lo_m, hi_m = by_station.loc[low], by_station.loc[high]
        row["mark_rise_m"] = float(hi_m["h"] - lo_m["h"])
        row["published"] = row["mark_rise_m"] / chainage
        # the top of the stretch first, as for the other two groups
        m = _measure_section(net, names, None, (hi_m["x"], hi_m["y"]),
                             (lo_m["x"], lo_m["y"]), chainage,
                             snap_m=BENCHMARK_STREET_OFFSET_M)
        row = fill(row, m, "mean_grade", BENCHMARK_GRADE_TOLERANCE)
        if m["status"] == "measured":
            row["model_rise_m"] = -m["rise_m"]
        rows.append(row)

    df = pd.DataFrame(rows)
    for g, label in (("td", "Transport Department"), ("ratio", "published ratio"),
                     ("benchmark", "between bench marks")):
        sub = df[(df["group"] == g) & df["computed"].notna()]
        if len(sub):
            log.info("steep streets (%s): %d of %d measured sections ok",
                     label, int((sub["verdict"] == "ok").sum()), len(sub))
    return df


# --------------------------------------------------------------------------
# level corridors
# --------------------------------------------------------------------------
def _window(edges, bbox):
    """Restrict an edge table to a WGS84 bounding box."""
    if bbox is None:
        return edges
    from shapely.geometry import box
    from pyproj import Transformer
    tr = Transformer.from_crs(CRS_GEOGRAPHIC, edges.crs, always_xy=True)
    x0, y0 = tr.transform(bbox[0], bbox[2])
    x1, y1 = tr.transform(bbox[1], bbox[3])
    return edges[edges.geometry.intersects(box(x0, y0, x1, y1))]


def check_flat_corridors(edges, corridors=None, marks=None) -> pd.DataFrame:
    """Do the level streets measure level, and the others not?

    Each corridor is every edge of its streets (mapped sidewalks apart).
    ``gain_per_km`` is the climbing per kilometre averaged over the two
    directions, so that it does not depend on which way each piece happens
    to have been drawn; ``elev_min`` and ``elev_max`` are the band the
    street stays within.  A street is ``level`` when both are small (see
    ``LEVEL_MAX_CLIMB_PER_KM`` and ``LEVEL_MAX_RANGE_M``).  ``discovered``
    says whether one of its streets is in the names of a corridor the
    analysis found, and is None when no corridor table was given.

    The evidence is the official bench marks whose locality names the
    street and which stand within ``BENCHMARK_STREET_OFFSET_M`` of it: how
    many, and the lowest and highest of their heights.
    """
    import shapely
    from shapely.strtree import STRtree

    rows = []
    marks = load_benchmarks() if marks is None else marks
    for label, spec in KNOWN_FLAT.items():
        streets = spec["streets"]
        sub = _window(edges[street_mask(edges, streets)], spec.get("bbox"))
        row = {"corridor": label, "streets": "; ".join(streets),
               "expected": spec["expect"], "street_km": 0.0,
               "gain_per_km": np.nan, "mean_abs_grade": np.nan,
               "elev_min": np.nan, "elev_max": np.nan, "n_marks": 0,
               "mark_min": np.nan, "mark_max": np.nan,
               "verdict": "not in this run", "as_expected": None,
               "discovered": None}
        if sub.empty:
            rows.append(row)
            continue
        km = float(sub["length_m"].sum() / 1000)
        w = sub["length_m"].to_numpy(dtype="float64")
        row["street_km"] = km
        row["gain_per_km"] = float((sub["cum_gain_fwd"].sum()
                                    + sub["cum_loss_fwd"].sum()) / 2 / km)
        row["mean_abs_grade"] = float((sub["avg_grade_fwd"].abs() * w).sum() / w.sum())
        row["elev_min"] = float(sub["elev_min"].min())
        row["elev_max"] = float(sub["elev_max"].max())
        level = (row["gain_per_km"] < LEVEL_MAX_CLIMB_PER_KM
                 and row["elev_max"] - row["elev_min"] < LEVEL_MAX_RANGE_M)
        row["verdict"] = "level" if level else "not level"
        row["as_expected"] = row["verdict"] == spec["expect"]
        if len(marks):
            # the marks whose stated locality is this street, and which do
            # stand beside it: at a junction a mark within reach of one
            # street often belongs to the one that crosses it
            loc = marks["locality"].fillna("").str.upper()
            own = marks[np.logical_or.reduce([loc.str.contains(s.upper(), regex=False)
                                              for s in streets])]
            if len(own):
                tree = STRtree(np.asarray(sub.geometry.values))
                pts = shapely.points(own["x"].to_numpy(), own["y"].to_numpy())
                hit = np.unique(tree.query(pts, predicate="dwithin",
                                           distance=BENCHMARK_STREET_OFFSET_M)[0])
                if hit.size:
                    h = own["h"].to_numpy()[hit]
                    row["n_marks"] = int(hit.size)
                    row["mark_min"], row["mark_max"] = float(h.min()), float(h.max())
        if corridors is not None:
            names = corridors["street_names"].fillna("")
            row["discovered"] = bool(any(any(s in n for s in streets) for n in names))
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# routing checks
# --------------------------------------------------------------------------
def _graph_node_xy(edges, graph):
    """Coordinates of every node of a routing graph: (ids, x, y)."""
    import shapely

    geoms = np.asarray(edges.geometry.values)
    first = shapely.get_point(geoms, 0)
    last = shapely.get_point(geoms, -1)
    xy = pd.DataFrame({
        "node": np.r_[edges["u"].to_numpy(dtype=object), edges["v"].to_numpy(dtype=object)],
        "x": np.r_[shapely.get_x(first), shapely.get_x(last)],
        "y": np.r_[shapely.get_y(first), shapely.get_y(last)],
    }).drop_duplicates("node")
    xy = xy[xy["node"].isin(set(graph.node_ids))]
    return xy["node"].to_numpy(dtype=object), xy["x"].to_numpy(), xy["y"].to_numpy()


def _nearest_graph_node(graph, edges, lon=None, lat=None, xy=None, table=None):
    """Nearest routable node to a point: (node id, distance in metres)."""
    if xy is None:
        xy = _to_grid(edges, lon, lat)
    ids, x, y = table if table is not None else _graph_node_xy(edges, graph)
    d = np.hypot(x - xy[0], y - xy[1])
    i = int(np.argmin(d))
    return ids[i], float(d[i])


def _street_lengths(table, arcs) -> pd.Series:
    """Metres of a route on each named street, longest first."""
    t = table.iloc[arcs]
    return (t.assign(street=t["name"].fillna("(unnamed)"))
            .groupby("street")["length_m"].sum().sort_values(ascending=False))


def check_contour_trip(ctx, corridors=None) -> dict:
    """Does the flat objective go round on the contour road, not over?

    Hong Kong's counterpart of upstream's Wiggle check, and measured the
    same way.  A trip between two points at much the same height cannot be
    judged by how much it climbs per kilometre; what matters is **excess
    climbing**, how much more a route climbs than the unavoidable
    difference between its ends, and the honest test is that number for the
    flat route against the same number for the *shortest* route between the
    same two points.

    The trip is routed on foot and by bicycle with the pipeline's own
    router.  For each route the result records what it does (distance,
    climb, high point, steepest pitch, the streets that carry it), and how
    much of it runs on the contour roads local knowledge would name
    (``CONTOUR_TRIP["round"]``) and on the streets that go over
    (``CONTOUR_TRIP["over"]``).  Nothing is assumed about either.
    """
    from .config import ROUTING_PROFILES
    from .routing import route, route_geometry

    edges = ctx.edges
    out: dict = {"label": CONTOUR_TRIP["label"], "from": CONTOUR_TRIP["from"][0],
                 "to": CONTOUR_TRIP["to"][0], "routes": {}, "status": "not run"}
    ends = [_to_grid(edges, *CONTOUR_TRIP[k][1]) for k in ("from", "to")]
    for mode in ("walk", "bike"):
        graph = ctx.graphs.get(mode)
        if graph is None:
            continue
        table = _graph_node_xy(edges, graph)
        (a, da), (b, db) = (_nearest_graph_node(graph, edges, xy=p, table=table)
                            for p in ends)
        if max(da, db) > TRIP_SNAP_M:
            out.setdefault("reason", "the trip is not in this run")
            continue
        if not graph.connected(graph.index_of(a), graph.index_of(b)):
            out.setdefault("reason", f"the two ends are not joined by {mode}")
            continue
        res = {}
        for pname in ("shortest", "balanced", "min_climb"):
            arcs, s = route(graph, a, b, ROUTING_PROFILES[pname])
            if not arcs:
                continue
            t = graph.table.iloc[arcs]
            by_street = _street_lengths(graph.table, arcs)
            net = abs(s["end_elev_m"] - s["start_elev_m"])
            res[pname] = {
                "km": s["distance_m"] / 1000.0, "gain_m": s["elev_gain_m"],
                "net_m": net, "excess_m": max(0.0, s["elev_gain_m"] - net),
                "max_grade": s["max_grade"],
                "high_point_m": float(t[["start_elev", "end_elev"]].to_numpy().max()),
                "steps_m": float(t.loc[t["cls"] == "steps", "length_m"].sum()),
                "streets": [(n, float(v)) for n, v in by_street.head(6).items()
                            if n != "(unnamed)"],
                "round_share": float(by_street.reindex(list(CONTOUR_TRIP["round"]))
                                     .fillna(0).sum() / s["distance_m"]),
                "over_share": float(by_street.reindex(list(CONTOUR_TRIP["over"]))
                                    .fillna(0).sum() / s["distance_m"]),
                "arcs": arcs,
            }
        if "shortest" not in res or "min_climb" not in res:
            continue
        # the best flat option is whichever climb-averse objective achieves
        # the least excess climbing without an unreasonable detour
        short_r = res["shortest"]
        cands = [(k, res[k]) for k in ("min_climb", "balanced")
                 if k in res and res[k]["km"] <= 1.35 * short_r["km"]] \
            or [("min_climb", res["min_climb"])]
        flat_name, flat_r = min(cands, key=lambda kr: kr[1]["excess_m"])
        share = np.nan
        if corridors is not None and len(corridors):
            cor = corridors[corridors["mode"] == mode] if "mode" in corridors else corridors
            geom = route_geometry(graph, flat_r["arcs"], edges)
            if geom is not None and len(cor):
                import shapely
                near = shapely.union_all(np.asarray(cor.geometry.buffer(12.0).values))
                share = float(geom.intersection(near).length / geom.length)
        for r in res.values():
            r.pop("arcs")
        out["routes"][mode] = {
            "objectives": res, "flat_objective": flat_name,
            "snap_m": (da, db),
            "excess_flat_m": flat_r["excess_m"],
            "excess_shortest_m": short_r["excess_m"],
            "corridor_share": share,
            "discovered": bool(np.isfinite(share) and share >= 0.5),
            "verdict": ("goes round" if flat_r["excess_m"] <= 0.5 * short_r["excess_m"]
                        else "no better than shortest"),
        }
    if out["routes"]:
        out["status"] = "run"
        for mode, r in out["routes"].items():
            log.info("contour trip by %s: shortest wastes %.1f m of climbing, "
                     "the flat route %.1f m (%s)", mode, r["excess_shortest_m"],
                     r["excess_flat_m"], r["verdict"])
    return out


def check_passes(ctx, marks=None, sampler=None) -> pd.DataFrame:
    """The model's pass heights against the gaps' published ones.

    The pass analysis (``passes.py``) answers "what is the lowest high point
    anyone must reach between these two places", exactly, with a merge
    tree.  Asked of the whole network about two places either side of one
    of Hong Kong Island's gaps, the answer is rarely the gap: a walker with
    unlimited patience goes round by the coast, far lower.  So the question
    is put the way a gap is defined.  For each gap, take an official bench
    mark on a road down each side, and only the streets and paths within
    a circle that just holds both marks; the lowest crossing between the
    two marks *within that circle* is the model's height for the gap, and
    the edge the merge tree blames for it is where the model puts the gap.
    (Each mark is entered at the lower end of the edge it stands beside.)

    Beside it goes a second, independent reading: the high point of the
    shortest walking route between the same two marks on the whole network.
    Both are compared with the saddle height mapped in OpenStreetMap and
    with the bench mark, if there is one, standing within ``GAP_MARK_M`` of
    the mapped saddle.  The verdict is ``agrees``, ``higher`` or ``lower``
    (than whichever reference is nearer).

    A circle can hold a second, lower gap.  Wan Chai Gap is under 500 m
    along the ridge from Magazine Gap and more than forty metres lower, and
    the mark on Peak Road stands between the two, so once Wan Chai Gap Road
    is in the network the lowest way from one Magazine Gap mark to the
    other is round by Wan Chai Gap: a true answer, to a question about a
    different gap.  When the crossing found is further
    than ``GAP_NEAR_M`` from the mapped saddle, its height says nothing
    about this gap, and the gap is read from the shortest route instead,
    provided that route tops out at the gap; ``judged`` says which reading
    was used (``crossing`` or ``route``).  Only when neither is at the gap
    is the verdict ``lower way round``.

    Given a ``sampler`` of the unsmoothed LiDAR, ``raw_m`` is what the
    terrain model itself reads at the top of the pass edge.  It separates
    the two ways a pass can disagree with its published height: the model's
    processing has moved it, or the published figure is rough.
    """
    import shapely
    from shapely.geometry import Point

    from .config import ROUTING_PROFILES
    from .passes import build_bottleneck_tree, pass_height
    from .routing import route

    cols = ["gap", "name_zh", "published_m", "pass_m", "pass_street",
            "pass_offset_m", "raw_m", "route_high_m", "route_offset_m",
            "mark_m", "mark_station", "judged", "diff_m", "verdict", "why"]
    edges = ctx.edges
    graph = ctx.graphs.get("walk")
    marks = load_benchmarks() if marks is None else marks
    if graph is None:
        return _not_run("no walking graph", cols)
    by_station = marks.set_index("station") if len(marks) else None
    geoms = np.asarray(edges.geometry.values)
    bounds = shapely.bounds(geoms)
    tb = edges.total_bounds
    walk = edges["walk_ok"].to_numpy()
    walk_idx = np.flatnonzero(walk)
    eid_arr = edges["edge_id"].to_numpy()
    u_arr = edges["u"].to_numpy(dtype=object)
    v_arr = edges["v"].to_numpy(dtype=object)
    e_idx = edges.set_index("edge_id")
    tree_log = get_logger("hk_flat_routes.passes")   # it logs every build
    rows = []
    for name, name_zh, lon, lat, published, stations in KNOWN_GAPS:
        gx, gy = _to_grid(edges, lon, lat)
        gap = Point(gx, gy)
        row = dict.fromkeys(cols, np.nan)
        row.update({"gap": name, "name_zh": name_zh, "published_m": published,
                    "pass_street": "", "mark_station": "", "why": "",
                    "judged": "", "verdict": "not in this run"})
        rows.append(row)
        if by_station is not None:
            near = marks[np.hypot(marks["x"] - gx, marks["y"] - gy) <= GAP_MARK_M]
            if len(near):
                k = near.iloc[int(np.argmin(np.hypot(near["x"] - gx, near["y"] - gy)))]
                row["mark_m"], row["mark_station"] = float(k["h"]), k["station"]
        if stations is None:
            # No crossing can be set up, but where a mark stands at the gap
            # the model can at least be read beside it, as in check 1.
            row["verdict"] = "not run"
            row["why"] = "no bench mark on the road down the far side"
            if not (tb[0] <= gx <= tb[2] and tb[1] <= gy <= tb[3]):
                row["verdict"] = "not in this run"
            elif np.isfinite(row["mark_m"]):
                q = by_station.loc[row["mark_station"]]
                d = shapely.distance(geoms[walk_idx], Point(q["x"], q["y"]))
                if d.size and d.min() <= BENCHMARK_MAX_OFFSET_M:
                    k = int(walk_idx[int(np.argmin(d))])
                    prof = ctx.profiles[int(eid_arr[k])]
                    z = float(np.interp(geoms[k].project(Point(q["x"], q["y"])),
                                        prof[0], prof[1]))
                    row["why"] += f"; beside the mark the model reads {z:.1f} m"
            continue
        if by_station is None:
            row["verdict"], row["why"] = "not run", marks.attrs.get("reason", "")
            continue
        if any(s not in by_station.index for s in stations):
            row["verdict"] = "not run"
            row["why"] = "a station is no longer in the published file"
            continue
        ends = [(float(by_station.loc[s, "x"]), float(by_station.loc[s, "y"]))
                for s in stations]
        reach = max(gap.distance(Point(p)) for p in ends)
        if not (tb[0] <= gx - reach and gx + reach <= tb[2]
                and tb[1] <= gy - reach and gy + reach <= tb[3]):
            continue                              # a mark is outside the run
        # Each mark stands beside a road that climbs to the gap.  Hill roads
        # have few junctions, so the nearest junction can be far off and may
        # even be over the top; the *lower* end of the edge the mark is on
        # is always on the mark's own side.
        anchors = []
        for q in ends:
            pt = Point(q)
            off = BENCHMARK_STREET_OFFSET_M
            near = walk_idx[(bounds[walk_idx, 0] <= q[0] + off)
                            & (bounds[walk_idx, 2] >= q[0] - off)
                            & (bounds[walk_idx, 1] <= q[1] + off)
                            & (bounds[walk_idx, 3] >= q[1] - off)]
            d = shapely.distance(geoms[near], pt) if near.size else np.array([])
            if not d.size or d.min() > off:
                break
            k = int(near[int(np.argmin(d))])
            z = ctx.profiles[int(eid_arr[k])][1]
            both = sorted(((float(z[0]), u_arr[k], shapely.get_point(geoms[k], 0)),
                           (float(z[-1]), v_arr[k], shapely.get_point(geoms[k], -1))),
                          key=lambda t: t[0])
            pick = next((t for t in both if graph.index_of(t[1]) is not None), None)
            if pick is None:
                break
            anchors.append(pick)
        if len(anchors) < 2:
            row["verdict"] = "not matched"
            row["why"] = "a bench mark is not beside a street of the network"
            continue
        a, b = anchors[0][1], anchors[1][1]
        radius = max(reach, *(gap.distance(t[2]) for t in anchors)) + 200.0
        if not (tb[0] <= gx - radius and gx + radius <= tb[2]
                and tb[1] <= gy - radius and gy + radius <= tb[3]):
            continue                              # the circle leaves the run

        # the lowest crossing within the circle
        box = ((bounds[:, 0] <= gx + radius) & (bounds[:, 2] >= gx - radius)
               & (bounds[:, 1] <= gy + radius) & (bounds[:, 3] >= gy - radius))
        idx = np.flatnonzero(box & walk)
        idx = idx[shapely.distance(geoms[idx], gap) <= radius]
        level = tree_log.level
        tree_log.setLevel("WARNING")
        try:
            tree = build_bottleneck_tree(edges.iloc[idx], mode="walk")
        finally:
            tree_log.setLevel(level)
        hit = pass_height(tree, a, b)
        if hit is None:
            row["verdict"] = "not matched"
            row["why"] = "the two marks are not joined within the circle"
            continue
        h, eid = hit
        nm = e_idx.loc[eid, "name"]
        row["pass_m"] = h
        row["pass_street"] = nm if isinstance(nm, str) else f"({e_idx.loc[eid, 'cls']})"
        row["pass_offset_m"] = float(e_idx.loc[eid, "geometry"].distance(gap))
        if sampler is not None:
            d, z = ctx.profiles[int(eid)]
            top = e_idx.loc[eid, "geometry"].interpolate(float(d[int(np.argmax(z))]))
            row["raw_m"] = float(sampler.sample(np.array([top.x]),
                                                np.array([top.y]))[0])

        # the high point of the shortest walk between the same two marks
        if graph.connected(graph.index_of(a), graph.index_of(b)):
            arcs, _s = route(graph, a, b, ROUTING_PROFILES["shortest"])
            if arcs:
                on = e_idx.loc[graph.table.iloc[arcs]["edge_id"].to_numpy()]
                top = on["elev_max"].to_numpy()
                k = int(np.argmax(top))
                row["route_high_m"] = float(top[k])
                row["route_offset_m"] = float(on["geometry"].iloc[k].distance(gap))

        # which reading is this gap's: the crossing if it is at the gap,
        # else the shortest route's high point if that is (see the docstring)
        at_gap = row["pass_offset_m"] <= GAP_NEAR_M
        by_route = (not at_gap and np.isfinite(row["route_high_m"])
                    and row["route_offset_m"] <= GAP_NEAR_M)
        row["judged"] = "route" if by_route else "crossing"
        height = row["route_high_m"] if by_route else h
        refs = [published] + ([row["mark_m"]] if np.isfinite(row["mark_m"]) else [])
        row["diff_m"] = min((height - r for r in refs), key=abs)
        if not at_gap and not by_route:
            row["verdict"] = "lower way round"
        elif abs(row["diff_m"]) <= GAP_TOLERANCE_M:
            row["verdict"] = "agrees"
        else:
            row["verdict"] = "higher" if row["diff_m"] > 0 else "lower"
    return pd.DataFrame(rows, columns=cols)


# --------------------------------------------------------------------------
# internal invariants
# --------------------------------------------------------------------------
def check_invariants(ctx) -> pd.DataFrame:
    """Identities the edge tables must satisfy whatever the terrain.

    Upstream asserts these in its test suite.  They are repeated here so a
    reader of the report sees them hold on the published build, and each is
    reported with the worst deviation found, not just a tick.
    """
    d = ctx.directed
    e = ctx.edges
    rows = []

    def add(name, checked, bad, worst, unit):
        rows.append({"invariant": name, "checked": int(checked),
                     "violations": int(bad), "worst": float(worst),
                     "unit": unit, "verdict": "holds" if bad == 0 else "VIOLATED"})

    counts = d.groupby("edge_id").size()
    add("every edge has exactly two directions", len(e),
        int((counts != 2).sum()) + abs(len(counts) - len(e)),
        float((counts - 2).abs().max()) if len(counts) else 0.0, "arcs")

    res = (d["cum_gain"] - d["cum_loss"] - d["net_change"]).abs()
    add("gain - loss = net change, per directed edge", len(d),
        (res > 1e-3).sum(), res.max(), "m")

    f = d[d["direction"] == "fwd"].set_index("edge_id")
    r = d[d["direction"] == "rev"].set_index("edge_id").reindex(f.index)
    mirror = np.maximum((f["cum_gain"] - r["cum_loss"]).abs(),
                        (f["cum_loss"] - r["cum_gain"]).abs())
    add("gain(u->v) = loss(v->u)", len(f), (mirror > 1e-6).sum(), mirror.max(), "m")
    anti = (f["net_change"] + r["net_change"]).abs()
    add("net change(u->v) = -net change(v->u)", len(f), (anti > 1e-6).sum(),
        anti.max(), "m")

    a = d[["from_node", "start_elev"]].rename(columns={"from_node": "node",
                                                       "start_elev": "z"})
    b = d[["to_node", "end_elev"]].rename(columns={"to_node": "node",
                                                   "end_elev": "z"})
    g = pd.concat([a, b], ignore_index=True).groupby("node")["z"]
    spread = g.max() - g.min()
    add("one elevation per node, whichever edge reaches it", len(spread),
        (spread > 1e-6).sum(), spread.max(), "m")

    lo, hi = ELEVATION_RANGE_M
    out = (e["elev_min"] < lo) | (e["elev_max"] > hi)
    worst = max(lo - float(e["elev_min"].min()), float(e["elev_max"].max()) - hi, 0.0)
    add(f"elevations between {lo:g} and {hi:g} m above Principal Datum", len(e),
        out.sum(), worst, "m")

    over = e["max_abs_grade"] > ELEVATION.max_plausible_grade + 1e-9
    add(f"no maximum gradient above the {ELEVATION.max_plausible_grade:.0%} "
        "plausibility ceiling", len(e), over.sum(),
        max(float(e["max_abs_grade"].max()) - ELEVATION.max_plausible_grade, 0.0),
        "")

    ths = sorted(int(c.rsplit("_", 1)[1]) for c in d.columns
                 if c.startswith("d_above_"))
    nested = np.zeros(len(d), dtype=bool)
    worst = 0.0
    for lo_t, hi_t in zip(ths[:-1], ths[1:]):
        gap = d[f"d_above_{hi_t}"] - d[f"d_above_{lo_t}"]
        nested |= (gap > 1e-6).to_numpy()
        worst = max(worst, float(gap.max()))
    if ths:
        gap = d[f"d_above_{ths[0]}"] - d["length_m"] * (1 + 1e-6) - 1e-3
        nested |= (gap > 0).to_numpy()
        worst = max(worst, float(gap.max()))
    add("distance above each gradient threshold is nested and no longer "
        "than the edge", len(d), nested.sum(), max(worst, 0.0), "m")

    steps = d[d["cls"] == "steps"]
    add("no stairway is open to bicycles", len(steps),
        steps["bike_traversable"].sum(), float(steps["bike_traversable"].sum()),
        "arcs")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------
def run_validation(ctx, corridors=None, write: bool = True,
                   dem_crosscheck: bool = True) -> dict:
    """Run every check and write a markdown report.

    ``dem_crosscheck=False`` skips the comparison with the 5 m model, which
    reads raw LiDAR and so cannot change with any model parameter; the
    sensitivity harness uses that to save a minute per configuration.
    """
    with step("validating elevation model against known ground truth", log):
        marks = load_benchmarks()
        net = StreetNet(ctx.edges, ctx.profiles)
        bench = check_benchmarks(ctx.edges, ctx.profiles, marks)
        if dem_crosscheck:
            dem = check_dem_agreement(ctx.edges)
        else:
            dem = _not_run("skipped: it does not depend on the model",
                           ["x", "y", "z_lidar", "z_dtm5", "diff", "built_up"])
        steep = check_steep_streets(ctx.edges, ctx.profiles, marks, net=net)
        flat = check_flat_corridors(ctx.edges, corridors, marks)
        try:
            trip = check_contour_trip(ctx, corridors)
        except Exception as exc:                      # pragma: no cover
            log.warning("contour trip check failed: %s", exc)
            trip = {"label": CONTOUR_TRIP["label"], "routes": {},
                    "status": "not run", "reason": f"the check failed ({exc})"}
        try:
            passes = check_passes(ctx, marks, sampler=_raw_sampler()
                                  if dem_crosscheck else None)
        except Exception as exc:                      # pragma: no cover
            log.warning("pass check failed: %s", exc)
            passes = _not_run(f"the check failed ({exc})")
        invariants = check_invariants(ctx)

    res = {"benchmarks": bench, "dem": dem, "steep": steep, "flat": flat,
           "trip": trip, "passes": passes, "invariants": invariants}
    if write:
        _write_report(ctx, res)
    return res


# --------------------------------------------------------------------------
# the report
# --------------------------------------------------------------------------
def _pct(v, digits: int = 1) -> str:
    return "n/a" if v is None or not np.isfinite(v) else f"{100 * v:.{digits}f}%"


def _m(v, digits: int = 1, sign: bool = False) -> str:
    if v is None or not np.isfinite(v):
        return "n/a"
    return f"{v:+.{digits}f} m" if sign else f"{v:.{digits}f} m"


def _proper(text) -> str:
    """A locality from the bench mark file, which is all capitals, as a name.

    ``str.title`` alone capitalises the letter after every apostrophe, which
    is right for D'Aguilar and wrong for Hiram's and King's.
    """
    import re
    return re.sub(r"(?<=[a-z])'S\b", "'s", str(text).title())


def _coverage(ctx) -> list[str]:
    """One paragraph saying what this run covers."""
    e = ctx.edges
    line = (f"This run covers **{len(e):,} street edges**, "
            f"{e['length_m'].sum() / 1000:,.0f} km")
    graph = ctx.graphs.get("walk") if ctx.graphs else None
    if graph is not None and graph.node_comp is not None:
        n = int(np.unique(graph.node_comp).size)
        if n > 1:
            line += (f", in {n} pieces with no way between them on foot "
                     "(landmasses, and whatever the edge of the run cuts off)")
    return [line + ". References outside it are listed as *not in this run*.", ""]


def _write_report(ctx, res: dict) -> None:
    bench, dem, steep, flat = res["benchmarks"], res["dem"], res["steep"], res["flat"]
    trip, passes, inv = res["trip"], res["passes"], res["invariants"]
    L: list[str] = ["# Validation report", "",
                    "Seven checks of the Hong Kong model against ground truth "
                    "it was not built from: official bench marks, an "
                    "independent terrain model, documented gradients, streets "
                    "known to be level, a trip everyone walks the flat way "
                    "round, the gaps, and the model's own internal "
                    "identities. The first four follow upstream's "
                    "(flattensf's) checks for San Francisco, against Hong "
                    "Kong's references; every figure below is generated by "
                    "`python -m hk_flat_routes validate`, and heights are in "
                    "metres above the Hong Kong Principal Datum.", ""]
    L += _coverage(ctx)
    misses: list[str] = []
    kinds: set[str] = set()                 # which notes section 8 needs

    # ---- 1 ---------------------------------------------------------------
    L += ["## 1. Street elevations against the official bench marks", ""]
    if len(bench):
        d = bench["diff"]
        a = d.abs()
        L += [f"The Survey and Mapping Office publishes "
              f"**{bench.attrs.get('n_published', 0):,}** bench marks, each a "
              "levelled height above the Principal Datum with a position on "
              "the HK1980 Grid to the nearest metre. A mark is compared when "
              f"it stands within {BENCHMARK_MAX_OFFSET_M:g} m of the "
              "centreline of a road that is on the ground; the model's "
              "elevation is read from the edge's profile at the nearest "
              "point of the centreline.", "",
              f"- Marks compared in this run: **{len(bench):,}**",
              f"- Model minus mark: median **{d.median():+.2f} m**, median "
              f"absolute deviation **{_mad(d):.2f} m**",
              f"- Within 0.5 m: **{100 * (a <= 0.5).mean():.0f}%**; within 1 m: "
              f"**{100 * (a <= 1).mean():.0f}%**; within 2 m: "
              f"**{100 * (a <= 2).mean():.0f}%**", "",
              "The model is expected to read a few decimetres low. A bench "
              "mark's height is that of the top of the mark, which stands "
              "about 0.2 m above the ground it is set in, and it is set in "
              "the pavement or a wall beside the road, a kerb's height above "
              "the carriageway the centreline follows.", "",
              "The spread is given as a median and a median absolute "
              "deviation, not a mean and an RMS, on purpose. A mark's "
              "position is good to a metre and it can be up to "
              f"{BENCHMARK_MAX_OFFSET_M:g} m to one side of the centreline; "
              "on a road cut into a hillside that puts some marks on top of "
              "the retaining wall above the road or at the foot of the one "
              "below it. Those few differ by metres for reasons that have "
              "nothing to do with the terrain model, and an RMS would report "
              "little else.", "",
              "| Height of mark | Marks | Median difference | MAD | Within 1 m | Within 2 m |",
              "|---|---|---|---|---|---|"]
        bands = ((-np.inf, 20.0, "below 20 m"), (20.0, 100.0, "20 to 100 m"),
                 (100.0, np.inf, "above 100 m"))
        for lo, hi, label in bands:
            g = bench[(bench["h_mark"] >= lo) & (bench["h_mark"] < hi)]
            if len(g):
                L.append(f"| {label} | {len(g)} | {g['diff'].median():+.2f} m | "
                         f"{_mad(g['diff']):.2f} m | "
                         f"{100 * (g['diff'].abs() <= 1).mean():.0f}% | "
                         f"{100 * (g['diff'].abs() <= 2).mean():.0f}% |")
        worst = bench.reindex(a.sort_values(ascending=False).index).head(5)
        L += ["", "The five largest differences:", "",
              "| Station | Locality | Mark | Model | Difference | Off the centreline | Street |",
              "|---|---|---|---|---|---|---|"]
        for _, r in worst.iterrows():
            L.append(f"| {r['station']} | {_proper(r['locality'])} | "
                     f"{r['h_mark']:.1f} m | {r['z_model']:.1f} m | "
                     f"{r['diff']:+.1f} m | {r['offset_m']:.0f} m | "
                     f"{r['name'] or '(unnamed ' + str(r['cls']) + ')'} |")
        ok = abs(d.median()) <= 1.0 and (a <= 2).mean() >= 0.90
        L += ["", "**Verdict:** " + (
              "the street elevations are in the Principal Datum and correctly "
              "placed; the bulk of them agree with levelled heights to a few "
              "decimetres." if ok else
              "the street elevations do NOT agree with the bench marks to the "
              "expected standard (median within 1 m, nine in ten within 2 m)."), ""]
        if not ok:
            misses.append("**Bench marks**: the median difference or the share "
                          "within 2 m is outside the expected standard; see section 1.")
    else:
        L += [f"_Not run: {bench.attrs.get('reason', 'no bench marks')}._", ""]

    # ---- 2 ---------------------------------------------------------------
    L += ["## 2. LiDAR against the independent 5 m terrain model", ""]
    if len(dem):
        L += ["The Lands Department's 5 m model comes from 2015 aerial "
              "photography, shares nothing with the 2020 LiDAR survey, and is "
              "on the same grid and datum. Its publisher gives its accuracy as "
              "+/- 5 m and says it includes elevated roads and shows the "
              "height of the vegetation where land is vegetated; its heights "
              "are whole metres. So it is compared only at points on the "
              f"run's streets ({len(dem):,} of them, drawn at random from "
              f"{dem.attrs.get('n_tiles', 0)} LiDAR tiles), and the built-up "
              "grid, where there is nothing over the street to photograph, is "
              "kept apart from the hill roads under trees.", "",
              "| Street points | Count | Median (LiDAR - 5 m) | MAD | Within 2 m | Within 5 m |",
              "|---|---|---|---|---|---|"]
        for label, g in (("Built-up", dem[dem["built_up"]]),
                         ("Elsewhere", dem[~dem["built_up"]]),
                         ("All", dem)):
            if len(g):
                L.append(f"| {label} | {len(g):,} | {g['diff'].median():+.2f} m | "
                         f"{_mad(g['diff']):.2f} m | "
                         f"{100 * (g['diff'].abs() < 2).mean():.1f}% | "
                         f"{100 * (g['diff'].abs() < 5).mean():.1f}% |")
        b = dem[dem["built_up"]] if dem["built_up"].any() else dem
        ok = abs(b["diff"].median()) <= 2.0 and (b["diff"].abs() < 5).mean() >= 0.90
        L += ["", "A built-up point counts as agreeing when the two models are "
              "within the 5 m model's own stated accuracy. "
              + ("The LiDAR reads lower, as it should: one model is the "
                 "ground and the other includes whatever stood on it."
                 if b["diff"].median() < 0 else
                 "The LiDAR does not read lower, which a model of the bare "
                 "ground should."), "",
              "**Verdict:** " + (
                  "two independently produced models agree on the street grid "
                  "to within the coarser one's accuracy, so the LiDAR tiles "
                  "are correctly georeferenced and in the expected datum "
                  "throughout. The 5 m model is no use beyond that: section 1 "
                  "is the measure of accuracy." if ok else
                  "the two models do NOT agree on the built-up street grid to "
                  "within the 5 m model's stated accuracy."), ""]
        if not ok:
            misses.append("**5 m terrain model**: disagreement beyond its "
                          "stated accuracy on built-up streets; see section 2.")
    else:
        L += [f"_Not run: {dem.attrs.get('reason', 'no result')}._", ""]

    # ---- 3 ---------------------------------------------------------------
    L += ["## 3. Steep streets against documented gradients", "",
          "Upstream compares a published grade with the steepest edge of "
          "that name anywhere in the city. A Hong Kong street name covers "
          "kilometres of hairpins, slip roads and flyovers, and the "
          "references are sections of it, so each is followed through the "
          "network from one documented end to the other and measured on the "
          "joined-up elevation profile. *Steepest 50 m* and *steepest 100 m* "
          "are the greatest gradients sustained over that length.", ""]

    td = steep[steep["group"] == "td"]
    L += ["### 3a. The Transport Department's long steep downhill roads", "",
          f"Maximum gradients from the department's 2008 review of "
          f"{TD_ANNEX_SECTIONS} road sections (Legislative Council paper "
          f"CB(1)1977/07-08(04)); these are the {len(td)} that could be "
          "placed on the map: an end located, a length that fits, and no "
          "doubt which way the section runs from that end. A section "
          "given by one end is measured for its stated length from that "
          "end, as far as the last junction within it. The paper does not "
          "say over what length its maximum is "
          "taken, and bench marks on the same roads show it to be a "
          "sustained gradient, not the steepest pitch. So the model gives "
          "two figures to bracket it, its gradient over the whole section "
          f"and over the section's steepest {TD_WINDOW_M:g} m, and a section "
          "agrees when the stated maximum lies between them, give or take "
          f"{100 * TD_TOLERANCE:g} percentage points for rounding. "
          "*Over-reported* means the model is steeper than the stated "
          "maximum over the whole section; *under-reported*, that even its "
          "steepest 100 m falls short.", "",
          "| Road | Section | Length stated / measured | Stated maximum | "
          "Whole section | Steepest 100 m | Steepest 50 m | Verdict |",
          "|---|---|---|---|---|---|---|---|"]
    for _, r in td.iterrows():
        ln = ("n/a" if not np.isfinite(r["length_m"])
              or r["verdict"] == "not in this run" else f"{r['length_m']:.0f} m")
        if not np.isfinite(r["computed"]):
            L.append(f"| {r['street']} | {r['section']} | "
                     f"{r['stated_length_m']:.0f} m / {ln} | {_pct(r['published'])} | "
                     f"n/a | n/a | n/a | {r['verdict']} |")
            continue
        L.append(f"| {r['street']} | {r['section']} | "
                 f"{r['stated_length_m']:.0f} m / {ln} | **{_pct(r['published'])}** | "
                 f"{_pct(r['mean_grade'])} | {_pct(r['steepest_100'])} | "
                 f"{_pct(r['steepest_50'])} | {r['verdict']} |")
    measured = td[td["computed"].notna()]
    absent = int((td["verdict"] == "not in this run").sum())
    lost = int((td["verdict"] == "not matched").sum())
    if len(measured):
        tail = "; ".join(t for t in (f"{absent} not in this run" if absent else "",
                                     f"{lost} not matched" if lost else "") if t)
        L += ["", f"**Verdict:** the stated maximum lies between the model's "
              f"two figures on **{int((measured['verdict'] == 'ok').sum())} of "
              f"{len(measured)}** measured sections"
              + (f" ({tail})." if tail else "."), ""]
    else:
        L += ["", "**Verdict:** none of these is in this run.", ""]

    ratio = steep[steep["group"] == "ratio"]
    L += ["### 3b. Signed and published ratios on the old streets", "",
          "Ratios for the streets that climb straight up the hillside above "
          "Central and Sai Ying Pun, from Wikipedia's street articles and "
          "OpenStreetMap's `incline` tags, which mostly copy the steep-hill "
          "signs. No primary source was found for them, so this table "
          "carries less weight than the other two. Several are ladder "
          "streets, part roadway and part flights of steps, so every class "
          "of way counts here and the last column says how much of the "
          "section is stairway. A sign states the steepest pitch, so the "
          f"ratio is set against the steepest {RATIO_WINDOW_M:g} m (in bold), "
          f"and counts as agreeing within {100 * RATIO_TOLERANCE:g} percentage "
          "points.", "",
          "| Street | Section | Published | **Steepest 50 m** | Steepest 100 m | "
          "Mean | Difference | Verdict | Steps |",
          "|---|---|---|---|---|---|---|---|---|"]
    for _, r in ratio.iterrows():
        df_ = "n/a" if not np.isfinite(r["diff"]) else f"{100 * r['diff']:+.1f}"
        c = "n/a" if not np.isfinite(r["computed"]) else f"**{_pct(r['computed'])}**"
        L.append(f"| {r['street']} {r['name_zh']} | {r['section']} | "
                 f"{r['ratio']} ({_pct(r['published'])}) | {c} | "
                 f"{_pct(r['steepest_100'])} | {_pct(r['mean_grade'])} | {df_} | "
                 f"{r['verdict']} | {_pct(r['steps_share'], 0)} |")
    L += _steep_verdict(ratio, f"{100 * RATIO_TOLERANCE:g} percentage points")

    bm = steep[steep["group"] == "benchmark"]
    L += ["### 3c. Mean gradient between two bench marks on one road", "",
          "The strongest of the three, because both heights are levelled: "
          "the rise between two marks on the same road, over the distance "
          "between them along the road (measured on OpenStreetMap "
          "centrelines by the research for this port), against the model's "
          "own rise over its own distance between the same two points. It "
          "counts as agreeing within "
          f"{100 * BENCHMARK_GRADE_TOLERANCE:g} percentage point.", "",
          "| Road | Marks | Rise by the marks / in the model | Distance researched / "
          "in the model | Mean by the marks | **Mean in the model** | "
          "Difference | Verdict |", "|---|---|---|---|---|---|---|---|"]
    for _, r in bm.iterrows():
        measured = np.isfinite(r["computed"])
        rise = (f"{r.get('mark_rise_m', np.nan):.1f} m / "
                + (f"{r.get('model_rise_m', np.nan):.1f} m" if measured else "n/a")
                if np.isfinite(r.get("mark_rise_m", np.nan)) else "n/a")
        ln = "n/a" if not np.isfinite(r["length_m"]) else f"{r['length_m']:,.0f} m"
        df_ = "n/a" if not measured else f"{100 * r['diff']:+.1f}"
        c = "n/a" if not measured else f"**{_pct(r['computed'])}**"
        L.append(f"| {r['street']} | {r['section'].replace('bench marks ', '')} | "
                 f"{rise} | {r['stated_length_m']:,.0f} m / {ln} | "
                 f"{_pct(r['published'])} | {c} | {df_} | {r['verdict']} |")
    L += _steep_verdict(bm, f"{100 * BENCHMARK_GRADE_TOLERANCE:g} percentage point")

    for _, r in steep.iterrows():
        if r["verdict"] in ("under-reported", "over-reported"):
            what = {"td": "the Transport Department's maximum",
                    "ratio": "the published ratio",
                    "benchmark": "the gradient between its bench marks"}[r["group"]]
            got = (f"{_pct(r['mean_grade'])} over the whole section and "
                   f"{_pct(r['steepest_100'])} over its steepest 100 m"
                   if r["group"] == "td" else _pct(r["computed"]))
            misses.append(f"**{r['street']}** ({r['section']}) is "
                          f"{r['verdict']}: {got} against {what} of "
                          f"{_pct(r['published'])}.")
            kinds.add("benchmark" if r["group"] == "benchmark" else r["verdict"])
        elif r["verdict"] == "not matched":
            misses.append(f"**{r['street']}** ({r['section']}) could not be "
                          f"matched to the network: {r.get('why', '')}.")
            kinds.add("not matched")

    # ---- 4 ---------------------------------------------------------------
    L += ["## 4. Level corridors, contour roads and roads that are not level", "",
          "A street is called *level* when it climbs less than "
          f"{LEVEL_MAX_CLIMB_PER_KM:g} m per kilometre (the mean of the two "
          f"directions) and stays within a {LEVEL_MAX_RANGE_M:g} m band from "
          "end to end. The first is upstream's test. The second is needed "
          "here because the roads Hong Kong must *not* call level are long "
          "and gentle: a road that rises steadily for kilometres climbs "
          "little in any one of them. The bench marks are the official ones "
          "whose locality names the street and which stand within "
          f"{BENCHMARK_STREET_OFFSET_M:g} m of it.", "",
          "| Corridor | Km | Climb per km | Elevation in the model | Bench "
          "marks (lowest to highest) | Expected | Verdict | Found as a corridor? |",
          "|---|---|---|---|---|---|---|---|"]
    for _, r in flat.iterrows():
        if r["verdict"] == "not in this run":
            L.append(f"| {r['corridor']} | n/a | n/a | n/a | n/a | "
                     f"{r['expected']} | not in this run | n/a |")
            continue
        mk = ("none" if not r["n_marks"] else
              f"{int(r['n_marks'])}: {r['mark_min']:.1f} to {r['mark_max']:.1f} m")
        L.append(f"| {r['corridor']} | {r['street_km']:.1f} | "
                 f"{r['gain_per_km']:.1f} m | {r['elev_min']:.1f} to "
                 f"{r['elev_max']:.1f} m | {mk} | {r['expected']} | "
                 f"**{r['verdict']}** | "
                 f"{'n/a' if r['discovered'] is None else 'yes' if r['discovered'] else 'no'} |")
    seen = flat[flat["verdict"] != "not in this run"]
    good = int(seen["as_expected"].fillna(False).astype(bool).sum())
    L += ["", f"**Verdict:** **{good} of {len(seen)}** corridors in this run "
          "come out as expected"
          + (f" ({len(flat) - len(seen)} not in this run)." if len(seen) < len(flat) else "."),
          "",
          "Kilometres count every carriageway and named footway of the "
          "street. A street's band in the model is wider than its bench "
          "marks' wherever the street runs on beyond the marks, and, on the "
          "contour roads, wherever the bare-earth survey dips into a gully "
          "the road crosses on a culvert; that residue is the climb per "
          "kilometre a level hillside road still shows. A corridor is "
          "*found* when one of its streets appears in the names of a "
          "corridor the analysis discovered for itself, which a street that "
          "leads to few neighbourhoods need not be"
          + ("." if flat["discovered"].notna().any() else
             " (n/a here: the corridor analysis has not been run)."), ""]
    for _, r in seen.iterrows():
        if not r["as_expected"]:
            kinds.add("level")
            misses.append(f"**{r['corridor']}** measures {r['verdict']} "
                          f"({r['gain_per_km']:.1f} m of climbing per km, "
                          f"{r['elev_min']:.0f} to {r['elev_max']:.0f} m) where "
                          f"it should be {r['expected']}.")

    # ---- 5 ---------------------------------------------------------------
    L += ["## 5. Round on the contour road, not over the hill", ""]
    if trip.get("routes"):
        L += [f"From {trip['from']} to {trip['to']}: both ends are part of "
              "the way up the same hillside, so a route has little height to "
              "gain and every metre it climbs beyond that is wasted. This is "
              "Hong Kong's counterpart of upstream's Wiggle check, and is "
              "measured the same way, as *excess climbing*: what a route "
              "climbs beyond the difference in height between its ends. The "
              "streets are the ones the router chose, longest first.", ""]
        names = {"shortest": "Shortest", "balanced": "Balanced",
                 "min_climb": "Flattest"}
        for mode, r in trip["routes"].items():
            L += [f"**{'On foot' if mode == 'walk' else 'By bicycle'}**", "",
                  "| Objective | Distance | Climb | Excess climb | High point | "
                  "Steepest pitch | Streets carrying it |",
                  "|---|---|---|---|---|---|---|"]
            for pname, o in r["objectives"].items():
                streets = ", ".join(f"{n} ({v:.0f} m)" for n, v in o["streets"][:4])
                L.append(f"| {names[pname]} | {o['km']:.2f} km | "
                         f"{o['gain_m']:.0f} m | **{o['excess_m']:.0f} m** | "
                         f"{o['high_point_m']:.0f} m | {_pct(o['max_grade'])} | "
                         f"{streets} |")
            s, f = r["objectives"]["shortest"], r["objectives"][r["flat_objective"]]
            round_names = " / ".join(CONTOUR_TRIP["round"])
            over_names = " / ".join(CONTOUR_TRIP["over"])
            # "0% against 0%" says nothing; say that neither goes that way
            if round(100 * f["over_share"]) == 0 and round(100 * s["over_share"]) == 0:
                over = f"Neither touches {over_names}, higher up the hill."
            else:
                over = (f"{100 * f['over_share']:.0f}% of it is on {over_names}, "
                        f"higher up the hill, against "
                        f"{100 * s['over_share']:.0f}% of the shortest.")
            L += ["", f"The shortest route throws away "
                  f"{r['excess_shortest_m']:.0f} m of climbing and the flat one "
                  f"{r['excess_flat_m']:.0f} m, for "
                  f"{100 * (f['km'] / s['km'] - 1):+.0f}% of distance. "
                  f"The flat route tops out at {f['high_point_m']:.0f} m and "
                  f"the shortest at {s['high_point_m']:.0f} m. "
                  f"{100 * f['round_share']:.0f}% of the flat route runs on "
                  f"the contour roads ({round_names}) against "
                  f"{100 * s['round_share']:.0f}% of the shortest. " + over
                  + ("" if not np.isfinite(r["corridor_share"]) else
                     f" {100 * r['corridor_share']:.0f}% of the flat route "
                     "lies along corridors the analysis found for itself.")
                  + f" **Verdict: {r['verdict']}.**", ""]
            if r["verdict"] != "goes round":
                misses.append(f"**The contour trip** by "
                              f"{'foot' if mode == 'walk' else 'bicycle'}: the "
                              f"flat route wastes {r['excess_flat_m']:.0f} m of "
                              f"climbing against the shortest route's "
                              f"{r['excess_shortest_m']:.0f} m, more than half.")
        L += ["A flat route *goes round* when it wastes at most half the "
              "climbing the shortest one does (upstream's test for the "
              "Wiggle).", ""]
    else:
        L += [f"_Not run: {trip.get('reason', 'the trip is not in this run')}._", ""]

    # ---- 6 ---------------------------------------------------------------
    L += ["## 6. The gaps", ""]
    if len(passes):
        L += ["The pass analysis finds, exactly, the lowest high point "
              "anyone must reach between two places. Between two places "
              "either side of one of these gaps the answer on the whole "
              "network is rarely the gap, because a walker in no hurry goes "
              "round by the coast. So each gap is asked about as a gap is "
              "defined: take an official bench mark on the road down each "
              "side and only the streets and paths within a circle that "
              "just holds both; the lowest crossing between the two marks "
              "inside that circle is the model's pass. Beside it is the high "
              "point of the shortest walking route between the same marks, "
              "found by the router on the whole network. Published heights "
              "are OpenStreetMap's for the saddle. They are map readings, and "
              "at the gaps where a bench mark stands as well the two differ "
              "by several metres, so where a mark stands within "
              f"{GAP_MARK_M:g} m of the mapped saddle its levelled height is "
              "given too, and the model is judged against whichever is "
              f"nearer, to {GAP_TOLERANCE_M:g} m. A circle can also hold a "
              "second, lower gap, and the lowest crossing is then over that "
              "one. So when the crossing found is more than "
              f"{GAP_NEAR_M:g} m from the mapped saddle it is not taken as "
              "this gap's height: the gap is read from the shortest route "
              "instead, if that route tops out at the gap, and the verdict "
              "says so. The figure judged is the one in bold.", "",
              "| Gap | Published | Bench mark at the gap | Lowest crossing "
              "in the model | On | From the mapped saddle | Raw LiDAR "
              "there | High point of the shortest route | Difference | Verdict |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in passes.iterrows():
            mk = ("none" if not np.isfinite(r["mark_m"]) else
                  f"{r['mark_m']:.1f} m ({r['mark_station']})")
            if not np.isfinite(r["pass_m"]):
                why = f" ({r['why']})" if r["why"] else ""
                L.append(f"| {r['gap']} {r['name_zh']} | {r['published_m']:.0f} m | "
                         f"{mk} | n/a | | | | | | {r['verdict']}{why} |")
                continue
            by_route = r["judged"] == "route"
            crossing, top = f"{r['pass_m']:.1f} m", _m(r["route_high_m"])
            if by_route:
                top = f"**{top}**"
            else:
                crossing = f"**{crossing}**"
            L.append(f"| {r['gap']} {r['name_zh']} | {r['published_m']:.0f} m | "
                     f"{mk} | {crossing} | {r['pass_street']} | "
                     f"{r['pass_offset_m']:.0f} m | {_m(r['raw_m'])} | "
                     f"{top} | {r['diff_m']:+.1f} m | {r['verdict']}"
                     + (" (on the shortest route; the lowest crossing is "
                        "another way round)" if by_route else "") + " |")
        seen = passes[np.isfinite(passes["pass_m"])]
        L += ["", f"**Verdict:** **{int((seen['verdict'] == 'agrees').sum())} of "
              f"{len(seen)}** gaps measured in this run agree with a published "
              "or levelled height"
              + (f" ({len(passes) - len(seen)} not measured)."
                 if len(seen) < len(passes) else "."), "",
              "*Raw LiDAR there* is the terrain model itself, unsmoothed, at "
              "the top of the edge the pass is on: where it agrees with the "
              "model and not with the published height, it is the published "
              "height that is rough. *Lower way round* would mean the model "
              "found, inside the circle, a crossing lower than the gap and "
              "away from it, and that the shortest route did not top out at "
              "the gap either; the street named is where.", ""]
        for _, r in seen.iterrows():
            if r["verdict"] in ("higher", "lower"):
                kinds.add("gap")
                got = (f"{r['route_high_m']:.1f} m on the shortest route"
                       if r["judged"] == "route" else
                       f"{r['pass_m']:.1f} m on {r['pass_street']}")
                misses.append(f"**{r['gap']}** comes out {abs(r['diff_m']):.1f} m "
                              f"{r['verdict']} than its nearer reference: "
                              f"{got} against "
                              f"a mapped {r['published_m']:.0f} m"
                              + (f" and a bench mark at {r['mark_m']:.1f} m"
                                 if np.isfinite(r["mark_m"]) else "") + ".")
            elif r["verdict"] == "lower way round":
                kinds.add("gap")
                misses.append(f"**{r['gap']}**: the model crosses at "
                              f"{r['pass_m']:.1f} m on {r['pass_street']}, "
                              f"{r['pass_offset_m']:.0f} m from the mapped "
                              f"saddle, lower than the gap's "
                              f"{r['published_m']:.0f} m.")
    else:
        L += [f"_Not run: {passes.attrs.get('reason', 'no result')}._", ""]

    # ---- 7 ---------------------------------------------------------------
    L += ["## 7. Internal invariants", "",
          "Identities the tables must satisfy whatever the terrain; upstream "
          "asserts them in its test suite, and they are recomputed here on "
          "the build this report describes.", "",
          "| Invariant | Checked | Violations | Worst deviation | Verdict |",
          "|---|---|---|---|---|"]
    for _, r in inv.iterrows():
        worst = (f"{r['worst']:.3g} {r['unit']}".strip() if r["worst"] else "none")
        L.append(f"| {r['invariant']} | {r['checked']:,} | {r['violations']:,} | "
                 f"{worst} | {r['verdict']} |")
        if r["verdict"] != "holds":
            misses.append(f"**Invariant violated**: {r['invariant']} "
                          f"({r['violations']:,} cases).")
    L += ["", f"**Verdict:** **{int((inv['verdict'] == 'holds').sum())} of "
          f"{len(inv)}** hold.", ""]

    # ---- 8 ---------------------------------------------------------------
    L += ["## 8. Targets the model does *not* reproduce", ""]
    if misses:
        L += [f"- {m}" for m in misses]
        L += ["", "None of these was tuned away. What lies behind each kind:", ""]
        L += [text for kind, text in _MISS_NOTES if kind in kinds]
        L += [""]
    else:
        L += ["_Every target in this run was reproduced._", ""]

    VALIDATION_MD.parent.mkdir(parents=True, exist_ok=True)
    VALIDATION_MD.write_text("\n".join(L), encoding="utf-8")
    log.info("wrote %s", VALIDATION_MD.name)


#: What lies behind each kind of disagreement; section 8 prints the ones
#: that occur in the run.
_MISS_NOTES = (
    ("not matched",
     "- **Not matched.** The network holds only ways open to walkers or "
     "cyclists, under the names the map gives them. A documented section "
     "that runs on a flyover, or through a stretch the map tags as a road "
     "tunnel, closed to both, or on a road the access rules have (rightly "
     "or wrongly) closed, has nothing to be measured on. That "
     "is a statement about the street network, not about the elevations."),
    ("under-reported",
     "- **Under-reported.** A published ratio is the steepest pitch, "
     "sometimes a few tens of metres long, and on a ladder street that "
     "pitch is a flight of steps. The elevation profile is smoothed over "
     f"about {2 * ELEVATION.smooth_window_m:g} m along the street to suppress "
     "survey noise, which takes the "
     "top off the shortest pitches. Since the object is to find flat routes, "
     "shaving the peak of a wall is a far cheaper error than inventing "
     "gradients on level ground."),
    ("over-reported",
     "- **Over-reported.** The terrain model is bare earth and has no "
     "decks. Where a road runs on a viaduct or a retaining structure that "
     "the map does not flag as a bridge, the model follows the ground "
     "underneath, and a hillside road gains dips and pitches it does not "
     "have."),
    ("benchmark",
     "- **Between bench marks.** Some marks are older than the road now "
     "beside them. Where a road has been rebuilt on a new line or level, the "
     "nearest centreline to a mark is no longer the road it was levelled on, "
     "and the rise between two marks is not the rise of today's road; "
     "section 1's largest differences show which marks those are."),
    ("level",
     "- **Level, or not.** The same bare-earth dips give a level hillside "
     "road climbing it does not have; and a long street takes in ends that "
     "its bench marks do not."),
    ("gap",
     "- **Gaps.** The published heights are map readings. Where the raw "
     "LiDAR at the pass reads as the model does, the difference is in the "
     "published figure; where it does not, the model's handling of that "
     "edge (a bridge, a flight of steps) has moved it."),
)


def _steep_verdict(sub: pd.DataFrame, tolerance: str) -> list[str]:
    """The one-line verdict under a table of gradients."""
    measured = sub[sub["computed"].notna()]
    absent = int((sub["verdict"] == "not in this run").sum())
    lost = int((sub["verdict"] == "not matched").sum())
    unrun = sub[sub["verdict"] == "not run"]
    tail = "; ".join(t for t in (f"{absent} not in this run" if absent else "",
                                 f"{lost} not matched" if lost else "",
                                 f"{len(unrun)} not run" if len(unrun) else "") if t)
    if measured.empty and len(unrun):
        # the reference itself is missing (no bench mark file), which is
        # not the same thing as the streets being outside the run
        why = next((w for w in unrun.get("why", pd.Series(dtype=object))
                    if isinstance(w, str) and w), "its reference is missing")
        return ["", f"**Verdict:** not run: {why}.", ""]
    if measured.empty:
        return ["", "**Verdict:** none of these is in this run.", ""]
    return ["", f"**Verdict:** **{int((measured['verdict'] == 'ok').sum())} of "
            f"{len(measured)}** measured sections agree within {tolerance}; "
            f"median difference {100 * measured['diff'].median():+.1f} points"
            + (f" ({tail})." if tail else "."), ""]
