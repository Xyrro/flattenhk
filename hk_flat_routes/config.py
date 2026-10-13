"""Central configuration: paths, CRS, grade thresholds and routing weights.

Everything tunable in the analysis lives here so that experiments are
reproducible and the cost model is auditable.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
#: Where the code is: the checkout, or ``site-packages`` for a copy installed
#: with ``pip install .``.
CODE_ROOT = Path(__file__).resolve().parent.parent


def _data_root(code_root: Path, env_root: str | None, cwd: Path) -> Path:
    """Where ``data/``, ``outputs/`` and ``site/`` are kept.

    ``HKFR_ROOT`` when it is set, and otherwise the checkout the code runs
    from.  A copy installed into ``site-packages`` has no checkout around
    it, and 14 GB of terrain does not belong among the installed packages:
    such a copy works in the current directory instead.
    """
    if env_root:
        return Path(env_root).absolute()
    if {"site-packages", "dist-packages"} & set(code_root.parts):
        return cwd
    return code_root


PROJECT_ROOT = _data_root(CODE_ROOT, os.environ.get("HKFR_ROOT"), Path.cwd())
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
#: ``HKFR_RUN_DIR`` redirects every processed and output path under one
#: directory, so an experiment (see ``sensitivity.py``) can rebuild the whole
#: pipeline with different parameters without touching the main results.
_RUN_DIR = os.environ.get("HKFR_RUN_DIR")
PROCESSED_DIR = Path(_RUN_DIR) / "processed" if _RUN_DIR else DATA_DIR / "processed"
OUTPUT_DIR = Path(_RUN_DIR) / "outputs" if _RUN_DIR else PROJECT_ROOT / "outputs"
#: The route finder as a static site, deployed to GitHub Pages from here.
SITE_DIR = Path(_RUN_DIR) / "site" if _RUN_DIR else PROJECT_ROOT / "site"
PRODUCT_NAME = "Flatten HK"
CITY_NAME = "Hong Kong"
#: Where this port's source lives and where its site is served.  Both can be
#: overridden from the environment, so a fork builds with its own links.
REPO_URL = os.environ.get("HKFR_REPO_URL", "https://github.com/Xyrro/flattenhk")
SITE_URL = os.environ.get("HKFR_SITE_URL", "https://xyrro.github.io/flattenhk/")
if not SITE_URL.endswith("/"):
    SITE_URL += "/"
#: A custom domain for GitHub Pages (written to ``site/CNAME``).  Empty for
#: a project site served from ``<user>.github.io/<repo>``, which must not
#: ship a CNAME file.
SITE_DOMAIN = os.environ.get("HKFR_SITE_DOMAIN", "")
#: The project this one is a port of; the page, the README and the licence
#: credit it from here.
UPSTREAM_NAME = "Flatten SF"
UPSTREAM_SITE = "https://flattensf.com/"
UPSTREAM_REPO = "https://github.com/almostimplemented/flattensf"
UPSTREAM_AUTHOR = "Drew Edwards"
UPSTREAM_AUTHOR_URL = "https://almostimplemented.com"

for _d in (RAW_DIR, PROCESSED_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Coordinate reference systems
# --------------------------------------------------------------------------
#: Geographic CRS of the source vector data (Overture) and of all map output.
CRS_GEOGRAPHIC = "EPSG:4326"
#: Projected CRS used for *every* length, slope and distance computation.
#: HK1980 Grid -- the native CRS of the CEDD LiDAR terrain model, so
#: elevation sampling needs no reprojection of the raster. It is a
#: transverse Mercator centred on the territory with unit scale factor;
#: scale distortion is below 1 part in 10,000 anywhere in Hong Kong.
CRS_PROJECTED = "EPSG:2326"

# --------------------------------------------------------------------------
# Study area
# --------------------------------------------------------------------------
#: Analysis bounding box (lon_min, lon_max, lat_min, lat_max).
#: Covers the whole Hong Kong SAR, from Tai O to Tung Ping Chau and from
#: Po Toi to the Shenzhen River.  The box also takes in the south of
#: Shenzhen; the street network is clipped to the SAR boundary afterwards.
STUDY_BBOX = (113.8200, 114.4500, 22.1300, 22.5800)

#: Which statutory planning scheme areas stand for "neighborhoods".  The
#: Town Planning Board publishes 151, from Central District to stretches of
#: country-park fringe; the analysis wants a few dozen places people would
#: name, comparable in number to San Francisco's 37.  These are every plan
#: on Hong Kong Island (S/H...) and in Kowloon (S/K...), the New
#: Territories' new towns, and the settled parts of Lantau: 56 units once
#: the two reclamation strips below are folded into their parents.
NEIGHBORHOOD_PLAN_PREFIXES = ("S/H", "S/K")
NEIGHBORHOOD_PLANS = (
    "S/KC", "S/TW", "S/TWW", "S/TY", "S/ST", "S/MOS", "S/TKO", "S/TP",
    "S/FSS", "S/YL", "S/TSW", "S/TM", "S/SK-SKT",
    "S/I-TCTC", "S/I-DB", "S/I-MWF", "S/SLC", "S/I-TOTC",
)
#: Plans that are an extension of another, not a place of their own:
#: Central District (Extension) and Wan Chai North are the reclaimed
#: waterfronts of Central and Wan Chai.
NEIGHBORHOOD_MERGE = {"S/H24": "S/H4", "S/H25": "S/H5"}

#: Neighborhoods left out of pair routing altogether.  Nothing needs to be
#: listed for the harbour: pairs are routed only within a landmass (see
#: ``MIN_COMPONENT_NODES``), so an island is simply not paired with the rest.
EXCLUDED_NEIGHBORHOODS: tuple = ()

# --------------------------------------------------------------------------
# Elevation sampling / smoothing
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ElevationConfig:
    #: Spacing (m) of elevation samples along each edge. Upstream chose 5 m
    #: empirically in San Francisco: at 10 m the short steep pitches were
    #: measurably clipped (a documented 41% street read 36.8%), while 5 m
    #: recovers them. Hong Kong's stepped streets and stairways are shorter
    #: and steeper still. Cost is about 9.4 million sample points
    #: territory-wide.
    sample_spacing_m: float = 5.0
    #: Minimum number of samples per edge (endpoints always included).
    min_samples: int = 3
    #: Half-width (m) of the Savitzky-Golay window applied along the profile.
    #: 25 m either side => ~50 m window, shorter than a city block, so real
    #: block-scale grade is preserved while kerb/vehicle/vegetation artefacts
    #: in the lidar DEM are suppressed. Upstream tested narrower windows in
    #: San Francisco and rejected them: at a 12.5 m half-width localised DEM
    #: artefacts survived and pushed two streets to the plausibility clip,
    #: whereas 25 m returned them within two points of their documented
    #: gradients. With it, Chai Wan Road and Shatin Pass Road come out at
    #: 10.6% and 16.8% here against the Transport Department's 11% and 17%.
    smooth_window_m: float = 25.0
    #: Polynomial order of the Savitzky-Golay filter.
    smooth_polyorder: int = 2
    #: Dead-band (m). Elevation wiggles smaller than this are not counted as
    #: cumulative gain/loss. Standard practice in altimetry; without it a flat
    #: street accumulates tens of metres of phantom climbing from DEM noise.
    gain_deadband_m: float = 0.5
    #: Grades above this magnitude are treated as DEM artefacts and clipped.
    #: The steepest roads in Hong Kong open to traffic are about 1 in 4
    #: (25%: Centre Street, Peel Street); stairways and ladder streets reach
    #: 50-60%, so the clip is generous and only removes physically
    #: impossible values.
    max_plausible_grade: float = 0.60
    #: Elevation on bridges/tunnels is taken as a linear interpolation between
    #: the endpoints instead of from the DEM (the DEM samples the ground or
    #: water surface beneath the structure).
    interpolate_structures: bool = True
    #: Standard deviation (m) of the Gaussian applied to the DEM before any
    #: sampling; 0 disables it.  San Francisco uses 3 m on a 1 m DEM.  Hong
    #: Kong's hillside roads are terraces four to eight metres wide between
    #: a retaining wall and a drop, and a 3 m Gaussian averages the wall and
    #: the drop into the road; 1.5 m (three cells of the 0.5 m model) still
    #: damps kerbs and survey noise and keeps the terrace a terrace.
    dem_sigma_m: float = 1.5
    #: How far (m) either side of the mapped centreline to look for the road
    #: itself, where the ground falls away across it.  A street's centreline
    #: in OpenStreetMap is good to a couple of metres; on a level street
    #: that costs nothing, but on a path cut into a 60% hillside two metres
    #: sideways is more than a metre up or down, and it wanders, so a level
    #: path (Bowen Road, a catchwater) reads as a string of small climbs.
    #: A road is flat across its width and the hillside is not, so each
    #: sample is taken at the flattest spot within this distance, with a
    #: preference for staying put (``lateral_penalty``).  0 disables it.
    lateral_search_m: float = 6.0
    #: Step (m) of that search.
    lateral_step_m: float = 1.0
    #: Cross-slope (rise over run) a sample must show at the centreline
    #: before the search is tried at all; below it the centreline stands.
    lateral_min_cross_slope: float = 0.06
    #: Cost of moving sideways, in cross-slope per metre moved: the search
    #: leaves the centreline only for ground that much flatter.
    lateral_penalty: float = 0.04
    #: Window (m) of the running median applied along each street before the
    #: Savitzky-Golay filter.  A bare-earth model has no bridge decks, so a
    #: path that crosses a gully on an untagged culvert or footbridge dips
    #: into the gully and climbs out again; the median removes dips and
    #: bumps narrower than half the window and leaves any sustained slope
    #: exactly as it is.  0 disables it.
    despike_window_m: float = 35.0
    #: Ground lower than this (m above Principal Datum) is treated as having
    #: no data.  Nothing a street runs on in Hong Kong is below the sea:
    #: reclaimed land is finished at four or five metres, and the Principal
    #: Datum is itself more than a metre under mean sea level.  What the
    #: 2020 survey recorded down there are construction pits (the Kai Tak
    #: runway park was a hole ten metres deep that winter, and has streets
    #: across it now); sampled at face value they put a ten-metre dip in a
    #: level road.  Treated as missing, such stretches are bridged from the
    #: ground either side like any other gap.
    min_ground_elev_m: float = -1.0
    #: Bridge and tunnel ends.  The node where a deck meets the ground is
    #: mapped a little past the abutment as often as not, and there the
    #: bare-earth model is already falling away under the deck (or, at a
    #: tunnel mouth, already climbing the hill above it).  Taken at face
    #: value that drags the end of every bridge down to the stream it
    #: crosses and puts a five-metre dip into a level path at each one.
    #: So within ``abutment_blank_m`` of a structure the model is not
    #: trusted: the end's elevation is extrapolated from the next
    #: ``abutment_fit_m`` of the approach, and replaces the sampled one when
    #: they differ by more than ``abutment_min_shift_m``.  0 disables it.
    abutment_blank_m: float = 10.0
    abutment_fit_m: float = 25.0
    abutment_min_shift_m: float = 1.0


def _overrides() -> dict:
    """Parameter overrides from ``HKFR_OVERRIDES`` (a JSON object).

    Used by the sensitivity harness to rebuild the pipeline under different
    settings. Keys are ``ElevationConfig`` or ``AnalysisConfig`` field names.
    """
    raw = os.environ.get("HKFR_OVERRIDES")
    if not raw:
        return {}
    import json
    try:
        parsed = json.loads(raw)
    except ValueError:
        parsed = None
    # an object and nothing else: ``dict()`` would also take a list of pairs
    if not isinstance(parsed, dict):
        raise ValueError(f"HKFR_OVERRIDES is not a JSON object: {raw!r}")
    return parsed


_OV = _overrides()


def _coerce(name: str, value, default):
    """An override as the type of the field's default, or a ``ValueError``.

    JSON has one kind of number, so a spacing written as 10 has to become
    10.0, and a number that arrives as a string ("1", from a shell
    one-liner) is taken as that number.  What cannot be read as the field's
    type is refused instead of stored: a 2.5 for a count would otherwise be
    cut to 2 further down, and any non-empty string counts as true.
    """
    import math

    kind = type(default)
    wanted = {bool: "true or false", int: "a whole number", float: "a number"}
    try:
        if kind is bool:
            if isinstance(value, bool) or value in (0, 1):
                return bool(value)
            raise ValueError
        if kind in (int, float):
            if isinstance(value, bool):
                raise ValueError
            number = float(value)
            if not math.isfinite(number) or (kind is int and number != int(number)):
                raise ValueError
            return kind(number)
    except (TypeError, ValueError):
        raise ValueError(f"HKFR_OVERRIDES: {name} takes {wanted[kind]} (its "
                         f"default is {default!r}), not {value!r}") from None
    return value


def _apply(instance, overrides: dict):
    """``instance`` with the overrides that name one of its fields applied."""
    picked = {k: _coerce(k, v, getattr(instance, k))
              for k, v in overrides.items() if k in instance.__dataclass_fields__}
    return replace(instance, **picked) if picked else instance


def _refuse_unknown(overrides: dict, *configs) -> None:
    """Fail on an override that names no parameter of ``configs``.

    Dropping it quietly, as this used to, builds a run that is labelled as
    a perturbation and is really the baseline: ``smooth_window`` for
    ``smooth_window_m`` is all it takes.
    """
    known = sorted({name for c in configs for name in c.__dataclass_fields__})
    unknown = sorted(set(overrides) - set(known))
    if unknown:
        raise ValueError(f"HKFR_OVERRIDES: unknown parameter(s) {unknown}; "
                         f"the parameters are {known}")


ELEVATION = _apply(ElevationConfig(), _OV)

# --------------------------------------------------------------------------
# Grade thresholds
# --------------------------------------------------------------------------
#: Grade thresholds (as fractions) at which "distance above grade" is tallied.
#: 3% = noticeable on a bike; 5% = sustained-effort threshold; 8% = hard
#: (1 in 12, the steepest ramp Hong Kong's barrier-free design manual
#: allows); 10% = very hard; 15% = extreme, walk-your-bike territory.
GRADE_THRESHOLDS = (0.03, 0.05, 0.08, 0.10, 0.15)

#: Percentile used for the "typical steep" grade statistic of an edge.
GRADE_PERCENTILE = 95

#: Minimum edge length (m) for its *maximum* grade to be treated as reliable.
#: Over a 5 m stub a single decimetre of DEM artefact reads as a 20% grade
#: (upstream's worst example was a 5 m connector in a flat part of San
#: Francisco reporting 41%). Short edges keep their metrics but are excluded
#: from maximum-grade tests, which are instead judged on average grade.
#: Two in five of Hong Kong's edges are this short: its footways are mapped
#: in small pieces.
MIN_RELIABLE_GRADE_LENGTH_M = 15.0

# --------------------------------------------------------------------------
# Routing cost model
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CostWeights:
    """Weights for the additive edge cost model.

    The cost of traversing a directed edge is, in "equivalent metres":

        cost = length_m * mode_multiplier
             + alpha * cumulative_gain_m
             + beta  * sum_k(w_k * distance_above_threshold_k)
             + gamma * extreme_grade_distance_m
             + turn/one-way handling

    ``alpha`` is the horizontal distance a walker/rider would happily trade
    for one metre of climbing.  The classic Naismith rule (walking) is ~7.9 m
    per metre climbed; cyclists are far more climb-averse, so the bicycle
    default is higher.
    """
    name: str = "balanced"
    #: metres of equivalent detour accepted per metre of climbing
    alpha: float = 8.0
    #: multiplier on the graded steep-distance penalty
    beta: float = 1.0
    #: multiplier on the extreme-grade penalty
    gamma: float = 1.0
    #: per-threshold penalty weights (equivalent metres per metre travelled
    #: above the threshold). Applied cumulatively, so a 12% grade incurs the
    #: 3%, 5%, 8% and 10% penalties -- this makes the penalty grow
    #: nonlinearly (super-linearly) with grade.
    threshold_penalties: tuple = (0.25, 0.75, 2.0, 4.0, 10.0)
    #: Extra penalty (equivalent metres per metre) for distance above the
    #: highest threshold, scaled by gamma.
    extreme_extra: float = 10.0
    #: Cap on total cost inflation relative to plain length, to stop a single
    #: objective producing absurd detours. ``None`` disables the cap.
    max_detour_ratio: float | None = None
    #: Whether the mode's per-class comfort multipliers apply. The "shortest"
    #: objective sets this False so that it minimises *distance only*, as the
    #: analysis specification requires. Without the flag, the bicycle comfort
    #: weights (a trunk road at 1.9x, a cycle track at 0.85x) made the
    #: "shortest" bicycle route longer in real metres than the flat route,
    #: which made every distance-penalty comparison meaningless.
    use_class_multiplier: bool = True


#: The four routing objectives required by the analysis.
ROUTING_PROFILES: dict[str, CostWeights] = {
    # A. Shortest -- distance only.
    "shortest": CostWeights(name="shortest", alpha=0.0, beta=0.0, gamma=0.0,
                            threshold_penalties=(0, 0, 0, 0, 0), extreme_extra=0.0,
                            use_class_multiplier=False),
    # B. Minimum climbing -- climbing dominates, grade shape ignored.
    "min_climb": CostWeights(name="min_climb", alpha=120.0, beta=0.0, gamma=0.0,
                             threshold_penalties=(0, 0, 0, 0, 0), extreme_extra=0.0),
    # C. Grade-averse -- steep *segments* dominate, total climbing secondary.
    "grade_averse": CostWeights(name="grade_averse", alpha=4.0, beta=12.0, gamma=6.0,
                                threshold_penalties=(0.25, 1.0, 3.0, 6.0, 15.0),
                                extreme_extra=15.0),
    # D. Balanced "flat but reasonable".
    "balanced": CostWeights(name="balanced", alpha=14.0, beta=2.0, gamma=2.0,
                            threshold_penalties=(0.25, 0.75, 2.0, 4.0, 10.0),
                            extreme_extra=10.0),
}

#: Sweep used to trace the distance/climb Pareto frontier. Each value scales
#: *every* climbing and grade term of the balanced profile together (alpha,
#: beta and gamma), so 0 is pure distance, 1 is the balanced objective and
#: large values approach the flattest possible route. Scaling only alpha,
#: as an earlier version did, left beta and gamma in force at the "shortest"
#: end and the frontier never reached the true shortest path.
PARETO_LAMBDA_SWEEP = (0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0,
                       5.0, 8.0, 15.0)

# --------------------------------------------------------------------------
# Travel modes
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ModeConfig:
    name: str
    #: Overture ``class`` values allowed for this mode.
    allowed_classes: frozenset = field(default_factory=frozenset)
    #: Overture ``subclass`` values that are always excluded.
    excluded_subclasses: frozenset = field(default_factory=frozenset)
    #: Access modes checked against Overture ``access_restrictions``.
    access_modes: tuple = ()
    #: Per-class multiplier on length, expressing comfort/stress rather than
    #: distance. 1.0 = neutral.
    class_multiplier: dict = field(default_factory=dict)
    #: If True, one-way restrictions are enforced (bicycles) rather than
    #: ignored (pedestrians).
    respect_oneway: bool = False


#: Classes that are never usable on foot or by bike, in either mode.
NEVER_ROUTABLE_CLASSES = frozenset({
    "motorway",           # grade-separated freeways
    "motorway_link",
    # rail/water subtypes are filtered by subtype, these are belt-and-braces
    "light_rail", "subway", "tram", "standard_gauge", "funicular", "rail",
    # Upstream routes over ``unknown``.  In Hong Kong's extract the class
    # holds 2,633 segments of two kinds: OpenStreetMap ways tagged
    # ``highway=construction`` with nothing to say what is being built (the
    # planned roads of Kwu Tung North and Hung Shui Kiu; Overture sets
    # ``is_under_construction`` only when the future class is given), and
    # segments whose one source is TomTom, which carry no name, no flag and
    # no access rule to judge them by.  Together they were 174 km of the
    # routable network.  The one kind kept is the short connector: see
    # ``UNKNOWN_LINK_MAX_M``.
    "unknown",
})
#: A segment of class ``unknown`` no longer than this (m) that joins two
#: streets of the network is kept, as a service road.  Dropping the class
#: outright also took out some fifty stubs of ten metres or so between an
#: estate road and the street outside, most of them TomTom's, and the walk
#: from a bus terminus to the block beside it became a walk round the
#: estate (Loong Fung Terrace to The Legend, 370 m, became 2 km).  A road
#: under construction is hundreds of metres long and stays out.
UNKNOWN_LINK_MAX_M = 50.0

#: Subclasses excluded from the routable graph for *both* modes.
NEVER_ROUTABLE_SUBCLASSES = frozenset({"driveway", "parking_aisle"})

#: Mapped sidewalks and crossings.  Travel is modelled along street
#: centrelines, because including the whole sidewalk network would represent
#: every street two or three times, which distorts corridor aggregation and
#: street naming.  San Francisco could simply drop them.  In Hong Kong a
#: footbridge, a subway or a stairway very often lands on a mapped sidewalk
#: rather than on the street, so they are kept exactly where the centreline
#: network needs them to stay connected (``network.select_sidepaths``).
SIDEPATH_SUBCLASSES = frozenset({"sidewalk", "crosswalk"})

#: Main-road classes: a tunnel on one of these is a vehicular tunnel.
ROAD_TUNNEL_CLASSES = frozenset({"motorway", "trunk", "primary", "secondary"})
#: A run of tunnel on a main road -- its pieces taken together, however the
#: data cuts them -- is treated as a vehicular tunnel, closed to walkers and
#: cyclists, when it is longer than this (m) in all.  The extract has 102
#: such runs.  The 21 longer than a kilometre are all named tunnels (Kai Tak
#: Tunnel, at 1.2 km, is the shortest), and the data itself closes every one
#: of them but a tube of Tate's Cairn Tunnel and 51 m of a Kai Tak slip
#: road.  In the shorter runs it leaves no more than 261 m open at a time.
#: An earlier limit of 80 m on each piece closed ordinary streets that pass
#: under a building or a flyover: Cheung Tung Road at Sunny Bay, the only
#: road along north Lantau a bicycle may use, and Tuen Fat Road and Tuen Hi
#: Road in Tuen Mun town centre.
ROAD_TUNNEL_MAX_M = 1000.0
#: A shorter tunnel is closed to a mode where the data itself bars the mode
#: from any piece of it: a tunnel is one thing, and a mapper who tags one of
#: its ways may miss the next.  "Any piece of it" takes in the other tube.
#: The two tubes of a tunnel share no node, only a name, and lie side by
#: side, 7 to 53 m apart in the extract (Lion Rock Tunnel's are the farthest
#: apart); two runs of tunnel that carry one name within this distance (m)
#: of each other are one tunnel.  That closes 0.7 km the data leaves open:
#: 106 m of slip road in a tube of Tseung Lam Highway that is otherwise
#: closed, and the second tube of Tseung Lam Highway (261 m), of the Ma On
#: Shan Bypass (163 m) and of Smithfield (161 m).  No other open run has a
#: closed one of its name within 500 m.  What stays open to walkers under a
#: roof is 1.3 km in 23 pieces of 101 m at most, in tunnels the data bars
#: nobody from at any point: the covered streets above, and with them 101 m
#: of Ching Cheung Road and a 70 m stub of the Central Kowloon Bypass.
ROAD_TUNNEL_TUBE_GAP_M = 60.0

#: Classes whose viaducts a walker may use only where the data permits
#: walking by name.  OpenStreetMap bars walkers from most of Hong Kong's
#: trunk roads but leaves the rest untagged, the viaducts of interchanges
#: among them, and an open viaduct is level and direct: the walk from Tsing
#: Yi to Mong Kok ran over the Kwai Chung Interchange.  A trunk road at
#: grade with no rule against walking (Gloucester Road, Lung Cheung Road)
#: stays open, and so do its slip roads at grade: closing those as well
#: left the carriageway between two interchanges a dead end, and a pin
#: that landed on it (Diamond Hill Station, on Lung Cheung Road) was
#: walked 400 m along the road to the next junction and back.
VIADUCT_PERMIT_REQUIRED_CLASSES = frozenset({"trunk"})
#: A run of bridge on such a road is a viaduct when it is longer than this
#: (m).  Shorter ones are a culvert or a subway's roof under a road that is
#: at grade on both sides (23 m of Chatham Road North, 86 m of Nga Cheung
#: Road).
VIADUCT_MIN_M = 100.0

#: Land nobody walks or rides through, which OpenStreetMap maps as plain
#: service roads because the gate is a point and the fence a line: the
#: container terminals and cargo working areas, the power stations, the
#: dockyard and the depots, and military land.  A service road is closed to
#: both modes where it lies inside an Overture land-use polygon of subtype
#: ``military`` (other than the classes exempted below), or of class
#: ``industrial`` whose name holds one of these phrases.  The land-use
#: theme has no class for a port or a power station, so the names are all
#: there is to go on.
CLOSED_LAND_NAMES = ("container terminal", "river trade terminal",
                     "power station", "cargo working area",
                     "cargo handling area", "cargo terminal", "superterminal",
                     "dockyard", "depot", "gas production")
#: Military land that is open country: the Castle Peak firing range is 22
#: square kilometres of hillside crossed by hikers' paths and closed only on
#: firing days.
CLOSED_LAND_EXEMPT_CLASSES = frozenset({"range"})
#: Street classes the closure applies to.  Paths and footways are left as
#: mapped.
CLOSED_LAND_STREET_CLASSES = frozenset({"service"})

#: No edge is left longer than this (m): the browser payload stores lengths
#: as 16-bit counts of 5 cm (3.2 km at most).
MAX_EDGE_LENGTH_M = 2000.0
#: A piece no longer than this (m) is contracted: its two ends become one
#: node.  Such pieces are real links between two connectors of the data, a
#: few decimetres apart, and deleting them (as upstream does) cuts the
#: network there.
MIN_EDGE_LENGTH_M = 0.5

#: How far (m) from a stretch of street its other-language name may be
#: taken from another stretch that carries both (``network.harmonise_names``).
NAME_PAIR_RADIUS_M = 1000.0

WALK = ModeConfig(
    name="walk",
    allowed_classes=frozenset({
        "trunk", "primary", "secondary", "tertiary", "residential",
        "living_street", "unclassified", "service", "pedestrian",
        "footway", "path", "steps", "track", "cycleway", "bridleway",
    }),
    excluded_subclasses=NEVER_ROUTABLE_SUBCLASSES,
    access_modes=("foot",),
    class_multiplier={
        # Stairs are walkable but slow and unpleasant with any load.
        "steps": 1.6,
        # Big roads are unpleasant but passable on foot.
        "trunk": 1.15,
        "primary": 1.05,
    },
    respect_oneway=False,   # pedestrians are not bound by one-way streets
)

BIKE = ModeConfig(
    name="bike",
    allowed_classes=frozenset({
        "trunk", "primary", "secondary", "tertiary", "residential",
        "living_street", "unclassified", "service", "pedestrian",
        "footway", "path", "cycleway", "track",
    }),
    # Steps are excluded outright for bicycles -- a route suitable for a
    # pedestrian is emphatically not necessarily rideable.
    excluded_subclasses=NEVER_ROUTABLE_SUBCLASSES,
    access_modes=("bicycle",),
    class_multiplier={
        "cycleway": 0.85,       # protected/dedicated bike infrastructure
        "living_street": 0.9,
        "residential": 0.95,
        "trunk": 1.9,           # e.g. Gloucester Road, Lung Cheung Road -- high stress
        "primary": 1.45,
        "secondary": 1.2,
        "footway": 1.5,         # rideable only where bicycles are permitted
        "path": 1.1,
        "pedestrian": 1.4,
    },
    respect_oneway=True,
)

MODES: dict[str, ModeConfig] = {"walk": WALK, "bike": BIKE}

#: Hong Kong is not one landmass on foot or by bike: nothing crosses
#: Victoria Harbour but tunnels closed to both, Lantau hangs off a bridge
#: with no footway, and Lamma, Cheung Chau and Peng Chau are reached by
#: ferry.  Routing therefore keeps *every* strongly connected component of
#: at least this many nodes, not only the largest one.  Components joined
#: by one-way paths (Wan Chai North, which a bicycle can ride into and not
#: out of) make up one landmass together with what lies on those paths, and
#: a route is looked for only between two points of the same landmass.  The
#: floor drops mapping fragments (a podium walkway nobody joined to the
#: street, a fenced compound) while keeping the smallest inhabited islands.
MIN_COMPONENT_NODES = 40
#: Component ranks are packed into one byte for the browser (0 = none).
MAX_COMPONENTS = 255

#: Classes bicycles may never use even when the class list would allow it.
BIKE_FORBIDDEN_CLASSES = frozenset({"steps", "bridleway"})

#: Classes a bicycle may use only where the data permits bicycles by name.
#: Cycling on a footpath is an offence in Hong Kong (Summary Offences
#: Ordinance, Cap. 228 s.4(8)) and in the country parks outside the
#: designated mountain-bike trails, while OpenStreetMap leaves ``bicycle``
#: unset on almost every footway (1,377 of the 52,824 footway segments that
#: are not sidewalks or crossings carry a rule that names it), so "no rule"
#: has to mean "no" here, where upstream's default is "yes".
#: Shared paths and mountain-bike trails are tagged and stay open; cycle
#: tracks are their own class.
BIKE_PERMIT_REQUIRED_CLASSES = frozenset({"footway", "pedestrian", "path"})

#: Dismount-and-push links.  Hong Kong's cycle tracks are joined to the roads
#: and to one another by a few metres of plain footway (a kerb link, a
#: crossing), which the rule above closes, leaving 76 of the 269 km of cycle
#: track as islands no route can reach.  A cyclist gets off and pushes
#: there, so a walkable run of those classes no longer than this (m) that
#: joins two points of the cycling network is opened to bicycles as a
#: pushed link (edge column ``bike_push``) ...
BIKE_PUSH_MAX_M = 50.0
#: ... and charged this many times its length wherever the class comfort
#: multipliers apply: walking pace beside riding pace.
BIKE_PUSH_MULTIPLIER = 3.0
#: A pushed link steeper than this from end to end is not opened after
#: all: a few metres of "footway" that rise at a quarter or a half are a
#: flight of steps the map did not class as one (25 m off Dyer Avenue in
#: Hung Hom climbs 12 m), and nobody pushes a bicycle up those.
BIKE_PUSH_MAX_GRADE = 0.20

# --------------------------------------------------------------------------
# Analysis parameters
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class AnalysisConfig:
    #: Number of representative access points sampled per neighborhood.
    #: One primary point is used for the pair matrix; the rest support
    #: robustness checks.
    points_per_neighborhood: int = 1
    #: Corridor detection: keep edges whose importance score is at or above
    #: this quantile of the positive-score distribution.
    corridor_score_quantile: float = 0.93
    #: Minimum corridor length (m) after merging contiguous segments.
    corridor_min_length_m: float = 350.0
    #: Maximum average grade for an edge to be eligible as corridor material.
    corridor_max_avg_grade: float = 0.055
    #: Corridors are grown only through edges below this maximum grade.
    corridor_max_edge_grade: float = 0.09
    #: Pass detection: elevation band (m) used to group saddle candidates.
    pass_cluster_radius_m: float = 400.0
    #: Number of Pareto sample pairs reported in detail.
    pareto_detail_pairs: int = 14
    #: Which qualifying intersection to use as a neighborhood's access point:
    #: 0 is the one nearest the street-weighted centre, 1 the next nearest,
    #: and so on. Non-zero values exist for robustness checks only.
    point_rank: int = 0


ANALYSIS = _apply(AnalysisConfig(), _OV)
_refuse_unknown(_OV, ElevationConfig, AnalysisConfig)

# --------------------------------------------------------------------------
# Representative neighborhood pairs highlighted in the written analysis
# --------------------------------------------------------------------------
#: Names are the planning areas' own, exactly as ``load_neighborhoods``
#: stores them.  Both ends of a pair are on one landmass, because nothing
#: else can be routed: eight pairs on Hong Kong Island, six in Kowloon, five
#: that reach into the New Territories and one on Lantau.  All twenty were
#: confirmed connected on the walking network when the neighborhood set was
#: chosen.  A pair whose ends have no network in a partial build (an
#: urban-core test window has no Sha Tin) is simply left out of the report.
FEATURED_PAIRS = (
    # Hong Kong Island
    ("Central District", "Mid-Levels West"),
    ("Sai Ying Pun & Sheung Wan", "The Peak Area"),
    ("Kennedy Town & Mount Davis", "Causeway Bay"),
    ("Wan Chai", "Aberdeen & Ap Lei Chau"),
    ("Causeway Bay", "Stanley"),
    ("North Point", "Chai Wan"),
    ("Pok Fu Lam", "Central District"),
    ("Wong Nai Chung", "Jardine's Lookout & Wong Nai Chung Gap"),
    # Kowloon
    ("Tsim Sha Tsui", "Kowloon Tong"),
    ("Mong Kok", "Kwun Tong (South)"),
    ("Cheung Sha Wan", "Tsz Wan Shan, Diamond Hill & San Po Kong"),
    ("Hung Hom", "Ho Man Tin"),
    ("Kai Tak", "Kwun Tong (North)"),
    ("Shek Kip Mei", "Ngau Tau Kok & Kowloon Bay"),
    # Kowloon to the New Territories, and within them
    ("Kowloon Tong", "Sha Tin"),
    ("Kwai Chung", "Sha Tin"),
    ("Tsuen Wan", "Tsing Yi"),
    ("Tseung Kwan O", "Kwun Tong (South)"),
    ("Sha Tin", "Tai Po"),
    # Lantau
    ("Tung Chung Town Centre Area", "Discovery Bay"),
)

#: What the written analysis calls each landmass.  A landmass is one
#: connected piece of the walking network (see ``MIN_COMPONENT_NODES``) and
#: carries no name of its own, so it is recognised by the neighborhoods found
#: on it: the name goes to the landmass holding the most of the listed
#: ones.  The three groups are those measured on the street data -- Hong
#: Kong Island with Ap Lei Chau; Kowloon, the New Territories and Tsing Yi;
#: Lantau with the airport island.  A landmass holding none of these (an
#: outlying island, or a fragment cut off by the edge of a test window) is
#: named after its own largest neighborhood instead.
LANDMASS_NAMES = (
    ("Hong Kong Island",
     ("Central District", "Wan Chai", "North Point", "Aberdeen & Ap Lei Chau")),
    ("Kowloon and the New Territories",
     ("Tsim Sha Tsui", "Mong Kok", "Kwun Tong (South)", "Sha Tin", "Tsuen Wan",
      "Yuen Long")),
    ("Lantau",
     ("Tung Chung Town Centre Area", "Mui Wo Fringe", "South Lantau Coast")),
)

#: Where the route page opens before anyone types.  Each end is (label,
#: place names to try in the search index, fallback), the fallback being a
#: (lon, lat) pair.  Statue Square in Central to the Main Building of the
#: University of Hong Kong, chosen by scoring pairs of well-known places on
#: the frontier search: the shortest walk goes straight up the stepped
#: streets of Sheung Wan and Sai Ying Pun; the flattest keeps to the old
#: waterfront as far as Western Street and saves about half the climbing
#: for a fifth more distance, and the frontier between them holds some
#: thirty distinct routes.
DEFAULT_TRIP = (
    ("Statue Square, Central", (), (114.15995, 22.28105)),
    ("The University of Hong Kong", (), (114.13790, 22.28390)),
)


def profile(name: str) -> CostWeights:
    """Look up a routing profile by name."""
    try:
        return ROUTING_PROFILES[name]
    except KeyError:
        raise KeyError(f"unknown routing profile {name!r}; "
                       f"choose from {sorted(ROUTING_PROFILES)}") from None


def with_alpha(weights: CostWeights, alpha: float) -> CostWeights:
    """Return a copy of ``weights`` with a different climbing weight."""
    return replace(weights, alpha=alpha, name=f"{weights.name}_a{alpha:g}")


def with_scale(weights: CostWeights, lam: float) -> CostWeights:
    """Scale every climbing/grade term of ``weights`` by ``lam``.

    ``lam == 0`` is pure distance, with the comfort multipliers switched off
    so that it coincides exactly with the ``shortest`` objective.
    """
    return replace(weights, alpha=weights.alpha * lam, beta=weights.beta * lam,
                   gamma=weights.gamma * lam,
                   use_class_multiplier=(lam > 0) and weights.use_class_multiplier,
                   name=f"{weights.name}_x{lam:g}")
