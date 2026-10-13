"""Build a routable street graph from Overture transportation segments.

Topology comes from Overture ``connectors``: every segment lists the
connector IDs it touches together with the fractional position ``at`` along
its own geometry.  Splitting each segment at those positions and keying nodes
by connector ID yields exact topology with no geometric snapping tolerance,
and grade-separated crossings correctly stay unconnected.  ``at`` is a
fraction of the segment's length on the ground, so the cutting is done in
the projected CRS: a fraction of the length in degrees puts a connector
part-way along a winding segment tens of metres from where it is.

Access is derived from Overture ``access_restrictions``:

* ``denied`` + ``heading=backward`` -- one-way streets (``heading=forward``
  for one that runs against the way the segment is drawn).  Enforced for
  bicycles, ignored for pedestrians, since OSM ``oneway`` describes vehicle
  movement.
* ``denied``/``allowed``/``designated`` + ``mode=[foot|bicycle|...]`` --
  explicit per-mode permissions.
* ``recognized=[as_private]`` or ``using=[as_customer|at_destination]`` --
  private or destination-only access; excluded from through routing.

A rule that names a mode outranks a rule that applies to every mode,
whatever order they appear in.

Several things are done differently from the San Francisco original, all
because of what Hong Kong's data looks like:

* **Flags, rules and names that cover part of a segment are applied to that
  part.**  Overture joins consecutive OpenStreetMap ways into one segment
  and keeps ``is_bridge``, ``is_tunnel``, access rules and names as ranges
  along it (``between``).  Half of Hong Kong's bridge flags and most of its
  tunnel flags are such ranges.  Segments are therefore cut at the range
  ends as well as at connectors, so a flyover is a structure only where it
  is one (otherwise the whole hillside road it belongs to is flattened into
  a ramp), a tunnel closed to walkers is closed without closing the street
  either side of it, and no piece carries the name of the street it runs
  into.
* **Road tunnels are closed to walkers and cyclists** unless the data says
  otherwise by name: every vehicular tunnel in Hong Kong is, by law, and a
  tunnel left open would be the flattest route through any hill.
* **So are a few places the data leaves open and nobody can go**: to
  walkers, the viaducts of trunk roads; to both, the service roads inside
  container terminals, cargo yards, power stations, depots and military
  land (``closed_land``).
* **Sidewalks and crossings are used where the centrelines need them.**
  Travel is still modelled on street centrelines, but in Hong Kong a
  footbridge, a subway or a hillside stair very often lands on a mapped
  sidewalk rather than on the street, and dropping every sidewalk leaves it
  hanging.  A sidewalk or crossing is kept where it joins two points of the
  centreline network that are otherwise unconnected or much farther apart
  (``select_sidepaths``).
* **Bicycles are pushed across short footway links.**  Footways are closed
  to cycling unless signed for it, and the cycle tracks are joined to the
  roads and to one another by a few metres of footway; those links are
  opened as dismount-and-push links (``bike_push``).
* **Fragments are dropped** (see ``config.MIN_COMPONENT_NODES``), and no
  edge is left longer than the browser payload can carry.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (BIKE_FORBIDDEN_CLASSES, BIKE_PERMIT_REQUIRED_CLASSES,
                     BIKE_PUSH_MAX_M, BIKE_PUSH_MULTIPLIER,
                     CLOSED_LAND_EXEMPT_CLASSES, CLOSED_LAND_NAMES,
                     CLOSED_LAND_STREET_CLASSES, CRS_GEOGRAPHIC, CRS_PROJECTED,
                     MAX_EDGE_LENGTH_M,
                     MIN_COMPONENT_NODES, MIN_EDGE_LENGTH_M, MODES,
                     NAME_PAIR_RADIUS_M, NEVER_ROUTABLE_CLASSES,
                     NEVER_ROUTABLE_SUBCLASSES, PROCESSED_DIR,
                     ROAD_TUNNEL_CLASSES, ROAD_TUNNEL_MAX_M,
                     ROAD_TUNNEL_TUBE_GAP_M, SIDEPATH_SUBCLASSES,
                     UNKNOWN_LINK_MAX_M, VIADUCT_MIN_M,
                     VIADUCT_PERMIT_REQUIRED_CLASSES)
from .utils import get_logger, progress, step

log = get_logger("hk_flat_routes.network")

EDGES_PARQUET = PROCESSED_DIR / "edges_raw.parquet"

#: access_type values that permit travel.
_PERMIT = {"allowed", "designated"}
#: conditional qualifiers that mean "not available for through travel".
#: ``as_permitted`` is deliberately not one of them, as it is upstream: it is
#: OpenStreetMap's ``access=permissive``, a way the owner lets the public
#: use, and in Hong Kong that is how the paths through estates and campuses
#: and stretches of old roads such as Wan Chai Gap Road are tagged.
_PRIVATE_RECOGNIZED = {"as_private", "as_employee", "as_student"}
_DESTINATION_USING = {"at_destination", "as_customer", "as_delivery", "to_deliver"}
#: mode tokens that subsume the pedestrian / bicycle modes. A bicycle is a
#: vehicle (OpenStreetMap's ``vehicle=no`` closes a street to cycles too).
_MODE_PARENTS = {
    "foot": {"foot", "pedestrian"},
    "bicycle": {"bicycle", "vehicle"},
}


def _as_list(value) -> list:
    """Coerce an Arrow-derived value (ndarray / list / None / NaN) to a list.

    Arrow list columns arrive as numpy object arrays after ``to_pandas``, and
    ``bool(ndarray)`` raises, so every nested column is funnelled through here.
    """
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, float) and np.isnan(value):
        return []
    return [value]


def _as_dict(value) -> dict:
    """Coerce an Arrow struct value to a dict."""
    if isinstance(value, dict):
        return value
    return {}


# --------------------------------------------------------------------------
# access parsing
# --------------------------------------------------------------------------
def _rule_modes(rule: dict) -> set[str] | None:
    modes = _as_list(_as_dict(rule.get("when")).get("mode"))
    return set(modes) if modes else None


def _rule_applies(rule: dict, mode: str) -> bool:
    """Does an access rule constrain ``mode``?"""
    modes = _rule_modes(rule)
    if modes is None:
        return True                      # applies to all modes
    return bool(modes & _MODE_PARENTS[mode])


def _covers(between, at: float | None) -> bool | None:
    """Does a ``between`` range cover position ``at`` of the segment?

    ``None`` when the rule covers part of the segment and no position was
    given, i.e. when the question cannot be answered.
    """
    rng = _as_list(between)
    if len(rng) < 2:
        return True
    if at is None:
        return None
    return float(rng[0]) - 1e-9 <= at <= float(rng[1]) + 1e-9


def evaluate_access(restrictions, mode: str, default: bool = True,
                    at: float | None = None) -> dict:
    """Resolve Overture access rules for one travel mode.

    Returns a dict with ``allowed`` (through travel permitted at all),
    ``oneway_forward_only`` (backward travel prohibited),
    ``oneway_backward_only`` (forward travel prohibited: a one-way street
    drawn against its direction of travel, OpenStreetMap's ``oneway=-1``),
    ``restricted`` (access is conditional/private) and ``explicit`` (a rule
    naming this mode permitted it).

    Rules that apply to every mode are resolved first and rules that name
    ``mode`` after them, each group in document order, so a mode-specific
    rule always has the last word over a general one.  Rules carrying
    ``between`` apply to only part of the segment.  Given ``at``, the
    position along the segment being asked about, such a rule applies when
    its range covers that position and not otherwise; without ``at`` it is
    recorded but does not veto the whole segment, because vetoing would
    delete usable street from the network.
    """
    allowed = default
    oneway_forward_only = False
    oneway_backward_only = False
    restricted = False
    partial = False
    explicit = False

    rules = [_as_dict(r) for r in _as_list(restrictions)]
    rules.sort(key=lambda r: 0 if _rule_modes(r) is None else 1)   # stable
    for rule in rules:
        if not _rule_applies(rule, mode):
            continue
        when = _as_dict(rule.get("when"))
        if when.get("during"):
            # time-conditional (e.g. peak-hour bans); ignored, noted only
            continue
        if len(_as_list(when.get("vehicle"))) > 0:
            # a height, width, weight or length limit: it arrives as a
            # denial with no mode, and bars nobody on foot or on a bicycle
            # (818 Hong Kong segments, 111 km, would otherwise be closed
            # to walkers by a headroom sign)
            continue
        covered = _covers(rule.get("between"), at)
        if covered is False:
            continue                      # a rule for another stretch
        is_partial = covered is None
        atype = rule.get("access_type")
        heading = when.get("heading")

        if heading:
            # directional restriction == one-way; a permit for the closed
            # direction lifts it again (e.g. a contraflow bike lane)
            if atype == "denied" or atype in _PERMIT:
                if heading == "backward":
                    oneway_forward_only = atype == "denied"
                elif heading == "forward":
                    oneway_backward_only = atype == "denied"
            continue

        recognized = set(_as_list(when.get("recognized")))
        using = set(_as_list(when.get("using")))
        if recognized & _PRIVATE_RECOGNIZED or using & _DESTINATION_USING:
            # conditional access -- treat as unavailable for through routing
            if atype in _PERMIT:
                restricted = True
                if not is_partial:
                    allowed = False
            continue

        if is_partial:
            partial = True
            continue
        allowed = atype in _PERMIT
        if _rule_modes(rule) is not None:
            explicit = allowed
        if allowed and _rule_modes(rule) is not None:
            # an unconditional permit naming this mode also lifts a general
            # private/destination restriction read earlier
            restricted = False

    return {"allowed": allowed, "oneway_forward_only": oneway_forward_only,
            "oneway_backward_only": oneway_backward_only,
            "restricted": restricted, "partial_rules": partial,
            "explicit": explicit}


def _flags(road_flags) -> set[str]:
    out: set[str] = set()
    for entry in _as_list(road_flags):
        for v in _as_list(_as_dict(entry).get("values")):
            out.add(v)
    return out


def _flag_ranges(road_flags) -> dict[str, list[tuple[float, float]]]:
    """Each road flag with the stretches of the segment it applies to."""
    out: dict[str, list[tuple[float, float]]] = {}
    for entry in _as_list(road_flags):
        e = _as_dict(entry)
        rng = _as_list(e.get("between"))
        span = (float(rng[0]), float(rng[1])) if len(rng) >= 2 else (0.0, 1.0)
        for v in _as_list(e.get("values")):
            out.setdefault(v, []).append(span)
    return out


def _flags_at(ranges: dict, at: float) -> set[str]:
    return {f for f, spans in ranges.items()
            if any(lo - 1e-9 <= at <= hi + 1e-9 for lo, hi in spans)}


#: Road flags whose range ends become cuts, so each piece carries them whole.
_CUT_FLAGS = ("is_bridge", "is_tunnel", "is_link", "is_abandoned",
              "is_under_construction", "is_proposed")
_DEAD_FLAGS = frozenset({"is_abandoned", "is_under_construction", "is_proposed"})

_CJK = r"\u2e80-\u9fff\uf900-\ufaff\uff00-\uffef"
_BILINGUAL_RE = re.compile(rf"^(.*[{_CJK}][^A-Za-z]*?)\s+([A-Za-z].*)$")
_HAS_CJK_RE = re.compile(rf"[{_CJK}]")
#: Languages of ``names.common`` tried for the Chinese name, in order.
_ZH_KEYS = ("zh-Hant-HK", "zh-Hant", "zh", "yue-Hant", "yue", "zh-Hans")


def split_name(names, at: float | None = None) -> tuple[str | None, str | None]:
    """(English, Chinese) names of an Overture feature.

    Hong Kong's streets carry both.  OpenStreetMap writes them together in
    ``name`` ("彌敦道 Nathan Road") and usually also apart in ``name:en`` and
    ``name:zh``, which Overture hands on as ``names.common``.  The tagged
    languages are preferred; a bilingual ``primary`` is split at the first
    word in Latin script when they are missing.

    Overture also joins consecutive ways into one segment, and a segment
    can change its name part-way: one runs as Nathan Road for the first
    half and Austin Road for the rest.  Such names arrive in ``names.rules``
    with the stretch they apply to (``between``), and ``primary`` is then
    right for only part of the segment.  Given ``at``, the position along
    the segment, the names are those of that position:

    * a stretch with a name of its own (a ranged rule in no particular
      language) takes it, and its English and Chinese from the rules for the
      same stretch;
    * where the street's name runs the whole segment, a ranged rule in one
      language speaks for that language only (a ``zh-Hans`` name tagged on
      half of Tuen Mun Road does not take the road's English and
      Traditional Chinese names away from that half);
    * a stretch no rule names is unnamed: it is an unnamed way (a footway,
      a cycle track) that Overture joined to a named one, and ``primary``
      belongs to the other.
    """
    n = _as_dict(names)
    primary = n.get("primary")
    common = n.get("common")
    if isinstance(common, dict):
        pass
    elif common is None:
        common = {}
    else:                       # an Arrow map arrives as a list of pairs
        common = dict(_as_list(common))
    langs = common

    if at is not None:
        whole: dict = {}
        local: dict = {}
        ranged = False
        for rule in _as_list(n.get("rules")):
            r = _as_dict(rule)
            if r.get("variant") != "common" or not r.get("value"):
                continue
            rng = _as_list(r.get("between"))
            if len(rng) < 2:
                whole.setdefault(r.get("language"), r["value"])
                continue
            ranged = True
            if float(rng[0]) - 1e-9 <= at <= float(rng[1]) + 1e-9:
                local.setdefault(r.get("language"), r["value"])
        if ranged and None in local:
            # this stretch has a name of its own: the segment-wide values
            # describe some other part of it
            primary, langs = local[None], local
        elif ranged and None in whole:
            # the street's name runs the whole way; a rule for part of it
            # adds or replaces only the language it is written in
            primary, langs = whole[None], {**whole, **common, **local}
        elif ranged:
            # no rule names this stretch outright: it has what the rules
            # for it say in one language or another, which is usually nothing
            primary, langs = None, local
            if local and not (local.get("en")
                              or any(local.get(k) for k in _ZH_KEYS)):
                primary = next(iter(local.values()))

    en = langs.get("en")
    zh = next((langs[k] for k in _ZH_KEYS if langs.get(k)), None)
    if isinstance(primary, str) and primary.strip():
        primary = primary.strip()
        m = _BILINGUAL_RE.match(primary)
        if m:
            zh = zh or m.group(1).strip()
            en = en or m.group(2).strip()
        elif _HAS_CJK_RE.search(primary):
            zh = zh or primary
        else:
            en = en or primary
    return (en or None), (zh or None)


# --------------------------------------------------------------------------
# segment loading and splitting
# --------------------------------------------------------------------------
def load_segments(path: Path) -> pd.DataFrame:
    """Load the cached Overture segment extract into a DataFrame."""
    import pyarrow.parquet as pq
    table = pq.read_table(path)
    log.info("loaded %d raw Overture segments", table.num_rows)
    return table.to_pandas()


def _short_links(df: pd.DataFrame) -> np.ndarray:
    """Segments of class ``unknown`` that are a short link between streets.

    Such segments are taken in runs: pieces that follow one another through
    connectors no kept segment uses.  True for every piece of a run that is
    no longer than ``UNKNOWN_LINK_MAX_M`` in all and touches the kept
    network at two connectors or more (see ``config.UNKNOWN_LINK_MAX_M``).
    Lengths are taken in metres on the projected line.
    """
    import geopandas as gpd
    import scipy.sparse as sp
    import shapely
    from scipy.sparse.csgraph import connected_components

    out = np.zeros(len(df), dtype=bool)
    unknown = (df["cls"] == "unknown").to_numpy()
    if not unknown.any() or "connectors" not in df.columns or "geometry" not in df.columns:
        return out

    kept = (~df["cls"].isin(NEVER_ROUTABLE_CLASSES)
            & ~df["subclass"].fillna("").isin(NEVER_ROUTABLE_SUBCLASSES)).to_numpy()
    on_street = {_as_dict(c).get("connector_id")
                 for conns in df["connectors"].to_numpy()[kept] for c in _as_list(conns)}
    idx = np.flatnonzero(unknown)
    lines = gpd.GeoSeries(shapely.from_wkb(df["geometry"].to_numpy()[idx]),
                          crs=CRS_GEOGRAPHIC).to_crs(CRS_PROJECTED)
    metres = lines.length.to_numpy()
    ends = [[_as_dict(c).get("connector_id") for c in _as_list(conns)]
            for conns in df["connectors"].to_numpy()[idx]]

    # join the pieces that share a connector off the kept network
    first: dict = {}
    a, b = [], []
    for k, ids in enumerate(ends):
        for c in ids:
            if c is None or c in on_street:
                continue
            if c in first:
                a.append(first[c]); b.append(k)
            else:
                first[c] = k
    n = len(idx)
    _, run = connected_components(
        sp.coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n)), directed=False)
    total = np.bincount(run, weights=metres)
    touches: dict = {}
    for k, ids in enumerate(ends):
        touches.setdefault(run[k], set()).update(c for c in ids if c in on_street)
    for k in range(n):
        out[idx[k]] = total[run[k]] <= UNKNOWN_LINK_MAX_M and len(touches[run[k]]) >= 2
    return out


def _base_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Drop everything that is never part of a walk/bike street network."""
    n0 = len(df)
    df = df[df["subtype"] == "road"].copy()
    log.info("  subtype=road: %d -> %d", n0, len(df))

    df["cls"] = df["class"].fillna("unknown")
    link = _short_links(df)
    df.loc[link, "cls"] = "service"
    df = df[~df["cls"].isin(NEVER_ROUTABLE_CLASSES)]
    df = df[~df["subclass"].fillna("").isin(NEVER_ROUTABLE_SUBCLASSES)]
    log.info("  after class/subclass filter: %d (%d short links of class "
             "unknown kept as service roads)", len(df), int(link.sum()))
    # Abandoned, proposed and unfinished stretches are dropped piece by
    # piece once the segment has been cut (the flags are ranges).
    # Freeway ramps are tagged is_link on a motorway-ish class; the class
    # filter already removed motorway_link.  The links that hang off trunk
    # roads stay in the table: a bicycle may use them, and a walker where
    # the data permits it (see ``resolve_access``).
    return df


def _name_cuts(names) -> list[float]:
    """Positions along a segment where its name changes.

    The ends of the ranged name rules, kept only where ``split_name`` gives
    a different answer on the two sides: most ranged rules repeat one name
    in another language, or stop where a rule for the same name begins.
    """
    stops = {0.0, 1.0}
    for rule in _as_list(_as_dict(names).get("rules")):
        r = _as_dict(rule)
        rng = _as_list(r.get("between"))
        if r.get("variant") == "common" and len(rng) >= 2:
            stops.update(float(a) for a in rng[:2] if 0.0 < float(a) < 1.0)
    if len(stops) == 2:
        return []
    stops = sorted(stops)
    named = [split_name(names, at=0.5 * (a + b))
             for a, b in zip(stops[:-1], stops[1:])]
    return [stops[k + 1] for k in range(len(named) - 1)
            if named[k] != named[k + 1]]


def _range_cuts(row, length_m: float) -> list[float]:
    """Positions where a flag, an access rule or a name starts or stops
    applying."""
    out: set[float] = set()
    ranges = _flag_ranges(row["road_flags"])
    for flag in _CUT_FLAGS:
        for lo, hi in ranges.get(flag, ()):
            out.update((lo, hi))
    for rule in _as_list(row["access_restrictions"]):
        rng = _as_list(_as_dict(rule).get("between"))
        if len(rng) >= 2:
            out.update((float(rng[0]), float(rng[1])))
    out.update(_name_cuts(row["names"]))
    # a cut within a metre of either end only makes a sliver
    eps = 1.0 / max(length_m, 1.0)
    return sorted(a for a in out if eps < a < 1.0 - eps)


def _split_geometries(df: pd.DataFrame):
    """Split each segment at its connectors, returning an edge-level table.

    Segments are also cut wherever a structure flag, an access rule or a
    name starts or stops applying part-way along, so that every piece is
    wholly a bridge, a tunnel, open or closed, and has one name.  Those
    cuts make nodes of their own (``cut:<segment>:<position>``), shared by
    the two pieces either side.

    Positions along a segment are fractions of its length in metres, so the
    lines are projected first and the pieces come back in ``CRS_PROJECTED``.
    Cutting the longitude/latitude line instead, as upstream does, measures
    whatever part of the segment runs east-west 7.5% short (a degree of
    longitude is that much shorter than one of latitude at 22 degrees
    north), so a connector part-way along a winding hill road lands in the
    wrong place, and in a different wrong place on each segment through it:
    the ends of the pieces meeting at one node were more than a metre apart
    at 24,717 of 171,286 nodes, and up to 42 m.  Cut in metres, they agree
    to a centimetre.
    """
    import geopandas as gpd
    import shapely
    from shapely.ops import substring

    geoms = gpd.GeoSeries(shapely.from_wkb(df["geometry"].values),
                          crs=CRS_GEOGRAPHIC).to_crs(CRS_PROJECTED).values
    rows: list[dict] = []
    n_synthetic = n_cut = n_dead = 0

    for i, (idx, row) in enumerate(progress(df.iterrows(), desc="  splitting segments",
                                            total=len(df), unit="seg")):
        line = geoms[i]
        if line is None or line.is_empty or line.length == 0:
            continue
        conns = [_as_dict(c) for c in _as_list(row["connectors"])]
        cut_at = {float(np.clip(c["at"], 0.0, 1.0)): c["connector_id"]
                  for c in conns if c.get("connector_id")}
        eps = 1.0 / max(line.length, 1.0)
        for a in _range_cuts(row, line.length):
            if all(abs(a - b) > eps for b in cut_at):
                cut_at[a] = f"cut:{row['id']}:{a:.6f}"
                n_cut += 1
        cuts = sorted(cut_at.items())
        # ensure both ends are nodes
        if not cuts or cuts[0][0] > 1e-9:
            cuts.insert(0, (0.0, f"synth:{row['id']}:start"))
            n_synthetic += 1
        if cuts[-1][0] < 1.0 - 1e-9:
            cuts.append((1.0, f"synth:{row['id']}:end"))
            n_synthetic += 1
        ranges = _flag_ranges(row["road_flags"])
        names = row["names"]
        has_ranges = any(len(_as_list(_as_dict(r).get("between"))) >= 2
                         for r in _as_list(_as_dict(names).get("rules")))
        name_en, name_zh = split_name(names)
        for k in range(len(cuts) - 1):
            a_at, a_id = cuts[k]
            b_at, b_id = cuts[k + 1]
            if b_at - a_at <= 1e-9 or a_id == b_id:
                continue
            mid = 0.5 * (a_at + b_at)
            flags = _flags_at(ranges, mid)
            if flags & _DEAD_FLAGS:
                n_dead += 1
                continue
            piece = substring(line, a_at, b_at, normalized=True)
            if piece.is_empty or piece.length == 0:
                continue
            if has_ranges:              # the name may change along the way
                name_en, name_zh = split_name(names, at=mid)
            rows.append({
                "segment_id": row["id"],
                "u": a_id, "v": b_id,
                "part": k,
                "start_at": a_at, "end_at": b_at, "mid_at": mid,
                "cls": row["cls"],
                "subclass": row["subclass"],
                # the name shown is the English one where there is one
                "name": name_en or name_zh,
                "name_zh": name_zh if name_en else None,
                "flags": flags,
                "access_restrictions": row["access_restrictions"],
                "geometry": piece,
            })
    log.info("  split into %d edges (%d synthetic end nodes, %d cuts at flag, "
             "rule and name ranges, %d abandoned or unbuilt pieces dropped)",
             len(rows), n_synthetic, n_cut, n_dead)
    return rows


def _join_ends(geoms, u, v):
    """Move the ends of the pieces that meet at each node onto one point.

    Every piece ending at a node is cut at the node's own position, so
    their ends agree to a few millimetres already; this makes them agree
    exactly, and brings together the ends that a contracted piece
    (``_contract_short_pieces``) used to hold a few decimetres apart.  The
    point is the mean of the ends.  Returns the new geometries and how far
    each end was moved, the start of every piece followed by the end of
    every piece.
    """
    import shapely

    coords, index = shapely.get_coordinates(geoms, return_index=True)
    first = np.flatnonzero(np.r_[True, index[1:] != index[:-1]])
    last = np.r_[first[1:], len(index)] - 1
    ends = np.concatenate([first, last])
    _, node = np.unique(np.concatenate([u, v]), return_inverse=True)
    n = np.bincount(node)
    mean = np.column_stack([np.bincount(node, weights=coords[ends, k]) / n
                            for k in (0, 1)])
    moved = np.hypot(*(coords[ends] - mean[node]).T)
    coords[ends] = mean[node]
    return shapely.set_coordinates(np.array(geoms, dtype=object), coords), moved


def _contract_short_pieces(u, v, length, max_len: float = MIN_EDGE_LENGTH_M):
    """Merge the two ends of every piece no longer than ``max_len``.

    Returns ``(u, v, keep)``: the node labels after merging and the pieces
    that still join two different nodes.  A short piece is a real link
    between two connectors a few decimetres apart -- where a footway meets
    the kerb, typically -- and upstream deletes it, which leaves the two
    ways it joined unconnected: 38 such pairs of points were more than 50 m
    apart on the walking network, two of them over 4 km.  Contracting keeps
    the link and loses nothing but its length.  A merged node takes the
    smallest of the labels it replaces.
    """
    import scipy.sparse as sp

    u = np.asarray(u, dtype=object); v = np.asarray(v, dtype=object)
    short = np.asarray(length, dtype="float64") <= max_len
    ids, inv = np.unique(np.concatenate([u, v]), return_inverse=True)
    a = inv[:len(u)]; b = inv[len(u):]
    adj = sp.coo_matrix((np.ones(int(short.sum()), dtype=np.int8),
                         (a[short], b[short])), shape=(len(ids), len(ids)))
    _, label = sp.csgraph.connected_components(adj, directed=False)
    # ``ids`` is sorted, so the first node of each group has its smallest label
    _, first = np.unique(label, return_index=True)
    merged = ids[first][label]
    u = merged[a]; v = merged[b]
    return u, v, u != v


# --------------------------------------------------------------------------
# sidewalks and crossings
# --------------------------------------------------------------------------
def select_sidepaths(u, v, length, core, side, radius: float = 3000.0,
                     stretch: float = 1.3, slack: float = 30.0,
                     join_stranded: bool = True,
                     what: str = "sidewalks and crossings") -> np.ndarray:
    """Which sidewalk and crossing edges the centreline network needs.

    ``core`` marks the edges travel is modelled on (streets, footways,
    steps, paths) and ``side`` the mapped sidewalks and crossings.  A run
    of sidepath edges between two nodes of the core network is kept when
    the core network itself does not join those two nodes, or joins them
    only by a way more than ``stretch`` times longer (and more than
    ``slack`` metres longer): a footbridge that lands on a sidewalk, a
    subway under a road with no crossing at grade, the pavement of a bridge
    whose carriageway is closed to walkers.  A sidewalk that runs from one
    junction of its own street to the next offers nothing the centreline
    does not, and is left out.  That is not every sidewalk beside an open
    street: wherever a footway or a stair lands on a sidewalk in the middle
    of a block, the sidewalk from there to the next crossing is a
    connection the centreline lacks and is kept, so most of what is kept
    does run beside a street the table also holds (nine tenths of it within
    15 m of a walkable centreline), and totals of length count those
    streets twice.

    Runs are searched up to ``radius`` metres, which is far enough to judge
    every run in the territory on its merits (a search to 20 km keeps the
    same edges).  At 300 m the 1.6 km of pavement on Tsing Tsuen Bridge was
    never looked at, and the walk from Tsuen Wan to Tsing Yi went round by
    Kwai Chung, 5.3 km for 2.2.  Whatever the first pass leaves unconnected
    is then joined to the rest by the shortest sidepath connections there
    are (a minimum spanning forest over the remaining components), so
    nothing reachable on the full network is stranded; that pass has no
    limit of length, and ``join_stranded=False`` leaves it out.

    The same search opens short footway links to pushed bicycles, with the
    cycling network as ``core`` (see ``resolve_access``); ``what`` names
    the edges in the log.

    Returns a boolean mask over the edges: the sidepath edges to keep.
    """
    import heapq
    from collections import defaultdict

    u = np.asarray(u); v = np.asarray(v)
    length = np.asarray(length, dtype="float64")
    core = np.asarray(core, dtype=bool); side = np.asarray(side, dtype=bool)
    keep = np.zeros(len(u), dtype=bool)
    if not side.any():
        return keep

    ids, inv = np.unique(np.concatenate([u, v]), return_inverse=True)
    a = inv[:len(u)]; b = inv[len(u):]
    core_adj: dict[int, list] = defaultdict(list)
    side_adj: dict[int, list] = defaultdict(list)
    for e in np.flatnonzero(core):
        core_adj[int(a[e])].append((int(b[e]), float(length[e])))
        core_adj[int(b[e])].append((int(a[e]), float(length[e])))
    for e in np.flatnonzero(side):
        side_adj[int(a[e])].append((int(b[e]), float(length[e]), int(e)))
        side_adj[int(b[e])].append((int(a[e]), float(length[e]), int(e)))
    terminals = sorted(n for n in side_adj if n in core_adj)

    # ---- pass 1: sidepath runs that shorten, or make, a connection ------
    for src in terminals:
        dist = {src: 0.0}
        pred: dict[int, tuple[int, int]] = {}
        heap = [(0.0, src)]
        found: list[int] = []
        while heap:
            d, n = heapq.heappop(heap)
            if d > dist.get(n, np.inf):
                continue
            if n != src and n in core_adj:
                found.append(n)             # a run ends at the core network
                continue
            for m, w, e in side_adj[n]:
                nd = d + w
                if nd <= radius and nd < dist.get(m, np.inf):
                    dist[m] = nd; pred[m] = (n, e)
                    heapq.heappush(heap, (nd, m))
        found = [t for t in found if t > src]        # each pair once
        if not found:
            continue
        limit = max(max(stretch * dist[t], dist[t] + slack) for t in found)
        cdist = {src: 0.0}
        heap = [(0.0, src)]
        while heap:
            d, n = heapq.heappop(heap)
            if d > cdist.get(n, np.inf):
                continue
            for m, w in core_adj[n]:
                nd = d + w
                if nd <= limit and nd < cdist.get(m, np.inf):
                    cdist[m] = nd
                    heapq.heappush(heap, (nd, m))
        for t in found:
            if cdist.get(t, np.inf) <= max(stretch * dist[t], dist[t] + slack):
                continue
            n = t
            while n != src:
                n, e = pred[n]
                keep[e] = True
    if not join_stranded:
        log.info("  %s: %d of %d edges kept (%.0f of %.0f km)", what,
                 int(keep.sum()), int(side.sum()), length[keep].sum() / 1000,
                 length[side].sum() / 1000)
        return keep

    # ---- pass 2: join what is still apart, by the shortest connections --
    parent = np.arange(len(ids))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for e in np.flatnonzero(core | keep):
        ra, rb = find(int(a[e])), find(int(b[e]))
        if ra != rb:
            parent[ra] = rb
    kept_nodes = set(core_adj)
    for e in np.flatnonzero(keep):
        kept_nodes.add(int(a[e])); kept_nodes.add(int(b[e]))
    sources = sorted(n for n in side_adj if n in kept_nodes)
    dist = {n: 0.0 for n in sources}
    owner = {n: n for n in sources}
    pred = {}
    heap = [(0.0, n) for n in sources]
    heapq.heapify(heap)
    while heap:
        d, n = heapq.heappop(heap)
        if d > dist.get(n, np.inf):
            continue
        for m, w, e in side_adj[n]:
            nd = d + w
            if nd < dist.get(m, np.inf):
                dist[m] = nd; owner[m] = owner[n]; pred[m] = (n, e)
                heapq.heappush(heap, (nd, m))
    links = []
    for e in np.flatnonzero(side & ~keep):
        x, y = int(a[e]), int(b[e])
        if x in owner and y in owner and owner[x] != owner[y]:
            links.append((dist[x] + float(length[e]) + dist[y], int(e), x, y))
    links.sort()
    n_joined = 0
    for _, e, x, y in links:
        rx, ry = find(owner[x]), find(owner[y])
        if rx == ry:
            continue
        parent[rx] = ry
        n_joined += 1
        keep[e] = True
        for n in (x, y):
            while n in pred:
                n, pe = pred[n]
                keep[pe] = True
    log.info("  %s: %d of %d edges kept (%.0f of %.0f km); "
             "%d stranded pieces joined in the second pass", what,
             int(keep.sum()), int(side.sum()), length[keep].sum() / 1000,
             length[side].sum() / 1000, n_joined)
    return keep


def harmonise_names(name: pd.Series, name_zh: pd.Series, points,
                    radius: float = NAME_PAIR_RADIUS_M):
    """Give every stretch of a street the same pair of names.

    OpenStreetMap tags most of a street "軒尼詩道 Hennessy Road" and the odd
    stretch of it just "軒尼詩道", or just "Hennessy Road".  Left alone, the
    page would list one walk down Hennessy Road as two streets and offer a
    junction of Hennessy Road with itself.  A stretch with a Chinese name
    alone takes the English name of the nearest stretch that carries that
    Chinese name with an English one, and the other way round.

    Nearest, and within ``radius`` metres: names are not unique across the
    territory, and a pairing taken from wherever it is commonest gave a
    Club Street near Tin Shui Wai the Chinese name of another 28 km away.
    ``points`` holds a few points of each edge (an array of shape
    ``(edges, k, 2)``: its two ends and its middle, say); two stretches are
    as far apart as their closest points, so a stretch that continues a
    street shares a node with it and is at no distance at all.

    Returns the completed ``(name, name_zh)``: ``name`` English where the
    street has an English name at all, ``name_zh`` its Chinese name or None.
    """
    points = np.asarray(points, dtype="float64")
    en = name.to_numpy(dtype=object); zh = name_zh.to_numpy(dtype=object)
    named = name.notna().to_numpy(); paired = named & name_zh.notna().to_numpy()
    by_en: dict[str, list[int]] = {}
    by_zh: dict[str, list[int]] = {}
    for i in np.flatnonzero(paired):
        by_en.setdefault(en[i], []).append(i)
        by_zh.setdefault(zh[i], []).append(i)
    out_en = en.copy(); out_zh = zh.copy()
    for i in np.flatnonzero(named & ~paired):
        cjk = bool(_HAS_CJK_RE.search(en[i])) and not re.search("[A-Za-z]", en[i])
        carriers = (by_zh if cjk else by_en).get(en[i])
        if not carriers:
            continue
        d = np.linalg.norm(points[carriers][:, :, None] - points[i][None, None],
                           axis=-1).min(axis=(1, 2))
        near = int(d.argmin())
        if d[near] > radius:
            continue
        if cjk:
            out_en[i], out_zh[i] = en[carriers[near]], en[i]
        else:
            out_zh[i] = zh[carriers[near]]
    return (pd.Series(out_en, index=name.index, name=name.name),
            pd.Series(out_zh, index=name_zh.index, name=name_zh.name))


def _components(u, v, mask):
    """Connected components of the graph of the edges in ``mask``.

    Returns ``(idx, label, nodes)``: the masked edges, the component each
    of them belongs to and the number of nodes in each component.
    """
    import scipy.sparse as sp

    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return idx, np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    ids, inv = np.unique(np.concatenate([np.asarray(u)[idx], np.asarray(v)[idx]]),
                         return_inverse=True)
    a = inv[:idx.size]; b = inv[idx.size:]
    adj = sp.coo_matrix((np.ones(idx.size, dtype=np.int8), (a, b)),
                        shape=(len(ids), len(ids))).tocsr()
    _, labels = sp.csgraph.connected_components(adj, directed=False)
    return idx, labels[a], np.bincount(labels)


def _component_sizes(u, v, mask) -> np.ndarray:
    """For each edge in ``mask``, the node count of its connected component
    in the graph of the masked edges (0 for edges outside the mask)."""
    out = np.zeros(len(u), dtype=np.int64)
    idx, label, nodes = _components(u, v, mask)
    out[idx] = nodes[label]
    return out


def _run_lengths(u, v, length, mask) -> np.ndarray:
    """For each edge in ``mask``, the total length of its run: the masked
    edges joined to it end to end through shared nodes (0 outside the mask)."""
    out = np.zeros(len(u), dtype="float64")
    idx, label, _ = _components(u, v, mask)
    if idx.size:
        out[idx] = np.bincount(label, weights=np.asarray(length, dtype="float64")[idx])[label]
    return out


def _tunnels(gdf, mask) -> np.ndarray:
    """The tunnel each piece in ``mask`` belongs to (-1 outside the mask).

    The pieces of one tube follow one another through shared nodes.  The
    two tubes of a tunnel share no node, only a name, and lie side by side:
    runs that carry the same name within ``ROAD_TUNNEL_TUBE_GAP_M`` of one
    another are numbered as one tunnel.
    """
    import scipy.sparse as sp
    import shapely

    out = np.full(len(gdf), -1, dtype=np.int64)
    idx, run, nodes = _components(gdf["u"].to_numpy(), gdf["v"].to_numpy(), mask)
    if idx.size == 0:
        return out
    geoms = np.asarray(gdf.geometry.values[idx], dtype=object)
    name = gdf["name"].to_numpy(dtype=object)[idx]
    a, b = shapely.STRtree(geoms).query(geoms, predicate="dwithin",
                                        distance=ROAD_TUNNEL_TUBE_GAP_M)
    twin = np.array([isinstance(name[i], str) and name[i] == name[j]
                     for i, j in zip(a, b)], dtype=bool)
    beside = sp.coo_matrix((np.ones(int(twin.sum()), dtype=np.int8),
                            (run[a[twin]], run[b[twin]])),
                           shape=(len(nodes), len(nodes)))
    _, tunnel = sp.csgraph.connected_components(beside, directed=False)
    out[idx] = tunnel[run]
    return out


def closed_land(path: Path | None = None):
    """The sites nobody may walk or ride through, from Overture's land use.

    Military land other than the firing ranges, and the industrial sites
    named as container terminals or power stations (``config.CLOSED_LAND_*``
    has the reasons).  Returns a GeoDataFrame of ``name`` and polygon in the
    projected CRS.  ``download`` fetches the land-use extract with the
    street network and stops when it cannot, so the extract is missing only
    from a data directory filled before it did; such a build gets an empty
    table here, and a warning that the sites stay open.
    """
    import geopandas as gpd
    import pyarrow.parquet as pq
    import shapely

    from .download import BASE_PARQUETS

    path = BASE_PARQUETS["land_use"] if path is None else Path(path)
    if not path.exists():
        log.warning("no land-use extract at %s: service roads inside "
                    "container terminals, power stations and military land "
                    "stay open", path)
        return gpd.GeoDataFrame({"name": []}, geometry=[], crs=CRS_PROJECTED)
    t = pq.read_table(path, columns=["names", "subtype", "class",
                                     "geometry"]).to_pandas()
    # the English name where there is one, as for a street
    name = t["names"].map(lambda n: next(filter(None, split_name(n)), ""))
    military = ((t["subtype"] == "military")
                & ~t["class"].isin(CLOSED_LAND_EXEMPT_CLASSES))
    named = (t["class"] == "industrial") & name.str.lower().map(
        lambda s: any(phrase in s for phrase in CLOSED_LAND_NAMES))
    keep = (military | named).to_numpy()
    land = gpd.GeoDataFrame(
        {"name": name[keep].to_numpy()},
        geometry=shapely.from_wkb(t["geometry"].to_numpy()[keep]),
        crs=CRS_GEOGRAPHIC).to_crs(CRS_PROJECTED)
    land = land[land.geom_type.isin(["Polygon", "MultiPolygon"])]
    land["geometry"] = land.geometry.buffer(0)
    return land.reset_index(drop=True)


def _inside(geoms, land) -> np.ndarray:
    """For each line, the row of ``land`` its midpoint lies in (-1: none)."""
    import shapely

    out = np.full(len(geoms), -1, dtype=np.int64)
    if len(geoms) and len(land):
        mid = shapely.line_interpolate_point(np.asarray(geoms, dtype=object),
                                             0.5, normalized=True)
        tree = shapely.STRtree(land.geometry.to_numpy())
        line, site = tree.query(mid, predicate="within")
        out[line] = site
    return out


def _split_long_edges(gdf, max_len: float = MAX_EDGE_LENGTH_M):
    """Cut edges longer than ``max_len`` into equal parts.

    The browser payload carries lengths as 16-bit counts of 5 cm, which
    tops out at 3.2 km; a catchwater path or a reservoir road can run
    farther than that between junctions.
    """
    import geopandas as gpd
    from shapely.ops import substring

    long_ = gdf["length_m"] > max_len
    if not long_.any():
        return gdf
    parts = []
    for _, row in gdf[long_].iterrows():
        n = int(np.ceil(row["length_m"] / max_len))
        for k in range(n):
            piece = row.copy()
            piece["geometry"] = substring(row.geometry, k / n, (k + 1) / n,
                                          normalized=True)
            if k > 0:
                piece["u"] = f"long:{row['segment_id']}:{row['part']}:{k}"
            if k < n - 1:
                piece["v"] = f"long:{row['segment_id']}:{row['part']}:{k + 1}"
            piece["length_m"] = piece["geometry"].length
            parts.append(piece)
    log.info("  %d edges over %.0f m cut into %d", int(long_.sum()), max_len,
             len(parts))
    out = pd.concat([gdf[~long_], gpd.GeoDataFrame(parts, crs=gdf.crs)],
                    ignore_index=True)
    return gpd.GeoDataFrame(out, geometry="geometry", crs=gdf.crs)


def _reverse_pieces(gdf, reverse):
    """Turn the pieces in ``reverse`` round: ends swapped, line reversed.

    A piece keeps its place in its segment through ``part``, which
    ``metrics.contiguous_runs`` reads to join consecutive pieces of one
    segment end to end.  A reversed piece no longer follows on from its
    neighbour, so it is renumbered ``-(part + 1)``: a segment reversed whole
    keeps its pieces consecutive, in the new direction, and a reversed piece
    is never consecutive with one that was left alone.
    """
    import geopandas as gpd
    import shapely

    reverse = np.asarray(reverse, dtype=bool)
    if not reverse.any():
        return gdf
    gdf = gdf.copy()
    u = gdf["u"].to_numpy(dtype=object); v = gdf["v"].to_numpy(dtype=object)
    gdf["u"] = np.where(reverse, v, u)
    gdf["v"] = np.where(reverse, u, v)
    gdf["part"] = np.where(reverse, -(gdf["part"].to_numpy() + 1),
                           gdf["part"].to_numpy())
    geom = np.array(gdf.geometry.values, dtype=object)
    geom[reverse] = shapely.reverse(geom[reverse])
    return gdf.set_geometry(gpd.GeoSeries(geom, index=gdf.index, crs=gdf.crs))


def cut_pieces(segments_path: Path, clip_to_city: bool = True):
    """The Overture segments as pieces between nodes, in ``CRS_PROJECTED``.

    Filtered to what could ever be walked or ridden, split at connectors
    and at the ends of flag, rule and name ranges, with the shortest pieces
    contracted and the pieces outside the SAR boundary dropped.  Access is
    not yet decided: each piece carries its segment's rules and its own
    position along the segment (``mid_at``) for ``resolve_access``.
    """
    import geopandas as gpd

    df = load_segments(segments_path)
    with step("filtering segments", log):
        df = _base_filter(df)
    with step("splitting segments at connectors", log):
        rows = _split_geometries(df)

    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=CRS_PROJECTED)
    with step("contracting short pieces", log):
        length = gdf.geometry.length.to_numpy()
        u, v, keep = _contract_short_pieces(gdf["u"], gdf["v"], length)
        short = length <= MIN_EDGE_LENGTH_M
        merged = np.unique(u[short])
        gdf["u"] = u; gdf["v"] = v
        gdf = gdf[keep].reset_index(drop=True)
        geoms, moved = _join_ends(gdf.geometry.values, gdf["u"].to_numpy(),
                                  gdf["v"].to_numpy())
        at_merged = np.isin(np.concatenate([gdf["u"], gdf["v"]]), merged)
        gdf = gdf.set_geometry(gpd.GeoSeries(geoms, crs=CRS_PROJECTED))
        gdf["length_m"] = gdf.geometry.length
        log.info("  %d pieces of %.1f m or less contracted into %d nodes "
                 "(%d longer pieces between the same two ends dropped as "
                 "loops); %d edges", int(short.sum()), MIN_EDGE_LENGTH_M,
                 len(merged), int((~keep & ~short).sum()), len(gdf))
        log.info("  piece ends brought onto their nodes: by up to %.3f m at "
                 "the merged nodes and %.4f m at the others",
                 moved[at_merged].max(initial=0.0),
                 moved[~at_merged].max(initial=0.0))

    if clip_to_city:
        from .neighborhoods import city_boundary
        with step("clipping network to the Hong Kong SAR boundary", log):
            boundary = city_boundary()
            mid = gdf.geometry.interpolate(0.5, normalized=True)
            keep = gpd.GeoSeries(mid, crs=gdf.crs).within(boundary)
            n_before = len(gdf)
            gdf = gdf[keep.to_numpy()].reset_index(drop=True)
            log.info("  %d -> %d edges inside Hong Kong", n_before, len(gdf))
    return gdf


def resolve_access(gdf, land=None):
    """Decide, piece by piece, who may use the street.

    ``gdf`` is the table of ``cut_pieces``; ``land`` the closed sites of
    ``closed_land``, or None for none.  Adds the structure flags, and for
    each mode ``<mode>_allowed`` (what the data's rules say), ``<mode>_ok``
    (routable: the class admits the mode, the rules allow it and none of
    the closures below applies), ``<mode>_oneway`` and
    ``<mode>_restricted``; and ``bike_push`` for the footway links a bicycle
    is pushed across.  Pieces that are one-way against the way they are
    drawn come back turned round, so the returned table is a new one.
    """
    gdf = gdf.copy()
    # structure flags: each piece is wholly one thing (see _split_geometries)
    gdf["is_bridge"] = gdf["flags"].apply(lambda f: "is_bridge" in f)
    gdf["is_tunnel"] = gdf["flags"].apply(lambda f: "is_tunnel" in f)
    gdf["is_structure"] = gdf["is_bridge"] | gdf["is_tunnel"]
    gdf["is_sidepath"] = gdf["subclass"].fillna("").isin(SIDEPATH_SUBCLASSES)
    u = gdf["u"].to_numpy(); v = gdf["v"].to_numpy()
    length = gdf["length_m"].to_numpy()

    # Every vehicular tunnel in Hong Kong is closed to pedestrians and
    # cycles (the tunnel ordinances and by-laws, and Cap. 374 for the
    # expressway ones).  OpenStreetMap says so for all of them but one tube
    # of Tate's Cairn Tunnel, four kilometres under Lion Rock's neighbour,
    # and one slip road of Kai Tak Tunnel, and an open tunnel is the
    # flattest way through any hill, so the rule is applied here to the
    # main-road classes.  A tunnel is judged whole: its pieces are gathered
    # into runs that share nodes, and every piece of a long run is closed,
    # however short the piece (the Kai Tak slip road is 51 m of a 1.3 km
    # run).  A shorter tunnel is closed to a mode where the data bars the
    # mode from any piece of it, the other tube included (``_tunnels``): one
    # tube of Tseung Lam Highway, of the Ma On Shan Bypass and of Smithfield
    # is tagged and the other is not.  The rest stay open, because a tunnel
    # flag on a main road is more often a street under a building or a
    # flyover than a tunnel; so do the service roads under podiums and
    # stations, which are tagged as tunnels and are ordinary ground-level
    # streets with a roof; and so does anything the data opens to the mode
    # by name.
    tunnel = (gdf["is_tunnel"] & gdf["cls"].isin(ROAD_TUNNEL_CLASSES)).to_numpy()
    long_tunnel = _run_lengths(u, v, length, tunnel) > ROAD_TUNNEL_MAX_M
    tube = _tunnels(gdf, tunnel)

    # The gated sites: see config.CLOSED_LAND_NAMES.
    site = np.full(len(gdf), -1, dtype=np.int64)
    if land is not None and len(land):
        street = gdf["cls"].isin(CLOSED_LAND_STREET_CLASSES).to_numpy()
        site[street] = _inside(gdf.geometry.values[street], land)
    if (site >= 0).any():
        km = pd.Series(length[site >= 0] / 1000).groupby(
            land["name"].to_numpy()[site[site >= 0]]).sum().sort_values(ascending=False)
        log.info("  %.1f km of service road lies inside %d closed sites: %s",
                 km.sum(), len(km), "; ".join(
                     f"{name or '(unnamed)'} {x:.1f}" for name, x in km.head(8).items()))

    # Viaducts of trunk roads, for walkers only: see
    # config.VIADUCT_PERMIT_REQUIRED_CLASSES.
    deck = (gdf["cls"].isin(VIADUCT_PERMIT_REQUIRED_CLASSES).to_numpy()
            & gdf["is_bridge"].to_numpy())
    viaduct = _run_lengths(u, v, length, deck) > VIADUCT_MIN_M

    with step("evaluating per-mode access restrictions", log):
        heading = {}
        against = np.zeros(len(gdf), dtype=bool)
        for mode_name, mode in MODES.items():
            res = [evaluate_access(r, mode.access_modes[0], at=at)
                   for r, at in zip(gdf["access_restrictions"], gdf["mid_at"])]
            allowed = np.array([r["allowed"] for r in res], dtype=bool)
            forward = np.array([r["oneway_forward_only"] for r in res], dtype=bool)
            backward = np.array([r["oneway_backward_only"] for r in res], dtype=bool)
            explicit = np.array([r["explicit"] for r in res], dtype=bool)
            heading[mode_name] = (forward, backward)
            gdf[f"{mode_name}_allowed"] = allowed
            gdf[f"{mode_name}_restricted"] = [r["restricted"] for r in res]
            # class eligibility
            ok = gdf["cls"].isin(mode.allowed_classes).to_numpy() & allowed
            if mode_name == "bike":
                ok &= ~gdf["cls"].isin(BIKE_FORBIDDEN_CLASSES).to_numpy()
                ok &= ~gdf["is_sidepath"].to_numpy()   # no cycling on the pavement
                # ... nor on a footway or trail, unless it is signed for it
                ok &= (~gdf["cls"].isin(BIKE_PERMIT_REQUIRED_CLASSES).to_numpy()
                       | explicit)
            if mode.respect_oneway:
                ok &= ~(forward & backward)     # closed in both directions
                against |= backward & ~forward
            # the closures; a rule that names the mode lifts any of them
            closures = [
                ("road tunnel", long_tunnel),
                ("tunnel the data closes in part",
                 tunnel & ~long_tunnel & np.isin(tube, tube[tunnel & ~allowed])),
                ("closed site", site >= 0)]
            if mode_name == "walk":
                closures.append(("trunk viaduct", viaduct))
            closed = np.zeros(len(gdf), dtype=bool)
            notes = []
            for label, where in closures:
                where = where & ~explicit
                closed |= where
                notes.append(f"{label} {length[where & ok].sum() / 1000:.1f} km")
            gdf[f"{mode_name}_ok"] = ok & ~closed
            log.info("  %s: %d of %d edges routable; closed by rule where the "
                     "data leaves them open: %s", mode_name,
                     int(gdf[f"{mode_name}_ok"].sum()), len(gdf), ", ".join(notes))
        # A one-way street drawn against its direction of travel (75
        # segments carry OpenStreetMap's oneway=-1; the Kowloon City flyover
        # is one) is turned round, so that "one-way" always means from u to
        # v and nothing downstream needs a second flag.
        gdf = _reverse_pieces(gdf, against)
        for mode_name, (forward, backward) in heading.items():
            gdf[f"{mode_name}_oneway"] = np.where(against, backward, forward)
        log.info("  %d pieces that are one-way against their direction turned "
                 "round", int(against.sum()))
        u = gdf["u"].to_numpy(); v = gdf["v"].to_numpy()

    # Dismount-and-push links: see config.BIKE_PUSH_MAX_M.  The search is
    # the one that chooses sidewalks, with the cycling network in the place
    # of the centrelines: a run of walkable footway is opened when it joins
    # two points of the cycling network that are otherwise unconnected, or
    # farther apart than the push is worth (the pushed length is charged
    # BIKE_PUSH_MULTIPLIER times, so that many times farther).  Steps,
    # bridges and tunnels are never opened, nor a footway the data closes
    # to bicycles by name, and there is no second pass: nothing longer than
    # the limit is pushed, whatever it would connect.
    with step("opening short footway links to pushed bicycles", log):
        pushable = (gdf["walk_ok"] & ~gdf["bike_ok"] & gdf["bike_allowed"]
                    & gdf["cls"].isin(BIKE_PERMIT_REQUIRED_CLASSES)
                    & ~gdf["is_structure"]).to_numpy()
        push = select_sidepaths(u, v, length, gdf["bike_ok"].to_numpy(), pushable,
                                radius=BIKE_PUSH_MAX_M,
                                stretch=BIKE_PUSH_MULTIPLIER,
                                join_stranded=False, what="pushed links")
        gdf["bike_push"] = push
        gdf["bike_ok"] = gdf["bike_ok"] | push
        gdf["bike_oneway"] = gdf["bike_oneway"] & ~push
    return gdf


def build_edges(segments_path: Path, force: bool = False,
                clip_to_city: bool = True,
                land_use_path: Path | None = None) -> "pd.DataFrame":
    """Produce the undirected edge table with geometry, class and access.

    The Overture extract covers a bounding box, which also catches Shenzhen
    north of the boundary (outside the terrain model, and not reachable on
    foot without a border crossing).  Clipping to the Hong Kong SAR boundary
    keeps the analysis to the territory.

    ``land_use_path`` is Overture's land-use extract, for the sites closed
    by ``closed_land``; the downloaded one by default.
    """
    import geopandas as gpd
    import shapely

    if EDGES_PARQUET.exists() and not force:
        log.info("cached %s", EDGES_PARQUET.name)
        return gpd.read_parquet(EDGES_PARQUET)

    gdf = cut_pieces(segments_path, clip_to_city=clip_to_city)
    gdf = resolve_access(gdf, closed_land(land_use_path))

    with step("choosing the sidewalks and crossings the network needs", log):
        # a sidewalk or crossing a bicycle is pushed along is kept already
        side = (gdf["is_sidepath"] & gdf["walk_ok"] & ~gdf["bike_push"]).to_numpy()
        core = (gdf["walk_ok"] & (~gdf["is_sidepath"] | gdf["bike_push"])).to_numpy()
        chosen = select_sidepaths(gdf["u"].to_numpy(), gdf["v"].to_numpy(),
                                  gdf["length_m"].to_numpy(), core, side)
        pushed = (gdf["is_sidepath"] & gdf["bike_push"]).to_numpy()
        km = gdf["length_m"].to_numpy() / 1000
        log.info("  with the %.0f km a bicycle is pushed along, %.0f of the "
                 "%.0f km of walkable sidewalk and crossing is kept",
                 km[pushed].sum(), km[chosen | pushed].sum(),
                 km[side | pushed].sum())
        gdf = gdf[~gdf["is_sidepath"].to_numpy() | chosen
                  | pushed].reset_index(drop=True)

    with step("dropping fragments", log):
        u = gdf["u"].to_numpy(); v = gdf["v"].to_numpy()
        big = np.zeros(len(gdf), dtype=bool)
        for mode_name in MODES:
            ok = gdf[f"{mode_name}_ok"].to_numpy()
            big |= _component_sizes(u, v, ok) >= MIN_COMPONENT_NODES
        log.info("  %d of %d edges (%.0f of %.0f km) are in a component of at "
                 "least %d nodes for walking or cycling", int(big.sum()), len(gdf),
                 gdf.loc[big, "length_m"].sum() / 1000,
                 gdf["length_m"].sum() / 1000, MIN_COMPONENT_NODES)
        gdf = gdf[big].reset_index(drop=True)

    gdf = _split_long_edges(gdf)
    with step("pairing English and Chinese street names", log):
        geom = gdf.geometry.values
        points = np.stack([shapely.get_coordinates(
            shapely.line_interpolate_point(geom, f, normalized=True))
            for f in (0.0, 0.5, 1.0)], axis=1)
        gdf["name"], gdf["name_zh"] = harmonise_names(gdf["name"], gdf["name_zh"],
                                                      points)

    # bicycle infrastructure / low-stress proxy from the road class.  A
    # footway or path is a shared path only where a bicycle may ride it:
    # not where it is closed to cycling, and not where it is pushed.
    ridden = (gdf["bike_ok"] & ~gdf["bike_push"]).to_numpy()
    gdf["bike_facility"] = np.where(
        gdf["cls"] == "cycleway", "dedicated_cycleway",
        np.where(gdf["cls"].isin(["path", "footway"]) & ridden,
                 "shared_path",
                 np.where(gdf["cls"] == "living_street", "living_street",
                          np.where(gdf["cls"] == "pedestrian", "car_free_street", ""))))
    gdf["bike_facility"] = np.where(gdf["bike_push"], "", gdf["bike_facility"])
    gdf["low_stress"] = gdf["bike_facility"].astype(bool) | gdf["cls"].isin(
        ["residential", "living_street", "pedestrian"])

    gdf["edge_id"] = np.arange(len(gdf), dtype=np.int64)
    keep = ["edge_id", "segment_id", "u", "v", "part", "cls", "subclass", "name",
            "name_zh", "length_m", "is_bridge", "is_tunnel", "is_structure",
            "walk_ok", "walk_oneway", "walk_restricted",
            "bike_ok", "bike_oneway", "bike_restricted", "bike_push",
            "bike_facility", "low_stress", "geometry"]
    gdf = gdf[keep]
    # Written beside the table and moved into place.  A run directory may
    # hold a symbolic link to the main build's table here
    # (``sensitivity._share``); writing through it would overwrite the main
    # build, where replacing it only replaces the link.
    part = EDGES_PARQUET.with_name(EDGES_PARQUET.name + ".part")
    gdf.to_parquet(part)
    part.replace(EDGES_PARQUET)
    log.info("wrote %s (%d edges)", EDGES_PARQUET.name, len(gdf))
    return gdf


# --------------------------------------------------------------------------
# graph assembly
# --------------------------------------------------------------------------
def largest_component(edges, mode: str):
    """Restrict an edge table to the largest connected component for a mode."""
    import networkx as nx
    sub = edges[edges[f"{mode}_ok"]]
    g = nx.Graph()
    g.add_edges_from(zip(sub["u"], sub["v"]))
    if g.number_of_nodes() == 0:
        raise RuntimeError(f"no routable edges for mode {mode}")
    comp = max(nx.connected_components(g), key=len)
    keep = sub["u"].isin(comp) & sub["v"].isin(comp)
    log.info("mode %s: largest component has %d nodes, %d of %d edges",
             mode, len(comp), int(keep.sum()), len(sub))
    return sub[keep].copy(), comp
