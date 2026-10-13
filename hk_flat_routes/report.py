"""Generate the written analysis of the major findings.

Every number in the report is read from the analysis outputs rather than
typed in, so the prose cannot drift away from the data.

Upstream's report was half generated figures and half hand-written San
Francisco geography (the Wiggle, the Mission valley floor, the pass in
Golden Gate Park).  This port goes further and types in no finding about
Hong Kong at all: every street, corridor, pass and neighborhood named below
is looked up in the outputs of the run the report is generated from, and
every sentence that draws a conclusion is chosen by a comparison of those
numbers.  The same code therefore reports an urban-core test build and the
whole territory without an edit, and cannot state something a run did not
find.  The only place names that do not come from a run are the three
landmass names in ``config.LANDMASS_NAMES`` and the harbour between the
first two, which is named only when a run has both shores and has routed
pairs on each.  The closing section on limits describes the method, not
the territory, and its figures are computed like the rest.

Two things are specific to Hong Kong and shape the whole document.

*Landmasses.*  The territory is several pieces with no walking link between
them, and neighborhood pairs are routed only within a piece.  Each
neighborhood's piece is read from the ``component`` column of its access
point, the pieces are named by the neighborhoods found on them
(``config.LANDMASS_NAMES``), and the headline, the corridors and the passes
are given per landmass as well as overall.

*Units.*  Kilometres, metres, metres per kilometre and percent throughout;
heights are metres above the Hong Kong Principal Datum (mPD).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import (ANALYSIS, BIKE, BIKE_PERMIT_REQUIRED_CLASSES,
                     BIKE_PUSH_MAX_M, BIKE_PUSH_MULTIPLIER, CITY_NAME,
                     ELEVATION, FEATURED_PAIRS, LANDMASS_NAMES,
                     MIN_COMPONENT_NODES, OUTPUT_DIR, PARETO_LAMBDA_SWEEP,
                     PROCESSED_DIR, ROUTING_PROFILES, SIDEPATH_SUBCLASSES, WALK)
from .utils import get_logger

log = get_logger("hk_flat_routes.report")

REPORT_MD = OUTPUT_DIR / "findings.md"

#: Rows shown per landmass in the corridor and pass tables.
TOP_PER_LANDMASS = 8
#: A pass that cuts one neighborhood off from the rest, and stands no more
#: than this (m) above that neighborhood's own access point, is described as
#: the height of the neighborhood rather than as a ridge in front of it.
HILLTOP_M = 15.0
#: A pair's lowest crossing counts as "one end of the trip" when it is no
#: more than this (m) above the higher of the two access points.
RISE_TOLERANCE_M = 5.0
#: Corridors that never drop below this height (mPD) are reported apart
#: from those on low ground, and called level below this mean gradient.
HIGH_CORRIDOR_M = 50.0
LEVEL_GRADE = 0.02
#: An access point more than this (m) above the median height of its
#: neighbourhood's own streets is said to be so, and a pass at its doorstep
#: is not called the height of the neighbourhood.  The same figure as the
#: band ``neighborhoods.ACCESS_HEIGHT_BAND_M`` holds access points to, so
#: this singles out exactly the points that rule could not bring down.
ACCESS_ABOVE_STREETS_M = 10.0
#: What a way that is not a street is called in the tables.
OFF_STREET = {"path": "hill path", "track": "track", "steps": "steps",
              "footway": "footpath"}


# --------------------------------------------------------------------------
# small formatting helpers
# --------------------------------------------------------------------------
def _km(m) -> float:
    return m / 1000.0


def _list(names, limit: int | None = None) -> str:
    """'A', 'A and B', 'A, B and C'; with ``limit``, 'A, B and 5 more'."""
    names = [str(n) for n in names]
    if limit is not None and len(names) > limit + 1:
        return f"{', '.join(names[:limit])} and {len(names) - limit} more"
    if len(names) <= 1:
        return "".join(names)
    return f"{', '.join(names[:-1])} and {names[-1]}"


def _text(v) -> str:
    """A cell value as text, with NaN and None as the empty string."""
    return v.strip() if isinstance(v, str) else ""


def _way_name(row) -> str:
    """Street name of a pass or barrier, or what kind of unnamed way it is."""
    name = _text(row.get("name"))
    if name:
        return name
    cls = _text(row.get("cls"))
    return f"(unnamed {cls})" if cls else "(unnamed)"


def _way_label(row) -> str:
    """As ``_way_name``, saying so when the way is not a street.

    The steepest things a shortest route crosses in Hong Kong are mostly
    ridge paths and old footpaths over the hills, many of them named only
    in Chinese, and a table headed "street" that lists them reads as if a
    road went up there.
    """
    kind = OFF_STREET.get(_text(row.get("cls")))
    if not kind:
        return _way_name(row)
    name = _text(row.get("name"))
    return f"{name} ({kind})" if name else f"(unnamed {kind})"


def _where(row) -> str:
    from .passes import place_label
    d = row.get("nearest_distance_m")
    return place_label(row.get("neighborhood"), row.get("nearest_neighborhood"),
                       float(d) if d is not None and pd.notna(d) else None)


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def _load():
    import geopandas as gpd
    from .corridors import CORRIDORS_GPKG
    from .pairs import PAIRS_PARQUET, PARETO_PARQUET
    from .passes import (BASIN_PASSES_CSV, BASINS_CSV, PASS_MATRIX_CSV,
                         PASS_MATRIX_COLUMNS, read_barriers, read_passes)

    out = {
        "edges": gpd.read_parquet(PROCESSED_DIR / "edges_metrics.parquet"),
        "pairs": pd.read_parquet(PAIRS_PARQUET),
        "corridors": gpd.read_file(CORRIDORS_GPKG),
        "passes": read_passes(),
        "barriers": read_barriers(),
        # written by ``neighborhoods.load_neighborhoods`` and
        # ``choose_representative_points``; named here rather than imported
        # so that writing the report does not load the download machinery
        "neighborhoods": gpd.read_file(PROCESSED_DIR / "neighborhoods.gpkg"),
    }
    for mode in ("walk", "bike"):
        path = PROCESSED_DIR / f"neighborhood_points_{mode}.gpkg"
        if path.exists():
            out[f"points_{mode}"] = gpd.read_file(path)
    if PARETO_PARQUET.exists():
        out["pareto"] = pd.read_parquet(PARETO_PARQUET)
    out["pass_matrix"] = (pd.read_csv(PASS_MATRIX_CSV) if PASS_MATRIX_CSV.exists()
                          else pd.DataFrame(columns=list(PASS_MATRIX_COLUMNS)))
    for key, path in (("basins", BASINS_CSV), ("basin_passes", BASIN_PASSES_CSV)):
        if path.exists():
            try:
                out[key] = pd.read_csv(path)
            except pd.errors.EmptyDataError:
                pass
    return out


# --------------------------------------------------------------------------
# landmasses
# --------------------------------------------------------------------------
def _pieces(edges, mode: str = "walk"):
    """Connected pieces of a mode's network.

    Returns ``(piece_of_node, n_nodes, km)``: a dict from node id to piece
    number, and each piece's node count and street length.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    col = f"{mode}_ok"
    sub = edges[edges[col]] if col in edges.columns else edges
    n = len(sub)
    codes, nodes = pd.factorize(pd.concat([sub["u"], sub["v"]], ignore_index=True))
    a, b = codes[:n], codes[n:]
    g = coo_matrix((np.ones(n, dtype="int8"), (a, b)), shape=(len(nodes),) * 2)
    k, lab = connected_components(g, directed=False)
    km = np.bincount(lab[a], weights=sub["length_m"].to_numpy("float64"),
                     minlength=k) / 1000.0
    return dict(zip(nodes, lab)), np.bincount(lab, minlength=k), km


def name_landmasses(members: dict, area: dict | None = None) -> dict:
    """Name each landmass from the neighborhoods found on it.

    ``members`` maps a landmass key to the neighborhood names on it.  Each
    name in ``config.LANDMASS_NAMES`` goes to the one landmass holding the
    most of its listed neighborhoods (ties to the landmass with more
    neighborhoods in all); a landmass left without one is called after its
    largest neighborhood by ``area``, so a fragment at the edge of a test
    window, or an outlying island with one planning area, still gets a
    name a reader can place.
    """
    area = area or {}
    names: dict = {}
    for label, anchors in LANDMASS_NAMES:
        best, best_score = None, (0, 0)
        for key, mem in members.items():
            if key in names:
                continue
            score = (len(set(anchors) & set(mem)), len(mem))
            if score[0] > 0 and score > best_score:
                best, best_score = key, score
        if best is not None:
            names[best] = label
    taken = set(names.values())
    for key, mem in members.items():
        if key in names:
            continue
        lead = max(mem, key=lambda n: (area.get(n, 0.0), n))
        label = lead if len(mem) == 1 else f"{lead} and {len(mem) - 1} more"
        while label in taken:            # two fragments led by the same name
            label += " (another piece)"
        names[key] = label
        taken.add(label)
    return names


def _groups_from_pairs(pairs: pd.DataFrame) -> dict:
    """{neighborhood: group number} from which pairs have a route."""
    parent: dict = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for o, dst in zip(pairs["origin"], pairs["destination"]):
        parent[find(o)] = find(dst)
    roots = sorted({find(x) for x in parent})
    return {x: roots.index(find(x)) for x in parent}


def _landmasses(d) -> None:
    """Work out which landmass each neighborhood is on; store it in ``d``.

    The landmass is the connected component of the walking graph that the
    neighborhood's access point lies on, as recorded by
    ``choose_representative_points`` in the ``component`` column.  (A file
    written before that column existed is handled by finding the pieces of
    the walkable network afresh.)  Street length per landmass comes from
    the same pieces.
    """
    edges, nb = d["edges"], d["neighborhoods"]
    pts = d.get("points_walk")
    piece_of, piece_nodes, piece_km = _pieces(edges, "walk")
    area = dict(zip(nb["neighborhood"], nb["area_km2"])) \
        if "area_km2" in nb.columns else {}

    all_names = list(nb["neighborhood"])
    by_component = False
    if pts is not None:
        has = pts[pts["node"].notna()]
        node_of = dict(zip(has["neighborhood"], has["node"]))
        if "component" in has.columns and has["component"].notna().all():
            comp_of = {n: int(c) for n, c in zip(has["neighborhood"], has["component"])}
            by_component = True
        else:
            comp_of = {n: int(piece_of[v]) for n, v in node_of.items()
                       if v in piece_of}
    else:
        # no points file at all: two neighborhoods share a landmass exactly
        # when the pair table has a route between them
        node_of, comp_of = {}, _groups_from_pairs(d["pairs"])

    members: dict = {}
    for n in sorted(comp_of):
        members.setdefault(comp_of[n], []).append(n)
    names = name_landmasses(members, area)

    walk = d["pairs"][(d["pairs"]["mode"] == "walk")
                      & (d["pairs"]["profile"] == "shortest")]
    lms = []
    used_pieces = set()
    for key, mem in members.items():
        piece = next((int(piece_of[node_of[n]]) for n in mem
                      if node_of.get(n) in piece_of), None)
        if piece is not None:
            used_pieces.add(piece)
        inside = walk[walk["origin"].isin(mem) & walk["destination"].isin(mem)]
        lms.append({
            "key": key, "name": names[key], "neighborhoods": mem,
            "km": float(piece_km[piece]) if piece is not None else float("nan"),
            "possible": len(mem) * (len(mem) - 1),
            "routed": int(len(inside)),
        })
    lms.sort(key=lambda m: (-len(m["neighborhoods"]), m["name"]))

    big = np.flatnonzero(piece_nodes >= MIN_COMPONENT_NODES)
    # Without access points (no points file) no landmass can be matched to
    # its piece of network, and every piece would be counted as holding no
    # neighbourhood; nothing is said about the other pieces then.
    other = [int(p) for p in big if int(p) not in used_pieces] if node_of else []
    lm_of = {n: m["name"] for m in lms for n in m["neighborhoods"]}
    d["lm"] = lms
    # True when a landmass's key is its component rank in the walking graph,
    # which is also how the corridor table tags its rows
    d["lm_by_component"] = by_component
    d["lm_of"] = lm_of
    d["lm_names"] = [m["name"] for m in lms]
    d["no_network"] = sorted(set(all_names) - set(comp_of))
    d["n_pieces"] = int(len(big))
    d["other_pieces_km"] = sorted((float(piece_km[p]) for p in other), reverse=True)
    d["area"] = area
    # the walking pair table, tagged with the landmass both ends are on
    p = d["pairs"]
    d["pairs"] = p.assign(landmass=p["origin"].map(lm_of),
                          landmass_to=p["destination"].map(lm_of))


def _with_pairs(d) -> list[dict]:
    """Landmasses that have at least one routed pair, largest first."""
    return [m for m in d["lm"] if m["routed"] > 0]


def _intro(d) -> list[str]:
    nb = d["neighborhoods"]
    n_all = len(nb)
    n_net = n_all - len(d["no_network"])
    L = [f"# {CITY_NAME}'s flat street network: findings", "",
         "*Generated by `python -m hk_flat_routes report`. Every figure, and "
         "every street, corridor, pass and neighbourhood named, is computed "
         "from the analysis outputs of the run this file was generated from; "
         "see `validation_report.md` for the checks against known ground "
         "truth. Distances are in kilometres and metres, heights in metres "
         "above Principal Datum (mPD).*", ""]
    if d["no_network"]:
        L += [f"*Scope of this run: {n_net} of the {n_all} neighbourhoods "
              f"(Town Planning Board planning areas) have a street network in "
              f"it. The other {len(d['no_network'])} have none and appear "
              f"nowhere below: {_list(d['no_network'])}.*", ""]
    else:
        L += [f"*Scope of this run: all {n_all} neighbourhoods (Town Planning "
              f"Board planning areas).*", ""]
    return L


def _landmass_section(d) -> list[str]:
    lms = d["lm"]
    n_net = sum(len(m["neighborhoods"]) for m in lms)
    routed = sum(m["routed"] for m in lms)
    possible = n_net * (n_net - 1)
    several = len(lms) > 1
    L = ["## Several landmasses, not one city" if several
         else "## The network of this run", ""]
    lead = (f"On foot, {CITY_NAME} is not one network. " if d["n_pieces"] > 1
            else "")
    L += [f"{lead}The walkable streets of this run fall into "
          f"{d['n_pieces']} connected piece{'s' if d['n_pieces'] != 1 else ''} "
          f"of at least {MIN_COMPONENT_NODES} intersections each, and a route "
          f"exists only between two places on the same piece. "
          f"{len(lms)} of them "
          f"hold{'s' if len(lms) == 1 else ''} a neighbourhood's access "
          f"point:", "",
          "| Landmass | Neighbourhoods | Walkable street | Ordered pairs routed |",
          "|---|---|---|---|"]
    for m in lms:
        km = f"{m['km']:,.0f} km" if np.isfinite(m["km"]) else "n/a"
        L.append(f"| {m['name']} | {len(m['neighborhoods'])} | {km} | "
                 f"{m['routed']:,} |")
    L.append("")
    if d["other_pieces_km"]:
        o = d["other_pieces_km"]
        one = len(o) == 1
        L += [f"The other {'piece' if one else f'{len(o)} pieces'} "
              f"({sum(o):,.0f} km of street"
              + ("" if one else f" in all, the largest {o[0]:,.0f} km")
              + f") hold{'s' if one else ''} no neighbourhood of the set and "
              f"take{'s' if one else ''} no part in the pair analysis, though "
              f"the route finder covers "
              f"{'it' if one else 'them'}.", ""]

    cross = d["pairs"][(d["pairs"]["mode"] == "walk")
                       & (d["pairs"]["landmass"] != d["pairs"]["landmass_to"])]
    if possible > routed:
        L += [f"Neighbourhood pairs are routed only within a landmass. That "
              f"gives {routed:,} ordered pairs on foot, where {n_net} "
              f"neighbourhoods on a single landmass would give "
              f"{possible:,}; the other {possible - routed:,} have no "
              f"walking route at all, and the pair table has {len(cross)} "
              f"rows that join two landmasses.", ""]
    elif routed:
        L += [f"All {n_net} neighbourhoods of this run are on one landmass, "
              f"and every one of the {routed:,} ordered pairs between them "
              f"has a walking route.", ""]

    # The harbour.  Said only when the run has both shores in it and has
    # routed pairs on each, so that "no route across" is a finding of the
    # routing and not the silence of a stage that never ran.
    by_name = {m["name"]: m for m in lms}
    isl, kln = (by_name.get(LANDMASS_NAMES[0][0]), by_name.get(LANDMASS_NAMES[1][0]))
    if isl and kln and isl["routed"] and kln["routed"]:
        n = 2 * len(isl["neighborhoods"]) * len(kln["neighborhoods"])
        got = cross[cross["landmass"].isin([isl["name"], kln["name"]])
                    & cross["landmass_to"].isin([isl["name"], kln["name"]])]
        both = d["pairs"][d["pairs"]["landmass"] != d["pairs"]["landmass_to"]]
        if got.empty:
            L += [f"**Nothing crosses Victoria Harbour on foot.** Not one of "
                  f"the {n:,} ordered pairs between {isl['name']} "
                  f"({len(isl['neighborhoods'])} neighbourhoods) and "
                  f"{kln['name']} ({len(kln['neighborhoods'])}) has a "
                  f"walking route"
                  + (", nor a cycling one" if both.empty else "")
                  + ". The street data holds no link between the two that "
                  "is open to a walker, so every figure below is about "
                  "getting around *within* a landmass.", ""]
        else:
            L += [f"{len(got):,} of the {n:,} ordered pairs between "
                  f"{isl['name']} and {kln['name']} have a walking route in "
                  f"this run.", ""]
    alone = [m for m in lms if len(m["neighborhoods"]) == 1]
    if alone and several:
        L += [f"{_list([m['neighborhoods'][0] for m in alone])} "
              f"{'is' if len(alone) == 1 else 'are each'} alone on "
              f"{'its' if len(alone) == 1 else 'their'} piece of network in "
              f"this run, with no other neighbourhood to be paired with.", ""]
    return L


# --------------------------------------------------------------------------
def _profile(walk: pd.DataFrame, name: str) -> pd.DataFrame:
    return walk[walk["profile"] == name]


def _sweep_least(d, mode: str = "walk"):
    """Least climbing found for each ordered pair anywhere in the weight sweep.

    The objective the tables call flattest is strongly climb-averse, not a
    minimum, and the sweep behind the trade-off section runs past it; where
    a sentence claims that nothing climbs less, this is what it is checked
    against.  A series indexed by (origin, destination), or ``None`` when
    the run has no sweep.
    """
    pa = d.get("pareto")
    if pa is None or pa.empty:
        return None
    pa = pa[pa["mode"] == mode]
    return pa.groupby(["origin", "destination"])["elev_gain_m"].min()


def _headline(d) -> list[str]:
    e, p = d["edges"], d["pairs"]
    walk = p[(p["mode"] == "walk")]
    short = _profile(walk, "shortest")
    flat = _profile(walk, "min_climb")
    bal = _profile(walk, "balanced")
    ga = _profile(walk, "grade_averse")

    walkable = e[e["walk_ok"]] if "walk_ok" in e.columns else e
    km = walkable["length_m"].sum() / 1000
    total_km = e["length_m"].sum() / 1000
    L = [
        "## The headline",
        "",
        f"The street network modelled here is {total_km:,.0f} km long, of "
        f"which {km:,.0f} km is walkable. It climbs an average of "
        f"{walkable['cum_gain_fwd'].sum()/km:.1f} m for every kilometre of "
        f"street. But that average conceals a usable low-elevation network. "
        f"Across all {len(short):,} ordered neighbourhood pairs, on foot:",
        "",
        "| Objective | Mean distance | Mean climb | Mean steepest grade | "
        "Distance penalty | Climbing avoided |",
        "|---|---|---|---|---|---|",
    ]
    for label, g in (("Shortest (distance only)", short),
                     ("Balanced", bal),
                     ("Flattest (strongly climb-averse)", flat),
                     ("Grade-averse", ga)):
        if g.empty:
            continue
        L.append(
            f"| {label} | {_km(g['distance_m'].mean()):.2f} km | "
            f"{g['elev_gain_m'].mean():.0f} m | "
            f"{g['max_grade'].mean():.1%} | "
            f"{100*(g['detour_ratio'].mean()-1):+.0f}% | "
            f"{g['gain_saved_pct'].mean():.0f}% |")
    if short.empty or flat.empty:
        return L + [""]
    extra = 100 * (flat["detour_ratio"].mean() - 1)
    saved = flat["gain_saved_pct"].mean()
    alpha = ROUTING_PROFILES["min_climb"].alpha
    L += [
        "",
        f"**About {extra:.0f}% more walking buys about {saved:.0f}% less "
        f"climbing.** That is the central result. The flattest route, here "
        f"and below, is the route of the strongly climb-averse objective, "
        f"which accepts {alpha:.0f} m of extra walking to save one metre of "
        f"climb: it is on average {extra:.0f}% longer than the shortest "
        f"one, it avoids {saved:.0f}% of the ascent, and it takes the "
        f"typical steepest pitch from {short['max_grade'].mean():.0%} to "
        f"{flat['max_grade'].mean():.0%}.",
        "",
    ]
    least = _sweep_least(d)
    if least is not None:
        by_pair = flat.set_index(["origin", "destination"])
        f_gain, s_gain = by_pair["elev_gain_m"], by_pair["shortest_gain_m"]
        # a pair the sweep does not hold keeps the flattest route's figure
        low = least.reindex(by_pair.index).fillna(f_gain)
        under = f_gain - low
        if (under > 1.0).any():
            # averaged as ``gain_saved_pct`` is: a level pair saves nothing
            best = (100 * (s_gain - low) / s_gain).where(s_gain > 0, 0.0).mean()
            L += [f"It is not the least climbing possible. The sweep of "
                  f"weights behind the trade-off section further down goes "
                  f"on to objectives more climb-averse still, and it holds "
                  f"a route with more than 1 m less climbing than the "
                  f"flattest route for {int((under > 1.0).sum()):,} of the "
                  f"{len(flat):,} pairs ({int((under > 5.0).sum()):,} by "
                  f"more than 5 m, {under.max():.0f} m at most). Taking the "
                  f"least climbing found for every pair would put the "
                  f"climbing avoided at {best:.1f}% where the flattest "
                  f"route gives {saved:.1f}%.", ""]

    # --- the same, landmass by landmass ---------------------------------
    lms = _with_pairs(d)
    if len(lms) > 1:
        L += ["The landmasses do not share that result equally:", "",
              "| Landmass | Ordered pairs | Shortest route | Flattest route | "
              "Extra distance | Climbing avoided | Mean steepest grade |",
              "|---|---|---|---|---|---|---|"]
        rows = []
        for m in lms:
            s = short[short["landmass"] == m["name"]]
            f = flat[flat["landmass"] == m["name"]]
            if s.empty or f.empty:
                continue
            rows.append((m["name"], 100 * (f["detour_ratio"].mean() - 1),
                         f["gain_saved_pct"].mean(), s["elev_gain_m"].mean()))
            L.append(
                f"| {m['name']} | {len(s):,} | "
                f"{_km(s['distance_m'].mean()):.2f} km / "
                f"{s['elev_gain_m'].mean():.0f} m | "
                f"{_km(f['distance_m'].mean()):.2f} km / "
                f"{f['elev_gain_m'].mean():.0f} m | "
                f"{rows[-1][1]:+.0f}% | {rows[-1][2]:.0f}% | "
                f"{s['max_grade'].mean():.0%} to {f['max_grade'].mean():.0%} |")
        L += ["", "Each route cell is mean distance / mean cumulative climb.", ""]
        if len(rows) > 1:
            most = max(rows, key=lambda r: r[2])
            fewest = min(rows, key=lambda r: r[2])
            hilliest = max(rows, key=lambda r: r[3])
            L += [f"The flat route pays best on {most[0]}, where it avoids "
                  f"{most[2]:.0f}% of the climbing for {most[1]:.0f}% more "
                  f"distance, and least on {fewest[0]} ({fewest[2]:.0f}% for "
                  f"{fewest[1]:.0f}%). The shortest routes climb most on "
                  f"{hilliest[0]}: {hilliest[3]:.0f} m on average.", ""]

    # --- grade-averse against flattest ----------------------------------
    if not ga.empty:
        g_gain, f_gain = ga["elev_gain_m"].mean(), flat["elev_gain_m"].mean()
        g_det = 100 * (ga["detour_ratio"].mean() - 1)
        g_max, f_max = ga["max_grade"].mean(), flat["max_grade"].mean()
        L += ["The grade-averse objective is worth separating out. It climbs "
              f"{'more' if g_gain > f_gain else 'less'} in total than the "
              f"flattest route ({g_gain:.0f} m against {f_gain:.0f} m) and "
              f"costs {'more' if g_det > extra else 'less'} distance "
              f"({g_det:+.0f}% against {extra:+.0f}%), and it brings the mean "
              f"steepest pitch to {g_max:.1%} where the flattest route "
              f"leaves {f_max:.1%}."
              + (" Total climbing and peak steepness are genuinely different "
                 "objectives, and a single definition of \"flat\" cannot "
                 "serve both: minimising total ascent will happily send you "
                 "up one short wall, and avoiding walls will make you climb "
                 "a little more overall."
                 if g_gain > f_gain and g_max < f_max else ""), ""]

    # --- what the shortest route is like --------------------------------
    cap = ELEVATION.max_plausible_grade
    at_cap = int((short["max_grade"] >= cap - 1e-6).sum())
    steps_km = e[e["cls"] == "steps"]["length_m"].sum() / 1000
    L += ["A note on the baseline. The shortest pedestrian route minimises "
          "distance only, as specified, and a distance-minimising walker "
          f"takes every stairway and hillside path on the way (the walkable "
          f"network has {steps_km:,.0f} km of steps): the mean steepest pitch "
          f"on a shortest walking route is {short['max_grade'].mean():.0%}, "
          f"and {at_cap:,} of the {len(short):,} "
          f"({100*at_cap/len(short):.0f}%) reach the model's {cap:.0%} "
          f"plausibility ceiling. That is what minimising distance means "
          f"here, and it is a large part of why the flat alternatives "
          f"matter.", ""]
    if "path_gain_m" in short.columns and short["elev_gain_m"].sum() > 0:
        def share(g, col, of):
            return 100 * g[col].sum() / g[of].sum() if g[of].sum() > 0 else 0.0
        L += ["Much of the baseline's climbing is not on streets at all. "
              f"{share(short, 'path_gain_m', 'elev_gain_m'):.0f}% of the "
              f"climbing on the shortest routes is done on hill paths, "
              f"tracks and steps, which make up "
              f"{share(short, 'path_distance_m', 'distance_m'):.0f}% of "
              f"their length; on the flattest routes it is "
              f"{share(flat, 'path_gain_m', 'elev_gain_m'):.0f}% of the "
              f"climbing and "
              f"{share(flat, 'path_distance_m', 'distance_m'):.0f}% of the "
              f"length. Few people's shortest walk between two districts "
              f"goes straight over the hill between them on a path, so part "
              f"of the climbing the flat route avoids is climbing a walker "
              f"who keeps to streets would not have done either.", ""]
    steep = 0.15
    f_steep = 100 * (flat["max_grade"] > steep).mean()
    line = (f"A steep pitch can survive on a *flat* route too: "
            f"{f_steep:.0f}% of the flattest routes still include a pitch "
            f"above {steep:.0%}. A short stairway costs only a few hundred "
            f"equivalent metres under the cost model, so when the "
            f"alternative is a longer detour than that, the model takes the "
            f"stairs -- which is what a pedestrian does too.")
    if not ga.empty:
        g_steep = 100 * (ga["max_grade"] > steep).mean()
        line += (f" The grade-averse objective is the one that refuses "
                 f"them: {g_steep:.0f}% of its routes include such a pitch.")
    L += [line, ""]
    return L


def _cell(g) -> str:
    if g.empty:
        return "n/a"
    r = g.iloc[0]
    return (f"{_km(r['distance_m']):.2f} km / {r['elev_gain_m']:.0f} m "
            f"/ max {r['max_grade']:.0%}")


def _featured(d) -> list[str]:
    p = d["pairs"]
    walk = p[p["mode"] == "walk"]
    flat = _profile(walk, "min_climb")
    if flat.empty:
        return []
    L = ["## Specific answers", "",
         "### What is the flattest reasonable route between two "
         "neighbourhoods?", "",
         "| Landmass | From | To | Shortest | Flattest | Balanced |",
         "|---|---|---|---|---|---|"]
    shown = 0
    for o, dst in FEATURED_PAIRS:
        sub = walk[(walk["origin"] == o) & (walk["destination"] == dst)]
        if sub.empty:
            continue
        shown += 1
        L.append(f"| {d['lm_of'].get(o, '')} | {o} | {dst} | "
                 f"{_cell(sub[sub['profile']=='shortest'])} | "
                 f"{_cell(sub[sub['profile']=='min_climb'])} | "
                 f"{_cell(sub[sub['profile']=='balanced'])} |")
    L += ["", "Each cell is distance / cumulative climb / steepest gradient."
          + (f" {len(FEATURED_PAIRS) - shown} of the {len(FEATURED_PAIRS)} "
             f"featured pairs have an end with no street network in this "
             f"run and are left out."
             if shown < len(FEATURED_PAIRS) else ""), ""]

    # --- where it pays most ---------------------------------------------
    # one row per unordered pair: the direction that saves the more
    key = [tuple(sorted(t)) for t in zip(flat["origin"], flat["destination"])]
    best = (flat.assign(_key=key).sort_values("gain_saved_m", ascending=False)
            .drop_duplicates("_key").head(8))
    L += ["### Where does flat routing pay off most?", "",
          "The neighbourhood pairs where choosing the flat route avoids the "
          "most climbing (each pair once, in its better direction):", "",
          "| From | To | Shortest | Flattest | Climbing avoided | "
          "Metres of climb saved per extra metre walked |",
          "|---|---|---|---|---|---|"]
    for _, r in best.iterrows():
        eff = r["climb_saved_per_extra_m"]
        L.append(
            f"| {r['origin']} | {r['destination']} | "
            f"{_km(r['shortest_distance_m']):.2f} km / "
            f"{r['shortest_gain_m']:.0f} m | "
            f"{_km(r['distance_m']):.2f} km / {r['elev_gain_m']:.0f} m | "
            f"{r['shortest_gain_m']-r['elev_gain_m']:.0f} m | "
            + (f"{eff:.2f} |" if np.isfinite(eff) else "n/a |"))
    L.append("")

    # --- and where it does not ------------------------------------------
    little, level = 5.0, 10.0
    none = flat[flat["gain_saved_m"] < little]
    already = none[none["shortest_gain_m"] < level]
    hard = none[none["shortest_gain_m"] >= level]
    line = (f"For {len(none):,} of the {len(flat):,} ordered pairs "
            f"({100*len(none)/len(flat):.0f}%) the flattest route avoids less "
            f"than {little:.0f} m of climbing. In {len(already):,} of those "
            f"the shortest route is already nearly level (under "
            f"{level:.0f} m of climb); in the other {len(hard):,} the climb "
            f"is there and the flattest route does not get round it.")
    least = _sweep_least(d)
    if least is not None and len(hard):
        # the same test against the least climbing found anywhere in the sweep
        key = pd.MultiIndex.from_arrays([hard["origin"], hard["destination"]])
        more = hard["shortest_gain_m"].to_numpy() - least.reindex(key).to_numpy()
        n_more = int(np.nansum(more >= little))
        line += (f" Nor, for {len(hard) - n_more:,} of them, does any route "
                 f"of the wider sweep of weights described further down"
                 + (f"; for the other {n_more:,} one of its routes does "
                    f"avoid {little:.0f} m or more ({np.nanmax(more):.0f} m "
                    f"at most)." if n_more else "."))
    L += ["### And where does it not?", "", line, ""]

    # per neighbourhood: every trip that starts or ends there
    ends = pd.concat([flat.assign(place=flat["origin"]),
                      flat.assign(place=flat["destination"])], ignore_index=True)
    by = ends.groupby("place").agg(
        shortest_gain=("shortest_gain_m", "mean"), gain=("elev_gain_m", "mean"),
        detour=("detour_ratio", "mean"), n=("elev_gain_m", "size"))
    by["avoided"] = 100 * (1 - by["gain"] / by["shortest_gain"].where(
        by["shortest_gain"] > 0))
    by = by.dropna(subset=["avoided"]).sort_values("avoided", ascending=False)
    if len(by) >= 6:
        k = min(5, len(by) // 2)
        L += ["Neighbourhood by neighbourhood, over every trip that starts or "
              f"ends there, the {k} that gain most from flat routing and the "
              f"{k} that gain least:", "",
              "| Neighbourhood | Landmass | Mean climb, shortest | Mean climb, "
              "flattest | Climbing avoided | Extra distance |",
              "|---|---|---|---|---|---|"]
        for name, r in pd.concat([by.head(k), by.tail(k)]).iterrows():
            L.append(f"| {name} | {d['lm_of'].get(name, '')} | "
                     f"{r['shortest_gain']:.0f} m | {r['gain']:.0f} m | "
                     f"{r['avoided']:.0f}% | {100*(r['detour']-1):+.0f}% |")
        L.append("")
    return L


# --------------------------------------------------------------------------
def _corridor_landmass(d, cor: pd.DataFrame) -> pd.Series:
    """The landmass each walking corridor is on.

    The corridor table says so itself (``landmass``, the component rank the
    corridor was detected on) when it comes from ``detect_corridors`` and
    the landmasses here are keyed the same way.  A table without that
    column is placed by the planning areas each corridor passes through,
    on the landmass most of them are on; a corridor that touches none goes
    with the neighborhood whose polygon it lies nearest to.
    """
    import geopandas as gpd

    lm_of = d["lm_of"]
    known = [None] * len(cor)
    if d.get("lm_by_component") and "landmass" in cor.columns:
        by_key = {m["key"]: m["name"] for m in d["lm"]}
        known = [by_key.get(int(k)) if pd.notna(k) else None
                 for k in cor["landmass"]]
    out = []
    loose = []
    for i, v in enumerate(cor["neighborhoods"] if "neighborhoods" in cor.columns
                          else [None] * len(cor)):
        lms = [lm_of[n] for n in _text(v).split("; ") if n in lm_of]
        if known[i] is not None:
            out.append(known[i])
        elif lms:
            out.append(pd.Series(lms).mode().iloc[0])
        else:
            out.append(None)
            loose.append(i)
    if loose and len(lm_of):
        nb = d["neighborhoods"]
        nb = nb[nb["neighborhood"].isin(lm_of)][["neighborhood", "geometry"]]
        part = gpd.GeoDataFrame(geometry=cor.geometry.iloc[loose].values, crs=cor.crs)
        if nb.crs != part.crs:
            nb = nb.to_crs(part.crs)
        near = gpd.sjoin_nearest(part, nb, how="left")
        near = near[~near.index.duplicated()]
        for j, i in enumerate(loose):
            out[i] = lm_of.get(near["neighborhood"].get(j))
    return pd.Series(out, index=cor.index, dtype="object")


def _through(v, limit: int = 3) -> str:
    return _list([n for n in _text(v).split("; ") if n], limit=limit)


def _corridor_kinds(walk: pd.DataFrame) -> pd.Series:
    """What ground each corridor is on: 'low', 'contour', 'over' or ''.

    ``low`` lies wholly below the lowland threshold.  The two high kinds
    never drop below ``HIGH_CORRIDOR_M``: ``contour`` holds a mean gradient
    of ``LEVEL_GRADE`` or less, a level road kept up on a hillside, and
    ``over`` does not, which makes it the gentlest way up or across high
    ground and not a flat street at all.  Everything else is in between.
    """
    from .passes import BASIN_ELEV_M
    high = walk["elev_min_m"] >= HIGH_CORRIDOR_M
    level = walk["mean_abs_grade"] <= LEVEL_GRADE
    kind = pd.Series("", index=walk.index, dtype="object")
    kind[walk["elev_max_m"] <= BASIN_ELEV_M] = "low"
    kind[high & level] = "contour"
    kind[high & ~level] = "over"
    return kind


def _corridors(d) -> list[str]:
    c = d["corridors"]
    title = "## The corridors flat routes use"
    if c.empty or "mode" not in c.columns:
        return [title, "", "No corridor met the thresholds in this run.", ""]
    walk = c[c["mode"] == "walk"].sort_values("total_score", ascending=False).copy()
    walk["landmass"] = _corridor_landmass(d, walk)
    walk["kind"] = _corridor_kinds(walk)
    e = d["edges"]
    walkable = e[e["walk_ok"]] if "walk_ok" in e.columns else e
    net_gain = walkable["cum_gain_fwd"].sum() / (walkable["length_m"].sum() / 1000)
    L = [title, "",
         "These were *discovered*, not listed: the analysis aggregated how "
         "often each street segment carried a climb-averse route between "
         "neighbourhoods, weighted by the climbing those routes avoided, and "
         "merged the high-scoring segments into contiguous corridors. No "
         "corridor was named in advance. A corridor is named by the streets "
         "that make up most of it, and it can only serve pairs on its own "
         "landmass. A street qualifies by being used and by being gentle "
         f"itself (a mean gradient of {ANALYSIS.corridor_max_avg_grade:.1%} "
         "at most), not by being low: a corridor may be on low ground, "
         "a level road high on a hillside, or simply the least steep way "
         "over a ridge, and the last part of this section says how many "
         "are which. Climb per km is the mean of the two directions of "
         f"travel. This run found {len(walk)} walking corridors, "
         f"{walk['length_km'].sum():.0f} km in all.", ""]
    note = {"contour": ", a contour road", "over": ", a way over high ground"}
    for m in d["lm"]:
        sub = walk[walk["landmass"] == m["name"]]
        if sub.empty:
            continue
        L += [f"### {m['name']}", "",
              f"{len(sub)} corridors, {sub['length_km'].sum():.1f} km; the "
              f"{min(TOP_PER_LANDMASS, len(sub))} most important:", "",
              "| Corridor | Length | Mean grade | Climb per km | Pairs served "
              "| Neighbourhoods linked | Elevation | Runs through |",
              "|---|---|---|---|---|---|---|---|"]
        for _, r in sub.head(TOP_PER_LANDMASS).iterrows():
            L.append(
                f"| {r['corridor_name']} | {r['length_km']:.1f} km | "
                f"{r['mean_abs_grade']:.1%} | {r['gain_per_km']:.1f} m | "
                f"{int(r['pair_count_max'])} | {int(r['neighborhood_span'])} | "
                f"{r['elev_min_m']:.0f}-{r['elev_max_m']:.0f} mPD"
                f"{note.get(r['kind'], '')} | "
                f"{_through(r.get('neighborhoods')) or '(no planning area)'} |")
        L.append("")
        top = sub.iloc[0]
        line = (f"**{top['corridor_name']}** ({top['length_km']:.1f} km, "
                f"{top['mean_abs_grade']:.1%} mean gradient) is the most "
                f"important {'corridor' if top['kind'] == 'over' else 'flat corridor'} "
                f"on {m['name']}. Its busiest block carries a climb-averse "
                f"route for {int(top['pair_count_max'])} of the landmass's "
                f"{m['routed']:,} ordered pairs")
        if "flat_routes_saved_m" in sub.columns and pd.notna(top["flat_routes_saved_m"]):
            line += (f". The flattest routes of {int(top['flat_routes']):,} "
                     f"ordered pairs run along some part of it, and between "
                     f"them those routes climb "
                     f"{top['flat_routes_saved_m']/1000:.1f} km less than "
                     f"the shortest routes for the same pairs.")
        else:
            # a corridor table from before the routes were counted: all it
            # has is the score, which is a share and not a total
            line += (f", and its share of the climbing those routes avoid, "
                     f"apportioned by length over the three climb-averse "
                     f"objectives, comes to {top['climb_saved_m']/1000:.1f} km.")
        line += (f" It stays between {top['elev_min_m']:.0f} and "
                 f"{top['elev_max_m']:.0f} mPD"
                 + (f" and runs through {_through(top.get('neighborhoods'), 6)}."
                    if _text(top.get("neighborhoods")) else "."))
        L += [line, ""]
        if len(sub) > 1:
            sec = sub.iloc[1]
            L += [f"**{sec['corridor_name']}** ({sec['length_km']:.1f} km, "
                  f"{sec['mean_abs_grade']:.1%}, "
                  f"{int(sec['pair_count_max'])} pairs, "
                  f"{sec['elev_min_m']:.0f}-{sec['elev_max_m']:.0f} mPD) is "
                  f"the second.", ""]

    lost = walk[walk["landmass"].isna()]
    if len(lost):
        L += [f"{len(lost)} further corridors could not be placed on a "
              f"landmass with neighbourhoods.", ""]

    # --- what kind of ground the corridors are on -----------------------
    from .passes import BASIN_ELEV_M
    low = walk[walk["kind"] == "low"]
    level = walk[walk["kind"] == "contour"]
    over = walk[walk["kind"] == "over"]
    n_high = len(level) + len(over)
    L += ["### Low ground, contour roads and ways over", "",
          f"For scale: the walkable network as a whole climbs "
          f"{net_gain:.1f} m per kilometre of street, and these corridors "
          f"{walk['gain_per_km'].median():.1f} m at the median. "
          f"{len(low)} of the {len(walk)} lie wholly below "
          f"{BASIN_ELEV_M:.0f} mPD, {low['length_km'].sum():.0f} km of the "
          f"{walk['length_km'].sum():.0f} km: flat because the ground is."
          + (f" {n_high} never drop below {HIGH_CORRIDOR_M:.0f} mPD. "
             f"{len(level)} of those hold a mean gradient of "
             f"{LEVEL_GRADE:.0%} or less: level routes kept high on a "
             f"hillside rather than on low ground." if n_high else ""), ""]

    def table(part) -> list[str]:
        out = ["| Corridor | Landmass | Length | Mean grade | Climb per km | "
               "Elevation | Pairs served |", "|---|---|---|---|---|---|---|"]
        for _, r in part.head(8).iterrows():
            out.append(f"| {r['corridor_name']} | {_text(r['landmass'])} | "
                       f"{r['length_km']:.1f} km | {r['mean_abs_grade']:.1%} | "
                       f"{r['gain_per_km']:.1f} m | "
                       f"{r['elev_min_m']:.0f}-{r['elev_max_m']:.0f} mPD | "
                       f"{int(r['pair_count_max'])} |")
        return out + [""]

    if len(level):
        L += table(level)
    if len(over):
        one = len(over) == 1
        lo, hi = over["mean_abs_grade"].min(), over["mean_abs_grade"].max()
        L += [f"The other {len(over)} {'is' if one else 'are'} neither "
              f"level nor low. {'Its' if one else 'Their'} mean gradient is "
              + (f"{lo:.1%}" if one else f"{lo:.1%} to {hi:.1%}")
              + f", and {'it is' if one else 'they are'} in the list "
              f"because the climb-averse routes between the places "
              f"{'it joins' if one else 'they join'} use "
              f"{'it' if one else 'them'}: {'it is' if one else 'each is'} "
              f"the gentlest way up or over high ground that those routes "
              f"found, which is a different thing from a flat street.", ""]
        L += table(over)
    return L


# --------------------------------------------------------------------------
def _sides(pm: pd.DataFrame, edge_id) -> tuple[list, list]:
    """The two groups of neighborhoods a pass stands between.

    Every pair bound by the pass has one end on each side, so the pairs
    form a complete two-sided graph: one side is everything paired with
    any one neighborhood, the other side is the rest.  The smaller side is
    returned first.
    """
    rows = pm[pm["pass_edge_id"] == edge_id]
    if rows.empty:
        return [], []
    first = rows["neighborhood_a"].iloc[0]
    b = set(rows.loc[rows["neighborhood_a"] == first, "neighborhood_b"]) \
        | set(rows.loc[rows["neighborhood_b"] == first, "neighborhood_a"])
    a = (set(rows["neighborhood_a"]) | set(rows["neighborhood_b"])) - b
    a, b = sorted(a), sorted(b)
    return (a, b) if len(a) <= len(b) else (b, a)


def _separates(a: list, b: list, z_of: dict, above: dict | None = None) -> str:
    """'X (at 12 mPD) from the other 18', 'X (at 2 mPD) from Y', 'A and B from 5 others'.

    ``above`` holds the neighbourhoods whose access point stands well above
    their own streets (``_points_above_streets``); for one of those the
    height is given as the point's, with what it stands above.
    """
    if not a or not b:
        return ""
    if len(a) == 1:
        z = z_of.get(a[0])
        at = f" (at {z:.0f} mPD)" if z is not None and np.isfinite(z) else ""
        if above and a[0] in above:
            z, med = above[a[0]]
            at = (f" (its access point at {z:.0f} mPD, its streets at "
                  f"{med:.0f} mPD)")
        # a landmass of two neighbourhoods has one pass, between the two
        rest = b[0] if len(b) == 1 else f"the other {len(b)}"
        return f"{a[0]}{at} from {rest}"
    small = _list(a) if len(a) == 2 else f"{len(a)} neighbourhoods"
    return f"{small} from {len(b)} others"


def _points_above_streets(d) -> dict:
    """{neighbourhood: (access point mPD, median street mPD)} for the access
    points that stand more than ``ACCESS_ABOVE_STREETS_M`` above the median
    height of their neighbourhood's ordinary streets.

    Access points are chosen within that distance of the median wherever an
    intersection qualifies there, so this is empty unless the rule had
    nothing to choose from; a points file written before the heights were
    recorded says nothing either way.
    """
    pts = d.get("points_walk")
    if pts is None or not {"elev_m", "street_median_m"} <= set(pts.columns):
        return {}
    gap = pts["elev_m"] - pts["street_median_m"]
    over = pts[pts["node"].notna() & (gap > ACCESS_ABOVE_STREETS_M)]
    return {n: (float(z), float(m)) for n, z, m in
            zip(over["neighborhood"], over["elev_m"], over["street_median_m"])}


def _pass_kinds(sub, pm: pd.DataFrame, z_of: dict,
                above: dict | None = None) -> list[dict]:
    """Classify each pass of one landmass by what it does.

    The minimax tree adds neighborhoods to the lowland one at a time, in
    order of the height each must reach to join the rest, so most passes
    have a single neighborhood on one side.  Whether such a pass is worth
    calling one depends on where that neighborhood itself sits:

    ``hill``    the pass is within ``HILLTOP_M`` of the neighborhood's own
                access point -- the neighborhood is on a hill and the
                "pass" is its doorstep;
    ``point``   the same, where the access point itself stands well above
                the neighborhood's streets (``above``) -- the height is
                that of the point chosen, not of the place;
    ``ridge``   the pass stands well above the access point -- the
                neighborhood is low, and a ridge lies in front of it;
    ``divide``  both sides hold more than one neighborhood.
    """
    above = above or {}
    out = []
    for _, r in sub.iterrows():
        x, y = _sides(pm, r["edge_id"])
        rec = {"row": r, "a": x, "b": y, "lone": None, "above": float("nan")}
        if len(x) == 1:
            z = z_of.get(x[0], float("nan"))
            rec["lone"] = x[0]
            rec["above"] = float(r["pass_elev_m"] - z)
            rec["kind"] = ("ridge" if rec["above"] > HILLTOP_M else
                           "point" if x[0] in above else "hill")
        else:
            rec["kind"] = "divide"
        out.append(rec)
    return out


def _low_route_sentence(rows: pd.DataFrame, short_m: dict, height: float) -> str:
    """How far the way that keeps to a pass's height is, for the pairs it binds.

    A lowest crossing is a statement about height alone.  For a divide it
    can describe a route nobody would take: the lowest way between two
    districts either side of a ridge may be right round it.  ``rows`` are
    the divide's rows of the pass matrix, with ``low_route_m`` (the length
    of the shortest route that never rises above the pass) where the run
    worked it out; ``short_m`` maps an ordered pair to its shortest
    distance.  Empty when either is missing.
    """
    if "low_route_m" not in rows.columns:
        return ""
    t = rows.assign(shortest=[short_m.get((a, b), np.nan) for a, b in
                              zip(rows["neighborhood_a"], rows["neighborhood_b"])])
    t = t.dropna(subset=["low_route_m", "shortest"])
    if t.empty:
        return ""
    far = t.loc[(t["low_route_m"] - t["shortest"]).idxmax()]
    return (f" That is a statement about height, not about a route anyone "
            f"would take. For the {len(t):,} pairs it divides, the shortest "
            f"route that never rises above {height:.0f} mPD is "
            f"{_km(t['low_route_m'].median()):.0f} km at the median, where "
            f"their shortest route is {_km(t['shortest'].median()):.0f} km; "
            f"from {far['neighborhood_a']} to {far['neighborhood_b']} it is "
            f"{_km(far['low_route_m']):.0f} km against "
            f"{_km(far['shortest']):.0f} km.")


def _passes(d) -> list[str]:
    pz, pm = d["passes"], d["pass_matrix"]
    L = ["## Passes, saddles and barriers", "",
         "The question \"how much climbing is unavoidable between these two "
         "parts of the city?\" is a **minimax** problem, not a shortest-path "
         "one: what matters is the lowest summit you can possibly cross. "
         "Solving it over a minimum bottleneck spanning tree gives, for every "
         "pair of neighbourhoods on the same landmass, the exact elevation "
         "of the lowest available crossing and the block on which it "
         "happens. Pairs on different landmasses have no crossing at all and "
         "are not in the table.", ""]
    if pm.empty or pz.empty:
        return L + ["No pair of neighbourhoods in this run has a pass "
                    "between them.", ""] + _barriers(d) + _basins(d)

    # access-point heights tell a ridge from a neighbourhood on a hill
    walk = d["pairs"][d["pairs"]["mode"] == "walk"]
    z_of = walk.groupby("origin")["start_elev_m"].first().to_dict()
    pm = pm.assign(landmass=pm["neighborhood_a"].map(d["lm_of"]))
    ends = np.maximum(pm["neighborhood_a"].map(z_of).to_numpy("float64"),
                      pm["neighborhood_b"].map(z_of).to_numpy("float64"))
    # how far the lowest crossing rises above the higher end of the trip
    pm["rise"] = (pm["pass_elev_m"].to_numpy("float64") - ends).clip(min=0.0)

    worst = pm.loc[pm["pass_elev_m"].idxmax()]
    flat_share = 100 * (pm["rise"] <= RISE_TOLERANCE_M).mean()
    above = _points_above_streets(d)
    pts = d.get("points_walk")
    if pts is not None and "street_median_m" in pts.columns:
        held = (f" An access point is taken within "
                f"{ACCESS_ABOVE_STREETS_M:.0f} m of the median height of its "
                f"neighbourhood's ordinary streets wherever an intersection "
                f"qualifies there, so an end that high is where the "
                f"neighbourhood's streets are")
        held += (f"; {len(above)} could not be and "
                 f"stand{'s' if len(above) == 1 else ''} higher "
                 f"({_list(sorted(above), limit=6)})." if above else ".")
    else:
        held = ""
    # the shortest walking distance of each pair, for the detour a low
    # crossing costs (the same either way: a walker ignores one-way rules)
    short = _profile(walk, "shortest")
    short_m = dict(zip(zip(short["origin"], short["destination"]),
                       short["distance_m"]))
    L += [f"Across the {len(pm):,} pairs of neighbourhoods that can be walked "
          f"between, the lowest possible crossing averages "
          f"{pm['pass_elev_m'].mean():.0f} mPD and reaches "
          f"{worst['pass_elev_m']:.0f} mPD at worst "
          f"({worst['neighborhood_a']} to {worst['neighborhood_b']}). Only "
          f"{len(pz)} distinct street segments act as the binding constraint "
          f"for any pair -- the real passes.", "",
          f"{'Most' if flat_share >= 50 else 'Much'} of that height is not "
          f"a ridge in the way but the neighbourhoods themselves. For "
          f"{flat_share:.0f}% of the pairs the "
          f"lowest crossing is within {RISE_TOLERANCE_M:.0f} m of the higher "
          f"of the two access points: the unavoidable summit is one end of "
          f"the trip. Over all pairs the crossing stands "
          f"{pm['rise'].mean():.0f} m above the higher end on average."
          + held, ""]

    lm_pass = pz["edge_id"].map(pm.groupby("pass_edge_id")["landmass"].first())
    for m in d["lm"]:
        sub_m = pm[pm["landmass"] == m["name"]]
        sub = pz[(lm_pass == m["name"]).to_numpy()]
        if sub_m.empty or sub.empty:
            continue
        hi = sub_m.loc[sub_m["pass_elev_m"].idxmax()]
        top_rise = sub_m.loc[sub_m["rise"].idxmax()]
        blocked = int((sub_m["rise"] > RISE_TOLERANCE_M).sum())
        L += [f"### {m['name']}", "",
              f"{len(sub_m):,} pairs, {len(sub)} passes. The lowest crossing "
              f"averages {sub_m['pass_elev_m'].mean():.0f} mPD and is highest "
              f"between {hi['neighborhood_a']} and {hi['neighborhood_b']}, at "
              f"{hi['pass_elev_m']:.0f} mPD. {blocked:,} of the pairs "
              f"({100*blocked/len(sub_m):.0f}%) have a ridge between them "
              f"that rises more than {RISE_TOLERANCE_M:.0f} m above both "
              f"ends"
              + (f", by {top_rise['rise']:.0f} m at most "
                 f"({top_rise['neighborhood_a']} to "
                 f"{top_rise['neighborhood_b']})." if blocked else "."), "",
              "| Pass | Where | Lowest possible crossing | Pairs forced over "
              "it | Gradient there | What it separates |",
              "|---|---|---|---|---|---|"]
        kinds = _pass_kinds(sub, pm, z_of, above)
        for k in kinds[:TOP_PER_LANDMASS]:
            r = k["row"]
            L.append(f"| {_way_label(r)} | {_where(r)} | "
                     f"{r['pass_elev_m']:.0f} mPD | {int(r['pairs_served'])} | "
                     f"{r['max_abs_grade']:.1%} | "
                     f"{_separates(k['a'], k['b'], z_of, above)} |")
        L.append("")

        # --- the pass that binds most pairs -----------------------------
        k = kinds[0]
        top = k["row"]
        line = (f"The pass that binds most pairs on {m['name']} is "
                f"**{_way_label(top)}**, {_where(top)}, at "
                f"{top['pass_elev_m']:.0f} mPD: the constraint for "
                f"{int(top['pairs_served'])} of the {len(sub_m):,} pairs "
                f"({100*top['pairs_served']/len(sub_m):.0f}%)")
        if k["kind"] == "hill":
            line += (f", every one that involves {k['lone']}, whose access "
                     f"point is itself at {z_of[k['lone']]:.0f} mPD. This is "
                     f"less a saddle than the height of the neighbourhood: "
                     f"whoever goes there climbs to it whatever the route.")
        elif k["kind"] == "point":
            z, med = above[k["lone"]]
            line += (f", every one that involves {k['lone']}. Its access "
                     f"point is at {z:.0f} mPD, {z - med:.0f} m above the "
                     f"median height of the neighbourhood's own streets "
                     f"({med:.0f} mPD), so this is the height of the point "
                     f"chosen for it rather than of the neighbourhood: most "
                     f"of {k['lone']} lies lower, and a trip to its lower "
                     f"streets need not come up here.")
        elif k["kind"] == "ridge":
            line += (f", every one that involves {k['lone']}. Its access "
                     f"point is at {z_of[k['lone']]:.0f} mPD, "
                     f"{k['above']:.0f} m lower: the neighbourhood lies "
                     f"behind a ridge, and every way in goes over it.")
        else:
            line += (f", every one between {_list(k['a'], limit=6)} on one "
                     f"side and {_list(k['b'], limit=6)} on the other. "
                     f"Anyone going between the two groups reaches that "
                     f"height whatever route they choose; where it crosses, "
                     f"the gradient is {top['max_abs_grade']:.1%}.")
        L += [line, ""]

        # --- hills, ridges and divides ----------------------------------
        n = {kind: sum(1 for k in kinds if k["kind"] == kind)
             for kind in ("hill", "point", "ridge", "divide")}
        L += [f"The {len(kinds)} passes are of three kinds. {n['hill']}: "
              f"the doorstep of a neighbourhood that stands higher than "
              f"the ground between it and the rest (the pass is within "
              f"{HILLTOP_M:.0f} m of its access point). "
              f"{n['ridge']}: a ridge standing in front of a single lower "
              f"neighbourhood. {n['divide']}: a divide between two groups of "
              f"more than one neighbourhood each."
              + (f" The other {n['point']} "
                 f"{'is' if n['point'] == 1 else 'are'} the doorstep of an "
                 f"access point and not of a neighbourhood: "
                 + _list(sorted(k["lone"] for k in kinds
                                if k["kind"] == "point"), limit=6)
                 + f", whose point{'s stand' if n['point'] > 1 else ' stands'} "
                 f"more than {ACCESS_ABOVE_STREETS_M:.0f} m above the median "
                 f"height of {'their' if n['point'] > 1 else 'its'} own "
                 f"streets." if n["point"] else ""), ""]
        ridges = sorted((k for k in kinds if k["kind"] == "ridge"),
                        key=lambda k: -k["above"])
        if ridges:
            L += ["The ridges, highest above the neighbourhood they shut in "
                  "first:", "",
                  "| Neighbourhood | Its access point | Lowest way in or out "
                  "| On | Where | Climb that cannot be avoided |",
                  "|---|---|---|---|---|---|"]
            for k in ridges[:6]:
                r = k["row"]
                L.append(f"| {k['lone']} | {z_of[k['lone']]:.0f} mPD | "
                         f"{r['pass_elev_m']:.0f} mPD | {_way_label(r)} | "
                         f"{_where(r)} | {k['above']:.0f} m |")
            L.append("")
        divides = sorted((k for k in kinds if k["kind"] == "divide"),
                         key=lambda k: -int(k["row"]["pairs_served"]))
        if divides:
            k = divides[0]
            r = k["row"]
            line = (f"The divide that binds most pairs is {_way_label(r)}, "
                    f"{_where(r)}, at {r['pass_elev_m']:.0f} mPD: "
                    f"{_list(k['a'], limit=6)} lie on one side of it and "
                    f"{_list(k['b'], limit=6)} on the other.")
            line += _low_route_sentence(pm[pm["pass_edge_id"] == r["edge_id"]],
                                        short_m, float(r["pass_elev_m"]))
            L += [line, ""]

    L += _barriers(d)
    L += _basins(d)
    return L


def _barriers(d) -> list[str]:
    b = d["barriers"]
    if b.empty or "unavoidability" not in b.columns:
        return []
    un = pd.to_numeric(b["unavoidability"], errors="coerce").fillna(0.0)
    hard, soft = 0.5, 0.15
    off = b["cls"].isin(OFF_STREET) if "cls" in b.columns \
        else pd.Series(False, index=b.index)
    L = ["### Barriers with an alternative, and barriers without", "",
         "A steep street or path that carries heavy shortest-path traffic "
         "but almost none once climbing is penalised has a flat alternative "
         "nearby. One that keeps its traffic under every objective does "
         f"not. Of the {len(b):,} highest-scoring barrier segments kept, "
         f"{int((un < soft).sum()):,} are avoidable (the flat routes keep "
         f"under {soft:.0%} of the traffic), {int((un > hard).sum()):,} are "
         f"unavoidable (over {hard:.0%}) and the rest fall between. "
         f"{int(off.sum()):,} of the {len(b):,} are not streets but "
         f"footpaths, hill paths, tracks and flights of steps, which is "
         f"where a distance-minimising route does its steepest climbing; "
         f"{int((off & (un < soft)).sum()):,} of those are avoidable.", "",
         "| Street or path | Where | Gradient | Pairs via shortest route | "
         "Still via the flat route | Verdict |",
         "|---|---|---|---|---|---|"]
    top = b.assign(_un=un).nlargest(10, "barrier_score")
    for _, r in top.iterrows():
        u = float(r["_un"])
        verdict = ("**unavoidable**" if u > hard else
                   "avoidable" if u < soft else "partly avoidable")
        flat_use = pd.to_numeric(r.get("flat_use_per_objective"), errors="coerce")
        L.append(f"| {_way_label(r)} | {_where(r)} | "
                 f"{r['max_abs_grade']:.1%} | {int(r['shortest_use'])} | "
                 f"{0 if pd.isna(flat_use) else flat_use:.0f} | {verdict} |")
    L.append("")
    worst = b.assign(_un=un)[un > hard].nlargest(5, "barrier_score")
    if len(worst):
        # ranked by barrier score (pairs x gradient x length), like the table,
        # so the order below is not simply the order of the pair counts
        L += ["The highest-scoring of the unavoidable ones: "
              + "; ".join(f"{_way_label(r)}, {_where(r)} "
                          f"({r['max_abs_grade']:.0%}, "
                          f"{int(r['shortest_use'])} pairs)"
                          for _, r in worst.iterrows()) + ".", ""]
    return L


def _basins(d) -> list[str]:
    bs = d.get("basins")
    if bs is None or bs.empty:
        return []
    from .passes import BASIN_MIN_KM
    thr = float(bs["threshold_m"].iloc[0])
    bs = bs.sort_values("length_km", ascending=False)
    L = ["### The lowland basins", "",
         f"Street below {thr:.0f} mPD forms {len(bs)} separate basins of at "
         f"least {BASIN_MIN_KM:.0f} km, {bs['length_km'].sum():,.0f} km in "
         f"all: the flat districts, each named here by the neighbourhoods "
         f"with most of its street. The largest:", "",
         "| Basin | Street below the threshold |", "|---|---|"]
    for _, r in bs.head(8).iterrows():
        L.append(f"| {_text(r['label']) or '(no planning area)'} | "
                 f"{r['length_km']:,.0f} km |")
    L.append("")
    bp = d.get("basin_passes")
    if bp is not None and len(bp) and "pass_elev_m" in bp.columns:
        bp = bp.sort_values("pass_elev_m")
        km_of = dict(zip(bs["basin"].astype(int), bs["length_km"]))
        label_of = dict(zip(bs["basin"].astype(int), bs["label"]))

        def side(v) -> str:
            """A group of basins, named by the largest one in it."""
            ids = [int(float(x)) for x in str(v).split(",") if x.strip()]
            ids = [i for i in ids if i in km_of]
            if not ids:
                return ""
            lead = max(ids, key=lambda i: km_of[i])
            name = _text(label_of[lead]) or "(no planning area)"
            return name if len(ids) == 1 else f"{name} and {len(ids) - 1} smaller"

        L += ["Leaving one basin for the next means climbing at least this "
              "high. Each row is the crest at which two groups of basins "
              "first join, lowest first:", "",
              "| Lowest crossing | On | Where | Joins | to |",
              "|---|---|---|---|---|"]
        for _, r in bp.head(10).iterrows():
            L.append(f"| {r['pass_elev_m']:.0f} mPD | {_way_name(r)} | "
                     f"{_where(r)} | {side(r.get('basins_a'))} | "
                     f"{side(r.get('basins_b'))} |")
        L.append("")
    return L


# --------------------------------------------------------------------------
def _frontier_rows(g: pd.DataFrame) -> list[str]:
    """Table rows for one pair's frontier of distance against climbing.

    ``g`` is the pair's rows of the frontier file, which is non-dominated
    over three objectives (distance, climbing, steepest gradient).  Sorted
    by distance, such rows can climb *more* as they get longer -- they are
    there for their gradient -- and a table of them reads as a mistake.  The
    rows kept here are the ones no other beats on both distance and
    climbing, judged on the figures as printed: two routes 3 m and a
    decimetre of climb apart print as the same row twice, or as a longer
    route that saves nothing.
    """
    g = g.sort_values(["distance_m", "elev_gain_m"])
    rows: list[tuple[str, float, str]] = []
    for _, row in g.iterrows():
        dist = f"{_km(row['distance_m']):.2f} km"
        climb = float(f"{row['elev_gain_m']:.0f}")
        cell = f"| {dist} | {climb:.0f} m | {row['max_grade']:.0%} |"
        if rows and climb >= rows[-1][1]:
            continue                        # longer, and no less climbing
        if rows and dist == rows[-1][0]:
            rows.pop()                      # as short, and more climbing
        rows.append((dist, climb, cell))
    return [cell for _, _, cell in rows]


def _pareto(d) -> list[str]:
    if "pareto" not in d or d["pareto"].empty:
        return []
    pa = d["pareto"]
    pa = pa[(pa["mode"] == "walk") & pa["pareto_optimal"]]

    # for every pair, how much detour does halving the climb cost?
    rows = []
    for (o, dst), g in pa.groupby(["origin", "destination"]):
        g = g.sort_values("distance_m")
        s = g.iloc[0]                       # the pure-distance anchor
        if s["elev_gain_m"] <= 0:
            continue
        half = g[g["elev_gain_m"] <= 0.5 * s["elev_gain_m"]]
        best = s["elev_gain_m"] - g["elev_gain_m"].min()
        far = g.loc[g["elev_gain_m"].idxmin(), "distance_m"] - s["distance_m"]
        near = g[g["distance_m"] <= s["distance_m"] + 0.5 * far]
        rows.append({
            "landmass": d["lm_of"].get(o),
            "halvable": len(half) > 0,
            "detour_to_halve": (half["distance_m"].min() / s["distance_m"] - 1)
            if len(half) else np.nan,
            "best_saved_pct": 100 * (1 - g["elev_gain_m"].min() / s["elev_gain_m"]),
            "best_detour": g.loc[g["elev_gain_m"].idxmin(), "distance_m"]
            / s["distance_m"] - 1,
            # of the flattest route's saving, the share that half of its
            # extra distance already buys (pairs with a saving worth the name)
            "half_share": (s["elev_gain_m"] - near["elev_gain_m"].min()) / best
            if best >= RISE_TOLERANCE_M and far > 0 else np.nan,
        })
    r = pd.DataFrame(rows)
    n_pairs = pa.groupby(["origin", "destination"]).ngroups

    top = max(PARETO_LAMBDA_SWEEP)
    L = ["## The distance / climbing trade-off", "",
         f"For every one of the {n_pairs:,} ordered pairs, a single weight "
         f"scales the climbing and gradient terms of the balanced objective "
         f"from zero (pure distance) to {top:g} times their size, and the "
         f"strongly climb-averse objective of the tables above is added as "
         f"one more point. The sweep runs past that objective: at its far "
         f"end a metre of climb is priced at "
         f"{top * ROUTING_PROFILES['balanced'].alpha:.0f} m of walking, "
         f"against {ROUTING_PROFILES['min_climb'].alpha:.0f} m. The routes "
         f"found trace the frontier between distance, cumulative climbing "
         f"and peak gradient, and \"the least-climbing route\" below is the "
         f"one with the least climbing anywhere in the sweep. The useful "
         f"question is where the knee is: how much detour buys how much of "
         f"the climbing.", ""]

    def lines(r, who: str) -> list[str]:
        out = [f"- **{100*r['halvable'].mean():.0f}% of {who} can halve "
               f"their climbing** by some route"]
        if r["halvable"].any():
            out[0] += (f", and the median detour that costs is "
                       f"**{100*r['detour_to_halve'].median():.0f}%**. "
                       f"{100*(r['detour_to_halve'] <= 0.10).mean():.0f}% of "
                       f"all {who} can halve it within a 10% detour, "
                       f"{100*(r['detour_to_halve'] <= 0.20).mean():.0f}% "
                       f"within 20%.")
        else:
            out[0] += "."
        out.append(f"- Taken to the least-climbing route, the median pair "
                   f"sheds **{r['best_saved_pct'].median():.0f}%** of its "
                   f"climbing for a median "
                   f"**{100*r['best_detour'].median():.0f}%** more distance.")
        return out

    if len(r):
        L += lines(r, "pairs") + [""]
        lms = [m for m in _with_pairs(d) if (r["landmass"] == m["name"]).any()]
        if len(lms) > 1:
            L += ["| Landmass | Pairs that can halve their climbing | Median "
                  "detour to halve it | Median saving on the least-climbing "
                  "route | Median extra distance |", "|---|---|---|---|---|"]
            for m in lms:
                q = r[r["landmass"] == m["name"]]
                det = q["detour_to_halve"].median()
                L.append(f"| {m['name']} | {100*q['halvable'].mean():.0f}% | "
                         + (f"{100*det:.0f}%" if np.isfinite(det) else "n/a")
                         + f" | {q['best_saved_pct'].median():.0f}% | "
                         f"{100*q['best_detour'].median():.0f}% |")
            L.append("")

    # frontiers in detail: the first featured pairs of each landmass
    shown: dict = {}
    tables: list[str] = []
    for o, dst in FEATURED_PAIRS:
        lm = d["lm_of"].get(o)
        if shown.get(lm, 0) >= 2 or sum(shown.values()) >= 4:
            continue
        g = pa[(pa["origin"] == o) & (pa["destination"] == dst)]
        if g.empty:
            continue
        shown[lm] = shown.get(lm, 0) + 1
        tables += [f"**{o} to {dst}** ({lm})", "",
                   "| Distance | Climb | Steepest grade |", "|---|---|---|"]
        tables += _frontier_rows(g) + [""]
    if tables:
        L += ["The frontiers of a few pairs in detail. Each table lists, "
              "shortest first, the routes that no other route of the sweep "
              "beats on both distance and climbing, so climbing falls as "
              "distance rises. The frontier file keeps more: a route also "
              "stays in it for lowering the steepest gradient, even where "
              "it climbs more than a shorter one.", ""] + tables

    half = r["half_share"].dropna() if len(r) else pd.Series(dtype=float)
    if len(half):
        med = float(half.median())
        shape = ("The frontiers are concave: the first part of the extra "
                 "distance removes most of the climbing, and what follows "
                 "buys little." if med >= 0.6 else
                 "The frontiers are back-loaded: most of the saving comes "
                 "only with the last part of the detour." if med <= 0.4 else
                 "The frontiers are close to straight: climbing comes off "
                 "roughly in proportion to the extra distance, with no "
                 "sharp knee.")
        walk = d["pairs"][d["pairs"]["mode"] == "walk"]
        bal, flat = _profile(walk, "balanced"), _profile(walk, "min_climb")
        L += [f"How are the frontiers shaped? Take the least-climbing "
              f"route's extra distance and its saving as the whole of what "
              f"is on offer. Over the {len(half):,} pairs with at least "
              f"{RISE_TOLERANCE_M:.0f} m to save, half of that extra "
              f"distance buys {100*med:.0f}% of the saving at the median, "
              f"and more than half of it for "
              f"{100*(half > 0.5).mean():.0f}% of pairs. {shape}"
              + (f" The balanced objective takes "
                 f"{100*(bal['detour_ratio'].mean()-1):+.0f}% distance for "
                 f"{bal['gain_saved_pct'].mean():.0f}% of the climbing, "
                 f"against {100*(flat['detour_ratio'].mean()-1):+.0f}% for "
                 f"{flat['gain_saved_pct'].mean():.0f}% on the strongly "
                 f"climb-averse one, and it holds the mean steepest pitch to "
                 f"{bal['max_grade'].mean():.0%} against "
                 f"{flat['max_grade'].mean():.0%}."
                 if len(bal) and len(flat) else ""), ""]
    return L


def _modes(d) -> list[str]:
    p, e = d["pairs"], d["edges"]
    w = p[(p["mode"] == "walk") & (p["profile"] == "min_climb")]
    b = p[(p["mode"] == "bike") & (p["profile"] == "min_climb")]
    if w.empty or b.empty or "bike_ok" not in e.columns:
        return []
    steps_km = e[e["cls"] == "steps"]["length_m"].sum() / 1000
    walk_km = e[e["walk_ok"]]["length_m"].sum() / 1000
    # A pushed link is open to bicycles in the edge table so that a route
    # can use it, but nobody rides it: it is a footway the cyclist walks.
    pushed = (e["bike_push"].to_numpy(dtype=bool, na_value=False)
              if "bike_push" in e.columns else np.zeros(len(e), dtype=bool))
    ridden = e[e["bike_ok"].to_numpy(dtype=bool) & ~pushed]
    bike_km = ridden["length_m"].sum() / 1000
    push_km = e[pushed]["length_m"].sum() / 1000
    permit = sorted(BIKE_PERMIT_REQUIRED_CLASSES)
    in_permit = e[e["cls"].isin(permit)]
    pw = in_permit[in_permit["walk_ok"]]["length_m"].sum() / 1000
    pb = ridden[ridden["cls"].isin(permit)]["length_m"].sum() / 1000
    mult = BIKE.class_multiplier
    calm = min(mult, key=mult.get)
    harsh = max(mult, key=mult.get)
    L = ["## Walking is not cycling", "",
         f"The two networks are modelled separately, and they are not "
         f"interchangeable. Of the {walk_km:,.0f} km a walker may use, a "
         f"bicycle may be ridden on {bike_km:,.0f} km. The {steps_km:,.0f} "
         f"km of steps are a genuine part of the pedestrian network and no "
         f"use on a bicycle; the bicycle graph excludes them outright. Ways "
         f"of class {_list(permit)} are open to bicycles only where the "
         f"data permits them by name, because riding on a footpath is not "
         f"allowed: that keeps {pb:,.0f} km of the {pw:,.0f} km a walker "
         f"has of them."
         + (f" A further {push_km:,.0f} km of them, in runs of at most "
            f"{BIKE_PUSH_MAX_M:.0f} m that join two parts of the cycling "
            f"network, is open to a bicycle pushed on foot, at "
            f"{BIKE_PUSH_MULTIPLIER:g} times its length." if push_km else "")
         + f" Bicycle costs also carry stress weights by road "
         f"class (from {mult[calm]:g} of its length on a {calm} to "
         f"{mult[harsh]:g} on a {harsh} road) and respect one-way "
         f"restrictions, which pedestrians do not.", ""]

    key = ["origin", "destination"]
    both = w.merge(b, on=key, suffixes=("_w", "_b"))
    only_walk = len(w) - len(both)
    L += [f"On foot {len(w):,} ordered pairs have a route; by bicycle "
          f"{len(b):,}"
          + (f", so {only_walk:,} pairs can be walked but not ridden"
             if only_walk > 0 else "")
          + f". Over the {len(both):,} pairs both can make, the flattest "
          f"bicycle route averages {_km(both['distance_m_b'].mean()):.2f} km "
          f"and {both['elev_gain_m_b'].mean():.0f} m of climbing against "
          f"{_km(both['distance_m_w'].mean()):.2f} km and "
          f"{both['elev_gain_m_w'].mean():.0f} m on foot, a difference of "
          f"{100*(both['distance_m_b'].mean()/both['distance_m_w'].mean()-1):+.0f}% "
          f"in distance and "
          f"{100*(both['elev_gain_m_b'].mean()/both['elev_gain_m_w'].mean()-1):+.0f}% "
          f"in climbing.", ""]
    more = b[b["gain_saved_m"] < -0.5]
    if len(more):
        L += [f"By bicycle the flattest route is not always the one that "
              f"climbs least. Its objective weighs road class as well as "
              f"climbing, and for {len(more):,} of the {len(b):,} pairs it "
              f"climbs more than the shortest route does, by "
              f"{-more['gain_saved_m'].min():.0f} m at most: a calmer road "
              f"taken at the price of a little height.", ""]
    if len(both):
        gap = (both["distance_m_b"] / both["distance_m_w"])
        far = both.assign(_r=gap).nlargest(3, "_r")
        L += ["The pairs where the bicycle goes furthest out of the "
              "walker's way: "
              + "; ".join(f"{r['origin']} to {r['destination']} "
                          f"({_km(r['distance_m_b']):.1f} km against "
                          f"{_km(r['distance_m_w']):.1f} km)"
                          for _, r in far.iterrows()) + ".", ""]
    pb_pts = d.get("points_bike")
    if pb_pts is not None and "component" in pb_pts.columns:
        ok = pb_pts[pb_pts["node"].notna()]
        groups = ok.groupby("component")["neighborhood"].apply(list)
        walk_groups = len(d["lm"])
        if len(groups) != walk_groups:
            L += [f"The landmasses differ too: the neighbourhoods fall into "
                  f"{len(groups)} separate groups by bicycle against "
                  f"{walk_groups} on foot.", ""]
    return L


def _top_corridor_phrase(sd: pd.DataFrame) -> str:
    """'the top corridor is X in every run', or an honest count."""
    leads = sd["top_corridor"].fillna("").map(lambda v: v.split(" - ")[0])
    counts = leads.value_counts()
    top, n = counts.index[0], int(counts.iloc[0])
    if n == len(sd):
        return f"the top corridor is {top} in every run"
    return f"the top corridor is {top} in {n} of {len(sd)} runs"


def _robustness(d) -> list[str]:
    """Summarise the sensitivity analysis, if it has been run.

    The sensitivity harness chooses its own reference streets and may add
    or drop columns, so each row of the table is written only if the
    columns it needs are there.
    """
    path = OUTPUT_DIR / "sensitivity.csv"
    if not path.exists():
        return []
    sd = pd.read_csv(path, index_col="tag")
    if "baseline" not in sd.index or len(sd) < 2:
        return []
    base = sd.loc["baseline"]
    others = sd.drop(index="baseline")

    def has(*cols) -> bool:
        return all(c in sd.columns and others[c].notna().any() for c in cols)

    L = ["## How much of this depends on the modelling choices?", "",
         "Every figure above was recomputed with the whole pipeline rebuilt "
         f"under {len(others)} one-at-a-time changes to the elevation "
         "parameters and the choice of access intersection "
         "(`outputs/sensitivity.md` has the full tables).", "",
         "| Finding | Baseline | Range across all perturbations |",
         "|---|---|---|"]
    if has("min_climb_detour_pct"):
        L.append(f"| Flattest route: extra distance | {base['min_climb_detour_pct']:+.0f}% | "
                 f"{others['min_climb_detour_pct'].min():+.0f}% to "
                 f"{others['min_climb_detour_pct'].max():+.0f}% |")
    if has("min_climb_gain_saved_pct"):
        L.append(f"| Flattest route: climbing avoided | {base['min_climb_gain_saved_pct']:.0f}% | "
                 f"{others['min_climb_gain_saved_pct'].min():.0f}% to "
                 f"{others['min_climb_gain_saved_pct'].max():.0f}% |")
    if has("grade_averse_max_grade_pct"):
        L.append(f"| Grade-averse: mean steepest pitch | {base['grade_averse_max_grade_pct']:.1f}% | "
                 f"{others['grade_averse_max_grade_pct'].min():.1f}% to "
                 f"{others['grade_averse_max_grade_pct'].max():.1f}% |")
    if has("edge_overlap_pct"):
        L.append(f"| Corridor material shared with baseline (by length) | 100% | "
                 f"{others['edge_overlap_pct'].min():.0f}% to "
                 f"{others['edge_overlap_pct'].max():.0f}% |")
    if has("lead_streets_shared"):
        n_lead = (int(base["lead_streets_total"]) if has("lead_streets_total")
                  else int(sd["lead_streets_shared"].max()))
        L.append(f"| Lead streets of the top corridors kept | "
                 f"{n_lead} of {n_lead} | "
                 f"{int(others['lead_streets_shared'].min())} to "
                 f"{int(others['lead_streets_shared'].max())} of {n_lead} |")
    if has("top_pass_nbhd", "top_pass_m"):
        L.append(f"| Dominant pass | {base['top_pass_nbhd']}, {base['top_pass_m']:.0f} mPD | "
                 f"same location in {int((sd['top_pass_nbhd'] == base['top_pass_nbhd']).sum())} "
                 f"of {len(sd)} runs; {others['top_pass_m'].min():.0f}-"
                 f"{others['top_pass_m'].max():.0f} mPD |")
    if has("gap_pass_m"):
        pub = (f" (published {base['gap_published_m']:.0f} mPD)"
               if has("gap_published_m") and pd.notna(base.get("gap_published_m"))
               else "")
        L.append(f"| Reference gap: lowest crossing{pub} | "
                 f"{base['gap_pass_m']:.1f} mPD | {others['gap_pass_m'].min():.1f}-"
                 f"{others['gap_pass_m'].max():.1f} mPD |")
    if has("trip_excess_flat_m", "trip_excess_shortest_m"):
        L.append(f"| Reference trip: excess climb, flat vs shortest | "
                 f"{base['trip_excess_flat_m']:.1f} vs "
                 f"{base['trip_excess_shortest_m']:.1f} m | flat "
                 f"{others['trip_excess_flat_m'].min():.1f}-"
                 f"{others['trip_excess_flat_m'].max():.1f} m, shortest "
                 f"{others['trip_excess_shortest_m'].min():.1f}-"
                 f"{others['trip_excess_shortest_m'].max():.1f} m |")
    if has("bm_median_m"):
        L.append(f"| Survey benchmarks: median height error | "
                 f"{base['bm_median_m']:+.2f} m | {others['bm_median_m'].min():+.2f} "
                 f"to {others['bm_median_m'].max():+.2f} m |")
    # any reference street the harness tracked: columns named grade_<street>
    # (the objective called grade_averse has columns that begin the same way)
    # what the model's figure is, and what the documented one beside it is,
    # depends on the kind of reference (sensitivity._STEEP): the Transport
    # Department states a sustained maximum, which should sit at or under
    # the model's steepest 100 m, not on it
    try:
        from .sensitivity import _STEEP
        kind_of = {street: group for group, street in _STEEP}
    except ImportError:
        kind_of = {}
    label = {"td": ("steepest 100 m", "stated maximum {:.1f}%"),
             "ratio": ("steepest 50 m", "signed {:.1f}%"),
             "benchmark": ("mean between bench marks", "{:.1f}% by the marks")}
    for col in [c for c in sd.columns if c.startswith("grade_")
                and not c.startswith("grade_averse_") and has(c)]:
        street = col[len("grade_"):]
        what, ref = label.get(kind_of.get(street), ("gradient", "documented {:.1f}%"))
        pub = base.get(f"published_{street}")
        pub = f" ({ref.format(pub)})" if pub is not None and pd.notna(pub) else ""
        L.append(f"| {street}: {what}{pub} | {base[col]:.1f}% | "
                 f"{others[col].min():.1f}% to {others[col].max():.1f}% |")
    L.append("")
    if has("min_climb_gain_saved_pct"):
        dev = (others["min_climb_gain_saved_pct"] - base["min_climb_gain_saved_pct"]).abs()
        worst = dev.idxmax()
        change = f" ({others.loc[worst, 'change']})" if "change" in sd.columns else ""
        L += [f"The perturbation that shifts the headline most is "
              f"**{worst}**{change}, at "
              f"{others.loc[worst, 'min_climb_gain_saved_pct']:.0f}% climbing "
              f"avoided against {base['min_climb_gain_saved_pct']:.0f}% at "
              f"baseline.", ""]
    if has("edge_overlap_pct") and "top_corridor" in sd.columns:
        least = others["edge_overlap_pct"].idxmin()
        drifters = sorted({s for v in others.get("lead_streets_new", pd.Series(dtype=str)).fillna("")
                           for s in str(v).split("; ") if s})
        L += [f"The corridors are where the model is least rigid, and it is "
              f"worth being precise about how. The *street* that qualifies as "
              f"corridor material is {others['edge_overlap_pct'].min():.0f}-"
              f"{others['edge_overlap_pct'].max():.0f}% the same by length, and "
              f"{_top_corridor_phrase(sd)}; what changes is where each "
              f"corridor is cut and therefore what it is called, most under "
              f"**{least}** ({others['edge_overlap_pct'].min():.0f}%)."
              + (f" Streets that drift in and out of the top twelve: "
                 f"{', '.join(drifters)}. They should be read as a tier "
                 f"rather than a ranking." if drifters else ""), ""]
    return L


def _outside_share(d) -> tuple[float, float]:
    """Walkable street (km, share) whose midpoint is in no planning area."""
    import geopandas as gpd

    e = d["edges"]
    walkable = e[e["walk_ok"]] if "walk_ok" in e.columns else e
    mid = gpd.GeoDataFrame({"length_m": walkable["length_m"].to_numpy()},
                           geometry=walkable.geometry.interpolate(
                               0.5, normalized=True).values, crs=e.crs)
    nb = d["neighborhoods"][["neighborhood", "geometry"]]
    if nb.crs != mid.crs:
        nb = nb.to_crs(mid.crs)
    j = gpd.sjoin(mid, nb, how="left", predicate="within")
    j = j[~j.index.duplicated()]
    out = j.loc[j["neighborhood"].isna(), "length_m"].sum()
    total = walkable["length_m"].sum()
    return out / 1000.0, (out / total if total else 0.0)


def _limits(d) -> list[str]:
    e, nb = d["edges"], d["neighborhoods"]
    walkable = e[e["walk_ok"]] if "walk_ok" in e.columns else e
    walk_km = walkable["length_m"].sum() / 1000
    L = ["## What this analysis does not tell you", ""]

    if "is_structure" in walkable.columns:
        s_km = walkable[walkable["is_structure"].astype(bool)]["length_m"].sum() / 1000
        L.append(
            "- **Elevation is the ground, not the deck.** The terrain model "
            "is bare earth: it has no footbridges, flyovers, elevated "
            "walkways or tunnels in it. Every stretch flagged as one "
            f"({s_km:,.0f} km of the walkable network, "
            f"{100*s_km/walk_km:.0f}%) is given a straight ramp between the "
            "ground at its two ends, so a footbridge counts as level and "
            "the climb to its deck and down again is not counted, while a "
            "bridge the map does not flag is sampled from the ground "
            "beneath it.")
    steps_km = e[e["cls"] == "steps"]["length_m"].sum() / 1000
    L.append(
        "- **Escalators and lifts are not modelled.** A route's climb is "
        "the climb of the ground it covers, whether or not something would "
        f"carry you up it. The {steps_km:,.0f} km of steps are all charged "
        f"as stairs (at {WALK.class_multiplier.get('steps', 1.0):g} times "
        "their length): the model has no notion of an escalator, a lift or "
        "a moving walkway.")
    if "subclass" in walkable.columns:
        side = walkable[walkable["subclass"].isin(SIDEPATH_SUBCLASSES)]
        L.append(
            "- **Travel is modelled on street centrelines, with selected "
            "sidewalks.** Mapped sidewalks and crossings are kept only where "
            "the centreline network needs them to stay connected "
            f"({side['length_m'].sum()/1000:,.0f} km here); taking them all "
            "would represent every street two or three times and wreck the "
            "corridor aggregation. Pedestrian distances are therefore "
            "block-scale, not door-to-door, and the model lets a walker "
            "cross a street at any junction, including where the real "
            "crossing is a footbridge or a subway some way along.")
    # Survey dates as given in the dataset's CSDI metadata record
    # (cedd_rcd_1629267205233_87895): flown 20 December 2019 to 2 February
    # 2020.  "The 2020 survey" is its name, not the year all of it was flown.
    L.append(
        "- **The terrain is the ground of early 2020.** The LiDAR survey "
        "was flown between December 2019 and February 2020. Ground formed "
        "or reshaped since -- new reclamation, a road cut into a slope -- "
        "is missing from it or wrong in it, and a street mapped since then "
        "is sampled from the ground as it was.")

    pts = d.get("points_walk")
    line = ("- **One access point per planning area.** Each neighbourhood is "
            "represented by a single street-network-weighted, "
            "intersection-snapped point.")
    if pts is not None and "street_median_m" in pts.columns:
        n_above = len(_points_above_streets(d))
        line += (f" It is taken within {ACCESS_ABOVE_STREETS_M:.0f} m of the "
                 f"median height of the area's ordinary streets, because a "
                 f"planning area here typically runs from the waterfront up "
                 f"the hillside behind it and one point cannot stand for "
                 f"both"
                 + (f"; {n_above} of the points could not be and "
                    f"stand{'s' if n_above == 1 else ''} higher."
                    if n_above else ".")
                 + " A trip to the upper streets of such an area climbs "
                 "more than the figures here say.")
    if "area_km2" in nb.columns and len(nb):
        with_net = nb[~nb["neighborhood"].isin(d["no_network"])]
        big = with_net.nlargest(3, "area_km2")
        line += (f" The planning areas in this run range from "
                 f"{with_net['area_km2'].min():.1f} to "
                 f"{with_net['area_km2'].max():.1f} km2, and the largest -- "
                 + _list(f"{n} ({a:.1f} km2)" for n, a in
                         zip(big["neighborhood"], big["area_km2"]))
                 + " -- are served worse by one point than the compact ones.")
    if pts is not None and "offset_m" in pts.columns:
        far = pts[pts["node"].notna() & (pts["offset_m"] > 400)]
        line += (f" {len(far)} access points lie more than 400 m from their "
                 f"polygon's centroid.")
    L.append(line)

    out_km, out_share = _outside_share(d)
    L.append(
        "- **The planning areas do not tile the territory.** "
        f"{out_km:,.0f} km of the walkable network ({100*out_share:.0f}%) "
        "lies in no planning area of the set: countryside, most of which "
        "is in no plan, and districts whose plans were not taken as "
        "neighbourhoods. Those streets are routed over like any others, "
        "but no trip starts or ends on them, and a pass or barrier there "
        "is located by the nearest planning area.")
    if len(d["lm"]) > 1:
        L.append(
            "- **No ferries and no railways.** The network is streets and "
            "paths only, so two landmasses that a ferry or a train joins "
            "are simply unconnected here: the report has no cross-harbour "
            "route rather than one with a ferry leg.")
    L += [
        "- **No traffic, surface quality, signals or safety.** The bicycle "
        "stress weights are a crude proxy for road class, not a level-of-"
        "traffic-stress model, and nothing here accounts for signal delay, "
        "crowding, pavement condition or collision risk.",
        "- **Access rules are OpenStreetMap's.** Whether a way is open to "
        "walkers or to bicycles is read from OpenStreetMap tagging through "
        "Overture Maps, with the conservative defaults described above; no "
        "official register of footpaths, cycle tracks or prohibited zones "
        "was used.",
        "",
    ]
    return L


def write_report() -> Path:
    d = _load()
    _landmasses(d)
    L = _intro(d)
    L += _landmass_section(d)
    if d["pairs"].empty:
        L += ["No neighbourhood pair could be routed in this run, so there "
              "is nothing further to report.", ""]
    else:
        L += _headline(d)
        L += _featured(d)
        L += _corridors(d)
        L += _passes(d)
        L += _pareto(d)
        L += _modes(d)
        L += _robustness(d)
    L += _limits(d)
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("\n".join(L), encoding="utf-8")
    return REPORT_MD
