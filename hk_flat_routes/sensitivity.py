"""Sensitivity analysis: how much do the findings depend on the parameters?

Every number in the report rests on a handful of modelling choices -- how
densely the DEM is sampled, how hard it is smoothed, the dead-band, the three
corrections Hong Kong's hillsides needed, and which intersection stands in
for each neighbourhood.  A sceptical reader's first question is whether the
headline moves when those change.  This harness answers it by rebuilding the
pipeline under a one-at-a-time grid around the baseline and tabulating what
comes out.

Each configuration runs in its own subprocess with ``HKFR_RUN_DIR`` and
``HKFR_OVERRIDES`` set, so the main results are never touched.  Only the
street network is shared between configurations, because it is the one
expensive artefact no parameter here can change; upstream also shared its DEM
mosaic, but Hong Kong's terrain model is read tile by tile from
``data/raw``, which every run already has in common.  Runs are cached: a
configuration whose summary already exists is not rebuilt.

The harness works on whatever the baseline run covers.  Started with
``HKFR_RUN_DIR`` pointing at a run directory it perturbs *that* run (its
network, its extent), keeping the configurations in a ``sensitivity`` folder
inside it; without, it perturbs the main build.  No count or street in the
written report is assumed: a reference outside the run prints as "n/a".

    python -m hk_flat_routes sensitivity
    python -m hk_flat_routes sensitivity --only baseline no_despike

A configuration costs one elevation sampling and one analysis: 15 to 30
minutes each for the whole territory, measured with three running side by
side on a 10-core laptop.  The fourteen run one after another, so the full
grid takes four to five hours.  ``--only`` runs a part of it and writes
tables of that part alone; several terminals each given a different part
share the work, every configuration having its own directory, and a last
run with no ``--only`` then finds every summary cached and writes the full
tables in seconds.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (ANALYSIS, CODE_ROOT, ELEVATION, OUTPUT_DIR, PROCESSED_DIR,
                     PROJECT_ROOT)
from .utils import get_logger, step

log = get_logger("hk_flat_routes.sensitivity")

#: One directory per configuration, beside the baseline's ``processed``:
#: ``data/sensitivity`` for the main build, ``<run>/sensitivity`` for a run.
RUNS_DIR = PROCESSED_DIR.parent / "sensitivity"
SENS_CSV = OUTPUT_DIR / "sensitivity.csv"
SENS_MD = OUTPUT_DIR / "sensitivity.md"

BASELINE = {
    "sample_spacing_m": ELEVATION.sample_spacing_m,
    "smooth_window_m": ELEVATION.smooth_window_m,
    "gain_deadband_m": ELEVATION.gain_deadband_m,
    "dem_sigma_m": ELEVATION.dem_sigma_m,
    "lateral_search_m": ELEVATION.lateral_search_m,
    "despike_window_m": ELEVATION.despike_window_m,
    "abutment_blank_m": ELEVATION.abutment_blank_m,
    "point_rank": ANALYSIS.point_rank,
}

#: One-at-a-time perturbations around the baseline, with the reason each is
#: worth testing.  The first six and the last two are upstream's; the three
#: switches in between turn off, one at a time, the corrections this port
#: added for roads cut into hillsides (see ``config.ElevationConfig``).
GRID = [
    ("baseline", {}, "the configuration used for every published figure"),
    ("spacing_2.5m", {"sample_spacing_m": 2.5}, "denser DEM sampling"),
    ("spacing_10m", {"sample_spacing_m": 10.0}, "coarser DEM sampling"),
    ("window_12.5m", {"smooth_window_m": 12.5}, "half the profile smoothing"),
    ("window_50m", {"smooth_window_m": 50.0}, "double the profile smoothing"),
    ("deadband_0.25m", {"gain_deadband_m": 0.25}, "half the dead-band"),
    ("deadband_1m", {"gain_deadband_m": 1.0}, "double the dead-band"),
    ("sigma_0m", {"dem_sigma_m": 0.0}, "no spatial pre-filter on the DEM"),
    ("sigma_3m", {"dem_sigma_m": 3.0},
     "double the spatial pre-filter (upstream's setting)"),
    ("no_terrace_search", {"lateral_search_m": 0.0},
     "sample the mapped centreline; no search for the road surface"),
    ("no_despike", {"despike_window_m": 0.0},
     "no running median along the street"),
    ("no_abutment_fix", {"abutment_blank_m": 0.0},
     "trust the terrain model right up to bridge and tunnel ends"),
    ("point_rank_1", {"point_rank": 1}, "second-nearest access intersection"),
    ("point_rank_2", {"point_rank": 2}, "third-nearest access intersection"),
]

#: Reference sections followed through every configuration, one of each
#: kind in the validation report (see ``validate.py`` for the sources): a
#: Transport Department maximum, two signed ratios on ladder streets, and a
#: mean gradient between bench marks.  Keys are (group, street).
_STEEP = [("td", "Old Peak Road"), ("ratio", "Centre Street"),
          ("ratio", "Peel Street"), ("benchmark", "Hatton Road")]
#: Level streets followed likewise: the tram line, Kowloon's spine, and a
#: contour road a hundred metres up, which is where the hillside
#: corrections earn their keep.
_FLAT = ["Hennessy Road", "Nathan Road", "Bowen Road"]
#: The gap whose height is followed through every configuration.
_GAP = "Wong Nai Chung Gap"

#: Artefacts of the baseline that no parameter in ``GRID`` can change.
_SHARED = ("edges_raw.parquet", "neighborhoods.gpkg")


# --------------------------------------------------------------------------
# per-run summary (executed inside the subprocess, under the overrides)
# --------------------------------------------------------------------------
def summarize_current_run(tag: str) -> Path:
    """Write ``outputs/summary.json`` for the configuration now in force."""
    import geopandas as gpd
    from .corridors import CORRIDORS_GPKG
    from .pairs import PAIRS_PARQUET
    from .passes import PASSES_GEOJSON
    from .pipeline import build_context
    from .utils import configure_gdal_for_proxy
    from .validate import run_validation

    configure_gdal_for_proxy()
    ctx = build_context()
    pairs = pd.read_parquet(PAIRS_PARQUET)
    cor = gpd.read_file(CORRIDORS_GPKG)
    passes = gpd.read_file(PASSES_GEOJSON)
    edges = ctx.edges

    out: dict = {"tag": tag,
                 "params": {k: getattr(ANALYSIS if k == "point_rank" else ELEVATION, k)
                            for k in BASELINE},
                 "n_edges": int(len(edges)), "n_routes": int(len(pairs))}

    walk = pairs[pairs["mode"] == "walk"]
    out["n_pairs"] = int(walk[["origin", "destination"]].drop_duplicates().shape[0])
    for pname, g in walk.groupby("profile"):
        out[f"{pname}_detour_pct"] = float(100 * (g["detour_ratio"].mean() - 1))
        out[f"{pname}_gain_saved_pct"] = float(g["gain_saved_pct"].mean())
        out[f"{pname}_gain_m"] = float(g["elev_gain_m"].mean())
        out[f"{pname}_max_grade_pct"] = float(100 * g["max_grade"].mean())
        out[f"{pname}_dist_km"] = float(g["distance_m"].mean() / 1000.0)

    km = edges[edges["walk_ok"]]["length_m"].sum() / 1000
    out["network_gain_per_km"] = float(
        edges[edges["walk_ok"]]["cum_gain_fwd"].sum() / km)

    # the validation checks that can move with the parameters; the
    # comparison with the 5 m terrain model reads raw LiDAR and cannot
    val = run_validation(ctx, cor, write=False, dem_crosscheck=False)
    steep = val["steep"]
    for group, street in _STEEP:
        row = steep[(steep["group"] == group) & (steep["street"] == street)]
        got = row["computed"].iloc[0] if len(row) else np.nan
        pub = row["published"].iloc[0] if len(row) else np.nan
        out[f"grade_{street}"] = float(got * 100) if np.isfinite(got) else np.nan
        out[f"published_{street}"] = float(pub * 100) if np.isfinite(pub) else np.nan
    flat = val["flat"]
    for street in _FLAT:
        row = flat[flat["streets"].str.split("; ").map(lambda s: street in s)]
        out[f"gainkm_{street}"] = (float(row["gain_per_km"].iloc[0])
                                   if len(row) else np.nan)
    bench = val["benchmarks"]
    out["bm_marks"] = int(len(bench))
    if len(bench):
        d = bench["diff"]
        out["bm_median_m"] = float(d.median())
        out["bm_mad_m"] = float((d - d.median()).abs().median())
        out["bm_within_1m_pct"] = float(100 * (d.abs() <= 1).mean())

    cw = cor[cor["mode"] == "walk"].sort_values("total_score", ascending=False)
    out["top_corridors"] = cw["corridor_name"].head(12).tolist()
    out["corridor_count"] = int(len(cw))
    out["top_corridor_km"] = float(cw["length_km"].iloc[0]) if len(cw) else np.nan

    if len(passes):
        p0 = passes.iloc[0]
        out["top_pass_m"] = float(p0["pass_elev_m"])
        out["top_pass_pairs"] = int(p0["pairs_served"])
        # a pass in the countryside lies in no planning area: it is placed
        # by the nearest one, as the report places it
        where = p0.get("neighborhood")
        if where is None or pd.isna(where):
            where = p0.get("nearest_neighborhood")
        out["top_pass_nbhd"] = "" if where is None or pd.isna(where) else str(where)
    out["n_passes"] = int(len(passes))

    trip = (val["trip"].get("routes") or {}).get("walk")
    if trip:
        out["trip_excess_flat_m"] = float(trip["excess_flat_m"])
        out["trip_excess_shortest_m"] = float(trip["excess_shortest_m"])
        out["trip_goes_round"] = bool(trip["verdict"] == "goes round")
        out["trip_discovered"] = bool(trip["discovered"])
    gaps = val["passes"]
    if len(gaps):
        g = gaps[gaps["gap"] == _GAP]
        if len(g) and np.isfinite(g["pass_m"].iloc[0]):
            out["gap_pass_m"] = float(g["pass_m"].iloc[0])
            out["gap_published_m"] = float(g["published_m"].iloc[0])
        out["gaps_agree"] = int((gaps["verdict"] == "agrees").sum())
        out["gaps_measured"] = int(gaps["pass_m"].notna().sum())

    path = OUTPUT_DIR / "summary.json"
    path.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    log.info("wrote %s", path)
    return path


# --------------------------------------------------------------------------
# orchestration (executed in the parent, with the baseline configuration)
# --------------------------------------------------------------------------
def _share(run_dir: Path) -> None:
    """Give a configuration the baseline's parameter-independent artefacts.

    Chiefly the street network.  Building it afresh would cost minutes per
    configuration and, worse, would build the whole territory from the raw
    extract whatever the baseline covers; sharing it is what makes the
    harness perturb *this* run.  Linked where the filesystem allows, copied
    where it does not.
    """
    for name in _SHARED:
        src = PROCESSED_DIR / name
        dst = run_dir / "processed" / name
        if not src.exists() or dst.exists() or dst.is_symlink():
            continue
        try:
            dst.symlink_to(src.resolve())
        except OSError:
            shutil.copy2(src, dst)


def _clear(run_dir: Path) -> None:
    """Drop a configuration's own artefacts, keeping what it shares."""
    for sub in ("processed", "outputs"):
        for path in (run_dir / sub).glob("*"):
            if path.name in _SHARED or path.is_dir():
                continue
            path.unlink()


def _child_env(run_dir: Path, overrides: dict) -> dict:
    """The environment a configuration's subprocess runs in.

    The data root is pinned as well as the run directory: the child is
    started in the code's directory, not the caller's, and an installed
    copy of the package would otherwise take that for its data root.  The
    code root goes first on ``PYTHONPATH`` so that the child imports the
    package this process is running, installed or not.  (Both used to be
    ``PROJECT_ROOT``, which is the data root once ``HKFR_ROOT`` is set:
    every configuration then died with "No module named hk_flat_routes".)
    For the same reason the run directory is handed on as an absolute
    path: started under a relative ``HKFR_RUN_DIR``, the child would look
    for it from the code's directory and build somewhere else.
    """
    env = dict(os.environ)
    env["HKFR_ROOT"] = str(PROJECT_ROOT)
    env["HKFR_RUN_DIR"] = str(run_dir.absolute())
    env["HKFR_OVERRIDES"] = json.dumps(overrides)
    inherited = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(CODE_ROOT) + (os.pathsep + inherited if inherited else "")
    return env


def _run(tag: str, overrides: dict, force: bool = False) -> dict:
    run_dir = RUNS_DIR / tag
    summary = run_dir / "outputs" / "summary.json"
    if summary.exists() and not force:
        log.info("[%s] cached", tag)
        return json.loads(summary.read_text(encoding="utf-8"))

    (run_dir / "processed").mkdir(parents=True, exist_ok=True)
    (run_dir / "outputs").mkdir(parents=True, exist_ok=True)
    if force:
        _clear(run_dir)
    _share(run_dir)

    env = _child_env(run_dir, overrides)
    log_path = run_dir / "run.log"
    t0 = time.time()
    # ``build-network`` runs without ``--force`` on purpose: forced, it
    # would rebuild the shared network.  The elevation profiles and edge
    # metrics are rebuilt regardless, because this directory has none.
    with open(log_path, "w") as fh:
        for cmd in (["build-network"], ["analyze", "--force"],
                    ["summarize", "--tag", tag]):
            rc = subprocess.call([sys.executable, "-m", "hk_flat_routes", *cmd],
                                 env=env, stdout=fh, stderr=subprocess.STDOUT,
                                 cwd=str(CODE_ROOT))
            if rc:
                raise RuntimeError(f"[{tag}] `{' '.join(cmd)}` failed (rc={rc}); "
                                   f"see {log_path}")
    log.info("[%s] done in %.0fs", tag, time.time() - t0)
    return json.loads(summary.read_text(encoding="utf-8"))


def run_sensitivity(force: bool = False, only: list[str] | None = None) -> pd.DataFrame:
    known = [t for t, _o, _w in GRID]
    for tag in only or []:
        if tag not in known:
            raise ValueError(f"unknown configuration {tag!r}; choose from {known}")
    todo = [g for g in GRID if not only or g[0] in only]
    rows = []
    with step(f"sensitivity analysis over {len(todo)} configurations", log):
        for tag, overrides, _why in todo:
            rows.append(_run(tag, overrides, force=force))
    df = pd.DataFrame(rows)
    df = df.set_index("tag")
    _write(df)
    return df


def _jaccard(a: list, b: list) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 1.0


def _corridor_material(tag: str) -> dict | None:
    """Edge id -> length for the edges that qualified as corridor material.

    Reconstructed from the run's edge scores with the same quantile cut the
    corridor merger uses. ``None`` if the run directory is not available.
    """
    path = RUNS_DIR / tag / "processed" / "edge_corridor_scores.parquet"
    if not path.exists():
        return None
    s = pd.read_parquet(path)
    s = s[s["mode"] == "walk"]
    pos = s[s["score"] > 0]
    if pos.empty:
        return {}
    cut = float(np.quantile(pos["score"], ANALYSIS.corridor_score_quantile))
    top = s[s["score"] >= cut]
    return dict(zip(top["edge_id"].astype(int), top["length_m"].astype(float)))


def _edge_overlap(a: dict | None, b: dict | None) -> float:
    """Length-weighted Jaccard of two corridor-material edge sets, in %.

    Comparing corridor *names* is brittle: a merge boundary moving by one
    block renames a corridor, so two runs that agree on almost every metre
    of street can share no names at all. Comparing the edges themselves is
    what the question actually asks.
    """
    if a is None or b is None:
        return float("nan")
    inter = sum(l for e, l in a.items() if e in b)
    union = sum(a.values()) + sum(l for e, l in b.items() if e not in a)
    return 100.0 * inter / union if union else 100.0


def _lead_streets(names: list) -> set:
    """The leading street of each corridor name, e.g. 'Des Voeux Road West'."""
    return {n.split(" - ")[0] for n in names if n}


def _num(row, key: str, fmt: str = "{:.1f}", unit: str = "") -> str:
    """A cell for the tables: a number, or "n/a" when the run has none."""
    v = row.get(key, np.nan)
    try:
        ok = np.isfinite(v)
    except TypeError:
        ok = False
    return (fmt.format(v) + unit) if ok else "n/a"


def _span(series: pd.Series, fmt: str = "{:.0f}") -> str:
    """'lo-hi' of a column (one figure if they print alike), or 'n/a'."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return "n/a"
    lo, hi = fmt.format(s.min()), fmt.format(s.max())
    return lo if lo == hi else f"{lo}-{hi}"


def _has(df: pd.DataFrame, column: str) -> bool:
    """Does any configuration carry a value in this column?"""
    return column in df and bool(df[column].notna().any())


def _write(df: pd.DataFrame) -> None:
    why = {tag: w for tag, _o, w in GRID}
    base = df.loc["baseline"] if "baseline" in df.index else None
    ref = base if base is not None else df.iloc[0]
    df = df.copy()
    df["change"] = [why.get(t, "") for t in df.index]
    base_mat = _corridor_material("baseline") if base is not None else None
    base_lead = _lead_streets(base["top_corridors"]) if base is not None else set()
    df["edge_overlap_pct"] = [_edge_overlap(_corridor_material(t), base_mat)
                              for t in df.index]
    df["lead_streets_shared"] = [len(_lead_streets(r["top_corridors"]) & base_lead)
                                 for _, r in df.iterrows()]
    df["lead_streets_total"] = len(base_lead)
    df["lead_streets_new"] = ["; ".join(sorted(_lead_streets(r["top_corridors"]) - base_lead))
                              for _, r in df.iterrows()]
    df["top_corridor"] = [r["top_corridors"][0] if r["top_corridors"] else ""
                          for _, r in df.iterrows()]
    flat = df.drop(columns=["top_corridors", "params"], errors="ignore")
    flat.to_csv(SENS_CSV)

    n_top = int(max((len(r["top_corridors"]) for _, r in df.iterrows()), default=0))
    L = ["# Sensitivity analysis", "",
         f"Each of the {len(df)} configurations below rebuilds the pipeline -- "
         f"elevation sampling, metrics, {_num(ref, 'n_routes', '{:,.0f}')} "
         "routes, corridors and passes -- on the same street network "
         f"({_num(ref, 'n_edges', '{:,.0f}')} edges), with one parameter "
         "changed from the baseline. The question is whether the findings "
         "survive the modelling choices. (The harness is upstream's, from "
         "flattensf; the parameters and the reference streets are Hong "
         "Kong's.)", "",
         "Baseline: " + ", ".join(f"{k} = {v:g}" for k, v in BASELINE.items()), "",
         f"## The headline (walking, all {_num(ref, 'n_pairs', '{:,.0f}')} "
         "ordered pairs with a way between them)", "",
         "| Configuration | Change | Flattest: extra distance | Flattest: "
         "climbing avoided | Shortest: mean climb | Flattest: mean climb | "
         "Grade-averse: mean steepest |",
         "|---|---|---|---|---|---|---|"]
    for tag, r in df.iterrows():
        L.append(f"| {tag} | {why.get(tag, '')} | "
                 f"{_num(r, 'min_climb_detour_pct', '{:+.0f}', '%')} | "
                 f"{_num(r, 'min_climb_gain_saved_pct', '{:.0f}', '%')} | "
                 f"{_num(r, 'shortest_gain_m', '{:.0f}', ' m')} | "
                 f"{_num(r, 'min_climb_gain_m', '{:.0f}', ' m')} | "
                 f"{_num(r, 'grade_averse_max_grade_pct', '{:.1f}', '%')} |")

    steep = [s for _g, s in _STEEP]
    L += ["", "## Elevation model checks", "",
          "| Configuration | " + " | ".join(steep) + " | "
          + " | ".join(f"{n} climb/km" for n in _FLAT)
          + " | Network climb/km | Model minus bench mark (median; within 1 m) |",
          "|---" * (len(steep) + len(_FLAT) + 3) + "|"]
    for tag, r in df.iterrows():
        L.append(f"| {tag} | "
                 + " | ".join(_num(r, f"grade_{n}", "{:.1f}", "%") for n in steep)
                 + " | "
                 + " | ".join(_num(r, f"gainkm_{n}", "{:.1f}", " m") for n in _FLAT)
                 + f" | {_num(r, 'network_gain_per_km', '{:.1f}', ' m')} | "
                 f"{_num(r, 'bm_median_m', '{:+.2f}', ' m')}; "
                 f"{_num(r, 'bm_within_1m_pct', '{:.0f}', '%')} |")
    # what the model's figure is, and what the documented one beside it is:
    # the two are not the same quantity for the Transport Department's roads
    kinds = {"td": ("the steepest 100 m of the Transport Department's section",
                    "stated maximum {}"),
             "ratio": ("the steepest 50 m of the signed section", "signed {}"),
             "benchmark": ("the mean between two bench marks", "{} by the marks")}
    L += ["", "Each street's column holds: "
          + "; ".join(f"{s}, {kinds[g][0]} ("
                      + kinds[g][1].format(_num(ref, f'published_{s}', '{:.1f}', '%'))
                      + ")" for g, s in _STEEP)
          + ". The Transport Department's maximum is a sustained gradient, "
          "which the validation report expects to find between the section's "
          "mean and its steepest 100 m, so the figure here should sit at or "
          "above it, not on it. The streets, their sections and their sources are those of "
          "the validation report; the bench marks are the "
          f"{_num(ref, 'bm_marks', '{:,.0f}')} official ones beside a road in "
          "this run.", ""]

    L += ["## Corridors, passes and the contour trip", "",
          "Corridor overlap is measured on the street itself: the "
          "length-weighted share of corridor-material edges the run has in "
          "common with the baseline. Comparing corridor names would be "
          "misleading, since a merge boundary moving by one block renames a "
          "corridor without changing where it runs. The contour trip is the "
          "validation report's (University of Hong Kong to the Botanical "
          "Gardens): the climbing each route wastes beyond the difference "
          "between its ends.", "",
          "| Configuration | Corridors found | Corridor edges shared with "
          f"baseline | Lead streets of the top {n_top} kept | Streets that "
          f"enter the top {n_top} | Top corridor | Top pass | {_GAP} "
          f"(mapped at {_num(ref, 'gap_published_m', '{:.0f}', ' m')}) | "
          "Contour trip, excess climb (flat / shortest) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for tag, r in df.iterrows():
        top = r["top_corridor"].split(" - ")[0]
        L.append(f"| {tag} | {_num(r, 'corridor_count', '{:.0f}')} | "
                 f"{_num(r, 'edge_overlap_pct', '{:.0f}', '%')} | "
                 f"{int(r['lead_streets_shared'])}/{len(base_lead)} | "
                 f"{r['lead_streets_new'] or '&mdash;'} | "
                 f"{top} ({_num(r, 'top_corridor_km', '{:.1f}', ' km')}) | "
                 f"{r.get('top_pass_nbhd', '')} "
                 f"{_num(r, 'top_pass_m', '{:.0f}', ' m')}, "
                 f"{_num(r, 'top_pass_pairs', '{:.0f}')} pairs | "
                 f"{_num(r, 'gap_pass_m', '{:.1f}', ' m')} | "
                 f"{_num(r, 'trip_excess_flat_m', '{:.1f}', ' m')} / "
                 f"{_num(r, 'trip_excess_shortest_m', '{:.1f}', ' m')} |")

    if base is not None and len(df) > 1:
        others = df.drop(index="baseline")
        have_overlap = others["edge_overlap_pct"].notna().any()
        L += ["", "## Reading it", "",
              f"- The headline trade (extra distance for climbing avoided on the "
              f"flattest route) ranges from "
              f"{others['min_climb_detour_pct'].min():+.0f}% / "
              f"{others['min_climb_gain_saved_pct'].min():.0f}% to "
              f"{others['min_climb_detour_pct'].max():+.0f}% / "
              f"{others['min_climb_gain_saved_pct'].max():.0f}% across every "
              f"perturbation, against {base['min_climb_detour_pct']:+.0f}% / "
              f"{base['min_climb_gain_saved_pct']:.0f}% at baseline."]
        if have_overlap:
            drifters = sorted({s for v in others["lead_streets_new"]
                               for s in v.split("; ") if s})
            L += [f"- The corridor material stays "
                  f"{_span(others['edge_overlap_pct'])}% the same street, by "
                  f"length, under every perturbation, and "
                  f"{_span(others['lead_streets_shared'])} of the baseline's "
                  f"{len(base_lead)} lead streets keep their place. What moves "
                  f"is the exact extent and composite name of each corridor, "
                  f"most under **{others['edge_overlap_pct'].idxmin()}** "
                  f"({others['edge_overlap_pct'].min():.0f}%)"
                  + (", and a few borderline streets drift in and out at the "
                     f"margin: {', '.join(drifters)}." if drifters else ".")]
        if _has(df, "top_pass_nbhd"):
            L += [f"- The dominant pass is in {base.get('top_pass_nbhd', '')} in "
                  f"{int((df['top_pass_nbhd'] == base.get('top_pass_nbhd')).sum())} "
                  f"of {len(df)} configurations."]
        if _has(df, "bm_median_m") and _has(df, "bm_within_1m_pct"):
            L += [f"- Against the official bench marks the model's median "
                  f"difference is {_span(df['bm_median_m'], '{:+.2f}')} m in "
                  f"every configuration, with "
                  f"{_span(df['bm_within_1m_pct'])}% of marks within 1 m; the "
                  f"smallest share is under "
                  f"**{df['bm_within_1m_pct'].idxmin()}**."]
        if _has(df, "trip_excess_flat_m"):
            rounds = df["trip_goes_round"].fillna(False).astype(bool)
            L += [f"- On the contour trip the flat route goes round in "
                  f"{int(rounds.sum())} of {len(df)} configurations, wasting "
                  f"{_span(df['trip_excess_flat_m'], '{:.1f}')} m of climbing "
                  f"against the shortest route's "
                  f"{_span(df['trip_excess_shortest_m'], '{:.1f}')} m."]
        if _has(df, "gap_pass_m"):
            L += [f"- {_GAP} comes out at {_span(df['gap_pass_m'], '{:.1f}')} m "
                  f"across the configurations, against a mapped "
                  f"{_num(ref, 'gap_published_m', '{:.0f}')} m."]
        L += [""]
    SENS_MD.write_text("\n".join(L), encoding="utf-8")
    log.info("wrote %s and %s", SENS_CSV.name, SENS_MD.name)
