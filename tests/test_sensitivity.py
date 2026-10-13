"""Tests for the parameter-override plumbing and the sensitivity harness."""
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent


def _config_in_subprocess(env_extra: dict) -> dict:
    """Import the config in a fresh interpreter under the given environment."""
    code = (
        "import json; from hk_flat_routes import config as c, elevation as e, "
        "sensitivity as s\n"
        "print(json.dumps({'spacing': c.ELEVATION.sample_spacing_m,"
        " 'sigma': c.ELEVATION.dem_sigma_m, 'sigma_mod': e.DEM_SMOOTH_SIGMA_M,"
        " 'search': c.ELEVATION.lateral_search_m,"
        " 'despike': c.ELEVATION.despike_window_m,"
        " 'abutment': c.ELEVATION.abutment_blank_m,"
        " 'rank': c.ANALYSIS.point_rank, 'processed': str(c.PROCESSED_DIR),"
        " 'outputs': str(c.OUTPUT_DIR), 'raw': str(c.RAW_DIR),"
        " 'runs': str(s.RUNS_DIR), 'sens_md': str(s.SENS_MD)}))"
    )
    env = dict(os.environ)
    env.pop("HKFR_OVERRIDES", None); env.pop("HKFR_RUN_DIR", None)
    env.update(env_extra)
    env["PYTHONPATH"] = str(ROOT)
    out = subprocess.run([sys.executable, "-c", code], env=env, cwd=ROOT,
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_defaults_without_overrides():
    c = _config_in_subprocess({})
    # Hong Kong's baseline: a 1.5 m pre-filter (upstream uses 3 m) and the
    # three hillside corrections switched on
    assert c["spacing"] == 5.0 and c["sigma"] == 1.5 and c["rank"] == 0
    assert c["sigma_mod"] == c["sigma"]
    assert c["search"] > 0 and c["despike"] > 0 and c["abutment"] > 0
    assert c["processed"].endswith("data/processed")
    assert c["runs"].endswith("data/sensitivity")


def test_overrides_reach_every_module():
    c = _config_in_subprocess({"HKFR_OVERRIDES": json.dumps(
        {"sample_spacing_m": 10, "dem_sigma_m": 0, "point_rank": 2,
         "lateral_search_m": 0, "despike_window_m": 0, "abutment_blank_m": 0})})
    assert c["spacing"] == 10 and c["rank"] == 2
    # the elevation module reads sigma from the config, not a local constant
    assert c["sigma"] == 0 and c["sigma_mod"] == 0
    # the Hong Kong corrections can each be switched off from outside
    assert c["search"] == 0 and c["despike"] == 0 and c["abutment"] == 0


def test_unknown_override_keys_are_refused():
    """Ignored, as they used to be, a mistyped key builds a second baseline."""
    with pytest.raises(subprocess.CalledProcessError) as err:
        _config_in_subprocess({"HKFR_OVERRIDES": json.dumps({"smooth_window": 12.5})})
    assert "unknown parameter(s) ['smooth_window']" in err.value.stderr


def test_malformed_overrides_fail_loudly():
    with pytest.raises(subprocess.CalledProcessError):
        _config_in_subprocess({"HKFR_OVERRIDES": "{not json"})


def test_run_dir_redirects_processed_and_outputs_but_not_raw(tmp_path):
    c = _config_in_subprocess({"HKFR_RUN_DIR": str(tmp_path)})
    assert c["processed"] == str(tmp_path / "processed")
    assert c["outputs"] == str(tmp_path / "outputs")
    # raw data is shared: an experiment must never re-download 14 GB of DEM
    assert c["raw"].endswith("data/raw")


def test_a_run_directory_keeps_its_own_sensitivity_runs(tmp_path):
    """Pointed at a run, the harness perturbs that run and writes beside it."""
    c = _config_in_subprocess({"HKFR_RUN_DIR": str(tmp_path)})
    assert c["runs"] == str(tmp_path / "sensitivity")
    assert c["sens_md"] == str(tmp_path / "outputs" / "sensitivity.md")


# ------------------------------------------------------- sharing artefacts
def test_configurations_share_the_network_and_nothing_else(tmp_path, monkeypatch):
    from hk_flat_routes import sensitivity as S
    base = tmp_path / "processed"
    base.mkdir()
    (base / "edges_raw.parquet").write_text("network")
    (base / "neighborhoods.gpkg").write_text("polygons")
    (base / "edge_profiles.npz").write_text("depends on the parameters")
    monkeypatch.setattr(S, "PROCESSED_DIR", base)
    run = tmp_path / "sensitivity" / "no_despike"
    (run / "processed").mkdir(parents=True)
    (run / "outputs").mkdir()
    S._share(run)
    assert (run / "processed" / "edges_raw.parquet").read_text() == "network"
    assert (run / "processed" / "neighborhoods.gpkg").read_text() == "polygons"
    assert not (run / "processed" / "edge_profiles.npz").exists()
    S._share(run)                                   # a second call is harmless

    # a forced re-run drops the configuration's own results, not the network
    (run / "processed" / "edge_profiles.npz").write_text("stale")
    (run / "outputs" / "summary.json").write_text("{}")
    S._clear(run)
    assert not (run / "processed" / "edge_profiles.npz").exists()
    assert not (run / "outputs" / "summary.json").exists()
    assert (run / "processed" / "edges_raw.parquet").read_text() == "network"
    assert (base / "edges_raw.parquet").read_text() == "network"


def test_sharing_tolerates_a_baseline_without_a_network(tmp_path, monkeypatch):
    from hk_flat_routes import sensitivity as S
    monkeypatch.setattr(S, "PROCESSED_DIR", tmp_path / "nothing-here")
    run = tmp_path / "run"
    (run / "processed").mkdir(parents=True)
    S._share(run)
    assert list((run / "processed").iterdir()) == []


def test_a_configuration_runs_the_running_code_on_the_callers_data(tmp_path, monkeypatch):
    """With ``HKFR_ROOT`` elsewhere, the data root is not where the code is."""
    from hk_flat_routes import sensitivity as S
    data = tmp_path / "data-root"
    monkeypatch.setattr(S, "PROJECT_ROOT", data)
    monkeypatch.setattr(S, "RUNS_DIR", data / "data" / "sensitivity")
    monkeypatch.setattr(S, "PROCESSED_DIR", data / "data" / "processed")
    monkeypatch.setenv("PYTHONPATH", "/somewhere/else")
    calls = []

    def call(cmd, env, cwd, **kwargs):
        calls.append((cmd, env, cwd))
        if cmd[-2:] == ["--tag", "no_despike"]:
            (S.RUNS_DIR / "no_despike" / "outputs" / "summary.json").write_text('{"tag": "x"}')
        return 0
    monkeypatch.setattr(S.subprocess, "call", call)
    assert S._run("no_despike", {"despike_window_m": 0.0}) == {"tag": "x"}

    assert [c[0][3] for c in calls] == ["build-network", "analyze", "summarize"]
    for cmd, env, cwd in calls:
        assert cwd == str(ROOT)                      # where the package is
        assert env["PYTHONPATH"].split(os.pathsep) == [str(ROOT), "/somewhere/else"]
        assert env["HKFR_ROOT"] == str(data)         # where the data is
        assert env["HKFR_RUN_DIR"] == str(S.RUNS_DIR / "no_despike")
        assert json.loads(env["HKFR_OVERRIDES"]) == {"despike_window_m": 0.0}


def test_the_child_can_import_the_package_from_a_data_root_elsewhere(tmp_path, monkeypatch):
    """What failed: every configuration died with "No module named hk_flat_routes"."""
    from hk_flat_routes import sensitivity as S
    monkeypatch.setattr(S, "PROJECT_ROOT", tmp_path)
    monkeypatch.delenv("PYTHONPATH", raising=False)
    env = S._child_env(tmp_path / "run", {})
    assert env["PYTHONPATH"] == str(ROOT)
    code = ("from hk_flat_routes import config as c; "
            "print(c.RAW_DIR); print(c.PROCESSED_DIR)")
    out = subprocess.run([sys.executable, "-c", code], env=env, cwd=str(S.CODE_ROOT),
                         capture_output=True, text=True, check=True).stdout.split()
    assert out == [str(tmp_path / "data" / "raw"), str(tmp_path / "run" / "processed")]


def test_a_relative_run_directory_is_the_same_place_for_the_child(tmp_path, monkeypatch):
    """The child starts in the code's directory, not the caller's: handed a
    relative ``HKFR_RUN_DIR`` it would build beside the code instead."""
    from hk_flat_routes import sensitivity as S
    monkeypatch.setattr(S, "PROJECT_ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    env = S._child_env(Path("runs") / "sensitivity" / "no_despike", {})
    here = Path.cwd() / "runs" / "sensitivity" / "no_despike"
    assert env["HKFR_RUN_DIR"] == str(here)
    out = subprocess.run([sys.executable, "-c", "from hk_flat_routes import config as c; "
                          "print(c.PROCESSED_DIR)"], env=env, cwd=str(S.CODE_ROOT),
                         capture_output=True, text=True, check=True).stdout.strip()
    assert out == str(here / "processed")
    assert (here / "processed").is_dir()            # made where the caller meant


def test_unknown_configuration_is_refused(tmp_path, monkeypatch):
    from hk_flat_routes import sensitivity as S
    monkeypatch.setattr(S, "RUNS_DIR", tmp_path)
    with pytest.raises(ValueError, match="unknown configuration"):
        S.run_sensitivity(only=["no_such_thing"])


# ------------------------------------------------------------- the writer
def _fake_row(tag, **over):
    r = {
        "tag": tag, "params": {}, "n_edges": 108750, "n_routes": 6096,
        "n_pairs": 762,
        "shortest_detour_pct": 0.0, "shortest_gain_saved_pct": 0.0,
        "shortest_gain_m": 146.0, "shortest_max_grade_pct": 27.0,
        "shortest_dist_km": 6.3,
        "min_climb_detour_pct": 14.0, "min_climb_gain_saved_pct": 39.0,
        "min_climb_gain_m": 85.0, "min_climb_max_grade_pct": 17.0,
        "min_climb_dist_km": 7.2,
        "grade_averse_detour_pct": 48.0, "grade_averse_gain_saved_pct": 30.0,
        "grade_averse_gain_m": 99.0, "grade_averse_max_grade_pct": 10.8,
        "grade_averse_dist_km": 9.0,
        "balanced_detour_pct": 19.0, "balanced_gain_saved_pct": 33.0,
        "balanced_gain_m": 94.0, "balanced_max_grade_pct": 12.8,
        "balanced_dist_km": 7.4,
        "network_gain_per_km": 21.0,
        "grade_Old Peak Road": 21.8, "published_Old Peak Road": 17.0,
        "grade_Centre Street": 21.5, "published_Centre Street": 25.0,
        "grade_Peel Street": 24.4, "published_Peel Street": 25.0,
        "grade_Hatton Road": 13.6, "published_Hatton Road": 13.5,
        "gainkm_Hennessy Road": 1.9, "gainkm_Nathan Road": 3.5,
        "gainkm_Bowen Road": 11.4,
        "bm_marks": 558, "bm_median_m": -0.47, "bm_mad_m": 0.21,
        "bm_within_1m_pct": 86.0,
        "top_corridors": ["Des Voeux Road West - X", "King's Road - Y", "Nathan Road"],
        "corridor_count": 41, "top_corridor_km": 6.9,
        "top_pass_m": 171.0, "top_pass_pairs": 20, "top_pass_nbhd": "Kwun Tong (North)",
        "n_passes": 38, "trip_excess_flat_m": 44.9,
        "trip_excess_shortest_m": 92.1, "trip_goes_round": True,
        "trip_discovered": False, "gap_pass_m": 193.6, "gap_published_m": 191.0,
        "gaps_agree": 4, "gaps_measured": 6,
    }
    r.update(over)
    return r


def _write(tmp_path, monkeypatch, rows):
    from hk_flat_routes import sensitivity as S
    monkeypatch.setattr(S, "SENS_CSV", tmp_path / "s.csv")
    monkeypatch.setattr(S, "SENS_MD", tmp_path / "s.md")
    monkeypatch.setattr(S, "RUNS_DIR", tmp_path / "no-runs")   # no run dirs
    S._write(pd.DataFrame(rows).set_index("tag"))
    return ((tmp_path / "s.md").read_text(encoding="utf-8"),
            pd.read_csv(tmp_path / "s.csv", index_col="tag"))


def test_writer_produces_csv_and_markdown(tmp_path, monkeypatch):
    md, csv = _write(tmp_path, monkeypatch, [
        _fake_row("baseline"),
        _fake_row("no_despike", min_climb_gain_saved_pct=36.0,
                  top_corridors=["Des Voeux Road West - X", "Other", "Nathan Road"]),
    ])
    assert "| baseline |" in md and "| no_despike |" in md
    assert "no running median along the street" in md      # the reason column
    assert "| 2/3 |" in md                       # two of three lead streets kept
    assert "Other" in md                         # the street that entered
    assert "top_corridors" not in csv.columns    # list column dropped from CSV
    assert csv.loc["no_despike", "min_climb_gain_saved_pct"] == 36.0
    assert csv.loc["no_despike", "lead_streets_shared"] == 2
    assert csv.loc["no_despike", "lead_streets_total"] == 3
    assert pd.isna(csv.loc["no_despike", "edge_overlap_pct"])  # no run dirs


def test_writer_counts_are_the_runs_own_and_units_are_metric(tmp_path, monkeypatch):
    md, _ = _write(tmp_path, monkeypatch, [
        _fake_row("baseline", n_pairs=762, n_routes=6096, n_edges=108750),
        _fake_row("sigma_0m"),
    ])
    assert "Each of the 2 configurations" in md
    assert "6,096 routes" in md and "108,750 edges" in md
    assert "all 762 ordered pairs" in md
    assert "| 146 m | 85 m |" in md              # mean climbs, in metres
    assert "Kwun Tong (North) 171 m, 20 pairs" in md
    assert "44.9 m / 92.1 m" in md               # the contour trip
    assert "-0.47 m; 86%" in md                  # model against bench marks
    # documented figures, each with what the model's figure beside it is
    assert "(stated maximum 17.0%)" in md and "(signed 25.0%)" in md
    assert "(13.5% by the marks)" in md
    # nothing of the city this was ported from, and no imperial units
    for word in (" ft", " mi ", "Wiggle", "Filbert", "Embarcadero", "Valencia"):
        assert word not in md


def test_writer_survives_a_run_without_the_reference_streets(tmp_path, monkeypatch):
    """A window that holds none of the reference streets prints n/a."""
    gone = {k: float("nan") for k in _fake_row("x")
            if k.startswith(("grade_", "published_", "gainkm_", "bm_m", "bm_w",
                             "trip_excess", "gap_"))}
    gone["bm_marks"] = 0
    md, csv = _write(tmp_path, monkeypatch, [
        _fake_row("baseline", **gone), _fake_row("deadband_1m", **gone)])
    assert "n/a" in md
    assert "| baseline |" in md and "## Reading it" in md
    assert "bench marks the model" not in md     # that line needs a value
    assert pd.isna(csv.loc["baseline", "grade_Old Peak Road"])


def test_writer_without_a_baseline_still_writes(tmp_path, monkeypatch):
    md, csv = _write(tmp_path, monkeypatch, [_fake_row("no_despike")])
    assert "| no_despike |" in md and "## Reading it" not in md
    assert csv.loc["no_despike", "lead_streets_total"] == 0


def test_jaccard():
    from hk_flat_routes.sensitivity import _jaccard
    assert _jaccard(["a", "b"], ["a", "b"]) == 1.0
    assert _jaccard(["a", "b"], ["b", "c"]) == pytest.approx(1 / 3)
    assert _jaccard([], []) == 1.0


def test_edge_overlap_is_length_weighted():
    from hk_flat_routes.sensitivity import _edge_overlap, _lead_streets
    a = {1: 100.0, 2: 300.0}
    b = {2: 300.0, 3: 100.0}
    # shared 300 of a 500 m union
    assert _edge_overlap(a, b) == pytest.approx(60.0)
    assert _edge_overlap(a, a) == 100.0
    assert _edge_overlap({}, {}) == 100.0
    assert math.isnan(_edge_overlap(None, b))
    assert _lead_streets(["Des Voeux Road West - Catchick Street",
                          "Nathan Road", ""]) == {"Des Voeux Road West",
                                                  "Nathan Road"}


def test_grid_is_one_at_a_time_around_the_baseline():
    from hk_flat_routes.sensitivity import BASELINE, GRID
    tags = [t for t, _, _ in GRID]
    assert tags[0] == "baseline" and len(set(tags)) == len(tags)
    for tag, over, why in GRID[1:]:
        assert len(over) == 1, f"{tag} perturbs more than one parameter"
        (k, v), = over.items()
        assert k in BASELINE and v != BASELINE[k]
        assert why


def test_grid_perturbs_the_hong_kong_corrections():
    """Each hillside correction is switched off once, and sigma goes both ways."""
    from hk_flat_routes.config import ElevationConfig
    from hk_flat_routes.sensitivity import BASELINE, GRID
    fields = set(ElevationConfig.__dataclass_fields__) | {"point_rank"}
    touched: dict = {}
    for _tag, over, _why in GRID[1:]:
        (k, v), = over.items()
        assert k in fields, f"{k} is not a parameter the pipeline reads"
        touched.setdefault(k, []).append(v)
    for k in ("lateral_search_m", "despike_window_m", "abutment_blank_m"):
        assert touched[k] == [0.0]
    assert min(touched["dem_sigma_m"]) == 0.0
    assert max(touched["dem_sigma_m"]) > BASELINE["dem_sigma_m"]
    for k in ("sample_spacing_m", "smooth_window_m", "gain_deadband_m"):
        assert min(touched[k]) < BASELINE[k] < max(touched[k])
