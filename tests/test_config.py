"""Tests for the configuration's environment handling: where the data lives
and what ``HKFR_OVERRIDES`` will accept."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from hk_flat_routes import config

ROOT = Path(__file__).resolve().parent.parent


def _run(code: str, env_extra: dict, cwd: Path, pythonpath: Path = ROOT):
    env = {k: v for k, v in os.environ.items() if not k.startswith("HKFR_")}
    env.update(env_extra, PYTHONPATH=str(pythonpath))
    return subprocess.run([sys.executable, "-c", code], env=env, cwd=cwd,
                          capture_output=True, text=True)


# ------------------------------------------------------------ data root
def test_a_checkout_keeps_its_data_beside_the_code(tmp_path):
    code = tmp_path / "flattenhk"
    assert config._data_root(code, None, tmp_path / "elsewhere") == code
    assert config._data_root(code, "", tmp_path / "elsewhere") == code


def test_hkfr_root_wins_and_is_made_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert config._data_root(tmp_path / "code", "data-here", tmp_path) == tmp_path / "data-here"
    site = tmp_path / "venv" / "lib" / "python3.12" / "site-packages"
    assert config._data_root(site, str(tmp_path / "d"), tmp_path) == tmp_path / "d"


@pytest.mark.parametrize("packages", ["site-packages", "dist-packages"])
def test_an_installed_copy_works_in_the_current_directory(tmp_path, packages):
    site = tmp_path / "venv" / "lib" / "python3.12" / packages
    assert config._data_root(site, None, tmp_path / "work") == tmp_path / "work"


def test_importing_an_installed_copy_writes_nothing_into_site_packages(tmp_path):
    """What ``pip install .`` did: ``data/`` and ``outputs/`` among the packages."""
    site = tmp_path / "venv" / "lib" / "python3.12" / "site-packages"
    (site / "hk_flat_routes").mkdir(parents=True)
    for name in ("__init__.py", "config.py"):
        shutil.copy(ROOT / "hk_flat_routes" / name, site / "hk_flat_routes" / name)
    work = tmp_path / "work"
    work.mkdir()
    done = _run("from hk_flat_routes import config as c; "
                "print(c.PROJECT_ROOT); print(c.RAW_DIR); print(c.OUTPUT_DIR); print(c.CODE_ROOT)",
                {}, cwd=work, pythonpath=site)
    assert done.returncode == 0, done.stderr
    project, raw, outputs, code = [Path(line) for line in done.stdout.split()]
    assert project.resolve() == work.resolve()
    assert raw.resolve() == (work / "data" / "raw").resolve()
    assert outputs.resolve() == (work / "outputs").resolve()
    assert code.resolve() == site.resolve()
    assert sorted(p.name for p in site.iterdir()) == ["hk_flat_routes"]

    # and HKFR_ROOT still decides when it is set
    done = _run("from hk_flat_routes import config as c; print(c.PROJECT_ROOT)",
                {"HKFR_ROOT": str(tmp_path / "chosen")}, cwd=work, pythonpath=site)
    assert Path(done.stdout.strip()) == tmp_path / "chosen"


# ------------------------------------------------------------ overrides
def test_overrides_take_the_type_of_the_default():
    e = config._apply(config.ElevationConfig(), {
        "sample_spacing_m": 10, "dem_sigma_m": "0", "min_samples": 4.0,
        "interpolate_structures": 0})
    assert (e.sample_spacing_m, type(e.sample_spacing_m)) == (10.0, float)
    assert (e.dem_sigma_m, type(e.dem_sigma_m)) == (0.0, float)
    assert (e.min_samples, type(e.min_samples)) == (4, int)
    assert e.interpolate_structures is False
    a = config._apply(config.AnalysisConfig(), {"point_rank": "1"})
    assert (a.point_rank, type(a.point_rank)) == (1, int)


@pytest.mark.parametrize("name, value", [
    ("point_rank", 1.5), ("point_rank", "one"), ("point_rank", True),
    ("point_rank", None), ("point_rank", [1]),
    ("sample_spacing_m", "wide"), ("sample_spacing_m", False),
    ("sample_spacing_m", float("nan")), ("sample_spacing_m", float("inf")),
    ("sample_spacing_m", {"m": 5}),
    ("interpolate_structures", "false"), ("interpolate_structures", 2),
])
def test_an_override_that_is_not_the_fields_type_is_refused(name, value):
    for cfg in (config.ElevationConfig(), config.AnalysisConfig()):
        if name in cfg.__dataclass_fields__:
            with pytest.raises(ValueError, match=f"HKFR_OVERRIDES: {name} takes"):
                config._apply(cfg, {name: value})


def test_each_config_takes_its_own_overrides_and_ignores_the_others():
    both = {"sample_spacing_m": 2.5, "point_rank": 2}
    assert config._apply(config.ElevationConfig(), both).sample_spacing_m == 2.5
    assert config._apply(config.AnalysisConfig(), both).point_rank == 2
    config._refuse_unknown(both, config.ElevationConfig, config.AnalysisConfig)
    assert config._apply(config.ElevationConfig(), {}) == config.ElevationConfig()


def test_a_mistyped_parameter_is_an_error_not_a_silent_baseline():
    with pytest.raises(ValueError) as err:
        config._refuse_unknown({"smooth_window": 12.5, "sample_spacing_m": 10},
                               config.ElevationConfig, config.AnalysisConfig)
    assert "unknown parameter(s) ['smooth_window']" in str(err.value)
    assert "smooth_window_m" in str(err.value)       # the names it could have been


@pytest.mark.parametrize("raw", ['[["point_rank", 1]]', "[1, 2]", '"point_rank"',
                                 "null", "5", "{not json"])
def test_overrides_must_be_a_json_object(monkeypatch, raw):
    """A list of pairs used to pass, because ``dict()`` takes one."""
    monkeypatch.setenv("HKFR_OVERRIDES", raw)
    with pytest.raises(ValueError, match="HKFR_OVERRIDES is not a JSON object"):
        config._overrides()
    monkeypatch.setenv("HKFR_OVERRIDES", '{"point_rank": 1}')
    assert config._overrides() == {"point_rank": 1}
    monkeypatch.delenv("HKFR_OVERRIDES")
    assert config._overrides() == {}


def test_the_checks_run_when_the_package_is_imported(tmp_path):
    code = "from hk_flat_routes import config as c; print(repr(c.ANALYSIS.point_rank))"
    done = _run(code, {"HKFR_ROOT": str(tmp_path),
                       "HKFR_OVERRIDES": json.dumps({"point_rank": "1"})}, cwd=ROOT)
    assert done.stdout.strip() == "1"
    for bad, said in (({"smooth_window": 12.5}, "unknown parameter(s) ['smooth_window']"),
                      ({"point_rank": 1.5}, "point_rank takes a whole number")):
        done = _run(code, {"HKFR_ROOT": str(tmp_path),
                           "HKFR_OVERRIDES": json.dumps(bad)}, cwd=ROOT)
        assert done.returncode != 0 and said in done.stderr


def test_the_sensitivity_grid_passes_the_checks():
    from hk_flat_routes.sensitivity import GRID
    for _tag, overrides, _why in GRID:
        config._refuse_unknown(overrides, config.ElevationConfig, config.AnalysisConfig)
        config._apply(config.ElevationConfig(), overrides)
        config._apply(config.AnalysisConfig(), overrides)


# ------------------------------------------------------- the GDAL helper
def test_gdal_is_told_the_environments_ca_bundle_and_nothing_else(tmp_path, monkeypatch):
    from hk_flat_routes.utils import configure_gdal_for_proxy
    for name in ("REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "GDAL_HTTP_CAINFO",
                 "GDAL_HTTP_PROXY", "HTTPS_PROXY", "https_proxy",
                 "GDAL_DISABLE_READDIR_ON_OPEN", "VSI_CACHE", "VSI_CACHE_SIZE",
                 "GDAL_CACHEMAX"):
        monkeypatch.delenv(name, raising=False)
    configure_gdal_for_proxy()
    assert "GDAL_HTTP_CAINFO" not in os.environ and "CURL_CA_BUNDLE" not in os.environ
    assert "GDAL_HTTP_PROXY" not in os.environ

    bundle = tmp_path / "roots.pem"
    bundle.write_text("roots")
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(bundle))
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:3128")
    configure_gdal_for_proxy()
    assert os.environ["GDAL_HTTP_CAINFO"] == str(bundle)
    assert os.environ["GDAL_HTTP_PROXY"] == "http://proxy.example:3128"
