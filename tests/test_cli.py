"""Tests for the command line: encodings, ``--force`` and the help texts.

The stages themselves are replaced by stand-ins, so nothing here reads or
writes the build.
"""
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from hk_flat_routes import __main__ as cli

ROOT = Path(__file__).resolve().parent.parent
COMMANDS = ["sources", "download", "build-network", "site", "analyze", "validate",
            "map", "report", "all", "sensitivity", "summarize", "route"]


# ---------------------------------------------------------------- help
def test_every_subcommand_has_help():
    parser = cli.build_parser()
    sub = next(a for a in parser._actions if a.dest == "cmd")
    assert sorted(sub.choices) == sorted(COMMANDS)
    assert sorted(cli._DISPATCH) == sorted(COMMANDS)
    for cmd in [[]] + [[c] for c in COMMANDS]:
        with pytest.raises(SystemExit) as stop:
            parser.parse_args(cmd + ["--help"])
        assert stop.value.code == 0


# ------------------------------------------------------------ encodings
@pytest.mark.parametrize("encoding", ["cp1252", "cp950", "ascii"])
@pytest.mark.parametrize("args", [["--help"], ["sources"]])
def test_output_survives_a_console_that_is_not_utf8(tmp_path, encoding, args):
    """Both texts hold Chinese; cp1252 is redirected output on Western Windows."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("HKFR_")}
    env.update(PYTHONIOENCODING=encoding, PYTHONPATH=str(ROOT), HKFR_ROOT=str(tmp_path))
    done = subprocess.run([sys.executable, "-m", "hk_flat_routes", *args],
                          env=env, cwd=tmp_path, capture_output=True)
    assert done.returncode == 0, done.stderr.decode("ascii", "replace")[-400:]
    text = done.stdout.decode(encoding)
    assert "python -m hk_flat_routes" in text or "Data sources" in text
    if encoding == "cp950":                         # Traditional Chinese Windows
        assert "旺角" in text                       # shown, not replaced


def test_streams_that_already_cope_are_left_alone(monkeypatch):
    class Stream:
        def __init__(self, errors):
            self.errors, self.calls = errors, []

        def reconfigure(self, **kwargs):
            self.calls.append(kwargs)

    out, err = Stream("strict"), Stream("backslashreplace")
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)
    cli._tolerant_streams()
    assert out.calls == [{"errors": "replace"}] and err.calls == []
    monkeypatch.setattr(sys, "stdout", object())    # a stream with no such method
    cli._tolerant_streams()


def _as_a_western_windows_locale(monkeypatch):
    """Make a bare ``read_text()`` decode as cp1252, as it does there."""
    real = Path.read_text

    def read_text(self, encoding=None, errors=None, **kwargs):
        return real(self, encoding=encoding or "cp1252", errors=errors, **kwargs)
    monkeypatch.setattr(Path, "read_text", read_text)


def _stand_in(monkeypatch, name: str, **attrs):
    """Put a stand-in for a submodule where ``from .x import y`` finds it."""
    module = types.SimpleNamespace(**attrs)
    monkeypatch.setitem(sys.modules, f"hk_flat_routes.{name}", module)
    return module


#: Text of the kind both reports hold: Chinese, a dash, a degree sign.  One
#: byte of 道 in UTF-8 is 0x81, which cp1252 has no character for, so read
#: as cp1252 the file does not come out garbled: it cannot be read at all.
REPORT = "# Validation\n\n彌敦道 Nathan Road — 0.5° … 旺角\n"


def test_validate_prints_its_report_whatever_the_locale(tmp_path, monkeypatch, capsys):
    report = tmp_path / "validation_report.md"
    report.write_text(REPORT, encoding="utf-8")
    _stand_in(monkeypatch, "pipeline", build_context=lambda **k: "ctx")
    _stand_in(monkeypatch, "corridors", CORRIDORS_GPKG=tmp_path / "none.gpkg")
    _stand_in(monkeypatch, "validate", VALIDATION_MD=report,
              run_validation=lambda ctx, cor: None)
    _as_a_western_windows_locale(monkeypatch)
    assert cli.main(["validate"]) == 0
    assert capsys.readouterr().out == REPORT + "\n"


def test_sensitivity_prints_its_report_whatever_the_locale(tmp_path, monkeypatch, capsys):
    report = tmp_path / "sensitivity.md"
    report.write_text(REPORT, encoding="utf-8")
    seen = {}
    _stand_in(monkeypatch, "sensitivity", SENS_MD=report,
              run_sensitivity=lambda force, only: seen.update(force=force, only=only))
    _as_a_western_windows_locale(monkeypatch)
    assert cli.main(["sensitivity", "--force", "--only", "baseline"]) == 0
    assert capsys.readouterr().out == REPORT + "\n"
    assert seen == {"force": True, "only": ["baseline"]}


# ---------------------------------------------------------------- force
@pytest.fixture
def stages(tmp_path, monkeypatch):
    """Stand-ins for every stage ``site`` and ``map`` call, recording the calls."""
    calls = []

    def record(name, result=None):
        def stage(*args, **kwargs):
            calls.append((name, kwargs))
            return result
        return stage

    ctx = types.SimpleNamespace(edges=[], directed=[], neighborhoods=[], graphs={})
    _stand_in(monkeypatch, "pipeline", build_context=record("build_context", ctx),
              run_analysis=record("run_analysis", {
                  k: [] for k in ("pairs", "corridors", "passes", "barriers")}))
    _stand_in(monkeypatch, "elevation", build_dem_overview=record("build_dem_overview"))
    _stand_in(monkeypatch, "viz_interactive",
              make_route_page=record("make_route_page", tmp_path / "index.html"),
              make_interactive_map=record("make_interactive_map"))
    _stand_in(monkeypatch, "viz_static", make_backbone_map=record("make_backbone_map"),
              make_grade_map=record("make_grade_map"))
    _stand_in(monkeypatch, "corridors", CORRIDORS_GPKG=tmp_path / "corridors.gpkg")
    _stand_in(monkeypatch, "passes", read_passes=lambda: [], read_barriers=lambda: [],
              read_basins=lambda: [])
    monkeypatch.setitem(sys.modules, "geopandas",
                        types.SimpleNamespace(read_file=lambda path: []))
    return calls


@pytest.mark.parametrize("cmd, after", [("site", "make_route_page"),
                                        ("map", "make_backbone_map")])
def test_force_rebuilds_the_terrain_overview_before_it_is_drawn(stages, cmd, after):
    assert cli.main([cmd, "--force"]) == 0
    names = [name for name, _ in stages]
    assert ("build_dem_overview", {"force": True}) in stages
    assert names.index("build_dem_overview") < names.index(after)
    # the street network is another stage's to rebuild
    assert ("build_context", {}) in stages


@pytest.mark.parametrize("cmd", ["site", "map"])
def test_without_force_the_overview_is_left_to_its_cache(stages, cmd):
    assert cli.main([cmd]) == 0
    assert "build_dem_overview" not in [name for name, _ in stages]


def test_a_machine_without_tiles_can_still_force_the_page(stages, monkeypatch, caplog):
    def no_tiles(force=False):
        raise FileNotFoundError("data/raw/dem_index_2020.geojson")
    sys.modules["hk_flat_routes.elevation"].build_dem_overview = no_tiles
    with caplog.at_level("WARNING"):
        assert cli.main(["site", "--force"]) == 0
    assert "no terrain tiles to rebuild the overview from" in caplog.text
    assert "make_route_page" in [name for name, _ in stages]


@pytest.mark.parametrize("cmd", ["site", "map"])
def test_a_forced_rebuild_that_fails_on_a_bad_tile_is_not_passed_over(stages, cmd):
    """Only the absence of tiles is tolerated: a tile that cannot be read
    would otherwise leave the stale overview in a build that was forced."""
    import zipfile

    def bad_tile(force=False):
        raise zipfile.BadZipFile("File is not a zip file")
    sys.modules["hk_flat_routes.elevation"].build_dem_overview = bad_tile
    with pytest.raises(zipfile.BadZipFile):
        cli.main([cmd, "--force"])
    drawn = {"make_route_page", "make_backbone_map", "make_interactive_map"}
    assert not drawn & {name for name, _ in stages}


def test_the_real_stages_take_what_the_command_line_passes():
    """The stand-ins here accept anything.  The functions they stand in for
    belong to other modules and must take ``force`` by that name, or
    ``--force`` ends in a ``TypeError`` after the stage before it has run."""
    import inspect

    from hk_flat_routes import download, elevation, neighborhoods, pipeline, sensitivity
    for fn, passed in ((download.download_all, ["force"]),
                       (pipeline.build_context, ["force"]),
                       (pipeline.run_analysis, ["force"]),
                       (elevation.build_dem_overview, ["force"]),
                       (neighborhoods.load_neighborhoods, ["force"]),
                       (sensitivity.run_sensitivity, ["force", "only"])):
        taken = inspect.signature(fn).parameters
        assert set(passed) <= set(taken), f"{fn.__module__}.{fn.__name__}{tuple(taken)}"


@pytest.mark.parametrize("flag, forced", [([], False), (["--force"], True)])
def test_force_reaches_the_stages_that_cache(stages, monkeypatch, flag, forced):
    assert cli.main(["build-network", *flag]) == 0
    assert stages[-1] == ("build_context", {"force": forced})
    assert cli.main(["analyze", *flag]) == 0
    assert stages[-1] == ("run_analysis", {"force": forced})

    seen = {}
    _stand_in(monkeypatch, "download",
              download_all=lambda force: seen.update(force=force))
    assert cli.main(["download", *flag]) == 0
    assert seen == {"force": forced}
