"""Tests that the packaging and the documents say one thing: dependencies,
data sizes, the terms data is used under, the environment variables."""
import ast
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "hk_flat_routes"


def _text(*parts) -> str:
    return ROOT.joinpath(*parts).read_text(encoding="utf-8")


def _flat(text: str) -> str:
    """Text with its line breaks and indentation folded to single spaces."""
    return " ".join(text.split())


# ---------------------------------------------------------- dependencies
def _declared() -> dict[str, str]:
    block = re.search(r"^dependencies = \[(.*?)^\]", _text("pyproject.toml"), re.S | re.M)
    return dict(re.match(r"([A-Za-z0-9_.-]+)(.*)", d).groups()
                for d in re.findall(r'"([^"]+)"', block.group(1)))


def _imported() -> set[str]:
    found = set()
    for folder in (PACKAGE, ROOT / "tests", ROOT / "scripts"):
        for path in folder.rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Import):
                    found |= {a.name.split(".")[0] for a in node.names}
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    found.add(node.module.split(".")[0])
    return found - set(sys.stdlib_module_names) - {"hk_flat_routes"}


def test_requirements_txt_is_pyproject_plus_pytest():
    lines = [line.strip() for line in _text("requirements.txt").splitlines() if line.strip()]
    declared = [name + spec for name, spec in _declared().items()]
    assert lines == declared + ["pytest>=7.4"]


def test_nothing_is_declared_that_nothing_imports():
    #: packages used through another's interface, not imported by name
    indirect = {"pyogrio": "geopandas reads files with it",
                "aiohttp": "fsspec's https filesystem needs it"}
    names = {"pillow": "PIL"}
    imported = _imported()
    for dep in _declared():
        assert dep in indirect or names.get(dep, dep) in imported, f"{dep} is never imported"
    for gone in ("rtree", "folium", "branca"):
        assert gone not in _declared() and gone not in imported


def test_everything_imported_is_declared():
    optional = {"pytest", "playwright", "imageio_ffmpeg"}   # dev extra; a QA script's fallback
    names = {"PIL": "pillow"}
    declared = set(_declared())
    for module in _imported() - optional:
        assert names.get(module, module) in declared, f"{module} is not declared"


def test_pandas_floor_is_the_version_the_code_needs():
    """``groupby().apply(include_groups=...)`` arrived in pandas 2.2."""
    assert "include_groups" in _text("hk_flat_routes", "passes.py")
    floor = tuple(int(x) for x in re.findall(r"\d+", _declared()["pandas"]))
    assert floor >= (2, 2)


def test_the_wheel_carries_the_vendored_files():
    assert '"vendor/*"' in _text("pyproject.toml")
    from hk_flat_routes.download import IGEOCOM_INTERMEDIATE_PEM
    assert IGEOCOM_INTERMEDIATE_PEM.parent == PACKAGE / "vendor"
    assert IGEOCOM_INTERMEDIATE_PEM.exists()


# ------------------------------------------------------------------ sizes
def test_sizes_are_logged_in_the_units_the_documents_use():
    from hk_flat_routes.utils import human_bytes
    assert human_bytes(14_127_079_492) == "14.1GB"      # the terrain tiles
    assert human_bytes(17_057_436) == "17.1MB"          # the explorer page
    assert human_bytes(999) == "999.0B" and human_bytes(1000) == "1.0kB"


@pytest.mark.parametrize("parts, said", [
    (("README.md",), "14.1 GB in 3,253 files, about 14.3 GB"),
    (("README.md",), "(~14.3 GB)"),
    (("README.md",), "3,253 tiles (14.1 GB)"),
    (("data", "README.md"), "about 14.3 GB in all, 14.1 GB of it the terrain tiles"),
    ((".gitignore",), "14.1 GB"),
    ((".github", "workflows", "pages.yml"), "about 14.3 GB"),
    (("hk_flat_routes", "sources.py"), "14.1 GB zipped, in 3,253 files"),
])
def test_the_download_is_one_size_everywhere(parts, said):
    text = _flat(_text(*parts).replace("# ", ""))
    assert said in text
    for stale in ("13 GB", "13.2 GB", "14.5 GB", "14 GB"):
        assert stale not in text


def test_the_extracts_to_keep_are_the_ones_the_network_is_built_from():
    """Four, since the land use closes the gated sites; and one size for them."""
    from hk_flat_routes import download as D
    from hk_flat_routes import sources
    network = [D.SEGMENTS_PARQUET, D.CONNECTORS_PARQUET, D.DIVISIONS_PARQUET,
               D.BASE_PARQUETS["land_use"]]
    for parts in (("README.md",), ("data", "README.md")):
        text = _flat(_text(*parts))
        assert "four files" in text and "three files" not in text
        for path in network:
            assert path.name in text, f"{parts[-1]} does not list {path.name}"
    assert "which four files to keep" in _flat(_text(".gitignore").replace("# ", ""))
    assert not sources.DATASETS_BY_KEY["overture_land_use"].optional
    if all(path.exists() for path in network):      # the figure, where it can be checked
        size = sum(path.stat().st_size for path in network) / 1e6
        assert f"({size:.1f} MB together" in _flat(_text("README.md"))


# ------------------------------------------------------------------ terms
def test_the_bench_marks_terms_are_stated_once_and_repeated():
    from hk_flat_routes import sources
    assert sources.DATASETS_BY_KEY["bench_marks"].licence == sources.BENCHMARKS_TERMS
    for said in ("CSDI Portal Terms and Conditions of Use",
                 "a dataset of the CSDI Portal, fetched here from the Lands "
                 "Department's own copy",
                 "not redistributed beyond the stations quoted in the validation report"):
        assert said in sources.BENCHMARKS_TERMS
        for parts in (("README.md",), ("LICENSE",)):
            assert said in _flat(_text(*parts)), f"{parts[-1]} does not say {said!r}"


def test_every_adapted_project_has_its_notice():
    licence = _text("LICENSE")
    for holder in ("flattenhk contributors", "sf-flat-routes contributors",
                   "flatten_zrh contributors"):
        assert f"Copyright (c) 2026 {holder}" in licence
    assert licence.count("Permission is hereby granted, free of charge") == 3
    assert "flatten_zrh" in _text("scripts", "screenshots.py")
    assert "flatten_zrh" in _text("README.md")


def test_no_tomtom_segment_is_said_to_be_used():
    from hk_flat_routes import config
    if "unknown" in config.WALK.allowed_classes | config.BIKE.allowed_classes:
        pytest.skip("class unknown is still routable in this checkout")
    for parts in (("README.md",), ("LICENSE",), ("hk_flat_routes", "sources.py")):
        text = _flat(_text(*parts).replace('" "', ""))
        assert "TomTom" in text and "a few of the Overture records come from TomTom" not in text


def test_the_licence_names_only_files_the_repository_holds():
    licence = _flat(_text("LICENSE"))
    assert "58 of the 151 plans" in licence and "56 of the 151" not in licence
    listed = licence.split("Derived street geometry in this repository")[1].split("is a Derived")[0]
    assert "data/processed" not in listed
    assert "data/processed/" in _text(".gitignore")


def test_the_gradient_invariant_is_named_for_what_it_tests():
    """It tests the clipped maximum of each edge, not every gradient there is."""
    validate = _flat(_text("hk_flat_routes", "validate.py"))
    assert 'add(f"no maximum gradient above the' in validate
    assert "no maximum gradient over the 60% ceiling" in _flat(_text("README.md"))
    for text in (validate, _flat(_text("README.md"))):
        assert "no gradient above the" not in text and "no gradient over the" not in text


# ------------------------------------------------------------------ rules
def test_the_readme_states_the_rules_with_the_figures_the_code_uses():
    """The README gives the network's, the terrain model's and the page's
    rule parameters in words.  Each sentence is tied here to the constant
    it describes, so that one cannot move without the other."""
    import inspect

    from hk_flat_routes import config, network
    readme = _flat(_text("README.md"))
    side = inspect.signature(network.select_sidepaths).parameters
    e = config.ElevationConfig()                    # the defaults, whatever the environment
    stated = [
        (config.ROAD_TUNNEL_MAX_M, 1000.0, "A run longer than 1 km is closed by rule"),
        (config.ROAD_TUNNEL_CLASSES, {"motorway", "trunk", "primary", "secondary"},
         "A tunnel on a motorway, trunk, primary or secondary road is judged whole"),
        (config.FOOT_PERMIT_REQUIRED_RAMP_CLASSES, {"trunk"},
         "Its slip roads and bridges are closed to them unless the data permits walking by name"),
        (config.CLOSED_LAND_STREET_CLASSES, {"service"}, "Service roads inside the container terminals"),
        ("unknown" in config.NEVER_ROUTABLE_CLASSES, True, "Class `unknown` is dropped."),
        (config.BIKE_PERMIT_REQUIRED_CLASSES, {"footway", "pedestrian", "path"},
         "may use a footway, a pedestrian street or a path only where the data permits bicycles explicitly"),
        (config.BIKE_PUSH_MAX_M, 50.0, "a walkable run of footway of at most 50 m"),
        (config.BIKE_PUSH_MULTIPLIER, 3.0, "the link costs three times its length"),
        (side["radius"].default, 3000.0, "A run is followed for up to 3 km"),
        ((side["stretch"].default, side["slack"].default), (1.3, 30.0),
         "more than 1.3 times (and 30 m) farther apart"),
        (config.MIN_COMPONENT_NODES, 40, "strongly connected components of 40 nodes or more"),
        (e.dem_sigma_m, 1.5, "A Gaussian of sigma = 1.5 m"),
        ((e.lateral_search_m, e.lateral_min_cross_slope), (6.0, 0.06),
         "(more than 6% between two metres left and two metres right), each sample is taken "
         "at the flattest spot within 6 m of the centreline"),
        ((e.abutment_blank_m, e.abutment_blank_m + e.abutment_fit_m, e.abutment_min_shift_m),
         (10.0, 35.0, 1.0),
         "extrapolated from the ten to thirty-five metres of approach before it, replacing "
         "the sampled value when they differ by more than a metre"),
        (e.despike_window_m, 35.0, "A running median over 35 m"),
        (e.min_ground_elev_m, -1.0, "Ground below -1 mPD is treated as missing"),
        ((e.sample_spacing_m, 2 * e.smooth_window_m, e.gain_deadband_m), (5.0, 50.0, 0.5),
         "sampling at 5 m, Savitzky-Golay over 50 m per contiguous street, one elevation "
         "per intersection, and the 0.5 m dead-band"),
    ]
    for value, expected, said in stated:
        assert value == expected, f"the README says {said!r}"
        assert said in readme, f"the README no longer says {said!r}"

    # the page's own constants, for the sentence on an endpoint that is moved
    page = _text("hk_flat_routes", "web", "simple.js")
    for name, value, said in (("RESNAP_M", "400", "when a street of it is within 400 m"),
                              ("MOVED_NOTE_M", "50", "how far it moved when that is more than 50 m"),
                              ("MAX_ROUTES", "30", "with thirty routes on the slider in all")):
        assert re.search(rf"\b{name} = (\d+)", page).group(1) == value, name
        assert said in readme, f"the README no longer says {said!r}"


# ------------------------------------------------------------- environment
def test_every_environment_variable_the_package_reads_is_documented():
    read = set()
    for path in PACKAGE.glob("*.py"):
        read |= set(re.findall(r'environ(?:\.get)?[\[(]\s*"(HKFR_[A-Z_]+)"', path.read_text(encoding="utf-8")))
    assert {"HKFR_ROOT", "HKFR_RUN_DIR", "HKFR_OVERRIDES", "HKFR_OVERTURE_RELEASE"} <= read
    readme = _text("README.md")
    for name in sorted(read):
        assert f"`{name}" in readme, f"{name} is read but the README does not mention it"
    for name in ("HKFR_ROOT", "HKFR_RUN_DIR", "HKFR_OVERRIDES", "HKFR_OVERTURE_RELEASE"):
        assert f"| `{name}` |" in readme            # the table


def test_cached_data_is_not_tracked():
    ignored = _text(".gitignore").split()
    assert {"data/raw/", "data/processed/", "data/sensitivity/"} <= set(ignored)


def test_the_qa_scripts_carry_nothing_from_another_machine():
    for name in ("qa_loop_mode.py", "qa_map_cuts.py"):
        text = _text("tests", name)
        assert "from hk_flat_routes.config import SITE_DIR" in text
        for stale in ("/home/", "/opt/", "/tmp/", "-122.", "Dolores", "Marina", " mi loop"):
            assert stale not in text, f"{name} still holds {stale!r}"


# --------------------------------------------------- what is left to fill in
def test_the_readme_has_no_placeholder_left():
    """A ``{{NAME}}`` placeholder or a note in an HTML comment means a figure
    still has to be filled in from the outputs."""
    readme = _text("README.md")
    left = re.findall(r"<!--.*?-->", readme, re.S) + re.findall(r"\{\{.*?\}\}", readme)
    assert not left, f"{len(left)} to resolve, the first: {left[0][:120]}"
