"""The route page's sources, tested without a build.

``test_routepage.py`` drives the built page over the real graph.  The checks
here need neither: the notices and counts the page is written with, and the
routing engine (``web/engine.js``) run in a headless browser over a graph of
three or four nodes made up on the spot, where the right answer can be
worked out by hand.  The engine tests are skipped without Playwright and a
Chromium.
"""
from __future__ import annotations

import pytest

from hk_flat_routes import viz_interactive as vi
from hk_flat_routes.config import BIKE_PUSH_MULTIPLIER, ELEVATION


# ------------------------------------------------------- notices and counts
def test_the_engine_carries_both_mit_copyright_lines():
    head = vi._asset("engine.js").split("*/", 1)[0]
    assert "Copyright (c) 2026 sf-flat-routes contributors" in head
    assert "Copyright (c) 2026 flattenhk contributors" in head
    assert "LICENSE.txt" in head


def test_the_site_is_written_with_the_licences_its_files_are_under(tmp_path):
    vi._copy_licences(tmp_path)
    assert (tmp_path / "LICENSE-leaflet.txt").exists()
    text = (tmp_path / "LICENSE.txt").read_text(encoding="utf-8")
    # upstream's notice and the port's, each with its permission notice
    assert "Copyright (c) 2026 sf-flat-routes contributors" in text
    assert "Copyright (c) 2026 flattenhk contributors" in text
    assert text.count("Permission is hereby granted") >= 2


def test_the_licence_is_read_from_beside_the_code_and_may_be_absent(tmp_path, monkeypatch):
    """Not from the data root, which ``HKFR_ROOT`` may put anywhere; and a
    copy installed from a wheel has no LICENSE beside the package."""
    code, site = tmp_path / "code", tmp_path / "site"
    code.mkdir(); site.mkdir()
    (code / "LICENSE").write_text("the notices\n", encoding="utf-8")
    monkeypatch.setattr(vi, "CODE_ROOT", code)
    vi._copy_licences(site)
    assert (site / "LICENSE.txt").read_text(encoding="utf-8") == "the notices\n"
    (code / "LICENSE").unlink(); (site / "LICENSE.txt").unlink()
    vi._copy_licences(site)
    assert (site / "LICENSE-leaflet.txt").exists() and not (site / "LICENSE.txt").exists()


def _page(n_edges=1234, n_arcs=2468) -> str:
    return vi._route_page_html({"meta": {"n_edges": n_edges, "n_arcs": n_arcs}},
                               linked=False)


def test_how_this_works_counts_segments_not_directed_arcs():
    html = _page()
    assert "over 1,234\n    street and path segments" in html
    assert "2,468" not in html
    assert "/*__N_" not in html


def test_the_credits_name_both_government_portals():
    """The Government's terms ask for the portals to be named as well as
    the departments: in the credits, and on the map itself."""
    html = _page()
    credits = html[html.index("<summary>How this works</summary>"):html.index("</details>")]
    for name, link in (("CSDI Portal", "https://portal.csdi.gov.hk/"),
                       ("DATA.GOV.HK", "https://data.gov.hk/")):
        assert name in credits and f'href="{link}"' in credits
    control = vi._asset("simple.js")
    control = control[control.index("attributionControl.addAttribution("):]
    control = control[:control.index(");")]
    assert "CSDI Portal" in control and "portal.csdi.gov.hk" in control
    assert "DATA.GOV.HK" in control and "data.gov.hk" in control
    assert "OpenStreetMap" in control


def test_how_this_works_says_what_shorter_means_with_calm_streets_on():
    words = " ".join(_page().split())
    assert "the kilometres shown, which are the real ones, can fall" in words
    assert "(push)" in words


# ------------------------------------------------------------- the engine
#: Packs typed arrays into a Bundle the way webgraph.bundle lays them out,
#: and builds a Graph from a list of nodes and arcs.
_HELPERS = """
window.pack = (spec) => {
  const T = { u1: Uint8Array, i2: Int16Array, u2: Uint16Array, i4: Int32Array };
  let off = 0; const arrays = {}, parts = [];
  for (const [name, [t, vals]] of Object.entries(spec)) {
    const a = new T[t](vals);
    off = Math.ceil(off / 4) * 4;
    arrays[name] = { t, o: off, n: a.length };
    parts.push([off, new Uint8Array(a.buffer)]); off += a.byteLength;
  }
  const buf = new Uint8Array(off);
  for (const [o, b] of parts) buf.set(b, o);
  return new Bundle(buf, { arrays, strings: {} });
};
/* nodes: [lon, lat, z]; arcs: {from, to, len (m), gain (m), loss (m),
 * max (gradient), cls, modes (1 walk, 2 bike, 3 both), push} sorted by from */
window.tiny = (nodes, arcs, extra = {}) => {
  const indptr = new Array(nodes.length + 1).fill(0);
  for (const a of arcs) indptr[a.from + 1]++;
  for (let i = 0; i < nodes.length; i++) indptr[i + 1] += indptr[i];
  const used = (bit) => nodes.map((_, i) => arcs.some((a) => (a.modes & bit) && (a.from === i || a.to === i)) ? 1 : 0);
  const spec = {
    node_lon: ['i4', nodes.map((n) => Math.round(n[0] * 1e6))], node_lat: ['i4', nodes.map((n) => Math.round(n[1] * 1e6))],
    node_flags: ['u1', nodes.map((_, i) => used(1)[i] | (used(2)[i] << 1))],
    node_comp_walk: ['u1', extra.compWalk || used(1)], node_comp_bike: ['u1', extra.compBike || used(2)],
    node_elev: ['i2', nodes.map((n) => Math.round((n[2] || 0) * 20))], indptr: ['i4', indptr],
    arc_head: ['i4', arcs.map((a) => a.to)], arc_edge: ['i4', arcs.map((a, i) => i)],
    arc_len: ['u2', arcs.map((a) => Math.round(a.len * 20))], arc_gain: ['u2', arcs.map((a) => Math.round((a.gain || 0) * 100))],
    arc_loss: ['u2', arcs.map((a) => Math.round((a.loss || 0) * 100))], arc_maxgrade: ['i2', arcs.map((a) => Math.round((a.max || 0) * 10000))],
    arc_cls: ['u1', arcs.map((a) => a.cls || 0)], arc_flags: ['u1', arcs.map((a) => a.modes)],
    // the comfort weight of the class alone, as webgraph.py ships it
    edge_stress: ['u1', arcs.map((a) => [130, 100][a.cls || 0])],
  };
  if (extra.push) spec.arc_push = ['u1', arcs.map((a) => a.push ? 1 : 0)];
  const meta = Object.assign({
    n_nodes: nodes.length, n_arcs: arcs.length, n_edges: arcs.length, thresholds: [],
    classes: ['footway', 'residential'], multipliers: { walk: [1, 1], bike: [1.5, 1] },
    scales: { dm: 20, cm: 100, coord: 1e6, grade: 10000, stress: 100 },
    components: { walk: ['One'], bike: ['One'] },
  }, extra.meta || {});
  return new Graph(pack(spec), meta);
};
"""


@pytest.fixture(scope="module")
def engine():
    """A blank page with engine.js and the helpers above loaded."""
    sync_api = pytest.importorskip("playwright.sync_api",
                                   reason="playwright is not installed")
    errors: list = []
    with sync_api.sync_playwright() as pw:
        try:
            browser = pw.chromium.launch(args=["--no-sandbox", "--disable-gpu"])
        except Exception as exc:                            # pragma: no cover
            pytest.skip(f"no usable Chromium: {exc}")
        page = browser.new_page()
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.set_content("<!DOCTYPE html><html><body></body></html>")
        page.add_script_tag(content=vi._asset("engine.js"))
        page.add_script_tag(content=_HELPERS)
        assert not errors, errors
        yield page
        browser.close()


def test_the_nearest_node_is_the_nearest(engine):
    """Clusters of points with empty country between them, as Hong Kong's
    streets lie: the search must not stop at the first ring of cells that
    holds a candidate, and a degree of longitude is shorter than one of
    latitude.  Checked against a scan of every point, with and without a
    filter, inside the grid and well outside it."""
    out = engine.evaluate("""() => {
        let seed = 7; const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
        const xs = [], ys = [];
        for (let c = 0; c < 12; c++) {
            const cx = 113.85 + rnd() * 0.55, cy = 22.15 + rnd() * 0.40;
            for (let k = 0; k < 150; k++) { xs.push(cx + (rnd() - 0.5) * 0.02); ys.push(cy + (rnd() - 0.5) * 0.004); }
        }
        const X = Float64Array.from(xs), Y = Float64Array.from(ys), grid = new Grid(X, Y, 0.004);
        const kx = 111320 * Math.cos(22.35 * Math.PI / 180), ky = 110540;
        const metres = (i, x, y) => Math.hypot((X[i] - x) * kx, (Y[i] - y) * ky);
        const odd = (i) => i % 3 !== 0;
        let wrong = 0, wrongFiltered = 0, limitWrong = 0, tested = 0, worst = 0;
        for (let q = 0; q < 600; q++) {
            const x = 113.6 + rnd() * 1.1, y = 21.9 + rnd() * 0.9;
            for (const filter of [null, odd]) {
                let best = Infinity;
                for (let i = 0; i < X.length; i++) if (!filter || filter(i)) best = Math.min(best, metres(i, x, y));
                const got = grid.nearest(x, y, filter);
                const d = got < 0 ? Infinity : metres(got, x, y);
                // the grid measures with the cosine of its own mid-latitude: a metre of slack
                if (d > best + 1 || (filter && got >= 0 && !filter(got))) { if (filter) wrongFiltered++; else wrong++; worst = Math.max(worst, d - best); }
                const within = grid.nearest(x, y, filter, 400);
                if (best < 399 ? within < 0 || metres(within, x, y) > best + 1 : (best > 401 && within >= 0)) limitWrong++;
                tested++;
            }
        }
        return { tested, wrong, wrongFiltered, limitWrong, worst,
                 empty: new Grid(new Float64Array(0), new Float64Array(0), 0.004).nearest(114, 22) };
    }""")
    assert out["tested"] == 1200
    assert out["wrong"] == 0 and out["wrongFiltered"] == 0, out
    assert out["limitWrong"] == 0, out
    assert out["empty"] == -1


@pytest.mark.parametrize("told", [True, False])
def test_steepest_is_never_above_the_ceiling_gradients_are_held_to(engine, told):
    """A stub 2.3 m long that rises 2.52 m (Ladder Street has one) is 110%
    by net climb over length.  The figure is ``routing.arc_pitch``: held to
    the ceiling the bundle names, or in a bundle that names none to the
    arc's own stored maximum, which the build clips at that ceiling."""
    ceiling = ELEVATION.max_plausible_grade
    out = engine.evaluate("""([ceiling, told]) => {
        const meta = told ? { max_plausible_grade: ceiling, min_reliable_grade_length_m: 15 } : {};
        const g = tiny([[114.15, 22.28, 60], [114.1501, 22.28, 62.52], [114.1502, 22.28, 63.5], [114.1504, 22.28, 64.5]], [
            { from: 0, to: 1, len: 2.3, gain: 2.52, max: ceiling, modes: 3 },     // a stub over the ceiling
            { from: 1, to: 2, len: 10, gain: 1.0, max: 0.25, modes: 3 },         // short: its average stands in
            { from: 2, to: 3, len: 20, gain: 1.0, max: 0.30, modes: 3 },         // long: its stored maximum
            { from: 3, to: 2, len: 20, loss: 1.0, max: -0.02, modes: 3 },        // downhill
        ], { meta });
        return [0, 1, 2, 3].map((a) => g.summarise([a]).steepest).concat([g.summarise([0, 1, 2]).steepest, g.summarise([0]).max_grade]);
    }""", [ceiling, told])
    stub, short, long_, down, route, max_grade = out
    assert stub == pytest.approx(ceiling)
    assert short == pytest.approx(0.10)
    assert long_ == pytest.approx(0.30)
    assert down == 0
    assert route == pytest.approx(ceiling) and max_grade == pytest.approx(ceiling)


_PUSH_GRAPH = """(push) => {
    // A to B by a 40 m footpath (a pushed link when ``push`` is set), or
    // round by C on 100 m of street
    const g = tiny([[114.15, 22.28], [114.1504, 22.28], [114.1502, 22.2804]], [
        { from: 0, to: 1, len: 40, cls: 0, modes: 3, push: true },
        { from: 0, to: 2, len: 50, cls: 1, modes: 3 },
        { from: 2, to: 1, len: 50, cls: 1, modes: 3 },
    ], push ? { push: true, meta: { push_multiplier: push } } : {});
    const classes = { alpha: 0, beta: 0, gamma: 0, penalties: [], extreme: 0, use_class_multiplier: true };
    const plain = Object.assign({}, classes, { use_class_multiplier: false });
    const calm = Object.assign({}, plain, { stress: true });
    const via = (mode, w) => { const r = g.route(0, 1, mode, w); return { arcs: r.arcs, cost: r.cost, same: g.pathCost(r.arcs, mode, w) }; };
    return { bike: via('bike', classes), walk: via('walk', classes), bikePlain: via('bike', plain), bikeCalm: via('bike', calm),
             direct: g.pathCost([0], 'bike', classes), directCalm: g.pathCost([0], 'bike', calm), stressLen: g.arcLenStress[0] / g.DM,
             directBoth: g.pathCost([0], 'bike', Object.assign({}, classes, { stress: true })),
             stressM: g.summarise([0]).stress_m, distance: g.summarise([0]).distance_m,
             many: g.distances(0, 'bike', classes, [1])[0], hasPush: !!g.arcPush, mult: g.pushMult };
}"""


def test_a_pushed_link_counts_several_times_its_length_by_bike(engine):
    """Mirrors routing.edge_costs: the push multiplier on the length term in
    bike mode wherever the class multipliers apply.  The comfort-weighted
    length the route page's calm option searches on has it once as well
    (bikeways.stress): the bundle carries the class's weight alone, and the
    engine multiplies the push factor in from ``arc_push``."""
    k = BIKE_PUSH_MULTIPLIER
    out = engine.evaluate(_PUSH_GRAPH, k)
    assert out["hasPush"] and out["mult"] == k
    # 40 m of footway at the bike class multiplier of 1.5, pushed
    assert out["direct"] == pytest.approx(40 * 1.5 * k)
    assert out["bike"]["arcs"] == [1, 2] and out["bike"]["cost"] == pytest.approx(100)
    assert out["many"] == pytest.approx(100)
    # a walker is not pushing anything
    assert out["walk"]["arcs"] == [0] and out["walk"]["cost"] == pytest.approx(40)
    # no comfort multipliers, no push multiplier: the true shortest
    assert out["bikePlain"]["arcs"] == [0] and out["bikePlain"]["cost"] == pytest.approx(40)
    # calm streets: 40 m at the footway's comfort weight of 1.3, pushed
    assert out["stressLen"] == pytest.approx(40 * 1.3 * k)
    assert out["directCalm"] == pytest.approx(40 * 1.3 * k)
    assert out["stressM"] == pytest.approx(40 * 1.3 * k) and out["distance"] == pytest.approx(40)
    assert out["bikeCalm"]["arcs"] == [1, 2]
    # and once only, were the class multipliers ever applied to that length
    assert out["directBoth"] == pytest.approx(40 * 1.3 * k * 1.5)
    for r in (out["bike"], out["walk"], out["bikePlain"], out["bikeCalm"]):
        assert r["cost"] == pytest.approx(r["same"])


def test_a_bundle_without_pushed_links_routes_as_before(engine):
    out = engine.evaluate(_PUSH_GRAPH, 0)
    assert not out["hasPush"] and out["mult"] == 1
    assert out["direct"] == pytest.approx(60)
    assert out["bike"]["arcs"] == [0]
    assert out["stressLen"] == pytest.approx(52) and out["bikeCalm"]["arcs"] == [0]


def test_component_sizes_come_from_the_build_or_are_counted(engine):
    out = engine.evaluate("""() => {
        const nodes = [[114.15, 22.28], [114.151, 22.28], [114.152, 22.28], [114.153, 22.28], [114.154, 22.28]];
        const arcs = [{ from: 0, to: 1, len: 50, modes: 3 }, { from: 1, to: 0, len: 50, modes: 3 },
                      { from: 2, to: 3, len: 50, modes: 1 }, { from: 3, to: 4, len: 50, modes: 1 }];
        const comps = { compWalk: [2, 2, 1, 1, 1], compBike: [1, 1, 0, 0, 0] };
        const counted = tiny(nodes, arcs, comps);
        const given = tiny(nodes, arcs, Object.assign({ meta: { component_nodes: { walk: [30, 20], bike: [7] } } }, comps));
        return { counted: [counted.compNodes(1, 'walk'), counted.compNodes(2, 'walk'), counted.compNodes(1, 'bike'), counted.compNodes(0, 'bike'), counted.compNodes(3, 'walk')],
                 given: [given.compNodes(1, 'walk'), given.compNodes(2, 'walk'), given.compNodes(1, 'bike'), given.compNodes(0, 'walk')] };
    }""")
    assert out["counted"] == [3, 2, 2, 0, 0]
    assert out["given"] == [30, 20, 7, 0]
