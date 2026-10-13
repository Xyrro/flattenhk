"""End-to-end test of the explorer's in-browser router.

The map does its own routing, so the thing most worth guarding is that it
agrees with Python.  This test drives the real HTML file in a headless
browser, hands the JavaScript router the *exact* arc sequences Python chose,
and checks two things:

1. the metrics JavaScript computes for Python's own path match Python's
   figures (this isolates quantisation error from route choice);
2. the route JavaScript finds for itself is never more expensive, under its
   own cost model, than the one Python found (this is the optimality check).

Routes may legitimately differ where two paths tie on cost -- the two
implementations break ties differently -- so path identity is reported but
not asserted.

Hong Kong adds a third thing to agree on: where there is *no* route.  Every
sample pair lies on one landmass, and one more pair crosses the harbour;
Python and the browser must both say that pair cannot be walked or ridden,
and the page must say why instead of showing an empty panel.

Skipped unless Playwright, a Chromium build and a built map are all present:

    pip install playwright && playwright install chromium
    python -m hk_flat_routes map

The page is looked for in ``config.OUTPUT_DIR``, so a run redirected with
``HKFR_RUN_DIR`` tests its own map.
"""
from __future__ import annotations

import functools
import http.server
import threading

import pytest

from hk_flat_routes.config import (MIN_RELIABLE_GRADE_LENGTH_M, OUTPUT_DIR,
                                   PROCESSED_DIR)

playwright = pytest.importorskip("playwright.sync_api",
                                 reason="playwright is not installed")

#: The explorer, wherever this run writes its outputs.
EXPLORER_HTML = OUTPUT_DIR / "hk_flat_routes_map.html"

#: Pairs chosen to span flat and hilly ground and both modes, each within
#: one landmass: Hong Kong Island first, then Kowloon.  A pair is skipped
#: when a build does not reach one of its ends.
PAIRS = [
    ("Pok Fu Lam", "Shau Kei Wan"), ("Central District", "The Peak Area"),
    ("Aberdeen & Ap Lei Chau", "Central District"), ("Stanley", "Wan Chai"),
    ("Kennedy Town & Mount Davis", "Chai Wan"), ("Mid-Levels West", "Causeway Bay"),
    ("Tsim Sha Tsui", "Kowloon Tong"), ("Mong Kok", "Kwun Tong (South)"),
    ("Kai Tak", "Lai Chi Kok"), ("Hung Hom", "Tsz Wan Shan, Diamond Hill & San Po Kong"),
]

#: Across Victoria Harbour: no street or path joins the two, in either mode.
CROSS_HARBOUR = ("Central District", "Tsim Sha Tsui")


pytestmark = [
    pytest.mark.skipif(not EXPLORER_HTML.exists(),
                       reason="map not built; run `python -m hk_flat_routes map`"),
    pytest.mark.skipif(not (PROCESSED_DIR / "edges_directed.parquet").exists(),
                       reason="processed data not built"),
]


@pytest.fixture(scope="module")
def context():
    from hk_flat_routes.pipeline import build_context
    from hk_flat_routes.utils import configure_gdal_for_proxy

    configure_gdal_for_proxy()
    return build_context()


def _points(ctx, mode) -> dict:
    """Neighbourhood -> access node, where the network reaches one."""
    pts = ctx.points[mode]
    pts = pts[pts["node"].notna()]
    return dict(zip(pts["neighborhood"], pts["node"]))


@pytest.fixture(scope="module")
def reference(context):
    """Python's own routes for the sample pairs, with full arc sequences."""
    from hk_flat_routes.config import ROUTING_PROFILES
    from hk_flat_routes.routing import route

    rows = []
    for mode, graph in context.graphs.items():
        pts = _points(context, mode)
        for pname, w in ROUTING_PROFILES.items():
            cost = graph.build_costs(w)
            for o, d in PAIRS:
                if o not in pts or d not in pts:
                    continue
                arcs, s = route(graph, pts[o], pts[d], w, arc_cost=cost)
                if not arcs:
                    continue
                t = graph.table.iloc[arcs]
                rows.append({
                    "mode": mode, "profile": pname, "o": o, "d": d,
                    "path": [[int(e), 1 if dr == "rev" else 0]
                             for e, dr in zip(t["edge_id"], t["direction"])],
                    "cost": float(cost[arcs].sum()),
                    "distance_m": float(s["distance_m"]),
                    "gain": float(s["elev_gain_m"]),
                    "loss": float(s["elev_loss_m"]),
                    # the route's steepest pitch (``routing.arc_pitch``),
                    # and the largest stored gradient of any of its arcs
                    "pitch": float(s["max_grade"]),
                    "maxabs": float(s["max_abs_grade_any"]),
                    "shortest_arc": float(t["length_m"].min()),
                    "avgg": float(s["avg_abs_grade"]),
                    "th": [float(s[f"d_above_{t_}"]) for t_ in (3, 5, 8, 10, 15)],
                })
    if not rows:
        pytest.skip("this build reaches none of the sample pairs")
    return rows


@pytest.fixture(scope="module")
def served():
    """The outputs directory over HTTP: some browsers will not run a page's
    scripts from ``file://``, and the page needs nothing but itself."""
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args, **kwargs):
            pass

    handler = functools.partial(Quiet, directory=str(OUTPUT_DIR))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/{EXPLORER_HTML.name}"
    srv.shutdown()


@pytest.fixture(scope="module")
def browser():
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        try:
            b = pw.chromium.launch(args=["--no-sandbox", "--disable-gpu"])
        except Exception as exc:                           # pragma: no cover
            pytest.skip(f"no usable Chromium: {exc}")
        yield b
        b.close()


def _open(browser, url, errors, requests=None):
    """A fresh page with the explorer loaded and its first route drawn."""
    page = browser.new_page(viewport={"width": 1400, "height": 900})
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("console", lambda m: errors.append(m.text)
            if m.type == "error" else None)
    if requests is not None:
        page.on("request", lambda r: requests.append(r.url))
    page.goto(url, wait_until="load", timeout=240_000)
    page.wait_for_function(
        "window.App && App.graph && App.map"
        " && document.querySelector('#res .big, #res .warn, #res .sub')",
        timeout=240_000)
    return page


@pytest.fixture(scope="module")
def browser_results(reference, browser, served):
    """Drive the real HTML file and collect the JavaScript router's answers."""
    errors: list = []
    requests: list = []
    page = _open(browser, served, errors, requests)
    out = page.evaluate(_BROWSER_SCRIPT, reference)
    info = page.evaluate(
        "() => ({nodes: App.graph.n, arcs: App.graph.m, edges: App.geom.nEdges,"
        " drawn: App.network.lastDraw.edges, zoom: App.map.getZoom(),"
        " text: document.getElementById('side').innerText,"
        " attribution: document.querySelector('.leaflet-control-attribution').innerText})")
    page.close()
    return out, info, errors, requests


_BROWSER_SCRIPT = """(ref) => {
    const res = [];
    for (const r of ref) {
        const g = App.graph, w = App.meta.profiles[r.profile];
        const arcs = r.path.map(([e, rev]) => g.arcOf(e, rev));
        if (arcs.some(a => a < 0)) { res.push({...r, miss: 'arc'}); continue; }
        const sPy = g.summarise(arcs);
        const costPy = g.pathCost(arcs, r.mode, w);
        const src = App.pointNode(r.o, r.mode), dst = App.pointNode(r.d, r.mode);
        const rt = g.route(src, dst, r.mode, w);
        if (!rt) { res.push({...r, miss: 'route'}); continue; }
        const sJs = g.summarise(rt.arcs);
        res.push({
            mode: r.mode, profile: r.profile, o: r.o, d: r.d,
            same: arcs.length === rt.arcs.length
                  && arcs.every((a, i) => a === rt.arcs[i]),
            oneLandmass: g.comp(src, r.mode) > 0 && g.comp(src, r.mode) === g.comp(dst, r.mode),
            onPyPath: {distance_m: sPy.distance_m, gain: sPy.elev_gain_m,
                       loss: sPy.elev_loss_m, pitch: sPy.steepest,
                       maxabs: Math.max(...arcs.map(a => Math.abs(g.arcMaxGrade[a]))) / g.GRADE,
                       avgg: sPy.avg_abs_grade, th: sPy.thresholds,
                       cost: costPy},
            ownRoute: {distance_m: sJs.distance_m, gain: sJs.elev_gain_m,
                       cost: rt.cost},
        });
    }
    return res;
}"""


@pytest.fixture(scope="module")
def cross_harbour(context, browser, served):
    """Central District to Tsim Sha Tsui, asked of Python and of the page."""
    from hk_flat_routes.config import ROUTING_PROFILES
    from hk_flat_routes.routing import route

    o, d = CROSS_HARBOUR
    python = {}
    for mode, graph in context.graphs.items():
        pts = _points(context, mode)
        if o not in pts or d not in pts:
            pytest.skip("this build does not reach both sides of the harbour")
        a, b = graph.index_of(pts[o]), graph.index_of(pts[d])
        arcs, s = route(graph, pts[o], pts[d], ROUTING_PROFILES["shortest"])
        python[mode] = {"connected": bool(graph.connected(a, b)),
                        "arcs": len(arcs), "distance_m": s["distance_m"]}

    errors: list = []
    page = _open(browser, served, errors)
    engine = page.evaluate("""([o, d]) => { const out = {};
        for (const mode of ['walk', 'bike']) {
            const g = App.graph, a = App.pointNode(o, mode), b = App.pointNode(d, mode);
            out[mode] = {a, b, compA: g.comp(a, mode), compB: g.comp(b, mode),
                         nameA: g.compName(a, mode), nameB: g.compName(b, mode),
                         route: g.route(a, b, mode, App.meta.profiles.shortest)};
        }
        return out; }""", [o, d])
    # and through the page itself, the way a person would ask
    page.select_option("#o", o)
    page.select_option("#d", d)
    page.wait_for_function("document.querySelector('#res .warn')", timeout=20_000)
    panel = page.evaluate("""() => ({
        text: document.getElementById('res').innerText,
        line: App.routeLine.getLatLngs().length, cmp: App.cmpLine.getLatLngs().length,
        marks: App.marks.getLayers().length,
        profile: document.getElementById('profnote').textContent})""")
    # the same pair on a bicycle
    page.click("#mode button[data-v=bike]")
    page.wait_for_timeout(200)
    bike_text = page.evaluate("document.getElementById('res').innerText")
    # back to a pair that can be walked: the panel recovers
    page.click("#mode button[data-v=walk]")
    page.select_option("#d", "Wan Chai")
    page.wait_for_function("document.querySelector('#res .big')", timeout=20_000)
    after = page.evaluate("""() => ({text: document.getElementById('res').innerText,
        line: App.routeLine.getLatLngs().length})""")
    page.close()
    return {"python": python, "engine": engine, "panel": panel,
            "bike_text": bike_text, "after": after, "errors": errors}


@pytest.fixture(scope="module")
def map_clicks(browser, served):
    """Set a trip with the mouse, the way the hint on the page says to.

    The planning areas cover the whole built-up city and their outlines are
    showing when the page opens, so every click here lands inside one.
    """
    errors: list = []
    page = _open(browser, served, errors)
    page.click("#clear")
    # corridors and passes have popups of their own; out of the way
    for key in ("corridors", "passes"):
        box = page.locator(f"#chk_{key}")
        if box.count() and box.is_checked():
            box.click()
    # two neighbourhood access points on one landmass, each inside a planning
    # area and clear of every outline, as points in the map's container
    spots = page.evaluate("""() => {
        const g = App.graph, mode = App.state.mode, size = App.map.getSize();
        const areas = App.overlays.neighborhoods.getLayers();
        const out = [];
        for (const name of App.DATA.neighborhood_names) {
            const n = App.pointNode(name, mode);
            if (n < 0) continue;
            const c = App.map.latLngToContainerPoint([g.nodeLat(n), g.nodeLon(n)]);
            if (c.x < 20 || c.y < 20 || c.x > size.x - 20 || c.y > size.y - 20) continue;
            const p = App.map.containerPointToLayerPoint(c);
            const inside = areas.some(a => L.Polygon.prototype._containsPoint.call(a, p));
            const onOutline = areas.some(a => L.Polyline.prototype._containsPoint.call(a, p, true));
            if (!inside || onOutline) continue;
            if (out.length && g.comp(n, mode) !== out[0].comp) continue;
            out.push({name, x: c.x, y: c.y, comp: g.comp(n, mode)});
            if (out.length === 2) break;
        }
        return out; }""")
    if len(spots) < 2:
        page.close()
        pytest.skip("no two access points inside a planning area in view")
    frame = page.locator("#map").bounding_box()
    steps = []
    for s in spots:
        page.mouse.click(frame["x"] + s["x"], frame["y"] + s["y"])
        page.wait_for_timeout(300)
        steps.append(page.evaluate("""() => ({src: App.state.src, dst: App.state.dst,
            popups: document.querySelectorAll('.leaflet-popup-content').length})"""))
    result = page.evaluate("""() => ({line: App.routeLine.getLatLngs().length,
        text: document.getElementById('res').innerText})""")
    # a click on an outline itself still says which neighbourhood it is
    page.click("#clear")
    edge = page.evaluate("""() => {
        const size = App.map.getSize();
        for (const f of App.DATA.layers.neighborhoods.features) {
            const c = f.geometry.coordinates;
            for (const ring of (f.geometry.type === 'Polygon' ? c : c.flat())) for (const q of ring) {
                const p = App.map.latLngToContainerPoint([q[1], q[0]]);
                if (p.x > 60 && p.y > 60 && p.x < size.x - 60 && p.y < size.y - 60) return [p.x, p.y];
            }
        }
        return null; }""")
    outline = None
    if edge:
        page.mouse.click(frame["x"] + edge[0], frame["y"] + edge[1])
        page.wait_for_timeout(300)
        outline = page.evaluate("""() => ({src: App.state.src,
            popup: (document.querySelector('.leaflet-popup-content') || {}).innerText || ''})""")
    page.close()
    return {"steps": steps, "result": result, "outline": outline, "errors": errors}


@pytest.fixture(scope="module")
def bike_fragment(browser, served):
    """Two junctions on one landmass that no bicycle can travel between.

    The network open to bicycles is in more pieces than the land is:
    Discovery Bay on Lantau, Ap Lei Chau off Hong Kong Island.  Skipped in
    a build whose cycling network has no such fragment.
    """
    errors: list = []
    page = _open(browser, served, errors)
    pair = page.evaluate("""() => {
        const g = App.graph, bike = g.nodeComp.bike;
        if (!bike) return null;
        let a = -1;
        for (let i = 0; i < g.n && a < 0; i++) {
            const l = App.nodeLand[i];
            if (bike[i] && l && App.landComp.bike[l] !== bike[i]) a = i;
        }
        if (a < 0) return null;
        const main = App.landComp.bike[App.nodeLand[a]];
        let b = -1;
        for (let i = 0; i < g.n && b < 0; i++) if (bike[i] === main) b = i;
        return {a, b, main}; }""")
    if not pair:
        page.close()
        pytest.skip("the cycling network has no fragment in this build")
    page.click("#mode button[data-v=bike]")
    out = page.evaluate("""({a, b, main}) => {
        App.state.src = a; App.state.dst = b;
        document.getElementById('o').value = ''; document.getElementById('d').value = '';
        App.recompute();
        const key = App.warpKey();
        App.state.src = b; App.state.dst = a; App.recompute();
        return {text: document.getElementById('res').innerText,
                warn: !!document.querySelector('#res .warn'),
                groups: [...document.querySelectorAll('#o optgroup')].map(g => g.label),
                sameLand: App.nodeLand[a] === App.nodeLand[b],
                compA: App.graph.comp(a, 'bike'), compB: App.graph.comp(b, 'bike'),
                warpA: App.warpComp(a, 'bike'), warpB: App.warpComp(b, 'bike'), main,
                sameKey: key === App.warpKey()}; }""", pair)
    # the menus as they would be headed if every piece stood on one island,
    # and if the build had a name for none of them
    out["relabelled"] = page.evaluate("""() => { const names = App.meta.components, was = names.bike, got = {};
        for (const [key, name] of [['one', 'One Island'], ['none', '']]) {
            names.bike = was.map(() => name); App.fillSelects();
            got[key] = [...document.querySelectorAll('#o optgroup')].filter(g => !g.querySelector('option:disabled')).map(g => g.label);
        }
        names.bike = was; App.fillSelects();
        return got; }""")
    page.close()
    return out, errors


@pytest.fixture(scope="module")
def steepest_shown(browser, served):
    """A walk whose steepest stored gradient is on an arc too short to be
    judged by it, and what the panel then gives as the steepest climb."""
    errors: list = []
    page = _open(browser, served, errors)
    out = page.evaluate("""() => { const g = App.graph, w = App.meta.profiles.shortest;
        const names = App.DATA.neighborhood_names;
        let tried = 0;
        for (const o of names) for (const d of names) {
            const a = App.pointNode(o, 'walk'), b = App.pointNode(d, 'walk');
            if (o === d || a < 0 || b < 0 || g.comp(a, 'walk') !== g.comp(b, 'walk')) continue;
            if (++tried > 120) return null;
            const r = g.route(a, b, 'walk', w); if (!r) continue;
            const s = g.summarise(r.arcs);
            if (Math.abs(s.steepest - s.max_grade) < 0.002) continue;
            App.state.mode = 'walk'; App.state.profile = 'shortest'; App.state.custom = false;
            App.state.src = a; App.state.dst = b; App.recompute();
            const row = [...document.querySelectorAll('#res tr')].find(tr => tr.firstChild.textContent === 'Steepest climb');
            return { o, d, pitch: s.steepest, stored: s.max_grade, shown: row.lastChild.textContent }; }
        return null; }""")
    page.close()
    if out is None:
        pytest.skip("no walk between access points has its steepest stored gradient on a short arc")
    return out, errors


@pytest.fixture(scope="module")
def warp_costs(browser, served):
    """What the warp takes each arc to cost by bicycle, beside the router's
    own figure under the same weights, and a corner of its matrix beside
    the router's searches."""
    errors: list = []
    page = _open(browser, served, errors)
    out = page.evaluate("""async () => { const g = App.graph, w = App.warpWeights(), cost = Warp.arcCosts(g, 'bike', w);
        let worst = 0, pushed = 0, open = 0, leaks = 0;
        for (let a = 0; a < g.m; a++) {
            if (!(g.arcFlags[a] & 2)) { if (cost[a] !== Infinity) leaks++; continue; }
            open++; if (g.arcPush && g.arcPush[a]) pushed++;
            worst = Math.max(worst, Math.abs(cost[a] - g.pathCost([a], 'bike', w)));
        }
        // eight access points of one landmass: the matrix is the mean of
        // the two directions, a missing one taking the other's value
        const by = new Map();
        for (const name of App.DATA.neighborhood_names) { const n = App.pointNode(name, 'bike'); if (n < 0) continue;
            const c = g.comp(n, 'bike'); if (!by.has(c)) by.set(c, []); if (!by.get(c).includes(n)) by.get(c).push(n); }
        const nodes = [...by.values()].sort((x, y) => y.length - x.length)[0].slice(0, 8), K = nodes.length;
        const D = await Warp.costMatrix(g, nodes, 'bike', w, null, null);
        const rows = nodes.map(n => g.distances(n, 'bike', w, nodes));
        let off = 0;
        for (let i = 0; i < K; i++) for (let j = 0; j < K; j++) { if (i === j) continue;
            let a = rows[i][j], b = rows[j][i]; if (!isFinite(a)) a = b; if (!isFinite(b)) b = a;
            off = Math.max(off, Math.abs(D[i * K + j] - (a + b) / 2)); }
        return { worst, pushed, open, leaks, K, off, classes: w.use_class_multiplier, mult: g.pushMult }; }""")
    page.close()
    return out, errors


_WARP_READY = ("App.warp && !App._warping"
               " && !document.getElementById('busy').classList.contains('on')")


@pytest.fixture(scope="module")
def warp_result(browser, served):
    """Build the warped city in the browser and report its diagnostics.

    The page opens on a trip on one landmass, and that landmass is the one
    the warp redraws.
    """
    errors: list = []
    page = _open(browser, served, errors)
    page.click("#warptoggle")
    page.wait_for_function(_WARP_READY, timeout=180_000)
    out = page.evaluate("""async () => {
        const w = App.warp, mode = App.state.mode, g = App.graph;
        // every street vertex: finite, and moved only on the warped landmass
        const a = App.network.alt, r = App.geom.mercator(), st = App.geom.starts;
        let finite = true, movedOff = 0, movedOn = 0, onLand = 0;
        for (let i = 0; i < a.mx.length; i++) {
            if (!Number.isFinite(a.mx[i]) || !Number.isFinite(a.my[i])) { finite = false; break; }
        }
        for (let e = 0; e < App.geom.nEdges; e++) {
            let d = 0;
            for (let k = st[e] >> 1; k < st[e + 1] >> 1; k++) d += Math.abs(a.mx[k] - r.mx[k]) + Math.abs(a.my[k] - r.my[k]);
            if (App.edgeLand[e] === w.land) { onLand++; if (d > 0) movedOn++; }
            else if (d > 0) movedOff++;
        }
        // the neighbourhoods of that landmass: how well do distances on the
        // page, and on the ground, stand for the cost of travelling between them?
        const names = Object.keys(App.DATA.points[mode]).filter(n => App.onWarpedLand(App.pointNode(n, mode)));
        const nodes = names.map(n => App.pointNode(n, mode));
        const D = await Warp.costMatrix(g, nodes, mode, App.warpWeights(), null, null);
        const f = Warp.frame(g.nodeLat(nodes[0]), g.nodeLon(nodes[0]));
        const geo = nodes.map(n => f.toXY(g.nodeLon(n), g.nodeLat(n)));
        const page = nodes.map(n => { const p = w.transform(g.nodeLon(n), g.nodeLat(n)); return f.toXY(p[0], p[1]); });
        // residual after the best single scale, as SMACOF's stress is defined
        const misfit = P => { let de = 0, ee = 0, dd = 0; const K = P.length;
            for (let i = 0; i < K; i++) for (let j = i + 1; j < K; j++) {
                const e = Math.hypot(P[i][0] - P[j][0], P[i][1] - P[j][1]), d = D[i * K + j];
                de += d * e; ee += e * e; dd += d * d; }
            const s = de / ee; let err = 0;
            for (let i = 0; i < K; i++) for (let j = i + 1; j < K; j++) {
                const e = Math.hypot(P[i][0] - P[j][0], P[i][1] - P[j][1]), d = D[i * K + j];
                err += (s * e - d) * (s * e - d); }
            return Math.sqrt(err / dd); };
        return {anchors: w.anchors, stress: w.stress, ms: w.ms, cellM: w.cellM,
                small: w.small, name: w.name, land: w.land, mode: w.mode,
                finite, movedOff, movedOn, onLand, n: a.mx.length,
                dimmed: Array.from(w.dim).reduce((x, y) => x + y, 0),
                edges: App.geom.nEdges, places: names.length,
                misfitGeography: misfit(geo), misfitPage: misfit(page),
                readout: document.getElementById('warpinfo').innerText,
                hint: document.getElementById('pickhint').textContent};
    }""")
    # switching it off puts every street back
    page.click("#warptoggle")
    page.wait_for_timeout(300)
    out["restored"] = page.evaluate("App.network.alt === null && !App.warpOn")
    page.close()
    return out, errors


# --------------------------------------------------------------------- tests
def test_the_page_loads_without_javascript_errors(browser_results):
    _out, _info, errors, _requests = browser_results
    assert not errors, f"JavaScript errors on load: {errors[:4]}"


def test_the_page_fetches_nothing_but_itself(browser_results):
    """No online base map, no fonts, no tiles: the file is the whole page."""
    _out, _info, _errors, requests = browser_results
    outside = [u for u in requests if not u.startswith("http://127.0.0.1")
               and not u.startswith("data:") and not u.startswith("blob:")]
    assert not outside, f"the page asked for {outside[:3]}"


def test_the_browser_graph_matches_the_python_graph(browser_results):
    _out, info, _errors, _requests = browser_results
    import pandas as pd
    directed = pd.read_parquet(PROCESSED_DIR / "edges_directed.parquet")
    import geopandas as gpd
    edges = gpd.read_parquet(PROCESSED_DIR / "edges_metrics.parquet")
    n_arcs = int((directed["walk_traversable"]
                  | directed["bike_traversable"]).sum())
    assert info["edges"] == len(edges)
    # the packed arc count drops arcs outside both modes' components
    assert 0.9 * n_arcs <= info["arcs"] <= n_arcs


def test_the_page_is_metric_and_credits_its_sources(browser_results):
    """Kilometres and metres only, and the credits the data and upstream
    are owed."""
    import re

    _out, info, _errors, _requests = browser_results
    text = info["text"]
    imperial = re.findall(r"\d[\d,.]*\s?(?:ft|mi|feet|miles)\b", text)
    assert not imperial, f"imperial units on the page: {imperial[:5]}"
    assert re.search(r"\d\.\d\d km", text) and re.search(r"\d m climb", text)
    assert "San Francisco" not in text
    for credit in ("OpenStreetMap", "Overture", "CEDD", "Town Planning Board",
                   "Flatten SF", "Drew Edwards"):
        assert credit in text, f"{credit} is not credited in the panel"
        assert credit in info["attribution"], f"{credit} is not credited on the map"


def test_the_street_canvas_thins_the_network_when_zoomed_out(browser_results):
    """The whole network in view is drawn as a fraction of its edges."""
    _out, info, _errors, _requests = browser_results
    assert 0 < info["drawn"] < 0.8 * info["edges"], info


def test_every_python_route_resolves_in_the_browser(browser_results):
    out, _info, _errors, _requests = browser_results
    missing = [r for r in out if r.get("miss")]
    assert not missing, f"{len(missing)} routes did not resolve: {missing[:3]}"
    assert all(r["oneLandmass"] for r in out)


def test_metrics_agree_on_pythons_own_path(browser_results, reference):
    """Same arcs, same numbers -- to within the packing's quantisation."""
    out, _info, _errors, _requests = browser_results
    ref = {(r["mode"], r["profile"], r["o"], r["d"]): r for r in reference}
    bad = []
    for r in out:
        py = ref[(r["mode"], r["profile"], r["o"], r["d"])]
        js = r["onPyPath"]
        # distance error accumulates one quantum per arc, so scale with length
        tol = max(0.5, 1e-4 * py["distance_m"])
        if abs(js["distance_m"] - py["distance_m"]) > tol:
            bad.append((r, f"distance {js['distance_m']:.2f} vs "
                           f"{py['distance_m']:.2f}"))
        # so does climbing: half a centimetre an arc at worst, and a route
        # along Hong Kong Island is several hundred short arcs
        climb_tol = max(0.3, 0.005 * len(py["path"]))
        for k in ("gain", "loss"):
            if abs(js[k] - py[k]) > climb_tol:
                bad.append((r, f"{k} {js[k]:.3f} vs {py[k]:.3f}"))
        for k in ("maxabs", "avgg"):
            if abs(js[k] - py[k]) > 5e-4:
                bad.append((r, f"{k} {js[k]:.5f} vs {py[k]:.5f}"))
        # The steepest pitch is a stored gradient on an arc of 15 m or
        # more, as exact as the others.  On a shorter arc it is net climb
        # over length, from a length the bundle holds to 5 cm and heights
        # it holds to 1 cm: under the 60% ceiling that is 2.5 cm in all,
        # over the arc's length (2.5 points on a 1 m stub, 0.2 on 12 m).
        # The browser reports no pitch below zero.
        pitch_tol = 5e-4
        if py["shortest_arc"] < MIN_RELIABLE_GRADE_LENGTH_M:
            pitch_tol += 0.025 / py["shortest_arc"]
        if abs(js["pitch"] - max(py["pitch"], 0.0)) > pitch_tol:
            bad.append((r, f"pitch {js['pitch']:.5f} vs {py['pitch']:.5f}"))
        for i, (a, b) in enumerate(zip(js["th"], py["th"])):
            if abs(a - b) > max(1.0, 1e-3 * py["distance_m"]):
                bad.append((r, f"d_above[{i}] {a:.2f} vs {b:.2f}"))
    assert not bad, f"{len(bad)} metric mismatches, e.g. {bad[:4]}"


def test_the_panel_gives_the_steepest_pitch_as_the_pair_tables_do(steepest_shown):
    """Python's ``max_grade`` for a route is its steepest pitch
    (``routing.arc_pitch``): an arc under 15 m is judged on its average
    climb.  The panel showed the largest gradient stored for any arc, and a
    map that disagrees with its own CSV is worse than either figure."""
    out, errors = steepest_shown
    assert not errors, errors[:3]
    assert out["shown"] == f"{100 * out['pitch']:.1f}%", out
    assert out["shown"] != f"{100 * out['stored']:.1f}%", out


def test_the_browser_router_is_optimal(browser_results):
    """The browser's own route must never cost more than Python's."""
    out, _info, _errors, _requests = browser_results
    worse = [(r["mode"], r["profile"], r["o"], r["d"],
              r["ownRoute"]["cost"], r["onPyPath"]["cost"])
             for r in out
             if r["ownRoute"]["cost"] > r["onPyPath"]["cost"] * 1.0005 + 0.5]
    assert not worse, f"browser found costlier routes: {worse[:4]}"


def test_most_routes_are_identical_not_merely_equivalent(browser_results):
    """A sanity floor: tie-breaking should differ sometimes, not usually."""
    out, _info, _errors, _requests = browser_results
    same = sum(1 for r in out if r.get("same"))
    assert same >= 0.6 * len(out), (
        f"only {same}/{len(out)} arc sequences matched exactly, which "
        "suggests a cost-model difference rather than tie-breaking")


def test_both_sides_of_the_harbour_are_sampled(browser_results):
    """The sample is only worth its name if it covers more than one
    landmass (in a build that has more than one)."""
    out, _info, _errors, _requests = browser_results
    ends = {r["o"] for r in out}
    if {"Pok Fu Lam", "Tsim Sha Tsui"} <= ends:
        assert len({(r["o"], r["d"]) for r in out}) >= 6


# -------------------------------------------------------- across the harbour
def test_python_finds_no_route_across_the_harbour(cross_harbour):
    for mode, r in cross_harbour["python"].items():
        assert not r["connected"], mode
        assert r["arcs"] == 0 and r["distance_m"] == 0.0, mode


def test_the_browser_finds_no_route_across_the_harbour(cross_harbour):
    for mode, r in cross_harbour["engine"].items():
        assert r["a"] >= 0 and r["b"] >= 0, mode
        assert r["compA"] > 0 and r["compB"] > 0, mode
        assert r["compA"] != r["compB"], mode
        assert r["route"] is None, mode
        assert r["nameA"] == "Hong Kong Island", mode
        assert r["nameB"] == "Kowloon and the New Territories", mode


def test_the_page_says_why_there_is_no_route(cross_harbour):
    """Not an empty panel: the two landmasses by name, both ends marked,
    and no line left over from the last trip."""
    assert not cross_harbour["errors"], cross_harbour["errors"][:3]
    p = cross_harbour["panel"]
    assert "No walking route" in p["text"]
    assert "Hong Kong Island" in p["text"]
    assert "Kowloon and the New Territories" in p["text"]
    assert "Central District" in p["text"] and "Tsim Sha Tsui" in p["text"]
    assert "中區" in p["text"] and "尖沙咀" in p["text"]
    assert p["line"] == 0 and p["cmp"] == 0 and p["profile"] == ""
    assert p["marks"] == 2
    assert "No cycling route" in cross_harbour["bike_text"]
    # and a pair that can be walked brings the result back
    assert " km" in cross_harbour["after"]["text"]
    assert cross_harbour["after"]["line"] > 1


# ------------------------------------------------------------ the mouse
def test_a_click_inside_a_neighbourhood_sets_the_trip(map_clicks):
    """Leaflet's canvas takes a click anywhere inside a polygon as a click
    on it, and a layer with a popup keeps that click from the map.  With
    the outlines drawn as upstream draws them, no click on the city would
    ever set an origin."""
    assert not map_clicks["errors"], map_clicks["errors"][:3]
    first, second = map_clicks["steps"]
    assert first["src"] is not None and first["src"] >= 0 and first["dst"] is None
    assert second["src"] == first["src"] and second["dst"] is not None
    assert first["popups"] == 0 and second["popups"] == 0
    assert map_clicks["result"]["line"] > 1
    assert " km" in map_clicks["result"]["text"]


def test_a_click_on_an_outline_still_names_the_neighbourhood(map_clicks):
    outline = map_clicks["outline"]
    if outline is None:
        pytest.skip("no outline vertex in view")
    assert "Neighbourhood:" in outline["popup"]
    assert outline["src"] is None


# ------------------------------------------- pieces of the cycling network
def test_no_route_within_a_landmass_is_not_blamed_on_the_water(bike_fragment):
    """Both ends on Lantau, no way between them by bicycle: the page must
    not answer "choose two points on the same landmass"."""
    out, errors = bike_fragment
    assert not errors, errors[:3]
    assert out["sameLand"] and out["compA"] != out["compB"]
    assert out["warn"] and "No cycling route" in out["text"]
    assert "separate pieces" in out["text"]
    assert "same landmass" not in out["text"]


def test_the_menu_groups_of_one_island_can_be_told_apart(bike_fragment):
    """A piece of the cycling network is called after the island it stands
    on, and "a route exists only between two places in the same group" is
    no help under two groups both headed Lantau."""
    out, _ = bike_fragment
    assert len(out["groups"]) >= 2
    assert len(set(out["groups"])) == len(out["groups"]), out["groups"]
    # however many pieces share a name, or have none
    for labels in out["relabelled"].values():
        assert len(labels) >= 2 and len(set(labels)) == len(labels), labels
    assert out["relabelled"]["one"][:2] == ["One Island", "One Island: a separate part of the network"]
    assert out["relabelled"]["none"][0] == "A separate part of the network"


def test_a_fragment_is_warped_with_its_landmass(bike_fragment):
    """A warp fitted to a fragment's few junctions would carry the whole
    landmass along on an extrapolation; the landmass's main cycling network
    is laid out instead, whichever piece the origin is on."""
    out, _ = bike_fragment
    assert out["warpA"] == out["main"] == out["warpB"]
    assert out["warpA"] != out["compA"]
    assert out["sameKey"]


# ------------------------------------------------------------------ the warp
def test_the_warp_prices_every_arc_as_the_router_does(warp_costs):
    """The warp's matrix comes from a table of arc costs of its own
    (``Warp.arcCosts``), not from ``Graph.route``.  A pushed link left at
    its plain length there would draw every cycle track nearer the road
    than a ride between them is."""
    out, errors = warp_costs
    assert not errors, errors[:3]
    assert out["classes"], "the warp is expected to open on the class multipliers"
    assert out["open"] > 0 and out["leaks"] == 0, out
    assert out["worst"] < 1e-9, out
    assert out["K"] >= 4 and out["off"] < 1e-6, out


def test_the_warp_builds_without_errors(warp_result):
    out, errors = warp_result
    assert not errors, errors[:3]
    assert not out["small"]
    assert out["finite"] and out["n"] > 0
    assert out["restored"]


def test_the_warp_redraws_one_landmass_and_leaves_the_rest(warp_result):
    """Hong Kong Island moves; Kowloon stays where it is, greyed."""
    out, _ = warp_result
    assert out["name"], "the warped landmass has no name to show"
    assert out["name"] in out["readout"]
    assert out["movedOff"] == 0
    assert out["movedOn"] > 0.95 * out["onLand"] > 0
    assert out["dimmed"] == out["edges"] - out["onLand"]
    assert "off while the city is warped" in out["hint"]


def test_the_warp_keeps_to_upstreams_number_of_anchors(warp_result):
    """A one-kilometre lattice over the territory would be over a thousand
    anchors and a cost matrix of minutes; the spacing follows the landmass
    instead, and the count stays where San Francisco's is."""
    out, _ = warp_result
    assert 120 <= out["anchors"] <= 260, out["anchors"]
    assert out["cellM"] >= 150
    assert out["ms"] < 60_000


def test_the_warp_fits_the_cost_matrix_reasonably(warp_result):
    """Stress is the residual between page distance and climbing cost."""
    out, _ = warp_result
    assert out["stress"] < 0.25, f"stress {out['stress']:.3f} is too high to read"


def test_the_warped_page_tells_cost_better_than_geography_does(warp_result):
    """The whole point: between the neighbourhoods of the landmass, distance
    on the warped page stands for the cost of the trip more closely than
    distance on the ground does."""
    out, _ = warp_result
    assert out["places"] >= 6
    assert out["misfitPage"] < 0.8 * out["misfitGeography"], (
        f"page {out['misfitPage']:.3f} vs geography {out['misfitGeography']:.3f}")
