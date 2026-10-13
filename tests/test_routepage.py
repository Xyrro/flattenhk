"""End-to-end test of the route page (the shareable map).

Drives ``outputs/hk_flat_route_finder.html`` in a headless browser: the page
must load without errors, route its default trip, answer searches in English
and Chinese for a junction, an address, a station and a park, refuse a trip
across the harbour in words, and produce a route family whose ends
are what the slider labels promise -- the left end is the true shortest
path and the right end has the least climbing. One member is also checked
against Python under the same scaled weights, so the slider positions
cannot drift from the analysis.

The second half is the page around the router, which is where a review in
a real browser found it wanting: what is on the card after a refusal and
after the first paint, where the pins are put and what they are called,
the order of a bike family on calm streets, the phone layout, and the
small things (file names, links, keys).  ``test_routepage_source.py`` has
the checks that need no build.

Skipped unless Playwright, a Chromium build and a built page are present.
"""
from __future__ import annotations

import glob
import os

import pytest

from hk_flat_routes.config import BIKE_PUSH_MULTIPLIER, ELEVATION, PROCESSED_DIR
from hk_flat_routes.viz_interactive import SIMPLE_HTML, SITE_INDEX

playwright = pytest.importorskip("playwright.sync_api",
                                 reason="playwright is not installed")


def _chromium() -> str | None:
    """A Chromium to hand Playwright, or None to let it use its own."""
    for pattern in ("/opt/pw-browsers/chromium-*/chrome-linux/chrome",
                    os.path.expanduser(
                        "~/.cache/ms-playwright/chromium-*/chrome-linux/chrome")):
        hits = sorted(glob.glob(pattern))
        if hits:
            return hits[-1]
    return None


pytestmark = [
    pytest.mark.skipif(not SIMPLE_HTML.exists(),
                       reason="route page not built; run `python -m hk_flat_routes map`"),
    pytest.mark.skipif(not (PROCESSED_DIR / "edges_directed.parquet").exists(),
                       reason="processed data not built"),
]

_SEARCHES = ["nathan & jordan", "nathan rd and jordan rd", "彌敦道與佐敦道", "123 nathan road",
             "彌敦道123號", "victoria park", "維多利亞公園", "mong kok", "旺角", "central station",
             "中環站", "lion rock", "mei foo sun chuen", "hennessy rd", "k11", "sky100",
             "123號彌敦道", "nathan road 123", "central pier"]

_SCRIPT = """(queries) => {
    const fam = App.family, g = App.graph;
    const alphas = [0, 14, 120];
    const members = fam.unique.map(u => ({
        id: u.id, distance_m: u.stats.distance_m, gain: u.stats.elev_gain_m,
        arcs: u.arcs.map(a => [g.arcEdge[a], (g.arcFlags[a] & 4) ? 1 : 0]),
        costs: alphas.map(a => g.pathCost(u.arcs, App.state.mode, App.weights(a))),
    }));
    const search = {};
    for (const q of queries) search[q] = App.index.search(q).map(r => [r.name, r.kind]);
    const sl = document.getElementById('sl');
    sl.value = 0; sl.dispatchEvent(new Event('input'));
    const atZero = App.shown.id;
    sl.value = 1; sl.dispatchEvent(new Event('input'));
    const atOne = App.shown.id;
    sl.value = 0.5; sl.dispatchEvent(new Event('input'));
    const half = App.shown.id;
    return {
        from: App.state.from, to: App.state.to, members, alphas,
        search, atZero, atOne, half, hash: location.hash,
        places: App.index.places.length, intersections: App.index.intersections.length,
        hasAddresses: !!App.index.addr,
        frontier: { solutions: App._search.solutions.length, labels: App._search.labels,
                    expanded: App._search.expanded, truncated: App._search.truncated },
        snap: (() => { const g = App.graph, kx = 111320 * Math.cos(22.3 * Math.PI / 180), ky = 110540;
            // 15 km out to sea, and every node's own spot nudged 30 m east
            const far = App.pointAt(114.52, 22.30), near = [];
            for (let i = 0; i < g.n; i += 997) {
                if (!(g.nodeFlags[i] & 1)) continue;
                const p = App.pointAt(g.nodeLon(i) + 30 / kx, g.nodeLat(i));
                near.push({ node: p.node, onNode: p.node >= 0 && p.lon === g.nodeLon(p.node) && p.lat === g.nodeLat(p.node),
                            shift: p.shift, real: Math.hypot((p.lon - p.qlon) * kx, (p.lat - p.qlat) * ky) });
            }
            return { far: { node: far.node, lon: far.lon, lat: far.lat, label: far.label }, near }; })(),
    };
}"""


@pytest.fixture(scope="module")
def page_results():
    from playwright.sync_api import sync_playwright
    errors: list = []
    with sync_playwright() as pw:
        try:
            browser = pw.chromium.launch(executable_path=_chromium(),
                                         args=["--no-sandbox", "--disable-gpu"])
        except Exception as exc:                            # pragma: no cover
            pytest.skip(f"no usable Chromium: {exc}")
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: errors.append(m.text)
                if m.type == "error" and "fonts.g" not in m.text
                and "ERR_" not in m.text else None)
        page.goto(SIMPLE_HTML.resolve().as_uri(), wait_until="load", timeout=240_000)
        page.wait_for_function(
            "window.App && App.family && !App.family.partial"
            " && !document.getElementById('result').hidden",
            timeout=240_000)
        out = page.evaluate(_SCRIPT, _SEARCHES)
        # change the destination without touching the slider: the line on
        # the map must be the new trip's, not the old one's
        page.evaluate("""() => {
            const hit = App.index.search('victoria park')[0];
            window._hit = hit;
            App.setPoint('to', App.pointAt(hit.lon, hit.lat, hit.name), false);
            App.recompute('auto');
        }""")
        page.wait_for_function("App.family && !App.family.partial", timeout=120_000)
        out["retarget"] = page.evaluate("""() => {
            const hit = window._hit, before = 0;
            const u = App.shown, line = App._line.getLatLngs();
            const end = line[line.length - 1];
            return { before, after: line.length, sameAsShown: line.length === u.latlngs.length,
                     member: App.family.unique.includes(u),
                     endsAtPark: Math.abs(end.lat - hit.lat) < 0.004 && Math.abs(end.lng - hit.lon) < 0.004 };
        }""")
        # a trip across the harbour: both ends snap, and the page says why
        # there is nothing to draw
        page.evaluate("""() => {
            App.setPoint('from', App.pointAt(114.1578, 22.2816, 'Central'), false);
            App.setPoint('to', App.pointAt(114.1722, 22.2975, 'Tsim Sha Tsui'), false);
            App.recompute('auto');
        }""")
        page.wait_for_timeout(600)
        out["harbour"] = page.evaluate("""() => {
            const seen = (id) => { const e = document.getElementById(id), r = e.getBoundingClientRect();
                return { display: getComputedStyle(e).display, height: r.height, text: e.textContent.trim() }; };
            return {
            family: !!App.family, status: document.getElementById('status').textContent,
            comps: [App.graph.comp(App.state.from.node, 'walk'), App.graph.comp(App.state.to.node, 'walk')],
            names: [App.graph.compName(App.state.from.node, 'walk'), App.graph.compName(App.state.to.node, 'walk')],
            result: seen('result'), prof: seen('prof'), turns: seen('turns'), delta: seen('delta'),
            share: seen('share'), gpx: seen('gpx'),
            hash: location.hash, token: App.token(), lines: App.routeLayer.getLayers().length }; }""")
        # bike mode, Sha Tin Station to Tai Po Market Station: with calm
        # streets on the ride takes the cycle track along Tolo Harbour; off,
        # Tai Po Road, the shortest line and a main road.  (The trip this
        # used to ride, Tai Po Market to the Science Park, tells the two
        # apart only while the cycle track is hard to reach: with short
        # footway links opened to pushed bicycles, tried on the table of
        # the build before them, the shortest line there took 4.9 km of
        # the track as well.)
        page.evaluate("""() => {
            document.querySelector('#mode button[data-v=bike]').click();
            App.setPoint('from', App.pointAt(114.18691, 22.38213, 'Sha Tin Station'), false);
            App.setPoint('to', App.pointAt(114.17039, 22.44458, 'Tai Po Market Station'), false);
            App.recompute('auto');
        }""")
        page.wait_for_function("App.family && !App.family.partial", timeout=120_000)
        streets = """() => {
            const g = App.graph, u = App.family.shortest, km = {}, cls = {};
            for (const a of u.arcs) {
                const e = g.arcEdge[a];
                const n = App.geom.edgeInfo(e).name || '?';
                const c = App.geom.classes[App.geom.cls[e]];
                km[n] = (km[n] || 0) + g.arcLen[a] / g.DM;
                cls[c] = (cls[c] || 0) + g.arcLen[a] / g.DM;
            }
            return { streets: km, classes: cls, distance_m: u.stats.distance_m, stress_m: u.stats.stress_m,
                     calm: App.state.calm, token: App.token(), n: App.family.unique.length,
                     rowHidden: document.getElementById('calmrow').hidden,
                     monotone: App.family.unique.every((m, i, arr) => i === 0
                        || (m.stats[App.lenKey()] >= arr[i - 1].stats[App.lenKey()] - 1e-6
                            && m.stats.elev_gain_m <= arr[i - 1].stats.elev_gain_m + 1e-6)) };
        }"""
        out["calm_on"] = page.evaluate(streets)
        page.evaluate("() => document.getElementById('calm').click()")
        page.wait_for_function("App.family && !App.family.partial && !App.state.calm", timeout=120_000)
        out["calm_off"] = page.evaluate(streets)
        # loop mode: the engine's loops, then the toggle, then a shared link
        out["loops"] = page.evaluate("""() => {
            const g = App.graph, src = App.nearestNode(114.1880, 22.2820);     // Victoria Park
            const s = g.loops(src, 'walk', { targetM: 6000 });
            while (!s.step(1e9)) {}
            const tail = g.reverse().tail;
            return { src, ms: s.ms, accepted: s.accepted.length, median: s.medianGain,
              loops: s.loops.map(r => ({ length: r.length, gain: r.gain, overlap: r.overlap, round: r.round,
                first: tail[r.arcs[0]], last: g.head[r.arcs[r.arcs.length - 1]],
                connected: r.arcs.every((a, i) => i === 0 || tail[a] === g.head[r.arcs[i - 1]]) })) };
        }""")
        out["outback"] = page.evaluate("""() => {
            const g = App.graph, src = App.nearestNode(114.1968, 22.2365);    // Repulse Bay beach
            const run = (opts) => { const s = g.loops(src, 'walk', Object.assign({ targetM: 4000 }, opts));
              while (!s.step(1e9)) {} return s.loops.map(r => ({ gain: r.gain, kind: r.kind, overlap: r.overlap, length: r.length })); };
            return { loops: run({}), ob: run({ outBack: true }) };
        }""")
        page.evaluate("""() => {
            document.querySelector('#mode button[data-v=walk]').click();
            App.setPoint('from', App.pointAt(114.1880, 22.2820, 'Victoria Park'), false);
            App.setPoint('to', App.pointAt(114.1730, 22.2775, 'Wan Chai'), false);
            App.recompute('auto');
        }""")
        page.wait_for_function("App.family && !App.family.partial", timeout=120_000)
        page.click("#loopbtn")
        page.wait_for_function("App.family && App.family.loop", timeout=60_000)
        page.wait_for_timeout(400)
        out["loop_ui"] = page.evaluate("""() => ({
            looping: document.getElementById('card').classList.contains('looping'),
            toWidth: document.getElementById('tofield').getBoundingClientRect().width,
            pressed: document.getElementById('loopbtn').getAttribute('aria-pressed'),
            slMax: +document.getElementById('sl').max, hash: location.hash,
            stats: App.shown.stats.distance_m, target: App.family.targetM,
            delta: document.getElementById('delta').textContent,
            markers: App.markers.getLayers().length })""")
        page.click("#loopbtn")
        page.wait_for_function("App.family && !App.family.loop && !App.family.partial", timeout=120_000)
        page.wait_for_timeout(400)
        out["loop_off"] = page.evaluate("""() => ({ to: document.getElementById('to').value,
            toWidth: document.getElementById('tofield').getBoundingClientRect().width,
            hash: location.hash, slMax: +document.getElementById('sl').max })""")
        # a shared loop link reopens the same loop
        page.goto(SIMPLE_HTML.resolve().as_uri() + "#l~114.18800~22.28200~w~3.418~1~Victoria_20Park",
                  wait_until="load", timeout=240_000)
        page.reload(wait_until="load", timeout=240_000)
        page.wait_for_function("window.App && App.family && App.family.loop", timeout=240_000)
        # the GPX track and the units toggle
        out["gpx"] = page.evaluate("""() => { const g = App.gpx(); const pts = g.match(/<trkpt /g) || [];
            const eles = [...g.matchAll(/<ele>([-\\d.]+)<\\/ele>/g)].map(m => +m[1]);
            return { points: pts.length, latlngs: App.shown.latlngs.length, name: (g.match(/<name>(.*?)<\\/name>/) || [])[1],
              eleOk: eles.every(Number.isFinite), eleMin: Math.min(...eles), eleMax: Math.max(...eles),
              header: g.startsWith('<?xml version="1.0"'), closed: g.trim().endsWith('</gpx>') }; }""")
        out["units_default"] = page.evaluate("""() => ({ dist: document.getElementById('v_dist').textContent,
            pressed: document.querySelector('#units button[aria-pressed=true]').dataset.u })""")
        page.click("#units button[data-u=mi]"); page.wait_for_timeout(300)
        out["units_mi"] = page.evaluate("""() => ({ dist: document.getElementById('v_dist').textContent, climb: document.getElementById('v_climb').textContent,
            slMax: document.getElementById('sl').max, stored: localStorage.getItem('flattenhk.units') })""")
        # the same link opened by someone whose device is set to miles
        page.reload(wait_until="load", timeout=240_000)
        page.wait_for_function("window.App && App.family && App.family.loop", timeout=240_000)
        out["loop_link_mi"] = page.evaluate("""() => ({ mi: App.state.loopMi, target: App.family.targetM,
            slpos: document.getElementById('slpos').textContent, token: App.token(),
            pressed: document.querySelector('#units button[aria-pressed=true]').dataset.u })""")
        page.click("#units button[data-u=km]"); page.wait_for_timeout(300)
        out["units_km"] = page.evaluate("""() => ({ dist: document.getElementById('v_dist').textContent, climb: document.getElementById('v_climb').textContent,
            delta: document.getElementById('delta').textContent, slMax: document.getElementById('sl').max, slpos: document.getElementById('slpos').textContent,
            end1: document.getElementById('end1').textContent, mi: App.state.loopMi, stored: localStorage.getItem('flattenhk.units') })""")
        out["loop_link"] = page.evaluate("""() => ({ loop: App.state.loop, mi: App.state.loopMi,
            idx: App.state.loopIdx, from: document.getElementById('from').value,
            n: App.family.unique.length, slVal: +document.getElementById('sl').value })""")
        browser.close()
    return out, errors


def test_loops_close_on_themselves_and_are_flat(page_results):
    out, _ = page_results
    r = out["loops"]
    assert r["accepted"] >= 10 and len(r["loops"]) >= 2, r
    target = 6000
    for lp in r["loops"]:
        assert lp["first"] == r["src"] and lp["last"] == r["src"] and lp["connected"], lp
        assert abs(lp["length"] - target) <= 0.12 * target, lp
        assert lp["overlap"] <= 0.3 and lp["round"] >= 0.2, lp
    # the flattest loop climbs well under what a typical loop from here does
    assert r["loops"][0]["gain"] < 0.75 * r["median"], r
    assert r["ms"] < 8000, r


def test_out_and_backs_are_offered_only_when_allowed_and_are_no_hillier(page_results):
    out, _ = page_results
    r = out["outback"]
    assert all(lp["kind"] != "outback" for lp in r["loops"])
    # from Repulse Bay every loop has to climb off the beach; along the
    # shore and back is the flattest run there is
    # (overlap counts the second pass over a street, so there and back is 0.5)
    obs = [lp for lp in r["ob"] if lp["kind"] == "outback"]
    assert obs and obs[0]["overlap"] > 0.4, r
    assert r["ob"][0]["gain"] <= r["loops"][0]["gain"] + 1e-6, (r["ob"][0], r["loops"][0])
    assert abs(r["ob"][0]["length"] - 4000) <= 0.25 * 1609.344


def test_the_gpx_track_carries_every_point_with_an_elevation(page_results):
    out, _ = page_results
    g = out["gpx"]
    assert g["header"] and g["closed"], g
    assert g["points"] == g["latlngs"] > 100, g
    assert g["eleOk"] and -5 <= g["eleMin"] < g["eleMax"] < 960, g
    assert "loop from" in g["name"], g


def test_units_are_kilometres_by_default_and_switch_to_miles_and_back(page_results):
    out, _ = page_results
    default, mi, km = out["units_default"], out["units_mi"], out["units_km"]
    assert default["pressed"] == "km" and default["dist"].endswith("km"), default
    assert mi["dist"].endswith("mi") and mi["climb"].endswith("ft"), mi
    assert mi["slMax"] == "15" and mi["stored"] == "mi", mi
    assert km["dist"].endswith("km") and km["climb"].endswith("m") and " m of climbing" in km["delta"], km
    # the loop slider runs 2 to 24 km in km mode; the target itself is unchanged
    assert km["slMax"] == "24" and km["end1"] == "24 km" and km["slpos"].endswith("km loop"), km
    assert km["stored"] == "km"


def test_the_loop_button_folds_the_destination_away_and_back(page_results):
    out, errors = page_results
    on, off, link = out["loop_ui"], out["loop_off"], out["loop_link"]
    assert not errors, errors[:4]
    assert on["looping"] and on["pressed"] == "true" and on["toWidth"] < 2
    assert on["slMax"] == 24 and on["hash"].startswith("#l~")       # kilometres
    assert on["markers"] == 1
    assert abs(on["stats"] - on["target"]) <= 0.15 * on["target"]
    assert "loop" in on["delta"]
    assert off["to"] == "Wan Chai" and off["toWidth"] > 100
    assert off["slMax"] == 1 and off["hash"].startswith("#t~")
    # a shared 5.5 km loop (3.418 mi in the link) comes back as 5.5 km
    assert link["loop"] and abs(link["mi"] * 1.609344 - 5.5) < 0.01 and link["from"] == "Victoria Park"
    assert link["slVal"] == 5.5 and link["idx"] == min(1, link["n"] - 1)


def test_a_loop_link_keeps_its_length_whatever_units_the_reader_uses(page_results):
    """The link gives the length in miles to three places.  Read on a
    device set to miles it used to be rounded to the half-mile slider, so a
    5.5 km loop became a 3.5 mi one, 130 m longer and a different loop."""
    out, _ = page_results
    r = out["loop_link_mi"]
    assert r["pressed"] == "mi"
    assert r["mi"] == pytest.approx(3.418) and r["target"] == pytest.approx(3.418 * 1609.344)
    assert r["slpos"] == "3.4 mi loop", r
    assert "~3.418~" in r["token"], r


def test_calm_streets_take_a_bike_along_tolo_harbour_on_the_cycle_track(page_results):
    out, _ = page_results
    on, off = out["calm_on"], out["calm_off"]
    assert on["calm"] and not off["calm"]
    assert not on["rowHidden"]
    assert on["token"].split("~")[5] == "b" and off["token"].split("~")[5] == "bx"
    assert on["classes"].get("cycleway", 0) > 3000
    assert off["classes"].get("cycleway", 0) < 1000
    assert sum(v for k, v in off["streets"].items() if k.startswith("Tai Po Road")) > 2000
    assert sum(v for k, v in on["streets"].items() if k.startswith("Tai Po Road")) < 500
    # calm costs some real distance (a sixth to a quarter more here) and
    # buys a lot of comfort
    assert off["distance_m"] <= on["distance_m"] < off["distance_m"] * 1.35
    assert on["stress_m"] < off["stress_m"]
    # the family stays a frontier in the units it was searched in
    assert on["monotone"] and off["monotone"]
    assert on["n"] >= 1 and off["n"] >= 1


def test_a_trip_across_the_harbour_is_refused_in_words(page_results):
    """Nothing crosses Victoria Harbour on foot.  Both ends are real
    corners on their own shores, and the page names the two landmasses
    instead of drawing nothing."""
    out, _ = page_results
    h = out["harbour"]
    assert not h["family"], h
    assert h["comps"][0] != h["comps"][1] and all(c > 0 for c in h["comps"]), h
    assert h["names"] == ["Hong Kong Island", "Kowloon and the New Territories"], h
    assert "No way on foot between Hong Kong Island and Kowloon" in h["status"], h


def test_nothing_of_the_last_trip_is_left_on_the_card_after_a_refusal(page_results):
    """The ``hidden`` attribute was set all along; the stylesheet's own
    ``display`` rules overrode it, and the figures and profile of the trip
    before stayed under the refusal.  So this looks at what is drawn."""
    out, _ = page_results
    h = out["harbour"]
    for block in ("result", "prof", "turns", "share", "gpx"):
        assert h[block]["display"] == "none" and h[block]["height"] == 0, (block, h[block])
    assert h["delta"]["text"] == "" and h["lines"] == 0, h
    # and the address bar holds the trip that was refused, not the one before
    assert h["hash"] == "#" + h["token"] and "Tsim_20Sha_20Tsui" in h["hash"], h


def test_the_page_loads_and_routes_its_default_trip(page_results):
    out, errors = page_results
    assert not errors, errors[:4]
    assert out["from"] and out["to"]
    assert len(out["members"]) >= 2, "the default trip should offer a real choice"
    f = out["frontier"]
    assert f["solutions"] >= 2 and not f["truncated"], f
    assert f["labels"] < 4_000_000, f


def test_search_finds_junctions_addresses_and_places_in_both_languages(page_results):
    out, _ = page_results
    s = out["search"]
    assert out["intersections"] > 5000 and out["places"] > 20000 and out["hasAddresses"]
    assert s["nathan & jordan"][0] == ["Jordan Road & Nathan Road", "junction"]
    # abbreviations and full words match the same corner, and so does Chinese
    assert s["nathan rd and jordan rd"][0] == ["Jordan Road & Nathan Road", "junction"]
    assert s["彌敦道與佐敦道"][0] == ["佐敦道 & 彌敦道", "junction"]
    assert s["123 nathan road"][0] == ["123 Nathan Road", "address"]
    assert s["彌敦道123號"][0] == ["彌敦道123號", "address"]
    # the number may lead in Chinese and trail in English
    assert s["123號彌敦道"][0] == ["彌敦道123號", "address"]
    assert s["nathan road 123"][0] == ["123 Nathan Road", "address"]
    # a name that ends in digits is a place, not a house number on "K" street
    assert s["k11"][0][0] == "K11" and s["sky100"][0][0] == "Sky100"
    assert not any("address" in k for q in ("k11", "sky100") for _, k in s[q])
    # one pier under several spellings of its number is listed once
    piers = [n for n, k in s["central pier"] if k == "pier"]
    assert len(piers) == len({n.replace("No. ", "").replace("Number ", "") for n in piers})
    assert s["victoria park"][0] == ["Victoria Park", "park"]
    assert s["維多利亞公園"][0] == ["維多利亞公園", "park"]
    # an area outranks the station and the shops named after it
    assert s["mong kok"][0] == ["Mong Kok", "area"]
    assert s["旺角"][0] == ["旺角", "area"]
    assert s["central station"][0] == ["Central Station", "station"]
    assert s["中環站"][0] == ["中環站", "station"]
    assert s["lion rock"][0] == ["Lion Rock", "peak"]
    assert s["mei foo sun chuen"][0] == ["Mei Foo Sun Chuen", "estate"]
    assert any(k == "junction" and "Hennessy Road" in n for n, k in s["hennessy rd"])


def test_the_slider_ends_are_the_shortest_and_the_flattest(page_results):
    out, _ = page_results
    m = out["members"]
    first, last = m[0], m[-1]
    assert first["distance_m"] <= min(u["distance_m"] for u in m) + 1e-6
    assert last["gain"] <= min(u["gain"] for u in m) + 1e-6
    assert out["atZero"] == first["id"] and out["atOne"] == last["id"]
    assert 0 < out["half"] < len(m) - 1
    # the family is deduplicated: no two members share an arc sequence
    seqs = [tuple(map(tuple, u["arcs"])) for u in m]
    assert len(set(seqs)) == len(seqs)
    # and no member is dominated by another on both counts
    for a in m:
        for b in m:
            if a is b:
                continue
            assert not (b["distance_m"] <= a["distance_m"] - 1e-6
                        and b["gain"] <= a["gain"] - 1e-6), (a, b)


def test_the_family_is_monotone_in_distance_and_climbing(page_results):
    """Sliding right never shortens the route and never adds climbing: the
    defining property of a distance/climbing frontier sorted by distance,
    and the behaviour the slider's end labels promise."""
    out, _ = page_results
    seq = out["members"]
    for a, b in zip(seq, seq[1:]):
        assert b["distance_m"] >= a["distance_m"] - 1e-6
        assert b["gain"] <= a["gain"] + 1e-6


def test_a_new_trip_replaces_the_drawn_route_at_once(page_results):
    """Changing an endpoint redraws without the slider being touched."""
    out, _ = page_results
    r = out["retarget"]
    assert r["member"] and r["sameAsShown"] and r["endsAtPark"], r


def test_a_point_snaps_to_a_corner_nearby_and_to_none_far_away(page_results):
    """The pin and the route start must agree: a point a few steps off a
    street sits on the corner the route leaves from.  A point out at sea
    is not given a corner at all (upstream snaps any distance, which here
    put a click east of the territory on a trail 15 km away)."""
    out, _ = page_results
    near, far = out["snap"]["near"], out["snap"]["far"]
    assert len(near) > 100
    assert all(p["onNode"] for p in near)
    # the pin is the nearest corner (the node 30 m away, or a nearer one),
    # and ``shift`` is how far it is from what was asked for
    assert all(p["shift"] <= 31 and abs(p["shift"] - p["real"]) < 0.2 for p in near)
    assert far["node"] == -1 and (far["lon"], far["lat"]) == (114.52, 22.30), far
    assert far["label"] == "22.3000, 114.5200", far


def test_the_share_link_carries_the_trip(page_results):
    out, _ = page_results
    h = out["hash"]
    # one bare token that survives any host or chat client
    assert h.startswith("#t~") and "~0.500~" in h
    assert all(c.isalnum() or c in "._~-" for c in h[1:]), h


def test_the_frontier_contains_every_weighted_optimum(page_results):
    """Every route a weighted sum length + alpha * climbing would choose is
    a frontier point, so for each alpha the family's best member must cost
    no more, under Python's own evaluation, than Python's route for that
    alpha between the same two nodes. This pins the browser's frontier
    search to the analysis's cost model."""
    from hk_flat_routes.config import ROUTING_PROFILES, with_alpha
    from hk_flat_routes.pipeline import build_context
    from hk_flat_routes.routing import route
    from hk_flat_routes.utils import configure_gdal_for_proxy

    out, _ = page_results
    configure_gdal_for_proxy()
    ctx = build_context(modes=("walk",))
    graph = ctx.graphs["walk"]
    t = graph.table
    key = {(int(e), d): i for i, (e, d) in enumerate(zip(t["edge_id"], t["direction"]))}
    first = out["members"][0]
    arcs0 = [key[(e, "rev" if r else "fwd")] for e, r in first["arcs"]]
    src = t["from_node"].iloc[arcs0[0]]
    dst = t["to_node"].iloc[arcs0[-1]]
    for k, alpha in enumerate(out["alphas"]):
        w = with_alpha(ROUTING_PROFILES["shortest"], alpha)
        cost = graph.build_costs(w)
        py_arcs, _ = route(graph, src, dst, w, arc_cost=cost)
        py_cost = float(cost[py_arcs].sum())
        best = min(out["members"], key=lambda u: u["costs"][k])
        arcs = [key[(e, "rev" if r else "fwd")] for e, r in best["arcs"]]
        js_cost = float(cost[arcs].sum())
        # the search merges frontier points within 0.5 m of climbing (more
        # on a trip that climbs over 150 m, see simple.js) and tolerates a
        # fifth of that per node inside; quantisation adds ~5 cm per arc
        eps = max(0.5, first["gain"] / 300.0)
        tol = alpha * (eps + eps / 5.0 * len(arcs)) + 0.05 * len(arcs) + 1.0
        assert js_cost <= py_cost + tol, (alpha, js_cost, py_cost, tol)


# ------------------------------------------------------ the page around it
_READY = ("window.App && App.family && !App.family.partial"
          " && !document.getElementById('result').hidden")
#: a search has ended: in a family, or in words
_SETTLED = ("window.App && ((App.family && !App.family.partial) || "
            "/^No |same corner/.test(document.getElementById('status').textContent))")


class _Finder:
    """The route page in a browser of its own, with the keyboard."""

    def __init__(self, pw, width=1280, height=800, phone=False):
        try:
            self.browser = pw.chromium.launch(executable_path=_chromium(),
                                              args=["--no-sandbox", "--disable-gpu"])
        except Exception as exc:                            # pragma: no cover
            pytest.skip(f"no usable Chromium: {exc}")
        extra = {"is_mobile": True, "has_touch": True, "device_scale_factor": 2} if phone else {}
        self.page = self.browser.new_page(viewport={"width": width, "height": height}, **extra)
        self.errors: list = []
        self.page.on("pageerror", lambda e: self.errors.append(str(e)))
        self.page.on("console", lambda m: self.errors.append(m.text)
                     if m.type == "error" and "fonts.g" not in m.text
                     and "ERR_" not in m.text else None)
        self.page.goto(SIMPLE_HTML.resolve().as_uri(), wait_until="load", timeout=240_000)
        self.page.wait_for_function(_READY, timeout=240_000)

    def settle(self):
        self.page.wait_for_timeout(350)
        self.page.wait_for_function(_SETTLED, timeout=240_000)
        self.page.wait_for_timeout(700)

    def trip(self, a, b):
        """Set both ends from (lon, lat, name) and wait for the answer."""
        self.page.evaluate("""([a, b]) => {
            App.setPoint('from', App.pointAt(a[0], a[1], a[2]), false);
            App.setPoint('to', App.pointAt(b[0], b[1], b[2]), false);
            App.recompute('auto'); }""", [list(a), list(b)])
        self.settle()
        return self.state()

    def state(self):
        return self.page.evaluate("""() => { const st = App.state, g = App.graph, m = st.mode;
            const end = (p) => p && ({ node: p.node, label: p.label, moved: !!p.moved, shift: p.shift,
                                       comp: g.comp(p.node, m), walk: g.comp(p.node, 'walk') });
            return { from: end(st.from), to: end(st.to), family: !!App.family,
                     status: document.getElementById('status').textContent,
                     fields: [document.getElementById('from').value, document.getElementById('to').value],
                     turns: document.getElementById('turns').textContent }; }""")

    def in_miles(self):
        """The status line as it reads with the card switched to miles, and
        back again."""
        self.page.click("#units button[data-u=mi]")
        self.page.wait_for_timeout(400)
        said = self.page.evaluate("document.getElementById('status').textContent")
        self.page.click("#units button[data-u=km]")
        self.page.wait_for_timeout(400)
        return said

    def type_and_take(self, which, text):
        """Type a search with the real keyboard and take the suggestion of
        that name with the arrow keys (the first, if none is)."""
        self.page.click("#" + which)
        self.page.keyboard.press("ControlOrMeta+a")
        self.page.keyboard.type(text)
        self.page.wait_for_timeout(250)
        names = self.page.evaluate("""(w) => [...document.querySelectorAll('#' + w + '_s li .n')]
            .map(n => n.firstChild.textContent)""", which)
        for _ in range(names.index(text) if text in names else 0):
            self.page.keyboard.press("ArrowDown")
        self.page.keyboard.press("Enter")
        self.settle()
        return names


#: how much of a block of the card is on show
_BLOCK = """(id) => { const e = document.getElementById(id), k = e.querySelector('.k');
    const box = e.getBoundingClientRect(), cap = k ? k.getBoundingClientRect() : null;
    return { style: e.getAttribute('style') || '', client: e.clientHeight, scroll: e.scrollHeight,
             caption: cap ? cap.height > 0 && cap.bottom <= box.bottom + 0.5 : null }; }"""


@pytest.fixture(scope="module")
def card():
    """The page as it opens, on foot: the card, the slider, a download."""
    from playwright.sync_api import sync_playwright
    out: dict = {}
    with sync_playwright() as pw:
        f = _Finder(pw)
        page = f.page
        page.wait_for_timeout(900)          # every block has finished easing open
        out["result"] = page.evaluate(_BLOCK, "result")
        out["turns"] = page.evaluate(_BLOCK, "turns")
        out["words"] = page.evaluate("""() => ({
            how: document.querySelector('details').textContent.replace(/\\s+/g, ' '),
            edges: App.graph.meta.n_edges, arcs: App.graph.m,
            control: document.querySelector('.leaflet-control-attribution').textContent,
            links: [...document.querySelectorAll('.leaflet-control-attribution a')].map(a => a.href) })""")
        # the stats block opened afresh on a card with no figures on it yet,
        # as on a page's first route: the height it starts easing to
        out["opening"] = page.evaluate("""() => { const r = document.getElementById('result'), full = r.offsetHeight;
            r.hidden = true;
            for (const id of ['v_dist', 'v_climb', 'v_grade']) document.getElementById(id).innerHTML = '';
            App.drawStats(App.shown, App.shown);
            return { full, to: parseFloat(r.style.height) }; }""")
        page.wait_for_timeout(500)
        # the short end of the default trip, and every arc there is
        out["steep"] = page.evaluate("""() => { const g = App.graph, sl = document.getElementById('sl');
            sl.value = 0; sl.dispatchEvent(new Event('input'));
            let worst = 0;
            for (let a = 0; a < g.m; a++) worst = Math.max(worst, g.summarise([a]).steepest);
            return { shortest: App.shown.stats.steepest, family: Math.max(...App.family.unique.map(u => u.stats.steepest)), worst }; }""")
        page.wait_for_timeout(500)
        out["steep"]["shown"] = page.evaluate("document.getElementById('v_grade').textContent")
        # the keyboard on the slider
        page.focus("#sl")
        page.keyboard.press("End"); page.wait_for_timeout(150)
        page.keyboard.press("ArrowLeft"); page.wait_for_timeout(150)
        left = page.evaluate("[document.getElementById('slpos').textContent, App.family.unique.length]")
        page.keyboard.press("Home"); page.wait_for_timeout(150)
        page.keyboard.press("ArrowRight"); page.keyboard.press("ArrowRight"); page.wait_for_timeout(150)
        out["keys"] = {"left": left[0], "n": left[1],
                       "right": page.evaluate("document.getElementById('slpos').textContent")}
        # file names: the default trip at its flat end, and a trip named in Chinese
        page.keyboard.press("End"); page.wait_for_timeout(150)
        with page.expect_download() as got:
            page.click("#gpx")
        out["gpx_default"] = got.value.suggested_filename
        out["gpx_trip"] = page.evaluate("App.tripName()")
        page.evaluate("""() => { const a = App.state.from, b = App.state.to;
            App.setPoint('from', App.pointAt(a.qlon, a.qlat, '中環站'), false);
            App.setPoint('to', App.pointAt(b.qlon, b.qlat, '香港大學站'), false); App.recompute('auto'); }""")
        f.settle()
        with page.expect_download() as got:
            page.click("#gpx")
        out["gpx_chinese"] = got.value.suggested_filename
        # and a trip with more of a name than the file's is given room for
        page.evaluate("""() => { const a = App.state.from, b = App.state.to;
            App.setPoint('from', App.pointAt(a.qlon, a.qlat, 'Hong Kong International Airport Passenger Terminal Building'), false);
            App.setPoint('to', App.pointAt(b.qlon, b.qlat, 'The Chinese University of Hong Kong'), false); App.recompute('auto'); }""")
        f.settle()
        with page.expect_download() as got:
            page.click("#gpx")
        out["gpx_long"] = got.value.suggested_filename
        out["gpx_long_trip"] = page.evaluate("App.tripName()")
        # Labels for pins dropped on the walking network: all over it, and
        # wherever the nearest named corner within reach is on another
        # landmass (a fragment beside the shore, the shore beside a fragment).
        out["labels"] = page.evaluate("""() => { const g = App.graph, idx = App.index, names = App.geom.names, grid = App.nodeGrid;
            const kx = 111320 * Math.cos(22.35 * Math.PI / 180), ky = 110540, out = [];
            const m = (a, b) => Math.hypot((g.nodeLon(a) - g.nodeLon(b)) * kx, (g.nodeLat(a) - g.nodeLat(b)) * ky);
            const streets = (k) => (idx.nodeNames[k] || []).map(n => names[n - 1]);
            const corner = (k) => streets(k).length >= 2 ? 'near ' + streets(k).slice(0, 2).join(' & ') : null;
            // a pin's label, and how far the corner or street it names is
            // on the pin's own landmass (null: not there within 600 m)
            const pin = (i, where) => { const c = g.comp(i, 'walk'), label = idx.describe(i, grid, 'walk');
                if (!label.startsWith('near ')) return { label, where, kind: label.startsWith('on ') ? 'land' : 'coords', land: g.compName(i, 'walk') };
                const own = (f) => grid.nearest(g.nodeLon(i), g.nodeLat(i), (k) => g.comp(k, 'walk') === c && f(k), 600);
                let k = own((k) => corner(k) === label), kind = 'corner';
                if (k < 0) { k = own((k) => streets(k).some(n => 'near ' + n === label)); kind = 'street'; }
                return { label, where, kind, away: k < 0 ? null : m(i, k) }; };
            for (let i = 0; i < g.n; i += 499) if (g.comp(i, 'walk')) out.push(pin(i, 'anywhere'));
            let across = 0;
            for (let i = 0; i < g.n && across < 300; i++) { const c = g.comp(i, 'walk'); if (!c || corner(i)) continue;
                const k = grid.nearest(g.nodeLon(i), g.nodeLat(i), (k) => g.comp(k, 'walk') > 0 && corner(k) !== null, 500);
                if (k >= 0 && g.comp(k, 'walk') !== c) { out.push(pin(i, 'across')); across++; } }
            return out; }""")
        # a loop that cannot be found leaves no line of the search behind
        page.evaluate("""() => { App.setPoint('from', App.pointAt(114.0400, 22.2850, 'Peng Chau'), false); }""")
        page.click("#loopbtn")
        page.evaluate("""() => { const sl = document.getElementById('sl'); sl.value = sl.max;
            sl.dispatchEvent(new Event('input')); sl.dispatchEvent(new Event('change')); }""")
        page.wait_for_function(
            "/No loop|closest|flattest of|within/.test(document.getElementById('status').textContent)",
            timeout=120_000)
        page.wait_for_timeout(1200)
        out["loop"] = page.evaluate("""() => ({ status: document.getElementById('status').textContent,
            delta: document.getElementById('delta').textContent, family: !!App.family,
            display: getComputedStyle(document.getElementById('result')).display,
            hash: location.hash, token: App.token() })""")
        out["errors"] = f.errors
        f.browser.close()
    return out


def test_the_stats_keep_their_captions_once_the_page_has_loaded(card):
    """The first route is painted and then, a moment later, the family; the
    second paint used to catch the stats block half open and leave it at
    the height it had before the numbers were written (18 px of 40)."""
    r = card["result"]
    assert "height" not in r["style"] and "overflow" not in r["style"], r
    assert r["client"] == r["scroll"] > 30 and r["caption"], r
    t = card["turns"]
    assert "height" not in t["style"] and t["client"] == t["scroll"] > 10, t
    assert not card["errors"], card["errors"][:4]


def test_the_stats_open_to_the_height_of_their_figures(card):
    """A block eases open to the height it has when it is measured, and the
    stats were measured before their figures were written: on a page's
    first route they opened to the captions alone and jumped to the rest
    200 ms later."""
    o = card["opening"]
    assert o["full"] > 30 and o["to"] == pytest.approx(o["full"], abs=0.5), o


def test_steepest_is_never_over_the_ceiling_gradients_are_held_to(card):
    """Every stored gradient is clipped at 60%; the route page's own
    figure for a short arc (its average climb) was not, and read 110% at
    the short end of the trip the page opens on."""
    ceiling = ELEVATION.max_plausible_grade
    s = card["steep"]
    assert s["worst"] <= ceiling + 1e-9, s
    assert s["shortest"] <= ceiling + 1e-9 and s["family"] <= ceiling + 1e-9, s
    assert float(s["shown"].rstrip("%")) <= 100 * ceiling + 0.5, s


def test_an_arrow_key_on_the_slider_steps_one_route(card):
    k = card["keys"]
    assert k["n"] >= 4, "the default trip should have routes to step through"
    assert k["left"] == f"{k['n'] - 1} of {k['n']}", k
    assert k["right"] == f"3 of {k['n']}", k


def test_how_this_works_counts_the_segments_the_page_holds(card):
    w = card["words"]
    assert f"over {w['edges']:,} street and path segments" in w["how"], w["how"][:160]
    assert f"{w['arcs']:,}" not in w["how"]
    # both portals the Government's data came through, named on the map
    assert "CSDI Portal" in w["control"] and "DATA.GOV.HK" in w["control"], w["control"]
    assert any("portal.csdi.gov.hk" in h for h in w["links"])
    assert any("data.gov.hk" in h for h in w["links"])
    assert "CSDI Portal" in w["how"] and "DATA.GOV.HK" in w["how"]


def test_a_gpx_file_is_named_after_its_trip_in_any_script(card):
    """Whole words, and Chinese kept: a trip between two stations chosen
    in Chinese used to download as "flattenhk-to-flattest.gpx"."""
    name = card["gpx_default"]
    assert name.startswith("flattenhk-") and name.endswith("-flattest.gpx"), name
    words = [w.lower() for w in __import__("re").findall(r"[A-Za-z0-9]+", card["gpx_trip"])]
    stem = name[len("flattenhk-"):-len("-flattest.gpx")]
    assert stem.split("-") == words[:len(stem.split("-"))], (stem, words)
    assert card["gpx_chinese"] == "flattenhk-中環站-to-香港大學站-flattest.gpx", card["gpx_chinese"]
    # a name too long is cut between two words, and still says which end
    # of the slider it is: 80 characters in all
    long = card["gpx_long"]
    assert long.startswith("flattenhk-") and long.endswith("-flattest.gpx"), long
    stem = long[len("flattenhk-"):-len("-flattest.gpx")]
    whole = "-".join(w.lower() for w in __import__("re").findall(r"[A-Za-z0-9]+", card["gpx_long_trip"]))
    assert len(whole) > 80 and whole.startswith(stem + "-"), (stem, whole)
    assert 60 < len(stem) + len("-flattest") <= 80, long


def test_a_dropped_pin_is_named_after_somewhere_near_on_its_own_landmass(card):
    """No limit on either used to name a pin on Po Toi after a junction
    6.7 km away on Hong Kong Island."""
    labels = card["labels"]
    named = [r for r in labels if r["kind"] in ("corner", "street")]
    assert sum(r["kind"] == "corner" for r in named) > 100
    # on the pin's own landmass and within 500 m (simple.js NEAR_LABEL_M),
    # with a little for the grid's own metric
    off = [r for r in named if r["away"] is None or r["away"] > 505]
    assert not off, off[:3]
    # and where there is no junction to name, something true is said instead
    assert any(r["kind"] in ("street", "land") for r in labels)
    assert all(r["label"] == "on " + r["land"] for r in labels if r["kind"] == "land")
    assert not any(r["label"].startswith("near near") for r in labels)


def test_a_loop_that_fails_leaves_no_line_of_the_search_on_the_card(card):
    r = card["loop"]
    if r["family"]:
        pytest.skip("Peng Chau has a loop of the longest length in this build")
    assert r["status"].startswith("No "), r
    assert r["delta"] == "" and r["display"] == "none", r
    assert r["hash"] == "#" + r["token"] and r["hash"].startswith("#l~"), r


#: Trips long enough for a frontier of more than thirty routes by bike:
#: Tuen Mun Station to Sha Tin Station, Yuen Long Station to Tai Po Market
#: Station (91 and 40 solutions in the build these were chosen on).
_LONG_BIKE = [((113.97319, 22.39511, "Tuen Mun Station"), (114.18691, 22.38213, "Sha Tin Station")),
              ((114.03517, 22.44607, "Yuen Long Station"), (114.17039, 22.44458, "Tai Po Market Station"))]

#: A node on a piece of the mode's network with another piece 70 to 380 m
#: away, once for a fragment (under 150 nodes) and once for a larger piece;
#: with a second node of the other piece to route to, and a second of its own.
#: (The page must be in that mode: a node is used only if its own spot snaps
#: to it, which where two nodes share a spot it may not.)
_FIND_NEIGHBOURS = """(mode) => {
    const g = App.graph, kx = 111320 * Math.cos(22.35 * Math.PI / 180), ky = 110540;
    const m = (a, b) => Math.hypot((g.nodeLon(a) - g.nodeLon(b)) * kx, (g.nodeLat(a) - g.nodeLat(b)) * ky);
    const own = (n) => App.nearestNode(g.nodeLon(n), g.nodeLat(n)) === n;
    const out = {};
    for (let i = 0; i < g.n && !(out.small && out.big); i++) {
        const c = g.comp(i, mode); if (!c) continue;
        const size = g.compNodes(c, mode); if (size > 5000) continue;
        const kind = size < 150 ? 'small' : 'big'; if (out[kind] || !own(i)) continue;
        const j = App.nodeGrid.nearest(g.nodeLon(i), g.nodeLat(i), (k) => g.comp(k, mode) > 0 && g.comp(k, mode) !== c, 380);
        if (j < 0 || m(i, j) < 70) continue;
        const other = g.comp(j, mode);
        let far = -1, mate = -1;
        for (let k = 0; k < g.n && far < 0; k++) if (g.comp(k, mode) === other && m(k, j) > 300 && m(k, j) < 3000) far = k;
        for (let k = 0; k < g.n && mate < 0; k++) if (g.comp(k, mode) === c && m(k, i) > 30) mate = k;
        if (far < 0 || mate < 0) continue;
        const ll = (n) => [g.nodeLon(n), g.nodeLat(n)];
        out[kind] = { node: i, at: ll(i), far: ll(far), mate: ll(mate), size, gap: m(i, j),
                      sameWalk: g.comp(i, 'walk') > 0 && g.comp(i, 'walk') === g.comp(j, 'walk') };
    }
    return out; }"""


#: Two nodes of one landmass of the cycling network with a way from the
#: first to the second and none back: a landmass takes in the streets a
#: one-way system leads into and not out of (routing.component_ranks).
#: (The page must be in bike mode, as above.)
_FIND_TRAP = """() => { const g = App.graph, seen = new Set();
    const cost = new Float64Array(g.m); for (let a = 0; a < g.m; a++) cost[a] = g.arcLen[a] / g.DM;
    const ll = (n) => [g.nodeLon(n), g.nodeLat(n)];
    const own = (n) => App.nearestNode(g.nodeLon(n), g.nodeLat(n)) === n;
    for (let i = 0; i < g.n; i++) {
        const c = g.comp(i, 'bike'); if (!c || seen.has(c) || !own(i)) continue; seen.add(c);
        const there = g.tree(i, 'bike', cost), back = g.tree(i, 'bike', cost, { reverse: true });
        for (let k = 0; k < g.n; k++) {
            if (g.comp(k, 'bike') !== c || there.done[k] === back.done[k] || !own(k)) continue;
            const [a, b] = there.done[k] ? [i, k] : [k, i];
            return { way: ll(a), noWayBack: ll(b), nodes: [a, b], size: g.compNodes(c, 'bike') };
        }
    }
    return null; }"""


def _neighbours(f, mode):
    """Trips between neighbouring pieces of a mode's network (see above)."""
    found = f.page.evaluate(_FIND_NEIGHBOURS, mode)
    out = {}
    for kind, c in found.items():
        r = f.trip(c["at"] + ["A chosen place"], c["far"] + ["Another place"])
        r["case"] = c
        r["miles"] = f.in_miles()
        if kind == "small":
            # then a destination on the fragment itself, the start left as
            # the trip before it put it
            f.page.evaluate("""(ll) => { App.setPoint('to', App.pointAt(ll[0], ll[1], 'On the fragment'), false);
                App.recompute('auto'); }""", c["mate"])
            f.settle()
            r["after"] = f.state()
            # and the first trip again, as if the build had a name for the
            # walking landmass the fragment stands on, as it has for an island
            f.page.evaluate("""(n) => { const g = App.graph, names = g.meta.components.walk, c = g.comp(n, 'walk');
                window._name = [c, names[c - 1]]; if (c) names[c - 1] = 'Fragment Island'; }""", c["node"])
            r["named"] = f.trip(c["at"] + ["A chosen place"], c["far"] + ["Another place"])
            f.page.evaluate("""() => { const [c, was] = window._name; if (c) App.graph.meta.components.walk[c - 1] = was; }""")
        out[kind] = r
    return out


@pytest.fixture(scope="module")
def ends():
    """Where the ends of a trip are put, on foot and then by bike."""
    from playwright.sync_api import sync_playwright
    out: dict = {}
    with sync_playwright() as pw:
        f = _Finder(pw)
        page = f.page
        out["walk"] = _neighbours(f, "walk")
        # a click in Shenzhen, through the map's own handler
        f.trip((114.15995, 22.28105, "Statue Square"), (114.1379, 22.2839, "The University"))
        page.evaluate("""() => { App.state.focus = 'to';
            App.map.fire('click', { latlng: L.latLng(22.545, 114.06) }); }""")
        f.settle()
        out["click"] = f.state()
        out["click"]["miles"] = f.in_miles()
        # a summit no bicycle may be ridden to, asked for on foot and then by bike
        hit = page.evaluate("""() => { const g = App.graph;
            for (let i = 0; i < g.n; i++) {
                if (!(g.nodeFlags[i] & 1)) continue;
                if (App.nodeGrid.nearest(g.nodeLon(i), g.nodeLat(i), (k) => (g.nodeFlags[k] & 2) !== 0, 600) < 0) return [g.nodeLon(i), g.nodeLat(i)];
            }
            return null; }""")
        if hit:
            walked = f.trip(hit + ["Up a path"], (114.1758, 22.3376, "Kowloon Tong"))
            page.click("#mode button[data-v=bike]"); f.settle()
            ridden = f.state()
            page.click("#mode button[data-v=walk]"); f.settle()
            out["summit"] = {"walk": walked, "bike": ridden, "back": f.state()}
        page.click("#mode button[data-v=bike]"); f.settle()
        out["bike"] = _neighbours(f, "bike")
        # into a one-way trap, and out of it
        trap = page.evaluate(_FIND_TRAP)
        if trap:
            out["trap"] = {"case": trap,
                           "in": f.trip(trap["way"] + ["Outside"], trap["noWayBack"] + ["Inside"]),
                           "out": f.trip(trap["noWayBack"] + ["Inside"], trap["way"] + ["Outside"])}
        # Two junctions typed and taken from the list: one whose own node no
        # bicycle can use (a kerb corner, with a street beside it), and one
        # 200 m to 1.5 km off on the network that street belongs to.
        pair = page.evaluate("""() => { const g = App.graph, its = App.index.intersections;
            const kx = 111320 * Math.cos(22.35 * Math.PI / 180), ky = 110540;
            const once = new Map(); for (const it of its) once.set(it.name, (once.get(it.name) || 0) + 1);
            let own = 0, snapped = 0, unusable = 0, a = null, b = null;
            for (const it of its) {
                if (g.comp(it.node, 'bike') === 0) own++;
                const p = App.pointAt(it.lon, it.lat, it.name);
                if (p.node < 0) continue;
                snapped++; if (g.comp(p.node, 'bike') === 0) unusable++;
                if (!a && g.comp(it.node, 'bike') === 0 && once.get(it.name) === 1 && g.compNodes(g.comp(p.node, 'bike'), 'bike') > 5000) a = { name: it.name, comp: g.comp(p.node, 'bike'), lon: it.lon, lat: it.lat };
            }
            for (const it of its) {
                if (!a || b) break;
                const d = Math.hypot((it.lon - a.lon) * kx, (it.lat - a.lat) * ky);
                if (g.comp(it.node, 'bike') === a.comp && once.get(it.name) === 1 && d > 200 && d < 1500) b = { name: it.name };
            }
            // suggestions left on a node one mode cannot use although
            // another node where the same streets meet serves both
            const streets = (u) => App.index.nodeNames[u].map(i => App.geom.names[i - 1]).sort().join('|');
            const shared = new Set();
            for (let u = 0; u < g.n; u++) { const s = App.index.nodeNames[u]; if (s && s.length >= 2 && g.nodeFlags[u] === 3) shared.add(streets(u)); }
            const passedOver = its.filter(it => g.nodeFlags[it.node] !== 3 && shared.has(streets(it.node))).length;
            return { n: its.length, own, snapped, unusable, passedOver, a, b }; }""")
        out["junction_nodes"] = pair
        if pair["a"] and pair["b"]:
            f.type_and_take("from", pair["a"]["name"])
            out["junction_from"] = f.state()
            f.type_and_take("to", pair["b"]["name"])
            out["junctions"] = f.state()
        # a long ride with calm streets preferred (the default by bike)
        for a, b in _LONG_BIKE:
            r = f.trip(a, b)
            r.update(page.evaluate("""() => { if (!App.family) return { solutions: 0 };
                const key = App.lenKey(), sl = document.getElementById('sl'), n = App.family.unique.length, rows = [];
                for (let i = 0; i < n; i++) { sl.value = n > 1 ? i / (n - 1) : 1; sl.dispatchEvent(new Event('input'));
                    rows.push([App.shown.stats[key], App.shown.stats.elev_gain_m, App.shown.stats.distance_m]); }
                return { solutions: App._search.solutions.length, calm: App.calm(), key, rows }; }"""))
            out["long"] = r
            if r["solutions"] > 30:
                break
        # pushed links, where the bundle has any
        out["push"] = page.evaluate("""() => { const g = App.graph;
            if (!g.arcPush) return null;
            const tail = g.reverse().tail, w = g.meta.profiles.balanced, flat = Object.assign({}, w, { alpha: 0, beta: 0, gamma: 0 });
            // a pushed link on a piece of the cycling network too small to
            // keep is marked and not open: the count is of those in use
            let n = 0, a = -1;
            for (let k = 0; k < g.m; k++) { if (!g.arcPush[k] || !(g.arcFlags[k] & 2)) continue; n++; if (a < 0 && g.arcLen[k] > 200) a = k; }
            if (a < 0) return { n };
            const L = g.arcLen[a] / g.DM;
            return { n, mult: g.pushMult, meta: g.meta.push_multiplier, L, cls: g.meta.classes[g.arcCls[a]],
                     classMult: g.meta.multipliers.bike[g.arcCls[a]], stress: g.edgeStress[g.arcEdge[a]] / g.meta.scales.stress,
                     bike: g.pathCost([a], 'bike', flat), walk: (g.arcFlags[a] & 1) ? g.pathCost([a], 'walk', flat) : null,
                     walkMult: g.meta.multipliers.walk[g.arcCls[a]],
                     calm: g.arcLenStress[a] / g.DM, plain: g.pathCost([a], 'bike', Object.assign(App.weights(0), { stress: false })),
                     from: [g.nodeLon(tail[a]), g.nodeLat(tail[a])], to: [g.nodeLon(g.head[a]), g.nodeLat(g.head[a])] }; }""")
        if out["push"] and "from" in out["push"]:
            page.evaluate("() => { if (App.state.calm) document.getElementById('calm').click(); }")
            r = f.trip(out["push"]["from"] + ["One end of a link"], out["push"]["to"] + ["The other end"])
            r.update(page.evaluate("""() => ({ pushed: App.shown ? App.shown.arcs.filter(a => App.graph.arcPush[a]).length : 0,
                runs: App.shown ? App.runs(App.shown, 60).filter(r => r.push).map(r => App.runText(r)) : [] })"""))
            out["push"]["trip"] = r
            # two pushed pieces end to end under different names (one may
            # have none), as the street list gives them
            out["push"]["stretch"] = page.evaluate("""() => { const g = App.graph, geom = App.geom, tail = g.reverse().tail;
                const name = (a) => { const n = geom.name[g.arcEdge[a]]; return n ? geom.names[n - 1] : null; };
                const ok = (a) => g.arcPush[a] && (g.arcFlags[a] & 2);
                for (let a = 0; a < g.m; a++) { if (!ok(a) || !name(a)) continue;
                    for (let b = g.indptr[g.head[a]]; b < g.indptr[g.head[a] + 1]; b++) {
                        if (!ok(b) || g.head[b] === tail[a] || name(b) === name(a) || g.arcLen[b] >= g.arcLen[a]) continue;
                        return { names: [name(a), name(b)], lengths: [g.arcLen[a] / g.DM, g.arcLen[b] / g.DM],
                                 runs: App.runs({ arcs: [a, b] }, 60)
                                     .map(r => ({ name: r.name, push: r.push, length_m: r.length_m, arcs: r.arcs.length })) }; } }
                return null; }""")
        out["errors"] = f.errors
        f.browser.close()
    return out


def _check_no_silent_move(r, mode):
    """One trip from a piece of the network to its neighbour: the start is
    either where it was asked for, or moved and said to have been."""
    case, start = r["case"], r["from"]
    if not start["moved"]:
        assert start["node"] == case["node"] and start["label"] == "A chosen place", r
        assert r["fields"][0] == "A chosen place" and not r["family"], r
        assert r["status"].startswith("No way "), r
        assert "moved" not in r["status"], r
        return False
    # moved: off a fragment only, onto the destination's piece, and not far
    assert case["size"] < 150, r
    assert start["comp"] == r["to"]["comp"] and start["shift"] <= 400.5, r
    # it is no longer called what was asked for, and the status says how far it went
    assert start["label"] != "A chosen place" and r["fields"][0] == start["label"], r
    assert start["label"].startswith(("near ", "on ")) or start["label"][0].isdigit(), r
    metres = round(start["shift"] / 10) * 10
    assert f"Start moved {metres} m to the nearest street joined to the destination." in r["status"], r
    # and in feet once the card is switched to miles, not left in metres
    feet = round(start["shift"] * 3.28084 / 50) * 50
    assert f"Start moved {feet:,} ft to the nearest street" in r["miles"] and " m to " not in r["miles"], r
    return True


@pytest.mark.parametrize("mode", ["walk", "bike"])
def test_a_pin_is_never_moved_silently_and_never_off_a_real_place(ends, mode):
    """An end on a fragment of the network (a pier, a podium) is moved to
    the street beside it; that used to be done for any end within 400 m of
    the other's network, without a word and under the name the user chose,
    so that "Aberdeen Promenade" was a pin across the channel on Ap Lei
    Chau."""
    cases = ends[mode]
    if not cases:
        pytest.skip(f"no two pieces of the {mode} network lie 70 to 380 m apart in this build")
    if "small" in cases:
        moved = _check_no_silent_move(cases["small"], mode)
        if mode == "walk":
            assert moved, cases["small"]
        if moved:
            # asked next for a trip within the fragment, the pin goes back
            back = cases["small"]["after"]["from"]
            assert back["node"] == cases["small"]["case"]["node"] and not back["moved"], back
            assert back["label"] == "A chosen place"
            # and a place with a name of its own is not left for another,
            # however few its streets: Sharp Island is 44 nodes on foot
            named = cases["small"]["named"]
            if not cases["small"]["case"]["sameWalk"]:
                assert not named["from"]["moved"] and named["from"]["label"] == "A chosen place", named
                assert not named["family"] and named["status"].startswith("No way "), named
    if "big" in cases:
        assert not _check_no_silent_move(cases["big"], mode), cases["big"]


def test_two_places_on_one_island_are_not_said_to_be_on_different_islands(ends):
    """A cyclist's network is in more pieces than the land is."""
    statuses = [r["status"] for cases in (ends["walk"], ends["bike"]) for r in cases.values()
                if not r["family"]]
    assert all("different islands" not in s for s in statuses), statuses
    big = ends["bike"].get("big")
    if not big or not big["case"]["sameWalk"]:
        pytest.skip("no large piece of the bike network beside another on one landmass in this build")
    assert big["status"] == ("No way by bike between those two: the streets open to "
                             "bicycles do not join up between them."), big


def test_a_ride_that_can_only_be_made_the_other_way_round_is_said_to_be(ends):
    """Two places on one landmass need not have a route both ways: Wan Chai
    North can be ridden into from the rest of Hong Kong Island and, in the
    data, not out of.  That is neither "different islands" nor streets that
    "do not join up"."""
    if "trap" not in ends:
        pytest.skip("every landmass of the cycling network is strongly connected in this build")
    t = ends["trap"]
    for r in (t["in"], t["out"]):
        assert [r["from"]["node"], r["to"]["node"]] in (t["case"]["nodes"], t["case"]["nodes"][::-1]), r
        assert r["from"]["comp"] == r["to"]["comp"] > 0 and not r["from"]["moved"] and not r["to"]["moved"], r
    assert t["in"]["family"], t["in"]
    assert not t["out"]["family"], t["out"]
    assert t["out"]["status"] == ("No bikeable route from the start to the destination, only from the "
                                  "destination to the start: the one-way streets in the data do not lead back."), t["out"]


def test_a_point_far_from_any_street_is_refused_in_words(ends):
    c = ends["click"]
    assert not c["family"] and c["to"]["node"] == -1, c
    assert c["status"] == "No street or path open to walkers within 400 m of 22.5450, 114.0600.", c
    assert c["miles"] == "No street or path open to walkers within 1,300 ft of 22.5450, 114.0600.", c
    assert c["fields"][1] == "22.5450, 114.0600"
    if "summit" not in ends:
        pytest.skip("every walkable node has a bikeable one within 600 m in this build")
    s = ends["summit"]
    assert s["walk"]["from"]["node"] >= 0
    assert s["bike"]["from"]["node"] == -1 and not s["bike"]["family"], s["bike"]
    assert s["bike"]["status"] == "No street a bicycle may use within 400 m of Up a path.", s["bike"]
    # the place is kept, and on foot again it is where it was
    assert s["bike"]["fields"][0] == "Up a path"
    assert s["back"]["from"]["node"] == s["walk"]["from"]["node"]
    assert s["back"]["status"] == s["walk"]["status"]


def test_a_junction_taken_from_the_list_routes_by_bike(ends):
    """A junction suggestion stands on one node of the graph, and for 1,884
    of the 7,896 that was a kerb corner no bicycle can use; picked, it was
    used as it stood, and two junctions on Western Street were "on
    different islands"."""
    n = ends["junction_nodes"]
    assert n["unusable"] == 0 and n["snapped"] > 0.9 * n["n"], n
    # where the same streets meet at several nodes, the suggestion stands
    # on one both modes can use, so a walk and a ride start from one corner
    assert n["passedOver"] == 0, n
    assert not ends["errors"], ends["errors"][:4]
    if "junctions" not in ends:
        pytest.skip("every junction suggestion stands on a node a bicycle can use in this build")
    # the pick itself stands on a corner a bicycle can use, under the name
    # that was chosen: nothing had to move it there afterwards
    first = ends["junction_from"]
    assert first["from"]["label"] == n["a"]["name"] and first["from"]["comp"] == n["a"]["comp"], first
    assert not first["from"]["moved"] and "moved" not in first["status"], first
    j = ends["junctions"]
    assert j["fields"] == [n["a"]["name"], n["b"]["name"]], j
    # same network, so the answer is a route (or, on a landmass with a
    # one-way trap, that there is none this way), never "no way between"
    assert j["from"]["comp"] == j["to"]["comp"] == n["a"]["comp"], j
    assert j["family"] or j["status"].startswith("No bikeable route"), j


def test_a_long_calm_bike_family_runs_from_shortest_to_flattest(ends):
    """With calm streets on, the frontier is searched and sorted on
    comfort-weighted length.  A family of more than thirty is thinned, and
    the thinning sorted it again on real distance: climbing then went up
    and down along the slider and the "flattest" end was not the flattest."""
    r = ends["long"]
    assert r["solutions"] > 30, "neither trip has a frontier of over thirty routes in this build"
    assert r["calm"] and r["key"] == "stress_m"
    rows = r["rows"]
    assert len(rows) == 30
    for a, b in zip(rows, rows[1:]):
        assert b[0] >= a[0] - 1e-6, (a, b)
        assert b[1] <= a[1] + 1e-6, (a, b)
    assert rows[-1][1] == min(x[1] for x in rows)
    assert rows[0][0] == min(x[0] for x in rows)


def test_a_pushed_link_is_charged_and_named_in_the_browser_as_in_python(ends):
    """One real pushed link, priced by the page: the push multiplier once
    on its length under the class multipliers (routing.edge_costs) and once
    in its comfort-weighted length (bikeways.stress; the bundle ships the
    class's weight alone and the engine multiplies the rest in), and not at
    all on real distance or on foot."""
    from hk_flat_routes.bikeways import CLASS_STRESS, DEFAULT_STRESS

    p = ends["push"]
    if p is None or not p["n"]:
        pytest.skip("this bundle has no pushed links")
    if "L" not in p:
        pytest.skip("no pushed link over 10 m in this build")
    assert p["mult"] == p["meta"] == BIKE_PUSH_MULTIPLIER
    assert p["bike"] == pytest.approx(p["L"] * p["classMult"] * BIKE_PUSH_MULTIPLIER, abs=0.01)
    comfort = CLASS_STRESS.get(p["cls"], DEFAULT_STRESS)
    assert p["stress"] == pytest.approx(comfort, abs=0.006)
    assert p["calm"] == pytest.approx(p["L"] * comfort * BIKE_PUSH_MULTIPLIER, abs=0.05)
    assert p["plain"] == pytest.approx(p["L"], abs=0.01)
    if p["walk"] is not None:
        assert p["walk"] == pytest.approx(p["L"] * p["walkMult"], abs=0.01)
    # ridden end to end on real distance, the link is what the route takes,
    # and the list of streets says it is walked
    trip = p["trip"]
    if trip["pushed"]:
        assert trip["runs"] and all(t.endswith(" (push)") for t in trip["runs"]), trip
        assert "(push)" in trip["turns"], trip
    # one stretch on foot is one entry, under the name of its longest piece
    st = p["stretch"]
    if st:
        assert st["runs"] == [{"name": st["names"][0], "push": True, "arcs": 2,
                               "length_m": pytest.approx(sum(st["lengths"]))}], st


@pytest.fixture(scope="module")
def phone():
    """The page on a phone held upright (390 x 844)."""
    from playwright.sync_api import sync_playwright
    out: dict = {}
    with sync_playwright() as pw:
        f = _Finder(pw, 390, 844, phone=True)
        page = f.page
        page.wait_for_timeout(900)
        out["result"] = page.evaluate(_BLOCK, "result")
        out["turns"] = page.evaluate(_BLOCK, "turns")
        # a block as it is while it eases open: clipped, at a set height
        out["easing"] = page.evaluate("""() => { const r = document.getElementById('result'), card = document.getElementById('card');
            const full = r.scrollHeight; r.style.overflow = 'hidden'; r.style.height = full + 'px';
            const got = r.getBoundingClientRect().height; r.style.overflow = ''; r.style.height = '';
            return { full, got, scrolls: card.scrollHeight > card.clientHeight }; }""")
        out["credit"] = page.evaluate("""() => { const a = document.querySelector('.leaflet-control-attribution'), r = a.getBoundingClientRect();
            const at = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
            const card = document.getElementById('card').getBoundingClientRect();
            return { top: r.top, bottom: r.bottom, height: r.height, text: a.textContent, seen: !!at && a.contains(at),
                     cardTop: card.top, viewport: innerHeight }; }""")
        lists = {}
        for which in ("from", "to"):
            page.click("#" + which)
            page.keyboard.press("ControlOrMeta+a")
            page.keyboard.type("nathan road")
            page.wait_for_timeout(400)
            lists[which] = page.evaluate("""(w) => { const l = document.getElementById(w + '_s'), b = l.getBoundingClientRect();
                const f = document.getElementById(w).getBoundingClientRect(), first = l.querySelector('li').getBoundingClientRect();
                const at = document.elementFromPoint(first.left + 20, first.top + first.height / 2);
                return { top: b.top, bottom: b.bottom, n: l.querySelectorAll('li').length, fieldTop: f.top,
                         firstSeen: !!at && l.contains(at), viewport: innerHeight }; }""", which)
            page.keyboard.press("Escape")
            page.wait_for_timeout(300)
        out["lists"] = lists
        # and a refusal on the small screen
        r = f.trip((114.1578, 22.2816, "Central"), (114.1722, 22.2975, "Tsim Sha Tsui"))
        r["display"] = page.evaluate("""['result', 'prof', 'turns'].map(id => getComputedStyle(document.getElementById(id)).display)""")
        out["refusal"] = r
        out["errors"] = f.errors
        f.browser.close()
    return out


def test_on_a_phone_the_stats_and_the_street_list_are_on_the_card(phone):
    """A block left clipped (see the captions test) is squeezed to nothing
    by the docked card's flex layout: 2 px of stats, 7 px of streets."""
    r, t = phone["result"], phone["turns"]
    assert r["client"] == r["scroll"] > 30 and r["caption"], r
    assert t["client"] == t["scroll"] > 10, t
    assert not phone["errors"], phone["errors"][:4]
    assert phone["refusal"]["display"] == ["none", "none", "none"], phone["refusal"]


def test_on_a_phone_a_block_easing_open_is_not_squeezed(phone):
    """The docked card is a column of flex items at most 58% of the screen
    high, and a flex item that clips its content may be shrunk to nothing.
    A block is clipped for the 200 ms it takes to open; the card scrolls
    instead of squeezing it."""
    e = phone["easing"]
    assert e["scrolls"], "the card is expected to be taller than its share of the screen"
    assert e["got"] == pytest.approx(e["full"], abs=0.5), e


def test_on_a_phone_the_data_credits_stay_in_sight(phone):
    """The card is docked over the corner Leaflet puts them in, and the
    terms the data is used under require its sources to be seen."""
    c = phone["credit"]
    assert c["height"] > 0 and c["seen"], c
    assert c["bottom"] <= c["cardTop"], c
    assert "OpenStreetMap" in c["text"] and "HKSAR Government" in c["text"]


def test_on_a_phone_suggestions_open_upward_onto_the_screen(phone):
    for which, box in phone["lists"].items():
        assert box["n"] >= 5, (which, box)
        assert 0 <= box["top"] and box["bottom"] <= box["fieldTop"] <= box["viewport"], (which, box)
        assert box["firstSeen"], (which, box)


# ------------------------------------------------------------- the site
@pytest.fixture(scope="module")
def served_site():
    """The static site over HTTP, as GitHub Pages serves it."""
    import functools
    import http.server
    import threading

    if not SITE_INDEX.exists():
        pytest.skip("site not built")
    handler = functools.partial(http.server.SimpleHTTPRequestHandler,
                                directory=str(SITE_INDEX.parent))
    handler.log_message = lambda *a, **k: None
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/"
    srv.shutdown()


def test_the_site_loads_its_graph_over_http(served_site):
    from playwright.sync_api import sync_playwright
    errors: list = []
    with sync_playwright() as pw:
        try:
            browser = pw.chromium.launch(executable_path=_chromium(),
                                         args=["--no-sandbox", "--disable-gpu"])
        except Exception as exc:                            # pragma: no cover
            pytest.skip(f"no usable Chromium: {exc}")
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: errors.append(m.text)
                if m.type == "error" and "fonts.g" not in m.text
                and "ERR_" not in m.text else None)
        failed: list = []
        page.on("requestfailed", lambda r: failed.append(r.url)
                if served_site in r.url else None)
        page.goto(served_site, wait_until="load", timeout=240_000)
        page.wait_for_function(
            "window.App && App.family && !document.getElementById('result').hidden",
            timeout=240_000)
        out = page.evaluate("""() => ({
            inline: !!window.DATA.bundle, url: window.DATA.bundle_url,
            hillshade: window.DATA.hillshade && window.DATA.hillshade.url,
            shade: !!document.querySelector('img.hillshade') && document.querySelector('img.hillshade').naturalWidth,
            routes: App.family.unique.length, status: document.getElementById('status').textContent,
        })""")
        browser.close()
    assert not errors, errors[:4]
    assert not failed, failed
    assert not out["inline"] and out["url"].startswith("data/graph-")
    assert out["hillshade"].startswith("data/hillshade-") and out["shade"] > 1000
    assert out["routes"] >= 2
    import re
    index = SITE_INDEX.read_text(encoding="utf-8")
    refs = re.findall(r'(?:href|src)="([^"]+)"', index)
    local = [r for r in refs if not r.startswith("http")]
    assert any(re.match(r"app-[0-9a-f]{10}\.js$", r) for r in local), local
    for name in local + [".nojekyll", out["url"], out["hillshade"]]:
        assert (SITE_INDEX.parent / name).exists(), name
