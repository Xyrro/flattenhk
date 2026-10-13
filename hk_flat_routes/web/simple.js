/* Hong Kong flat routes -- the route page.
 * Adapted from flattensf (MIT) by Drew Edwards,
 * https://github.com/almostimplemented/flattensf
 * (the copyright lines and where the licence is are at the top of engine.js)
 *
 * One card: where from, where to, and a slider from the shortest route to
 * the flattest. Everything runs in the page: the graph and the cost model
 * come from engine.js, and place search is an offline index built from the
 * graph's own street names plus Overture places and addresses packed into
 * the bundle (see hk_flat_routes/places.py).
 *
 * The slider is a family of routes, not one route: the whole frontier of
 * distance against climbing between the two points, every route that no
 * other route beats on both counts, sorted from shortest to flattest. A
 * weighted sum (length + alpha * climbing, swept over alpha) finds only the
 * frontier's convex hull and jumps straight across its dents, which on
 * some trips is most of the interesting routes; the frontier is found by a
 * bi-objective search instead (Graph.pareto in engine.js). Along it,
 * distance only ever grows and climbing only ever falls, so the slider does
 * exactly what its ends say. (On a bike with calm streets preferred, the
 * distance that only grows is the comfort-weighted one the search ran on;
 * the kilometres shown can fall.) The shortest route appears at once and the
 * rest fills in over the next second or so; dragging is then instant, and
 * the map crossfades between neighbouring members.
 */
"use strict";

(function () {
  const MI = 1609.344, FT = 3.28084;
  const $ = (id) => document.getElementById(id);
  const DATA = window.DATA;
  const SITE_NAME = (DATA.site && DATA.site.name) || "Flatten HK";

  /* lambda sweep for the slider; the min-climb objective is appended as
   * the last stop so the right-hand end is literally "fewest feet climbed" */
  /* The flat end of the frontier is where a metre of climb is worth
   * ALPHA_MAX metres of walking (the analysis's minimum-climb weight is
   * 120). Past about 200 the router starts walking miles to save a few feet
   * (7.5 miles instead of 4.6 to save 55 ft, on the default trip), which
   * nobody would call a route, so the frontier is cut there. */
  const ALPHA_MAX = 200;
  /* frontier points closer than this in climbing are merged */
  const EPS_GAIN_CM = 50;
  /* the same tolerance inside the search, at intermediate nodes */
  const EPS_NODE_CM = 10;
  /* at most this many routes on the slider, spread evenly along the frontier */
  const MAX_ROUTES = 30;
  /* loop mode: the slider is the loop's length, in miles */
  const LOOP_MIN_MI = 1, LOOP_MAX_MI = 15, LOOP_STEP_MI = 0.5, LOOP_DEFAULT_MI = 5000 / 1609.344;   // a 5 km loop
  /* while loops are tried, the map and profile show a new one this often */
  // while loops are being tried the profile morphs from one candidate to
  // the next; each morph takes this long, and text changes crossfade
  const SCAN_MORPH_MS = 420, FADE_MS = 160, GROW_MS = 200;
  const PROF_PAD = 6;                        // the profile's side padding, px
  /* an end may be moved this far (m) to put both on one landmass, and only
   * off a fragment of the network with fewer nodes than this */
  const RESNAP_M = 400, FRAGMENT_NODES = 150;
  /* a point with no street the mode can use within this many metres is not
   * snapped at all, and a pin the page moves farther than MOVED_NOTE_M from
   * what was asked for is said to have moved */
  const SNAP_MAX_M = 400, MOVED_NOTE_M = 50;
  /* a dropped pin is named after a junction no farther off than this (m) */
  const NEAR_LABEL_M = 500;

  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
  const lerp = (a, b, t) => a + (b - a) * t;
  const ease = (t) => 1 - Math.pow(1 - t, 3);
  const easeInOut = (t) => t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;

  function hexToRgb(h) {
    h = h.replace("#", "");
    if (h.length === 3) h = h.split("").map((c) => c + c).join("");
    const n = parseInt(h, 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }
  /* three-stop ramp, shortest -> middle -> flattest, matching the slider */
  function ramp(t) {
    const stops = [hexToRgb(css("--short")), hexToRgb(css("--mid")), hexToRgb(css("--route"))];
    const x = clamp(t, 0, 1) * 2, i = Math.min(1, Math.floor(x)), k = x - i;
    return "rgb(" + stops[i].map((v, c) => Math.round(lerp(v, stops[i + 1][c], k))).join(",") + ")";
  }

  // kilometres and metres, or miles and feet: a preference kept on the device
  const KM = 1000;
  let units = "km";
  try { if (localStorage.getItem("flattenhk.units") === "mi") units = "mi"; } catch (e) { /* private mode */ }
  const distNum = (m) => { const v = units === "km" ? m / KM : m / MI; return v < 10 ? v.toFixed(1) : String(Math.round(v)); };
  const distUnit = () => units === "km" ? "km" : "mi";
  const climbNum = (m) => Math.round(units === "km" ? m : m * FT).toLocaleString();
  const climbUnit = () => units === "km" ? "m" : "ft";
  const fmtMi = (m) => distNum(m) + "<small>" + distUnit() + "</small>";
  const fmtFt = (m) => climbNum(m) + "<small>" + climbUnit() + "</small>";
  const distText = (m) => distNum(m) + " " + distUnit();
  const climbText = (m) => climbNum(m) + " " + climbUnit();
  // a short distance in a sentence, to the nearest 10 m or 50 ft
  const stepText = (m) => units === "km" ? (Math.round(m / 10) * 10) + " m"
    : (Math.round(m * FT / 50) * 50).toLocaleString() + " ft";
  // the loop slider runs in miles underneath; in kilometres it shows and
  // steps in kilometres (2 to 24, by halves). fromMi is where the thumb
  // rests; the length itself is kept exactly (a link made on the other
  // unit's grid keeps its loop) and exact() is what the labels print.
  const loopSlider = () => units === "km"
    ? { min: 2, max: 24, step: 0.5, unit: "km", toMi: (v) => v * KM / MI, exact: (mi) => mi * MI / KM, fromMi: (mi) => Math.round(mi * MI / KM * 2) / 2 }
    : { min: LOOP_MIN_MI, max: LOOP_MAX_MI, step: LOOP_STEP_MI, unit: "mi", toMi: (v) => v, exact: (mi) => mi, fromMi: (mi) => mi };
  const fmtPct = (g) => (g * 100).toFixed(g * 100 < 10 ? 1 : 0) + "<small>%</small>";

  /* -------------------------------------------------------- text matching */
  /* Street-type words collapse to their abbreviations on both the index and
   * the query, so "Nathan Rd", "Nathan Road" and "nathan" all match. Chinese
   * passes through untouched: it has no word breaks to normalise. */
  const ABBREV = { street: "st", avenue: "ave", boulevard: "blvd", drive: "dr", road: "rd",
    court: "ct", place: "pl", lane: "ln", terrace: "ter", highway: "hwy", path: "path",
    circuit: "cir", square: "sq", crescent: "cres", garden: "gdn", gardens: "gdns",
    building: "bldg", centre: "ctr", center: "ctr", station: "stn", estate: "est", mansion: "mans",
    north: "n", south: "s", east: "e", west: "w", saint: "st", mount: "mt" };
  const norm = (s) => s.toLowerCase()
    .replace(/[’']/g, "")
    .replace(/[^\p{L}\p{N}\s]/gu, " ")
    .replace(/\s+/g, " ").trim()
    .split(" ").map((w) => ABBREV[w] || w).join(" ");
  const CJK = /[⺀-鿿豈-﫿]/;
  const hasCJK = (s) => CJK.test(s);

  /* 0 exact, 1 starts with, 2 every token is a word prefix, 3 substring, -1 none.
   * Chinese is matched as a substring from two characters up ("旺角" finds
   * "旺角站" and "朗豪坊 旺角"), since a name there is one unbroken word. */
  function matchScore(nn, q, toks) {
    if (!nn) return -1;
    if (nn === q) return 0;
    if (nn.startsWith(q)) return 1;
    let all = true;
    for (const t of toks) { if (!(" " + nn).includes(" " + t)) { all = false; break; } }
    if (all) return 2;
    if (q.length >= (hasCJK(q) ? 2 : 3) && nn.includes(q)) return 3;
    return -1;
  }
  /* the better of a record's two names (English, Chinese) against a query */
  function matchEither(a, b, q, toks) {
    const x = matchScore(a, q, toks), y = b ? matchScore(b, q, toks) : -1;
    return x < 0 ? y : (y < 0 ? x : Math.min(x, y));
  }

  /* -------------------------------------------------------- search index */
  class Index {
    constructor(graph, geom, bundle) {
      this.graph = graph; this.geom = geom;
      // every street name is English where the street has one; its Chinese
      // name, where it has both, rides alongside for search and display
      this.zh = DATA.meta.names_zh || [];
      this.buildIntersections();
      // The place list is tens of thousands of names in two languages, and
      // normalising them all takes a noticeable moment on a phone. It is
      // put off until the first route is on the map (or the first search,
      // whichever comes first), so the page opens no later for carrying it.
      this.places = [];
      this._placesText = DATA.manifest.strings.places ? bundle.text("places") : null;
      if (this._placesText) setTimeout(() => this.buildPlaces(), 1500);
      this.addr = null;
      if (DATA.addr && DATA.manifest.strings.addr_streets) {
        const streets = JSON.parse(bundle.text("addr_streets"));
        const st = bundle.array("addr_street");
        const start = new Int32Array(streets.length + 1);
        for (let i = 0; i < st.length; i++) start[st[i] + 1]++;
        for (let i = 0; i < streets.length; i++) start[i + 1] += start[i];
        const zh = DATA.manifest.strings.addr_streets_zh ? JSON.parse(bundle.text("addr_streets_zh")) : [];
        this.addr = {
          streets, nn: streets.map(norm), zh, nz: streets.map((s, i) => (zh[i] ? norm(zh[i]) : "")), start,
          number: bundle.array("addr_number"),
          lon: bundle.array("addr_lon"), lat: bundle.array("addr_lat"),
          origin: DATA.addr.origin, step: DATA.addr.step,
        };
      }
    }

    buildPlaces() {
      if (!this._placesText) return;
      const p = JSON.parse(this._placesText);
      this._placesText = null;
      const alt = p.alt || [], gr = p.group_rank || [];
      const places = new Array(p.names.length);
      for (let i = 0; i < p.names.length; i++) {
        const a = alt[i] || "";
        places[i] = { name: p.names[i], alt: a, nn: norm(p.names[i]), na: a ? norm(a) : "",
          kind: p.groups[p.group[i]], gr: gr[p.group[i]], lon: p.lon[i], lat: p.lat[i] };
      }
      this.places = places;
    }

    /* the Chinese name of a street, by its index in the name table */
    zhOf(nameIdx) { return this.zh[nameIdx - 1] || ""; }

    /* "Nathan Road & Jordan Road": a node with two or more distinct street
     * names. Where several nodes carry the same names (the two carriageways,
     * and the kerb corners where two named pavements cross), the suggestion
     * stands on one both modes can use, if there is one; the pick is snapped
     * for the mode like any other point (see pick in buildUI). */
    buildIntersections() {
      const g = this.graph, geom = this.geom;
      const per = new Array(g.n);
      const add = (node, nameIdx) => {
        if (!nameIdx) return;
        let s = per[node];
        if (!s) { s = per[node] = []; }
        if (s.indexOf(nameIdx) < 0) s.push(nameIdx);
      };
      for (let u = 0; u < g.n; u++) {
        for (let a = g.indptr[u]; a < g.indptr[u + 1]; a++) {
          const ni = geom.name[g.arcEdge[a]];
          add(u, ni); add(g.head[a], ni);
        }
      }
      this.nodeNames = per;
      const seen = new Map();
      const items = [];
      const both = (u) => g.nodeFlags[u] === 3;
      for (let u = 0; u < g.n; u++) {
        const s = per[u];
        if (!s || s.length < 2) continue;
        const idx = s.slice().sort((x, y) => (geom.names[x - 1] < geom.names[y - 1] ? -1 : 1));
        const names = idx.map((i) => geom.names[i - 1]);
        const key = names.join("|");
        if (seen.has(key)) {
          const it = items[seen.get(key)];
          if (both(u) && !both(it.node)) { it.node = u; it.lon = g.nodeLon(u); it.lat = g.nodeLat(u); }
          continue;
        }
        seen.set(key, items.length);
        const zh = idx.map((i) => this.zhOf(i) || geom.names[i - 1]);
        items.push({ name: names.slice(0, 3).join(" & "), alt: zh.slice(0, 3).join(" & "),
          // each street is matched by either of its names
          parts: idx.map((i) => [norm(geom.names[i - 1]), norm(this.zhOf(i))]), node: u,
          lon: g.nodeLon(u), lat: g.nodeLat(u), kind: "junction" });
      }
      this.intersections = items;
    }

    /* A label for a dropped pin: "near" its own corner, or the nearest
     * named corner; failing that its street, the nearest named street, its
     * landmass, and last its coordinates. A corner or street counts only on
     * the pin's own landmass (for the mode) and within NEAR_LABEL_M. With
     * neither limit a pin on Po Toi, where no two named paths meet, was
     * "near Cape D'Aguilar Road & Shek O Road", a junction 6.7 km away on
     * Hong Kong Island, under a status line saying nothing joins the two. */
    describe(node, grid, mode) {
      const g = this.graph, names = this.nodeNames, comp = g.comp(node, mode);
      const street = (i) => this.geom.names[i - 1];
      const corner = (s) => "near " + s.slice(0, 2).map(street).join(" & ");
      const s = names[node];
      if (s && s.length >= 2) return corner(s);
      const nearest = (min) => grid ? grid.nearest(g.nodeLon(node), g.nodeLat(node),
        (i) => names[i] && names[i].length >= min && g.comp(i, mode) === comp, NEAR_LABEL_M) : -1;
      let near = nearest(2);
      if (near >= 0) return corner(names[near]);
      if (s && s.length === 1) return "near " + street(s[0]);
      near = nearest(1);
      if (near >= 0) return "near " + street(names[near][0]);
      const land = g.compName(node, mode);
      if (land) return "on " + land;
      return coordText(g.nodeLon(node), g.nodeLat(node));
    }

    search(raw, limit = 8) {
      const q = norm(raw);
      if (!q) return [];
      this.buildPlaces();
      const toks = q.split(" ");
      const qq = q.replace(/ /g, "");
      const cjk = hasCJK(q);
      const out = [];

      // "123 Nathan Road", "123號彌敦道", "彌敦道123號" or "nathan road 123" -- a
      // street address. A name that only ends in digits ("K11", "Sky100") is
      // not one: with no space, no 號 and no Chinese before the number, it is
      // left to the places below
      const am = (() => {
        let m = /^(\d+)\s*號?\s+(\D.*)$/u.exec(q) || /^(\d+)\s*號\s*(\D.*)$/u.exec(q);
        if (!m) { const c = /^(\d+)(\D.*)$/u.exec(q); if (c && hasCJK(c[2])) m = c; }
        if (m) return m;
        m = /^(\D+?)(\s*)(\d+)\s*(號?)$/u.exec(q);
        return m && (m[2] || m[4] || hasCJK(m[1])) ? [m[0], m[3], m[1].trim()] : null;
      })();
      if (am && this.addr) {
        const want = +am[1], sq = am[2], stoks = sq.split(" ");
        const hits = [];
        for (let i = 0; i < this.addr.streets.length; i++) {
          const sc = matchEither(this.addr.nn[i], this.addr.nz[i], sq, stoks);
          if (sc >= 0 && sc <= 2) hits.push([sc, i]);
        }
        hits.sort((a, b) => a[0] - b[0] || this.addr.streets[a[1]].length - this.addr.streets[b[1]].length);
        for (const [sc, si] of hits.slice(0, 4)) {
          const a = this.addr, lo = a.start[si], hi = a.start[si + 1];
          // numbers are sorted within a street: binary search for the nearest
          let l = lo, h = hi - 1;
          while (l < h) { const m = (l + h) >> 1; if (a.number[m] < want) l = m + 1; else h = m; }
          let best = l;
          if (l > lo && Math.abs(a.number[l - 1] - want) < Math.abs(a.number[l] - want)) best = l - 1;
          const num = a.number[best];
          const exact = num === want;
          // a near miss ranks under any place the same words name outright
          out.push({ score: exact ? -1 : sc + 2, name: num + " " + a.streets[si],
            alt: a.zh[si] ? a.zh[si] + num + "號" : "",
            kind: exact ? "address" : "nearest address", rank: 0,
            lon: a.origin[0] + a.lon[best] * a.step, lat: a.origin[1] + a.lat[best] * a.step });
        }
      }

      // "nathan & jordan", "彌敦道與佐敦道" -- a junction
      const parts = raw.toLowerCase().split(/\s+(?:and|at)\s+|\s*[&\/@+＆、與]\s*/).map(norm).filter(Boolean);
      const hit = (pair, p) => pair.some((n) => n && (n.startsWith(p) || (" " + n).includes(" " + p)
        || (hasCJK(p) && p.length >= 2 && n.includes(p))));
      if (parts.length === 2) {
        for (const it of this.intersections) {
          let ok = 0;
          for (const p of parts) { if (it.parts.some((pair) => hit(pair, p))) ok++; }
          if (ok === 2) out.push({ score: 1, rank: 1, ...it });
        }
      } else if (parts.length === 1) {
        // a street named in full lists its corners among the places that
        // begin with its name; a prefix of one ranks under them
        for (const it of this.intersections) {
          let sc = 0;
          for (const pair of it.parts) for (const n of pair) {
            if (n && n === q) sc = 1; else if (n && !sc && n.startsWith(q)) sc = 2;
          }
          if (sc) out.push({ score: sc, rank: sc === 1 ? 1 : 3, ...it });
        }
      }

      // places
      // areas and stations first, then mapped features and landmarks, then
      // everyday places: the index says how each kind ranks (group_rank)
      const KIND_RANK = { landmark: 1, transit: 1, civic: 1, shop: 2, food: 2, lodging: 2 };
      for (const p of this.places) {
        let sc = matchEither(p.nn, p.na, q, toks);
        if (sc < 0 && qq.length >= 4 && p.nn.replace(/ /g, "").startsWith(qq)) sc = 2;
        if (sc >= 0) out.push({ score: sc, rank: p.gr === undefined ? 1 + (KIND_RANK[p.kind] || 0) : p.gr, ...p });
      }

      out.sort((a, b) => a.score - b.score || a.rank - b.rank || a.name.length - b.name.length);
      // one junction per pair of streets is already guaranteed; dedupe places by name
      const seen = new Set(), res = [];
      for (const r of out) {
        const k = r.kind + "|" + r.name + "|" + (r.alt || "");
        if (seen.has(k)) continue;
        seen.add(k);
        // one place under two spellings ("Central Pier 3", "Central Pier
        // No. 3") shares its Chinese name and its spot: listed once
        if (r.alt && res.some((x) => x.alt === r.alt
          && Math.abs(x.lat - r.lat) < 0.003 && Math.abs(x.lon - r.lon) < 0.003)) continue;
        // typed in Chinese, the Chinese name leads; the other follows it
        const lead = cjk && r.alt ? r.alt : r.name, other = cjk && r.alt ? r.name : (r.alt || "");
        res.push(Object.assign({}, r, { name: lead, sub: other !== lead ? other : "" }));
        if (res.length >= limit) break;
      }
      return res;
    }
  }

  /* --------------------------------------------- faint street canvas layer */
  const StreetLayer = L.Layer.extend({
    initialize(geom) { this.geom = geom; },
    onAdd(map) {
      this._map = map;
      this._canvas = L.DomUtil.create("canvas", "leaflet-zoom-animated");
      map.getPanes().overlayPane.appendChild(this._canvas);
      map.on("moveend zoomend resize", this._redraw, this);
      map.on("zoomanim", this._animate, this);
      this._redraw();
    },
    onRemove(map) {
      L.DomUtil.remove(this._canvas);
      map.off("moveend zoomend resize", this._redraw, this);
      map.off("zoomanim", this._animate, this);
    },
    _animate(e) {
      const scale = this._map.getZoomScale(e.zoom);
      const offset = this._map._latLngToNewLayerPoint(
        this._map.getBounds().getNorthWest(), e.zoom, e.center);
      L.DomUtil.setTransform(this._canvas, offset, scale);
    },
    _redraw() {
      const map = this._map; if (!map) return;
      const size = map.getSize(), dpr = window.devicePixelRatio || 1;
      if (this._canvas.width !== size.x * dpr || this._canvas.height !== size.y * dpr) {
        this._canvas.width = size.x * dpr; this._canvas.height = size.y * dpr;
        this._canvas.style.width = size.x + "px"; this._canvas.style.height = size.y + "px";
      }
      const nw = map.getBounds().getNorthWest();
      L.DomUtil.setTransform(this._canvas, map.latLngToLayerPoint(nw), 1);
      const ctx = this._canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, size.x, size.y);
      const z = map.getZoom();
      if (z < 10.5) return;
      const b = map.getBounds().pad(0.05);
      const west = b.getWest(), east = b.getEast(), south = b.getSouth(), north = b.getNorth();
      // Zoomed out, the whole territory is in view: a quarter of a million
      // segments, most of them footpaths a few metres long. Only streets
      // (and long paths) are drawn then, and the short ones come in as the
      // view closes.
      const minLen = z >= 15 ? 0 : z >= 14 ? 10 : z >= 13 ? 20 : z >= 12 ? 40 : 80;
      const minPath = z >= 14.5 ? 0 : z >= 13 ? 60 : z >= 12 ? 250 : 800;
      const g = this.geom, DM = g.DM;
      if (!this._minor) {
        const MINOR = new Set(["footway", "steps", "path", "service", "track", "bridleway", "unknown"]);
        this._minor = g.classes.map((c) => MINOR.has(c));
      }
      const minor = this._minor;
      // pixel = mercator fraction * world size, less the view's pixel origin
      const { mx, my, x0, y0 } = g.mercator();
      const S = 256 * Math.pow(2, z), o = map.project(nw, z);
      const bx = x0 * S - o.x, by = y0 * S - o.y;
      ctx.strokeStyle = css("--street") || "#c9ccc9";
      ctx.lineWidth = z >= 16 ? 1.6 : z >= 14 ? 1.1 : 0.8;
      ctx.lineCap = "round";
      ctx.beginPath();
      for (let i = 0; i < g.nEdges; i++) {
        const q = i * 4;
        if (g.bbox[q + 2] < west || g.bbox[q] > east || g.bbox[q + 3] < south || g.bbox[q + 1] > north) continue;
        const len = g.len[i] / DM;
        if (len < (minor[g.cls[i]] ? minPath : minLen)) continue;
        const s = g.starts[i] >> 1, e = g.starts[i + 1] >> 1;
        ctx.moveTo(mx[s] * S + bx, my[s] * S + by);
        for (let k = s + 1; k < e; k++) ctx.lineTo(mx[k] * S + bx, my[k] * S + by);
      }
      ctx.stroke();
    },
  });

  /* -------------------------------------------------------------- the app */
  const App = {
    state: { mode: "walk", from: null, to: null, t: 1, focus: "from", calm: true,
      loop: false, loopMi: LOOP_DEFAULT_MI, loopIdx: 0, savedTo: null, outBack: false },
    family: null, shown: null, fading: null,

    async start() {
      $("status").textContent = "Loading the street network…";
      const buf = await loadBundle(DATA, (got, total) => {
        $("status").textContent = "Loading the street network… "
          + (total ? Math.round(100 * got / total) + "%" : (got / 1e6).toFixed(1) + " MB");
      });
      const bundle = new Bundle(buf, DATA.manifest);
      this.graph = new Graph(bundle, DATA.meta);
      this.geom = new Geometry(bundle, DATA.meta);
      DATA.bundle = null;
      const g = this.graph;
      const lonF = new Float32Array(g.n), latF = new Float32Array(g.n);
      for (let i = 0; i < g.n; i++) { lonF[i] = g.nodeLon(i); latF[i] = g.nodeLat(i); }
      this.nodeGrid = new Grid(lonF, latF, 0.004);
      this.index = new Index(g, this.geom, bundle);

      this.buildMap();
      this.buildUI();
      $("status").textContent = "";
      if (!this.readHash() && DATA.default && DATA.default.length === 2) {
        this.setPoint("from", this.placeToPoint(DATA.default[0]), false);
        this.setPoint("to", this.placeToPoint(DATA.default[1]), false);
      }
      this.recompute(true);
    },

    /* ---------------------------------------------------------------- map */
    buildMap() {
      const map = L.map("map", {
        zoomControl: false, attributionControl: true, preferCanvas: true,
        center: [22.302, 114.172], zoom: 12, minZoom: 10, maxZoom: 18, zoomSnap: 0.25, zoomAnimationThreshold: 8,
      });
      map.attributionControl.setPrefix("");
      // the Government's terms ask for the portals to be named as well as
      // the Government: the terrain comes from the CSDI Portal, the addresses
      // and several of the place layers from DATA.GOV.HK
      map.attributionControl.addAttribution(
        "Streets © <a href='https://www.openstreetmap.org/copyright'>OpenStreetMap</a> contributors via <a href='https://overturemaps.org'>Overture</a> · Elevation and places © HKSAR Government (<a href='https://portal.csdi.gov.hk/'>CSDI Portal</a>, <a href='https://data.gov.hk/'>DATA.GOV.HK</a>)");
      L.control.zoom({ position: "bottomright" }).addTo(map);
      this.map = map;

      if (DATA.hillshade) {
        const hs = L.imageOverlay(DATA.hillshade.url || DATA.hillshade.data_uri, DATA.hillshade.bounds,
          { opacity: 1, interactive: false, className: "hillshade" }).addTo(map);
        hs.on("error", () => map.getContainer().classList.add("no-shade"));
      } else {
        map.getContainer().classList.add("no-shade");
      }
      this.streets = new StreetLayer(this.geom).addTo(map);

      // sparse neighborhood names, hidden when zoomed far out or far in
      const labels = L.layerGroup().addTo(map);
      for (const l of DATA.labels || []) {
        L.marker([l.lat, l.lon], { interactive: false, keyboard: false,
          icon: L.divIcon({ className: "nblabel", html: l.n, iconSize: null }) }).addTo(labels);
      }
      const zoomClass = () => {
        const z = map.getZoom();
        map.getContainer().classList.toggle("z-low", z < 12);
        map.getContainer().classList.toggle("z-high", z >= 15.5);
      };
      map.on("zoomend", zoomClass); zoomClass();
      // street names are placed from screen positions, so they come off
      // while the map glides to a new zoom and are placed afresh after
      map.on("zoomstart", () => this.labelLayer.clearLayers());
      map.on("zoomend", () => { if (this._labelled) this.labelRoute(this._labelled); });

      this.familyLayer = L.layerGroup().addTo(map);
      this.routeLayer = L.layerGroup().addTo(map);
      this.markers = L.layerGroup().addTo(map);
      this.labelLayer = L.layerGroup().addTo(map);
      this.hoverLayer = L.layerGroup().addTo(map);
      // the route on show and the hover dot are SVG, not canvas: they fade
      // and move during zoom glides, and Safari has been seen to leave the
      // canvas blank after one; the faint family lines stay on the canvas
      this.svg = L.svg({ padding: 0.5 });

      // a pointer over the profile (or a finger dragged across it) marks
      // that point of the route on both the graph and the map
      const cv = $("prof");
      const at = (e) => {
        const r = cv.getBoundingClientRect();
        this.setHover(clamp((e.clientX - r.left - PROF_PAD) / (r.width - 2 * PROF_PAD), 0, 1));
      };
      cv.addEventListener("pointermove", at);
      cv.addEventListener("pointerdown", (e) => { at(e); try { cv.setPointerCapture(e.pointerId); } catch (err) { /* fine */ } });
      cv.addEventListener("pointerleave", () => this.setHover(null));
      for (const ev of ["pointerup", "pointercancel"]) cv.addEventListener(ev, (e) => { if (e.pointerType !== "mouse") this.setHover(null); });

      map.on("click", (e) => {
        const which = this.state.loop ? "from"
          : (!this.state.from ? "from" : (!this.state.to ? "to" : this.state.focus));
        this.setPoint(which, this.pointAt(e.latlng.lng, e.latlng.lat), true);
        this.recompute("auto");
      });
    },

    /* ---------------------------------------------------------- endpoints */
    /* the nearest corner the mode can use within SNAP_MAX_M (-1 if there
     * is none); with ``comp``, the nearest one on that landmass */
    nearestNode(lon, lat, comp) {
      const g = this.graph, mode = this.state.mode, bit = g.modeBit(mode);
      return this.nodeGrid.nearest(lon, lat, comp
        ? (i) => g.comp(i, mode) === comp
        : (i) => (g.nodeFlags[i] & bit) !== 0, SNAP_MAX_M);
    },
    /* A point is a routable street corner: whatever was clicked or searched
     * snaps to the nearest one, so the pin sits where the route actually
     * starts rather than in the harbour or the middle of a park. What was
     * asked for is kept (qlon, qlat) so the point can be snapped again, for
     * another mode or onto another landmass, and ``shift`` is how far the
     * pin is from it (m).
     *
     * Only so far, though. Upstream snaps any distance, and here a click in
     * Shenzhen became a pin 3.1 km away and a point in Macau one on Lantau,
     * 30 km off, with nothing to say so. With no street the mode can use
     * within SNAP_MAX_M the point stays where it was asked for with
     * ``node`` -1, and recompute() says why there is nothing to route. */
    pointAt(lon, lat, label, comp) {
      const node = this.nearestNode(lon, lat, comp);
      if (node < 0) {
        return { lon, lat, node: -1, qlon: lon, qlat: lat, shift: 0,
          label: label || coordText(lon, lat), named: !!label };
      }
      const g = this.graph, kx = 111320 * Math.cos(lat * Math.PI / 180), ky = 110540;
      return { lon: g.nodeLon(node), lat: g.nodeLat(node), node, qlon: lon, qlat: lat,
        shift: Math.hypot((g.nodeLon(node) - lon) * kx, (g.nodeLat(node) - lat) * ky),
        label: label || this.index.describe(node, this.nodeGrid, this.state.mode), named: !!label };
    },
    /* the same end snapped afresh from what was asked for, under the name
     * it was given: after a change of mode, or to undo a move */
    again(p) {
      return this.pointAt(p.qlon, p.qlat, p.moved ? p.asked : (p.named ? p.label : undefined));
    },
    /* Two ends on different landmasses cannot be joined. Sometimes one of
     * them only looks that way: a pier, a podium or a stray footpath is its
     * own little island in the data, with the real street a few steps off.
     * So an end on such a fragment (fewer than FRAGMENT_NODES nodes) is
     * moved onto the other end's landmass, if a corner of it lies within
     * RESNAP_M of what was asked for; otherwise there is no route to find.
     *
     * A real place is never left for another. Size alone used not to be
     * asked, and by bike, where a bridge closed to bicycles makes Ap Lei
     * Chau a network of its own (some 225 nodes, 229 m from Hong Kong
     * Island's), "Aberdeen Promenade" was pinned 309 m away across the
     * channel and the ride to it never left Ap Lei Chau. A fragment by its
     * size may also be a whole island (Sharp Island is 44 nodes on foot, Ma
     * Wan 122 by bike), so the pin stays on the walking landmass it was on
     * as well, unless that is a fragment too and one the build has no name
     * for.
     *
     * The pin that moves takes the name of the corner it lands on, not the
     * place that was asked for, and movedNote() says how far it went. */
    sameLandmass() {
      const g = this.graph, mode = this.state.mode;
      const { from, to } = this.state;
      const cf = g.comp(from.node, mode), ct = g.comp(to.node, mode);
      if (cf === ct) return true;
      const fragment = (c, m) => g.compNodes(c, m) < FRAGMENT_NODES;
      const moved = (p, comp) => {
        if (!fragment(g.comp(p.node, mode), mode)) return null;
        const q = this.pointAt(p.qlon, p.qlat, undefined, comp);
        if (q.node < 0 || q.shift > RESNAP_M) return null;
        const wa = g.comp(p.node, "walk"), wb = g.comp(q.node, "walk");
        if (wa && wb && wa !== wb && (g.compName(p.node, "walk") || !fragment(wa, "walk"))) return null;
        q.moved = true; q.asked = p.named ? p.label : undefined;
        return q;
      };
      const a = moved(to, cf), b = moved(from, ct);
      const pickTo = a && (!b || a.shift <= b.shift);
      const best = pickTo ? a : b;
      if (!best) return false;
      this.setPoint(pickTo ? "to" : "from", best, false);
      return true;
    },
    /* A pin moved for an earlier trip goes back to what was asked for, so
     * that where the ends are put depends on the trip alone and not on the
     * one before it. */
    unmove() {
      for (const w of ["from", "to"]) {
        const p = this.state[w];
        if (p && p.moved) this.setPoint(w, this.again(p), false);
      }
    },
    /* a pin the page moved is never moved silently: " Start moved 120 m
     * to the nearest street joined to the destination." */
    movedNote() {
      let out = "";
      for (const w of ["from", "to"]) {
        const p = this.state[w];
        if (!p || !p.moved || p.shift <= MOVED_NOTE_M) continue;
        out += " " + (w === "from" ? "Start" : "Destination") + " moved " + stepText(p.shift)
          + " to the nearest street joined to the " + (w === "from" ? "destination" : "start") + ".";
      }
      return out;
    },
    /* why there is no route, in words */
    noRouteText() {
      const g = this.graph, mode = this.state.mode, { from, to } = this.state;
      const bike = mode === "bike", how = bike ? "by bike" : "on foot";
      // One landmass, and still no way through: a landmass takes in the
      // streets a one-way system leads into and not back out of (by bike,
      // Wan Chai North: some 380 nodes that can be ridden to from the rest of
      // Hong Kong Island and, in the data, not left). If the trip can be
      // made the other way round, that is what there is to say.
      if (g.comp(from.node, mode) === g.comp(to.node, mode)) {
        const none = bike ? "No bikeable route" : "No route on foot";
        return g.route(to.node, from.node, mode, this.weights(0))
          ? none + " from the start to the destination, only from the destination to the start:"
            + " the one-way streets in the data do not lead back."
          : none + " between those points.";
      }
      // Two networks need not be two islands, which is what this used to
      // say of any pair it could not name. A cyclist's Hong Kong is the
      // walker's cut into pieces (Ap Lei Chau is some 225 nodes that no
      // bikeable street joins to Hong Kong Island's 11,000), and a piece
      // the build names after the island it stands on shares that
      // island's name.
      const a = g.compName(from.node, mode), b = g.compName(to.node, mode);
      const wf = g.comp(from.node, "walk"), wt = g.comp(to.node, "walk");
      if (a && b && a !== b && !(wf && wf === wt)) {
        return "No way " + how + " between " + a + " and " + b + ": nothing but road tunnels, railways and ferries crosses the water.";
      }
      return "No way " + how + " between those two: the " + (bike ? "streets open to bicycles" : "streets and paths")
        + " do not join up between them.";
    },
    /* and why an end could not be put on the network at all */
    noStreetText(p) {
      return (this.state.mode === "bike" ? "No street a bicycle may use" : "No street or path open to walkers")
        + " within " + stepText(SNAP_MAX_M) + " of " + p.label + ".";
    },
    placeToPoint(p) { return this.pointAt(p.lon, p.lat, p.label || p.name); },

    setPoint(which, pt, typed) {
      this.state[which] = pt;
      const input = $(which);
      input.value = pt ? pt.label : "";
      input.dataset.set = pt ? "1" : "";
      this.hideSuggest(which);
      this.drawMarkers();
      if (typed && !this.state.loop && !this.state[which === "from" ? "to" : "from"]) {
        $(which === "from" ? "to" : "from").focus();
      }
    },

    drawMarkers() {
      this.markers.clearLayers();
      for (const which of ["from", "to"]) {
        const p = this.state[which]; if (!p) continue;
        if (which === "to" && this.state.loop) continue;
        const m = L.marker([p.lat, p.lon], {
          draggable: true, keyboard: false, title: which === "from" ? "Start" : "Destination",
          icon: L.divIcon({ className: "pin-icon " + which, iconSize: [18, 18], iconAnchor: [9, 9] }),
        }).addTo(this.markers);
        m.on("dragend", () => {
          const ll = m.getLatLng();
          this.setPoint(which, this.pointAt(ll.lng, ll.lat), false);
          this.recompute(false);
        });
      }
    },

    /* ------------------------------------------------------------- search */
    buildUI() {
      for (const which of ["from", "to"]) {
        const input = $(which), list = $(which + "_s");
        let sel = -1, items = [];
        const render = () => {
          list.innerHTML = "";
          items.forEach((it, i) => {
            const li = document.createElement("li");
            li.setAttribute("role", "option");
            li.setAttribute("aria-selected", i === sel ? "true" : "false");
            li.innerHTML = "<span class='n'></span><span class='k'></span>";
            li.firstChild.textContent = it.name;
            if (it.sub) {
              const sub = document.createElement("small");
              sub.textContent = it.sub; li.firstChild.append(" ", sub);
            }
            li.lastChild.textContent = it.kind;
            li.addEventListener("mousedown", (e) => { e.preventDefault(); pick(i); });
            list.appendChild(li);
          });
          list.hidden = items.length === 0;
        };
        const pick = (i) => {
          const it = items[i]; if (!it) return;
          // a junction too: its own node may be a kerb corner a bicycle
          // cannot use (1,884 of 7,896 were), so it snaps like any place
          const pt = this.pointAt(it.lon, it.lat, it.name);
          items = []; render();
          this.setPoint(which, pt, true);
          // done typing here: on to the other field if it is still empty,
          // otherwise out, so the card shows the result (on a phone it is
          // folded to the fields while typing)
          const other = which === "from" ? "to" : "from";
          if (!this.state.loop && !this.state[other]) $(other).focus(); else input.blur();
          this.recompute("auto");
        };
        input.addEventListener("focus", () => {
          this.state.focus = which; $("card").classList.add("typing");
          if (input.dataset.set) input.select();
        });
        input.addEventListener("input", () => {
          input.dataset.set = "";
          items = this.index.search(input.value); sel = items.length ? 0 : -1; render();
        });
        input.addEventListener("keydown", (e) => {
          // while Chinese is being composed, Enter and the arrows belong to
          // the input method, not to the suggestion list
          if (e.isComposing || e.keyCode === 229) return;
          if (e.key === "ArrowDown" && items.length) { sel = (sel + 1) % items.length; render(); e.preventDefault(); }
          else if (e.key === "ArrowUp" && items.length) { sel = (sel - 1 + items.length) % items.length; render(); e.preventDefault(); }
          else if (e.key === "Enter") { if (sel >= 0) pick(sel); e.preventDefault(); }
          else if (e.key === "Escape") { items = []; render(); input.blur(); }
        });
        input.addEventListener("blur", () => {
          setTimeout(() => {
            items = []; render();
            if (!input.dataset.set && this.state[which]) input.value = this.state[which].label;
            if (document.activeElement !== $("from") && document.activeElement !== $("to")) $("card").classList.remove("typing");
          }, 120);
        });
        this["hide_" + which] = () => { items = []; render(); };
      }

      $("loopbtn").addEventListener("click", () => this.setLoop(!this.state.loop, true));
      $("swap").addEventListener("click", () => {
        if (this.state.loop) return;
        const a = this.state.from, b = this.state.to;
        this.setPoint("from", b, false); this.setPoint("to", a, false);
        this.recompute("auto");
      });
      for (const btn of $("mode").querySelectorAll("button")) {
        btn.addEventListener("click", () => {
          if (this.state.mode === btn.dataset.v) return;
          this.state.mode = btn.dataset.v;
          for (const b of $("mode").querySelectorAll("button")) b.setAttribute("aria-pressed", b === btn ? "true" : "false");
          $("calmrow").hidden = this.state.mode !== "bike";
          // endpoints may sit on stairs or a footpath that a bike cannot use
          this.resnap();
          this.recompute("auto");
        });
      }
      $("calm").addEventListener("change", () => {
        this.state.calm = $("calm").checked;
        this.recompute("auto");
      });
      $("outback").addEventListener("change", () => {
        this.state.outBack = $("outback").checked;
        this.recompute(true);            // a different kind of run: reframe it
      });
      const sl = $("sl");
      sl.addEventListener("input", () => {
        if (!this.state.loop) { this.state.t = +sl.value; this.show(); this.writeHash(); return; }
        // loop length: the label follows the thumb; the search runs once the
        // thumb settles, so dragging across the range does not queue searches
        this.state.loopMi = loopSlider().toMi(+sl.value);
        $("slpos").textContent = fmtLoop(this.state.loopMi);
        clearTimeout(this._loopTimer);
        this._loopTimer = setTimeout(() => this.recompute("auto"), 350);
      });
      sl.addEventListener("change", () => {
        if (!this.state.loop) return;
        clearTimeout(this._loopTimer);
        this.recompute("auto");
      });
      // an arrow key steps to the next route, not a thousandth of the way
      // along the slider (34 presses from one route of thirty to the next)
      sl.addEventListener("keydown", (e) => {
        const dir = { ArrowRight: 1, ArrowUp: 1, ArrowLeft: -1, ArrowDown: -1 }[e.key];
        if (!dir || this.state.loop || !this.family || this.family.partial) return;
        const n = this.family.unique.length;
        if (n < 2) return;
        e.preventDefault();
        sl.value = clamp(this.stepAt(this.state.t) + dir, 0, n - 1) / (n - 1);
        sl.dispatchEvent(new Event("input"));
      });
      for (const btn of $("units").querySelectorAll("button")) {
        btn.addEventListener("click", () => { if (btn.dataset.u !== units) this.setUnits(btn.dataset.u); });
      }
      this.setUnits(units);
      $("gpx").addEventListener("click", () => this.downloadGpx());
      $("share").addEventListener("click", () => {
        const url = this.shareUrl(), box = $("sharebox"), btn = $("share");
        const done = () => { btn.textContent = "Link copied"; setTimeout(() => { btn.textContent = "Copy link"; }, 1800); };
        const fallback = () => { box.value = url; box.hidden = false; box.focus(); box.select(); };
        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(url).then(done, fallback);
        } else fallback();
      });
      window.addEventListener("resize", () => { if (this.shown) this.drawProfile(this.shown, this.shown, 1); });
    },
    hideSuggest(which) { if (this["hide_" + which]) this["hide_" + which](); },
    /* snap both ends again from what was asked for (after a change of mode:
     * an end may sit on stairs or a footpath that a bike cannot use) */
    resnap() {
      for (const w of ["from", "to"]) {
        if (this.state[w]) this.setPoint(w, this.again(this.state[w]), false);
      }
    },

    /* ------------------------------------------------------------ routing */
    /* length + alpha * gain, on real length (no comfort multipliers), so the
     * family is a true distance-versus-climbing trade-off. On a bike with
     * calm streets on, length is comfort-weighted instead (engine.js
     * arcLenStress): a protected lane counts shorter, a busy arterial
     * longer, and the climbing axis is untouched. */
    calm() { return this.state.mode === "bike" && this.state.calm; },
    lenKey() { return this.calm() ? "stress_m" : "distance_m"; },
    weights(alpha) {
      return { alpha, beta: 0, gamma: 0, penalties: [0, 0, 0, 0, 0], extreme: 0,
        use_class_multiplier: false, stress: this.calm() };
    },

    /* Loop mode on or off. The To field folds away (its place is kept, so
     * turning the loop off brings the destination back) and the slider
     * becomes the loop's length. */
    setLoop(on, recompute) {
      if (this.state.loop === on) return;
      this.state.loop = on;
      $("card").classList.toggle("looping", on);
      const btn = $("loopbtn"), sl = $("sl");
      const label = on ? "Go from A to B" : "Make it a loop";
      btn.setAttribute("aria-pressed", on ? "true" : "false");
      btn.title = label; btn.setAttribute("aria-label", label);
      $("obrow").hidden = !on;
      if (on) {
        this.state.savedTo = this.state.to; this.state.to = null;
        this.loopSliderRange();
        $("slpos").textContent = fmtLoop(this.state.loopMi);
        $("to").tabIndex = -1;
      } else {
        this.state.to = this.state.savedTo; this.state.savedTo = null;
        sl.min = 0; sl.max = 1; sl.step = 0.001; sl.value = this.state.t;
        sl.setAttribute("aria-label", "From shortest to flattest");
        $("end0").textContent = "Shortest"; $("end1").textContent = "Flattest";
        $("slpos").textContent = "";
        $("to").tabIndex = 0;
        if (this.state.to) this.state.to = this.again(this.state.to);
      }
      $("to").value = this.state.to ? this.state.to.label : "";
      $("to").dataset.set = this.state.to ? "1" : "";
      this.drawMarkers();
      if (recompute) this.recompute(true);
    },

    /* the loop slider's range and labels in the current units */
    loopSliderRange() {
      const sl = $("sl"), r = loopSlider();
      sl.min = r.min; sl.max = r.max; sl.step = r.step; sl.value = r.fromMi(this.state.loopMi);
      sl.setAttribute("aria-label", "Loop length in " + (r.unit === "km" ? "kilometres" : "miles"));
      $("end0").textContent = r.min + " " + r.unit; $("end1").textContent = r.max + " " + r.unit;
      $("slpos").textContent = fmtLoop(this.state.loopMi);
    },
    /* switch between miles and kilometres and redraw what shows them */
    setUnits(u) {
      units = u;
      try { localStorage.setItem("flattenhk.units", u); } catch (e) { /* private mode */ }
      for (const b of $("units").querySelectorAll("button")) b.setAttribute("aria-pressed", b.dataset.u === u ? "true" : "false");
      if (this.state.loop) this.loopSliderRange();
      if (this._said) swapText($("status"), this._said());
      if (this.family && this.shown) {
        const m = this.shown;
        this.drawStats(m, m);
        if (!this._profAnim) this.drawProfile(m, m, 1);
      }
    },

    /* the flattest loops of about the chosen length from the start */
    recomputeLoop(fit) {
      this.unmove();
      const { from, mode, loopMi } = this.state;
      this.family = null;
      const gen = ++this._gen;
      $("sl").disabled = false;
      if (!from) {
        this.clearRoute();
        this.say("Where do you start? Type a place or click the map.");
        return;
      }
      if (from.node < 0) { this.refuse(() => this.noStreetText(from)); return; }
      const g = this.graph;
      this.say("Trying loops…");
      $("slpos").textContent = fmtLoop(loopMi);
      const search = g.loops(from.node, mode, { targetM: loopMi * MI, stress: this.calm(), outBack: this.state.outBack });
      this.scanStart(from, loopMi * MI);
      const run = () => {
        if (gen !== this._gen) return;
        // short slices, so the morph between candidates keeps its frame rate
        if (!search.step(12)) {
          if (search.last) this.scanShow(search.last);
          setTimeout(run, 0);
          return;
        }
        // a morph in flight finishes before the result takes over from it
        if (this._scan && this._scan.busy) { this._scan.onIdle = finish; return; }
        finish();
      };
      const finish = () => {
        if (gen !== this._gen) return;
        this.scanEnd();
        if (!search.loops.length) { this.refuse("No loop from here. Try another start."); return; }
        const members = search.loops.map((r) => Object.assign(this.member(r.arcs), { kind: r.kind }));
        members.forEach((m, i) => { m.id = i; });
        this.family = { unique: members, shortest: members[0], partial: false, loop: true,
          targetM: loopMi * MI, medianGain: search.medianGain, tried: search.accepted.length,
          shortfall: search.shortfall };
        this.state.loopIdx = Math.min(this._pendingLoopIdx || 0, members.length - 1);
        this._pendingLoopIdx = 0;
        this.shown = null;
        this.say(search.shortfall
          ? () => "No " + fmtLoop(loopMi) + " fits here; this is the closest."
          : "The flattest of " + search.tried + " loops tried.");
        this.drawFamily(true);
        this.show(true);
        // a loop that spills out of view, or has shrunk to a small part of
        // it (the length slider moved down), is refitted and recentred
        if (fit === true || (fit === "auto" && (!this.inView() || this.viewShare() < 0.35))) this.fit();
        this.writeHash();
        $("share").hidden = false; $("gpx").hidden = false; $("sharebox").hidden = true;
      };
      setTimeout(run, 0);
    },

    /* The scan: while the loop search runs, the map holds still and the
     * elevation profile morphs from one candidate loop to the next, the
     * stats following, so the search reads as the shape of the run being
     * worked out rather than a wait.  What was shown before fades out. */
    scanStart(from, targetM) {
      this.setHover(null); this._loopNext = null;
      const fl = this.familyLayer, rl = this.routeLayer, old = [];
      fl.eachLayer((l) => old.push(l)); rl.eachLayer((l) => old.push(l));
      if (old.length) fadeOut(old, 260, () => old.forEach((l) => { fl.removeLayer(l); rl.removeLayer(l); }));
      this.labelLayer.clearLayers();
      this._line = null; this._casing = null; this._labelled = null; this.shown = null;
      hideSoft($("turns"));
      swapText($("delta"), "About " + fmtLoop(targetM / MI).replace(" loop", "") + ", in every direction from here.");
      // the profile carries on from wherever it is; nothing drawn yet
      // means the first candidate rises out of a flat line
      const cur = !$("prof").hidden && this._profCur ? this._profCur : null;
      this._scan = { cur, target: null, busy: false, onIdle: null };
    },
    /* the latest candidate: morph to it once the current morph is done */
    scanShow(r) {
      const sc = this._scan;
      if (!sc || sc.busy || r === sc.target) return;
      sc.target = r;
      const s = this.graph.summarise(r.arcs);
      const m = { profile: resample(s.profile, 160), stats: s };
      let from = sc.cur || this._profCur;
      if (!from) {
        const z0 = Math.min(...m.profile.z);
        from = { profile: { z: new Float64Array(m.profile.z.length).fill(z0) }, stats: s, range: { min: z0, max: z0 } };
      }
      sc.cur = null;
      // the vertical scale follows each candidate, eased along with the shape
      const to = rangeOf([m]);
      this.figures(from.stats, s, 0);          // before the block is measured, as in drawStats
      showSoft($("result"));
      $("prof").dataset.colour = css("--route");
      sc.busy = true;
      this.animateProfile(from, m, SCAN_MORPH_MS, { range: { from: from.range, to }, stats: true, ease: easeInOut,
        done: () => { sc.busy = false; if (sc.onIdle) { const f = sc.onIdle; sc.onIdle = null; f(); } } });
    },
    /* the result then morphs in from wherever the scan left the profile */
    scanEnd() {
      if (!this._scan) return;
      this._scan = null;
      if (this._profAnim) cancelAnimationFrame(this._profAnim);
      this._handoff = !$("prof").hidden && this._profCur ? this._profCur : null;
    },

    /* There is nothing to draw, and the status line says why. The address
     * bar takes the trip that was refused: it used to keep the one before,
     * so a reload brought back a route the fields no longer described. */
    refuse(why) {
      this.clearRoute();
      this.say(why);
      this.writeHash();
    },
    /* The status line. A sentence that quotes a distance is given as a
     * function, so that it can be said again when the units change: "moved
     * 300 m" under a card switched to miles and feet. */
    say(text) {
      this._said = typeof text === "function" ? text : null;
      swapText($("status"), this._said ? text() : text);
    },

    recompute(fit) {
      if (this.state.loop) return this.recomputeLoop(fit);
      this.scanEnd();
      this.unmove();
      const { from, to } = this.state;
      this.family = null;
      const gen = ++this._gen;
      if (!from || !to) {
        this.clearRoute();
        this.say(!from && !to ? "Type two places, or click the map twice."
          : (!from ? "Where are you starting from?" : "Where to?"));
        return;
      }
      const lost = from.node < 0 ? from : (to.node < 0 ? to : null);
      if (lost) { this.refuse(() => this.noStreetText(lost)); return; }
      if (from.node === to.node) { this.refuse("Those are the same corner."); return; }
      if (!this.sameLandmass()) { this.refuse(this.noRouteText()); return; }
      // sameLandmass may have moved an end onto the other's landmass
      return this.recomputeRoute(fit, gen);
    },

    recomputeRoute(fit, gen) {
      const { from, to, mode } = this.state;
      const g = this.graph;
      if (from.node === to.node) { this.refuse("Those are the same corner."); return; }
      const shortest = g.route(from.node, to.node, mode, this.weights(0));
      if (!shortest) {
        const why = this.noRouteText();
        this.refuse(() => why + this.movedNote());
        return;
      }
      // the shortest and the flattest routes go up at once, so the map can
      // be fitted to the whole family; the frontier fills in between them
      const first = this.member(shortest.arcs);
      const flat = g.route(from.node, to.node, mode, this.weights(ALPHA_MAX)) || shortest;
      const fm = this.member(flat.arcs);
      const flatStats = fm.stats;
      const sameEnds = fm.arcs.length === first.arcs.length && fm.arcs.every((a, i) => a === first.arcs[i]);
      this.family = { unique: sameEnds ? [first] : [first, fm], shortest: first, partial: true };
      this.family.unique.forEach((m, i) => { m.id = i; });
      this.shown = null;
      this.say("Finding every route between shortest and flattest…");
      $("sl").disabled = true;
      this.drawFamily();
      this.show(true);
      if (fit === true || (fit === "auto" && !this.inView())) this.fit();
      this.writeHash();
      $("share").hidden = false; $("gpx").hidden = false; $("sharebox").hidden = true;

      // A trip across the New Territories can climb a kilometre and a half
      // by its shortest route. Half a metre of climbing is then too fine a
      // step to tell routes apart by (the frontier would hold thousands of
      // them and the search would run out of room before reaching its flat
      // end), so the tolerances grow with the trip: about three hundred
      // steps of climbing from the hilliest route to none.
      const gCap = Math.round(first.stats.elev_gain_m * g.CM) + 1;
      const eps = Math.max(EPS_GAIN_CM, Math.round(gCap / 300));
      const search = g.pareto(from.node, to.node, mode, {
        eps, epsNode: Math.max(EPS_NODE_CM, Math.round(eps / 5)), stress: this.calm(),
        dCap: Math.round(flatStats[this.lenKey()] * g.DM) + 1, gCap,
      });
      const run = () => {
        if (gen !== this._gen) return;          // the trip changed underneath us
        if (!search.step(30)) {
          setText($("status"), "Finding every route between shortest and flattest… "
            + search.solutions.length);
          setTimeout(run, 0);
          return;
        }
        this.finishFamily(search, first, fm, fit);
      };
      setTimeout(run, 0);
    },

    member(arcs) {
      const s = this.graph.summarise(arcs);
      return { arcs, stats: s, latlngs: this.graph.geometry(arcs, this.geom),
        profile: resample(s.profile, 160) };
    },

    /* the frontier is in, sorted shortest to flattest: pick the routes the
     * slider will step through */
    finishFamily(search, first, fm, fit) {
      // A long trip over the hills can leave thousands of routes on the
      // frontier. They are thinned on the search's own two numbers first,
      // so geometry and a profile are built for a few dozen, not for all.
      const sols = search.solutions.slice().sort((a, b) => a.weighted - b.weighted);
      const few = thinFrontier(sols.map((r) => ({ r, stats: { distance_m: r.weighted, elev_gain_m: r.gain } })), 3 * MAX_ROUTES);
      let members = few.map((x) => this.member(x.r.arcs));
      if (!members.length) members = [first];
      const key = this.lenKey();
      members.sort((a, b) => a.stats[key] - b.stats[key]);
      // The weighted flattest route is a frontier point by construction.
      // The search's tolerances can leave it out by a few feet, and if the
      // search was cut short it is missing altogether, so it closes the
      // family whenever it beats what the search found.
      const last = members[members.length - 1];
      if (fm.stats.elev_gain_m < last.stats.elev_gain_m - 1e-6) members.push(fm);
      // what the tests and the console look at, without holding on to the
      // search's label arrays
      this._search = { solutions: search.solutions, expanded: search.expanded, labels: search.labels, truncated: search.truncated };
      members = thinFrontier(members, MAX_ROUTES, key);
      members.forEach((m, i) => { m.id = i; });
      this.family = { unique: members, shortest: members[0], partial: false };
      $("sl").disabled = false;
      this.say(() => (members.length === 1
        ? "One route: the shortest is already the flattest."
        : members.length + " distinct routes, from shortest to flattest.") + this.movedNote());
      this.drawFamily();
      this.show(false);
      if (fit && !this.inView()) this.fit();
    },

    clearRoute() {
      this.setHover(null); this._loopNext = null;
      this.familyLayer.clearLayers(); this.routeLayer.clearLayers(); this.labelLayer.clearLayers();
      $("turns").hidden = true;
      this.shown = null; this._handoff = null; this._profCur = null;
      $("result").hidden = true; $("prof").hidden = true; cutText($("delta")); $("slpos").textContent = "";
      $("share").hidden = true; $("gpx").hidden = true; $("sharebox").hidden = true;
      $("sl").disabled = false;
    },

    /* slider position -> index into the family, evenly over its members */
    stepAt(t) {
      const n = this.family.unique.length;
      return clamp(Math.round(t * (n - 1)), 0, n - 1);
    },

    drawFamily(fade) {
      this.familyLayer.clearLayers();
      const lines = this.family.unique.map((u) =>
        L.polyline(u.latlngs, { color: css("--family"), weight: 2, opacity: fade ? 0 : 0.45, interactive: false,
          lineJoin: "round", lineCap: "round" }).addTo(this.familyLayer));
      if (fade) fadeIn(lines.map((l) => [l, 0.45]), 260);
    },

    /* show the family member for the current slider position */
    show(immediate) {
      if (!this.family) return;
      const loop = !!this.family.loop;
      const t = this.state.t, n = this.family.unique.length;
      const step = loop ? clamp(this.state.loopIdx, 0, n - 1) : this.stepAt(t);
      const u = this.family.unique[step];
      const colour = loop ? css("--route") : ramp(t);
      $("slpos").textContent = loop ? fmtLoop(this.state.loopMi)
        : (this.family.partial ? "" : (n === 1 ? "" : (step + 1) + " of " + n));
      const prev = this.shown;
      if (prev === u) { this.tintRoute(colour); return; }
      this.shown = u;
      // after a scan the profile and stats carry on from where it left them
      const hand = this._handoff; this._handoff = null;
      const from = hand || (prev && !immediate ? prev : null);
      this.drawRoute(u, colour, prev && !immediate ? prev : null, !!hand);
      this.drawStats(u, from || u);
      this.animateProfile(from || u, u, 300, hand ? { range: { from: hand.range, to: rangeOf(this.family.unique) } } : {});
      this.placeHoverDot();
    },

    tintRoute(colour) {
      if (this._line) this._line.setStyle({ color: colour });
      $("prof").dataset.colour = colour;
      if (this.shown) this.drawProfile(this.shown, this.shown, 1);
    },

    drawRoute(u, colour, prev, fade) {
      // crossfade: the old line fades out while the new one fades in
      const casing = css("--route-casing");
      if (this._line && prev) {
        const oldCase = this._casing, oldLine = this._line;
        fadeOut([oldCase, oldLine], 260, () => { this.routeLayer.removeLayer(oldCase); this.routeLayer.removeLayer(oldLine); });
      } else {
        this.routeLayer.clearLayers();
      }
      fade = fade || !!prev;
      this._casing = L.polyline(u.latlngs, { color: casing, weight: 10, opacity: fade ? 0 : 0.9, interactive: false,
        lineJoin: "round", lineCap: "round", renderer: this.svg }).addTo(this.routeLayer);
      this._line = L.polyline(u.latlngs, { color: colour, weight: 5, opacity: fade ? 0 : 1, interactive: false,
        lineJoin: "round", lineCap: "round", renderer: this.svg }).addTo(this.routeLayer);
      if (fade) fadeIn([[this._casing, 0.9], [this._line, 1]], 260);
      this._casing.bringToFront(); this._line.bringToFront();
      $("prof").dataset.colour = colour;
      this.labelRoute(u);
    },

    /* The streets a route follows, as runs of consecutive arcs sharing a
     * name: [{name, arcs, length_m, push}]. Unnamed stubs and runs under
     * minRun metres are dropped from the list. On a bike a stretch to be
     * walked (pushed links, engine.js arcPush) is a run of its own and
     * always kept, short and nameless as most are: it is where the rider
     * gets off. Its pieces are one run whatever each is called (a crossing,
     * a pavement named after its road, a nameless path), under the name of
     * the longest of them. */
    runs(u, minRun = 40) {
      const g = this.graph, geom = this.geom, out = [];
      const pushed = this.state.mode === "bike" && g.arcPush ? g.arcPush : null;
      let cur = null;
      for (const a of u.arcs) {
        const n = geom.name[g.arcEdge[a]], name = n ? geom.names[n - 1] : null;
        const L = g.arcLen[a] / g.DM, push = !!(pushed && pushed[a]);
        if (cur && push && cur.push) {
          cur.arcs.push(a); cur.length_m += L;
          if (name) {
            cur.by[name] = (cur.by[name] || 0) + L;
            if (cur.by[name] > (cur.by[cur.name] || 0)) cur.name = name;
          }
        } else if (cur && !push && !cur.push && cur.name === name) { cur.arcs.push(a); cur.length_m += L; }
        else {
          cur = { name, arcs: [a], length_m: L, push };
          if (push) cur.by = name ? { [name]: L } : {};
          out.push(cur);
        }
      }
      // drop unnamed stubs and short runs, then re-merge what that joins
      // up ("18th St, 18th St" either side of a nameless crossing)
      const kept = out.filter((r) => r.push || (r.name && r.length_m >= minRun)), merged = [];
      for (const r of kept) {
        const last = merged[merged.length - 1];
        if (last && last.name === r.name && last.push === r.push) { last.arcs = last.arcs.concat(r.arcs); last.length_m += r.length_m; }
        else merged.push({ name: r.name, arcs: r.arcs.slice(), length_m: r.length_m, push: r.push });
      }
      return merged;
    },
    /* a run as the street list prints it: "Tolo Harbour Cycle Track",
     * "Ting Kok Rd (push)", "footpath (push)" */
    runText(r) {
      return r.push ? (r.name ? shortStreet(r.name) : "footpath") + " (push)" : shortStreet(r.name);
    },

    /* Street names drawn along the highlighted route: one per named run,
     * rotated to the line's bearing, only where the run is long enough on
     * screen to carry its text, never overlapping another label. Redrawn
     * on zoom, since what fits changes. */
    labelRoute(u) {
      this.labelLayer.clearLayers();
      this._labelled = u;
      if (!u) return;
      const g = this.graph, geom = this.geom, map = this.map;
      const runs = this.runs(u, 80).filter((r) => !r.push).sort((a, b) => b.length_m - a.length_m);
      const placed = [];
      for (const r of runs) {
        if (placed.length >= 10) break;
        const text = shortStreet(r.name);
        const pts = [];
        for (const a of r.arcs) for (const ll of g.geometry([a], geom)) {
          const last = pts[pts.length - 1];
          if (!last || last[0] !== ll[0] || last[1] !== ll[1]) pts.push(ll);
        }
        if (pts.length < 2) continue;
        // pixel length along the run at this zoom, and its midpoint
        const px = pts.map((ll) => map.latLngToContainerPoint(ll));
        const seg = [];
        let total = 0;
        for (let i = 1; i < px.length; i++) { const d = px[i].distanceTo(px[i - 1]); seg.push(d); total += d; }
        const need = text.length * 6.6 + 28;
        if (total < need) continue;
        let acc = 0, i = 1;
        while (i < pts.length - 1 && acc + seg[i - 1] < total / 2) { acc += seg[i - 1]; i++; }
        const f = seg[i - 1] ? (total / 2 - acc) / seg[i - 1] : 0;
        const mid = px[i - 1].add(px[i].subtract(px[i - 1]).multiplyBy(f));
        if (placed.some((q) => q.distanceTo(mid) < need * 0.6)) continue;
        let deg = Math.atan2(px[i].y - px[i - 1].y, px[i].x - px[i - 1].x) * 180 / Math.PI;
        if (deg > 90) deg -= 180; else if (deg < -90) deg += 180;
        const m = L.marker(map.containerPointToLatLng(mid), { interactive: false, keyboard: false,
          icon: L.divIcon({ className: "rtlabel", iconSize: null,
            html: `<span style="--rot:${deg.toFixed(1)}deg"></span>` }) }).addTo(this.labelLayer);
        m.getElement().firstChild.textContent = text;
        placed.push(mid);
      }
    },

    /* the three figures, a fraction k of the way from one route's to another's */
    figures(p, s, k) {
      $("v_dist").innerHTML = fmtMi(lerp(p.distance_m, s.distance_m, k));
      $("v_climb").innerHTML = fmtFt(lerp(p.elev_gain_m, s.elev_gain_m, k));
      $("v_grade").innerHTML = fmtPct(lerp(p.steepest, s.steepest, k));
    },

    drawStats(u, prevU) {
      // The figures go in before the block is opened, which eases to the
      // height it measures then: on a page's first route it measured the
      // captions alone (18 px of 40), opened that far and jumped to the
      // rest when the easing let go.
      const s = u.stats, p = prevU.stats;
      this.figures(p, s, 0);
      showSoft($("result"));
      const runs = this.runs(u, 60);
      const box = $("turns");
      box.innerHTML = "";
      const render = (all) => {
        box.innerHTML = "";
        const show = all || runs.length <= 8 ? runs : runs.slice(0, 7);
        show.forEach((r, i) => {
          if (i) { const v = document.createElement("span"); v.className = "via"; v.textContent = "→"; box.appendChild(v); }
          box.appendChild(document.createTextNode(this.runText(r)));
        });
        if (show.length < runs.length) {
          const more = document.createElement("button");
          more.type = "button"; more.className = "link more";
          more.textContent = "+" + (runs.length - show.length) + " more";
          more.addEventListener("click", () => render(true));
          box.appendChild(more);
        }
      };
      render(false);
      if (runs.length) showSoft(box); else hideSoft(box);
      tween(260, (k) => this.figures(p, s, k));
      if (this.family.loop) { this.drawLoopDelta(u); return; }
      const sh = this.family.shortest.stats;
      if (u === this.family.shortest) {
        setText($("delta"), this.family.unique.length > 1
          ? (this.calm() ? "The shortest route on calm streets. Slide right to trade distance for less climbing."
            : "The shortest route. Slide right to trade distance for less climbing.")
          : "Shortest and flattest at once.", true);
      } else {
        const dd = s.distance_m - sh.distance_m, dc = sh.elev_gain_m - s.elev_gain_m;
        const pd = sh.distance_m ? Math.round(100 * Math.abs(dd) / sh.distance_m) : 0;
        const pc = sh.elev_gain_m ? Math.round(100 * dc / sh.elev_gain_m) : 0;
        // with calm streets on, the left end is the shortest by comfort-
        // weighted length, and a route to its right can be the shorter in
        // kilometres: its streets count for more (a busier road, a footway
        // to push along)
        const longer = Math.abs(dd) < 80 ? "about the same distance"
          : dd > 0 ? "<b class='up'>+" + distText(dd) + "</b> (" + pd + "% longer)"
            : "<b class='down'>−" + distText(-dd) + "</b> (" + pd + "% shorter, on less calm streets)";
        const less = dc <= 0 ? "no less climbing"
          : "<b class='down'>−" + climbText(dc) + "</b> of climbing (" + pc + "% less)";
        setText($("delta"), (this.calm() ? "vs. shortest on calm streets: " : "vs. shortest: ") + longer + ", " + less, true);
      }
    },

    /* under the stats: how this loop compares with a typical loop of the
     * same length from the same start, and a way to see the next-best one */
    drawLoopDelta(u) {
      const f = this.family, s = u.stats, box = $("delta");
      const med = f.medianGain, idx = f.unique.indexOf(u), n = f.unique.length;
      let html;
      const ob = u.kind === "outback" ? "Out and back. " : "";
      if (f.shortfall) html = ob + "The longest loop that fits from here.";
      else if (Number.isFinite(med) && med - s.elev_gain_m >= 3 && f.tried >= 5) {
        html = ob + "<b class='down'>−" + climbText(med - s.elev_gain_m)
          + "</b> of climbing vs. a typical " + fmtLoop(this.state.loopMi) + " from here.";
      } else html = ob + "About as flat as loops from here get.";
      swapText(box, (el) => {
        el.innerHTML = html;
        if (n > 1) {
          const b = document.createElement("button");
          b.type = "button"; b.className = "link"; b.id = "nextloop";
          b.textContent = "Another loop (" + (idx + 1) + " of " + n + ")";
          b.addEventListener("click", () => {
            // taps in quick succession each advance one more: the target
            // is counted from the last one asked for, not the one on show
            const cur = this._loopNext !== undefined && this._loopNext !== null ? this._loopNext : this.state.loopIdx;
            const next = (cur + 1) % n, nu = f.unique[next];
            this._loopNext = next;
            const go = () => { if (this.family !== f) return; this._loopNext = null; this.state.loopIdx = next; this.show(false); this.writeHash(); };
            if (this.inView(nu) && this.viewShare(nu) >= 0.35) { go(); return; }
            // the next loop needs a new view: the loop on show fades off,
            // the map glides to the next one, and that fades on (the same
            // order as a new search), rather than moving mid-crossfade
            const old = [this._casing, this._line].filter(Boolean);
            this._casing = null; this._line = null;
            this.labelLayer.clearLayers(); this._labelled = null;
            this.setHover(null);
            if (old.length) fadeOut(old, 200, () => old.forEach((l) => this.routeLayer.removeLayer(l)));
            clearTimeout(this._loopTap);
            this._loopTap = setTimeout(() => { if (this.family !== f || this._loopNext !== next) return; this.fit(nu); go(); }, old.length && !reducedMotion() ? 220 : 0);
          });
          el.append(" ", b);
        }
      });
    },

    /* ---------------------------------------------------------------- GPX */
    /* the route on show as a GPX track with elevation, for Strava, Garmin
     * Connect, Komoot and the rest */
    tripName() {
      const { from, to, loop } = this.state;
      return loop ? fmtLoop(this.state.loopMi) + " from " + from.label : from.label + " to " + (to ? to.label : "");
    },
    gpxName() {
      const which = this.gpxEnd();
      return this.tripName() + (which ? ", " + which : "");
    },
    /* "shortest" or "flattest" when the route on show is an end of its family */
    gpxEnd() {
      const f = this.family;
      if (!f || f.loop || f.unique.length < 2) return "";
      return this.shown === f.shortest ? "shortest" : this.shown === f.unique[f.unique.length - 1] ? "flattest" : "";
    },
    gpx() {
      const u = this.shown, s = u.stats, pts = u.latlngs;
      alongRoute(u, 0);                         // builds the cumulative length
      const cum = u._cum, total = cum[cum.length - 1] || 1, pd = s.profile.d, pz = s.profile.z;
      const span = pd[pd.length - 1] || 1;
      let j = 0;
      const ele = (i) => {
        const want = cum[i] / total * span;
        while (j < pd.length - 2 && pd[j + 1] < want) j++;
        const t = pd[j + 1] > pd[j] ? clamp((want - pd[j]) / (pd[j + 1] - pd[j]), 0, 1) : 0;
        return lerp(pz[j], pz[j + 1], t);
      };
      const esc = (x) => String(x).replace(/[<>&"]/g, (c) => ({ "<": "&lt;", ">": "&gt;", "&": "&amp;", '"': "&quot;" }[c]));
      const name = esc(this.gpxName()), desc = esc(distText(s.distance_m) + ", " + climbText(s.elev_gain_m) + " of climbing. " + SITE_NAME);
      const out = ['<?xml version="1.0" encoding="UTF-8"?>',
        '<gpx version="1.1" creator="' + esc(SITE_NAME) + '" xmlns="http://www.topografix.com/GPX/1/1">',
        "<metadata><name>" + name + "</name><desc>" + desc + "</desc><link href=\"" + esc(this.shareUrl()) + "\"><text>" + esc(SITE_NAME) + "</text></link></metadata>",
        "<trk><name>" + name + "</name><desc>" + desc + "</desc><trkseg>"];
      for (let i = 0; i < pts.length; i++) {
        out.push('<trkpt lat="' + pts[i][0].toFixed(6) + '" lon="' + pts[i][1].toFixed(6) + '"><ele>' + ele(i).toFixed(1) + "</ele></trkpt>");
      }
      out.push("</trkseg></trk></gpx>");
      return out.join("\n");
    },
    downloadGpx() {
      if (!this.shown) return;
      const text = this.gpx();
      // the trip in at most 80 characters, cut at a word, and then which
      // end of the family it is, whole: a plain cut at 60 made
      // "…-the-university-of-hong-kong-flattes" of the default trip
      const which = this.gpxEnd();
      const slug = (fileSlug(this.tripName(), 80 - (which ? which.length + 1 : 0)) + (which ? "-" + which : "")) || "route";
      const blob = new Blob([text], { type: "application/gpx+xml" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob); a.download = "flattenhk-" + slug + ".gpx";
      document.body.appendChild(a); a.click(); a.remove();
      setTimeout(() => URL.revokeObjectURL(a.href), 2000);
    },

    /* ------------------------------------------------------------ profile */
    /* the point a fraction of the way along the route on show, marked on
     * the map and on the graph; null clears it */
    setHover(f) {
      if (f !== null && (!this.shown || this._scan)) f = null;
      this._hover = f;
      if (f === null) { this.hoverLayer.clearLayers(); this._hoverDot = null; }
      else this.placeHoverDot();
      if (!this._profAnim && this.shown) this.drawProfile(this.shown, this.shown, 1);
    },
    placeHoverDot() {
      if (this._hover === null || this._hover === undefined || !this.shown) return;
      const ll = alongRoute(this.shown, this._hover);
      const fill = $("prof").dataset.colour || css("--route");
      if (!this._hoverDot) {
        this._hoverDot = L.circleMarker(ll, { radius: 6, color: "#fff", weight: 2.5, fillColor: fill, fillOpacity: 1,
          interactive: false, renderer: this.svg }).addTo(this.hoverLayer);
      } else { this._hoverDot.setLatLng(ll); this._hoverDot.setStyle({ fillColor: fill }); }
    },

    /* morph the profile from a to b; opts.range eases the vertical scale
     * between two {min, max} ranges, opts.stats carries the numbers along,
     * opts.done runs when the morph completes (not when it is cut short) */
    animateProfile(a, b, ms = 300, opts = {}) {
      const cv = $("prof");
      if (cv.hidden) { cv.hidden = false; cv.classList.add("fading"); void cv.offsetWidth; cv.classList.remove("fading"); }
      if (this._profAnim) cancelAnimationFrame(this._profAnim);
      const t0 = performance.now();
      const frame = (now) => {
        const k = (opts.ease || ease)(clamp((now - t0) / ms, 0, 1));
        this.drawProfile(a, b, k, opts.range);
        if (opts.stats) this.figures(a.stats, b.stats, k);
        if (k < 1) this._profAnim = requestAnimationFrame(frame);
        else { this._profAnim = null; if (opts.done) opts.done(); }
      };
      this._profAnim = requestAnimationFrame(frame);
    },

    drawProfile(a, b, k, range) {
      const cv = $("prof"), dpr = window.devicePixelRatio || 1;
      const W = cv.clientWidth || 360, H = cv.clientHeight || 92;
      if (cv.width !== W * dpr || cv.height !== H * dpr) { cv.width = W * dpr; cv.height = H * dpr; }
      const ctx = cv.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);
      const n = b.profile.z.length;
      const z = new Float64Array(n);
      for (let i = 0; i < n; i++) z[i] = lerp(a.profile.z[i], b.profile.z[i], k);
      const dist = lerp(a.stats.distance_m, b.stats.distance_m, k);
      // a fixed vertical scale across the family keeps the hills comparable;
      // a morph that changes scale (the scan, or coming out of it) eases it
      let zmin, zmax;
      if (range) { zmin = lerp(range.from.min, range.to.min, k); zmax = lerp(range.from.max, range.to.max, k); }
      else ({ min: zmin, max: zmax } = rangeOf(this.family ? this.family.unique : [a, b]));
      // what is on the canvas right now, so another morph can start from it
      this._profCur = { profile: { z }, range: { min: zmin, max: zmax }, stats: {
        distance_m: dist, elev_gain_m: lerp(a.stats.elev_gain_m, b.stats.elev_gain_m, k),
        steepest: lerp(a.stats.steepest, b.stats.steepest, k) } };
      const span = Math.max(zmax - zmin, 15);
      zmin -= span * 0.08; zmax = zmin + span * 1.2;
      const padL = PROF_PAD, padR = PROF_PAD, top = 8, bottom = 18;
      const X = (i) => padL + (W - padL - padR) * i / (n - 1);
      const Y = (v) => top + (H - top - bottom) * (1 - (v - zmin) / (zmax - zmin));
      const colour = cv.dataset.colour || css("--route");
      ctx.beginPath();
      ctx.moveTo(X(0), Y(zmin));
      for (let i = 0; i < n; i++) ctx.lineTo(X(i), Y(z[i]));
      ctx.lineTo(X(n - 1), Y(zmin)); ctx.closePath();
      ctx.fillStyle = colour; ctx.globalAlpha = 0.18; ctx.fill(); ctx.globalAlpha = 1;
      ctx.beginPath();
      for (let i = 0; i < n; i++) { if (i) ctx.lineTo(X(i), Y(z[i])); else ctx.moveTo(X(i), Y(z[i])); }
      ctx.strokeStyle = colour; ctx.lineWidth = 2; ctx.lineJoin = "round"; ctx.stroke();
      // labels: start and end elevation, the high point, the distance scale
      ctx.fillStyle = css("--muted"); ctx.font = "500 10px " + css("--mono");
      ctx.textBaseline = "alphabetic";
      let hi = 0; for (let i = 1; i < n; i++) if (z[i] > z[hi]) hi = i;
      const lab = (v) => climbText(v);
      ctx.textAlign = "left"; ctx.fillText(lab(z[0]), padL, H - 5);
      ctx.textAlign = "right"; ctx.fillText(lab(z[n - 1]), W - padR, H - 5);
      ctx.textAlign = "center"; ctx.fillText(distText(dist), W / 2, H - 5);
      const hov = this._hover !== null && this._hover !== undefined && !this._scan && k >= 1 ? this._hover : null;
      if (hov === null && hi > n * 0.06 && hi < n * 0.94 && z[hi] - Math.min(z[0], z[n - 1]) > 6) {
        ctx.textAlign = X(hi) < 40 ? "left" : X(hi) > W - 40 ? "right" : "center";
        ctx.fillStyle = css("--ink");
        ctx.fillText(lab(z[hi]), X(hi), Math.max(10, Y(z[hi]) - 5));
      }
      // the hovered point: a guide line, a dot, and its distance and height
      if (hov !== null) {
        const fi = hov * (n - 1), i0 = Math.floor(fi), i1 = Math.min(n - 1, i0 + 1);
        const zv = lerp(z[i0], z[i1], fi - i0), x = padL + (W - padL - padR) * hov, y = Y(zv);
        ctx.strokeStyle = css("--muted"); ctx.lineWidth = 1; ctx.globalAlpha = 0.5;
        ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, H - bottom); ctx.stroke(); ctx.globalAlpha = 1;
        ctx.beginPath(); ctx.arc(x, y, 5, 0, 2 * Math.PI);
        ctx.fillStyle = colour; ctx.fill(); ctx.strokeStyle = "#fff"; ctx.lineWidth = 2.5; ctx.stroke();
        const text = distText(hov * dist) + " · " + lab(zv);
        ctx.font = "600 10px " + css("--mono");
        const tw = ctx.measureText(text).width + 10;
        let tx = x - tw / 2; tx = clamp(tx, padL, W - padR - tw);
        const ty = y - 10 >= top + 14 ? y - 10 : y + 10;
        ctx.fillStyle = css("--card"); ctx.globalAlpha = 0.92;
        ctx.beginPath(); ctx.roundRect(tx, ty - 11, tw, 15, 4); ctx.fill(); ctx.globalAlpha = 1;
        ctx.fillStyle = css("--ink"); ctx.textAlign = "left"; ctx.fillText(text, tx + 5, ty);
      }
    },

    /* what the map is fitted to: the whole frontier between two places
     * (its routes share the ends), but only the loop on show, since the
     * other loops offered can lie in any direction from the start */
    familyBounds(u) {
      const b = L.latLngBounds([]);
      const members = this.family.loop ? [u || this.shown || this.family.unique[0]] : this.family.unique;
      for (const u of members) for (const ll of u.latlngs) b.extend(ll);
      return b;
    },
    /* is the whole family inside the part of the map the card does not cover? */
    inView(u) {
      if (!this.family) return true;
      const map = this.map, size = map.getSize(), card = $("card").getBoundingClientRect();
      const b = this.familyBounds(u);
      const sw = map.latLngToContainerPoint(b.getSouthWest()), ne = map.latLngToContainerPoint(b.getNorthEast());
      const wide = size.x > 640;
      const x0 = wide ? card.right + 16 : 16, y1 = wide ? size.y - 16 : card.top - 16;
      return sw.x >= x0 && ne.x <= size.x - 16 && ne.y >= 16 && sw.y <= y1;
    },
    /* how much of the uncovered map the family spans, in its larger
     * direction (1 = edge to edge) */
    viewShare(u) {
      if (!this.family) return 1;
      const map = this.map, size = map.getSize(), card = $("card").getBoundingClientRect();
      const b = this.familyBounds(u);
      const sw = map.latLngToContainerPoint(b.getSouthWest()), ne = map.latLngToContainerPoint(b.getNorthEast());
      const wide = size.x > 640;
      const w = wide ? size.x - card.right - 32 : size.x - 32, h = wide ? size.y - 32 : card.top - 32;
      return Math.max((ne.x - sw.x) / Math.max(w, 1), (sw.y - ne.y) / Math.max(h, 1));
    },
    fit(u) {
      if (!this.family) return;
      const b = this.familyBounds(u);
      const size = this.map.getSize();
      const wide = size.x > 640;
      const card = $("card").getBoundingClientRect();
      // generous margins that grow with the window: a long straight run
      // that reaches the edge reads as cut off; the zoom control sits
      // bottom right
      const my = Math.max(56, Math.round(size.y * 0.12)), mx = Math.max(64, Math.round(size.x * 0.06));
      // one short glide every time (the map's zoomAnimationThreshold is
      // raised so large changes animate too, instead of cutting)
      this.map.fitBounds(b, wide
        ? { paddingTopLeft: [card.right + mx, my], paddingBottomRight: [mx, my + 16], maxZoom: 15 }
        : { paddingTopLeft: [28, Math.max(36, Math.round(size.y * 0.12))], paddingBottomRight: [28, this.cardHeight() + 28], maxZoom: 15 });
    },
    /* the height the card is about to have: blocks easing open or shut
     * carry their target height inline, so the bottom sheet's final size
     * is known before the easing ends and the map can be fitted to it */
    cardHeight() {
      const card = $("card");
      // while typing on a phone the card is folded to the fields; measure it
      // as it will be once the result shows
      const typing = card.classList.contains("typing");
      if (typing) card.classList.remove("typing");
      let h = card.getBoundingClientRect().height;
      if (typing) card.classList.add("typing");
      for (const el of card.querySelectorAll("[style]")) {
        if (el.style.height) h += parseFloat(el.style.height) - el.getBoundingClientRect().height;
        if (el.style.marginTop) h += parseFloat(el.style.marginTop) - parseFloat(getComputedStyle(el).marginTop);
      }
      // the card's max-height is a share of the viewport, which the map
      // fills (its parent may be an unsized wrapper when the page is embedded)
      const max = getComputedStyle(card).maxHeight;
      if (max.endsWith("%")) h = Math.min(h, this.map.getSize().y * parseFloat(max) / 100);
      else if (max.endsWith("px")) h = Math.min(h, parseFloat(max));
      return h;
    },

    /* ------------------------------------------------------------ sharing */
    /* The trip travels in the URL fragment as one bare token of letters,
     * digits and . _ ~ - (labels hex-escaped), which survives any host,
     * link shortener or chat client that mangles key=value fragments. */
    token() {
      const { from, to, mode, t } = this.state;
      const c = (p) => p.lon.toFixed(5) + "~" + p.lat.toFixed(5);
      const m = mode === "bike" ? (this.state.calm ? "b" : "bx") : "w";
      if (this.state.loop) return ["l", c(from), m + (this.state.outBack ? "o" : ""), String(+this.state.loopMi.toFixed(3)), String(this.state.loopIdx), encLabel(from.label)].join("~");
      return ["t", c(from), c(to), mode === "bike" ? (this.state.calm ? "b" : "bx") : "w", t.toFixed(3),
        encLabel(from.label), encLabel(to.label)].join("~");
    },
    writeHash() {
      const { from, to } = this.state;
      if (!from || (!to && !this.state.loop)) return;
      try { history.replaceState(null, "", "#" + this.token()); } catch (e) { /* sandboxed */ }
    },
    shareUrl() {
      let base = "";
      try { base = location.href.split("#")[0]; } catch (e) { /* sandboxed */ }
      return base + "#" + this.token();
    },
    readHash() {
      let h = "";
      try { h = location.hash; } catch (e) { return false; }
      if (!h || h.length < 2) return false;
      const parts = h.slice(1).split("~");
      if (parts[0] === "l" && parts.length >= 6) return this.readLoopHash(parts);
      if (parts[0] !== "t" || parts.length < 8) return false;
      const nums = parts.slice(1, 5).map(Number);
      if (nums.some((v) => !Number.isFinite(v))) return false;
      if (parts[5] === "b" || parts[5] === "bx") {
        this.state.mode = "bike";
        this.state.calm = parts[5] === "b";
        $("calm").checked = this.state.calm;
        $("calmrow").hidden = false;
        for (const b of $("mode").querySelectorAll("button")) b.setAttribute("aria-pressed", b.dataset.v === "bike" ? "true" : "false");
      }
      const tt = parseFloat(parts[6]);
      if (Number.isFinite(tt)) { this.state.t = clamp(tt, 0, 1); $("sl").value = this.state.t; }
      this.setPoint("from", this.pointAt(nums[0], nums[1], decLabel(parts[7]) || undefined), false);
      this.setPoint("to", this.pointAt(nums[2], nums[3], decLabel(parts[8] || "") || undefined), false);
      return true;
    },
    /* #l~lon~lat~mode~miles~which~label (mode ending in o: out-and-back allowed) */
    readLoopHash(parts) {
      const lon = +parts[1], lat = +parts[2], mi = +parts[4];
      if (![lon, lat, mi].every(Number.isFinite)) return false;
      let m = parts[3] || "w";
      if (m.endsWith("o")) { m = m.slice(0, -1); this.state.outBack = true; $("outback").checked = true; }
      this.setMode(m);
      // the length as the link gives it, not rounded to this reader's
      // slider: a 5 km link (3.107 mi) opened with miles chosen used to
      // become a 3 mi loop of 4,828 m, a different loop from the sender's
      this.state.loopMi = clamp(mi, LOOP_MIN_MI, LOOP_MAX_MI);
      this.setPoint("from", this.pointAt(lon, lat, decLabel(parts[6] || "") || undefined), false);
      this.setLoop(true, false);
      this._pendingLoopIdx = Math.max(0, parseInt(parts[5], 10) || 0);
      return true;
    },
    setMode(tok) {
      if (tok !== "b" && tok !== "bx") return;
      this.state.mode = "bike";
      this.state.calm = tok === "b";
      $("calm").checked = this.state.calm;
      $("calmrow").hidden = false;
      for (const b of $("mode").querySelectorAll("button")) b.setAttribute("aria-pressed", b.dataset.v === "bike" ? "true" : "false");
    },
  };

  /* "5 km loop", "3.1 mi loop": the loop's true length to one decimal */
  function fmtLoop(mi) {
    const r = loopSlider();
    return (+r.exact(mi).toFixed(1)) + " " + r.unit + " loop";
  }

  /* A name for a file: letters and digits of any script, in lower case,
   * hyphens between, cut at a word if it runs past ``max`` characters.
   * (Kept to a-z and 0-9, a trip between 中環站 and 香港大學站 was
   * "flattenhk-to-flattest.gpx".) */
  function fileSlug(text, max) {
    const t = text.toLowerCase().replace(/[^\p{L}\p{N}]+/gu, "-").replace(/^-|-$/g, "");
    if (t.length <= max) return t;
    const cut = t.slice(0, max + 1), at = cut.lastIndexOf("-");
    return at > 0 ? cut.slice(0, at) : t.slice(0, max);
  }

  /* a point with no name to go by: "22.2810, 114.1600" */
  function coordText(lon, lat) { return lat.toFixed(4) + ", " + lon.toFixed(4); }

  function encLabel(s) {
    // by UTF-16 unit, so a character beyond the basic plane (Hong Kong
    // writes "lift" with one) goes out as its two halves and comes back whole
    let out = "";
    s = String(s || "");
    for (let i = 0; i < s.length; i++) {
      const ch = s[i], c = s.charCodeAt(i);
      if (/[A-Za-z0-9.\-]/.test(ch)) out += ch;
      else out += c < 256 ? "_" + c.toString(16).padStart(2, "0") : "_u" + c.toString(16).padStart(4, "0");
    }
    return out;
  }
  function decLabel(s) {
    return String(s || "").replace(/_u([0-9a-f]{4})|_([0-9a-f]{2})/gi, (m, u, b) => String.fromCharCode(parseInt(u || b, 16)));
  }

  /* ------------------------------------------------------------- helpers */
  const STREET_SHORT = { Street: "St", Avenue: "Ave", Boulevard: "Blvd", Drive: "Dr", Road: "Rd",
    Terrace: "Ter", Place: "Pl", Court: "Ct", Lane: "Ln", Highway: "Hwy", Parkway: "Pkwy" };
  function shortStreet(name) {
    return String(name).split(" ").map((w) => STREET_SHORT[w] || w).join(" ");
  }

  /* Keep at most k members, spread evenly along the frontier's length in
   * normalised (distance, climbing) space, always keeping both ends.
   * ``key`` is the length the frontier was searched and sorted on:
   * stress_m with calm streets on. Thinning and re-sorting such a family on
   * real distance, as this did, shuffled it: by bike from Yuen Long Station
   * to Tai Po Market Station climbing rose on 16 of the slider's 29 steps
   * and its "flattest" end climbed 168 m against the family's 121 m. */
  function thinFrontier(members, k, key = "distance_m") {
    const n = members.length;
    if (n <= k) return members;
    const d = members.map((m) => m.stats[key]), c = members.map((m) => m.stats.elev_gain_m);
    const dr = Math.max(1e-9, d[n - 1] - d[0]), cr = Math.max(1e-9, c[0] - c[n - 1]);
    const cum = [0];
    for (let i = 1; i < n; i++) {
      cum.push(cum[i - 1] + Math.hypot((d[i] - d[i - 1]) / dr, (c[i] - c[i - 1]) / cr));
    }
    const total = cum[n - 1], out = [], used = new Set();
    for (let j = 0; j < k; j++) {
      const target = total * j / (k - 1);
      let best = -1, bestErr = Infinity;
      for (let i = 0; i < n; i++) {
        if (used.has(i)) continue;
        const err = Math.abs(cum[i] - target);
        if (err < bestErr) { bestErr = err; best = i; }
      }
      used.add(best); out.push(members[best]);
    }
    return out.sort((a, b) => a.stats[key] - b.stats[key]);
  }

  /* the point a fraction f of the way along a route's geometry; the
   * cumulative length along its points is kept on the route */
  function alongRoute(u, f) {
    const pts = u.latlngs;
    if (!u._cum) {
      const c = new Float64Array(pts.length);
      const ky = 110540, kx = 111320 * Math.cos(pts[0][0] * Math.PI / 180);
      for (let i = 1; i < pts.length; i++) {
        const dx = (pts[i][1] - pts[i - 1][1]) * kx, dy = (pts[i][0] - pts[i - 1][0]) * ky;
        c[i] = c[i - 1] + Math.sqrt(dx * dx + dy * dy);
      }
      u._cum = c;
    }
    const c = u._cum, want = f * c[c.length - 1];
    let lo = 0, hi = c.length - 1;
    while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (c[mid] <= want) lo = mid; else hi = mid; }
    const t = c[hi] > c[lo] ? (want - c[lo]) / (c[hi] - c[lo]) : 0;
    return L.latLng(lerp(pts[lo][0], pts[hi][0], t), lerp(pts[lo][1], pts[hi][1], t));
  }
  function resample(prof, n) {
    const { d, z } = prof;
    const out = new Float64Array(n);
    if (!d.length) return { z: out };
    const total = d[d.length - 1] || 1;
    let j = 0;
    for (let i = 0; i < n; i++) {
      const x = total * i / (n - 1);
      while (j < d.length - 2 && d[j + 1] < x) j++;
      const span = d[j + 1] - d[j];
      out[i] = span > 0 ? lerp(z[j], z[j + 1], (x - d[j]) / span) : z[j];
    }
    return { z: out };
  }

  function tween(ms, fn) {
    const t0 = performance.now();
    const frame = (now) => { const k = clamp((now - t0) / ms, 0, 1); fn(ease(k)); if (k < 1) requestAnimationFrame(frame); };
    requestAnimationFrame(frame);
  }
  function fadeOut(layers, ms, done) {
    const start = layers.map((l) => l.options.opacity);
    tween(ms, (k) => { layers.forEach((l, i) => l.setStyle({ opacity: start[i] * (1 - k) })); if (k >= 1) done(); });
  }
  function fadeIn(pairs, ms) {
    tween(ms, (k) => { pairs.forEach(([l, o]) => l.setStyle({ opacity: o * k })); });
  }
  /* the elevation range of some profiles, optionally grown from a range */
  function rangeOf(members, from) {
    let min = from ? from.min : Infinity, max = from ? from.max : -Infinity;
    for (const u of members) for (const v of u.profile.z) { if (v < min) min = v; if (v > max) max = v; }
    return { min, max };
  }
  /* Text changes crossfade rather than jump: the element fades out, the
   * new content (a string, or a function that fills the element) goes in,
   * and it fades back.  Calls during a fade coalesce into the latest. */
  function swapText(el, fill) {
    if (typeof fill !== "function" && el.textContent === fill && !el._swap) return;
    el._swap = typeof fill === "function" ? () => fill(el) : () => { el.textContent = fill; };
    if (el._swapTimer) return;
    if (!el.textContent && !el.children.length) { el._swap(); el._swap = null; return; }
    el.classList.add("fading");
    el._swapTimer = setTimeout(() => {
      el._swapTimer = null;
      const f = el._swap; el._swap = null; if (f) settle(el, f);
      el.classList.remove("fading");
    }, FADE_MS);
  }
  /* empty an element now, and drop any change still waiting on its fade:
   * a loop search that failed within the fade of "About 5 km, in every
   * direction from here." used to be left with that line under "No loop
   * from here" */
  function cutText(el) {
    clearTimeout(el._swapTimer);
    el._swapTimer = null; el._swap = null;
    el.classList.remove("fading");
    el.textContent = "";
  }
  /* run a change to an element's content, easing its height between
   * what it was and what it has become so the blocks below slide */
  function settle(el, change) {
    const h0 = el.offsetHeight;
    change();
    const h1 = el.offsetHeight;
    if (h0 === h1 || reducedMotion()) return;
    el.style.transition = "none"; el.style.height = h0 + "px"; el.style.overflow = "hidden";
    void el.offsetWidth;
    el.style.transition = ""; el.style.height = h1 + "px";
    clearTimeout(el._growTimer);
    el._growTimer = setTimeout(() => { el.style.height = ""; el.style.overflow = ""; }, GROW_MS + 20);
  }
  function reducedMotion() { return window.matchMedia("(prefers-reduced-motion: reduce)").matches; }
  /* and set content without a fade (a counter, the slider); a fade in
   * flight just picks up the latest value instead */
  function setText(el, value, html) {
    const apply = () => { if (html) el.innerHTML = value; else el.textContent = value; };
    if (el._swapTimer) el._swap = apply; else apply();
  }
  /* hide and show blocks with the same fade, their space closing and
   * opening (the card's gap included) rather than snapping */
  function hideSoft(el) {
    if (el.hidden || el._hiding) return;
    el._hiding = true; el.classList.add("fading");
    setTimeout(() => {
      if (!el._hiding) return;                 // shown again meanwhile
      el.style.transition = "none"; el.style.height = el.offsetHeight + "px"; el.style.overflow = "hidden";
      void el.offsetWidth;
      el.style.transition = ""; el.style.height = "0px"; el.style.marginTop = "-10px";
      clearTimeout(el._growTimer);
      el._growTimer = setTimeout(() => {
        if (!el._hiding) return;
        el.hidden = true; el._hiding = false; el.classList.remove("fading");
        el.style.height = ""; el.style.marginTop = ""; el.style.overflow = "";
      }, GROW_MS + 20);
    }, reducedMotion() ? 0 : FADE_MS);
  }
  function showSoft(el) {
    const wasHiding = el._hiding; el._hiding = false;
    if (!el.hidden && !wasHiding && !el.classList.contains("fading")) return;
    clearTimeout(el._growTimer);
    if (el.hidden || wasHiding) {
      // open from whatever height it has now (0 when hidden) to its full height
      const h0 = el.hidden ? 0 : el.offsetHeight;
      el.hidden = false; el.classList.add("fading");
      el.style.transition = "none"; el.style.height = ""; el.style.marginTop = ""; el.style.overflow = "hidden";
      const h1 = el.offsetHeight;
      el.style.height = h0 + "px"; if (h0 === 0) el.style.marginTop = "-10px";
      void el.offsetWidth;
      el.style.transition = ""; el.style.height = h1 + "px"; el.style.marginTop = "";
      el._growTimer = setTimeout(() => { el.style.height = ""; el.style.overflow = ""; el.classList.remove("fading"); }, GROW_MS);
      return;
    }
    // Called again while the block is still easing open (the first route
    // is painted, and a moment later the whole family): the timer that
    // would have released the block is gone, so release it here. The
    // height it was opening to was measured before the numbers were
    // written, and left in place it cut the captions off the stats for
    // good (18 px of 40), and on a phone, where the card's flex layout
    // squeezes a clipped block, shut the stats and the street list.
    el.style.height = ""; el.style.overflow = ""; el.style.marginTop = "";
    el.classList.add("fading"); el.hidden = false;
    void el.offsetWidth;                       // flush, so the fade-in runs
    el.classList.remove("fading");
  }

  App.ALPHA_MAX = ALPHA_MAX; App._gen = 0;
  window.App = App;
  App.start().catch((err) => {
    console.error(err);
    $("status").textContent = "Could not start: " + err.message;
  });
})();
