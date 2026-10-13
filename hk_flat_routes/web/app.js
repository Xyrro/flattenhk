/* Hong Kong flat routes -- the explorer page (everything on screen).
 * Adapted from flattensf (MIT) by Drew Edwards,
 * https://github.com/almostimplemented/flattensf
 * Depends on engine.js.
 *
 * What differs from San Francisco, and why:
 *  - metric only: kilometres, metres and metres above Principal Datum (mPD);
 *  - Hong Kong is several landmasses with no walking or cycling link between
 *    them, so a pair of points may have no route, and the page says which
 *    two landmasses they are on instead of drawing nothing;
 *  - the network is about three times San Francisco's, so the street canvas
 *    works from coordinates projected once, culls to the view and thins by
 *    zoom, and the lowland basins are painted from the packed graph too;
 *  - street names are shown in English and in Chinese;
 *  - there is no online base map: the page is whole on its own. */
"use strict";

/* --------------------------------------------- street network canvas layer */
const BUCKETS = [
  { colour: "#2c7bb6", label: "under 3% - flat" },
  { colour: "#7fcdbb", label: "3-5% - gentle" },
  { colour: "#fed976", label: "5-8% - noticeable" },
  { colour: "#fd8d3c", label: "8-10% - hard" },
  { colour: "#e3492e", label: "10-15% - very hard" },
  { colour: "#7f1d1d", label: "over 15% - extreme" },
];

/* Classes that come in only as the view closes: footpaths, steps, service
 * lanes and trails are three quarters of Hong Kong's edges (76% of the
 * 260,000 in the whole territory), half of them under 19 m long, and at the
 * scale of the whole territory they are noise. */
const MINOR_CLASSES = new Set(["footway", "steps", "path", "service", "track",
  "bridleway", "unknown"]);

/* The shortest edge (m) worth drawing at a zoom: one figure for streets and
 * one for the minor classes, both set in pixels. A street shorter than a
 * third of a pixel adds nothing but a dot; a footpath has to be a pixel and
 * a half long before it earns a place beside the streets. A long trail or
 * catchwater path therefore still shows from far out, so the hills keep
 * their colours, and the short pieces arrive at the zoom at which they
 * could be told apart. From zoom 15 everything is drawn. Counted on the
 * urban core's 109,000 edges, that is 14,000 polylines with all of them in
 * view at zoom 10.5, 35,000 at zoom 12 and 53,000 at zoom 13, the closest
 * view that still holds the whole core. */
const STREET_PX = 0.35, MINOR_PX = 1.5;
function drawFloor(z, lat) {
  if (z >= 15) return { street: 0, minor: 0 };
  const mPerPx = 156543.03 * Math.cos(lat * Math.PI / 180) / Math.pow(2, z);
  return { street: STREET_PX * mPerPx, minor: MINOR_PX * mPerPx };
}

/* Bounding box of every edge in the Web Mercator fractions of
 * Geometry.mercator(): [x0, y0, x1, y1] per edge. */
function mercatorBoxes(geom, P) {
  const box = new Float32Array(geom.nEdges * 4), { mx, my } = P, starts = geom.starts;
  for (let i = 0; i < geom.nEdges; i++) {
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (let k = starts[i] >> 1, e = starts[i + 1] >> 1; k < e; k++) {
      const x = mx[k], y = my[k];
      if (x < x0) x0 = x; if (x > x1) x1 = x;
      if (y < y0) y0 = y; if (y > y1) y1 = y;
    }
    box[i * 4] = x0; box[i * 4 + 1] = y0; box[i * 4 + 2] = x1; box[i * 4 + 3] = y1;
  }
  return box;
}

/* A quarter of a million polylines is far too many for individual Leaflet
 * layers, so the network is painted straight onto a canvas from the packed
 * arrays. Every vertex is projected to Web Mercator once (engine.js,
 * Geometry.mercator), after which a repaint is a multiply and an add per
 * coordinate; edges outside the view are skipped by bounding box, and the
 * shortest ones by zoom (drawFloor).
 *
 * Options: ``mask`` (a per-edge array; only edges with a non-zero entry are
 * drawn) and ``colour`` (one colour for all of them, instead of the six
 * gradient classes) make the same painter serve the lowland-basin layer. */
const NetworkLayer = L.Layer.extend({
  initialize(geom, opts) {
    this.geom = geom;
    L.setOptions(this, opts || {});
    this.alt = null;     // {mx, my}: where the vertices are under a warp
    this.dim = null;     // per edge, 1 = draw de-emphasised (another landmass)
  },

  onAdd(map) {
    this._map = map;
    this._canvas = L.DomUtil.create("canvas", "netcanvas leaflet-zoom-animated");
    // under Leaflet's own canvas (routes, corridors, markers), which must
    // keep the mouse, and over any street canvas added before this one
    const pane = map.getPanes().overlayPane;
    const above = Array.from(pane.children).find(el => !el.classList.contains("netcanvas"));
    pane.insertBefore(this._canvas, above || null);
    map.on("moveend zoomend resize", this._redraw, this);
    map.on("zoomanim", this._animate, this);
    this._redraw();
  },

  onRemove(map) {
    L.DomUtil.remove(this._canvas);
    map.off("moveend zoomend resize", this._redraw, this);
    map.off("zoomanim", this._animate, this);
    this._map = null;
  },

  /* Draw the vertices somewhere else (the warped city), or back where they
   * are with null. */
  setAlt(alt, dim) {
    this.alt = alt || null;
    this.dim = alt ? (dim || null) : null;
    this._altBox = alt ? mercatorBoxes(this.geom, alt) : null;
  },

  _animate(e) {
    const scale = this._map.getZoomScale(e.zoom);
    const offset = this._map._latLngToNewLayerPoint(
      this._map.getBounds().getNorthWest(), e.zoom, e.center);
    L.DomUtil.setTransform(this._canvas, offset, scale);
  },

  _redraw() {
    const map = this._map;
    if (!map) return;
    const t0 = performance.now();
    const size = map.getSize(), dpr = window.devicePixelRatio || 1;
    const cv = this._canvas;
    if (cv.width !== Math.round(size.x * dpr) || cv.height !== Math.round(size.y * dpr)) {
      cv.width = Math.round(size.x * dpr); cv.height = Math.round(size.y * dpr);
      cv.style.width = size.x + "px"; cv.style.height = size.y + "px";
    }
    // the canvas covers the view exactly; its corner in layer coordinates
    // and in world pixels at this zoom
    const corner = map.containerPointToLayerPoint([0, 0]);
    L.DomUtil.setTransform(cv, corner, 1);
    const origin = map.getPixelOrigin().add(corner);

    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, size.x, size.y);

    const g = this.geom, n = g.nEdges, z = map.getZoom();
    const real = g.mercator();
    if (!this._box) this._box = mercatorBoxes(g, real);
    const P = this.alt || real, box = this.alt ? this._altBox : this._box;
    const { mx, my } = P;
    // pixel = Mercator fraction * world size, less the canvas corner
    const S = 256 * Math.pow(2, z);
    const bx = real.x0 * S - origin.x, by = real.y0 * S - origin.y;
    // the view in the same fractions, with a little margin
    const pad = 0.04 * Math.max(size.x, size.y);
    const vx0 = (-pad - bx) / S, vx1 = (size.x + pad - bx) / S;
    const vy0 = (-pad - by) / S, vy1 = (size.y + pad - by) / S;

    if (!this._minor) this._minor = g.classes.map(c => MINOR_CLASSES.has(c));
    const minor = this._minor, floor = drawFloor(z, map.getCenter().lat), DM = g.DM;
    const mask = this.options.mask, dim = this.dim, single = !!this.options.colour;
    // one pass decides what is drawn and in which colour class, so the six
    // strokes below only walk a byte array
    const vis = this._vis || (this._vis = new Uint8Array(n));
    let shown = 0;
    for (let i = 0; i < n; i++) {
      vis[i] = 0;
      if (mask && !mask[i]) continue;
      const o = i * 4;
      if (box[o + 2] < vx0 || box[o] > vx1 || box[o + 3] < vy0 || box[o + 1] > vy1) continue;
      if (g.len[i] / DM < (minor[g.cls[i]] ? floor.minor : floor.street)) continue;
      vis[i] = dim && dim[i] ? 7 : (single ? 1 : g.bucket[i] + 1);
      shown++;
    }

    const starts = g.starts;
    const stroke = (cls) => {
      ctx.beginPath();
      for (let i = 0; i < n; i++) {
        if (vis[i] !== cls) continue;
        const s = starts[i] >> 1, e = starts[i + 1] >> 1;
        let px = mx[s] * S + bx, py = my[s] * S + by;
        ctx.moveTo(px, py);
        for (let k = s + 1; k < e; k++) {
          const x = mx[k] * S + bx, y = my[k] * S + by;
          // a vertex less than half a pixel on is not worth a line segment,
          // but the last one always is: it is where the next street starts
          if (k < e - 1 && (x - px) * (x - px) + (y - py) * (y - py) < 0.25) continue;
          ctx.lineTo(x, y); px = x; py = y;
        }
      }
      ctx.stroke();
    };
    ctx.lineCap = "round"; ctx.lineJoin = "round";
    const wide = this.options.widen || 0;
    if (dim) {
      // the landmasses a warp leaves alone: there, but out of the way
      ctx.strokeStyle = "#5d6875"; ctx.globalAlpha = 0.34;
      ctx.lineWidth = z >= 15 ? 1.2 : 0.8;
      stroke(7);
    }
    if (single) {
      ctx.strokeStyle = this.options.colour;
      ctx.globalAlpha = this.options.alpha || 0.6;
      ctx.lineWidth = (z >= 15 ? 2.0 : z >= 13 ? 1.35 : 1.0) + wide;
      stroke(1);
    } else {
      // flat first, steep last, so the walls read on top
      for (let bk = 0; bk < 6; bk++) {
        ctx.strokeStyle = BUCKETS[bk].colour;
        // The hills should read, but this is a map about flat streets, so the
        // flat classes are not allowed to disappear underneath them.
        ctx.globalAlpha = bk >= 3 ? 0.82 : 0.95;
        ctx.lineWidth = bk >= 3
          ? (z >= 15 ? 2.3 : z >= 13 ? 1.5 : 1.0)
          : (z >= 15 ? 2.0 : z >= 13 ? 1.35 : 1.0);
        stroke(bk + 1);
      }
    }
    ctx.globalAlpha = 1;
    // for the console and the tests: how much the last repaint drew
    this.lastDraw = { edges: shown, ms: performance.now() - t0, zoom: z };
  },
});

/* ------------------------------------------------------------- formatting */
function esc(v) {
  return String(v).replace(/[&<>"']/g, ch => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
}

/* Heights are metres above Principal Datum (mPD), Hong Kong's height datum,
 * about 1.2 m below mean sea level. */
function fmt(v, how) {
  if (v === null || v === undefined || v === "" || Number.isNaN(v)) return "—";
  switch (how) {
    case "pct": return (v * 100).toFixed(1) + "%";
    case "m0": return Math.round(v).toLocaleString() + " m";
    case "m1": return (+v).toFixed(1) + " m";
    case "mpd": return Math.round(v).toLocaleString() + " mPD";
    case "mkm": return (+v).toFixed(1) + " m/km";
    case "km": return (+v).toFixed(2) + " km";
    case "int": return (+v).toLocaleString();
    case "km2": return (+v).toFixed(2) + " km²";
    case "bool": return v ? "yes" : "no";
    default: return esc(v);
  }
}

const zhSpan = zh => (zh ? ` <span class="zh" lang="zh-Hant-HK">${esc(zh)}</span>` : "");

/* --------------------------------------------------------------- the app */
const App = {
  state: {
    mode: "walk", profile: "balanced",
    src: null, dst: null, picking: "src",
    alpha: null, beta: null, gamma: null, custom: false,
  },

  async start(DATA) {
    this.DATA = DATA;
    const buf = await inflate(DATA.bundle);
    const bundle = new Bundle(buf, DATA.manifest);
    this.meta = DATA.meta;
    this.graph = new Graph(bundle, DATA.meta);
    this.geom = new Geometry(bundle, DATA.meta);
    DATA.layers = JSON.parse(bundle.text("layers"));
    // which lowland basin each street is in (0 = none); see viz_interactive.py
    this.edgeBasin = bundle.has("edge_basin") ? bundle.array("edge_basin") : null;
    DATA.bundle = null;          // release the base64 string

    const g = this.graph;
    const lonF = new Float32Array(g.n), latF = new Float32Array(g.n);
    for (let i = 0; i < g.n; i++) { lonF[i] = g.nodeLon(i); latF[i] = g.nodeLat(i); }
    this.nodeGrid = new Grid(lonF, latF, 0.004);

    // edge midpoints, for click-to-inspect on the network layer
    const mx = new Float32Array(this.geom.nEdges), my = new Float32Array(this.geom.nEdges);
    for (let i = 0; i < this.geom.nEdges; i++) {
      const o = i * 4;
      mx[i] = (this.geom.bbox[o] + this.geom.bbox[o + 2]) / 2;
      my[i] = (this.geom.bbox[o + 1] + this.geom.bbox[o + 3]) / 2;
    }
    this.edgeGrid = new Grid(mx, my, 0.004);

    // the Chinese name of each street that has one, by its English name
    this.zhOf = new Map();
    const zh = this.meta.names_zh || [];
    this.meta.names.forEach((name, i) => { if (zh[i]) this.zhOf.set(name, zh[i]); });

    this.buildLand();
    this.buildMap();
    this.buildUI();
    this.setDefaultPair();
    this.recompute();
  },

  /* ------------------------------------------------------------ landmasses */
  /* One landmass number per junction and per street, whatever the travel
   * mode. The router's components are per mode (a cyclist's Hong Kong is
   * the walker's cut into more pieces, footpaths being closed to bikes), and
   * a street no mode can use belongs to none of them; but when one landmass
   * is redrawn and the rest are greyed, every street on it has to go along.
   * So the walking component is the landmass, a bike-only junction takes the
   * landmass most of its component stands on, and an unroutable street takes
   * that of a junction beside it. 0 means none could be found. */
  buildLand() {
    const g = this.graph, n = g.n;
    const walk = g.nodeComp.walk, bike = g.nodeComp.bike;
    const land = new Uint8Array(n);
    if (!walk) {
      // a bundle without component arrays is one landmass
      for (let i = 0; i < n; i++) land[i] = g.nodeFlags[i] ? 1 : 0;
    } else {
      const toWalk = new Uint8Array(256);
      if (bike) {
        const votes = new Map();
        for (let i = 0; i < n; i++) {
          if (!bike[i] || !walk[i]) continue;
          const k = bike[i] * 256 + walk[i];
          votes.set(k, (votes.get(k) || 0) + 1);
        }
        const best = new Int32Array(256);
        for (const [k, c] of votes) {
          const b = k >> 8;
          if (c > best[b]) { best[b] = c; toWalk[b] = k & 255; }
        }
      }
      for (let i = 0; i < n; i++) land[i] = walk[i] || (bike ? toWalk[bike[i]] : 0);
    }
    this.nodeLand = land;

    // And the other way round: the component of each mode's graph that
    // stands for a landmass, which is the one with most junctions on it.
    // A cyclist's network is in more pieces than the land is: in the
    // whole-territory build more than a third of its landmasses are
    // fragments of a landmass that has a larger one (Discovery Bay on
    // Lantau, Ap Lei Chau off Hong Kong Island), some 40 to 430 junctions
    // each (see warpComp and noRouteHTML).
    const mainComp = comp => {
      const out = new Uint8Array(256);
      if (!comp) { out[1] = 1; return out; }
      const votes = new Map();
      for (let i = 0; i < n; i++) {
        if (!comp[i] || !land[i]) continue;
        const k = land[i] * 256 + comp[i];
        votes.set(k, (votes.get(k) || 0) + 1);
      }
      const best = new Int32Array(256);
      for (const [k, c] of votes) {
        const l = k >> 8;
        if (c > best[l]) { best[l] = c; out[l] = k & 255; }
      }
      return out;
    };
    this.landComp = { walk: mainComp(walk), bike: mainComp(bike) };

    const nE = this.geom.nEdges, eland = new Uint8Array(nE);
    for (let u = 0; u < n; u++) {
      for (let a = g.indptr[u]; a < g.indptr[u + 1]; a++) {
        const e = g.arcEdge[a];
        if (!eland[e]) eland[e] = land[u] || land[g.head[a]];
      }
    }
    // unroutable streets: the landmass of the nearest junction within a few
    // hundred metres (the cells either side of the street's own)
    const c = this.geom.coords, starts = this.geom.starts;
    for (let e = 0; e < nE; e++) {
      if (eland[e] || starts[e + 1] <= starts[e]) continue;
      const x = c[starts[e]], y = c[starts[e] + 1];
      let best = 0, bestD = Infinity;
      for (const i of this.nodeGrid.near(x, y, 1)) {
        if (!land[i]) continue;
        const d = (g.nodeLon(i) - x) ** 2 + (g.nodeLat(i) - y) ** 2;
        if (d < bestD) { bestD = d; best = land[i]; }
      }
      eland[e] = best;
    }
    this.edgeLand = eland;
  },

  /* What to call the landmass a junction is on for a mode. */
  landName(node, mode) {
    return this.graph.compName(node, mode)
      || (this.meta.components && this.meta.components.walk || [])[this.nodeLand[node] - 1]
      || "";
  },

  /* A neighbourhood's access point as a junction of the graph (-1 if the
   * mode has none there). Snapping stays within what the mode can route. */
  pointNode(name, mode) {
    const p = (this.DATA.points[mode] || {})[name];
    if (!p) return -1;
    const bit = this.graph.modeBit(mode);
    return this.nodeGrid.nearest(p[0], p[1],
      i => (this.graph.nodeFlags[i] & bit) !== 0);
  },

  /* "Central District 中區", as HTML or as plain text */
  nbHTML(name) { return esc(name) + zhSpan((this.DATA.neighborhood_zh || {})[name]); },
  nbText(name) {
    const zh = (this.DATA.neighborhood_zh || {})[name];
    return zh ? `${name} ${zh}` : name;
  },
  /* "Queen's Road Central 皇后大道中": a street's name in both languages.
   * The name table holds the English name where there is one and the
   * Chinese otherwise, with the Chinese of the first kind alongside. */
  streetHTML(name, zh) {
    if (!name) return "(unnamed)";
    const other = zh || this.zhOf.get(name);
    return esc(name) + (other && other !== name ? zhSpan(other) : "");
  },

  /* ---------------------------------------------------------------- map */
  /* The extent of the street data itself. The neighbourhood polygons are
   * no guide: they stop at the country parks, and in a partial build they
   * cover districts the network does not reach. */
  dataBounds() {
    const b = this.geom.bbox;
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (let o = 0; o < b.length; o += 4) {
      if (b[o] < x0) x0 = b[o]; if (b[o + 1] < y0) y0 = b[o + 1];
      if (b[o + 2] > x1) x1 = b[o + 2]; if (b[o + 3] > y1) y1 = b[o + 3];
    }
    if (!(x1 >= x0)) return L.latLngBounds([22.15, 113.83], [22.56, 114.44]);
    return L.latLngBounds([y0, x0], [y1, x1]);
  },

  buildMap() {
    const bounds = this.dataBounds();
    const map = L.map("map", {
      preferCanvas: true, center: bounds.getCenter(), zoom: 12,
      // the whole territory is 65 km across and has to fit a laptop screen;
      // quarter zoom steps let it fit snugly
      minZoom: 9, maxZoom: 18, zoomSnap: 0.25, zoomControl: true,
      maxBounds: bounds.pad(0.6), maxBoundsViscosity: 0.8,
    });
    // set at once, not flown to: nothing is drawn twice on the way in
    this._fitCity = () => map.fitBounds(bounds, { padding: [14, 14], animate: false });
    this.map = map;
    // bottom right, above the credits: they run the width of a small map and
    // would cover a scale bar in the usual corner
    L.control.scale({ imperial: false, metric: true, position: "bottomright" }).addTo(map);
    // No base map is fetched. Upstream laid CARTO's dark tiles under the
    // network; the Atlanta port reports that they now answer with an "API
    // key required" tile (not checked here) and went to a plain ground too.
    // The streets carry the page, and it works with no connection at all.
    map.attributionControl.addAttribution(
      'Streets &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
      + ' contributors via <a href="https://overturemaps.org">Overture Maps</a>'
      + ' &middot; elevation: CEDD 2020 LiDAR DTM &copy; HKSAR Government'
      + ' (<a href="https://portal.csdi.gov.hk/">CSDI</a>)'
      + ' &middot; planning areas &copy; Town Planning Board'
      + ' &middot; /*__CITY_NAME__*/ port of <a href="/*__UPSTREAM_SITE__*/">/*__UPSTREAM_NAME__*/</a>'
      + ' by /*__UPSTREAM_AUTHOR__*/');

    // A line weight or a marker chosen for a city on screen is a smear when
    // the whole territory is, which San Francisco never has to be: corridors
    // and passes thin as the view pulls back.
    const farScale = () => { const z = map.getZoom(); return z >= 12.5 ? 1 : z >= 11 ? 0.7 : 0.5; };

    this.overlays = {};
    this.network = new NetworkLayer(this.geom).addTo(map);
    this.overlays.network = this.network;

    const layers = this.DATA.layers;
    const basinM = this.DATA.basin_elev_m;
    this.basinWord = basinM ? `below ${basinM} mPD` : "low ground";
    const topScore = Math.max(1, ...((layers.corridors || { features: [] }).features
      .map(f => +f.properties.total_score || 0)));
    this.vectorDefs = {
      neighborhoods: () => ({ color: "#8492a0", weight: 1.1, opacity: 0.8, fill: false, dashArray: "4,3" }),
      bike_network: () => ({ color: "#39d98a", weight: 2.0, opacity: 0.9 }),
      low_stress: () => ({ color: "#c792ea", weight: 2.6, opacity: 0.95 }),
      barriers: () => ({ color: "#ff5f4d", weight: 3.0, opacity: 0.85 }),
      // the widest corridor is the one that serves most; the score grows
      // with the number of neighbourhood pairs, so it is taken relative to
      // the best one here rather than to San Francisco's
      corridors: f => ({
        color: f.properties.mode === "bike" ? "#9be7ff" : "#2c7bb6",
        weight: (4.2 + 3.4 * Math.min((+f.properties.total_score || 0) / topScore, 1)) * farScale(),
        opacity: 0.95, lineCap: "round",
      }),
    };
    const defs = [
      ["neighborhoods", "Neighbourhood boundaries (planning areas)", true],
      ["basins", `Lowland basins (street ${this.basinWord})`, false],
      ["bike_network", "Cycle tracks and paths open to bicycles (OSM-derived)", false],
      ["low_stress", "Car-free / living streets", false],
      ["barriers", "Steep barriers (unavoidable climbs)", false],
      ["corridors", "Flat corridors (discovered)", true],
    ];
    this.layerDefs = [{ key: "network", label: "Streets coloured by gradient", on: true }];
    for (const [key, label, on] of defs) {
      if (key === "basins") {
        // painted from the packed graph, like the network: most of the
        // city's streets are in one basin or another
        if (!this.edgeBasin || !layers.basins) continue;
        this.overlays.basins = new NetworkLayer(this.geom, {
          mask: this.edgeBasin, colour: "#3f9b6d", alpha: 0.6, widen: 0.5,
        });
      } else {
        if (!layers[key] || !layers[key].features.length) continue;
        // a layer that starts switched off is built when it is first shown
        if (on) this.overlay(key).addTo(map);
      }
      this.layerDefs.push({ key, label, on });
    }
    if (layers.passes && layers.passes.features.length) {
      const passRadius = () => 6 * farScale();
      this.overlays.passes = L.layerGroup(
        layers.passes.features.map(f => {
          const c = f.geometry.coordinates;
          const flat = Array.isArray(c[0][0]) ? c[0] : c;
          const mid = flat[Math.floor(flat.length / 2)];
          return L.circleMarker([mid[1], mid[0]], {
            radius: passRadius(), color: "#4a3b10", weight: 1.4,
            fillColor: "#ffd166", fillOpacity: 0.95,
          }).bindPopup(this.popupHTML("passes", f.properties));
        })).addTo(map);
      map.on("zoomend", () => this.overlays.passes.eachLayer(m => m.setRadius(passRadius())));
      this.layerDefs.push({ key: "passes", label: "Critical passes / saddles", on: true });
    }

    this.cmpLine = L.polyline([], {
      color: "#c3ced9", weight: 3.4, opacity: 0.95, dashArray: "7,6",
    }).addTo(map);
    this.routeHalo = L.polyline([], { color: "#01080d", weight: 10, opacity: 0.85 }).addTo(map);
    this.routeLine = L.polyline([], { color: "#ffffff", weight: 4.4, opacity: 1 }).addTo(map);
    this.marks = L.layerGroup().addTo(map);

    map.on("click", e => this.onMapClick(e));
    map.on("zoomend", () => {
      if (this.overlays.corridors) this.overlays.corridors.setStyle(this.vectorDefs.corridors);
    });
    this._fitCity();

    const legend = L.control({ position: "bottomleft" });
    legend.onAdd = () => {
      const div = L.DomUtil.create("div", "legend");
      div.innerHTML =
        '<div class="key"><i style="background:#ffffff;height:4px"></i> your route</div>'
        + '<div class="key"><i style="background:#8a97a4"></i> shortest route (comparison)</div>'
        + '<div class="key"><i style="background:#2c7bb6;height:5px"></i> flat corridor</div>'
        + '<div class="key"><i style="background:#ff5f4d"></i> steep barrier</div>'
        + '<div class="key"><i style="background:#ffd166;height:9px;width:9px;border-radius:50%"></i> critical pass</div>';
      return div;
    };
    legend.addTo(map);
  },

  /* The Leaflet layer for one of the GeoJSON overlays, built on first use.
   *
   * A neighbourhood answers a click on its outline only. Leaflet's canvas
   * renderer counts a click anywhere inside a polygon as a click on it,
   * filled or not, and a layer with a popup keeps the click from the map;
   * the planning areas cover all of the built-up city, so with the outlines
   * showing (as they are when the page opens) no click on the map would
   * ever set an origin. */
  overlay(key) {
    if (!this.overlays[key]) {
      this.overlays[key] = L.geoJSON(this.DATA.layers[key], {
        style: this.vectorDefs[key],
        onEachFeature: (f, l) => {
          l.bindPopup(() => this.popupHTML(key, f.properties));
          if (key === "neighborhoods" && l instanceof L.Polygon) {
            l._containsPoint = p => L.Polyline.prototype._containsPoint.call(l, p, true);
          }
        },
      });
    }
    return this.overlays[key];
  },

  popupHTML(kind, props) {
    const spec = {
      corridors: [["corridor_name", "Corridor", "streets"], ["mode", "Mode"],
        ["length_km", "Length", "km"], ["mean_abs_grade", "Mean gradient", "pct"],
        ["max_grade", "Max gradient", "pct"], ["gain_per_km", "Climb per km", "mkm"],
        ["pair_count_max", "Neighbourhood pairs served", "int"],
        ["neighborhood_span", "Neighbourhoods spanned", "int"],
        ["climb_saved_m", "Climbing avoided (total)", "m0"],
        ["elev_min_m", "Lowest point", "mpd"], ["elev_max_m", "Highest point", "mpd"],
        ["neighborhoods", "Passes through"], ["street_names", "Streets", "streets"]],
      passes: [["name", "Street", "street"], ["neighborhood", "Neighbourhood", "nb"],
        ["nearest_neighborhood", "Nearest neighbourhood", "nb"],
        ["pass_elev_m", "Lowest possible crossing", "mpd"],
        ["pairs_served", "Neighbourhood pairs forced over it", "int"],
        ["max_abs_grade", "Max gradient", "pct"],
        ["neighborhoods_separated", "Separates"]],
      barriers: [["name", "Street", "street"], ["neighborhood", "Neighbourhood", "nb"],
        ["nearest_neighborhood", "Nearest neighbourhood", "nb"],
        ["max_abs_grade", "Max gradient", "pct"], ["length_m", "Length", "m0"],
        ["shortest_use", "Pairs via shortest route", "int"],
        ["flat_use_per_objective", "... still via the flat route", "int"],
        ["unavoidability", "Unavoidable share", "pct"]],
      bike_network: [["name", "Street", "street"], ["bike_facility", "Facility"],
        ["cls", "Road class"], ["max_abs_grade", "Max gradient", "pct"],
        ["length_m", "Length", "m0"]],
      low_stress: [["name", "Street", "street"], ["cls", "Class"],
        ["bike_facility", "Facility"], ["length_m", "Length", "m0"]],
      neighborhoods: [["neighborhood", "Neighbourhood", "nb"], ["area_km2", "Area", "km2"]],
    }[kind] || [];
    let h = "";
    for (const [k, lab, how] of spec) {
      if (!(k in props) || props[k] === null || props[k] === "") continue;
      // the nearest neighbourhood stands in only where the street is in none
      if (k === "nearest_neighborhood" && props.neighborhood) continue;
      let v;
      if (how === "street") v = this.streetHTML(props[k], props[k + "_zh"]);
      else if (how === "nb") v = this.nbHTML(props[k]);
      else if (how === "streets") v = this.streetList(props[k]);
      else v = fmt(props[k], how);
      h += `<div><b>${lab}:</b> ${v}</div>`;
    }
    return h || "<div>(no attributes)</div>";
  },

  /* A corridor's "A - B - C" or "A; B; C" of street names, with the Chinese
   * names of the same streets on a second line. */
  streetList(text) {
    const sep = String(text).includes(";") ? "; " : " - ";
    const parts = String(text).split(sep);
    const zh = parts.map(p => this.zhOf.get(p.trim())).filter(Boolean);
    return esc(text) + (zh.length
      ? `<br><span class="zh" lang="zh-Hant-HK">${esc(zh.join(sep === "; " ? "；" : " - "))}</span>` : "");
  },

  /* --------------------------------------------------------------- input */
  onMapClick(e) {
    if (this.warpOn) return;            // the warped page is not a place to click
    const { lat, lng } = e.latlng;
    const bit = this.graph.modeBit(this.state.mode);
    // only junctions the mode can route from: every one of them is on a
    // landmass, so a pick can always be routed somewhere
    const node = this.nodeGrid.nearest(lng, lat,
      i => (this.graph.nodeFlags[i] & bit) !== 0);
    if (node < 0) return;

    // a click well away from any street is an inspection, not a routing pick
    const dLon = (this.graph.nodeLon(node) - lng) * Math.cos(lat * Math.PI / 180);
    const dLat = this.graph.nodeLat(node) - lat;
    const metres = Math.hypot(dLon, dLat) * 111320;
    if (metres > 260) { this.inspect(e.latlng); return; }

    // the point is no longer the neighbourhood the menu names
    if (this.state.picking === "src") {
      this.state.src = node; this.state.picking = "dst";
      document.getElementById("o").value = "";
    } else {
      this.state.dst = node; this.state.picking = "src";
      document.getElementById("d").value = "";
    }
    this.syncPickButtons();
    this.recompute();
  },

  inspect(latlng) {
    const row = this.edgeGrid.nearest(latlng.lng, latlng.lat);
    if (row < 0) return;
    const info = this.geom.edgeInfo(row);
    let html = `<div><b>Street:</b> ${this.streetHTML(info.name)}</div>`
      + `<div><b>Gradient class:</b> ${BUCKETS[info.bucket].label}</div>`
      + `<div><b>Max gradient:</b> ${fmt(info.max_grade, "pct")}</div>`
      + `<div><b>Mean gradient:</b> ${fmt(info.avg_grade, "pct")}</div>`
      + `<div><b>Climb per km:</b> ${fmt(info.gain_per_km, "mkm")}</div>`
      + `<div><b>Length:</b> ${fmt(info.length_m, "m0")}</div>`
      + `<div><b>Road class:</b> ${esc(info.cls)}</div>`;
    const basin = this.edgeBasin && this.edgeBasin[row]
      ? (this.DATA.layers.basins.rows || [])[this.edgeBasin[row] - 1] : null;
    if (basin) {
      html += `<div><b>Lowland basin:</b> ${esc(basin.basin_label || "unnamed")}`
        + ` (${fmt(basin.length_km, "km")} of street ${this.basinWord})</div>`;
    }
    const land = (this.meta.components && this.meta.components.walk || [])[this.edgeLand[row] - 1];
    if (land) html += `<div><b>Landmass:</b> ${esc(land)}</div>`;
    L.popup({ maxWidth: 300 }).setLatLng(latlng).setContent(html).openOn(this.map);
  },

  /* The trip the page opens on: the first guided example, and failing that
   * the first two neighbourhoods that share a landmass. */
  setDefaultPair() {
    const mode = this.state.mode, g = this.graph;
    let pair = null;
    for (const ex of (this.DATA.examples || [])) {
      const a = this.pointNode(ex.o, mode), b = this.pointNode(ex.d, mode);
      if (a >= 0 && b >= 0 && g.comp(a, mode) === g.comp(b, mode)) { pair = [ex.o, ex.d]; break; }
    }
    if (!pair) {
      const first = new Map();
      for (const name of this.DATA.neighborhood_names) {
        const node = this.pointNode(name, mode);
        if (node < 0) continue;
        const c = g.comp(node, mode);
        if (first.has(c)) { pair = [first.get(c), name]; break; }
        first.set(c, name);
      }
    }
    if (!pair) return;
    this.state.src = this.pointNode(pair[0], mode);
    this.state.dst = this.pointNode(pair[1], mode);
    document.getElementById("o").value = pair[0];
    document.getElementById("d").value = pair[1];
  },

  weights() {
    const base = this.meta.profiles[this.state.profile];
    if (!this.state.custom) return base;
    return Object.assign({}, base, {
      alpha: this.state.alpha, beta: this.state.beta, gamma: this.state.gamma,
    });
  },

  /* ----------------------------------------------------------- recompute */
  recompute() {
    const s = this.state, g = this.graph;
    const out = document.getElementById("res");
    const clearLines = () => {
      this.lastRoute = null;
      this.routeLine.setLatLngs([]); this.routeHalo.setLatLngs([]);
      this.cmpLine.setLatLngs([]);
      this.drawProfile(null);
    };
    if (s.src == null || s.dst == null || s.src < 0 || s.dst < 0 || s.src === s.dst) {
      out.innerHTML = '<p class="sub">Click the map to set an origin, then a '
        + 'destination. Or pick neighbourhoods from the menus.</p>';
      clearLines(); this.marks.clearLayers();
      this.followWarp();
      return;
    }
    this.drawMarks();

    // Two points on different landmasses have no route, and the search that
    // would find that out visits a whole landmass first. The components
    // are known, so say so at once, and say which two they are.
    const ca = g.comp(s.src, s.mode), cb = g.comp(s.dst, s.mode);
    if (!ca || !cb || ca !== cb) {
      out.innerHTML = this.noRouteHTML();
      clearLines();
      this.followWarp();
      return;
    }

    const t0 = performance.now();
    const w = this.weights();
    const r = g.route(s.src, s.dst, s.mode, w);
    const sh = g.route(s.src, s.dst, s.mode, this.meta.profiles.shortest);
    const ms = performance.now() - t0;

    if (!r) {
      out.innerHTML = '<p class="sub">No route found between those two points '
        + 'in this travel mode.</p>';
      clearLines();
      this.followWarp();
      return;
    }

    const sum = g.summarise(r.arcs);
    const shSum = sh ? g.summarise(sh.arcs) : null;
    this.lastRoute = { line: g.geometry(r.arcs, this.geom),
                       cmp: sh && this.state.profile !== "shortest" && !this.state.custom
                         ? g.geometry(sh.arcs, this.geom) : [] };
    this.renderResult(sum, shSum, ms, r.settled);
    this.drawProfile(sum);
    if (this.followWarp()) return;      // the lines are drawn when the warp is ready
    this.drawRouteLines();
  },

  /* What to call an end of the trip in a sentence: its neighbourhood, if it
   * was chosen from the menu. */
  endName(which) {
    const name = document.getElementById(which === "src" ? "o" : "d").value;
    return name ? ` (${this.nbHTML(name)})` : "";
  },

  /* Central District to Tsim Sha Tsui: there is nothing to draw, and an
   * empty panel would look like a fault. Name the two landmasses.
   *
   * Two points on one landmass can be apart as well, for a cyclist: the
   * network open to bicycles is in pieces (in the whole-territory build
   * Discovery Bay's 430 or so junctions are no part of the 2,700 of the
   * rest of Lantau's). To say then that both are "on Lantau" and to choose
   * two points on the same landmass would be no help, so that case has its
   * own words.
   *
   * A third case is a piece the build could not name: a small network the
   * data joins to nothing else (a few of the twenty or so walking
   * landmasses of the whole-territory build, some 40 to 120 junctions
   * each), or in a partial build a district the window cut off. That need
   * not be one of Hong Kong's landmasses, and the sentence about tunnels
   * and ferries is not said of it. */
  noRouteHTML() {
    const s = this.state, bike = s.mode === "bike";
    const where = node => {
      const name = this.landName(node, s.mode);
      return name ? `on <b>${esc(name)}</b>`
        : "on a separate part of the street network";
    };
    const head = `<div class="warn"><b>No ${bike ? "cycling" : "walking"} route between these two points.</b> `;
    const named = !!(this.landName(s.src, s.mode) && this.landName(s.dst, s.mode));
    const land = this.nodeLand[s.src];
    if (land && land === this.nodeLand[s.dst]) {
      const name = (this.meta.components && this.meta.components.walk || [])[land - 1];
      return head
        + `The origin${this.endName("src")} and the destination${this.endName("dst")} are `
        + `${name ? `both on <b>${esc(name)}</b>` : "on the same landmass"}, but on separate pieces of its `
        + `${bike ? "cycling" : "walking"} network: no street or path open to `
        + `${bike ? "bicycles" : "walkers"} joins the two`
        + `${bike ? " (footpaths, steps and road tunnels are closed to them). Walking, there may be a way" : ""}. `
        + `Choose two points on the same piece.</div>`;
    }
    return head
      + `The origin${this.endName("src")} is ${where(s.src)}; `
      + `the destination${this.endName("dst")} is ${where(s.dst)}. `
      + (named
        ? `Nothing joins the two ${bike ? "by bicycle" : "on foot"}: Hong Kong's landmasses are `
          + `linked only by road tunnels, bridges without a footway, railways and ferries, `
          + `and this map routes along streets and paths alone. `
          + `Choose two points on the same landmass.</div>`
        : `No street or path in the data joins the two ${bike ? "by bicycle" : "on foot"}, `
          + `and this map routes along streets and paths alone. `
          + `Choose two points on the same part of the network.</div>`);
  },

  drawMarks() {
    const s = this.state, g = this.graph;
    this.marks.clearLayers();
    const mk = (node, colour, label) => L.circleMarker(
      this.place(g.nodeLat(node), g.nodeLon(node), this.onWarpedLand(node)),
      { radius: 7, color: "#04121a", weight: 2, fillColor: colour, fillOpacity: 1 })
      .bindPopup(`<b>${label}</b><br>${g.nodeZ(node).toFixed(1)} m above Principal Datum`
        + (this.landName(node, s.mode) ? `<br>${esc(this.landName(node, s.mode))}` : ""));
    mk(s.src, "#7fcdbb", "Origin").addTo(this.marks);
    mk(s.dst, "#e3492e", "Destination").addTo(this.marks);
  },

  /* a lat/lon as currently displayed: warped and morphed, or as it is.
   * ``on`` is false for a point on a landmass the warp leaves alone. */
  place(lat, lon, on = true) {
    if (!on || !this.warpOn || !this.warp || this.warp.small) return [lat, lon];
    const [wlon, wlat] = this.warp.transform(lon, lat);
    const t = this.warpT;
    return [lat + (wlat - lat) * t, lon + (wlon - lon) * t];
  },

  /* is this junction on the landmass the current warp redraws? */
  onWarpedLand(node) {
    return !!this.warp && node != null && node >= 0 && this.nodeLand[node] === this.warp.land;
  },

  drawRouteLines() {
    if (this.state.src != null && this.state.dst != null
      && this.state.src >= 0 && this.state.dst >= 0) this.drawMarks();
    if (!this.lastRoute) return;
    // a route lies on one landmass: its origin's
    const on = this.onWarpedLand(this.state.src);
    const w = pts => pts.map(ll => this.place(ll[0], ll[1], on));
    this.routeHalo.setLatLngs(w(this.lastRoute.line));
    this.routeLine.setLatLngs(w(this.lastRoute.line));
    this.cmpLine.setLatLngs(w(this.lastRoute.cmp));
  },

  renderResult(sum, sh, ms, settled) {
    const km = sum.distance_m / 1000;
    let h = `<div class="big">${km.toFixed(2)} km &middot; `
      + `${Math.round(sum.elev_gain_m).toLocaleString()} m climb</div><table>`;
    h += `<tr><td class="k">Distance</td><td class="v">${km.toFixed(2)} km</td></tr>`;
    h += `<tr><td class="k">Elevation gain</td><td class="v">`
      + `${sum.elev_gain_m.toFixed(0)} m</td></tr>`;
    h += `<tr><td class="k">Elevation loss</td><td class="v">`
      + `${sum.elev_loss_m.toFixed(0)} m</td></tr>`;
    h += `<tr><td class="k">Climb per km</td><td class="v">`
      + `${(km > 0 ? sum.elev_gain_m / km : 0).toFixed(1)} m/km</td></tr>`;
    // the route's steepest pitch, the figure the pair tables call max_grade
    // (routing.arc_pitch): an arc under 15 m is judged on its average climb
    h += `<tr><td class="k">Steepest climb</td><td class="v">`
      + `${(sum.steepest * 100).toFixed(1)}%</td></tr>`;
    h += `<tr><td class="k">Mean gradient</td><td class="v">`
      + `${(sum.avg_abs_grade * 100).toFixed(1)}%</td></tr>`;
    this.meta.thresholds.forEach((t, i) => {
      h += `<tr><td class="k">Distance climbing over ${t}%</td><td class="v">`
        + `${Math.round(sum.thresholds[i]).toLocaleString()} m</td></tr>`;
    });
    h += "</table>";
    if (sh && sh.distance_m > 0 && (this.state.profile !== "shortest" || this.state.custom)) {
      const dKm = (sum.distance_m - sh.distance_m) / 1000;
      const dM = sh.elev_gain_m - sum.elev_gain_m;
      h += `<div class="cmp">
        <div class="card"><div class="t">Extra distance</div><div class="n">`
        + `${dKm >= 0 ? "+" : ""}${dKm.toFixed(2)} km `
        + `(${(100 * (sum.distance_m / sh.distance_m - 1)).toFixed(0)}%)</div></div>
        <div class="card"><div class="t">Climbing saved</div>`
        + `<div class="n ${dM > 0 ? "good" : "bad"}">`
        + `${dM >= 0 ? "" : "+"}${Math.abs(Math.round(dM))} m</div></div></div>`;
      h += `<div class="note">Shortest route, shown dashed: `
        + `${(sh.distance_m / 1000).toFixed(2)} km with `
        + `${Math.round(sh.elev_gain_m)} m of climbing, steepest `
        + `${(sh.steepest * 100).toFixed(0)}%.</div>`;
    }
    const land = this.landName(this.state.src, this.state.mode);
    h += `<div class="note">${land ? `On ${esc(land)}. ` : ""}`
      + `Routed in the browser in ${ms.toFixed(0)} ms `
      + `(${settled.toLocaleString()} junctions settled, ${sum.n_edges} street `
      + `segments).</div>`;
    document.getElementById("res").innerHTML = h;
  },

  drawProfile(sum) {
    const cv = document.getElementById("prof");
    const ctx = cv.getContext("2d");
    const W = cv.width, H = cv.height;
    ctx.clearRect(0, 0, W, H);
    const note = document.getElementById("profnote");
    if (!sum || !sum.profile.d.length) { note.textContent = ""; return; }
    const d = sum.profile.d, e = sum.profile.z, n = d.length;
    // a route across the New Territories has thousands of junctions, more
    // than a spread argument list will carry
    let emin = Infinity, emax = -Infinity;
    for (let i = 0; i < n; i++) { if (e[i] < emin) emin = e[i]; if (e[i] > emax) emax = e[i]; }
    const span = Math.max(emax - emin, 8);
    const padL = 56, padR = 16, padT = 14, padB = 30;
    const x = i => padL + (d[i] / d[n - 1]) * (W - padL - padR);
    const y = v => padT + (1 - (v - emin) / span) * (H - padT - padB);

    for (let i = 1; i < n; i++) {
      const dz = e[i] - e[i - 1], dd = Math.max(d[i] - d[i - 1], 1e-6);
      const g = Math.abs(dz / dd);
      ctx.fillStyle = g < 0.03 ? "#2c7bb6" : g < 0.05 ? "#7fcdbb" : g < 0.08 ? "#fed976"
        : g < 0.10 ? "#fd8d3c" : g < 0.15 ? "#e3492e" : "#7f1d1d";
      ctx.globalAlpha = 0.8;
      ctx.beginPath();
      ctx.moveTo(x(i - 1), y(e[i - 1])); ctx.lineTo(x(i), y(e[i]));
      ctx.lineTo(x(i), H - padB); ctx.lineTo(x(i - 1), H - padB);
      ctx.closePath(); ctx.fill();
    }
    ctx.globalAlpha = 1;
    ctx.strokeStyle = "#e9eef3"; ctx.lineWidth = 1.6;
    ctx.beginPath();
    for (let i = 0; i < n; i++) { i ? ctx.lineTo(x(i), y(e[i])) : ctx.moveTo(x(i), y(e[i])); }
    ctx.stroke();
    ctx.strokeStyle = "#39434e"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(padL, H - padB); ctx.lineTo(W - padR, H - padB); ctx.stroke();
    ctx.fillStyle = "#93a1af"; ctx.font = "11px sans-serif"; ctx.textAlign = "right";
    ctx.fillText(Math.round(emax) + " m", padL - 5, padT + 9);
    ctx.fillText(Math.round(emin) + " m", padL - 5, H - padB - 1);
    ctx.fillText((d[n - 1] / 1000).toFixed(2) + " km", W - padR, H - 9);
    ctx.textAlign = "left"; ctx.fillText("0", padL, H - 9);
    note.textContent = `Elevation ${Math.round(emin)}–${Math.round(emax)} mPD, `
      + `sampled at every junction. Shading shows gradient.`;
  },
};

/* ------------------------------------------------------------ UI wiring */
Object.assign(App, {
  buildUI() {
    const oSel = document.getElementById("o"), dSel = document.getElementById("d");
    this.fillSelects();
    const setFrom = (sel, which) => {
      const name = sel.value;
      if (!name) return;
      const node = this.pointNode(name, this.state.mode);
      if (node < 0) return;
      this.state[which] = node;
      this.recompute();
    };
    oSel.onchange = () => setFrom(oSel, "src");
    dSel.onchange = () => setFrom(dSel, "dst");

    document.getElementById("swap").onclick = () => {
      const t = this.state.src; this.state.src = this.state.dst; this.state.dst = t;
      const to = oSel.value; oSel.value = dSel.value; dSel.value = to;
      this.recompute();
    };
    document.getElementById("clear").onclick = () => {
      this.state.src = null; this.state.dst = null; this.state.picking = "src";
      oSel.value = ""; dSel.value = "";
      this.syncPickButtons(); this.recompute();
    };

    document.querySelectorAll("#pickrow button").forEach(b => {
      b.onclick = () => { this.state.picking = b.dataset.pick; this.syncPickButtons(); };
    });
    this.syncPickButtons();

    const seg = (id, field, after) => {
      const box = document.getElementById(id);
      box.querySelectorAll("button").forEach(b => {
        b.onclick = () => {
          box.querySelectorAll("button").forEach(x => x.setAttribute("aria-pressed", "false"));
          b.setAttribute("aria-pressed", "true");
          this.state[field] = b.dataset.v;
          if (after) after();
          this.recompute();
        };
      });
    };
    seg("mode", "mode", () => this.modeChanged());
    seg("pref", "profile", () => { this.state.custom = false; this.syncSliders(); });

    // weight sliders
    const defs = [
      ["alpha", "Climbing weight α", 0, 250, 1,
       "equivalent metres of detour accepted per metre climbed"],
      ["beta", "Steep-grade penalty β", 0, 30, 0.5,
       "multiplier on the cumulative grade-threshold penalties"],
      ["gamma", "Extreme-grade penalty γ", 0, 20, 0.5,
       "multiplier on the penalty above the highest threshold"],
    ];
    const box = document.getElementById("sliders");
    for (const [key, label, min, max, step, help] of defs) {
      const wrap = document.createElement("div");
      wrap.className = "slider";
      wrap.innerHTML = `<div class="row"><span>${label}</span><b id="lab_${key}"></b></div>`
        + `<input type="range" id="sl_${key}" min="${min}" max="${max}" step="${step}">`;
      wrap.title = help;
      box.appendChild(wrap);
      const input = wrap.querySelector("input");
      input.oninput = () => {
        this.state[key] = parseFloat(input.value);
        this.state.custom = true;
        document.getElementById("lab_" + key).textContent = input.value;
        document.getElementById("customnote").style.display = "block";
        this.recompute();
      };
    }
    document.getElementById("resetw").onclick = () => {
      this.state.custom = false; this.syncSliders(); this.recompute();
    };
    this.syncSliders();

    // guided examples, so the map demonstrates the findings unprompted
    const ebox = document.getElementById("examples");
    for (const ex of (this.DATA.examples || [])) {
      const btn = document.createElement("button");
      btn.className = "act";
      btn.innerHTML = `<b>${esc(ex.o)} &rarr; ${esc(ex.d)}</b><br><span class="exnote">`
        + `${esc(ex.note)}</span>`;
      btn.onclick = () => {
        this.state.mode = ex.mode || "walk";
        this.state.profile = ex.profile || "min_climb";
        this.state.custom = false;
        document.querySelectorAll("#mode button").forEach(b =>
          b.setAttribute("aria-pressed", String(b.dataset.v === this.state.mode)));
        document.querySelectorAll("#pref button").forEach(b =>
          b.setAttribute("aria-pressed", String(b.dataset.v === this.state.profile)));
        this.fillSelects();
        const src = this.pointNode(ex.o, this.state.mode);
        const dst = this.pointNode(ex.d, this.state.mode);
        this.state.src = src >= 0 ? src : null;
        this.state.dst = dst >= 0 ? dst : null;
        oSel.value = ex.o; dSel.value = ex.d;
        this.syncSliders();
        this.recompute();
      };
      ebox.appendChild(btn);
    }

    // layers
    const lbox = document.getElementById("layers");
    for (const def of this.layerDefs) {
      const wrap = document.createElement("label");
      wrap.innerHTML = `<input type="checkbox" id="chk_${def.key}" `
        + `${def.on ? "checked" : ""}> ${esc(def.label)}`;
      lbox.appendChild(wrap);
      wrap.querySelector("input").onchange = ev => {
        const lyr = this.overlays[def.key] || this.overlay(def.key);
        if (!lyr) return;
        if (ev.target.checked) {
          // the outlines and corridors a warp shows are its own copies
          if (this.warpOn && (def.key === "neighborhoods" || def.key === "corridors")) {
            this.redrawWarpedOverlays();
          } else { lyr.addTo(this.map); this.raise(); }
        } else {
          this.map.removeLayer(lyr);
          if (this.warpOn) this.redrawWarpedOverlays();
        }
      };
    }
    const kbox = document.getElementById("key");
    BUCKETS.forEach(b => {
      const el = document.createElement("div");
      el.className = "key";
      el.innerHTML = `<i style="background:${b.colour}"></i> ${b.label}`;
      kbox.appendChild(el);
    });
    this.raise();
  },

  /* The two neighbourhood menus, grouped by landmass for the current mode,
   * so that what can be routed to what is visible before anything is
   * chosen. A cyclist's landmasses are not the walker's, hence per mode. */
  fillSelects() {
    const g = this.graph, mode = this.state.mode;
    const groups = new Map();
    for (const name of this.DATA.neighborhood_names) {
      const node = this.pointNode(name, mode);
      const c = node >= 0 ? g.comp(node, mode) : 0;
      if (!groups.has(c)) groups.set(c, []);
      groups.get(c).push(name);
    }
    // landmasses largest first, as the build ranked them; "none" last
    const order = Array.from(groups.keys()).sort((a, b) => (a || 1e9) - (b || 1e9));
    const names = (this.meta.components || {})[mode === "bike" ? "bike" : "walk"] || [];
    for (const id of ["o", "d"]) {
      const sel = document.getElementById(id), keep = sel.value;
      sel.textContent = "";
      sel.appendChild(new Option("(map point)", ""));
      // A piece of the cycling network with no name of its own is called
      // after the island it stands on (Discovery Bay is "Lantau"), so two
      // groups can have one name: the smaller says that it is a piece
      // apart. A third piece of one island, or a second with no name at
      // all, is numbered, so that no two groups are headed alike.
      const used = new Map();
      for (const c of order) {
        const og = document.createElement("optgroup");
        const name = c ? names[c - 1] : "";
        const before = used.get(name) || 0;
        used.set(name, before + 1);
        og.label = !c ? `No ${mode === "bike" ? "cycling" : "walking"} network here`
          : (!name ? "A separate part of the network"
            : before ? `${name}: a separate part of the network` : name)
            + (before > (name ? 1 : 0) ? ` (${name ? before : before + 1})` : "");
        for (const name of groups.get(c)) {
          const opt = new Option(this.nbText(name), name);
          if (!c) opt.disabled = true;
          og.appendChild(opt);
        }
        sel.appendChild(og);
      }
      sel.value = keep;
      if (sel.value !== keep) sel.value = "";
    }
  },

  /* A mode change moves the ends of the trip: a neighbourhood's access
   * point is chosen per mode, and a picked junction may not be rideable
   * (steps are not, and most footpaths are closed to bicycles). */
  modeChanged() {
    const g = this.graph, mode = this.state.mode, bit = g.modeBit(mode);
    this.fillSelects();
    for (const [key, id] of [["src", "o"], ["dst", "d"]]) {
      const name = document.getElementById(id).value;
      const nd = this.state[key];
      if (name) {
        const node = this.pointNode(name, mode);
        if (node >= 0) { this.state[key] = node; continue; }
        document.getElementById(id).value = "";
      }
      if (nd != null && nd >= 0 && (g.nodeFlags[nd] & bit) === 0) {
        this.state[key] = this.nodeGrid.nearest(
          g.nodeLon(nd), g.nodeLat(nd), i => (g.nodeFlags[i] & bit) !== 0);
      }
    }
  },

  syncPickButtons() {
    document.querySelectorAll("#pickrow button").forEach(b => {
      b.setAttribute("aria-pressed", String(b.dataset.pick === this.state.picking));
    });
    document.getElementById("pickhint").textContent =
      this.state.picking === "src"
        ? "Click the map to set the ORIGIN."
        : "Click the map to set the DESTINATION.";
  },

  syncSliders() {
    const base = this.meta.profiles[this.state.profile];
    if (!this.state.custom) {
      this.state.alpha = base.alpha; this.state.beta = base.beta;
      this.state.gamma = base.gamma;
      document.getElementById("customnote").style.display = "none";
    }
    for (const key of ["alpha", "beta", "gamma"]) {
      const sl = document.getElementById("sl_" + key);
      if (!sl) continue;
      sl.value = this.state[key];
      document.getElementById("lab_" + key).textContent = String(this.state[key]);
    }
  },

  raise() {
    [this.cmpLine, this.routeHalo, this.routeLine].forEach(l => l.bringToFront());
    this.marks.eachLayer(l => l.bringToFront());
  },
});

// exposed so the page can be driven from the console and from tests
window.App = App;

window.addEventListener("DOMContentLoaded", () => {
  const res = document.getElementById("res");
  res.innerHTML = '<p class="sub">Unpacking the street graph…</p>';
  App.start(window.DATA).catch(err => {
    res.innerHTML = '<p class="sub">Could not start: ' + esc(err.message || err)
      + "</p>";
    throw err;
  });
});

/* ------------------------------------------------------------ the warp */
/* One landmass is redrawn at a time (warp.js says why): the one the origin
 * is on, or with no origin the one in the middle of the map. Its streets,
 * outlines, corridors and the route move; everything else stays where
 * geography puts it and is greyed. */
Object.assign(App, {
  warpOn: false, warpT: 1.0, warpLambda: 1.0, warp: null, warpCache: {}, warpJob: 0,

  warpWeights() {
    const base = this.meta.profiles.balanced;
    const lam = this.warpLambda;
    return Object.assign({}, base, {
      alpha: base.alpha * lam, beta: base.beta * lam, gamma: base.gamma * lam,
      use_class_multiplier: lam > 0,
    });
  },

  /* The junction whose landmass is to be warped. */
  warpSeed() {
    const s = this.state, g = this.graph;
    if (s.src != null && s.src >= 0 && g.comp(s.src, s.mode)) return s.src;
    const c = this.map.getCenter(), bit = g.modeBit(s.mode);
    return this.nodeGrid.nearest(c.lng, c.lat, i => (g.nodeFlags[i] & bit) !== 0);
  },

  /* The component of the mode's graph that the warp of a junction's
   * landmass is fitted to: the mode's main network on that landmass, which
   * for a walker is the landmass itself. A cyclist's network has fragments
   * as well (Discovery Bay on Lantau, Ap Lei Chau off Hong Kong Island). A
   * deformation fitted to the few hundred junctions of a fragment says
   * nothing about the rest of the island, and carrying the whole island
   * along on it throws the far side kilometres across the page (from
   * Discovery Bay, whose 89 anchors moved 480 m on average, the streets of
   * Lantau moved 4 km on average and up to 9 km); so from an origin in a
   * fragment it is still the landmass's main network that is laid out, and
   * the fragment rides along as the footpaths do. */
  warpComp(seed, mode) {
    if (seed < 0) return 0;
    const main = this.landComp[mode === "bike" ? "bike" : "walk"][this.nodeLand[seed]];
    return main || this.graph.comp(seed, mode);
  },

  warpKey() {
    const seed = this.warpSeed();
    return `${this.state.mode}|${this.warpLambda}|${this.warpComp(seed, this.state.mode)}`;
  },

  /* Build (or fetch) the warp for the current mode, weight and landmass.
   * The cost matrix takes seconds on the larger landmasses, so the build
   * yields to the page as it goes and reports how far it has got. */
  async ensureWarp(onProgress) {
    const key = this.warpKey(), mode = this.state.mode;
    if (!this.warpCache[key]) {
      const seed = this.warpSeed();
      const w = await Warp.build(this, mode, this.warpWeights(),
        this.warpComp(seed, mode), onProgress);
      w.key = key; w.mode = mode;
      w.land = seed < 0 ? 0 : this.nodeLand[seed];
      // the landmass by its own name, whatever the mode's piece of it is called
      w.name = seed < 0 ? "" : ((this.meta.components && this.meta.components.walk || [])[w.land - 1]
        || this.landName(seed, mode));
      if (!w.small) {
        w.merc = Warp.warpMercator(this.geom, this.edgeLand, w.land, w.transform);
        w.dim = new Uint8Array(this.geom.nEdges);
        for (let i = 0; i < w.dim.length; i++) w.dim[i] = this.edgeLand[i] === w.land ? 0 : 1;
      }
      // each warp holds a copy of every vertex, so only a few are kept
      const keys = Object.keys(this.warpCache);
      if (keys.length >= 6) delete this.warpCache[keys[0]];
      this.warpCache[key] = w;
    }
    return this.warpCache[key];
  },

  /* lerp between the real and the warped city */
  warpedMerc() {
    const w = this.warp, t = this.warpT, real = this.geom.mercator();
    if (t >= 1) return w.merc;
    const n = real.mx.length;
    if (!this._lerp || this._lerp.mx.length !== n) {
      this._lerp = { mx: new Float32Array(n), my: new Float32Array(n) };
    }
    const { mx, my } = this._lerp;
    for (let k = 0; k < n; k++) {
      mx[k] = real.mx[k] + (w.merc.mx[k] - real.mx[k]) * t;
      my[k] = real.my[k] + (w.merc.my[k] - real.my[k]) * t;
    }
    return this._lerp;
  },

  /* With the warp on, a change of origin, mode or weight may call for a
   * different warp. Returns true when one is being built (the view is
   * redrawn when it is ready). */
  followWarp() {
    if (!this.warpOn) return false;
    if (this.warp && this.warp.key === this.warpKey() && !this._warping) return false;
    this.applyWarpView();
    return true;
  },

  applyWarpView() {
    const on = this.warpOn, job = ++this.warpJob;
    document.getElementById("warpstate").hidden = !on;
    const busy = document.getElementById("busy");
    if (on) {
      busy.textContent = "Warping the city…";
      busy.classList.add("on");
      this._warping = true;
      // let the busy notice paint before the build starts
      setTimeout(async () => {
        try {
          const w = await this.ensureWarp(frac => {
            if (job === this.warpJob) busy.textContent = `Warping the city… ${Math.round(frac * 100)}%`;
          });
          if (job !== this.warpJob || !this.warpOn) return;   // overtaken
          this.warp = w;
          if (w.small) this.network.setAlt(null);
          else this.network.setAlt(this.warpedMerc(), w.dim);
          this.fitWarp();
          this.network._redraw();
          for (const key of ["passes", "barriers", "basins", "bike_network", "low_stress"]) {
            if (this.overlays[key] && this.map.hasLayer(this.overlays[key])) {
              this.map.removeLayer(this.overlays[key]);
              const chk = document.getElementById("chk_" + key);
              if (chk) chk.checked = false;
            }
          }
          for (const key of ["neighborhoods", "corridors"]) {
            if (this.overlays[key] && this.map.hasLayer(this.overlays[key])) {
              this.map.removeLayer(this.overlays[key]);
            }
          }
          this.redrawWarpedOverlays();
          this.drawRouteLines();
          this.renderWarpReadout();
        } finally {
          if (job === this.warpJob) { busy.classList.remove("on"); this._warping = false; }
        }
      }, 30);
    } else {
      busy.classList.remove("on");
      this._warping = false;
      this.network.setAlt(null);
      this.network._redraw();
      if (this.warpLayers) { this.warpLayers.forEach(l => this.map.removeLayer(l)); this.warpLayers = null; }
      this.drawWarpLabels();
      for (const key of ["neighborhoods", "corridors"]) {
        const chk = document.getElementById("chk_" + key);
        if (this.overlays[key] && chk && chk.checked) this.overlays[key].addTo(this.map);
      }
      this.drawRouteLines();
      this.raise();
    }
    this.syncPickButtons();
  },

  redrawWarpedOverlays() {
    if (this.warpLayers) this.warpLayers.forEach(l => this.map.removeLayer(l));
    const w = this.warp, mode = this.state.mode;
    const layers = [];
    if (!w) { this.warpLayers = layers; return; }
    const move = (lon, lat) => { const p = this.place(lat, lon); return [p[1], p[0]]; };
    const shown = key => {
      const chk = document.getElementById("chk_" + key);
      return this.DATA.layers[key] && (!chk || chk.checked);
    };
    // a neighbourhood goes with the landmass its access point is on; a
    // corridor with the one its middle lies on
    const nbOn = f => this.onWarpedLand(this.pointNode(f.properties.neighborhood, mode));
    const lineOn = f => {
      let c = f.geometry.coordinates;
      while (Array.isArray(c[0][0])) c = c[Math.floor(c.length / 2)];
      const mid = c[Math.floor(c.length / 2)];
      return this.onWarpedLand(this.nodeGrid.nearest(mid[0], mid[1], i => this.nodeLand[i] > 0));
    };
    if (shown("neighborhoods")) {
      layers.push(L.geoJSON(Warp.warpGeoJSON(this.DATA.layers.neighborhoods, move, nbOn), {
        style: f => ({ color: "#aab6c2", weight: 1.3, opacity: f.off ? 0.3 : 0.85,
                       fill: false, dashArray: "4,3" }),
        onEachFeature: (f, l) => l.bindPopup(this.popupHTML("neighborhoods", f.properties)),
      }).addTo(this.map));
    }
    if (shown("corridors")) {
      layers.push(L.geoJSON(Warp.warpGeoJSON(this.DATA.layers.corridors, move, lineOn), {
        style: f => ({ color: f.properties.mode === "bike" ? "#9be7ff" : "#2c7bb6",
                       weight: 3.6, opacity: f.off ? 0.25 : 0.9, lineCap: "round" }),
        onEachFeature: (f, l) => l.bindPopup(this.popupHTML("corridors", f.properties)),
      }).addTo(this.map));
    }
    this.warpLayers = layers;
    this.drawWarpLabels();
    this.raise();
  },

  /* Neighbourhood names on the warped landmass, so the deformation can be
   * read. The flat districts of a landmass are close together in cost and
   * are drawn close together, the north shore of Hong Kong Island most of
   * all, so a name that would sit on top of one already placed is left out
   * until the view is zoomed far enough to part them. */
  drawWarpLabels() {
    if (this.warpLabels) { this.map.removeLayer(this.warpLabels); this.warpLabels = null; }
    if (!this.warpOn || !this.warp) return;
    const mode = this.state.mode, pts = this.DATA.points[mode] || {};
    const labels = L.layerGroup(), placed = [];
    for (const name of Object.keys(pts)) {
      if (!this.onWarpedLand(this.pointNode(name, mode))) continue;
      const [lon, lat] = pts[name];
      const ll = this.place(lat, lon);
      const p = this.map.latLngToContainerPoint(ll);
      const w = 6.6 * name.length + 10, h = 16;
      const box = [p.x - w / 2, p.y - h / 2, p.x + w / 2, p.y + h / 2];
      if (placed.some(b => box[0] < b[2] && box[2] > b[0] && box[1] < b[3] && box[3] > b[1])) continue;
      placed.push(box);
      const el = document.createElement("span");
      el.textContent = name;
      L.marker(ll, {
        icon: L.divIcon({ className: "nblabel", html: el, iconSize: null }),
        interactive: false,
      }).addTo(labels);
    }
    this.warpLabels = labels.addTo(this.map);
  },

  /* Bring the whole redrawn landmass into view. Upstream zooms to the
   * route; here a route is often one corner of an island, and the
   * deformation only reads at the scale of the landmass. */
  fitWarp() {
    const w = this.warp, t = this.warpT;
    if (!w || w.small || !w.anchorLL.length) return;
    const pts = w.anchorLL.map(a => [a.from[0] + (a.to[0] - a.from[0]) * t,
                                     a.from[1] + (a.to[1] - a.from[1]) * t]);
    this.map.fitBounds(L.latLngBounds(pts).pad(0.04), { animate: false });
  },

  renderWarpReadout() {
    const w = this.warp;
    if (!w) return;
    const bike = w.mode === "bike";
    const what = `<b>${esc(w.name || "This part of the network")}</b>, ${bike ? "by bicycle" : "on foot"}`;
    if (w.small) {
      document.getElementById("warpinfo").innerHTML =
        `<div>${what}: too small to warp (${w.anchors} anchor junctions). `
        + `Set an origin on a larger landmass.</div>`;
      return;
    }
    const a = this.meta.profiles.balanced.alpha * this.warpLambda;
    document.getElementById("warpinfo").innerHTML =
      `<div>${what}: <b>${a.toFixed(0)} m</b> of ${bike ? "riding" : "walking"} per metre of climb &middot; `
      + `${w.anchors} anchors about ${Math.round(w.cellM / 10) * 10} m apart &middot; `
      + `stress ${(100 * w.stress).toFixed(1)}% &middot; `
      + `mean shift ${Math.round(w.meanShiftM).toLocaleString()} m &middot; `
      + `${(w.ms / 1000).toFixed(1)} s</div>`;
  },

  bindWarpUI() {
    const toggle = document.getElementById("warptoggle");
    toggle.onclick = () => {
      this.warpOn = !this.warpOn;
      toggle.setAttribute("aria-pressed", String(this.warpOn));
      toggle.textContent = this.warpOn ? "Show the real city" : "Show the warped city";
      this.applyWarpView();
    };
    const lam = document.getElementById("sl_lambda");
    const lamLab = document.getElementById("lab_lambda");
    lam.value = this.warpLambda; lamLab.textContent = this.warpLambda.toFixed(1);
    lam.oninput = () => { lamLab.textContent = (+lam.value).toFixed(1); };
    lam.onchange = () => {
      this.warpLambda = parseFloat(lam.value);
      if (this.warpOn) this.applyWarpView();
    };
    this.map.on("zoomend", () => { if (this.warpOn && this.warp && !this._warping) this.drawWarpLabels(); });
    const morph = document.getElementById("sl_morph");
    const morphLab = document.getElementById("lab_morph");
    morph.value = this.warpT; morphLab.textContent = Math.round(this.warpT * 100) + "%";
    morph.oninput = () => {
      this.warpT = parseFloat(morph.value);
      morphLab.textContent = Math.round(this.warpT * 100) + "%";
      if (!this.warpOn || !this.warp || this.warp.small || this._warping) return;
      this.network.setAlt(this.warpedMerc(), this.warp.dim);
      this.network._redraw();
      this.redrawWarpedOverlays();
      this.drawRouteLines();
    };
  },
});

// the warp controls exist only once the page has its graph
const _origStart = App.start;
App.start = async function (DATA) {
  await _origStart.call(this, DATA);
  this.bindWarpUI();
  const prevSync = this.syncPickButtons.bind(this);
  this.syncPickButtons = () => {
    prevSync();
    if (this.warpOn) {
      document.getElementById("pickhint").textContent =
        "Map clicks are off while the city is warped. Choose neighbourhoods from the menus.";
    }
  };
};
