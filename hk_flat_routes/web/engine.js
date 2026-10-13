/* Hong Kong flat routes -- the routing engine.
 * Adapted from flattensf (MIT) by Drew Edwards,
 * https://github.com/almostimplemented/flattensf
 *
 * Copyright (c) 2026 sf-flat-routes contributors
 * Copyright (c) 2026 flattenhk contributors
 * Released under the MIT licence; the permission notice that goes with
 * both lines is in LICENSE (LICENSE.txt beside this file on the site).
 *
 * The whole routable graph is embedded (see hk_flat_routes/webgraph.py), so
 * routing happens in the browser: a Dijkstra over half a million directed
 * arcs with the same cost model Python uses. Shared by the explorer (app.js)
 * and the simple route page (simple.js).
 */
"use strict";

/* ------------------------------------------------------------------ decode */
const TYPES = {
  i1: Int8Array, u1: Uint8Array, i2: Int16Array, u2: Uint16Array,
  i4: Int32Array, u4: Uint32Array, f4: Float32Array, f8: Float64Array,
};

function base64ToBytes(str) {
  const bin = atob(str);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes;
}

/* The whole graph ships as one gzipped buffer. Inflating it in the browser
 * costs a few hundred milliseconds and saves about two thirds of the file
 * size, which matters a great deal more. */
async function inflateBytes(bytes) {
  // a host that serves .gz with Content-Encoding: gzip hands us the plain
  // bytes already; the gzip magic number tells the two cases apart
  if (!(bytes.length > 2 && bytes[0] === 0x1f && bytes[1] === 0x8b)) return bytes;
  if (typeof DecompressionStream === "undefined") {
    throw new Error("this browser has no DecompressionStream; please use a "
      + "current version of Chrome, Firefox, Edge or Safari");
  }
  const stream = new Blob([bytes]).stream()
    .pipeThrough(new DecompressionStream("gzip"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

async function inflate(b64str) {
  return inflateBytes(base64ToBytes(b64str));
}

/* Fetch the packed graph as a separate file (the hosted site), reporting
 * progress, or inflate the inline copy (the single-file page). */
async function loadBundle(data, onProgress) {
  if (data.bundle) return inflate(data.bundle);
  const res = await fetch(data.bundle_url);
  if (!res.ok) throw new Error("could not load " + data.bundle_url + " (" + res.status + ")");
  const total = +res.headers.get("Content-Length") || data.bundle_bytes || 0;
  const reader = res.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); got += value.length;
    if (onProgress) onProgress(got, total);
  }
  const all = new Uint8Array(got);
  let off = 0;
  for (const c of chunks) { all.set(c, off); off += c.length; }
  return inflateBytes(all);
}

/* Views onto the inflated buffer -- no copying, no JSON number parsing. */
class Bundle {
  constructor(buf, manifest) {
    this.buf = buf;
    this.manifest = manifest;
    this.decoder = new TextDecoder();
  }
  has(name) { return !!this.manifest.arrays[name]; }
  array(name) {
    const m = this.manifest.arrays[name];
    if (!m) throw new Error("missing array " + name);
    const T = TYPES[m.t];
    // the offset need not be aligned for the view type, so copy when it is not
    if ((this.buf.byteOffset + m.o) % T.BYTES_PER_ELEMENT === 0) {
      return new T(this.buf.buffer, this.buf.byteOffset + m.o, m.n);
    }
    const bytes = this.buf.slice(m.o, m.o + m.n * T.BYTES_PER_ELEMENT);
    return new T(bytes.buffer, 0, m.n);
  }
  text(name) {
    const m = this.manifest.strings[name];
    if (!m) throw new Error("missing string " + name);
    return this.decoder.decode(this.buf.subarray(m.o, m.o + m.b));
  }
}

function decodePolylineAt(s, start, end) {
  let i = start, lat = 0, lon = 0;
  const out = [];
  while (i < end) {
    let b, shift = 0, result = 0;
    do { b = s.charCodeAt(i++) - 63; result |= (b & 0x1f) << shift; shift += 5; } while (b >= 0x20);
    lat += (result & 1) ? ~(result >> 1) : (result >> 1);
    shift = 0; result = 0;
    do { b = s.charCodeAt(i++) - 63; result |= (b & 0x1f) << shift; shift += 5; } while (b >= 0x20);
    lon += (result & 1) ? ~(result >> 1) : (result >> 1);
    out.push(lon / 1e5, lat / 1e5);
  }
  return out;
}

/* A binary min-heap of (float key, int value) pairs on typed arrays. */
class MinHeap {
  constructor(cap = 1 << 16) { this.k = new Float64Array(cap); this.v = new Int32Array(cap); this.n = 0; }
  push(key, val) {
    if (this.n === this.k.length) {
      const nk = new Float64Array(this.k.length * 2), nv = new Int32Array(this.v.length * 2);
      nk.set(this.k); nv.set(this.v); this.k = nk; this.v = nv;
    }
    const k = this.k, v = this.v;
    let i = this.n++;
    k[i] = key; v[i] = val;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (k[p] <= k[i]) break;
      const tk = k[p], tv = v[p]; k[p] = k[i]; v[p] = v[i]; k[i] = tk; v[i] = tv; i = p;
    }
  }
  /* removes the minimum; its key is left in this.topKey */
  pop() {
    const k = this.k, v = this.v;
    const top = v[0]; this.topKey = k[0];
    this.n--;
    if (this.n > 0) {
      k[0] = k[this.n]; v[0] = v[this.n];
      let i = 0;
      for (;;) {
        const l = 2 * i + 1, r = l + 1; let m = i;
        if (l < this.n && k[l] < k[m]) m = l;
        if (r < this.n && k[r] < k[m]) m = r;
        if (m === i) break;
        const tk = k[m], tv = v[m]; k[m] = k[i]; v[m] = v[i]; k[i] = tk; v[i] = tv; i = m;
      }
    }
    return top;
  }
}

/* -------------------------------------------------------------- the graph */
class Graph {
  constructor(bundle, meta) {
    this.meta = meta;
    this.n = meta.n_nodes;
    this.lon = bundle.array("node_lon");
    this.lat = bundle.array("node_lat");
    this.nodeFlags = bundle.array("node_flags");
    // Which landmass each node is on, per mode (0 = the mode cannot use the
    // node). Hong Kong Island, Kowloon with the New Territories, Lantau and
    // the outlying islands are separate graphs for a walker or a cyclist,
    // and a route exists only between two nodes that share one. Sharing one
    // is not quite enough for a cyclist: a landmass takes in the streets a
    // one-way system leads into and not back out of, so a route may exist
    // one way and not the other, and route() returning null says so.
    this.nodeComp = {
      walk: bundle.has("node_comp_walk") ? bundle.array("node_comp_walk") : null,
      bike: bundle.has("node_comp_bike") ? bundle.array("node_comp_bike") : null,
    };
    this.nodeElev = bundle.array("node_elev");
    this.indptr = bundle.array("indptr");
    this.head = bundle.array("arc_head");
    this.arcEdge = bundle.array("arc_edge");
    this.arcLen = bundle.array("arc_len");
    this.arcGain = bundle.array("arc_gain");
    this.arcLoss = bundle.array("arc_loss");
    this.arcMaxGrade = bundle.array("arc_maxgrade");
    this.arcCls = bundle.array("arc_cls");
    this.arcFlags = bundle.array("arc_flags");
    this.m = this.head.length;
    // Dismount-and-push links: the short footpaths that join a cycle track
    // to the road, which a cyclist may take only on foot. One byte per arc,
    // absent from a bundle built before there were any. In bike mode such
    // an arc's length counts push_multiplier times over wherever a comfort
    // multiplier is charged: beside the class multipliers (pushFactor) and
    // in the comfort-weighted lengths (arcLenStress below).
    this.arcPush = bundle.has("arc_push") ? bundle.array("arc_push") : null;
    this.pushMult = this.arcPush ? (meta.push_multiplier || 1) : 1;
    // The mean gradient and the distance-above-threshold arrays feed the
    // explorer's grade-averse objectives. The route page's bundle leaves
    // them out (it trades distance against climbing only), and they read
    // as zero here when absent.
    const zeros = new Uint16Array(bundle.has("arc_meangrade") ? 0 : this.m);
    this.arcMeanGrade = bundle.has("arc_meangrade") ? bundle.array("arc_meangrade") : zeros;
    this.hasGrades = bundle.has("arc_meangrade");
    this.th = meta.thresholds.map(t => bundle.has("arc_th" + t) ? bundle.array("arc_th" + t) : zeros);

    this.DM = meta.scales.dm; this.CM = meta.scales.cm;
    this.COORD = meta.scales.coord; this.GRADE = meta.scales.grade;

    // Comfort-weighted arc lengths for bike mode, in the same units as
    // arcLen: a block of busy arterial counts as more than its length, a
    // block with a protected lane as less (bikeways.py has the table), and
    // a pushed link as its comfort-weighted length push_multiplier times
    // over. The bundle carries the weight of the class alone, a byte an
    // edge, so the push factor goes in here, once. Searches that take a
    // ``stress`` flag route on these instead.
    this.arcLenStress = this.arcLen;
    if (bundle.has("edge_stress") || this.pushMult !== 1) {
      this.edgeStress = bundle.has("edge_stress") ? bundle.array("edge_stress") : null;
      const q = meta.scales.stress || 100, s = new Int32Array(this.m);
      for (let a = 0; a < this.m; a++) {
        const w = this.edgeStress ? this.edgeStress[this.arcEdge[a]] : q;
        s[a] = Math.round(this.arcLen[a] * w / q * (this.arcPush && this.arcPush[a] ? this.pushMult : 1));
      }
      this.arcLenStress = s;
    }

    // scratch arrays reused across searches, with a visit stamp so nothing
    // has to be cleared between runs
    this.dist = new Float64Array(this.n);
    this.prev = new Int32Array(this.n);
    this.seen = new Int32Array(this.n);
    this.done = new Uint8Array(this.n);
    this.stamp = 0;
    this.heapKey = new Float64Array(1 << 16);
    this.heapVal = new Int32Array(1 << 16);
  }

  nodeLon(i) { return this.lon[i] / this.COORD; }
  nodeLat(i) { return this.lat[i] / this.COORD; }
  nodeZ(i) { return this.nodeElev[i] / this.DM; }
  modeBit(mode) { return mode === "bike" ? 2 : 1; }
  /* landmass of a node for a mode: 0 when the mode cannot use the node */
  comp(node, mode) {
    if (node < 0) return 0;
    const c = this.nodeComp[mode === "bike" ? "bike" : "walk"];
    return c ? c[node] : ((this.nodeFlags[node] & this.modeBit(mode)) ? 1 : 0);
  }
  /* its name ("Hong Kong Island"), where the build could give it one */
  compName(node, mode) {
    const names = (this.meta.components || {})[mode === "bike" ? "bike" : "walk"] || [];
    return names[this.comp(node, mode) - 1] || "";
  }
  /* how many nodes a landmass has for a mode: the build's own count where
   * the bundle carries it (meta.component_nodes), counted here otherwise */
  compNodes(comp, mode) {
    const key = mode === "bike" ? "bike" : "walk";
    const given = (this.meta.component_nodes || {})[key];
    if (given) return given[comp - 1] || 0;
    this._compNodes = this._compNodes || {};
    if (!this._compNodes[key]) {
      const count = new Int32Array(256);
      for (let i = 0; i < this.n; i++) count[this.comp(i, key)]++;
      this._compNodes[key] = count;
    }
    return comp > 0 ? this._compNodes[key][comp] : 0;
  }
  lengths(stress) { return stress ? this.arcLenStress : this.arcLen; }

  /* Cost of one arc in equivalent metres. Mirrors routing.edge_costs.
   * ``pushK`` is pushFactor() for the mode and weights. */
  arcCost(a, w, mult, len = this.arcLen, pushK = 1) {
    let c = (len[a] / this.DM) * mult[this.arcCls[a]];
    if (pushK !== 1 && this.arcPush[a]) c *= pushK;
    c += w.alpha * (this.arcGain[a] / this.CM);
    if (w.beta) {
      let pen = 0;
      for (let k = 0; k < this.th.length; k++) {
        const p = w.penalties[k];
        if (p) pen += p * (this.th[k][a] / this.DM);
      }
      c += w.beta * pen;
    }
    if (w.gamma && w.extreme) {
      c += w.gamma * w.extreme * (this.th[this.th.length - 1][a] / this.DM);
    }
    return c > 1e-6 ? c : 1e-6;
  }

  multipliers(mode, w) {
    const base = this.meta.multipliers[mode] || [];
    if (w.use_class_multiplier === false) return base.map(() => 1);
    return base;
  }
  /* What the length of a pushed link is multiplied by under these weights:
   * the push multiplier in bike mode wherever the class multipliers apply,
   * as in routing.edge_costs. The comfort-weighted lengths carry it
   * already (the constructor puts it in), so a search on those is not
   * charged twice. */
  pushFactor(mode, w) {
    return mode === "bike" && w.use_class_multiplier !== false && !w.stress ? this.pushMult : 1;
  }

  /* Dijkstra with a flat binary heap, stopping once the target settles. */
  route(src, dst, mode, w) {
    if (src === dst || src < 0 || dst < 0) return null;
    const bit = this.modeBit(mode);
    const mult = this.multipliers(mode, w), len = this.lengths(w.stress), pushK = this.pushFactor(mode, w);
    const { dist, prev, seen, done, indptr, head } = this;
    const stamp = ++this.stamp;

    let hk = this.heapKey, hv = this.heapVal, hn = 0;
    const push = (key, val) => {
      if (hn === hk.length) {
        const nk = new Float64Array(hk.length * 2), nv = new Int32Array(hv.length * 2);
        nk.set(hk); nv.set(hv); hk = this.heapKey = nk; hv = this.heapVal = nv;
      }
      let i = hn++;
      hk[i] = key; hv[i] = val;
      while (i > 0) {
        const p = (i - 1) >> 1;
        if (hk[p] <= hk[i]) break;
        const tk = hk[p], tv = hv[p];
        hk[p] = hk[i]; hv[p] = hv[i]; hk[i] = tk; hv[i] = tv;
        i = p;
      }
    };
    const pop = () => {
      const top = hv[0], topk = hk[0];
      hn--;
      if (hn > 0) {
        hk[0] = hk[hn]; hv[0] = hv[hn];
        let i = 0;
        for (;;) {
          const l = 2 * i + 1, r = l + 1;
          let s = i;
          if (l < hn && hk[l] < hk[s]) s = l;
          if (r < hn && hk[r] < hk[s]) s = r;
          if (s === i) break;
          const tk = hk[s], tv = hv[s];
          hk[s] = hk[i]; hv[s] = hv[i]; hk[i] = tk; hv[i] = tv;
          i = s;
        }
      }
      return [topk, top];
    };

    seen[src] = stamp; dist[src] = 0; prev[src] = -1; done[src] = 0;
    push(0, src);
    let settled = 0;
    while (hn > 0) {
      const [dv, u] = pop();
      if (seen[u] !== stamp || done[u]) continue;
      done[u] = 1; settled++;
      if (u === dst) break;
      for (let a = indptr[u]; a < indptr[u + 1]; a++) {
        if ((this.arcFlags[a] & bit) === 0) continue;
        const v = head[a];
        if (seen[v] === stamp && done[v]) continue;
        const nd = dv + this.arcCost(a, w, mult, len, pushK);
        if (seen[v] !== stamp || nd < dist[v]) {
          seen[v] = stamp; dist[v] = nd; prev[v] = a; done[v] = 0;
          push(nd, v);
        }
      }
    }
    if (seen[dst] !== stamp || !done[dst]) return null;

    // walk back: prev[] holds the arc used to reach each node
    const arcs = [];
    let v = dst, guard = 0;
    while (v !== src) {
      const a = prev[v];
      if (a < 0 || guard++ > this.n) return null;
      arcs.push(a);
      v = this.arcTail(a);
    }
    arcs.reverse();
    return { arcs, cost: dist[dst], settled };
  }

  /* Costs from one source to a set of target nodes: the same search as
   * route(), run until every target has settled. Used for the warp's cost
   * matrix, where one source serves ~100 targets at a time. */
  distances(src, mode, w, targets) {
    const bit = this.modeBit(mode);
    const mult = this.multipliers(mode, w), len = this.lengths(w.stress), pushK = this.pushFactor(mode, w);
    const { dist, seen, done, indptr, head } = this;
    const stamp = ++this.stamp;
    const want = new Int32Array(this.n);
    let remaining = 0;
    for (const t of targets) { if (!want[t]) { want[t] = 1; remaining++; } }
    let hk = this.heapKey, hv = this.heapVal, hn = 0;
    const push = (key, val) => {
      if (hn === hk.length) {
        const nk = new Float64Array(hk.length * 2), nv = new Int32Array(hv.length * 2);
        nk.set(hk); nv.set(hv); hk = this.heapKey = nk; hv = this.heapVal = nv;
      }
      let i = hn++; hk[i] = key; hv[i] = val;
      while (i > 0) {
        const p = (i - 1) >> 1;
        if (hk[p] <= hk[i]) break;
        const tk = hk[p], tv = hv[p]; hk[p] = hk[i]; hv[p] = hv[i]; hk[i] = tk; hv[i] = tv; i = p;
      }
    };
    const pop = () => {
      const top = hv[0], topk = hk[0]; hn--;
      if (hn > 0) {
        hk[0] = hk[hn]; hv[0] = hv[hn];
        let i = 0;
        for (;;) {
          const l = 2 * i + 1, r = l + 1; let m = i;
          if (l < hn && hk[l] < hk[m]) m = l;
          if (r < hn && hk[r] < hk[m]) m = r;
          if (m === i) break;
          const tk = hk[m], tv = hv[m]; hk[m] = hk[i]; hv[m] = hv[i]; hk[i] = tk; hv[i] = tv; i = m;
        }
      }
      return [topk, top];
    };
    seen[src] = stamp; dist[src] = 0; done[src] = 0; push(0, src);
    while (hn > 0 && remaining > 0) {
      const [dv, u] = pop();
      if (seen[u] !== stamp || done[u]) continue;
      done[u] = 1;
      if (want[u]) remaining--;
      for (let a = indptr[u]; a < indptr[u + 1]; a++) {
        if ((this.arcFlags[a] & bit) === 0) continue;
        const v = head[a];
        if (seen[v] === stamp && done[v]) continue;
        const nd = dv + this.arcCost(a, w, mult, len, pushK);
        if (seen[v] !== stamp || nd < dist[v]) { seen[v] = stamp; dist[v] = nd; done[v] = 0; push(nd, v); }
      }
    }
    return targets.map(t => (seen[t] === stamp && done[t]) ? dist[t] : Infinity);
  }

  /* One Dijkstra over a per-arc cost array: forward from src, or with
   * reverse, backward into src (dist[v] is then the cost of v -> src and
   * prev[v] the first arc of that path). It can stop at a target, and edges
   * marked in pen cost (1 + penK) times as much, which is how a loop's way
   * home is pushed off the streets it went out on. len and gain carry the
   * real length and climbing (m) of each tree path. */
  tree(src, mode, cost, { reverse = false, target = -1, pen = null, penNode = null, penK = 0, minPerM = 0 } = {}) {
    const n = this.n, bit = this.modeBit(mode), rev = reverse ? this.reverse() : null;
    const dist = new Float64Array(n).fill(Infinity), prev = new Int32Array(n).fill(-1);
    const done = new Uint8Array(n), len = new Float64Array(n), gain = new Float64Array(n);
    const heap = new MinHeap();
    // with a target and a known floor on cost per metre (minPerM), the
    // search is A*: straight-line distance to the target times that floor
    // never overestimates, so the route found is still the cheapest
    const astar = target >= 0 && minPerM > 0;
    const cosT = Math.cos(this.nodeLat(src) * Math.PI / 180);
    const tx = astar ? this.nodeLon(target) * 111320 * cosT : 0, ty = astar ? this.nodeLat(target) * 110540 : 0;
    const h = astar ? (v) => minPerM * Math.hypot(this.nodeLon(v) * 111320 * cosT - tx, this.nodeLat(v) * 110540 - ty) : () => 0;
    dist[src] = 0; heap.push(h(src), src);
    while (heap.n > 0) {
      const u = heap.pop();
      if (done[u]) continue;
      const du = dist[u];
      done[u] = 1;
      if (u === target) break;
      const lo = reverse ? rev.indptr[u] : this.indptr[u], hi = reverse ? rev.indptr[u + 1] : this.indptr[u + 1];
      for (let k = lo; k < hi; k++) {
        const a = reverse ? rev.arcs[k] : k;
        if ((this.arcFlags[a] & bit) === 0) continue;
        const v = reverse ? rev.tail[a] : this.head[a];
        let c = cost[a];
        if ((pen !== null && pen[this.arcEdge[a]]) || (penNode !== null && penNode[v])) c *= 1 + penK;
        const nd = du + c;
        if (nd < dist[v]) {
          dist[v] = nd; prev[v] = a; heap.push(nd + h(v), v);
          len[v] = len[u] + this.arcLen[a] / this.DM; gain[v] = gain[u] + this.arcGain[a] / this.CM;
        }
      }
    }
    return { src, dist, prev, len, gain, done, reverse };
  }

  /* Corners a loop may turn at: real intersections, where at least three
   * ways meet and at least one is a street (not only footways, service
   * lanes or steps), so a loop never turns around at the end of a
   * parking-lot lane. Cached per mode. */
  corners(mode) {
    this._corners = this._corners || {};
    if (this._corners[mode]) return this._corners[mode];
    const STREETS = new Set(["trunk", "primary", "secondary", "tertiary", "residential", "unclassified",
      "living_street", "pedestrian", "cycleway", "path"]);
    const street = this.meta.classes.map((c) => STREETS.has(c));
    const bit = this.modeBit(mode), { indptr, arcs, tail } = this.reverse();
    const ok = new Uint8Array(this.n), seen = new Int32Array(this.n).fill(-1);
    for (let u = 0; u < this.n; u++) {
      let deg = 0, onStreet = false;
      const touch = (v, a) => {
        if ((this.arcFlags[a] & bit) === 0) return;
        if (street[this.arcCls[a]]) onStreet = true;
        if (seen[v] !== u) { seen[v] = u; deg++; }
      };
      for (let a = this.indptr[u]; a < this.indptr[u + 1]; a++) touch(this.head[a], a);
      for (let k = indptr[u]; k < indptr[u + 1]; k++) touch(tail[arcs[k]], arcs[k]);
      ok[u] = deg >= 3 && onStreet ? 1 : 0;
    }
    this._corners[mode] = ok;
    return ok;
  }

  /* arcs from the tree's root to v (forward tree) or from v to the root
   * (reverse tree) */
  treePath(t, v) {
    const tail = this.reverse().tail, arcs = [];
    if (!t.done[v]) return null;
    if (!t.reverse) {
      for (let guard = 0; v !== t.src; guard++) {
        const a = t.prev[v]; if (a < 0 || guard > this.n) return null;
        arcs.push(a); v = tail[a];
      }
      return arcs.reverse();
    }
    for (let guard = 0; v !== t.src; guard++) {
      const a = t.prev[v]; if (a < 0 || guard > this.n) return null;
      arcs.push(a); v = this.head[a];
    }
    return arcs;
  }

  /* Flat loops: routes that start and end at src and run about targetM
   * metres, as flat as the streets around src allow.
   *
   * A loop is built from turnaround corners. One search outward and one
   * inward from src give every corner's flattest way out and way back.
   * The corners around src are split into sectors by bearing; in each,
   * the corner that looks cheapest to reach and return from is tried as a
   * petal (out to it, home by a different way: the streets used on the way
   * out cost penK times more on the way back), pairs of corners about a
   * sixth of a turn apart as triangles (out, across, home), and for loops
   * over five miles (eight kilometres), three corners a quarter turn apart.
   * A loop whose length misses the target by more than a tolerance is
   * retried once with a corner scaled nearer or farther. Loops that
   * repeat too much of themselves (an out-and-back) are dropped; the rest
   * are ranked by climbing, length breaking ties.
   *
   * Like pareto(), it returns a search object; step(budgetMs) until done.
   * search.last is the most recent well-shaped loop built, so the page can
   * show the scan as it happens.
   * Results: search.loops (best first, mutually distinct) and
   * search.accepted (every loop that met the length and overlap tests). */
  loops(src, mode, { targetM, alpha = 30, stress = false, sectors = 24, tol = 0.12,
    maxOverlap = 0.3, penK = 3, keep = 3, nearM = 110, perArc = 8, minRound = 0.2,
    tolM = 0.25 * 1609.344, retries = 3, outBack = false } = {}) {
    const g = this, T = targetM, L = g.lengths(stress);
    // a few metres per arc keeps routes off zigzags through tiny segments
    const cost = new Float64Array(g.m);
    for (let a = 0; a < g.m; a++) cost[a] = L[a] / g.DM + alpha * g.arcGain[a] / g.CM + perArc;
    const search = { loops: [], accepted: [], loose: [], all: [], tried: 0, done: false, ms: 0, shortfall: false };
    // a loop counts when it is within tolM of the target (or tol of it, if
    // that is tighter, for short loops); corners are rescaled up to
    // `retries` times to land in that window
    const t0 = performance.now();
    const band = Math.min(tol * T, tolM);
    let F = null, B = null, bySector = null;
    const jobs = [];
    const cosLat = Math.cos(g.nodeLat(src) * Math.PI / 180);
    const sectorOf = (v) => {
      const dx = (g.nodeLon(v) - g.nodeLon(src)) * cosLat, dy = g.nodeLat(v) - g.nodeLat(src);
      return Math.floor(((Math.atan2(dy, dx) + Math.PI) / (2 * Math.PI)) * sectors) % sectors;
    };
    // the corner in a sector whose way out is about `want` metres and whose
    // estimated loop (flattest out + flattest back) climbs least
    const pick = (s, want, spread, withBack) => {
      let best = -1, bestScore = Infinity;
      for (const v of bySector[s]) {
        const lf = F.len[v];
        if (Math.abs(lf - want) > spread) continue;
        const score = F.gain[v] + (withBack ? B.gain[v] : 0) + 0.002 * Math.abs(lf - want);
        if (score < bestScore) { bestScore = score; best = v; }
      }
      return best;
    };
    // cost per metre is never below the shortest-looking length factor
    // (0.8 on a protected lane in calm-streets mode), less a little for the
    // quantised lengths: the floor that keeps A* exact
    const minPerM = stress ? 0.78 : 0.98;
    const leg = (from, to, pen, penNode) => {
      const t = g.tree(from, mode, cost, { target: to, pen, penNode, penK, minPerM });
      return t.done[to] ? g.treePath(t, to) : null;
    };
    const mark = (pen, arcs) => { for (const a of arcs) pen[g.arcEdge[a]] = 1; };
    // corners within nearM of the streets already used, except around the
    // start, which every leg has to pass through: the way home avoids them,
    // so a loop is not the same street out and the next street back
    const XM = 111320 * Math.cos(g.nodeLat(src) * Math.PI / 180), YM = 110540;
    const px = (v) => g.nodeLon(v) * XM, py = (v) => g.nodeLat(v) * YM;
    const sx = px(src), sy = py(src);
    let nodeGrid = null;   // cell -> nodes, over every corner the loops can reach
    const cellKey = (cx, cy) => cx * 100003 + cy;
    const markNear = (penNode, arcs) => {
      const r2 = nearM * nearM, home2 = (1.5 * nearM) ** 2;
      for (const a of arcs) {
        const v0 = g.head[a], x0 = px(v0), y0 = py(v0);
        const cx = Math.floor(x0 / nearM), cy = Math.floor(y0 / nearM);
        for (let i = -1; i <= 1; i++) for (let j = -1; j <= 1; j++) {
          const cell = nodeGrid.get(cellKey(cx + i, cy + j)); if (!cell) continue;
          for (const v of cell) {
            if (penNode[v]) continue;
            const x = px(v), y = py(v);
            if ((x - x0) ** 2 + (y - y0) ** 2 < r2 && (x - sx) ** 2 + (y - sy) ** 2 >= home2) penNode[v] = 1;
          }
        }
      }
    };
    const measure = (arcs) => {
      let len = 0, gain = 0, rep = 0;
      const seen = new Map();
      for (const a of arcs) {
        const l = g.arcLen[a] / g.DM, e = g.arcEdge[a];
        len += l; gain += g.arcGain[a] / g.CM;
        if (seen.has(e)) rep += l; else seen.set(e, 1);
      }
      let area = 0, prevX = sx, prevY = sy;
      for (const a of arcs) { const x = px(g.head[a]) - sx, y = py(g.head[a]) - sy; area += (prevX - sx) * y - x * (prevY - sy); prevX = x + sx; prevY = y + sy; }
      const round = len > 0 ? 4 * Math.PI * Math.abs(area / 2) / (len * len) : 0;
      return { len, gain, overlap: len > 0 ? rep / len : 1, round };
    };
    const consider = (arcs, kind) => {
      if (!arcs || !arcs.length) return null;
      search.tried++;
      const m = measure(arcs);
      const r = { arcs, length: m.len, gain: m.gain, overlap: m.overlap, round: m.round, kind };
      // an out-and-back is allowed to be what it is
      const shaped = kind === "outback" || (m.overlap <= maxOverlap && m.round >= minRound);
      if (shaped) { search.all.push(r); search.last = r; }
      if (shaped && Math.abs(m.len - T) <= band) search.accepted.push(r);
      else if (shaped && Math.abs(m.len - T) <= tol * T) search.loose.push(r);
      return r;
    };
    const petal = (s, want, tries) => {
      const w = pick(s, want, Math.max(150, 0.06 * T), true);
      if (w < 0) return;
      const out = g.treePath(F, w); if (!out) return;
      const pen = new Uint8Array(g.meta.n_edges), penNode = new Uint8Array(g.n);
      mark(pen, out); markNear(penNode, out);
      const back = leg(w, src, pen, penNode); if (!back) return;
      const r = consider(out.concat(back), "petal");
      if (r && tries > 0 && Math.abs(r.length - T) > band) {
        jobs.push(() => petal(s, want * T / r.length, tries - 1));
      }
    };
    // the flattest way out to a turnaround about half the target away, and
    // the flattest way back, which is usually the same streets
    const outAndBack = (s, want, tries) => {
      const w = pick(s, want, Math.max(150, 0.06 * T), true);
      if (w < 0) return;
      const out = g.treePath(F, w), back = g.treePath(B, w);
      if (!out || !back) return;
      const r = consider(out.concat(back), "outback");
      if (r && tries > 0 && Math.abs(r.length - T) > band) {
        jobs.push(() => outAndBack(s, want * T / r.length, tries - 1));
      }
    };
    // out to the first corner, across each next one, then home; every leg
    // avoids the streets the loop has already used
    const polygon = (ss, want, tries) => {
      const sp = Math.max(150, 0.06 * T);
      const vs = ss.map((s, i) => pick(s, want, sp, i === ss.length - 1));
      if (vs.some((v) => v < 0) || new Set(vs).size < vs.length) return;
      const pen = new Uint8Array(g.meta.n_edges), penNode = new Uint8Array(g.n);
      let arcs = g.treePath(F, vs[0]); if (!arcs) return;
      mark(pen, arcs); markNear(penNode, arcs);
      for (const [from, to] of vs.slice(1).map((v, i) => [vs[i], v]).concat([[vs[vs.length - 1], src]])) {
        const l = leg(from, to, pen, penNode); if (!l) return;
        mark(pen, l); markNear(penNode, l); arcs = arcs.concat(l);
      }
      const r = consider(arcs, vs.length === 2 ? "triangle" : "polygon");
      if (r && tries > 0 && Math.abs(r.length - T) > band) {
        jobs.push(() => polygon(ss, want * T / r.length, tries - 1));
      }
    };
    search.step = (budgetMs = 30) => {
      const start = performance.now();
      if (!F) {
        F = g.tree(src, mode, cost);
        B = g.tree(src, mode, cost, { reverse: true });
        bySector = Array.from({ length: sectors }, () => []);
        const corner = g.corners(mode);
        nodeGrid = new Map();
        for (let v = 0; v < g.n; v++) {
          if (!F.done[v] || F.len[v] > 0.75 * T) continue;
          const k = cellKey(Math.floor(px(v) / nearM), Math.floor(py(v) / nearM));
          let c = nodeGrid.get(k); if (!c) nodeGrid.set(k, c = []); c.push(v);
        }
        for (let v = 0; v < g.n; v++) {
          if (!F.done[v] || !B.done[v] || v === src || !corner[v]) continue;
          const lf = F.len[v];
          if (lf < 0.1 * T || lf > 0.7 * T) continue;
          bySector[sectorOf(v)].push(v);
        }
        for (let s = 0; s < sectors; s++) {
          if (outBack) jobs.push(() => outAndBack(s, 0.5 * T, retries));
          jobs.push(() => petal(s, 0.38 * T, retries));
          jobs.push(() => petal(s, 0.46 * T, retries));
        }
        for (const gap of [Math.round(sectors / 6), Math.round(sectors / 4)]) {
          for (let s = 0; s < sectors; s++) jobs.push(() => polygon([s, (s + gap) % sectors], (gap > sectors / 5 ? 0.27 : 0.3) * T, retries));
        }
        // longer loops: three corners a quarter turn apart, which keeps the
        // loop on land where two far corners would fall in the sea
        if (T > 8000) {
          const q = Math.max(1, Math.round(sectors / 4));
          for (let s = 0; s < sectors; s++) jobs.push(() => polygon([s, (s + q) % sectors, (s + 2 * q) % sectors], 0.21 * T, retries));
        }
        if (performance.now() - start > budgetMs) return false;
      }
      while (jobs.length) {
        jobs.shift()();
        if (performance.now() - start > budgetMs) return false;
      }
      // rank, then keep loops that are not near-copies of a better one. When
      // nothing came close enough to the target (a long loop from the edge
      // of the city), offer the loop that came closest, and say so.
      let pool = search.accepted.slice();
      if (!pool.length && search.loose.length) {
        pool = search.loose.slice(); search.shortfall = true;
      }
      if (!pool.length && search.all.length) {
        const closest = search.all.slice().sort((x, y) => Math.abs(x.length - T) - Math.abs(y.length - T))[0];
        pool = [closest]; search.shortfall = true;
      }
      pool.sort((x, y) => (x.gain - y.gain) || (Math.abs(x.length - T) - Math.abs(y.length - T)));
      const edges = (r) => new Set(r.arcs.map((a) => g.arcEdge[a]));
      const picked = [];
      for (const r of pool) {
        const er = edges(r);
        const similar = picked.some((p) => {
          let both = 0; for (const e of er) if (p.edges.has(e)) both++;
          return both / Math.min(er.size, p.edges.size) > 0.6;
        });
        if (!similar) picked.push({ r, edges: er });
        if (picked.length >= keep) break;
      }
      search.loops = picked.map((p) => p.r);
      const gains = search.accepted.map((r) => r.gain).sort((x, y) => x - y);
      search.medianGain = gains.length ? gains[Math.floor(gains.length / 2)] : NaN;
      search.ms = performance.now() - t0;
      search.done = true;
      return true;
    };
    return search;
  }

  /* Reverse adjacency (arcs grouped by head node), built on first use. */
  reverse() {
    if (this._rev) return this._rev;
    const n = this.n, m = this.m;
    const indptr = new Int32Array(n + 1);
    for (let a = 0; a < m; a++) indptr[this.head[a] + 1]++;
    for (let i = 0; i < n; i++) indptr[i + 1] += indptr[i];
    const arcs = new Int32Array(m), tail = new Int32Array(m), fill = indptr.slice(0, n);
    for (let u = 0; u < n; u++) {
      for (let a = this.indptr[u]; a < this.indptr[u + 1]; a++) { tail[a] = u; arcs[fill[this.head[a]]++] = a; }
    }
    this._rev = { indptr, arcs, tail };
    return this._rev;
  }

  /* Exact lower bound from every node to dst on one per-arc weight (a
   * reverse Dijkstra), in that weight's own integer units. */
  boundsTo(dst, mode, weight) {
    const { indptr, arcs, tail } = this.reverse();
    const bit = this.modeBit(mode);
    const dist = new Float64Array(this.n).fill(Infinity);
    const done = new Uint8Array(this.n);
    const heap = new MinHeap();
    dist[dst] = 0; heap.push(0, dst);
    while (heap.n > 0) {
      const u = heap.pop(), du = heap.topKey;
      if (done[u] || du > dist[u]) continue;
      done[u] = 1;
      for (let k = indptr[u]; k < indptr[u + 1]; k++) {
        const a = arcs[k];
        if ((this.arcFlags[a] & bit) === 0) continue;
        const v = tail[a], nd = du + weight[a];
        if (nd < dist[v]) { dist[v] = nd; heap.push(nd, v); }
      }
    }
    return dist;
  }

  /* The whole distance-versus-climbing frontier between two nodes: every
   * route that no other route beats on both counts, not only the ones a
   * weighted sum can reach. This is BOA* (Hernandez et al., bi-objective
   * A* with lazy dominance checks): labels (node, length, gain) expand in
   * order of (length + bound, gain + bound), and a label is dropped when it
   * reaches a node with no less climbing than a label that got there first,
   * which, because of the expansion order, was also no longer. The check is
   * one comparison per label, which is what makes the search affordable in
   * a browser.
   *
   * ``eps`` (cm of gain) merges frontier points that differ by less than
   * that in climbing, which keeps the frontier to a readable size, and
   * ``epsNode`` applies the same tolerance at intermediate nodes, where it
   * trades a little exactness (the tolerance can accumulate along a path)
   * for a much smaller search; ``dCap``
   * (5 cm units) and ``gCap`` (cm) bound the search to routes no longer
   * than the flattest route worth showing and no hillier than the shortest.
   * ``maxLabels`` is a safety valve: the search then stops with the short
   * end of the frontier, which it finds first.
   *
   * Returns a search object; call step(budgetMs) until it reports done, so
   * the page can keep painting. */
  pareto(src, dst, mode, { eps = 50, epsNode = eps, dCap = Infinity, gCap = Infinity, maxLabels = 4e6, stress = false } = {}) {
    const g = this;
    const bit = g.modeBit(mode), len = g.lengths(stress);
    const h1 = g.boundsTo(dst, mode, len), h2 = g.boundsTo(dst, mode, g.arcGain);
    const g2min = new Float64Array(g.n).fill(Infinity);
    let cap = 1 << 16;
    let lNode = new Int32Array(cap), lG1 = new Int32Array(cap), lG2 = new Int32Array(cap),
      lParent = new Int32Array(cap), lArc = new Int32Array(cap);
    let nl = 0;
    const grow = () => {
      cap *= 2;
      const r = (old, T) => { const a = new T(cap); a.set(old); return a; };
      lNode = r(lNode, Int32Array); lG1 = r(lG1, Int32Array); lG2 = r(lG2, Int32Array);
      lParent = r(lParent, Int32Array); lArc = r(lArc, Int32Array);
    };
    const K = 1 << 20;   // f1 in the high bits, f2 in the low: expansion order (f1, f2)
    const heap = new MinHeap();
    const add = (node, g1, g2, parent, arc) => {
      if (nl === cap) grow();
      lNode[nl] = node; lG1[nl] = g1; lG2[nl] = g2; lParent[nl] = parent; lArc[nl] = arc;
      heap.push((g1 + h1[node]) * K + Math.min(g2 + h2[node], K - 1), nl);
      nl++;
    };
    const search = { solutions: [], done: false, expanded: 0, labels: 0, truncated: false };
    if (src === dst || !Number.isFinite(h1[src])) { search.done = true; return search; }
    add(src, 0, 0, -1, -1);

    search.step = (budgetMs = 30) => {
      const t0 = performance.now();
      let n = 0;
      while (heap.n > 0) {
        if ((++n & 1023) === 0 && performance.now() - t0 > budgetMs) return false;
        const x = heap.pop();
        const node = lNode[x], g1 = lG1[x], g2 = lG2[x];
        if (g2 + (node === dst ? eps : epsNode) > g2min[node] || g2 + h2[node] + eps > g2min[dst]) continue;
        g2min[node] = g2;
        search.expanded++;
        if (node === dst) {
          const arcs = [];
          for (let y = x; lParent[y] >= 0; y = lParent[y]) arcs.push(lArc[y]);
          arcs.reverse();
          let real = 0;
          for (const a of arcs) real += g.arcLen[a];
          search.solutions.push({ arcs, length: real / g.DM, weighted: g1 / g.DM, gain: g2 / g.CM });
          continue;
        }
        for (let a = g.indptr[node]; a < g.indptr[node + 1]; a++) {
          if ((g.arcFlags[a] & bit) === 0) continue;
          const v = g.head[a], n1 = g1 + len[a], n2 = g2 + g.arcGain[a];
          if (n1 + h1[v] > dCap || n2 + h2[v] > gCap) continue;
          if (n2 + epsNode > g2min[v] || n2 + h2[v] + eps > g2min[dst]) continue;
          add(v, n1, n2, x, a);
        }
        if (nl > maxLabels) { search.truncated = true; break; }
      }
      search.labels = nl;
      search.done = true;
      return true;
    };
    return search;
  }

  /* Arc index for an (edge id, reversed) pair. Built on first use; the map
   * itself does not need it, but it makes the router addressable from tests
   * and from the console. */
  arcOf(edgeId, reversed) {
    if (!this._arcLookup) {
      const lut = new Int32Array(this.arcEdge.length ? 0 : 0);
      this._arcLookup = new Map();
      for (let a = 0; a < this.m; a++) {
        this._arcLookup.set(this.arcEdge[a] * 2 + ((this.arcFlags[a] & 4) ? 1 : 0), a);
      }
    }
    const a = this._arcLookup.get(edgeId * 2 + (reversed ? 1 : 0));
    return a === undefined ? -1 : a;
  }

  /* Total cost of an explicit arc sequence, for comparison against other
   * implementations of the same cost model. */
  pathCost(arcs, mode, w) {
    const mult = this.multipliers(mode, w), len = this.lengths(w.stress), pushK = this.pushFactor(mode, w);
    let c = 0;
    for (const a of arcs) c += this.arcCost(a, w, mult, len, pushK);
    return c;
  }

  /* tail node of an arc, by binary search over the CSR row pointers */
  arcTail(a) {
    const p = this.indptr;
    let lo = 0, hi = this.n;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (p[mid + 1] <= a) lo = mid + 1; else hi = mid;
    }
    return lo;
  }

  /* Aggregate a route exactly as routing.summarise_route does. Its
   * ``max_grade``, the route's steepest pitch, is ``steepest`` here (and
   * never below zero); ``max_grade`` here is the largest gradient stored
   * for any arc, whatever the arc's length. */
  summarise(arcs) {
    let dist = 0, gain = 0, loss = 0, maxg = -Infinity, wgrade = 0, stressed = 0, steep = 0;
    const th = new Array(this.th.length).fill(0);
    // what a short arc is and what its gradient is held to, as the build
    // has them (routing.arc_pitch); a bundle from before it said so used
    // 15 m, and its arcs' own stored maxima are clipped at the ceiling
    const minLen = this.meta.min_reliable_grade_length_m || 15, ceiling = this.meta.max_plausible_grade;
    for (const a of arcs) {
      const L = this.arcLen[a] / this.DM;
      dist += L;
      stressed += this.arcLenStress[a] / this.DM;
      gain += this.arcGain[a] / this.CM;
      loss += this.arcLoss[a] / this.CM;
      const g = this.arcMaxGrade[a] / this.GRADE;
      if (g > maxg) maxg = g;
      // the steepest figure the route page shows, and Python's steepest
      // pitch: over a block shorter than 15 m a peak grade is lidar noise
      // (upstream met a 3 m stub that read 45%), so there the block's
      // average climb stands in, as the analysis does for its gradient
      // tests (config.MIN_RELIABLE_GRADE_LENGTH_M). The average is held to
      // the 60% ceiling every sampled gradient is held to: net climb over
      // length has no such limit, and on some 1,200 of the territory's
      // 211,000 short arcs it read over 60% (2.5 m up a 2.3 m stub of
      // Ladder Street: 110%; the worst, 1209%).
      const sg = L >= minLen ? g : Math.min(ceiling === undefined ? Math.max(g, 0) : ceiling,
        Math.max(0, (this.arcGain[a] - this.arcLoss[a]) / this.CM / Math.max(L, 1e-6)));
      if (sg > steep) steep = sg;
      wgrade += (this.arcMeanGrade[a] / this.GRADE) * L;
      for (let k = 0; k < th.length; k++) th[k] += this.th[k][a] / this.DM;
    }
    if (!arcs.length) maxg = 0;
    const prof = this.profile(arcs);
    return {
      distance_m: dist, stress_m: stressed, elev_gain_m: gain, elev_loss_m: loss, steepest: steep,
      max_grade: maxg, avg_abs_grade: dist > 0 ? wgrade / dist : 0,
      thresholds: th, n_edges: arcs.length,
      start_elev_m: prof.z.length ? prof.z[0] : 0,
      end_elev_m: prof.z.length ? prof.z[prof.z.length - 1] : 0,
      profile: prof,
    };
  }

  /* Elevation series along a route, sampled at every intersection. */
  profile(arcs) {
    const d = [], z = [];
    let acc = 0;
    if (!arcs.length) return { d, z };
    const first = this.arcTail(arcs[0]);
    d.push(0); z.push(this.nodeZ(first));
    for (const a of arcs) {
      acc += this.arcLen[a] / this.DM;
      d.push(acc); z.push(this.nodeZ(this.head[a]));
    }
    return { d, z };
  }

  /* Route geometry in [lat,lon] pairs, honouring per-arc direction. */
  geometry(arcs, geom) {
    const out = [];
    for (const a of arcs) {
      let pts = geom.edgeCoords(this.arcEdge[a]);
      if (this.arcFlags[a] & 4) {
        const rev = [];
        for (let i = pts.length - 2; i >= 0; i -= 2) rev.push(pts[i], pts[i + 1]);
        pts = rev;
      }
      for (let i = 0; i < pts.length; i += 2) {
        const ll = [pts[i + 1], pts[i]];
        const last = out[out.length - 1];
        if (!last || last[0] !== ll[0] || last[1] !== ll[1]) out.push(ll);
      }
    }
    return out;
  }
}

/* ------------------------------------------------- edge geometry + indexes */
class Geometry {
  constructor(bundle, meta) {
    this.s = bundle.text("geom");
    this.off = bundle.array("geom_off");
    this.nEdges = meta.n_edges;
    this.name = bundle.array("edge_name");
    this.cls = bundle.array("edge_cls");
    this.len = bundle.array("edge_len");
    // display attributes only the explorer uses; absent from the route
    // page's bundle
    const opt = (name, T) => bundle.has(name) ? bundle.array(name) : new T(meta.n_edges);
    this.bucket = opt("edge_bucket", Uint8Array);
    this.maxgrade = opt("edge_maxgrade", Uint16Array);
    this.avggrade = opt("edge_avggrade", Int16Array);
    this.gainkm = opt("edge_gainkm", Uint16Array);
    this.lowstress = opt("edge_lowstress", Uint8Array);
    this.names = meta.names;
    this.classes = meta.classes;
    this.GRADE = meta.scales.grade; this.DM = meta.scales.dm;

    // decode every edge once into flat coordinate storage
    const coords = [], starts = new Int32Array(this.nEdges + 1);
    let n = 0;
    for (let i = 0; i < this.nEdges; i++) {
      const pts = decodePolylineAt(this.s, this.off[i], this.off[i + 1]);
      starts[i] = n;
      for (let k = 0; k < pts.length; k++) coords.push(pts[k]);
      n += pts.length;
    }
    starts[this.nEdges] = n;
    this.coords = Float32Array.from(coords);
    this.starts = starts;
    this.s = null;   // the encoded string is no longer needed

    // per-edge bounding boxes, for viewport culling
    this.bbox = new Float32Array(this.nEdges * 4);
    for (let i = 0; i < this.nEdges; i++) {
      let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
      for (let k = starts[i]; k < starts[i + 1]; k += 2) {
        const x = this.coords[k], y = this.coords[k + 1];
        if (x < x0) x0 = x; if (x > x1) x1 = x;
        if (y < y0) y0 = y; if (y > y1) y1 = y;
      }
      this.bbox[i * 4] = x0; this.bbox[i * 4 + 1] = y0;
      this.bbox[i * 4 + 2] = x1; this.bbox[i * 4 + 3] = y1;
    }
  }

  /* Every vertex in Web Mercator, as fractions of the world's width
   * measured from the first vertex (so that single precision is ample:
   * a hundredth of a pixel at zoom 18). Painting a few hundred thousand
   * streets is then one multiply and one add per coordinate, instead of a
   * projection call per vertex. Built on first use. */
  mercator() {
    if (this._merc) return this._merc;
    const c = this.coords, n = c.length >> 1;
    const X = (lon) => (lon + 180) / 360;
    const Y = (lat) => { const s = Math.sin(lat * Math.PI / 180); return 0.5 - Math.log((1 + s) / (1 - s)) / (4 * Math.PI); };
    const x0 = n ? X(c[0]) : 0, y0 = n ? Y(c[1]) : 0;
    const mx = new Float32Array(n), my = new Float32Array(n);
    for (let i = 0; i < n; i++) { mx[i] = X(c[2 * i]) - x0; my[i] = Y(c[2 * i + 1]) - y0; }
    this._merc = { mx, my, x0, y0 };
    return this._merc;
  }

  /* flat [lon,lat,...] for an edge id (ids are a contiguous 0..n-1 range) */
  edgeCoords(i) {
    if (i < 0 || i >= this.nEdges) return [];
    return this.coords.subarray(this.starts[i], this.starts[i + 1]);
  }

  edgeInfo(row) {
    return {
      name: this.name[row] ? this.names[this.name[row] - 1] : null,
      cls: this.classes[this.cls[row]],
      bucket: this.bucket[row],
      max_grade: this.maxgrade[row] / this.GRADE,
      avg_grade: this.avggrade[row] / this.GRADE,
      length_m: this.len[row] / this.DM,
      gain_per_km: this.gainkm[row] / this.DM,
      low_stress: !!this.lowstress[row],
    };
  }
}

/* metres in a degree of latitude, and in one of longitude at the equator */
const M_PER_LAT = 110540, M_PER_LON = 111320;

/* Uniform grid for nearest-node and nearest-edge queries. xs are longitudes
 * and ys latitudes, in degrees, and so is the cell size. */
class Grid {
  constructor(xs, ys, cell) {
    this.cell = cell;
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (let i = 0; i < xs.length; i++) {
      if (xs[i] < x0) x0 = xs[i]; if (xs[i] > x1) x1 = xs[i];
      if (ys[i] < y0) y0 = ys[i]; if (ys[i] > y1) y1 = ys[i];
    }
    this.x0 = x0; this.y0 = y0;
    // a degree of longitude in degrees of latitude: 0.93 at Hong Kong's 22
    // degrees north. Distances are compared in degrees of latitude.
    this.kx = xs.length ? M_PER_LON / M_PER_LAT * Math.cos((y0 + y1) / 2 * Math.PI / 180) : 1;
    this.nx = Math.max(1, Math.ceil((x1 - x0) / cell) + 1);
    this.ny = Math.max(1, Math.ceil((y1 - y0) / cell) + 1);
    const counts = new Int32Array(this.nx * this.ny + 1);
    const cellOf = new Int32Array(xs.length);
    for (let i = 0; i < xs.length; i++) {
      const cx = Math.min(this.nx - 1, Math.max(0, ((xs[i] - x0) / cell) | 0));
      const cy = Math.min(this.ny - 1, Math.max(0, ((ys[i] - y0) / cell) | 0));
      const c = cy * this.nx + cx;
      cellOf[i] = c; counts[c + 1]++;
    }
    for (let c = 0; c < counts.length - 1; c++) counts[c + 1] += counts[c];
    this.start = counts;
    this.items = new Int32Array(xs.length);
    const fill = counts.slice();
    for (let i = 0; i < xs.length; i++) this.items[fill[cellOf[i]]++] = i;
    this.xs = xs; this.ys = ys;
  }

  near(x, y, rings = 2) {
    const cx = Math.min(this.nx - 1, Math.max(0, ((x - this.x0) / this.cell) | 0));
    const cy = Math.min(this.ny - 1, Math.max(0, ((y - this.y0) / this.cell) | 0));
    const out = [];
    for (let r = 0; r <= rings; r++) {
      for (let dy = -r; dy <= r; dy++) {
        for (let dx = -r; dx <= r; dx++) {
          if (Math.max(Math.abs(dx), Math.abs(dy)) !== r) continue;
          const gx = cx + dx, gy = cy + dy;
          if (gx < 0 || gy < 0 || gx >= this.nx || gy >= this.ny) continue;
          const c = gy * this.nx + gx;
          for (let k = this.start[c]; k < this.start[c + 1]; k++) out.push(this.items[k]);
        }
      }
      if (out.length && r >= 1) break;
    }
    return out;
  }

  /* The nearest item that passes ``filter``, and no farther than ``maxM``
   * metres if that is given; -1 when there is none.
   *
   * Rings of cells are scanned outward from the point's own, and the scan
   * goes on past the first hit: an item in ring r is at least r - 1 cells
   * away, so only when that is farther than the best found is the best
   * the nearest. Upstream stopped at the first ring with a candidate and
   * compared squared degrees unscaled, which in a city of streets is the
   * same thing. Among Hong Kong's islands and country parks it was not: of
   * 2,847 random points within 3 km of the walking network the old search
   * gave 262 a farther corner than the nearest, 25 of them on another
   * landmass. */
  nearest(x, y, filter, maxM = Infinity) {
    const { cell, kx, nx, ny, xs, ys, start, items } = this;
    const cx = Math.min(nx - 1, Math.max(0, ((x - this.x0) / cell) | 0));
    const cy = Math.min(ny - 1, Math.max(0, ((y - this.y0) / cell) | 0));
    const max = maxM / M_PER_LAT;
    let best = -1, bestD = max * max;
    const scan = (gx, gy) => {
      if (gx < 0 || gy < 0 || gx >= nx || gy >= ny) return;
      const c = gy * nx + gx;
      for (let k = start[c]; k < start[c + 1]; k++) {
        const i = items[k];
        const dx = (xs[i] - x) * kx, dy = ys[i] - y, d = dx * dx + dy * dy;
        if (d <= bestD && (d < bestD || best < 0) && (!filter || filter(i))) { bestD = d; best = i; }
      }
    };
    const rings = Math.max(nx, ny);
    for (let r = 0; r <= rings; r++) {
      // nothing in this ring or beyond it can beat the best so far
      const floor = Math.max(0, r - 1) * cell * Math.min(kx, 1);
      if (floor * floor > bestD) break;
      for (let dx = -r; dx <= r; dx++) { scan(cx + dx, cy - r); if (r) scan(cx + dx, cy + r); }
      for (let dy = -r + 1; dy < r; dy++) { scan(cx - r, cy + dy); scan(cx + r, cy + dy); }
    }
    return best;
  }
}


window.Graph = Graph; window.Geometry = Geometry; window.Grid = Grid;
window.Bundle = Bundle; window.inflate = inflate; window.loadBundle = loadBundle; window.MinHeap = MinHeap;
