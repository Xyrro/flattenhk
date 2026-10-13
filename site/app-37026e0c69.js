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
    // "the peak tower" and "peak tower" are one name
    .replace(/^the (?=\S)/, "")
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
