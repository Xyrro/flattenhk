/* The warped city.
 * Adapted from flattensf (MIT) by Drew Edwards,
 * https://github.com/almostimplemented/flattensf
 *
 * Geography answers "how far", and in a city built on hillsides that is
 * the wrong question. Here the city is redrawn so that distance on the page
 * means *climbing cost*: the equivalent-metre cost the router charges, in
 * which a metre of ascent counts as some multiple of a metre of walking.
 * Places separated by a ridge move apart; places joined by a flat corridor
 * pull together. The multiple is the viewer's to set, and the same weights
 * drive the routes, so the picture and the directions always agree.
 *
 * Method, in brief: about 200 anchor junctions (the neighbourhoods' access
 * points plus a lattice over the landmass) get a full cost matrix from the
 * in-page router; stress majorisation (SMACOF, unit weights) lays them out
 * so page distance matches cost, starting from their true positions so the
 * result is the least deformation that fits; a Procrustes fit turns and
 * scales the layout back onto geography so north stays up; and a thin-plate
 * spline through the anchors' displacements carries every street vertex,
 * neighbourhood outline and route along for the ride.
 *
 * What Hong Kong changes:
 *
 *  - One landmass at a time. Hong Kong Island, Kowloon with the New
 *    Territories, Lantau and the outlying islands have no walking or cycling
 *    link between them, so there is no cost from an anchor on one to an
 *    anchor on another, and nothing for the layout to fit. The warp is
 *    built for a single landmass of the mode's graph (Graph.comp); the
 *    rest of the map is left where it is (app.js greys it).
 *  - The lattice is sized to the landmass. Upstream's one-kilometre lattice
 *    gives San Francisco some 175 anchors; over the whole territory it would
 *    give on the order of a thousand, and the cost matrix is one search of
 *    the landmass per anchor. So the spacing is chosen to give about the
 *    same number of anchors whatever the landmass: 740 m on Hong Kong
 *    Island, 640 m on Lantau, 2.3 km across Kowloon and the New
 *    Territories, and never under 150 m, which is where the small islands
 *    end up.
 *  - The anchors are street junctions, not trail junctions (see anchors).
 *  - The searches run on arc costs worked out once, and the spline is
 *    sampled onto a grid and read back by interpolation, because there are
 *    three times as many street vertices to carry as in San Francisco. The
 *    walking warp of Kowloon and the New Territories, 156,000 junctions,
 *    takes about seven seconds on a desktop; Hong Kong Island's a second
 *    and a half.
 *
 * What it does not change is the method, and the method does not promise a
 * picture that still looks like a map. A hilltop is far, in cost, from
 * everything round it, and the layout can only put it at the edge; on Hong
 * Kong Island, which is mostly hill, the flat north shore gathers in the
 * middle and the rest fans out around it.
 */
"use strict";

const Warp = {
  /* Anchors aimed for on the lattice; the access points come on top. */
  TARGET: 170,
  /* No lattice is finer than this (m): anchors a block apart measure the
   * same streets twice. */
  MIN_CELL_M: 150,
  /* Below this many anchors there is no shape to deform. */
  MIN_ANCHORS: 8,
  /* When the street junctions cannot supply this share of the target even
   * at the finest lattice, the landmass is anchored on all its junctions
   * (see anchors). */
  MIN_STREET_SHARE: 0.5,

  /* --------------------------------------------------------- local frame */
  frame(lat0, lon0) {
    const kx = 111320 * Math.cos(lat0 * Math.PI / 180), ky = 110540;
    return {
      toXY: (lon, lat) => [(lon - lon0) * kx, (lat - lat0) * ky],
      toLL: (x, y) => [lon0 + x / kx, lat0 + y / ky],
    };
  },

  /* ----------------------------------------------------- anchor selection */
  /* Junctions that a street reaches, as against a footpath, steps, a
   * service lane or a trail alone. */
  STREETS: ["trunk", "primary", "secondary", "tertiary", "residential",
    "unclassified", "living_street", "pedestrian", "cycleway"],
  streetNodes(g, mode) {
    const wanted = new Set(this.STREETS), bit = g.modeBit(mode);
    const isStreet = g.meta.classes.map(c => wanted.has(c));
    const on = new Uint8Array(g.n);
    for (let u = 0; u < g.n; u++) {
      for (let a = g.indptr[u]; a < g.indptr[u + 1]; a++) {
        if ((g.arcFlags[a] & bit) && isStreet[g.arcCls[a]]) { on[u] = 1; on[g.head[a]] = 1; }
      }
    }
    return on;
  },

  /* The neighbourhood access points on this landmass plus a lattice of its
   * junctions: in each lattice cell that has any, the junction nearest the
   * cell's centre, unless that is out in a corner (water or hillside takes
   * up the middle of the cell, and the next cell will have a better one).
   *
   * ``comp`` is the component of the mode's graph to cover (Graph.comp).
   * The spacing starts from the bounding box of the candidates and is
   * corrected a few times towards the target count, since how much of the
   * box is street differs from an island to a peninsula.
   *
   * Only street junctions are candidates. Most of Hong Kong Island is
   * country park, and a lattice over it lands mostly on trail junctions,
   * hundreds of metres up paths steeper than any street. At the balanced
   * weights a metre of path steeper than 15% costs over fifty equivalent
   * metres, so those anchors are immensely far from everything, they
   * outnumber the anchors in town, and the layout that fits them shrinks
   * the city to a dot in a star of hiking trails. The picture is meant to
   * show how the hills part the places people live, so it is anchored
   * where the streets are, and the trails ride along on the deformation in
   * between. An island with too few streets for that (Cheung Chau and
   * Lamma are villages of footpaths) is anchored on every junction
   * instead. */
  anchors(app, mode, comp, target = this.TARGET) {
    const g = app.graph, bit = g.modeBit(mode);
    const street = this.streetNodes(g, mode);
    const all = [];
    let lon0 = Infinity, lat0 = Infinity, lon1 = -Infinity, lat1 = -Infinity;
    for (let i = 0; i < g.n; i++) {
      if (g.comp(i, mode) !== comp || !comp) continue;
      all.push(i);
      const lon = g.nodeLon(i), lat = g.nodeLat(i);
      if (lon < lon0) lon0 = lon; if (lon > lon1) lon1 = lon;
      if (lat < lat0) lat0 = lat; if (lat > lat1) lat1 = lat;
    }
    const f = this.frame((lat0 + lat1) / 2, (lon0 + lon1) / 2);
    if (!all.length) return { nodes: [], frame: f, cellM: 0, box: null };
    // the whole landmass, for the displacement grid
    const lo = f.toXY(lon0, lat0), hi = f.toXY(lon1, lat1);
    const box = [lo[0], lo[1], hi[0], hi[1]];

    const choose = (member) => {
      const xs = new Float64Array(member.length), ys = new Float64Array(member.length);
      let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
      member.forEach((i, k) => {
        const p = f.toXY(g.nodeLon(i), g.nodeLat(i));
        xs[k] = p[0]; ys[k] = p[1];
        if (p[0] < x0) x0 = p[0]; if (p[0] > x1) x1 = p[0];
        if (p[1] < y0) y0 = p[1]; if (p[1] > y1) y1 = p[1];
      });
      const W = Math.max(x1 - x0, 1), H = Math.max(y1 - y0, 1);
      const lattice = (cell) => {
        const nx = Math.ceil(W / cell) + 1, ny = Math.ceil(H / cell) + 1;
        const best = new Int32Array(nx * ny).fill(-1);
        const bestD = new Float64Array(nx * ny).fill(Infinity);
        for (let k = 0; k < member.length; k++) {
          const cx = Math.floor((xs[k] - x0) / cell), cy = Math.floor((ys[k] - y0) / cell);
          const d = Math.hypot(xs[k] - (x0 + (cx + 0.5) * cell), ys[k] - (y0 + (cy + 0.5) * cell));
          const c = cy * nx + cx;
          if (d < bestD[c]) { bestD[c] = d; best[c] = member[k]; }
        }
        const out = [];
        for (let c = 0; c < best.length; c++) if (best[c] >= 0 && bestD[c] <= cell * 0.6) out.push(best[c]);
        return out;
      };
      let cell = Math.max(this.MIN_CELL_M, Math.sqrt(W * H / target));
      let picked = lattice(cell);
      for (let it = 0; it < 8; it++) {
        if (picked.length >= 0.9 * target && picked.length <= 1.15 * target) break;
        const next = Math.max(this.MIN_CELL_M, cell * Math.sqrt(Math.max(picked.length, 1) / target));
        if (Math.abs(next - cell) < 1) break;
        cell = next; picked = lattice(cell);
      }
      return { picked, cell };
    };
    let { picked, cell } = choose(all.filter(i => street[i]));
    let streetsOnly = true;
    if (picked.length < this.MIN_STREET_SHARE * target) {
      ({ picked, cell } = choose(all));
      streetsOnly = false;
    }

    const nodes = new Set();
    const pts = app.DATA.points[mode] || {};
    const ok = i => (g.nodeFlags[i] & bit) !== 0;
    for (const name of Object.keys(pts)) {
      const n = app.nodeGrid.nearest(pts[name][0], pts[name][1], ok);
      if (n >= 0 && g.comp(n, mode) === comp) nodes.add(n);
    }
    for (const n of picked) nodes.add(n);
    return { nodes: Array.from(nodes), frame: f, cellM: cell, box, streetsOnly };
  },

  /* ------------------------------------------------------- cost matrix */
  /* The cost of every arc under ``w``, once: the matrix runs one search per
   * anchor over the same weights. Infinity closes an arc to the mode. */
  arcCosts(graph, mode, w) {
    const bit = graph.modeBit(mode), mult = graph.multipliers(mode, w);
    const len = graph.lengths(w.stress), pushK = graph.pushFactor(mode, w);
    const cost = new Float64Array(graph.m);
    for (let a = 0; a < graph.m; a++) {
      cost[a] = (graph.arcFlags[a] & bit) ? graph.arcCost(a, w, mult, len, pushK) : Infinity;
    }
    return cost;
  },

  /* Costs from one junction to the anchors: Dijkstra over the precomputed
   * arc costs, stopped once every anchor has settled. ``slot`` maps a node
   * to its anchor index plus one. The same search as Graph.distances,
   * which prices every arc afresh each time it is called. */
  costsFrom(graph, src, cost, slot, K, scratch) {
    const { dist, seen, heap } = scratch, stamp = ++scratch.stamp;
    const { indptr, head } = graph;
    const out = new Float64Array(K).fill(Infinity);
    heap.n = 0;
    dist[src] = 0; seen[src] = stamp; heap.push(0, src);
    let left = K;
    while (heap.n > 0 && left > 0) {
      const u = heap.pop(), du = heap.topKey;
      if (du > dist[u]) continue;            // a stale entry: u settled sooner
      if (slot[u] && out[slot[u] - 1] === Infinity) { out[slot[u] - 1] = du; left--; }
      for (let a = indptr[u]; a < indptr[u + 1]; a++) {
        const c = cost[a];
        if (c === Infinity) continue;
        const v = head[a], nd = du + c;
        if (seen[v] !== stamp || nd < dist[v]) { seen[v] = stamp; dist[v] = nd; heap.push(nd, v); }
      }
    }
    return out;
  },

  /* The whole matrix, symmetrised. Yields to the page between searches
   * (``tick`` resolves after a repaint) so a large landmass does not
   * freeze the tab, and reports progress as a fraction.
   *
   * Two anchors of one landmass can have a way between them in one
   * direction only (a cyclist's landmass takes in the streets a one-way
   * system leads into and not back out of, as Wan Chai North is on Hong
   * Kong Island); the pair then takes the cost of the way there is. */
  async costMatrix(graph, nodes, mode, w, onProgress, tick) {
    const K = nodes.length, D = new Float64Array(K * K);
    const cost = this.arcCosts(graph, mode, w);
    const slot = new Int32Array(graph.n);
    nodes.forEach((n, i) => { slot[n] = i + 1; });
    const scratch = { dist: new Float64Array(graph.n), seen: new Int32Array(graph.n),
                      heap: new MinHeap(), stamp: 0 };
    let last = performance.now();
    for (let i = 0; i < K; i++) {
      D.set(this.costsFrom(graph, nodes[i], cost, slot, K, scratch), i * K);
      if (tick && performance.now() - last > 60) {
        if (onProgress) onProgress((i + 1) / K);
        await tick();
        last = performance.now();
      }
    }
    // symmetrise: the cost of a round trip, halved
    for (let i = 0; i < K; i++) for (let j = i + 1; j < K; j++) {
      let a = D[i * K + j], b = D[j * K + i];
      if (!isFinite(a)) a = b; if (!isFinite(b)) b = a;
      const m = isFinite(a) ? (a + b) / 2 : 0;
      D[i * K + j] = m; D[j * K + i] = m;
    }
    return D;
  },

  /* -------------------------------------------------- stress majorisation */
  smacof(D, X0, iters = 150) {
    const K = X0.length;
    let X = X0.map(p => [p[0], p[1]]);
    const stress = X => {
      let s = 0, t = 0;
      for (let i = 0; i < K; i++) for (let j = i + 1; j < K; j++) {
        const d = D[i * K + j];
        const e = Math.hypot(X[i][0] - X[j][0], X[i][1] - X[j][1]);
        s += (e - d) * (e - d); t += d * d;
      }
      return Math.sqrt(s / t);
    };
    let prev = stress(X);
    for (let it = 0; it < iters; it++) {
      // Guttman transform with unit weights: X' = (1/K) B(X) X
      const Y = X.map(() => [0, 0]);
      for (let i = 0; i < K; i++) {
        let bii = 0;
        for (let j = 0; j < K; j++) {
          if (i === j) continue;
          const dx = X[i][0] - X[j][0], dy = X[i][1] - X[j][1];
          const e = Math.hypot(dx, dy);
          const bij = e > 1e-9 ? -D[i * K + j] / e : 0;
          Y[i][0] += bij * X[j][0]; Y[i][1] += bij * X[j][1];
          bii -= bij;
        }
        Y[i][0] += bii * X[i][0]; Y[i][1] += bii * X[i][1];
        Y[i][0] /= K; Y[i][1] /= K;
      }
      X = Y;
      const s = stress(X);
      if (prev - s < 1e-7) { prev = s; break; }
      prev = s;
    }
    return { X, stress: prev };
  },

  /* ------------------------------------------- Procrustes onto geography */
  /* Rotate, reflect and scale X so it sits over G as closely as possible. */
  procrustes(X, G) {
    const K = X.length;
    const mean = P => P.reduce((a, p) => [a[0] + p[0] / K, a[1] + p[1] / K], [0, 0]);
    const mx = mean(X), mg = mean(G);
    const Xc = X.map(p => [p[0] - mx[0], p[1] - mx[1]]);
    const Gc = G.map(p => [p[0] - mg[0], p[1] - mg[1]]);
    const fit = reflect => {
      const Xr = reflect ? Xc.map(p => [-p[0], p[1]]) : Xc;
      let a = 0, b = 0, c = 0, d = 0, nx = 0;
      for (let i = 0; i < K; i++) {
        a += Xr[i][0] * Gc[i][0]; b += Xr[i][0] * Gc[i][1];
        c += Xr[i][1] * Gc[i][0]; d += Xr[i][1] * Gc[i][1];
        nx += Xr[i][0] ** 2 + Xr[i][1] ** 2;
      }
      const th = Math.atan2(b - c, a + d);
      const cs = Math.cos(th), sn = Math.sin(th);
      const s = ((a + d) * cs + (b - c) * sn) / nx;
      const out = Xr.map(p => [mg[0] + s * (cs * p[0] - sn * p[1]),
                               mg[1] + s * (sn * p[0] + cs * p[1])]);
      let err = 0;
      for (let i = 0; i < K; i++) err += (out[i][0] - G[i][0]) ** 2 + (out[i][1] - G[i][1]) ** 2;
      return { out, err };
    };
    const r0 = fit(false), r1 = fit(true);
    return r0.err <= r1.err ? r0.out : r1.out;
  },

  /* ---------------------------------------------------- thin-plate spline */
  /* Interpolates the anchors' displacements to any point. Coordinates are
   * in kilometres to keep the system well conditioned; `smooth` relaxes the
   * exact fit a little so no single anchor can tear the map. */
  tps(P, V, smooth = 0.05) {
    const K = P.length, n = K + 3;
    const phi = r => (r > 1e-12 ? r * r * Math.log(r) : 0);
    const solve = rhs => {
      const A = Array.from({ length: n }, () => new Float64Array(n));
      const bvec = new Float64Array(n);
      for (let i = 0; i < K; i++) {
        for (let j = 0; j < K; j++) {
          A[i][j] = phi(Math.hypot(P[i][0] - P[j][0], P[i][1] - P[j][1]));
        }
        A[i][i] += smooth;
        A[i][K] = 1; A[i][K + 1] = P[i][0]; A[i][K + 2] = P[i][1];
        A[K][i] = 1; A[K + 1][i] = P[i][0]; A[K + 2][i] = P[i][1];
        bvec[i] = rhs[i];
      }
      // Gaussian elimination with partial pivoting
      for (let c = 0; c < n; c++) {
        let piv = c;
        for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[piv][c])) piv = r;
        if (piv !== c) { [A[c], A[piv]] = [A[piv], A[c]]; const t = bvec[c]; bvec[c] = bvec[piv]; bvec[piv] = t; }
        const p = A[c][c] || 1e-12;
        for (let r = c + 1; r < n; r++) {
          const f = A[r][c] / p;
          if (!f) continue;
          for (let k = c; k < n; k++) A[r][k] -= f * A[c][k];
          bvec[r] -= f * bvec[c];
        }
      }
      const x = new Float64Array(n);
      for (let r = n - 1; r >= 0; r--) {
        let s = bvec[r];
        for (let k = r + 1; k < n; k++) s -= A[r][k] * x[k];
        x[r] = s / (A[r][r] || 1e-12);
      }
      return x;
    };
    const wx = solve(V.map(v => v[0])), wy = solve(V.map(v => v[1]));
    const px = Float64Array.from(P.map(p => p[0])), py = Float64Array.from(P.map(p => p[1]));
    return (x, y) => {
      let ux = wx[K] + wx[K + 1] * x + wx[K + 2] * y;
      let uy = wy[K] + wy[K + 1] * x + wy[K + 2] * y;
      for (let k = 0; k < K; k++) {
        const r = Math.hypot(x - px[k], y - py[k]);
        if (r > 1e-12) { const f = r * r * Math.log(r); ux += wx[k] * f; uy += wy[k] * f; }
      }
      return [ux, uy];
    };
  },

  /* The spline costs one logarithm per anchor per point, and there are some
   * 800,000 street vertices to move. It is smooth on the scale of the
   * anchor spacing, so it is sampled once on a grid many times finer than
   * that (at most some 60,000 points, over the landmass and a margin) and
   * read back by bilinear interpolation. Outside the grid the spline itself
   * answers. ``disp`` and the result take and return kilometres. */
  field(disp, box, cellM) {
    const margin = Math.max(2 * cellM, 1500);
    const x0 = (box[0] - margin) / 1000, y0 = (box[1] - margin) / 1000;
    const W = (box[2] - box[0] + 2 * margin) / 1000, H = (box[3] - box[1] + 2 * margin) / 1000;
    const step = Math.max(cellM / 16000, Math.sqrt(W * H / 60000), 0.02);
    const nx = Math.ceil(W / step) + 1, ny = Math.ceil(H / step) + 1;
    const ux = new Float32Array(nx * ny), uy = new Float32Array(nx * ny);
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const u = disp(x0 + i * step, y0 + j * step);
      ux[j * nx + i] = u[0]; uy[j * nx + i] = u[1];
    }
    return (x, y) => {
      const fx = (x - x0) / step, fy = (y - y0) / step;
      if (!(fx >= 0 && fy >= 0 && fx < nx - 1 && fy < ny - 1)) return disp(x, y);
      const i = Math.floor(fx), j = Math.floor(fy), a = fx - i, b = fy - j;
      const k = j * nx + i;
      return [
        (ux[k] * (1 - a) + ux[k + 1] * a) * (1 - b) + (ux[k + nx] * (1 - a) + ux[k + nx + 1] * a) * b,
        (uy[k] * (1 - a) + uy[k + 1] * a) * (1 - b) + (uy[k + nx] * (1 - a) + uy[k + nx + 1] * a) * b,
      ];
    };
  },

  /* ------------------------------------------------------------ driver */
  /* Build the warp of one landmass (component ``comp`` of the mode's graph)
   * for a weighting. Resolves to a transform lon,lat -> warped lon,lat plus
   * diagnostics; ``small`` is set, and the transform is the identity, when
   * the component has too few junctions to say anything. */
  async build(app, mode, w, comp, onProgress) {
    const t0 = performance.now();
    const tick = () => new Promise(resolve => setTimeout(resolve, 0));
    const { nodes, frame, cellM, box } = this.anchors(app, mode, comp);
    const g = app.graph, K = nodes.length;
    if (K < this.MIN_ANCHORS) {
      return { transform: (lon, lat) => [lon, lat], small: true, stress: 0, anchors: K,
               cellM, meanShiftM: 0, ms: performance.now() - t0, anchorLL: [] };
    }
    const G = nodes.map(n => frame.toXY(g.nodeLon(n), g.nodeLat(n)));
    const D = await this.costMatrix(g, nodes, mode, w, onProgress, tick);
    // start from geography scaled to the cost scale, so the result is the
    // smallest deformation that fits
    let sumD = 0, sumG = 0;
    for (let i = 0; i < K; i++) for (let j = i + 1; j < K; j++) {
      sumD += D[i * K + j]; sumG += Math.hypot(G[i][0] - G[j][0], G[i][1] - G[j][1]);
    }
    const scale = sumG > 0 ? sumD / sumG : 1;
    const X0 = G.map(p => [p[0] * scale, p[1] * scale]);
    const { X, stress } = this.smacof(D, X0);
    const Xa = this.procrustes(X, G);
    const Pkm = G.map(p => [p[0] / 1000, p[1] / 1000]);
    const Vkm = Xa.map((p, i) => [(p[0] - G[i][0]) / 1000, (p[1] - G[i][1]) / 1000]);
    await tick();
    const disp = this.field(this.tps(Pkm, Vkm), box, cellM);
    const transform = (lon, lat) => {
      const [x, y] = frame.toXY(lon, lat);
      const [ux, uy] = disp(x / 1000, y / 1000);
      return frame.toLL(x + ux * 1000, y + uy * 1000);
    };
    // how much the city stretched, as the mean anchor displacement
    let moved = 0;
    for (let i = 0; i < K; i++) moved += Math.hypot(Xa[i][0] - G[i][0], Xa[i][1] - G[i][1]);
    return {
      transform, small: false, stress, anchors: K, cellM, meanShiftM: moved / K,
      ms: performance.now() - t0,
      anchorLL: nodes.map((n, i) => ({ from: [g.nodeLat(n), g.nodeLon(n)],
                                       to: frame.toLL(Xa[i][0], Xa[i][1]).reverse() })),
    };
  },

  /* Every street vertex under a transform, in the Web Mercator fractions
   * the street canvas draws from (Geometry.mercator in engine.js). Only the
   * edges whose landmass is ``land`` move; the rest keep their place. */
  warpMercator(geom, edgeLand, land, transform) {
    const real = geom.mercator(), c = geom.coords, starts = geom.starts;
    const mx = Float32Array.from(real.mx), my = Float32Array.from(real.my);
    for (let i = 0; i < geom.nEdges; i++) {
      if (edgeLand[i] !== land) continue;
      for (let k = starts[i] >> 1, e = starts[i + 1] >> 1; k < e; k++) {
        const [lon, lat] = transform(c[2 * k], c[2 * k + 1]);
        const s = Math.sin(lat * Math.PI / 180);
        mx[k] = (lon + 180) / 360 - real.x0;
        my[k] = 0.5 - Math.log((1 + s) / (1 - s)) / (4 * Math.PI) - real.y0;
      }
    }
    return { mx, my };
  },

  /* Apply a transform to a flat [lon,lat,...] Float32Array. */
  warpCoords(coords, transform) {
    const out = new Float32Array(coords.length);
    for (let k = 0; k < coords.length; k += 2) {
      const [lon, lat] = transform(coords[k], coords[k + 1]);
      out[k] = lon; out[k + 1] = lat;
    }
    return out;
  },

  /* Apply a transform to a GeoJSON layer. With ``keep``, a feature it
   * rejects (one on another landmass) is passed through where it is and
   * marked ``off``, so the caller can draw it out of the way. */
  warpGeoJSON(gj, transform, keep) {
    const walk = c => (typeof c[0] === "number")
      ? transform(c[0], c[1]) : c.map(walk);
    return {
      type: "FeatureCollection",
      features: gj.features.map(f => {
        const on = !keep || keep(f);
        return {
          type: "Feature", properties: f.properties, off: !on,
          geometry: on ? { type: f.geometry.type, coordinates: walk(f.geometry.coordinates) }
            : f.geometry,
        };
      }),
    };
  },
};
window.Warp = Warp;
