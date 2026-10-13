"""Pack the routable graph into a compact binary payload for the browser.

The interactive map used to ship ~10,000 precomputed routes, which meant it
could only answer questions about the 36 neighborhood access points.  It is
cheaper *and* far more useful to ship the graph itself: 168,000 directed arcs
and 70,000 nodes pack into a few megabytes, and a Dijkstra over that runs in
well under a second in JavaScript.  The map can then route between any two
points the viewer clicks, under any cost weights, with the cost model
evaluated in the browser exactly as Python evaluates it.

Packing
-------
Everything is a typed array.  All of them, plus the encoded geometry and the
vector overlays, are concatenated into a single buffer, gzipped, and
base64-encoded once; the browser inflates it with ``DecompressionStream`` and
takes ``TypedArray`` views straight onto the result.  That avoids parsing
megabytes of JSON numbers, and because the arrays are mostly small integers
with long runs of zeros, compression takes the payload down by roughly a
factor of three.  Quantisation is chosen so it is invisible at the scale the
metrics are reported:

* coordinates: ``int32`` micro-degrees (~0.1 m)
* lengths and steep distances: ``uint16`` in 5 cm units
* climbing: ``uint16`` centimetres
* gradients: ``int16`` hundredths of a percent

Geometry is one concatenated Google-encoded polyline string with per-edge
character offsets, which is already ASCII and so costs nothing to embed.

Parity with Python
------------------
Per-mode arc availability is computed here exactly as
``routing.build_route_graph`` computes it, *including* the restriction to the
landmasses worth keeping (Hong Kong Island, Kowloon and the New Territories,
Lantau, the outlying islands), so a route found in the browser is the same
route Python finds. ``tests/test_webgraph.py`` asserts that on real data.
"""
from __future__ import annotations

import base64
import gzip

import numpy as np
import pandas as pd

from .bikeways import stress as bike_stress
from .config import (BIKE_PUSH_MULTIPLIER, ELEVATION, GRADE_THRESHOLDS,
                     MIN_RELIABLE_GRADE_LENGTH_M, MODES, ROUTING_PROFILES)
from .utils import get_logger, step

log = get_logger("hk_flat_routes.webgraph")

_TH = [int(t * 100) for t in GRADE_THRESHOLDS]

#: Quantisation scales. Chosen so that round-trip error is far below the
#: precision at which any metric is reported.
DM = 20.0          # 5 cm units per metre (max edge 1212 m fits uint16)
CM = 100.0         # centimetres per metre
GRADE_Q = 10000.0  # int16 units per unit gradient (0.01% resolution)
COORD_Q = 1e6      # micro-degrees
STRESS_Q = 100.0   # hundredths of a comfort multiplier


def _b64(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def _u16(values, scale: float, what: str = "") -> np.ndarray:
    """Quantise to uint16.  With ``what`` given, a value that does not fit
    is an error rather than a silent clamp: a clamped length or climb makes
    the browser's route totals quietly wrong."""
    q = np.round(np.asarray(values, dtype="float64") * scale)
    if what and q.size and (np.nanmax(q) > 65535 or np.nanmin(q) < 0):
        raise OverflowError(
            f"{what}: {np.nanmin(q) / scale:g}..{np.nanmax(q) / scale:g} does "
            f"not fit uint16 at 1/{scale:g}")
    return np.clip(q, 0, 65535).astype("<u2")


def _i16(values, scale: float, what: str = "") -> np.ndarray:
    q = np.round(np.asarray(values, dtype="float64") * scale)
    if what and q.size and (np.nanmax(q) > 32767 or np.nanmin(q) < -32768):
        raise OverflowError(
            f"{what}: {np.nanmin(q) / scale:g}..{np.nanmax(q) / scale:g} does "
            f"not fit int16 at 1/{scale:g}")
    return np.clip(q, -32768, 32767).astype("<i2")


def encode_polyline(coords, precision: int = 5) -> str:
    """Google encoded-polyline format for a sequence of (lon, lat)."""
    factor = 10 ** precision
    out: list[str] = []
    prev_lat = prev_lon = 0

    def enc(v: int) -> str:
        v = ~(v << 1) if v < 0 else (v << 1)
        chunks = []
        while v >= 0x20:
            chunks.append(chr((0x20 | (v & 0x1f)) + 63))
            v >>= 5
        chunks.append(chr(v + 63))
        return "".join(chunks)

    for lon, lat in coords:
        ilat = int(round(lat * factor))
        ilon = int(round(lon * factor))
        out.append(enc(ilat - prev_lat))
        out.append(enc(ilon - prev_lon))
        prev_lat, prev_lon = ilat, ilon
    return "".join(out)


# --------------------------------------------------------------------------
def _mode_availability(directed: pd.DataFrame, mode: str,
                       node_index: dict):
    """Per-arc availability for a mode, matching ``build_route_graph``.

    Keeps the same landmasses Python keeps (``component_ranks``), with the
    same arcs, so that anything the browser can route, Python can route
    identically.  Returns the arc mask and, per node, the rank of its
    landmass plus one (0 for a node the mode cannot use).
    """
    from .routing import component_ranks

    ok = directed[f"{mode}_traversable"].to_numpy()
    fi = directed["from_node"].map(node_index).to_numpy()
    ti = directed["to_node"].map(node_index).to_numpy()
    n = len(node_index)
    # nodes the mode never touches are components of one, and so are dropped
    comp, _core = component_ranks(fi[ok], ti[ok], n, label=f"{mode}: ")
    avail = ok & (comp[fi] >= 0) & (comp[ti] >= 0)
    sizes = np.bincount(comp[comp >= 0])
    log.info("  %s: %d of %d arcs available on %d landmass(es) (largest %s nodes)",
             mode, int(avail.sum()), int(ok.sum()), sizes.size,
             ", ".join(f"{int(v):,}" for v in sizes[:6]))
    return avail, (comp + 1).astype("<u1")


#: Vertices that move an edge's drawn line by less than this (m) are dropped
#: from the browser payload.
GEOM_TOLERANCE_M = 1.5

#: Arrays only the explorer reads.  The route page is the one people load on
#: a phone, so its bundle leaves them out (about a fifth of the file).
EXPLORER_ONLY_ARRAYS = frozenset(
    {"arc_meangrade", "edge_bucket", "edge_maxgrade", "edge_avggrade",
     "edge_gainkm", "edge_lowstress"}
    | {f"arc_th{t}" for t in _TH})

#: A point on each landmass someone might try to route across to, and what
#: to call it when they do.  A component is named after the first of these
#: that has one of its nodes within 1.2 km.
LANDMASSES = (
    ("Hong Kong Island", 114.1720, 22.2790),
    ("Kowloon and the New Territories", 114.1717, 22.3200),
    ("Lantau", 113.9430, 22.2870),
    ("Lamma Island", 114.1100, 22.2260),
    ("Cheung Chau", 114.0290, 22.2090),
    ("Peng Chau", 114.0400, 22.2850),
    ("Ma Wan", 114.0600, 22.3500),
    ("Po Toi", 114.2620, 22.1660),
    ("Tung Lung Chau", 114.2890, 22.2530),
    ("Tap Mun", 114.3610, 22.4720),
    ("Tung Ping Chau", 114.4300, 22.5440),
    ("Kat O", 114.2930, 22.5390),
    ("Ap Chau", 114.2700, 22.5510),
    ("Kau Sai Chau", 114.3170, 22.3640),
    ("Sharp Island", 114.2890, 22.3640),
    ("Hei Ling Chau", 114.0330, 22.2520),
    ("the airport's boundary crossing island", 113.9520, 22.3160),
)


def _component_names(lon, lat, comp, over=None) -> list[str]:
    """A name for each landmass of one mode, by rank ("" if none).

    ``over`` is ``(comp, names)`` of the walking landmasses.  A bicycle's
    landmass that no reference point names is a piece of a walker's (the
    cycle tracks of Discovery Bay, the streets of Ap Lei Chau beyond a
    bridge closed to bicycles), and is called after the one most of its
    nodes lie on.  Two of a mode's landmasses can therefore carry the same
    name, which says only where they are: whether a route exists is a
    matter of their ranks.
    """
    n = int(comp.max()) if len(comp) else 0
    out = [""] * n
    kx = 111320.0 * np.cos(np.radians(22.35)); ky = 110540.0
    taken: set[int] = set()
    for name, x, y in LANDMASSES:
        ok = comp > 0
        if not ok.any():
            break
        d2 = ((lon - x) * kx) ** 2 + ((lat - y) * ky) ** 2
        d2 = np.where(ok, d2, np.inf)
        i = int(np.argmin(d2))
        if d2[i] > 1200.0 ** 2:
            continue
        c = int(comp[i])
        if c in taken:
            continue
        taken.add(c)
        out[c - 1] = name
    if over is not None:
        base, base_names = over
        for c in range(1, n + 1):
            under = base[(comp == c) & (base > 0)]
            if not out[c - 1] and len(under):
                out[c - 1] = base_names[int(np.bincount(under).argmax()) - 1]
    return out


def build_payload(edges, directed: pd.DataFrame) -> dict:
    """Pack nodes, arcs, geometry and display attributes for the browser."""
    with step("packing the routable graph for the browser", log):
        # ---- nodes -------------------------------------------------
        nodes = pd.unique(pd.concat([directed["from_node"],
                                     directed["to_node"]], ignore_index=True))
        node_index = {n: i for i, n in enumerate(nodes)}

        ll = edges.to_crs("EPSG:4326")
        lon = np.zeros(len(nodes)); lat = np.zeros(len(nodes))
        seen = np.zeros(len(nodes), dtype=bool)
        for u, v, geom in zip(edges["u"], edges["v"], ll.geometry):
            c = geom.coords
            for node, (x, y) in ((u, c[0]), (v, c[-1])):
                i = node_index.get(node)
                if i is not None and not seen[i]:
                    lon[i] = x; lat[i] = y; seen[i] = True

        # ---- node elevations, for route elevation profiles ---------
        # Elevations are reconciled to one value per intersection upstream,
        # so a profile sampled at arc boundaries is exact at every corner.
        za = directed[["from_node", "start_elev"]].rename(
            columns={"from_node": "node", "start_elev": "z"})
        zb = directed[["to_node", "end_elev"]].rename(
            columns={"to_node": "node", "end_elev": "z"})
        zz = pd.concat([za, zb], ignore_index=True).groupby("node")["z"].median()
        node_z = zz.reindex(nodes).fillna(0.0).to_numpy()

        # ---- per-mode availability ---------------------------------
        flags = np.zeros(len(directed), dtype="<u1")
        node_flags = np.zeros(len(nodes), dtype="<u1")
        node_comp = {}
        for bit, mode in enumerate(("walk", "bike")):
            avail, comp = _mode_availability(directed, mode, node_index)
            flags |= (avail.astype("<u1") << bit)
            node_flags |= ((comp > 0).astype("<u1") << bit)
            # which landmass a node is on, per mode: a route exists only
            # between two nodes that share one
            node_comp[mode] = comp
        keep = flags > 0
        d = directed[keep].reset_index(drop=True)
        flags = flags[keep]
        # bit 2 marks a reverse traversal, so the geometry is drawn backwards
        flags |= ((d["direction"].to_numpy() == "rev").astype("<u1") << 2)

        # ---- arcs, sorted into CSR order ---------------------------
        fi = d["from_node"].map(node_index).to_numpy()
        ti = d["to_node"].map(node_index).to_numpy()
        order = np.argsort(fi, kind="stable")
        fi, ti, d, flags = fi[order], ti[order], d.iloc[order], flags[order]
        indptr = np.zeros(len(nodes) + 1, dtype="<i4")
        np.add.at(indptr, fi + 1, 1)
        indptr = np.cumsum(indptr).astype("<i4")

        classes = sorted(set(edges["cls"].dropna().unique())
                         | set(d["cls"].dropna().unique()))
        cls_idx = {c: i for i, c in enumerate(classes)}

        # Loss, maximum gradient and mean absolute gradient are carried
        # per arc rather than re-derived in the browser, so that the map
        # reports exactly the same figures as the analysis outputs. Deriving
        # them from the node elevation series would be close but not equal,
        # and a map that disagrees with its own CSV is worse than a slightly
        # larger file.
        #
        # A route's steepest pitch is the one figure the engine does put
        # together itself (``routing.arc_pitch``): an arc's sampled maximum
        # from 15 m up, its net climb over its length below that.  It knows
        # the length only as rounded here, and 234 of the full build's
        # 520,489 arcs are under 15 m and round up to it.  Those are packed
        # one unit short, so that the page and the pair table judge every
        # arc on the same figure.
        arc_len = _u16(d["length_m"], DM, "arc length (m)")
        short = d["length_m"].to_numpy() < MIN_RELIABLE_GRADE_LENGTH_M
        arc_len[short] = np.minimum(
            arc_len[short], int(np.ceil(MIN_RELIABLE_GRADE_LENGTH_M * DM)) - 1)
        # A bicycle's dismount-and-push links, where a bicycle can take the
        # arc at all: a link on a fragment too small to keep is, to the
        # walkers who alone can reach it, a footway like any other.
        push = np.zeros(len(d), dtype=bool)
        if "bike_push" in d.columns:
            push = (d["bike_push"].to_numpy(dtype=bool, na_value=False)
                    & ((flags >> 1) & 1).astype(bool))
        arcs = {
            "head": ti.astype("<i4"),
            "edge": d["edge_id"].to_numpy().astype("<i4"),
            "len": arc_len,
            "gain": _u16(d["cum_gain"], CM, "arc climb (m)"),
            "loss": _u16(d["cum_loss"], CM, "arc descent (m)"),
            "maxgrade": _i16(d["max_grade"], GRADE_Q),
            "meangrade": _u16(d["mean_abs_grade"], GRADE_Q),
            "cls": d["cls"].map(cls_idx).fillna(0).to_numpy().astype("<u1"),
            "flags": flags,
            # 1 on a pushed link, whose length the engine multiplies by
            # ``meta.push_multiplier`` in bike mode wherever it applies the
            # class multipliers, and in the lengths it weights by
            # ``edge_stress``
            "push": push.astype("<u1"),
        }
        for t in _TH:
            arcs[f"th{t}"] = _u16(d[f"d_above_{t}"], DM)

        # ---- geometry: one polyline string with character offsets ---
        # Hong Kong's ways are drawn finely (trails most of all): five to
        # six vertices an edge against San Francisco's two and a half, which
        # would make geometry the largest thing in the file.  Vertices that
        # move the line by less than GEOM_TOLERANCE_M are dropped for
        # display.  Lengths, climbing and elevation were measured on the
        # full geometry and are carried separately, so nothing the router
        # counts changes; the ends of an edge never move.
        drawn = edges.geometry.simplify(GEOM_TOLERANCE_M, preserve_topology=False)
        drawn = drawn.set_crs(edges.crs).to_crs("EPSG:4326")
        geom_parts: list[str] = []
        offs = np.zeros(len(edges) + 1, dtype="<i4")
        pos = n_coords = 0
        for i, geom in enumerate(drawn):
            coords = list(geom.coords)
            n_coords += len(coords)
            s = encode_polyline(coords)
            geom_parts.append(s)
            pos += len(s)
            offs[i + 1] = pos
        geom = "".join(geom_parts)
        log.info("  geometry: %d vertices drawn of %d (%.1f m tolerance)",
                 n_coords, sum(len(g.coords) for g in edges.geometry),
                 GEOM_TOLERANCE_M)

        # ---- per-edge display attributes ---------------------------
        names = edges["name"].fillna("")
        name_values = sorted(set(names) - {""})
        if len(name_values) >= 65535:
            raise OverflowError(f"{len(name_values)} street names do not fit "
                                "the uint16 name index")
        name_idx = {n: i + 1 for i, n in enumerate(name_values)}
        # the Chinese name of each street that has both, aligned to
        # ``names``: the page shows the English one and searches on either
        zh_of: dict[str, str] = {}
        if "name_zh" in edges.columns:
            pairs = edges[["name", "name_zh"]].dropna()
            for en, grp in pairs.groupby("name")["name_zh"]:
                zh_of[en] = grp.value_counts().index[0]
        names_zh = [zh_of.get(n, "") for n in name_values]
        buckets = np.digitize(edges["max_abs_grade"].fillna(0.0).to_numpy(),
                              [0.03, 0.05, 0.08, 0.10, 0.15]).astype("<u1")
        # Climb per kilometre is the mean of the two directions of travel,
        # as the corridor table and the validation report have it.  The
        # forward climb alone, which upstream ships, is nothing at all on a
        # block that happens to be drawn downhill: 18,574 of the full
        # build's 50,279 walkable edges of 15 m or more with a mean gradient
        # of 3% or more read 0 m per kilometre in the explorer.
        climb = edges["cum_gain_fwd"]
        if "cum_loss_fwd" in edges.columns:
            climb = (climb + edges["cum_loss_fwd"]) / 2.0
        # edge ids are a contiguous 0..n-1 range by construction, so storing
        # them (and a reverse lookup in the browser) would be pure overhead
        assert np.array_equal(edges["edge_id"].to_numpy(),
                              np.arange(len(edges))), \
            "edge ids are expected to be a contiguous range"
        edge_attrs = {
            "name": names.map(lambda n: name_idx.get(n, 0)).to_numpy().astype("<u2"),
            "cls": edges["cls"].map(cls_idx).fillna(0).to_numpy().astype("<u1"),
            "bucket": buckets,
            "maxgrade": _u16(edges["max_abs_grade"].fillna(0.0), GRADE_Q),
            # net change over length: on an edge a metre or two long that
            # can run far past the ceiling every sampled gradient is held to
            "avggrade": _i16(edges["avg_grade_fwd"].fillna(0.0).clip(
                -ELEVATION.max_plausible_grade, ELEVATION.max_plausible_grade),
                GRADE_Q),
            "len": _u16(edges["length_m"], DM, "edge length (m)"),
            "gainkm": _u16((climb / (edges["length_m"] / 1000.0).clip(lower=1e-6))
                           .clip(0, 6000), DM),
            "lowstress": edges["low_stress"].fillna(False).to_numpy().astype("<u1"),
            # bike comfort multiplier of the class alone, hundredths (100 =
            # an ordinary block), a byte an edge.  The push factor of a
            # dismount-and-push link is not in it: the engine multiplies
            # that in from ``arc_push`` when it builds its comfort-weighted
            # lengths (the ``Graph`` constructor in ``web/engine.js``), and
            # a weight that already held it would be charged nine times
            # over.  ``tests/test_webgraph.py`` runs the engine on a payload
            # built here to hold the two sides to that.
            "stress": np.clip(np.round(bike_stress(edges, push=False) * STRESS_Q),
                              1, 255).astype("<u1"),
        }

        arrays: dict[str, np.ndarray] = {
            "node_lon": np.round(lon * COORD_Q).astype("<i4"),
            "node_lat": np.round(lat * COORD_Q).astype("<i4"),
            "node_flags": node_flags,
            "node_comp_walk": node_comp["walk"],
            "node_comp_bike": node_comp["bike"],
            "node_elev": _i16(node_z, DM, "node elevation (m)"),
            "indptr": indptr,
            "geom_off": offs,
        }
        for k, v in arcs.items():
            arrays["arc_" + k] = v
        for k, v in edge_attrs.items():
            arrays["edge_" + k] = v

        landmass_names = {"walk": _component_names(lon, lat, node_comp["walk"])}
        landmass_names["bike"] = _component_names(
            lon, lat, node_comp["bike"],
            over=(node_comp["walk"], landmass_names["walk"]))
        meta = {
            "n_nodes": int(len(nodes)),
            "n_arcs": int(len(ti)),
            "n_edges": int(len(edges)),
            "classes": classes,
            "names": name_values,
            "names_zh": names_zh,
            "components": landmass_names,
            # nodes on each landmass, in the same order
            "component_nodes": {mode: np.bincount(node_comp[mode])[1:].tolist()
                                for mode in node_comp},
            "thresholds": _TH,
            "scales": {"dm": DM, "cm": CM, "grade": GRADE_Q, "coord": COORD_Q,
                       "stress": STRESS_Q},
            "profiles": {
                name: {
                    "alpha": w.alpha, "beta": w.beta, "gamma": w.gamma,
                    "penalties": list(w.threshold_penalties),
                    "extreme": w.extreme_extra,
                    "use_class_multiplier": bool(w.use_class_multiplier),
                }
                for name, w in ROUTING_PROFILES.items()
            },
            "multipliers": {
                mode: [MODES[mode].class_multiplier.get(c, 1.0) for c in classes]
                for mode in MODES
            },
            "push_multiplier": BIKE_PUSH_MULTIPLIER,
            # what a route's steepest pitch is made of (``routing.arc_pitch``,
            # and ``Graph.summarise`` in the engine): an arc's sampled
            # maximum from this length (m) up, and below it the arc's net
            # climb over its length, held to this ceiling
            "min_reliable_grade_length_m": MIN_RELIABLE_GRADE_LENGTH_M,
            "max_plausible_grade": ELEVATION.max_plausible_grade,
        }
        return {"arrays": arrays, "geom": geom, "meta": meta,
                "n_coords": n_coords}


_DTYPE_TAG = {"int32": "i4", "uint32": "u4", "int16": "i2",
              "uint16": "u2", "uint8": "u1", "int8": "i1",
              "float32": "f4", "float64": "f8"}


def bundle(graph: dict, strings: dict[str, str]) -> dict:
    """Concatenate arrays and strings, gzip, and base64-encode once.

    Returns the manifest the browser needs to take views onto the inflated
    buffer, plus the encoded bundle itself.
    """
    blobs: list[bytes] = []
    manifest_arrays: dict[str, dict] = {}
    manifest_strings: dict[str, dict] = {}
    off = 0

    for name, arr in graph["arrays"].items():
        raw = np.ascontiguousarray(arr).tobytes()
        tag = _DTYPE_TAG.get(arr.dtype.name)
        if tag is None:
            raise TypeError(f"unsupported dtype {arr.dtype} for {name}")
        manifest_arrays[name] = {"t": tag, "o": off, "n": int(arr.size)}
        blobs.append(raw)
        off += len(raw)

    for name, text in strings.items():
        raw = text.encode("utf-8")
        manifest_strings[name] = {"o": off, "b": len(raw)}
        blobs.append(raw)
        off += len(raw)

    flat = b"".join(blobs)
    # mtime=0 keeps the output byte-reproducible; gzip otherwise stamps the
    # current time into the header and the map churns on every rebuild
    packed = gzip.compress(flat, compresslevel=9, mtime=0)
    log.info("  bundle: %.2f MB raw -> %.2f MB gzipped -> %.2f MB base64",
             len(flat) / 1e6, len(packed) / 1e6,
             len(packed) * 4 / 3 / 1e6)
    return {
        "manifest": {"arrays": manifest_arrays, "strings": manifest_strings,
                     "bytes": len(flat)},
        "b64": base64.b64encode(packed).decode("ascii"),
        "gz": packed,
        "meta": graph["meta"],
    }


