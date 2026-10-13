"""Command-line interface.

    python -m hk_flat_routes sources        # dataset provenance table
    python -m hk_flat_routes download       # fetch and cache all source data
    python -m hk_flat_routes build-network   # street graph + elevation + metrics
    python -m hk_flat_routes site           # the route-finder page, and only that
    python -m hk_flat_routes analyze        # pairs, Pareto, corridors, passes
    python -m hk_flat_routes validate       # checks against known ground truth
    python -m hk_flat_routes map            # interactive + static maps
    python -m hk_flat_routes report         # written analysis of the findings
    python -m hk_flat_routes all            # everything, in order
    python -m hk_flat_routes route --from "Central District" --to "The Peak Area"
    python -m hk_flat_routes route --from 旺角 --to "Kowloon Tong" --mode bike

Every stage caches its output, so re-running is cheap; pass ``--force`` to
recompute a stage from scratch.  For ``site`` and ``map`` that means the
terrain overview the hillshade is drawn from; ``validate`` and ``report``
keep no cache and always start afresh.

``site`` needs only ``build-network``: the route finder routes in the
browser and uses none of the pair analysis, so the page can be rebuilt and
published without waiting for ``analyze``.  ``all`` does not call it,
because ``map`` writes the same page along with everything else.

Neighbourhoods are the Town Planning Board's planning areas and are named
as that board names them, in English or in Chinese.  Hong Kong Island,
Kowloon with the New Territories, and Lantau are separate landmasses with
no walking or cycling link between them, so ``route`` between two of them
reports that there is no route rather than inventing one.
"""
from __future__ import annotations

import argparse
import sys

from .utils import get_logger, setup_logging, step

log = get_logger("hk_flat_routes")


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--force", action="store_true",
                   help="recompute instead of using cached artefacts")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m hk_flat_routes",
        description="Hong Kong flat-route analysis: discovering the "
                    "territory's low-elevation street network (a port of "
                    "flattensf).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    for name, helptext in [
        ("sources", "print the dataset provenance table"),
        ("download", "download and cache every source dataset"),
        ("build-network", "build the street graph, sample elevation, "
                          "compute edge metrics"),
        ("site", "build only the route-finder page (needs build-network, "
                 "not analyze)"),
        ("analyze", "neighbourhood pairs, Pareto fronts, corridors, passes"),
        ("validate", "validate the model against known ground truth"),
        ("map", "render the interactive and static maps"),
        ("report", "write the analysis of major findings"),
        ("all", "download, build-network, analyze, validate, map, report"),
    ]:
        sp = sub.add_parser(name, help=helptext)
        _add_common(sp)

    sp = sub.add_parser("sensitivity",
                        help="rebuild the pipeline under perturbed parameters")
    _add_common(sp)
    sp.add_argument("--only", nargs="*", help="run only these configuration tags")

    # internal: run by ``sensitivity`` inside each rebuilt pipeline.  Given
    # no help text it stays out of the command list (upstream passed
    # argparse.SUPPRESS, which argparse prints verbatim for a subcommand).
    sm = sub.add_parser("summarize")
    _add_common(sm)
    sm.add_argument("--tag", required=True)

    rp = sub.add_parser(
        "route", help="route between two neighbourhoods (planning areas)",
        description="Compare the routing objectives between two "
                    "neighbourhoods, named in English or Chinese, e.g. "
                    "--from \"Central District\" --to \"The Peak Area\".")
    _add_common(rp)
    rp.add_argument("--from", dest="origin", required=True,
                    help="origin neighbourhood, e.g. \"Central District\" or 中區")
    rp.add_argument("--to", dest="dest", required=True,
                    help="destination neighbourhood, e.g. \"The Peak Area\"")
    rp.add_argument("--mode", default="walk", choices=["walk", "bike"])
    rp.add_argument("--profile", default=None,
                    help="one routing objective (default: compare all four)")
    return p


def _tolerant_streams() -> None:
    """Let the output through on a console that is not UTF-8.

    The help text, the sources table and both reports carry Chinese names
    and typographic marks.  Where the console's encoding has no such
    character (cp1252 for redirected output on Western Windows) printing
    them raised ``UnicodeEncodeError``, for ``validate`` after all its work
    was done.  The encoding is left as the console wants it; a character it
    cannot show is replaced.
    """
    for stream, errors in ((sys.stdout, "replace"), (sys.stderr, "backslashreplace")):
        if getattr(stream, "errors", None) == "strict" and hasattr(stream, "reconfigure"):
            stream.reconfigure(errors=errors)


def _show(path) -> None:
    """Print a report this package wrote, which is UTF-8 whatever the locale."""
    print(path.read_text(encoding="utf-8"))


def _refresh_overview(force: bool) -> None:
    """What ``--force`` means for ``site`` and ``map``.

    Neither keeps a cache of its own, but both draw the hillshade from the
    10 m terrain overview, which is built from the tiles on disk and then
    reused until they change.  On a machine that holds the processed data
    without the tiles there is nothing to rebuild it from, and the cached
    overview stands; any other failure (a tile that cannot be read) is the
    rebuild that was asked for failing, and stops the command.
    """
    if not force:
        return
    from .elevation import build_dem_overview
    try:
        build_dem_overview(force=True)
    except FileNotFoundError as exc:
        log.warning("no terrain tiles to rebuild the overview from (%s); "
                    "using the cached one", exc)


def cmd_sources(_args) -> int:
    from . import sources
    print(sources.format_table())
    return 0


def cmd_download(args) -> int:
    from .download import download_all
    download_all(force=args.force)
    return 0


def cmd_build_network(args) -> int:
    from .pipeline import build_context
    ctx = build_context(force=args.force)
    log.info("network ready: %d edges, %d directed arcs; modes: %s",
             len(ctx.edges), len(ctx.directed),
             ", ".join(f"{m} ({g.n_nodes} nodes, {g.n_arcs} arcs)"
                       for m, g in ctx.graphs.items()))
    return 0


def cmd_site(args) -> int:
    """Build the route finder alone.

    The page is assembled by ``viz_interactive.make_route_page``; it is
    imported here, not at the top of the module, so that every other
    command works in a checkout where the web half is missing or broken,
    and this one says what it needs instead of failing on import.
    """
    try:
        from .viz_interactive import make_route_page
    except ImportError as exc:
        print("the `site` command needs hk_flat_routes.viz_interactive."
              f"make_route_page(ctx), which could not be imported: {exc}",
              file=sys.stderr)
        return 2
    from .pipeline import build_context

    ctx = build_context()
    _refresh_overview(args.force)
    with step("route-finder site", log):
        path = make_route_page(ctx)
    log.info("wrote %s", path)
    return 0


def cmd_analyze(args) -> int:
    from .pipeline import build_context, run_analysis
    ctx = build_context()
    res = run_analysis(ctx, force=args.force)
    log.info("analysis complete: %d routes, %d corridors, %d passes, "
             "%d barriers", len(res["pairs"]), len(res["corridors"]),
             len(res["passes"]), len(res["barriers"]))
    return 0


def cmd_validate(args) -> int:
    import geopandas as gpd
    from .corridors import CORRIDORS_GPKG
    from .pipeline import build_context
    from .validate import VALIDATION_MD, run_validation
    ctx = build_context()
    cor = gpd.read_file(CORRIDORS_GPKG) if CORRIDORS_GPKG.exists() else None
    run_validation(ctx, cor)
    _show(VALIDATION_MD)
    return 0


def cmd_map(args) -> int:
    import geopandas as gpd
    from .corridors import CORRIDORS_GPKG
    from .passes import read_barriers, read_basins, read_passes
    from .pipeline import build_context
    from .viz_interactive import make_interactive_map
    from .viz_static import make_backbone_map, make_grade_map

    ctx = build_context()
    cor = gpd.read_file(CORRIDORS_GPKG)
    # read through the pass module, which hands back the expected columns
    # even from a run whose pass analysis found nothing
    pz = read_passes()
    ba = read_barriers()
    bs = read_basins()

    _refresh_overview(args.force)
    with step("static maps", log):
        make_backbone_map(cor, ctx.edges, ctx.neighborhoods, pz)
        make_grade_map(ctx.edges, ctx.neighborhoods)
    with step("interactive map", log):
        make_interactive_map(ctx, cor, pz, ba, basins=bs if len(bs) else None)
    return 0


def cmd_report(args) -> int:
    from .report import write_report
    path = write_report()
    log.info("wrote %s", path)
    return 0


def _fold(text) -> str:
    """A name reduced for matching: case, spacing and punctuation ignored."""
    return "".join(ch for ch in str(text).casefold() if ch.isalnum())


def resolve_neighborhood(query: str, names, names_zh=None):
    """Find the neighbourhood a user means.  Returns ``(name, candidates)``.

    Planning-area names are long and exactly punctuated ("Tsz Wan Shan,
    Diamond Hill & San Po Kong"), and half the people asking would write
    the Chinese one.  The query is matched, ignoring case, spacing and
    punctuation, first against the whole English or Chinese name and then
    as a part of one; ``name`` is set only when exactly one neighbourhood
    fits, and otherwise ``candidates`` lists the ones that do (all of them
    when none does).
    """
    names = list(names)
    names_zh = list(names_zh) if names_zh is not None else [None] * len(names)
    q = _fold(query)
    if not q:
        return None, names
    keys = [(n, {_fold(n)} | ({_fold(z)} if isinstance(z, str) and z else set()))
            for n, z in zip(names, names_zh)]
    exact = [n for n, ks in keys if q in ks]
    if len(exact) == 1:
        return exact[0], exact
    part = [n for n, ks in keys if any(q in k for k in ks)]
    if len(part) == 1:
        return part[0], part
    return None, (exact or part or names)


def cmd_route(args) -> int:
    from .config import ROUTING_PROFILES
    from .pipeline import build_context
    from .routing import route

    ctx = build_context()
    graph = ctx.graphs[args.mode]
    points = ctx.points[args.mode]
    pts = dict(zip(points["neighborhood"], points["node"]))
    zh = dict(zip(ctx.neighborhoods["neighborhood"],
                  ctx.neighborhoods["neighborhood_zh"])) \
        if "neighborhood_zh" in ctx.neighborhoods.columns else {}
    ends = []
    for query in (args.origin, args.dest):
        name, cand = resolve_neighborhood(query, list(pts),
                                          [zh.get(n) for n in pts])
        if name is None:
            what = "matches several neighbourhoods" if len(cand) < len(pts) \
                else "is not a neighbourhood"
            print(f"{query!r} {what}. "
                  f"{'Did you mean' if len(cand) < len(pts) else 'Available'}:\n  "
                  + "\n  ".join(f"{n}  {zh.get(n) or ''}".rstrip()
                                for n in sorted(cand)), file=sys.stderr)
            return 2
        ends.append(name)
    origin, dest = ends
    print(f"\n{origin}  ->  {dest}   [{args.mode}]")

    # --- pairs that have no route: say so, and say why -----------------
    for name in (origin, dest):
        node = pts[name]
        if node is None or node != node or graph.index_of(node) is None:
            print(f"No route: {name} has no access point on the {args.mode} "
                  f"network of this build.\n")
            return 1
    if origin == dest:
        print("No route: the origin and the destination are the same "
              "neighbourhood.\n")
        return 1
    si, ti = graph.index_of(pts[origin]), graph.index_of(pts[dest])
    if not graph.connected(si, ti):
        print(f"No route: {origin} and {dest} are on different landmasses of "
              f"the {args.mode} network, and nothing in it that is open to "
              f"{'walkers' if args.mode == 'walk' else 'cyclists'} joins "
              f"them. (Ferries and railways are not part of the network.)\n")
        return 1

    names = [args.profile] if args.profile else list(ROUTING_PROFILES)
    print(f"{'objective':14s} {'km':>7s} {'climb m':>9s} {'loss m':>8s} "
          f"{'max %':>6s} {'>5% m':>7s} {'>8% m':>7s} {'vs shortest':>24s}")
    base = None
    if names != ["shortest"] and "shortest" not in names:
        # one objective was asked for: it is still compared with the
        # shortest route, which is computed for that and not printed
        base = route(graph, pts[origin], pts[dest],
                     ROUTING_PROFILES["shortest"])[1]
    for pname in names:
        arcs, s = route(graph, pts[origin], pts[dest], ROUTING_PROFILES[pname])
        if not arcs:
            # connected, yet the search found nothing: report, do not divide
            print(f"{pname:14s} {'no route found':>44s}")
            continue
        if base is None:
            base = s
        cmp_ = ""
        if pname != "shortest" and base and base["distance_m"] > 0:
            dd = 100 * (s["distance_m"] / base["distance_m"] - 1)
            dg = s["elev_gain_m"] - base["elev_gain_m"]
            cmp_ = f"{dd:+5.0f}% dist, {dg:+6.0f} m climb"
        print(f"{pname:14s} {s['distance_m']/1000:7.2f} "
              f"{s['elev_gain_m']:9.0f} {s['elev_loss_m']:8.0f} "
              f"{s['max_grade']*100:6.1f} {s['d_above_5']:7.0f} "
              f"{s['d_above_8']:7.0f} {cmp_:>24s}")
    print()
    return 0


def cmd_sensitivity(args) -> int:
    from .sensitivity import SENS_MD, run_sensitivity
    run_sensitivity(force=args.force, only=args.only)
    _show(SENS_MD)
    return 0


def cmd_summarize(args) -> int:
    from .sensitivity import summarize_current_run
    summarize_current_run(args.tag)
    return 0


def cmd_all(args) -> int:
    for fn in (cmd_download, cmd_build_network, cmd_analyze, cmd_validate,
               cmd_map, cmd_report):
        rc = fn(args)
        if rc:
            return rc
    return 0


_DISPATCH = {
    "sources": cmd_sources, "download": cmd_download,
    "build-network": cmd_build_network, "site": cmd_site,
    "analyze": cmd_analyze,
    "validate": cmd_validate, "map": cmd_map, "report": cmd_report,
    "route": cmd_route, "all": cmd_all,
    "sensitivity": cmd_sensitivity, "summarize": cmd_summarize,
}


def main(argv: list[str] | None = None) -> int:
    _tolerant_streams()
    args = build_parser().parse_args(argv)
    setup_logging(getattr(args, "verbose", False))
    return _DISPATCH[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
