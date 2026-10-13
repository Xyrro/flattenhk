# Data directory

This directory is populated by the pipeline and is **not** tracked in git.

- `raw/` — cached source downloads, never modified after they land: the
  Overture Maps street extract and SAR boundary, the Town Planning Board's
  planning scheme areas, the place and address sources for the search
  index, the Lands Department 5 m terrain model and bench marks used for
  validation, and `raw/dem/`, the 3,253 zipped tiles of the CEDD 2020
  LiDAR terrain model (0.5 m). Populate with
  `python -m hk_flat_routes download` (about 14.3 GB in all, 14.1 GB of it
  the terrain tiles; the tile server manages a few MB/s, so allow an hour
  or more; the download resumes where it stopped, and a file that is not
  what was asked for, such as a maintenance page sent in place of a tile,
  is never kept).
- `processed/` — cached intermediate products: the edge table, per-edge
  elevation profiles, the directed edge metrics, the 10 m terrain overview
  the hillshade is drawn from, the neighborhood pair matrix and the route
  store. Populate with `python -m hk_flat_routes build-network` and
  `python -m hk_flat_routes analyze`.

Everything here is rebuilt from the sources listed by
`python -m hk_flat_routes sources`, with one exception that matters for
reproducing the published figures. Overture Maps hosts only its latest few
releases (three on 2026-10-13, the oldest 55 days old), and the release
this project is pinned to, 2026-09-23.1, should be expected to go the way
of the earlier ones. After that these four files, which the street network
is built from, cannot be downloaded again:

- `raw/overture_segments_hk.parquet` (40 MB), the street network;
- `raw/overture_connectors_hk.parquet` (12 MB), its junctions;
- `raw/overture_division_area_hk.parquet` (0.6 MB), the SAR boundary;
- `raw/overture_land_use_hk.parquet` (10 MB), the land use that marks the
  container terminals, power stations and military land whose service
  roads the network closes.

Keep a copy of them (they are ODbL data: © OpenStreetMap contributors, via
Overture Maps), for instance as a release asset beside the code, and put
them back into `raw/` before running the pipeline. Without them,
`download` stops with an error that names the releases still hosted;
`HKFR_OVERTURE_RELEASE=<release>` then builds from one of those instead,
which is different data, so the figures will not be the published ones
exactly. An extract that is already in `raw/` is used as it is, whatever
release is asked for. Each one written from now on records the release it
was read from (the four of the first published build do not, and are
taken for 2026-09-23.1), and `download` warns when a cached extract is
another release's. The three
`raw/overture_{infrastructure,land,places}_hk.parquet` files behind the
place search are in the same position, but the search is built without
them if they are missing.
