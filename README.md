# flattenhk 平路香港

The flattest walking or cycling route between any two places in Hong Kong,
and every route between it and the shortest.

> **This is a Hong Kong port of [Flatten SF](https://flattensf.com/)
> ([source](https://github.com/almostimplemented/flattensf)) by
> [Drew Edwards](https://almostimplemented.com) (@almostimplemented).** The
> idea, the method, the in-browser router, the route-finder page and most of
> the code are theirs, used under the MIT licence (Copyright (c) 2026
> sf-flat-routes contributors; the notice is kept in [`LICENSE`](LICENSE)).
> What is new here is the Hong Kong data, the changes Hong Kong's terrain
> and street data forced on the method, the validation against Hong Kong
> ground truth, and the write-up below.

Hong Kong is hillier than San Francisco and hilly in a different way. The
city is a set of narrow flat shelves, most of them reclaimed from the sea,
at the foot of slopes that rise at forty or fifty per cent to ridges four
to nine hundred metres up. Streets either lie on a shelf, run along a
slope on a terrace cut into it, or go straight up it as stairs. And the
territory is not one place on foot: nothing crosses Victoria Harbour but
road tunnels, railways and ferries, and Lantau hangs off a bridge with no
footway.

This project models that from the government's 0.5 m LiDAR terrain model
and the OpenStreetMap street graph, routes across it under several
competing definitions of "flat", and works out which streets the geography
forces low-gradient traffic onto.

**The central result: about 20% more walking buys about 49% less
climbing.** Averaged over all 1,354 ordered pairs of neighborhoods that can
be walked between, the flattest route (the strongly climb-averse objective,
which accepts 120 m of extra walking to save a metre of climb) is 20%
longer than the shortest one and avoids 49% of the ascent. (San
Francisco's figures are 14% and 39%.)

The full write-up is in **[`outputs/findings.md`](outputs/findings.md)**, and
the checks against Hong Kong ground truth are in
**[`outputs/validation_report.md`](outputs/validation_report.md)**.

## The route finder

Type where you are and where you are going, in English or Chinese, then
drag the slider from **shortest** to **flattest** and watch the route
change. The slider steps through the routes that no other route beats on
both distance and climbing (up to thirty of them, spread evenly along the
frontier), solved in your browser over the full street graph: sliding right never shortens the route and never adds climbing. The
numbers, the elevation profile and the list of streets follow along.

It opens on Statue Square in Central to the University of Hong Kong: the
shortest walk (2.9 km, 147 m of climbing) goes up Lyndhurst Terrace,
Graham Street and Staunton Street, then Ladder Street to Caine Road, and
the flattest (3.4 km, 68 m) keeps to the old waterfront along Des Voeux
Road West as far as Western Street: half the climbing for a fifth more
distance, with thirty routes on the slider in all.

Everything upstream's page does, this one does: walk or bike, *prefer calm
streets* on a bike, the **loop button** (the flattest loop of a chosen
length from one start, with optional out-and-backs), **GPX** export, a
shareable link, and a *km / mi* switch (kilometres by default here). See
[upstream's README](https://github.com/almostimplemented/flattensf#the-route-finder)
for how the frontier of routes is found (BOA\*, a bi-objective A\*).

Three things are Hong Kong's own:

- **Landmasses.** Hong Kong Island, Kowloon with the New Territories,
  Lantau, and each outlying island are separate networks (21 of them for
  walking, 15 for cycling). A route is found only within one; asked for
  Central to Tsim Sha Tsui, the page says there is no way on foot between
  Hong Kong Island and Kowloon rather than drawing nothing. By bicycle a
  one-way system can let a rider into a district and not out, and the
  page says that too. An endpoint that lands on
  a stray fragment of the network (a pier, a podium) is moved onto the
  other end's landmass when a street of it is within 400 m; the pin then
  takes the name of the corner it lands on, and the page says how far it
  moved when that is more than 50 m. A real place, an island included, is
  never left for another.
- **Bilingual search.** Place search is offline and in two languages:
  street junctions ("Nathan Road & Jordan Road", "彌敦道與佐敦道"), areas,
  MTR stations and their exits, estates, buildings, parks, piers and tram
  stops, and street addresses ("123 Nathan Road", "彌敦道123號"), from the
  Lands Department's gazetteer and iGeoCom, its building names, the Address
  Lookup Service and OpenStreetMap. A query typed in Chinese is answered in
  Chinese.
- **Scale.** The page carries the whole territory: 260,082 street and
  path segments, 518,392 directed walking arcs and 155,478 cycling ones,
  with 55,900 places and 125,000 street numbers, in one 10.4 MB download.
  A trip across town solves in about half a second.

The site is static: [`site/`](site/) holds the page, its CSS and JS, the
street graph and the hillshade, and any static host serves it as is. The
same page is also written as one self-contained file,
`outputs/hk_flat_route_finder.html`, which opens straight from disk.

### The explorer and the warped city

`outputs/hk_flat_routes_map.html` is upstream's second page, ported whole:
the gradient-coloured network with the corridors, passes, barriers, lowland
basins and bicycle layers switchable; any-to-any routing under the four
objectives; an inspector for every street; and the *warped city*, which
redraws a landmass so that distance on the page is effort rather than
metres. It is one self-contained 17 MB file. In Hong Kong the warp is laid
out one landmass at a time, and the page says which.

## What it produces

| Output | What it is |
|---|---|
| [`site/`](site/) | The route finder as a static site |
| `outputs/hk_flat_route_finder.html` | The same page as one self-contained file |
| `outputs/hk_flat_routes_map.html` | The explorer and the warped city |
| [`outputs/findings.md`](outputs/findings.md) | The written analysis, generated from the tables below |
| [`outputs/validation_report.md`](outputs/validation_report.md) | Seven checks against Hong Kong ground truth |
| [`outputs/sensitivity.md`](outputs/sensitivity.md), `sensitivity.csv` | The pipeline rebuilt under 13 perturbed configurations |
| `outputs/neighborhood_pairs.csv` | Every routed pair under the four objectives |
| `outputs/pareto_frontier.csv` | The distance / climbing frontier of every pair |
| `outputs/flat_corridors.{csv,geojson,gpkg}` | The discovered low-gradient corridors |
| `outputs/passes.{csv,geojson}`, `pass_matrix.csv`, `basin_passes.csv` | The lowest crossing between every pair of neighborhoods |
| `outputs/barriers.{csv,geojson}` | Steep streets, with and without a flat alternative |
| `outputs/lowland_basins.{csv,geojson}` | Connected street below 15 mPD |
| `outputs/hk_flat_backbone.{png,pdf}`, `hk_street_grades.png` | Static maps |

## Headline findings

All from [`outputs/findings.md`](outputs/findings.md), which has the
tables. Distances in kilometres, heights in metres above Principal Datum
(mPD).

- **20% more walking, 49% less climbing**, over the 1,354 ordered pairs of
  neighborhoods with a walking route. The mean steepest pitch on a trip
  falls from 42% (a distance-minimising walker takes every stairway on the
  way) to 25%, and to 16% under the grade-averse objective, which costs 35%
  more distance. Total climbing and peak steepness are different
  objectives, here as in San Francisco.
- **Part of that saving is not going over the mountain.** The shortest
  route takes any mapped way, and 38% of its climbing is done on hill
  paths, tracks and steps that make up a tenth of its length. Sha Tin to
  Tai Po is 589 m of climbing by the shortest path, over the hills of Tai
  Po Kau, and 22 m by the flattest, for 1.3 km more.
- **Hong Kong is not one city on foot.** The walkable network falls into 21
  pieces; three hold the 56 neighborhoods (Kowloon and the New Territories
  32, Hong Kong Island 19, Lantau 5), and of the 3,080 ordered pairs that 56
  places would give, 1,726 have no walking route at all. Not one of the
  1,216 pairs across Victoria Harbour does.
- **The landmasses differ.** The flat route avoids 50% of the climbing on
  the mainland, 44% on Hong Kong Island and 40% on Lantau, for 17 to 21%
  of extra distance.
- **The flat corridors are the reclamations and the old coast roads.** 86
  walking corridors, 158 km, of which 51 lie wholly below 15 mPD: Lai Chi
  Kok Road with Kwai Chung Road, the Shing Mun River and Tolo Harbour from
  Sha Tin to Tai Po, Boundary Street; on the Island, King's Road and Java
  Road, Gloucester Road, Des Voeux Road West and the Kennedy Town praya.
  Castle Peak Road along the coast is as much used and nearly as low (4 to
  32 mPD). Six are level roads held high on a hillside (Shek O Road at 114
  to 151 mPD, Sir Cecil's Ride at 200, Bowen Road at 122), and five more
  are neither level nor low but the gentlest way over high ground (Tai Po
  Road over Piper's Hill, Repulse Bay Road, Stubbs Road).
- **For half of the pairs the "pass" is a doorstep.** Between the 677 pairs
  of neighborhoods that can be walked between, only 53 street segments
  are ever the binding high point, and for 50% of pairs that point is
  within 5 m of one end of the trip: the neighborhood itself is up the
  hill (The Peak at 292 mPD, Jardine's Lookout at 162, Mid-Levels West at
  119). The true ridges are few: Tseung Kwan O behind 135 m, Discovery Bay
  behind 159 m, Stanley behind 76 m, Sai Kung behind 70 m. One divide,
  Castle Peak Road at Ting Kau (32 mPD), stands between the New
  Territories' towns and Kowloon for 160 pairs, though staying under it
  means a route of 46 km at the median where the shortest is 20 km.
- **Half of all pairs can halve their climbing** (53%), at a median detour
  of 13%. The frontiers are close to straight: climbing comes off roughly
  in proportion to the extra distance, with no sharp knee.
- **A bicycle is not a walker.** Of 12,928 km open to a walker a bicycle
  may be ridden on 5,014 km: no steps, and footways only where signed,
  with another 45 km of short footway links it is pushed along. Eight
  pairs can be walked and not ridden, and the cycling network has four
  separate groups of neighborhoods against three on foot: Discovery Bay's
  road tunnel is closed to bicycles and the paths out of it are footpaths.

## What had to change for Hong Kong

Upstream says "nothing in the method is specific to San Francisco", and
that is true of the method. The data is another matter. Each of these is a
place where running upstream's code unchanged on Hong Kong's data gives a
wrong answer, usually silently; each is measured in the code's comments and
tested.

| | San Francisco (upstream) | Hong Kong (here) |
|---|---|---|
| Terrain model | USGS 3DEP lidar, 1 m, four tiles, held in memory | CEDD 2020 LiDAR, **0.5 m**, 3,253 tiles (14.1 GB), read a tile at a time with a margin of its neighbours |
| The sea | has values in the DEM | is *nodata*; smoothing is a normalised convolution so waterfront streets are not dragged toward a fill value |
| Hillside roads | rare | everywhere: each sample is taken at the flattest spot within 6 m of the mapped centreline, because a path cut into a 60% slope mapped 2 m off reads as a string of climbs |
| Bridge ends | DEM at the node | re-levelled from the approach: the bare-earth model is already falling into the gully under the last metres of deck |
| Bridge/tunnel flags | whole segment | per stretch (`between`): two in five of Hong Kong's bridge flags, and half of its tunnel flags, cover only part of a segment |
| Access rules | whole segment | per stretch, and vehicle height/weight limits no longer close streets to walkers |
| Road tunnels | as tagged | as tagged, and a tunnel on a main road is judged whole: one of more than 1 km is closed to walkers and cyclists where the data does not say (one tube of Tate's Cairn Tunnel carries no rule against walking), a shorter one where the data closes any part of it |
| Trunk roads | open unless denied | their viaducts are closed to walkers unless signed for them: the data bars walkers from most and says nothing of the rest, the decks of interchanges among them |
| Gated land | not an issue | service roads inside container terminals, cargo yards, power stations, depots and military land are closed: the gate is a point on the map, which no access rule records |
| Class `unknown` | routable | dropped, but for links of at most 50 m between two streets: in Hong Kong it is roads still under construction and TomTom segments with nothing to judge them by |
| Bicycles on footways | allowed unless denied | allowed only where signed (cycling on a footpath is an offence); a footway of 50 m or less that joins two rideable streets is pushed along, at three times its length, unless it is as steep as stairs |
| Sidewalks | dropped | kept, whatever their length, where a footbridge, subway or stair lands on one or the carriageway is closed to walkers, and the centreline network needs it |
| Street names | one string | English and Chinese, resolved per stretch (one segment is Nathan Road, then Austin Road) |
| Connectivity | largest component | every landmass: each connected piece of 40 nodes or more, with pieces joined by one-way streets counted as one |
| Clip | neighborhood union + 250 m | the SAR boundary, exactly (Shenzhen is 50 m across a river) |
| Units | feet and miles | metres and kilometres |

## Methodology

Only what differs from upstream is described here; for the rest
(the dead-band on cumulative gain, the cost model and its four objectives,
corridor detection, the minimax pass algorithm, the packing of the graph
for the browser) see
[upstream's README](https://github.com/almostimplemented/flattensf#methodology),
which the code still follows line for line.

### Elevation

1. **Source.** The Digital Terrain Model from the Civil Engineering and
   Development Department's territory-wide airborne LiDAR survey of
   December 2019 to February 2020: bare earth at **0.5 m**, in HK1980 Grid
   (EPSG:2326), heights in metres above the Hong Kong Principal Datum,
   published as 3,253 tiles of 750 m x 600 m on the government's CSDI
   portal. HK1980 Grid is also the CRS of every length and slope
   computation, so no raster is ever reprojected.
2. **Tiles, not a mosaic.** One mosaic of the territory at 0.5 m would be
   12 billion cells.
   Sample points are bucketed by tile and each tile is read with a margin
   of its eight neighbours wider than the smoothing filter's reach, so the
   result is identical to smoothing one seamless raster (tested to the
   millimetre) in a few hundred megabytes of memory.
3. **Noise suppression, spatially.** A Gaussian of sigma = 1.5 m, as a
   *normalised* convolution: cells with no data carry no weight. Upstream
   uses 3 m on a 1 m DEM; on Hong Kong's hillside roads, four to eight
   metres wide between a retaining wall and a drop, 3 m averages the wall
   into the road.
4. **The terrace search.** Where the ground falls away across a street
   (more than 6% between two metres left and two metres right), each
   sample is taken at the flattest spot within 6 m of the centreline, with
   a penalty for leaving it and another for shifting sideways from one
   sample to the next, so that the search follows one ledge along the
   street and does not hop between ledges.
   Edges that meet at a junction then share one height there, the median
   of what each found. Bowen Road, which eight bench marks show level to
   within 4.3 m, reads 10 m of climbing per km with this step and 19 m
   without it; the steep streets do not change.
5. **Structures.** As upstream: a linear ramp between the ends of each
   bridge or tunnel, with mid-structure nodes solved as a Laplace problem.
   Here the ramps apply to exactly the flagged stretch of a segment, and
   the elevation of each structure *end* is extrapolated from the ten to
   thirty-five metres of approach before it, replacing the sampled value
   when they differ by more than a metre. The Laplace problem is solved
   directly, where upstream relaxes it for a fixed 400 sweeps, which a
   long chain of spans does not settle in. And a structure that meets no
   measured ground at all (a pier: the sea is nodata, and so is the
   seawall under its landing) is solved together with the streets it
   joins, from the nearest terrain under them.
6. **Gullies and pits.** A running median over 35 m along each street
   removes the dip where a path crosses a gully on an untagged culvert
   (and leaves a steady slope exactly as it is). Ground below -1 mPD is
   treated as missing: in the winter of 2020 it was a construction pit.
7. Then upstream's chain: sampling at 5 m, Savitzky-Golay over 50 m per
   contiguous street, one elevation per intersection, and the 0.5 m
   dead-band on cumulative gain. One thing differs. Both filters along a
   street count in samples, and Hong Kong's streets are cut into many
   short pieces whose samples crowd together, so each street is read at
   stations an even 5 m apart, filtered there, and the samples take their
   values back from the result.

**Validation** (full report in
[`outputs/validation_report.md`](outputs/validation_report.md)):

- **Official bench marks.** 1,325 of the Survey and Mapping Office's
  levelled marks stand within 12 m of a road on the ground: the model reads
  a median 0.42 m below them (a mark sits in the kerb or a wall, a little
  above the carriageway) with a median absolute deviation of 0.26 m; 88%
  are within 1 m and 97% within 2 m, alike below 20 m and above 100 m.
- **An independent terrain model.** Against the Lands Department's 5 m
  model (2015 photogrammetry) at 3,997 random street points: median -0.9 m,
  90% within that model's own stated accuracy of 5 m.
- **Documented gradients.** The Transport Department's stated maximum lies
  between the model's section mean and its steepest 100 m, to within the 2
  points the report allows for rounding, on 15 of 15 road sections that
  could be measured, and strictly inside on 10 of them. The other five
  miss by at most half a point: Old Peak Road's stated 17% is just under
  the 17.3% the model finds over the whole section, and Shatin Pass Road's
  17% just over the 16.8% of its steepest 100 m. Signed ratios on the ladder
  streets: 11 of 12 within 4 points (Peel Street 1:4, 24.3%; Centre Street
  1:4, 21.9%). The mean gradient between two bench marks on one road, the
  strongest test: 13 of 14 within 1 point, median difference 0.0 (Wan Chai
  Gap Road 21.0% against 21.0%; Jat's Incline 339.6 m of rise by the marks,
  339.3 m in the model).
- **Level streets.** 13 of 13 come out as expected: the tram line along
  Des Voeux Road at 1.5 m of climbing per km and Hennessy Road at 1.9,
  Nathan Road 3.5, and the contour roads Conduit Road 8.0 and Bowen Road
  10.2, while Kennedy Road, Victoria Road, Waterloo Road and Lung Cheung
  Road, long and gentle but not level, are told apart from them.
- **Round, not over.** From the University to the Botanical Gardens the
  flat route stays on Bonham Road and Caine Road and wastes 45 m of
  climbing where the shortest wastes 92 m (Hong Kong's counterpart of
  upstream's Wiggle check).
- **The gaps.** 6 of 7 passes measured agree with a published or levelled
  height to within 3 m (Wong Nai Chung Gap 193.6 m against 191, Wan Chai
  Gap 241.7 against 241, Tsuen Kam Au 478.5 against a bench mark's 478.2).
- **Internal invariants.** Nine identities (gain - loss = net change, one
  elevation per node, no maximum gradient over the 60% ceiling it is
  clipped to, no stairway open to bicycles and so on) hold on all 520,164
  directed edges.
- **Not reproduced**, and not tuned away: Smithfield cannot be followed
  (160 m of it is a road tunnel, closed), Castle Road reads 14% against a
  published 1:5, one bench-mark pair on Hiram's Highway is on a rebuilt
  road, and Quarry Gap reads 6 m high.

**Sensitivity** (`python -m hk_flat_routes sensitivity`; tables in
[`outputs/sensitivity.md`](outputs/sensitivity.md)). The elevations and
the analysis were rebuilt under thirteen one-at-a-time changes: sample
spacing 2.5 and 10 m, smoothing window 25 and 100 m (half and double the
baseline's 50 m), dead-band 0.25 and 1 m, spatial pre-filter 0 and 3 m
(upstream's), each of the three hillside corrections switched off in turn
(the terrace search, the running median, the re-levelling of bridge
ends), and the second- and third-nearest access intersection for every
neighborhood. The headline does not move: +20% of distance for 48 to 49%
of the climbing in every one of them. The bench-mark agreement does not
either (median -0.42 to -0.40 m, 87 or 88% within 1 m), nor the steep
reference streets (Old Peak Road 21.6 to 22.1%, Peel Street 23.9 to
25.1%), and the dominant pass is the same one at Ting Kau in all
fourteen runs. What the corrections buy shows on the contour roads:
Bowen Road reads 10.2 m of climbing per km at baseline, 19.2 without the
terrace search, 20.6 with upstream's 3 m pre-filter, 14.5 without the
running median and 13.8 without the re-levelling. The corridors are the
least rigid part, as upstream found: the street that qualifies as
corridor material is 73 to 95% the same by length, and 8 to 12 of the
top twelve lead streets keep their place, but where each corridor is
cut, and so what it is called, shifts. They should be read as a tier,
not a ranking.

### Street network

Overture Maps' transportation theme, release 2026-09-23.1
(OpenStreetMap-derived, ODbL), read as upstream reads it. What Hong Kong
adds:

- **Ranges.** Overture joins consecutive OpenStreetMap ways into one
  segment and keeps flags, access rules and names as ranges along it.
  Segments are cut at range ends as well as at connectors, so every piece
  is wholly a bridge or not, open or closed, and named for itself.
  Flagging whole segments, as upstream does, would turn 1,600 km of Hong
  Kong's network into straight ramps where 490 km is actually bridge or
  tunnel.
- **Class `unknown` is dropped.** Upstream routes over it. In Hong Kong's
  extract it holds OpenStreetMap's roads still under construction (the
  planned streets of Kwu Tung North and Hung Shui Kiu) and the 2,148
  segments Overture takes from TomTom, which carry no class, name or
  access rule. One kind is kept, as a service road: a run of at most 50 m
  that joins two streets of the network (171 segments in the built network, 4.0 km in all, 156 of them
  TomTom's). Without those
  the walk from a bus terminus to the block beside it became a walk round
  the estate.
- **Tunnels.** Pedestrians and cyclists are barred from every vehicular
  tunnel in Hong Kong, and the data says so for all but a tube of Tate's
  Cairn Tunnel and a slip road of Kai Tak Tunnel. A tunnel on a motorway,
  trunk, primary or secondary road is judged whole, its pieces taken
  together. A run longer than 1 km is closed by rule wherever the data
  does not already say. A shorter one is closed to walkers, or to
  cyclists, only where the data itself bars them from some piece of it,
  the other tube included: the data closes one tube of Tseung Lam
  Highway, of the Ma On Shan Bypass and of Smithfield and says nothing of
  the other. The rest stay as the data has them, because a tunnel flag on
  a main road is more often a street passing under a building or a
  flyover than a tunnel: Cheung Tung Road at Sunny Bay, or Tuen Hi Road
  in Tuen Mun, at 159 m the longest left open.
- **Trunk roads.** A trunk road at grade with no rule against walking
  (Gloucester Road, Lung Cheung Road) is open to walkers, and so are its
  slip roads at grade. Its viaducts, runs of bridge longer than 100 m, are
  closed to them unless the data permits walking by name: the data bars
  walkers from most of them and says nothing of the rest, the decks of
  Kwai Chung Interchange among them, and an open viaduct is level, direct
  and nowhere anyone walks (10 km closed by the rule).
- **Gated land.** Service roads inside the container terminals and public
  cargo working areas, the power stations, the dockyards, the bus and
  railway depots and military land (the firing range at Castle Peak
  excepted, which is open hillside) are closed to both modes
  (100 km closed by the rule). OpenStreetMap maps
  them as plain service roads, because a gate is a point and no rule on
  the road records it; the land-use polygons of Overture's base theme do,
  which makes that extract an input of the network and not only of the
  place search.
- **Bicycles** may use a footway, a pedestrian street or a path only where
  the data permits bicycles explicitly; about 1,400 of 53,000 footway
  segments carry any bicycle rule, so upstream's "no rule means yes" would
  put bikes on every pavement. Cycle tracks are their own class. They are
  joined to the roads and to one another by a few metres of plain footway,
  so a walkable run of footway of at most 50 m that joins two points of
  the cycling network is opened as a *pushed link*: the cyclist dismounts,
  and the link costs three times its length. A run is opened only where
  the two points are otherwise unconnected or more than three times (and
  30 m) farther apart by riding; never steps, a bridge or a tunnel, never a
  footway the data closes to bicycles by name, and not one that rises at
  more than 20% from end to end, which is a flight of steps in fact.
- **Sidewalks and crossings** are mapped as separate ways in parts of Hong
  Kong, and footbridges, subways and stairs land on them. Travel is still
  modelled on centrelines, but a run of sidewalk or crossing is kept when
  it joins two points of the centreline network that are otherwise
  unconnected or more than 1.3 times (and 30 m) farther apart. A run is
  followed for up to 3 km, so that a long one is judged like any other:
  the footways of Tsing Tsuen Bridge are 1.6 km of sidewalk beside a
  carriageway closed to walkers (796 km kept of 1,448 km mapped, 18 km of
  it only as pushed links).
  Dropping them all, as upstream does, cuts 272 km of street and path
  (2.2% of the walkable centrelines) off from the rest of its landmass.
- **Landmasses.** A landmass is a set of strongly connected components of
  40 nodes or more that one-way streets join in either direction, together
  with the streets on the way between them, and is the same in Python and
  in the browser. (One component per landmass, as it was at first, made an
  island of Wan Chai North for cyclists: a district the one-way system
  lets a bicycle into and, as mapped, not out of.) Between two points of
  one landmass there can still be a way in one direction only, and the
  page then says there is no route.

### Neighborhoods

The Town Planning Board's statutory planning scheme areas (one polygon per
Outline Zoning Plan), which are named in both languages: every plan on
Hong Kong Island and in Kowloon, the New Territories' new towns and the
settled parts of Lantau, 56 in all. They name places and seed access
points; they do not tile the territory, so the network is clipped to the
SAR boundary instead. Pairs are routed only within a landmass:
992 ordered pairs on Kowloon and the New Territories, 342 on Hong Kong
Island and 20 on Lantau.

Corridors are scored and cut **per landmass**: a single territory-wide
threshold would be set by the mainland's 992 pairs, and Lantau's twenty
could never clear it.

## Data sources

`python -m hk_flat_routes sources` prints the full table with limitations.

| Dataset | Publisher | Role | Terms |
|---|---|---|---|
| Overture Maps transportation, divisions, base and places, release 2026-09-23.1 | Overture Maps Foundation (OpenStreetMap; Meta and Microsoft for places) | Street network, SAR boundary, the gated sites closed in the network, mapped features and shops for search | ODbL 1.0; CDLA Permissive 2.0 / CC0 for places |
| Digital Terrain Model from the 2020 territory-wide LiDAR survey, 0.5 m | Civil Engineering and Development Department, HKSAR Government, via the CSDI Portal | All elevation | CSDI Portal Terms and Conditions of Use |
| Statutory planning scheme areas | Town Planning Board / Planning Department | Neighborhoods | Town Planning Board terms of use |
| Place Names, iGeoCom, Building names | Lands Department, via the CSDI Portal and its open-data site | Place search | CSDI Portal / DATA.GOV.HK terms |
| Address Lookup Service bulk file | Digital Policy Office | Address search | DATA.GOV.HK terms |
| Tram and ferry stops | Transport Department | Place search | DATA.GOV.HK terms |
| Digital Terrain Model, 5 m | Lands Department (Survey and Mapping Office) | Validation only | DATA.GOV.HK terms |
| Bench marks (Geodetic Survey Control Station Database) | Lands Department (Survey and Mapping Office) | Validation only; not redistributed beyond the stations quoted in the validation report | CSDI Portal Terms and Conditions of Use (a dataset of the portal, fetched from the Lands Department's own copy) |

The Government's terms allow free use, including commercial use, on
condition that the Government and the portal are identified as the source
and its intellectual property rights acknowledged. They are not open
licences in the Creative Commons sense (they include an indemnity and can
change), so read them before redistributing anything built here:
<https://portal.csdi.gov.hk/csdi-webpage/doc/TNC> and
<https://data.gov.hk/en/terms-and-conditions>. The raw terrain tiles are
not redistributed in this repository.

Overture's extract also holds 2,148 road segments that it takes from
TomTom, not OpenStreetMap (ODbL 1.0 as well). All are of class `unknown`,
of which only the short links described under Street network are kept:
156 of them, 3.7 km in pieces of 50 m at most, are in the network.

**Keep the Overture extracts.** Everything else can be downloaded again;
the Overture release cannot, once it is retired. Overture hosts only its
latest few releases (three on 2026-10-13, the oldest 55 days old).
The published figures were built from release 2026-09-23.1, and
reproducing them later needs the four files `download` wrote from it for
the network: `data/raw/overture_segments_hk.parquet`,
`overture_connectors_hk.parquet`, `overture_division_area_hk.parquet` and
`overture_land_use_hk.parquet` (62.6 MB together, ODbL). Archive them with
the code, for instance as a release asset;
[`data/README.md`](data/README.md) has the details. When the pinned
release is gone and the files are not there, `download` stops and names
the releases still hosted, and `HKFR_OVERTURE_RELEASE` builds from one of
those instead.

## Installation

Python 3.10+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # or: pip install -e ".[dev]"
```

The geospatial stack (GeoPandas, rasterio, pyproj, shapely, scipy,
networkx, pyarrow) installs from wheels; no system GDAL is required.

## Reproducing the analysis

```bash
python -m hk_flat_routes all            # everything, in order
```

Or stage by stage; each caches its output, so re-running is cheap:

```bash
python -m hk_flat_routes sources        # dataset provenance table
python -m hk_flat_routes download       # fetch and cache source data (~14.3 GB)
python -m hk_flat_routes build-network  # street graph, elevation, edge metrics
python -m hk_flat_routes site           # the route finder only (site/)
python -m hk_flat_routes analyze        # pairs, Pareto, corridors, passes
python -m hk_flat_routes validate       # checks against Hong Kong ground truth
python -m hk_flat_routes map            # explorer, static maps, and site/
python -m hk_flat_routes report         # written analysis
python -m hk_flat_routes sensitivity    # optional: the robustness tables
```

Add `--force` to recompute a stage instead of using its cache: the
downloads, the network with its elevations and the neighborhoods, the
analysis, and for `site` and `map` the terrain overview behind the
hillshade. `validate` and `report` keep no cache. Forcing `download`, or
`all`, fetches every source again, terrain included; a cached file that is
not what it should be is replaced without it. A forced download also reads
the Overture extracts again, which it cannot do once their release is
retired: to rebuild from kept extracts, force `build-network` and the
stages after it and leave `download` alone. Ad-hoc routing between
neighborhoods:

```bash
python -m hk_flat_routes route --from "Central District" --to "The Peak Area"
```

The download is almost all terrain: 14.1 GB in 3,253 files, about 14.3 GB
in all, from a server that manages a few MB/s, so about an hour; it
resumes where it stopped. A file that is not what was asked for (the
portals sometimes send a maintenance page with HTTP 200) is retried and
never cached. The place sources, the 5 m terrain model and the bench marks
are optional: `download` carries on without one it cannot fetch and lists
at the end what it went without. (Overture's land use is not one of them,
though the search reads it too: the network is built with it.) MTR station
exits come from iGeoCom alone, so without that file the search has none.

After that, on a 10-core laptop: `build-network` 10 minutes (almost all of
it reading terrain: 9.3 million sample points, then the terrace search on
81,000 hillside edges), `analyze` a minute, `validate` half a minute,
`map` a minute, `report` seconds.

`sensitivity` rebuilds the elevation and the analysis fourteen times, one
configuration after another: allow four to five hours, or split the
configurations across terminals with `--only` and finish with one run
without it, which finds every configuration cached and writes the tables.

Where things are read and written is set from the environment:

| Variable | What it does |
|---|---|
| `HKFR_ROOT` | The directory holding `data/`, `outputs/` and `site/`. The checkout by default; the current directory when the package was installed with `pip install .` and has no checkout around it; `data/` and `outputs/` are made there by the first stage that runs. |
| `HKFR_RUN_DIR` | Redirects `data/processed`, `outputs` and `site` into this directory, leaving `data/raw` shared: a second build that does not touch the first. `sensitivity` uses it for each configuration. |
| `HKFR_OVERRIDES` | A JSON object of parameter overrides for that build, such as `{"dem_sigma_m": 3}`. The names are the fields of `ElevationConfig` and `AnalysisConfig` in [`config.py`](hk_flat_routes/config.py); an unknown name or a value of the wrong type is an error. |
| `HKFR_OVERTURE_RELEASE` | The Overture release `download` reads, in place of the pinned 2026-09-23.1. Different data, so different figures. An extract already in `data/raw` is kept whatever this says. Each extract written from now on records the release it was read from (one that names none is taken for the pinned release), and `download` warns when that is not the release in force: move the `overture_*_hk.parquet` files away, or force the download, to change release. |

(`HKFR_SITE_URL`, `HKFR_REPO_URL` and `HKFR_SITE_DOMAIN` are described
under the next heading; `HKFR_FORCE_PROGRESS=1` shows the progress bars
when the output is not a terminal.)

### Deploying the route finder

`site/` is committed already built, because building it needs the cached
terrain that CI does not have.
[`.github/workflows/pages.yml`](.github/workflows/pages.yml) publishes it
to GitHub Pages on every push that touches it (one-time setup: Settings ->
Pages -> Source: **GitHub Actions**). The page URL used for the canonical
link and the social-preview tags is `SITE_URL` in
[`config.py`](hk_flat_routes/config.py), or the `HKFR_SITE_URL`
environment variable at build time, and the link to the source that the
page shows is `REPO_URL`, or `HKFR_REPO_URL`; set `HKFR_SITE_DOMAIN` only
for a custom domain. `python scripts/screenshots.py` regenerates the
screenshots and the preview card.

### Tests

```bash
python -m pytest tests/ -q
```

The unit tests need no data. The integration tests assert the model's
invariants against the processed data and skip without it; the browser
tests drive the built pages in headless Chromium (`pip install -e ".[dev]"
&& playwright install chromium`) and check that the browser's route
metrics equal Python's on an identical path. `tests/qa_loop_mode.py` and
`tests/qa_map_cuts.py` are upstream's two checks to run by hand, ported:
the first puts loop mode through every control at desktop and phone
sizes, the second records the map and looks for hard cuts.

## Project structure

```
hk_flat_routes/
  config.py           all tunable parameters: CRS, weights, thresholds, modes
  sources.py          dataset registry: URLs, dates, licences, limitations
  download.py         cached acquisition; Parquet row-group bbox pruning;
                      the terrain tile index and parallel tile download
  network.py          street graph from Overture segments + connectors,
                      with Hong Kong's rules (ranges, tunnels, trunk
                      ramps, gated land, sidewalks, pushed links,
                      bilingual names)
  elevation.py        tiled DEM, sampling, terrace search, structures
  metrics.py          per-directed-edge metrics: grades, gain, steep distance
  routing.py          cost model, landmasses, scipy-backed shortest paths
  neighborhoods.py    planning areas and representative access points
  pairs.py            neighborhood-pair matrix and Pareto frontiers
  corridors.py        corridor importance scoring and merging
  passes.py           minimax passes, lowland basins, barriers
  validate.py         checks against bench marks and documented gradients
  viz_static.py       publication maps (matplotlib + lidar hillshade)
  webgraph.py         packs the graph into a compressed browser payload
  bikeways.py         the bike comfort table
  places.py           offline bilingual place index, and the hillshade
  viz_interactive.py  assembles the two self-contained web pages
  web/                their HTML, CSS and JavaScript: the shared router
                      (engine.js), the route finder (simple.js), the
                      explorer (app.js) and the warped city (warp.js)
  sensitivity.py      rebuilds the pipeline under perturbed parameters
  report.py           generates outputs/findings.md from the outputs
  pipeline.py         stage orchestration
  __main__.py         CLI
  vendor/             Leaflet 1.9.4 (BSD-2-Clause), inlined into the pages;
                      the certificate the iGeoCom host leaves out of its chain
site/                 the route finder as a static site (built)
scripts/              screenshots and the social preview card
tests/                unit, integration and browser tests; two QA scripts
data/raw/             cached source data (never modified)
data/processed/       cached intermediate products
outputs/              deliverables
```

## Limitations

- **Elevation is the ground, not the deck.** A footbridge, a flyover, an
  elevated walkway or the Mid-Levels escalator is modelled as a level ramp
  between where it meets the ground, so the five or six metres up to a
  footbridge and down again are not counted, and a podium-level walkway
  through a building takes the ground's elevation. The Lands Department's
  3D Pedestrian Network has surveyed deck heights and would fix this; it
  is not used yet.
- **Escalators and lifts are not modelled.** Nothing in the street data
  marks them, so the hillside escalators count as the stairs beside them.
- **No ferries, no MTR.** Hong Kong Island and Kowloon are separate
  networks. The cross-harbour ferries are in the source data and could be
  added as level links.
- **The terrain is as it was in early 2020.** Streets built since (parts
  of Kai Tak, Tung Chung East, the airport's third runway) sit on whatever
  was there, bridged from the ground around them where that was a pit.
- **Unflagged structures leak.** A road carried along a hillside on a
  viaduct that OpenStreetMap does not tag as a bridge is sampled from the
  slope beneath it.
- **Travel is on street centrelines**, with sidewalks only where the
  network needs them: which side of Gloucester Road has the pavement, and
  where its railings let you cross, is not modelled.
- **Closed areas the map does not mark are open here.** Access comes from
  OpenStreetMap's tags, and some places anyone in Hong Kong knows to be
  closed carry none. The Frontier Closed Area is the largest: the page
  offers walks into Sha Tau Kok past the Shek Chung Au checkpoint and to
  Lo Wu Station, both of which need a Closed Area Permit. Hei Ling Chau, a
  prison island, is routable as a landmass of its own. Service roads
  inside the container terminals, the power stations and military land
  are closed by rule (see Street network); a gated yard or works that no
  land-use polygon covers is not, and footways inside those sites are
  left as mapped.
- **The rules that close roads are narrow.** The trunk-road rule keeps
  walkers off viaducts only. A trunk road at grade that the data does not
  bar them from stays open, pavement or no pavement (Tuen Mun Road at Fu
  Tei is one), and the rule does not bind bicycles at all, which may
  still be routed over a viaduct the data leaves open. The tunnel rule
  leaves a covered stretch of main road shorter than 1 km open wherever
  the data closes no part of it: 1.3 km in 23 pieces of 101 m at most,
  most of them streets under a building or a flyover, and among them
  101 m of Ching Cheung Road and a 70 m stub of the Central Kowloon
  Bypass.
- **A bicycle can be let into a district and not out.** One-way streets
  are followed as mapped, and a landmass is everything they join in
  either direction, so the page can find a ride somewhere and none back.
  Wan Chai North is the known case, some 380 junctions around the
  Convention Centre: every road out is closed to bicycles in the data or
  one-way inwards, and no run of footway short enough to be pushed along
  leads out of it.
- **The shortest route is the shortest on the map, not the shortest
  anyone walks.** The baseline that every saving is measured against
  minimises distance over every mapped way, hill paths included, so
  between two towns it may go straight over the mountain between them:
  Sha Tin Station to Tai Po Market Station by the shortest path crosses
  Grassy Hill at over 600 mPD (9.8 km and 741 m of climbing, against
  13.2 km and 30 m by the flattest). Part of "less climbing for a little
  more distance" is the saving from not doing that: 38% of the baseline's
  climbing is on hill paths, tracks and steps.
- **Planning areas are not neighborhoods as residents draw them** (Sham
  Shui Po is inside "Cheung Sha Wan"), each gets one access point, and the
  new-town ones are large.
- **Maximum gradient on short edges is unreliable**, as upstream; edges
  under 15 m are judged on average gradient. Two in five of Hong Kong's
  edges are that short.
- **The route finder counts climbing, not steepness**, as upstream.
- **No traffic, signals, surface or safety**; the bicycle comfort weights
  are a table over road class. Cycling in urban Hong Kong is on the road,
  and the page does not judge whether a road is one anyone should ride.

## Licence and attribution

Code in this repository is available under the MIT licence; most of it is
upstream's, (c) 2026 sf-flat-routes contributors, and the port's changes
are (c) 2026 flattenhk contributors. `scripts/screenshots.py` is adapted
from [flatten_zrh](https://github.com/MoritzFS/flatten_zrh), the Zurich
port of the same project (MIT, (c) 2026 flatten_zrh contributors), whose
packaging this port also follows. All three notices are in
[`LICENSE`](LICENSE). The data it consumes is not MIT: street geometry is
**© OpenStreetMap contributors, ODbL 1.0** (via Overture Maps, with a
few kilometres of short links from TomTom under the same licence), and any
redistribution of derived street geometry, including the files under
`outputs/` and `site/data/`, carries ODbL share-alike obligations.
Elevation, planning areas, place names and addresses are © the Government
of the Hong Kong Special Administrative Region, obtained from the CSDI
Portal and DATA.GOV.HK and used under their terms. The bench marks are
used under the CSDI Portal Terms and Conditions of Use (the Geodetic
Survey Control Station Database is a dataset of the CSDI Portal, fetched
here from the Lands Department's own copy), for validation only, and are
not redistributed beyond the stations quoted in the validation report.
Leaflet is bundled under BSD-2-Clause; see
[`hk_flat_routes/vendor/README.md`](hk_flat_routes/vendor/README.md),
which also accounts for the one CA certificate shipped beside it.
