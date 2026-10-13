# Sensitivity analysis

Each of the 14 configurations below rebuilds the pipeline -- elevation sampling, metrics, 10,800 routes, corridors and passes -- on the same street network (260,082 edges), with one parameter changed from the baseline. The question is whether the findings survive the modelling choices. (The harness is upstream's, from flattensf; the parameters and the reference streets are Hong Kong's.)

Baseline: sample_spacing_m = 5, smooth_window_m = 25, gain_deadband_m = 0.5, dem_sigma_m = 1.5, lateral_search_m = 6, despike_window_m = 35, abutment_blank_m = 10, point_rank = 0

## The headline (walking, all 1,354 ordered pairs with a way between them)

| Configuration | Change | Flattest: extra distance | Flattest: climbing avoided | Shortest: mean climb | Flattest: mean climb | Grade-averse: mean steepest |
|---|---|---|---|---|---|---|
| baseline | the configuration used for every published figure | +20% | 49% | 347 m | 124 m | 16.2% |
| spacing_2.5m | denser DEM sampling | +20% | 49% | 348 m | 123 m | 16.0% |
| spacing_10m | coarser DEM sampling | +20% | 49% | 341 m | 122 m | 14.5% |
| window_12.5m | half the profile smoothing | +20% | 49% | 347 m | 123 m | 19.1% |
| window_50m | double the profile smoothing | +20% | 48% | 341 m | 123 m | 16.0% |
| deadband_0.25m | half the dead-band | +20% | 48% | 348 m | 126 m | 16.2% |
| deadband_1m | double the dead-band | +20% | 49% | 346 m | 122 m | 16.2% |
| sigma_0m | no spatial pre-filter on the DEM | +20% | 49% | 348 m | 125 m | 16.5% |
| sigma_3m | double the spatial pre-filter (upstream's setting) | +20% | 48% | 346 m | 125 m | 15.7% |
| no_terrace_search | sample the mapped centreline; no search for the road surface | +20% | 48% | 348 m | 127 m | 15.6% |
| no_despike | no running median along the street | +20% | 49% | 355 m | 126 m | 16.9% |
| no_abutment_fix | trust the terrain model right up to bridge and tunnel ends | +20% | 49% | 348 m | 125 m | 17.5% |
| point_rank_1 | second-nearest access intersection | +20% | 48% | 345 m | 124 m | 16.3% |
| point_rank_2 | third-nearest access intersection | +20% | 49% | 344 m | 123 m | 16.2% |

## Elevation model checks

| Configuration | Old Peak Road | Centre Street | Peel Street | Hatton Road | Hennessy Road climb/km | Nathan Road climb/km | Bowen Road climb/km | Network climb/km | Model minus bench mark (median; within 1 m) |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 21.9% | 21.9% | 24.3% | 13.6% | 1.9 m | 3.5 m | 10.2 m | 36.7 m | -0.42 m; 88% |
| spacing_2.5m | 21.9% | 21.9% | 24.2% | 13.5% | 1.8 m | 3.5 m | 10.6 m | 36.8 m | -0.42 m; 88% |
| spacing_10m | 21.6% | 21.7% | 24.3% | 13.5% | 1.9 m | 3.4 m | 11.7 m | 35.8 m | -0.42 m; 88% |
| window_12.5m | 21.6% | 21.7% | 24.4% | 13.5% | 1.9 m | 3.4 m | 10.7 m | 36.6 m | -0.42 m; 88% |
| window_50m | 22.1% | 22.3% | 24.3% | 13.5% | 1.9 m | 3.4 m | 7.7 m | 35.9 m | -0.42 m; 87% |
| deadband_0.25m | 21.9% | 21.9% | 24.3% | 13.6% | 1.9 m | 3.5 m | 10.8 m | 36.8 m | -0.42 m; 88% |
| deadband_1m | 21.9% | 21.9% | 24.3% | 13.6% | 1.9 m | 3.5 m | 10.0 m | 36.5 m | -0.42 m; 88% |
| sigma_0m | 21.8% | 22.1% | 25.1% | 13.5% | 1.9 m | 3.5 m | 11.5 m | 36.9 m | -0.41 m; 88% |
| sigma_3m | 21.7% | 21.7% | 23.9% | 13.6% | 1.8 m | 3.4 m | 20.6 m | 36.5 m | -0.40 m; 88% |
| no_terrace_search | 21.8% | 21.8% | 24.3% | 13.6% | 1.9 m | 3.5 m | 19.2 m | 36.6 m | -0.42 m; 87% |
| no_despike | 22.0% | 21.9% | 24.3% | 13.6% | 1.9 m | 3.5 m | 14.5 m | 37.7 m | -0.42 m; 88% |
| no_abutment_fix | 21.9% | 21.9% | 24.3% | 13.6% | 1.9 m | 3.5 m | 13.8 m | 36.7 m | -0.42 m; 88% |
| point_rank_1 | 21.9% | 21.9% | 24.3% | 13.6% | 1.9 m | 3.5 m | 10.2 m | 36.7 m | -0.42 m; 88% |
| point_rank_2 | 21.9% | 21.9% | 24.3% | 13.6% | 1.9 m | 3.5 m | 10.2 m | 36.7 m | -0.42 m; 88% |

Each street's column holds: Old Peak Road, the steepest 100 m of the Transport Department's section (stated maximum 17.0%); Centre Street, the steepest 50 m of the signed section (signed 25.0%); Peel Street, the steepest 50 m of the signed section (signed 25.0%); Hatton Road, the mean between two bench marks (13.5% by the marks). The Transport Department's maximum is a sustained gradient, which the validation report expects to find between the section's mean and its steepest 100 m, so the figure here should sit at or above it, not on it. The streets, their sections and their sources are those of the validation report; the bench marks are the 1,325 official ones beside a road in this run.

## Corridors, passes and the contour trip

Corridor overlap is measured on the street itself: the length-weighted share of corridor-material edges the run has in common with the baseline. Comparing corridor names would be misleading, since a merge boundary moving by one block renames a corridor without changing where it runs. The contour trip is the validation report's (University of Hong Kong to the Botanical Gardens): the climbing each route wastes beyond the difference between its ends.

| Configuration | Corridors found | Corridor edges shared with baseline | Lead streets of the top 12 kept | Streets that enter the top 12 | Top corridor | Top pass | Wong Nai Chung Gap (mapped at 191 m) | Contour trip, excess climb (flat / shortest) |
|---|---|---|---|---|---|---|---|---|
| baseline | 86 | 100% | 12/12 | &mdash; | Lai Chi Kok Road (10.8 km) | Tsuen Wan West 32 m, 160 pairs | 193.6 m | 45.4 m / 91.8 m |
| spacing_2.5m | 78 | 83% | 12/12 | &mdash; | Lai Chi Kok Road (10.8 km) | Tsuen Wan West 32 m, 160 pairs | 193.6 m | 45.5 m / 92.8 m |
| spacing_10m | 93 | 77% | 11/12 | Yuen Shin Road | Lai Chi Kok Road (11.9 km) | Tsuen Wan West 32 m, 160 pairs | 193.4 m | 50.2 m / 94.1 m |
| window_12.5m | 91 | 79% | 11/12 | Taikoo Wan Road | Lai Chi Kok Road (9.9 km) | Tsuen Wan West 32 m, 160 pairs | 193.6 m | 44.6 m / 90.0 m |
| window_50m | 91 | 78% | 9/12 | Hoi On Road; Science Park Road; Tai Po Road – Tai Wo | Lai Chi Kok Road (12.8 km) | Tsuen Wan West 32 m, 160 pairs | 193.4 m | 44.8 m / 87.7 m |
| deadband_0.25m | 86 | 95% | 9/12 | Science Park Road; Tai Po Road – Tai Wo; Yuen Shin Road | Lai Chi Kok Road (9.7 km) | Tsuen Wan West 32 m, 160 pairs | 193.6 m | 45.7 m / 92.1 m |
| deadband_1m | 94 | 89% | 12/12 | &mdash; | Shui Chong Street (14.9 km) | Tsuen Wan West 32 m, 160 pairs | 193.6 m | 45.4 m / 91.8 m |
| sigma_0m | 91 | 80% | 11/12 | Taikoo Wan Road | Yuen Wo Road (18.4 km) | Tsuen Wan West 32 m, 160 pairs | 193.5 m | 44.1 m / 91.5 m |
| sigma_3m | 80 | 73% | 10/12 | Science Park Road; Yuen Shin Road | Lai Chi Kok Road (10.4 km) | Tsuen Wan West 32 m, 160 pairs | 194.0 m | 45.1 m / 88.7 m |
| no_terrace_search | 85 | 79% | 8/12 | Container Port Road South; Science Park Road; Tai Po Road – Tai Wo; Yuen Shin Road | Yuen Wo Road (4.6 km) | Tsuen Wan West 29 m, 152 pairs | 193.6 m | 45.2 m / 88.4 m |
| no_despike | 87 | 83% | 9/12 | Container Port Road South; Java Road; Taikoo Wan Road | Shui Chong Street (14.6 km) | Tsuen Wan West 32 m, 160 pairs | 193.7 m | 45.2 m / 93.3 m |
| no_abutment_fix | 86 | 91% | 11/12 | Science Park Road | Lai Chi Kok Road (10.8 km) | Tsuen Wan West 32 m, 160 pairs | 193.6 m | 51.9 m / 98.5 m |
| point_rank_1 | 93 | 90% | 11/12 | Des Voeux Road West | Lai Chi Kok Road (12.3 km) | Tsuen Wan West 32 m, 152 pairs | 193.6 m | 45.4 m / 91.8 m |
| point_rank_2 | 92 | 89% | 11/12 | Des Voeux Road West | Lai Chi Kok Road (12.3 km) | Tsuen Wan West 32 m, 152 pairs | 193.6 m | 45.4 m / 91.8 m |

## Reading it

- The headline trade (extra distance for climbing avoided on the flattest route) ranges from +20% / 48% to +20% / 49% across every perturbation, against +20% / 49% at baseline.
- The corridor material stays 73-95% the same street, by length, under every perturbation, and 8-12 of the baseline's 12 lead streets keep their place. What moves is the exact extent and composite name of each corridor, most under **sigma_3m** (73%), and a few borderline streets drift in and out at the margin: Container Port Road South, Des Voeux Road West, Hoi On Road, Java Road, Science Park Road, Tai Po Road – Tai Wo, Taikoo Wan Road, Yuen Shin Road.
- The dominant pass is in Tsuen Wan West in 14 of 14 configurations.
- Against the official bench marks the model's median difference is -0.42--0.40 m in every configuration, with 87-88% of marks within 1 m; the smallest share is under **window_50m**.
- On the contour trip the flat route goes round in 9 of 14 configurations, wasting 44.1-51.9 m of climbing against the shortest route's 87.7-98.5 m.
- Wong Nai Chung Gap comes out at 193.4-194.0 m across the configurations, against a mapped 191 m.
