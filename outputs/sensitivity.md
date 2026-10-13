# Sensitivity analysis

Each of the 4 configurations below rebuilds the pipeline -- elevation sampling, metrics, 10,800 routes, corridors and passes -- on the same street network (260,615 edges), with one parameter changed from the baseline. The question is whether the findings survive the modelling choices. (The harness is upstream's, from flattensf; the parameters and the reference streets are Hong Kong's.)

Baseline: sample_spacing_m = 5, smooth_window_m = 25, gain_deadband_m = 0.5, dem_sigma_m = 1.5, lateral_search_m = 6, despike_window_m = 35, abutment_blank_m = 10, point_rank = 0

## The headline (walking, all 1,354 ordered pairs with a way between them)

| Configuration | Change | Flattest: extra distance | Flattest: climbing avoided | Shortest: mean climb | Flattest: mean climb | Grade-averse: mean steepest |
|---|---|---|---|---|---|---|
| spacing_10m | coarser DEM sampling | +19% | 48% | 343 m | 127 m | 18.8% |
| deadband_0.25m | half the dead-band | +19% | 48% | 349 m | 129 m | 19.6% |
| deadband_1m | double the dead-band | +19% | 49% | 346 m | 126 m | 19.6% |
| no_terrace_search | sample the mapped centreline; no search for the road surface | +19% | 48% | 347 m | 130 m | 20.4% |

## Elevation model checks

| Configuration | Old Peak Road | Centre Street | Peel Street | Hatton Road | Hennessy Road climb/km | Nathan Road climb/km | Bowen Road climb/km | Network climb/km | Model minus bench mark (median; within 1 m) |
|---|---|---|---|---|---|---|---|---|---|
| spacing_10m | 21.8% | 22.3% | 24.4% | 13.5% | 1.9 m | 3.4 m | 15.1 m | 35.7 m | -0.42 m; 88% |
| deadband_0.25m | 21.8% | 21.5% | 24.4% | 13.6% | 1.9 m | 3.5 m | 12.6 m | 36.6 m | -0.42 m; 87% |
| deadband_1m | 21.8% | 21.5% | 24.4% | 13.6% | 1.9 m | 3.5 m | 11.3 m | 36.3 m | -0.42 m; 87% |
| no_terrace_search | 21.8% | 21.5% | 24.2% | 13.6% | 1.9 m | 3.5 m | 20.0 m | 36.2 m | -0.42 m; 87% |

Each street's column holds: Old Peak Road, the steepest 100 m of the Transport Department's section (stated maximum 17.0%); Centre Street, the steepest 50 m of the signed section (signed 25.0%); Peel Street, the steepest 50 m of the signed section (signed 25.0%); Hatton Road, the mean between two bench marks (13.5% by the marks). The Transport Department's maximum is a sustained gradient, which the validation report expects to find between the section's mean and its steepest 100 m, so the figure here should sit at or above it, not on it. The streets, their sections and their sources are those of the validation report; the bench marks are the 1,328 official ones beside a road in this run.

## Corridors, passes and the contour trip

Corridor overlap is measured on the street itself: the length-weighted share of corridor-material edges the run has in common with the baseline. Comparing corridor names would be misleading, since a merge boundary moving by one block renames a corridor without changing where it runs. The contour trip is the validation report's (University of Hong Kong to the Botanical Gardens): the climbing each route wastes beyond the difference between its ends.

| Configuration | Corridors found | Corridor edges shared with baseline | Lead streets of the top 12 kept | Streets that enter the top 12 | Top corridor | Top pass | Wong Nai Chung Gap (mapped at 191 m) | Contour trip, excess climb (flat / shortest) |
|---|---|---|---|---|---|---|---|---|
| spacing_10m | 88 | n/a | 0/0 | Boundary Street; Castle Peak Road – Castle Peak Bay; Castle Peak Road – Tai Lam; Castle Peak Road – Ting Kau; Castle Peak Road – Tsing Lung Tau; King's Road; Lai Chi Kok Road; Lam Tei Interchange; Science Park Road; Tai Po Road – Sha Tin Heights; Yuen Wo Road; unnamed corridor | Lai Chi Kok Road (12.6 km) | Tsuen Wan West 31 m, 126 pairs | 193.5 m | 50.6 m / 97.7 m |
| deadband_0.25m | 84 | n/a | 0/0 | Boundary Street; Castle Peak Road – Castle Peak Bay; Castle Peak Road – Tai Lam; Castle Peak Road – Ting Kau; Castle Peak Road – Tsing Lung Tau; Container Port Road South; Science Park Road; Tai Po Road – Sha Tin Heights; Tai Po Road – Tai Wo; Taikoo Wan Road; Yuen Wo Road; unnamed corridor | Castle Peak Road – Castle Peak Bay (4.9 km) | Tsuen Wan West 32 m, 133 pairs | 193.6 m | 44.9 m / 92.7 m |
| deadband_1m | 89 | n/a | 0/0 | Boundary Street; Castle Peak Road – Castle Peak Bay; Castle Peak Road – Tai Lam; Castle Peak Road – Ting Kau; Castle Peak Road – Tsing Lung Tau; King's Road; Kwai Chung Road; Lai Chi Kok Road; Science Park Road; Tai Po Road – Sha Tin Heights; Yuen Shin Road; Yuen Wo Road | Castle Peak Road – Castle Peak Bay (5.1 km) | Tsuen Wan West 32 m, 133 pairs | 193.6 m | 44.9 m / 92.1 m |
| no_terrace_search | 82 | n/a | 0/0 | Boundary Street; Castle Peak Road – Castle Peak Bay; Castle Peak Road – Tai Lam; Castle Peak Road – Ting Kau; Container Port Road South; King's Road; Prince Edward Road East; Science Park Road; Tai Po Road – Sha Tin Heights; Taikoo Wan Road; Yuen Wo Road; unnamed corridor | Container Port Road South (7.5 km) | Tsuen Wan West 29 m, 126 pairs | 193.6 m | 45.7 m / 88.8 m |