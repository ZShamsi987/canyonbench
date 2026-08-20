# Field-negative feasibility record

Frozen: 2026-08-20. This record supports the `utah_west_desert` amendment in
[configs/trace_sources.yaml](../configs/trace_sources.yaml) and must be cited in
the paper alongside the Group A water-negative clarification.

## The rule being measured

A field negative is admissible only when no cultivated pixel falls inside the
buffered maximum camera footprint. `evaluate_site` zeroes every mask outside
`_maximum_camera_footprint` before `negative_clear` is computed, so the decision
support is `negative_screen_half_extent_m` plus the 30 m safety buffer — a
10,130 m half extent, about 408 km².

Every figure below applies the frozen `CULTIVATED_CDL_CODES` to the USDA CDL
class raster over exactly that support, at a 4–5 km sampling lattice. Candidate
centres were first required to be clear of Annual NLCD class 82 where a cached
regional raster was available.

## Result 1 — the original cross-biome regions cannot supply the quota

| Region | CDL-clear centres |
|---|---|
| Nevada Great Basin, Wyoming High Plains, Nebraska Plains (combined) | **1 of 1,765** |

The quota is eight. Separation-feasible at the registered 22 km: one.

## Result 2 — the rule is not satisfiable outside barren terrain

| Terrain probed | CDL-clear centres |
|---|---|
| Barren playa (Death Valley control) | 52 / 100 |
| Salt flat (Great Salt Lake Desert control) | 23 / 100 |
| Chihuahuan desert mountain (Big Bend) | 9 / 1,009 |
| Cold montane conifer (Sierra Nevada) | 3 / 960 |
| Pacific montane conifer (Cascades) | 1 / 880 |
| Cold montane wilderness (Northern Rockies) | 0 / 1,440 |
| Humid temperate broadleaf forest (Appalachian) | 0 / 1,200 |
| Northern hardwood / boreal transition (Adirondack) | 0 / 676 |

CDL scatters hay and pasture codes through rangeland and even designated
wilderness, and one 30 m pixel disqualifies 408 km². Forested and montane
regions therefore cannot supply field negatives at any pool size.

**Consequence to disclose:** field negatives are confined to barren terrain in
every group, not only cross-biome. Barrenness is confounded with the
field-negative label, so a model can answer the field question from ground
texture without using field evidence — which is precisely what the intervention
measures are designed to detect. Report this as a limitation of the field class.

## Result 3 — the adopted stratum

`utah_west_desert`, spanning the Bonneville, Great Salt Lake, and Sevier
deserts:

| Measure | Value |
|---|---|
| CDL-clear centres | 217 of 2,469 |
| Mutually separated by 22 km | **14** (quota is 8) |
| NAIP coverage | 2021, pairing with CDL 2021 for exact G1 date alignment |
| Annual NLCD confirmation | 6 of the first 10 centres confirmed clear; the other 4 returned USGS WCS service errors, not rejections |

It overlaps neither `mojave_dryland` (already a regional_ood stratum) nor
`nevada_great_basin`, and it is a genuine biome shift from the Colorado Plateau
corridor: cold, internally drained salt playa in the Central Basin and Range.

Fourteen separation-feasible centres found by the probe lattice, for
cross-checking whatever the registered discovery process selects:

```
-113.17 39.46   -113.45 40.06   -113.65 40.22   -113.37 40.26
-113.73 40.42   -113.45 40.46   -113.89 40.58   -113.61 40.62
-113.29 40.66   -113.69 40.82   -113.41 40.86   -113.57 41.02
-112.45 41.02   -112.93 41.34
```

## Why this was not found earlier

The authoritative screen costs one CropScape request or one national-archive
window per candidate. CropScape has been returning 503, and reading whole
archives is what the login node killed for memory, so the pipeline could vet a
few dozen candidates a day against a need for several hundred. Three expanded
pools each returned "0 valid negatives," which was read as bad luck rather than
as the measurement it was.

[scripts/adroit/prefilter_field_negatives.py](../scripts/adroit/prefilter_field_negatives.py)
removes that constraint: it applies the same frozen codes to the CDL class
raster served as cloud-optimised GeoTIFFs, one windowed range read per
candidate. Forty candidates take 10.7 s against 22 s each, which is what made
these 8,000-candidate sweeps affordable.
