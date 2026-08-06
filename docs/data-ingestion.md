# Procedural source ingestion

This is the primary v4 data guide. The real balloon video is used for quality
calibration and optional deployment analysis, not primary labels.

## Required sources

For every base site, freeze:

| Layer | Preferred source | Independent check |
|---|---|---|
| RGB orthoimage | USDA NAIP | USGS high-resolution ortho fallback |
| Water | USGS NHD/3DHP | a second hydrography layer |
| Major road | Census TIGER/Line | OpenStreetMap |
| Cultivated field | USDA Cropland Data Layer | independent parcel/crop layer |
| Terrain | USGS 3DEP | optional, but required for depth matching when available |
| Exclusion detector | independent class detector | target-class score raster required |

Do not treat one map source as self-validating. Every feature requires a primary
and secondary mask, even when the candidate's target class is different.

## Site composition

The critical-path dataset has exactly 120 independent sites:

| Group | Water | Road | Field | Total |
|---|---:|---:|---:|---:|
| Flight corridor | 20 | 20 | 20 | 60 |
| Regional OOD | 12 | 12 | 12 | 36 |
| Cross-biome | 8 | 8 | 8 | 24 |
| Total | 40 | 40 | 40 | 120 |

Each group/class cell is evenly split between present (including registered
extinction) and negative sites. A site row must identify all source tile IDs and
all water-body, road-segment, and parcel IDs so split leakage can be rejected.

## Source preparation

1. Export a bounded orthoimage chip in a projected metric CRS appropriate to the
   site.
2. Rasterize the primary and secondary vector layers onto the exact imagery
   grid, or align source rasters to that grid.
3. Align the DEM and detector-score rasters identically.
4. Confirm masks are binary.
5. Compute SHA-256 for every source artifact.
6. Create a source manifest from
   `CanyonBench-data/manifests/source_manifest.example.json`.
7. Add the site to `CanyonBench-data/manifests/sites.yaml`.

Every non-negative site must have a target-class detector score and matching
source record. Full generation fails without it because G4 and the empirical
extinction band cannot otherwise be evaluated.

Commands:

```bash
canyonbench trace rasterize-geojson water.geojson imagery.tif water_primary.tif
canyonbench trace align-raster secondary-water.tif imagery.tif water_secondary.tif
canyonbench trace align-raster dem.tif imagery.tif dem_aligned.tif --continuous
```

GeoJSON must already use the imagery CRS. Reproject externally when needed; do
not guess a missing CRS.

## Automatic gates

Generation evaluates every target before rendering.

- G1 rejects date gaps beyond the frozen threshold; fields use the tighter
  threshold.
- G2 requires two-source positive overlap within tolerance, or two-source
  absence through a larger negative buffer.
- G3 measures component size, apparent width, boundary distance, interior reach,
  local contrast, aliasing, and occlusion. Resolvability gates on interior reach,
  so a feature that crosses the footprint and therefore touches the frame border
  is still resolvable; closest approach remains a reported diagnostic.
  Below-resolution positives may remain only in the extinction category.
- G4 is exclusion-only. A detector score can reject a site but never add a
  feature.

Save every pass/fail value. Do not hand-correct a failed gate. Fix the source or
replace the candidate and rerun.

## Split freeze

Splits are deterministic and stratified: 20% development, 20% validation, 60%
test. No split may share a source tile, feature ID, rounded coordinate, or
overlapping footprint. Every derivative of a site inherits the same split.

## Real-flight calibration

WORLD10 contains multiple reset sessions; use the recovered operational run
only. Camera date/time is known to be wrong and must not be used as calendar
truth. Verified file/clip order is valid. The logged speed and vertical velocity
channels are constant zero, so derive drift from successive GPS positions.
Heading is unvalidated. Spectral and thermal channels are excluded.

Extract only enough representative frames to calibrate image-quality
distributions. Keep the large footage in Google Drive/cloud-only storage.
`--evict-source-cache` remains available in the legacy clip tools to return
successfully read macOS File Provider ranges to cloud-only state.

Run:

```bash
canyonbench trace calibrate-quality /path/to/frames calibration/flight-quality.json
```

### Rebuilding the World-X calibration without local video storage

The camera metadata clock is invalid. The audited World-X clip order and the
verified sync anchor are authoritative instead. To replace the legacy
left-third-cropped calibration, run the storage-bounded full-frame rebuild from
the repository root:

```bash
.venv/bin/python scripts/recalibrate_flight_quality.py
```

It selects exactly 377 usable Launching/Floating clips in chronological audited
order, extracts only one uncropped midpoint frame at a time, measures it, and
evicts the Drive-backed source immediately. It retains neither the footage nor
the temporary frames; `flight-quality-provenance.json` records the clip order
and output-frame hashes.

This measures the registered proxies and hashes every calibration frame. The
calibration record—not the full video—is needed by the generator.
