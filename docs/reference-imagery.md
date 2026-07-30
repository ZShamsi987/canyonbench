# Imagery and map-source policy

## Storage-safe acquisition

Do not download statewide mosaics or an entire balloon corridor. Acquire one
bounded chip per frozen candidate site, keep large rasters in cloud/local
ignored storage, and commit only manifests, hashes, terms, and small reports.

The generator reads local GeoTIFFs because reproducible camera projection
requires immutable pixels. It does not depend on a live tile service during a
reported run. A QGIS or browser map service may be used to discover candidates;
the final bounded export must be hashed and dated.

The implemented default is:

```bash
canyonbench trace discover-sites \
  configs/trace_sources.yaml \
  /Users/zafirshamsi/CanyonBench-data/manifests/trace_candidates.yaml \
  --cache-dir /Users/zafirshamsi/CanyonBench-data/cache/site-discovery
canyonbench trace acquire-sources \
  configs/trace_sources.yaml \
  /Users/zafirshamsi/CanyonBench-data/manifests/trace_candidates.yaml \
  /Users/zafirshamsi/CanyonBench-data/sources \
  /Users/zafirshamsi/CanyonBench-data/manifests/trace_prepared_candidates.yaml \
  /Users/zafirshamsi/CanyonBench-data/reports/source_acquisition.json \
  --flight-source /Users/zafirshamsi/Downloads/World-10/WORLD10.txt
```

Acquisition is restart-safe and writes a `COMPLETE` marker only after the full
site and provenance record succeed. Use `--start` and `--limit` to batch.

## Orthoimagery

USDA NAIP is preferred. USGS high-resolution orthoimagery is the fallback. A
source record must include provider, product, version/year, acquisition date,
native resolution, URL, SHA-256, license/terms, and source tile IDs.

Do not assume every public web service grants identical redistribution rights.
Record the terms for the actual downloaded product and confirm redistribution
before public release. If redistribution is restricted, release the generator,
coordinates allowed by the terms, and hashes rather than the pixels.

## Feature layers

Primary and secondary sources must be genuinely independent enough to support a
consensus claim. Record provider/version/date for both. For OpenStreetMap,
preserve the relevant ODbL attribution and extraction date. For federal public
domain products, preserve the agency/product acknowledgment even when not
legally required.

## Coordinate systems

Each site may use a different projected metric EPSG CRS. All of its layers must
share that exact CRS, affine transform, bounds, width, and height before the
camera runs. WGS84 longitude/latitude in `sites.yaml` locates the camera center;
the renderer transforms it into the raster CRS.

## Source manifest

`source_manifest.json` is part of every site bundle. Its hash is repeated in
each view manifest. A reported release is invalid if an artifact hash or source
date changes after generation.

## Disk layout

Recommended ignored layout:

```text
CanyonBench-data/
  sources/site_0001/
    imagery.tif
    water_primary.tif
    water_secondary.tif
    road_primary.tif
    road_secondary.tif
    field_primary.tif
    field_secondary.tif
    dem.tif
    source_manifest.json
  generated/
  runs/
  releases/
```

Use cloud placeholders for inactive sites if necessary, but ensure all inputs
for one site are fully local during its build. After the validated generated
bundle is backed up, source chips can return to cloud-only state.
