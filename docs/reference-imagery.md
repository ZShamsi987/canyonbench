# Reference imagery without a large local download

CanyonBench uses the official USGS NAIP ImageServer as its registration
reference. A catalog query on July 26, 2026 found complete coverage of the 377
sampled-frame corridor in Arizona from the 2023 acquisition: 189 primary tiles,
0.3 metre nominal resolution, four bands, UTM projection. The service and data
are public domain. Acknowledgment is:

> Map services and data available from U.S. Geological Survey, National
> Geospatial Program.

The frozen source record is in the data repository at
`registration/reference/source.yaml`. Registration coordinates use NAD83 / UTM
zone 12N (`EPSG:26912`), which is metric and covers the full corridor.

## Recommended: stream the layer in QGIS

Do not download all 189 source tiles.

1. Open QGIS.
2. Choose **Layer → Add Layer → Add ArcGIS REST Server Layer**.
3. Choose **New**, name the connection `USGS NAIP`, and use:
   `https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer`
4. Connect, select `USGSNAIPImagery`, and add it.
5. Set the QGIS project CRS to `EPSG:26912`.
6. Zoom to the approximate frame location using the `lat` and `lon` columns in
   `metadata/frames_sampled.csv`.
7. Pan and zoom until stable landmarks in the aerial frame match the NAIP
   layer. QGIS requests only the visible map area and maintains its own bounded
   cache.

The balloon GPS is a search hint, not a claim that the pixel at that coordinate
is the center of the camera image. High-altitude and oblique views can place the
visible ground far from the platform position.

## Freeze only a needed chip

After an operator finds the matching map extent in QGIS, copy the extent in
WGS84 and fetch a bounded GeoTIFF:

```bash
canyonbench reference-chip work/reference/img_006806_reference.tif \
  --west -111.46 \
  --south 36.92 \
  --east -111.44 \
  --north 36.94 \
  --width-px 2000 \
  --height-px 2000
```

The command:

- selects natural-color bands from the 2023 primary NAIP imagery;
- reprojects the result to `EPSG:26912`;
- refuses either dimension above the USGS 4000-pixel service limit;
- writes `*.tif.reference.json` with the exact request, source terms, metric
  extent, byte count, and SHA-256;
- reuses a matching cached file instead of downloading it again; and
- refuses to overwrite a mismatched cache unless `--force` is explicit.

Keep chips under `work/reference/`; `work/` and GeoTIFFs are ignored by Git.
Commit control points, matrices, residuals, source YAML, and chip sidecars to
the data release. Large raster chips remain reproducible on-demand cache
objects, not Git history.

## Why this is the storage-safe choice

The full route is roughly 137 km long. Native 0.3 m imagery for the entire
corridor would be unnecessarily large, while Git is a poor transport for large
mutable rasters. Streaming plus small, checksummed exports preserves exact
provenance without keeping a statewide or route-wide imagery copy on every
computer.

Official source pages:

- USGS NAIP ImageServer:
  <https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer>
- USGS National Map terms:
  <https://www.usgs.gov/faqs/what-are-terms-uselicensing-map-services-and-data-national-map>
- USGS real-time GIS service access:
  <https://www.usgs.gov/the-national-map-data-delivery/gis-data-download>
