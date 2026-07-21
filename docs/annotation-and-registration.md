# Annotation and registration operations

The companion annotation manual is authoritative. Label Studio projects should paste the numbered rules verbatim. Two annotators independently label masks, presence, and quality; a shared 30-frame calibration set and 12-frame qualification set are tracked separately.

Masks are grayscale PNGs at exact frame dimensions with values only 0 and 255. Per-annotator names use `img_SSSSSS__ID.png`; adjudicated masks use `img_SSSSSS.png`. The validator flags non-binary masks and the mask utilities can detect connected foreground regions below four pixels.

Control points use point-like, stable landmarks. Use at least six, target eight, spread across all quadrants with a central point. Mark two as `holdout`; they do not participate in fitting. Reference coordinates must use a metric CRS before RMSE is interpreted in metres. Save the CRS, source imagery identifier/date/license, and ground footprint width.

Registration reliability is not a subjective flag. It is the held-out RMSE compared with one quarter of one 4x4 cell's ground width (`ground_width_m / 16`). Unreliable frames retain presence and vegetation-cover labels but have no grid record.

When calibrated horizontal/vertical field of view and altitude above local ground are available, `registration.geometry.estimate_ground_geometry` records the nadir/planar assumption, footprint dimensions, axis-specific metres per pixel, and the corresponding registration threshold. Do not substitute altitude above mean sea level for altitude above local ground without documenting the terrain model.

VARI thresholds are fitted only on the calibration split against human masks. Save the complete threshold-IoU curve. Never copy the selected threshold into a new split or release without recording its calibration provenance.
