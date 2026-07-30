# Architecture and invariants

CanyonBench-Trace is a fail-closed procedural data system.

```text
frozen orthoimage ─┐
primary masks ─────┤
secondary masks ───┤─> G1–G4 ─> site-independent split ─> virtual camera
optional DEM ──────┤                                      ├─ RGB
source manifest ───┘                                      ├─ exact masks
                                                           ├─ depth/camera
real-flight frames ─> quality calibration ─> one degradation on 240 views

clean view + exact mask ─> O1/O2/O3/(O4) target and matched-control curves
model + prompt ─> strict response ─> self deletion/sufficiency/controls
all predictions ─> deterministic metrics ─> site-aware inference
```

## Data boundary

The code repository contains algorithms, contracts, tests, prompts, and
preregistration. The data repository contains small public manifests and release
metadata. Raw source rasters, generated 1024-pixel views, interventions, run
responses, and held-out test material are ignored local/cloud artifacts.

## Base-site independence

A site is the only independent observation. Its eight clean renders, degraded
counterparts, prompts, operators, and intervention steps never cross splits.
Source tile IDs and feature IDs are audited for cross-split reuse. The release
builder hides all reserved-test coordinates and procedural seeds.

## Exact projection

The virtual camera computes one source quadrilateral and one perspective
transform. RGB uses cubic interpolation, masks use nearest-neighbor
interpolation, and depth uses linear interpolation; all share the same
homography and output dimensions. Input layers must already share CRS,
transform, bounds, width, and height. Any mismatch stops generation.

Because the source is already orthorectified, the synthesized view omits the
relief displacement a real camera records. That gap is quantified per view as
`d = r * relief / height_agl` and written into the view manifest, then aggregated
by `trace fidelity-report`. It corrupts no label: image and masks share the
transform. `dataset.inject_relief_displacement` can add a first-order DEM
displacement instead, applying one shared sampling grid to RGB, every mask, and
every continuous layer so labels stay exact; it is off in the frozen run and
requires a DEM for every site.

## Gate order

1. **G1 time alignment:** image and label dates must be within the registered
   gap; cultivated fields use the tightest threshold.
2. **G2 source consensus:** positives require agreement between two independent
   layers within a metric tolerance. Negatives require both sources to be empty
   across an expanded safety buffer.
3. **G3 resolvability/extinction:** components, apparent width, boundary
   distance, interior reach, local contrast, and occlusion are measured.
   Resolvability uses *interior reach* — how far the feature penetrates away from
   every frame border — not closest approach, because a road or river crossing the
   footprint necessarily touches the border while remaining fully resolvable.
   Closest approach stays reported as a diagnostic. Below-threshold positives are
   retained only as explicit extinction cases, and a view enters the extinction
   band only when apparent width, calibrated local contrast, the exclusion-only
   detector, and the human audit sample all agree there is no visible trace.
4. **G4 exclusion detector:** a weak detector may remove dubious candidates.
   It never adds a feature or changes a negative to positive.

## Intervention interpretation

O1–O3 are all primary. O4 is secondary until its artifact audit passes. Every
target edit is paired with the same operator and deletion level on an irrelevant
region matched on area, texture, edge density, brightness, center distance,
boundary complexity, and depth. Generation retries matched-region selection
and fails closed if the registered SMD bound cannot be met. O4 alone may be
marked secondary/rejected by its artifact proxy; O1–O3 remain required and are
audited by V2.

## Inference invariants

Every response is strict JSON with answer, confidence, selected cells, and a
ranking of exactly those cells. Parsing never repairs or coerces output. A run
is append-only and resumable by SHA-256 request ID. No-image prompts send no
image payload. Valid repeated calls are cached by source-image hash plus the
exact model, prompt, and schema settings while retaining one logical prediction
row per benchmark case. The detector adapter is a separate non-language
reference and is not sent language-only self-explanation or CAVE queries.

## Statistical invariants

Confidence intervals bootstrap groups and sites, never images. Comparisons are
paired within site. Mixed models include registered scale, geometry, quality,
prompt, intervention, and model-class effects plus site/group/feature
components. Benjamini–Hochberg correction covers the three primary endpoint
family: macro FPR, OCRS, and EFS.
