# Release checklist

## Before dataset freeze

1. Validate exactly 120 sites and the registered group/class balance.
2. Confirm all redistribution decisions and source acknowledgments.
3. Freeze source manifests, dates, versions, URLs, licenses, tile IDs, feature
   IDs, and SHA-256 values.
4. Freeze `configs/trace.yaml` and `configs/preregistration.yaml`.
5. Generate 960 clean and 240 singly degraded views.
6. Run G1–G4, pixel alignment, hash, quota, and split-leakage validation.
7. Generate O1–O4 target/distractor curves and report every matching SMD.
8. Complete V1, V2, and V6 plus the objective two-auditor sample.
9. If generator behavior changes after freeze, regenerate and restart model
   runs; do not patch individual test artifacts.

## Before results freeze

1. Freeze the target eight-model roster (minimum six, maximum nine) and
   provider prices. Preserve every required model category.
2. Run Tier A fully before Tier B/C.
3. Confirm request/cost manifests and format-failure accounting.
4. Run all required operator, grid, K, detector, source, CAVE, geographic, and
   quality ablations.
5. Tune CAVE on development only.
6. Compute site/group intervals and BH-adjusted primary comparisons.
7. Record results on held-out test once. Never use test to tune prompts,
   thresholds, operators, or exclusions.

## Package

```bash
canyonbench trace validate /path/to/generated --config configs/trace.yaml
canyonbench trace release /path/to/generated /path/to/releases/v4
canyonbench trace release-validate \
  /path/to/releases/v4/public /path/to/releases/v4/escrow
scripts/check.sh
```

The public package contains development/validation views, masks, depth, cameras,
interventions, source manifests, schemas, prompts, adapters, CAVE, analysis, and
environment information. The reserved 60% test coordinates, camera seeds,
degradation seeds, and intervention masks remain private. The escrow manifest
contains only opaque IDs and hashes.

GitHub holds code and small manifests. Use an appropriate dataset host and
archival DOI for large public artifacts. Validate the archive after upload and
record its checksum. Update citation/DOI placeholders only from the frozen
release.
