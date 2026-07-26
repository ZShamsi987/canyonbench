# CanyonBench

CanyonBench is a reproducible benchmark for measuring hallucination and spatial grounding in vision-language models on high-altitude aerial imagery. This repository contains the code. The separately versioned [CanyonBench data repository](https://github.com/ZShamsi987/canyonbench-data) contains release schemas, annotation templates, metadata, and—after curation—the publishable dataset.

The implementation follows four non-negotiable design choices from the project specification:

- ground truth describes visible image content, never the point directly beneath the balloon;
- spatial scores use only frames with validated frame-to-map registration;
- structured outputs are primary and deterministically scored;
- uncertainty is bootstrapped over trajectory segments, not correlated individual frames.

No raw flight data or imagery is committed here. The test suite uses generated fixtures, so the complete software can be installed and verified before the private source data is supplied.

## What is implemented

- operational-flight extraction from repeated power-cycle log segments, including embedded-header and zero-GPS filtering;
- clip inventory, anchor-based synchronization, one-frame-per-second extraction, left-third removal, deterministic elapsed-second naming, phase gates, perceptual/distance sampling, and master joins;
- homography fitting with held-out control points, metric RMSE, reliability thresholds, and overlay warping;
- storage-bounded USGS NAIP reference access with year-locked exports, checksums, and
  provenance sidecars;
- mask validation, green-cover fractions, 4x4 grounding labels, annotation agreement/adjudication, and calibrated VARI weak labels;
- strict presence, vegetation, grounding, false-premise, and caption probe contracts;
- resumable OpenAI-compatible inference with request budgets and immutable run manifests;
- deterministic core metrics, rule-based caption fallback, judge validation, calibration diagnostics, segment bootstrap confidence intervals, and controlled ascent regressions;
- release validation, provenance capture, Slurm templates, and publication-ready tidy result tables.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,registration,analysis]'
canyonbench --help
pytest
```

External executables used by data preparation are checked at runtime:

```bash
canyonbench doctor
```

Frame extraction requires `ffmpeg` and `ffprobe`. Registration requires the `registration` extra. Open-weight inference can use a project-specific model server through the OpenAI-compatible adapter; the heavy `open-weight` extra is only needed for custom local adapters.

## End-to-end workflow

```bash
# 1. Recover the operational flight and inventory private clips.
canyonbench flight-log data/private/WORLD10.txt work/flight.csv \
  --audit-output work/flight-segments.csv
canyonbench clips data/private/video work/clips.csv --order-by auto

# 2. Record a verified visual anchor, then extract second-indexed crops.
canyonbench sync work/clips.csv work/sync.json --anchor-clip clip_07.avi \
  --anchor-offset-s 113.0 --flight-elapsed-s 2742
canyonbench extract work/clips.csv work/sync.json work/frames_raw \
  --execute --resume
canyonbench name-frames work/clips.csv work/sync.json work/frames_raw work/frames_named \
  --mode hardlink

# 3. Join telemetry, apply the aerial phase gate, and sample correlated frames.
canyonbench build-frames work/frames_named work/flight.csv work/frames_candidates.csv \
  --drop-unmatched --unmatched-output work/unmatched-frames.csv
canyonbench sample work/frames_candidates.csv work/frames_sampled.csv \
  --min-interval-s 60 --distance-m 500 --phash-distance 8

# 4. Merge adjudicated annotations and validated registration outputs.
canyonbench reference-chip work/reference/selected-area.tif \
  --west -111.46 --south 36.92 --east -111.44 --north 36.94 \
  --width-px 2000 --height-px 2000
canyonbench build-release work/frames_sampled.csv /path/to/canyonbench-data work/release
canyonbench validate-release work/release

# 5. Run and score a constrained probe battery.
canyonbench run configs/run.example.yaml
canyonbench score work/release results/example/predictions.jsonl results/example/metrics.json
```

Commands default to safe, inspectable behavior: extraction prints commands unless `--execute` is passed; inference resumes by request key; API calls require an explicit budget and refuse to exceed it.

For oversized Google Drive for desktop inputs on macOS, add `--evict-source-cache` to
`clips` and `extract` so successfully read source ranges are returned to cloud-only state in
bounded batches.

Reference imagery is also remote-first. QGIS can stream the official USGS NAIP
ImageServer, and `reference-chip` downloads only a chosen bounding box into ignored
`work/reference/`. Every GeoTIFF gets a reproducible JSON sidecar. Do not download the
189 full-resolution source tiles covering the route. See
[docs/reference-imagery.md](docs/reference-imagery.md).

## Repository map

```text
src/canyonbench/
  pipeline/       telemetry, clips, sync, extraction, sampling, joins
  registration/   control points, homographies, validation, overlays
  groundtruth/    masks, grids, annotation agreement, VARI weak labels
  eval/           probes, adapters, inference, judge, metrics, statistics
  reporting/      tidy tables and plots
configs/          versioned probe and run contracts
slurm/            Adroit-oriented job templates
scripts/          reproducible orchestration helpers
tests/            unit and synthetic end-to-end coverage
docs/             design decisions and operational guides
```

See [docs/architecture.md](docs/architecture.md),
[docs/data-ingestion.md](docs/data-ingestion.md),
[docs/reference-imagery.md](docs/reference-imagery.md),
[docs/evaluation.md](docs/evaluation.md), and
[docs/releasing.md](docs/releasing.md) before running on private data.

## Reproducibility and scope

Every release and inference run records input hashes, configuration, package version, timestamp, and source commit. Results distinguish raw frame count from effective segment count. Ascent findings are labeled associations—not causal altitude effects—and controlled analyses include image quality, feature prevalence, and phase.

## License and citation

Code is MIT licensed. Dataset licensing is documented separately because source imagery, reference layers, and derived annotations can have different terms. Use `CITATION.cff` for the current software citation; replace release placeholders when the paper and DOI are frozen.
