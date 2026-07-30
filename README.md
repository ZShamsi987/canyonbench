# CanyonBench-Trace

CanyonBench-Trace is a procedural benchmark for testing whether vision-language
models use causal visual evidence in high-altitude aerial imagery. A calibrated
virtual camera renders the same geographic site at 3, 8, 16, and 24 km AGL in
nadir and moderate-oblique geometry. The exact same crop, resampling operation,
and homography are applied to RGB imagery, three independent feature masks, and
optional terrain depth. This makes every target location exact without human
control points.

The benchmark asks whether **water**, a **major road**, or a **cultivated field**
is visible; where the model says its evidence is; whether deleting that evidence
changes its answer; whether preserving only that evidence recovers its answer;
and whether the same model is merely sensitive to matched irrelevant edits.

This repository is the code and protocol. The sibling
[CanyonBench-data](https://github.com/ZShamsi987/canyonbench-data) repository is
the versioned data contract and will hold public manifests and releases. Large
source rasters, generated views, interventions, model runs, and reserved test
artifacts stay out of Git.

## Status

The complete v4 software path is implemented and tested with synthetic
georeferenced data. Real benchmark generation begins when the frozen site/source
manifest is supplied. No semantic annotation campaign is needed. Human
involvement is limited to two coauthors independently answering four objective
binary audit questions on a stratified 5–10% sample.

The previous dashcam-style annotation and manual registration workflow remains
in the package only as a legacy/optional real-flight tier. It is not used to
create primary v4 labels, splits, or scores.

**Start here: [RUNBOOK.md](RUNBOOK.md)** gives the exact command sequence across
both hosts and marks every step that blocks on a human. The hardware split is in
[docs/compute-and-storage.md](docs/compute-and-storage.md), the frozen day-by-day
handoff in [docs/execution-plan.md](docs/execution-plan.md), and the money,
pre-committed cuts, and risk register in
[docs/budget-and-risk.md](docs/budget-and-risk.md).

Execution needs exactly three things: an `OPENROUTER_API_KEY`, SSH to Adroit for
the CPU half, and SSH to a Lambda GPU instance for the served models.

## What is implemented

- strict v4 schemas for sites, source provenance, gates, cameras, quality,
  derived geometry, interventions, prompts, predictions, audits, and CAVE;
- exact aligned raster/vector preparation and virtual-camera rendering;
- 120-site quota enforcement and 20/20/60 site-independent splitting;
- gates G1–G4: temporal alignment, source consensus, resolvability/extinction,
  and exclusion-only detector screening;
- all 960 clean views and a balanced 240-view, exactly-one-degradation subset;
- O1 local blur, O2 matched background texture, O3 frequency removal, and
  secondary O4 inpainting at 25/50/75/100% target deletion;
- target-shape matched distractors, balance diagnostics, edit-artifact scores,
  model-self-evidence deletion/sufficiency, and random/texture controls;
- structured-output model adapters, including OpenAI-compatible endpoints,
  vLLM, a non-language HTTP detector, and offline fixtures;
- resumable Tier A/B/C runs with hard request/cost budgets, up to three parse
  retries, explicit format failures, and no silent coercion;
- all registered accuracy, localization, OCRS, SEN, SES, EFS, OSG, prompt-prior,
  extinction, acuity, and selective-risk measures;
- CAVE verification for neutral and false-premise contexts, development-only
  threshold tuning, component ablations, and reliability/coverage/cost frontier;
- site/group hierarchical bootstrap, paired site tests, mixed effects, and
  Benjamini–Hochberg correction;
- V1 synthetic inserts, V2 edit detectability, V3 operator agreement,
  V4 grid/K sensitivity support, V5 reference controls, and V6 suppression
  efficacy;
- per-view sim-to-real relief-displacement accounting (`d = r*dh/H`) with an
  optional, off-by-default DEM relief injection for oblique views;
- a D1 price pilot that measures real per-call tokens and re-prices the whole
  plan in dollars before any production spend;
- the GSD/apparent-width extinction figure and geometric ladder table, plus the
  human confirmation of the extinction band from the objective audit;
- the registered two-resource split: Adroit CPU SLURM jobs for acquisition,
  generation, gates, interventions, and analysis; a Lambda GPU driver that
  pre-retrieves weights, serves each model in turn under a capability-adaptive
  bfloat16 profile, and resumes an interrupted session at the exact request;
- self-verifying public development-release construction and coordinate-free
  held-out-test escrow hashes.

## Install and verify

Python 3.11 or newer is required.

```bash
cd /Users/zafirshamsi/CanyonBench
python3 -m pip install 'uv==0.11.30'
uv sync --frozen --extra trace --extra dev
source .venv/bin/activate
canyonbench doctor
pytest
ruff check .
mypy src/canyonbench
```

`uv.lock` is the exact cross-platform environment freeze. Use
`python -m pip install -e '.[trace,dev]'` only as an unlocked development
fallback.

The test suite does not require private imagery, API keys, or paid calls. It
creates synthetic GeoTIFFs, renders the full altitude/geometry lattice for a
smoke site, generates interventions, runs a fixture model, and scores it.

## Primary workflow

### 1. Freeze sources and sites

The repository includes the complete public-source acquisition path and its
frozen policy in [configs/trace_sources.yaml](configs/trace_sources.yaml).
Discovery has produced 230 candidate centers. To reproduce or resume:

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

The second command materializes 2 m NAIP imagery, two independent masks per
class, three independent exclusion layers, 3DEP terrain, exact attribution,
and hashes. It is atomic and restart-safe; use `--start`/`--limit` for chunks.
Large caches/sources are ignored by Git. Follow
[docs/data-ingestion.md](docs/data-ingestion.md) for the contracts.

### 2. Calibrate quality from the real flight

The WORLD10 telemetry and balloon video are not primary ground truth. Use usable
real frames to freeze the distribution of sharpness, haze, exposure,
saturation/contrast, compression, horizon frequency, and rig obstruction:

The frozen calibration already exists at
`CanyonBench-data/calibration/flight-quality.json`; it covers all 377 original
flight frames and has SHA-256
`c5e2cf71db76f7eeac512e9607e59a7118993d84e99eb01021bcfdd3481f59d7`.

Record the calibration file SHA-256. Dashcam timestamps are ignored; only the
verified clip order is used. GPS drift may be derived from position because
logged horizontal/down velocities are zero. Logged heading, thermal, and
spectral channels are not treated as validated.

### 3. Generate the dataset

Review [configs/trace.yaml](configs/trace.yaml). Gate a larger candidate list,
freeze the quota-balanced independent cohort, run the registered gate
ablations, and then build:

```bash
canyonbench trace select-sites \
  /path/to/candidate-sites.yaml \
  configs/trace.yaml \
  /Users/zafirshamsi/CanyonBench-data/manifests/sites.yaml
canyonbench trace gate-ablations \
  configs/trace.yaml \
  /Users/zafirshamsi/CanyonBench-data/reports/gate-ablations.json
canyonbench trace build configs/trace.yaml
canyonbench trace validate \
  /Users/zafirshamsi/CanyonBench-data/generated \
  --config configs/trace.yaml \
  --output /Users/zafirshamsi/CanyonBench-data/reports/dataset-validation.json
```

The build fails closed on missing sources, gate failures, pixel misalignment,
split leakage, or manifest mismatch. It writes immutable hashes for every view.

Then report the two instrument products of the frozen bundles — the sim-to-real
relief gap and the extinction ladder:

```bash
canyonbench trace fidelity-report \
  /Users/zafirshamsi/CanyonBench-data/generated \
  --output /Users/zafirshamsi/CanyonBench-data/reports/relief-displacement.json
canyonbench trace extinction-figure \
  /Users/zafirshamsi/CanyonBench-data/generated \
  /Users/zafirshamsi/CanyonBench-data/reports/figures
```

Relief displacement (`d = r*dh/H`) is absent from an orthophoto-derived view and
corrupts no label, because image and masks share one transform. It is reported,
not hidden. DEM-based injection exists behind
`dataset.inject_relief_displacement` and stays off in the frozen run.

### 4. Perform the small objective audit

```bash
canyonbench trace audit-sample \
  /Users/zafirshamsi/CanyonBench-data/generated \
  /Users/zafirshamsi/CanyonBench-data/audits/audit.csv \
  --auditor-1 AUD-KUNSH --auditor-2 AUD-ATHARVA --fraction 0.1

# After both auditors independently fill yes/no fields:
canyonbench trace audit-summary \
  /Users/zafirshamsi/CanyonBench-data/audits/audit.csv \
  /Users/zafirshamsi/CanyonBench-data/reports/audit-summary.json \
  --dataset-dir /Users/zafirshamsi/CanyonBench-data/generated
```

Auditors only check overlay alignment, resolvability, obvious edit artifacts,
and source mismatch. They do not draw masks or decide semantic labels.
Passing `--dataset-dir` joins the `feature_resolvable` votes onto the derived
flags, which supplies the fourth extinction criterion: humans confirm no visible
trace where width, contrast, and the detector already agree there is none. A
contradicted extinction view is regenerated or excluded, never relabeled.
`audit-sample` also creates native-scale review sheets; follow the data
repository's `AUDIT_GUIDE.md`.

### 5. Freeze and run the model roster

Use the ready roster in
[configs/trace_run.frozen.yaml](configs/trace_run.frozen.yaml). It freezes
three proprietary vendors, three Qwen3-VL open-weight sizes, EarthDial, the
non-language detector, and observed 2026-07-29 prices. Set credentials only in
environment variables and recheck the cost plan immediately before inference.
Preflight the request cap and the dollar projection before any paid call, then
measure the real price with the D1 pilot and re-project from the measurement:

```bash
canyonbench trace plan-run configs/trace_run.frozen.yaml \
  --output /Users/zafirshamsi/CanyonBench-data/reports/call-plan.json
canyonbench trace price-pilot configs/trace_run.frozen.yaml --calls 50
canyonbench trace plan-run configs/trace_run.frozen.yaml \
  --price-pilot /Users/zafirshamsi/CanyonBench-data/runs/frozen-2026-07-29/price_pilot/price_pilot.json \
  --output /Users/zafirshamsi/CanyonBench-data/reports/call-plan-measured.json
canyonbench trace run configs/trace_run.frozen.yaml
canyonbench trace reference-baselines \
  /Users/zafirshamsi/CanyonBench-data/generated \
  /Users/zafirshamsi/CanyonBench-data/results/final/reference-baselines.csv
canyonbench trace cave-tune \
  /Users/zafirshamsi/CanyonBench-data/generated \
  /Users/zafirshamsi/CanyonBench-data/runs/final/predictions.jsonl \
  /Users/zafirshamsi/CanyonBench-data/results/final/cave-thresholds.json
canyonbench trace cave-apply \
  /Users/zafirshamsi/CanyonBench-data/generated \
  /Users/zafirshamsi/CanyonBench-data/runs/final/predictions.jsonl \
  /Users/zafirshamsi/CanyonBench-data/results/final/cave-thresholds.json \
  /Users/zafirshamsi/CanyonBench-data/results/final/cave-decisions.jsonl
canyonbench trace cave-ablations \
  /Users/zafirshamsi/CanyonBench-data/generated \
  /Users/zafirshamsi/CanyonBench-data/runs/final/predictions.jsonl \
  /Users/zafirshamsi/CanyonBench-data/results/final/cave-thresholds.json \
  /Users/zafirshamsi/CanyonBench-data/results/final/cave-ablations.jsonl
canyonbench trace score \
  /Users/zafirshamsi/CanyonBench-data/generated \
  /Users/zafirshamsi/CanyonBench-data/runs/final/predictions.jsonl \
  /Users/zafirshamsi/CanyonBench-data/results/final/metrics.json \
  --cave-decisions /Users/zafirshamsi/CanyonBench-data/results/final/cave-decisions.jsonl \
  --cave-ablations /Users/zafirshamsi/CanyonBench-data/results/final/cave-ablations.jsonl \
  --cave-frontier /Users/zafirshamsi/CanyonBench-data/results/final/cave-thresholds.frontier.json
canyonbench trace report \
  /Users/zafirshamsi/CanyonBench-data/results/final/metrics.json \
  /Users/zafirshamsi/CanyonBench-data/results/final/metrics.rows.csv \
  /Users/zafirshamsi/CanyonBench-data/reports/final \
  --dataset-dir /Users/zafirshamsi/CanyonBench-data/generated
```

`trace report` writes every registered figure and the main table. Copy them flat
into `paper/` as described in [paper/README.md](paper/README.md); an absent
artifact renders as a visible `TBD` rather than an empty float.

Never put API keys in YAML or Git. Runs resume by a content-derived request ID
and stop before exceeding the declared request or dollar cap. Run the complete
Tier-A checkpoint for every model before B/C. Keep the costly 4/6/8-grid by
K=3/6/10 sensitivity analysis in a separately frozen run config with
`analyses: [sensitivity]`.
The registered roster uses a common 768-pixel encoded long edge and
`adapter.max_retries: 0`; the harness owns all three allowed retries so no
provider attempt bypasses request/cost accounting.

Before interpreting causal results, generate V1 controls from frozen negative
views and run the automatic V2 audit. V3 rank agreement and V6 detector
suppression efficacy are computed by `trace score`.

```bash
canyonbench trace instrument-v1 \
  /path/to/frozen-negative-rgb.png \
  /Users/zafirshamsi/CanyonBench-data/reports/v1-controls
canyonbench trace instrument-v1-run \
  /path/to/frozen-trace-run.yaml \
  /path/to/frozen-negative-rgb.png \
  /Users/zafirshamsi/CanyonBench-data/reports/v1-controls
canyonbench trace instrument-v2 \
  /Users/zafirshamsi/CanyonBench-data/generated \
  /Users/zafirshamsi/CanyonBench-data/reports/v2-edit-detectability.json
```

`instrument-v1` only materializes and hashes the controls.
`instrument-v1-run` performs the registered end-to-end positive-control test
against the frozen roster and writes its predictions and metrics.

### 6. Build the release

```bash
canyonbench trace release \
  /Users/zafirshamsi/CanyonBench-data/generated \
  /Users/zafirshamsi/CanyonBench-data/releases/v4
```

Development and validation artifacts are copied to `public/`. The 60% test
coordinates, camera/degradation seeds, and intervention masks remain private;
`escrow/held_out_hashes.json` proves what was frozen without disclosing them.

## Repository map

```text
src/canyonbench/trace/
  schemas.py        strict public contracts
  sources.py        raster alignment and GeoJSON rasterization
  geometry.py       shared virtual-camera transform
  fidelity.py       relief-displacement quantification and optional injection
  gates.py          G1–G4 inclusion logic
  render.py         view bundles and exact degradation subset
  degradation.py    real-flight calibration and single-factor edits
  interventions.py O1–O4, distractor matching, artifact proxy
  protocol.py       stratified tiers and model-self evidence edits
  runner.py         resumable strict-output inference
  planning.py       request and dollar preflight against the caps
  pilot.py          D1 price pilot and measured re-projection
  metrics.py        registered benchmark endpoints
  reporting.py      registered figures, ladder, and main table
  merge.py          join the per-host prediction logs for scoring
  cave.py           causal verification wrapper
  instruments.py    V1–V6 validation utilities
  statistics.py     site-aware inference
  audit.py          objective two-auditor checks
  validation.py     dataset integrity audit
  release.py        public/held-out packaging
src/canyonbench/
  compute.py        VRAM/instance contract, serving profiles, storage layout
configs/            generation, prompts, frozen roster, preregistration
slurm/              Adroit CPU jobs: preflight, acquire, build, instruments,
                    OpenRouter inference, analysis
scripts/adroit/     one-time Adroit setup
scripts/lambda/     GPU bootstrap, weight pre-retrieval, sequential serving,
                    dataset push and result fetch
tests/              unit and synthetic end-to-end verification
docs/               compute, source, evaluation, audit, budget, release guides
RUNBOOK.md          the execution order for both hosts
```

## Non-negotiable invariants

1. Labels come from frozen procedural masks, never model output or human control
   points.
2. RGB, masks, and depth receive one identical camera transform, including the
   optional relief-displacement resampling, so alignment never depends on the
   geometry option in force.
3. G4 can exclude a candidate but can never create a label.
4. Extinction cases remain explicit; they are not relabeled as negatives.
5. Only one calibrated degradation may be applied to a degraded view.
6. The independence unit is the base site, not its renders.
7. A source tile, road segment, water body, parcel, coordinate, or overlapping
   footprint cannot cross splits.
8. Malformed model output is retried at most three times, then recorded as a
   format failure.
9. Target deletion is interpreted only beside matched distractor and
   random/texture controls.
10. CAVE thresholds are tuned on development only and never against test.

## License and citation

Code is MIT licensed. Each data source retains its own terms; the source
manifest and release report must preserve those terms and redistribution
status. Do not publish a raster merely because the code can process it. Update
`CITATION.cff` and the paper DOI only when the release is frozen.
