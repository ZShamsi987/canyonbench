# RUNBOOK — exactly what to run, in order

Everything in this file is compute. The software is complete and tested; no code
needs to be written to execute the project.

**What you need, and nothing else:**

1. `OPENROUTER_API_KEY`
2. SSH access to Adroit (CPU work)
3. SSH access to a Lambda GPU instance (open-weight inference)

**Two hosts, one rule:** Adroit builds the dataset and runs everything that is
not a served model. Lambda serves the open-weight models. Nothing else is split.

Steps marked **[BLOCKS]** stop the pipeline until a human acts. Steps marked
**[OUTPUT]** produce something you must read before continuing.

---

## Stage 0 — One-time setup (both hosts, ~40 minutes)

### 0.1 Adroit

```bash
ssh <you>@adroit.princeton.edu
git clone https://github.com/ZShamsi987/canyonbench.git ~/CanyonBench
bash ~/CanyonBench/scripts/adroit/bootstrap.sh
```

Then add to `~/.bashrc` exactly what the script prints, including:

```bash
export OPENROUTER_API_KEY=sk-or-...      # your key, never in the repo
export CANYONBENCH_DATA=/scratch/network/$USER/CanyonBench-data
export CANYONBENCH_DATASET_DIR=$CANYONBENCH_DATA/generated
```

```bash
cd ~/CanyonBench && sbatch slurm/adroit_preflight.sbatch
```

**[OUTPUT]** Read `logs/cb-preflight-*.out`. It records the partitions, GPU
types, and maximum wall time your account actually has. If the wall-time limit is
under 12 hours, tell me and I will re-chunk the acquisition and API jobs.

### 0.2 Lambda

Launch **1× H100 80 GB PCIe** (or SXM5). A 40 GB A100 works for everything except
the 32B model — see `docs/compute-and-storage.md` if only A100s are available.

```bash
ssh ubuntu@<lambda-ip>
sudo mkdir -p /lambda/canyonbench && sudo chown -R $USER /lambda/canyonbench
git clone https://github.com/ZShamsi987/canyonbench.git /lambda/canyonbench/CanyonBench
bash /lambda/canyonbench/CanyonBench/scripts/lambda/bootstrap.sh
```

**[OUTPUT]** The script prints `READY` or `BLOCKED`. It verifies the persistent
filesystem, the GPU, native bfloat16, and vLLM. Do not proceed on `BLOCKED`.

You may terminate this instance now and relaunch later — everything it wrote
lives on the persistent filesystem.

---

## Stage 1 — Build the dataset (Adroit only, D1–D7)

Nothing here touches a GPU or spends money.

```bash
cd ~/CanyonBench

# 1.1 Discover, acquire, and freeze 120 independent sites  (~4-10 h, resumable)
sbatch slurm/adroit_acquire.sbatch

# 1.2 Generate 960 clean + 240 degraded views and all interventions  (~2-6 h)
sbatch slurm/adroit_build.sbatch

# 1.3 Instrument validation V1/V2 and the audit packet  (~1 h)
sbatch slurm/adroit_instruments.sbatch $CANYONBENCH_DATASET_DIR/site_0001/view_a3km_nadir/rgb.png
```

**[OUTPUT]** After 1.2, read `$CANYONBENCH_DATA/reports/dataset-validation.json`.
`passed: true` is required. Any error there is a real defect — send it to me
rather than regenerating around it.

**[BLOCKS] 1.4 — the objective audit.** `adroit_instruments.sbatch` writes
`$CANYONBENCH_DATA/audits/audit.csv` plus review sheets. Kunsh and Atharva each
fill the four binary columns independently, without discussing them first. Then:

```bash
canyonbench trace audit-summary \
  $CANYONBENCH_DATA/audits/audit.csv \
  $CANYONBENCH_DATA/reports/audit-summary.json \
  --dataset-dir $CANYONBENCH_DATASET_DIR
```

**[OUTPUT]** If `source_mismatch_view_rate` exceeds 0.10, the registered kill
criterion fires: narrow to two classes and tighten the gates. Do not proceed past
this without reading the summary.

### DATASET FREEZE

No generator change after this point without restarting every model run.

---

## Stage 2 — Run the models (D8–D14)

Stage 2A and 2B are independent and can run at the same time.

### 2A — Open-weight models on Lambda

```bash
# From Adroit: push the frozen bundle (20-40 GB; source tiles never move)
export LAMBDA_HOST=ubuntu@<lambda-ip>
bash scripts/lambda/sync_dataset.sh

# On Lambda, in one session:
ssh $LAMBDA_HOST
source /lambda/canyonbench/env.sh
cd /lambda/canyonbench/CanyonBench
bash scripts/lambda/prefetch_weights.sh      # C1: before any GPU time is billed
bash scripts/lambda/run_open_weight.sh       # C3: serves each model in turn

# From Adroit, when it finishes:
bash scripts/lambda/fetch_results.sh
```

Then terminate the Lambda instance. If the session dies partway, relaunch and
rerun `run_open_weight.sh` — it resumes at the exact request it stopped on (C2).

### 2B — Proprietary models over the API (from Adroit)

```bash
sbatch slurm/adroit_openrouter.sbatch
```

**[BLOCKS]** The job runs the 50-call price pilot first and **stops** if the
measured projection does not fit the $220 cap.

**[OUTPUT]** Read `$CANYONBENCH_DATA/reports/price-pilot.json` before the full
run proceeds — specifically `measured_projection_usd.nominal_usd` and
`authorized`. This is the one number that decides whether the run is affordable,
and it is measured rather than assumed.

### RESULTS FREEZE

---

## Stage 3 — Analyze and write (Adroit only, D15–D21)

```bash
sbatch slurm/adroit_analyze.sbatch
```

This merges both hosts' logs, tunes and applies CAVE, computes every metric with
site/group bootstrap, mixed effects, and Benjamini–Hochberg, and writes every
figure and the main table.

**[OUTPUT]** `$CANYONBENCH_DATA/reports/final/` contains the figures and
`model_summary.tex`. Copy them flat into `paper/`:

```bash
cp $CANYONBENCH_DATA/reports/final/{gsd_extinction,intervention_traces,\
accuracy_vs_efs,acuity_curves,extinction_by_altitude,cave_frontier}.pdf paper/
cp $CANYONBENCH_DATA/reports/final/model_summary.tex paper/
```

Then build the release and the paper:

```bash
canyonbench trace release $CANYONBENCH_DATASET_DIR $CANYONBENCH_DATA/release
bash scripts/check.sh
```

---

## What I need from you (and only this)

| # | Item | When | Why it cannot be automated |
|---|---|---|---|
| 1 | `OPENROUTER_API_KEY` exported on Adroit | Stage 0 | Credential; never committed |
| 2 | Lambda instance launched, ≥80 GB card | Stage 0.2 | Costs credit; your call which type |
| 3 | Adroit wall-time limit from the preflight output | Stage 0.1 | Determines whether jobs need re-chunking |
| 4 | Two auditors complete `audit.csv` | Stage 1.4 | Human judgment is the fourth extinction criterion |
| 5 | Approve the measured price projection | Stage 2B | Spends real money |
| 6 | An endpoint for the non-language detector | Stage 2A | See below |
| 7 | EarthDial served on port 8001 | Stage 2A | See below |

### On items 6 and 7

Two roster entries need a service that does not exist in this repository:

- `canyonbench-independent-detector-v1` — the §12 non-language upper reference,
  expected as an HTTP detector on `127.0.0.1:8010`.
- `akshaydudhane/EarthDial_4B_RGB` — expected on `127.0.0.1:8001` behind a wrapper
  that enforces the CanyonBench JSON schema, because EarthDial exposes no native
  structured-output API.

If plain vLLM serves EarthDial acceptably, `run_open_weight.sh` will start it on
:8001 automatically and nothing more is needed. If it fails the health check, the
script tells you exactly that and stops.

If either service is unavailable, the registered fallback is to drop that model
and record the omission — the roster validator permits 2–3 open-weight and 1–2
remote-sensing models, and the descope ladder covers it. Tell me and I will make
that change explicitly rather than letting it happen silently.

### Nothing else is needed

No annotation campaign, no manual registration, no model training, no hidden
evaluation server, and no GPU on Adroit.
