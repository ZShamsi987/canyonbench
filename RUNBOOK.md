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
export CANYONBENCH_HOME=/scratch/network/$USER/canyonbench-trace
export CANYONBENCH_DATA=/scratch/network/$USER/canyonbench-trace-data
git clone https://github.com/ZShamsi987/canyonbench.git "$CANYONBENCH_HOME"
bash "$CANYONBENCH_HOME/scripts/adroit/bootstrap.sh"
```

Then add to `~/.bashrc` exactly what the script prints, including:

```bash
export OPENROUTER_API_KEY=sk-or-...      # your key, never in the repo
export CANYONBENCH_HOME=/scratch/network/$USER/canyonbench-trace
export CANYONBENCH_DATA=/scratch/network/$USER/canyonbench-trace-data
export CANYONBENCH_DATASET_DIR=$CANYONBENCH_DATA/generated
```

```bash
cd "$CANYONBENCH_HOME" && sbatch slurm/adroit_preflight.sbatch
```

Nothing above edits `~/.bashrc`, loads a module, or touches any other directory.
The bootstrap **refuses to run** if the target path exists and is not this
project. To remove the project later: `rm -rf $CANYONBENCH_HOME $CANYONBENCH_DATA`.

**[OUTPUT]** Read `logs/cb-preflight-*.out`. It runs `qos` and `sacctmgr` and
records the partitions and wall-time limits your account actually has. Send me
that section: the jobs are already chunked to fit a 4-hour ceiling, and if your
limit is higher I can collapse them into fewer, faster submissions.

### 0.2 Lambda — **do this only once the dataset is built and audited**

An idle instance still bills, so launch it at Stage 2A, not before.

Launch **`gpu_1x_a100_sxm4` in us-east-1 ($1.99/hr)** and attach the
**`CanyonBench`** filesystem. A Lambda filesystem mounts only in its own region
and us-east-1 has no H100, so this is the only single-GPU card that can see your
data — and it is also the cheapest Lambda offers anywhere. It serves the 8B,
EarthDial, and the detector; the 32B and 235B are reached over the API because
neither fits 40 GB in bfloat16.

Budget: $40 of the $400 credit is reserved for storage, leaving **$340 for GPU** —
about 171 hours at $1.99/hr against a 15-hour projection.

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
cd "$CANYONBENCH_HOME"

# All of this writes to exactly two directories: $CANYONBENCH_HOME (the
# checkout) and $CANYONBENCH_DATA. Nothing else on Adroit is touched.

# 1.1 Acquire sources on the LOGIN node, where Adroit permits outbound network
# access.  Do not submit slurm/adroit_acquire.sbatch on Adroit: compute nodes
# have no DNS or TCP egress.
nohup setsid nice -n 19 bash scripts/adroit/acquire_login.sh \
  > "$CANYONBENCH_DATA/logs/acquire-login.log" 2>&1 < /dev/null &
echo "acquisition pid $!"
tail -f "$CANYONBENCH_DATA/logs/acquire-login.log"

# Continue only after that log ends with "ACQUIRE OK".  The login-node workflow
# writes one complete prepared manifest, which adroit_freeze.sbatch accepts.
sbatch slurm/adroit_freeze.sbatch

# After the freeze job succeeds, generate 960 clean + 240 degraded views and
# all interventions (~2-6 h).
sbatch slurm/adroit_build.sbatch

# Read dataset-validation.json and require passed: true before submitting the
# instrument job.  Then run V1/V2 and create the audit packet (~1 h).
sbatch slurm/adroit_instruments.sbatch $CANYONBENCH_DATASET_DIR/site_0001/view_a3km_nadir/rgb.png
```

Acquisition is intentionally a single niced, resumable login-node process. It
is I/O-bound, writes each artifact atomically, and skips already complete sites,
so it is safe to stop and restart. The historical Slurm-array script remains
for clusters whose compute nodes have egress, but is not usable on Adroit.

**[OUTPUT]** After 1.3, read `$CANYONBENCH_DATA/reports/dataset-validation.json`.
`passed: true` is required. Any error there is a real defect — send it to me
rather than regenerating around it.

**[BLOCKS] 1.5 — the objective audit. This happens NOW, before any model runs
and before any money is spent.** `adroit_instruments.sbatch` writes
`$CANYONBENCH_DATA/audits/audit.csv` plus one review sheet per audited view.

Send both auditors **`docs/AUDITOR-GUIDE.md`**, the CSV, and the
`audit.csv_assets/` folder. They need no cluster access and no software. Budget
3–5 hours each. They must work independently and must not compare answers before
both submit — the agreement rate between them is reported in the paper.

Their work gates the dataset freeze: if the audit finds problems, sites are
regenerated or dropped, which cannot happen after models have run against them.
Stage 2 can start the moment both CSVs are back. Then:

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
python scripts/lambda/serve_detector.py --port 8010 &   # the upper reference
export CANYONBENCH_ENDPOINT__CANYONBENCH_INDEPENDENT_DETECTOR_V1=http://127.0.0.1:8010
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
| 2 | Lambda `gpu_1x_a100_sxm4` in us-east-1 + `CanyonBench` filesystem | Stage 2A | Costs credit; launch only when the dataset is ready |
| 3 | Adroit wall-time limit from the preflight output | Stage 0.1 | Determines whether jobs need re-chunking |
| 4 | Two auditors complete `audit.csv` | Stage 1.5 | Human judgment is the fourth extinction criterion |
| 5 | Approve the measured price projection | Stage 2B | Spends real money |
| 6 | Start the detector service (one command, provided) | Stage 2A | See below |
| 7 | Confirm the OpenRouter prices are still current | Stage 2B | See below |

### On item 6 — endpoints, in plain terms

**An endpoint is not a service you sign up for.** It is the URL of a server
process running on your own Lambda box. There are only four kinds in this
project, and three of them start themselves:

| What | Who starts it | URL |
|---|---|---|
| The 3 proprietary models + Qwen 235B | nobody — public API | `https://openrouter.ai/api/v1` |
| Qwen 8B / 32B | `run_open_weight.sh` | `http://127.0.0.1:8000/v1` |
| EarthDial | `run_open_weight.sh` | `http://127.0.0.1:8001/v1` |
| Detector reference | you, one command | `http://127.0.0.1:8010` |

**In the normal case you give me nothing.** Those URLs are already in the frozen
roster, the scripts bind exactly those ports, and `127.0.0.1` is right because
the driver runs on the same machine. The only thing you actually run is:

```bash
python scripts/lambda/serve_detector.py --port 8010 &
curl -s localhost:8010/health     # expect {"status": "ok", ...}
```

Tell me an address only if you start something *somewhere else* — a different
port, or another host. Then export one variable rather than editing the roster:

```bash
export CANYONBENCH_ENDPOINT__CANYONBENCH_INDEPENDENT_DETECTOR_V1=http://127.0.0.1:9100
```

Pattern: `CANYONBENCH_ENDPOINT__` + model id upper-cased, non-alphanumerics → `_`.

If a service cannot be stood up at all, the registered fallback is to drop that
model and record the omission — the validator permits 2–3 open-weight and 1–2
remote-sensing models. Tell me and I will make that cut explicitly.

### On item 7 — prices to verify

Open https://openrouter.ai/models and confirm the input/output rates per million
tokens for exactly these five, then paste them to me:

| Model id in the roster | Frozen input $/M | Frozen output $/M | Predicted spend |
|---|---|---|---|
| `openai/gpt-5.6-sol` | 5.00 | 30.00 | $79.71 |
| `anthropic/claude-opus-5` | 5.00 | 25.00 | $76.32 |
| `google/gemini-3.1-pro-preview` | 2.00 | 12.00 | $31.88 |
| `qwen/qwen3-vl-32b-instruct` | 0.104 | 0.416 | $1.52 |
| `qwen/qwen3-vl-235b-a22b-instruct` | 0.20 | 0.88 | $2.97 |
| | | **predicted total** | **$192** |

That is the worst case, where every query names the full six-cell budget; at a
more realistic three cells it is about $149. Both fit the $220 allocation.
The two Qwen rates matter least but are the least certain — check them too.

Also confirm each slug still exists and still accepts image input plus structured
output. If one is gone, tell me the replacement and I will update the roster and
the decision log. The price pilot measures real token usage, but it cannot detect
a stale *price*.

**Two models are 82 percent of the bill.** If a slug is retired and you need a
replacement, the constraints are: a different vendor from the other two, image
input, structured output, and a rate that keeps the total under $220 (anything at
or below $5/M input and $30/M output does). Send me the candidate and its
published rates and I will re-run the projection before it goes in the roster.

### Nothing else is needed

No annotation campaign, no manual registration, no model training, no hidden
evaluation server, and no GPU on Adroit.
