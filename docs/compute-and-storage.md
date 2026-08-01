# Compute and storage

CPU-bound dataset construction and GPU-bound model inference run on different
resources. The split removes any dependence on GPU queue availability during
dataset generation and confines credited GPU time to the work that actually
needs VRAM.

## Division of work

| Workload | Resource | Why |
|---|---|---|
| Source tile acquisition (NAIP, 3DEP, NHD, TIGER, OSM, CDL) | Adroit CPU | Network- and I/O-bound; no GPU utility |
| Virtual-camera generation, mask rasterization, homography warping | Adroit CPU | GDAL/rasterio/OpenCV parallelize across cores |
| Gates G1–G4, resolvability, `derived.json` | Adroit CPU | Vector geometry and raster statistics |
| Intervention rendering (O1–O4, matched distractors) | Adroit CPU | Image processing; optionally GPU-accelerated, never GPU-dependent |
| Open-weight and RS-specialized VLM inference | Lambda GPU | The only workload needing substantial VRAM; vLLM with continuous batching |
| Proprietary VLM inference | OpenRouter API | No local compute; driven from an Adroit CPU node |
| Edit-detector training, synthetic inserts | Lambda GPU (incidental) | Minor; scheduled inside an existing inference session |
| Metrics, bootstrap, mixed effects, figures | Adroit CPU | Statistical computation |

## Precision constraint

Every reported model is served in **bfloat16**. Quantization changes model
behavior, and a benchmark whose subject is causal faithfulness cannot report a
quantized model under the named model's identity. Instances without native
bfloat16 are excluded regardless of nominal VRAM, which is why the 8× V100
configuration is not usable here despite having 128 GB in aggregate.

`trace compute-check --role lambda` fails the run if any visible device reports a
CUDA capability below 8.0.

## Minimum requirements by model class

| Model class | Weights (bf16) | Minimum VRAM | Recommended | System RAM |
|---|---|---|---|---|
| 7–8B VLM | ~16 GB | 24 GB | 40 GB | 64 GB |
| 12–14B VLM | ~28 GB | 40 GB | 80 GB | 64 GB |
| 26–34B VLM | ~52–68 GB | 80 GB | 80 GB | 128 GB |
| 70B+ VLM (not in the roster) | ~140 GB | 2×80 GB | 2×80 GB | 256 GB |
| Detector or segmenter | <8 GB | 16 GB | 24 GB | 32 GB |

The table is encoded in `canyonbench.compute.MODEL_CLASSES`, so the assessment
below is computed rather than transcribed.

## Instance selection

Selection is driven by VRAM per GPU, not aggregate GPU count: the workload is
request-parallel, so throughput scales through batching on one device, and extra
devices help only when a single model exceeds one card.

**The region decides the card.** A Lambda filesystem mounts only in its own
region, the project filesystem `CanyonBench` lives in **us-east-1**, and Lambda
offers no H100 there at all. Verified against the live API:

| Configuration | Rate | In us-east-1 | Verdict |
|---|---|---|---|
| 1× A100 40 GB SXM4 | **$1.99/hr** | **yes** | **Registered selection** — the only single-GPU card in region, and the cheapest anywhere |
| 1× H100 80 GB PCIe | $3.29/hr | no (us-west-3) | Unavailable in region |
| 1× H100 80 GB SXM5 | $4.29/hr | no (us-south-2/3, us-southeast-1) | Unavailable in region |
| 1× A10 24 GB PCIe | — | no | Fallback for 7–8B and the detector only |
| 8× A100 80 GB SXM4 | $22.32/hr | yes | Not recommended: pays for eight devices to use one |
| 8× Tesla V100 16 GB | — | — | **Excluded** — Volta lacks native bfloat16 |

**Launch `gpu_1x_a100_sxm4` in us-east-1 with the `CanyonBench` filesystem
attached.** It holds the 8B model (~16 GB), EarthDial (~8 GB), and the detector
(<8 GB) with KV-cache headroom; at 40 GB the profile is `max_model_len` 16384,
`max_num_seqs` 64.

Two Qwen sizes exceed a 40 GB card in bfloat16 — the 32B needs ~68 GB and the
235B roughly 470 GB — so both are reached over OpenRouter instead. Together they
add about $4.50 to the bill against $1.30/hr saved versus an H100, and the three
open-weight size levels stay intact.

## Credit position

The $400 of Lambda credit is split before anything is launched: **$40 reserved
for persistent storage** (~200 GB, billed per GB-month, and the filesystem must
survive to the end of the project) and **$340 for GPU time**.

| Card | Rate | Hours $340 buys | Projected hours | Headroom |
|---|---|---|---|---|
| **A100 40 GB SXM4 (registered)** | **$1.99/hr** | **~171 h** | 15 h | **~11×** |
| H100 80 GB PCIe (out of region) | $3.29/hr | ~103 h | 15 h | ~6.9× |

Projected GPU consumption is approximately 15 GPU-hours: 2 hours of setup and
weight retrieval, 6 hours for the roster across Tiers A–C, 2 hours of robustness
re-runs, and 5 hours of defect-recovery buffer. At the registered rate that is
about **$30 of the $340**, so the binding constraint is engineering time, not
compute.
The split is encoded in `canyonbench.compute.gpu_budget`.

The surplus is large enough to extend Tier B causal tracing from the 160-view
subset to the full 960-view lattice for the self-served models at no cash cost —
that is the first upgrade to spend it on, ahead of anything that costs money.

## Endpoints: what they are and where they come from

**There is no third-party service to sign up for.** An "endpoint" here is just
the URL of a server process running on your own Lambda machine. Every model in
the roster is reached over HTTP, and each one is either:

| Model | Server | Who starts it | URL |
|---|---|---|---|
| 3 proprietary + Qwen 32B + Qwen 235B | OpenRouter | nobody — it is a public API | `https://openrouter.ai/api/v1` |
| Qwen 8B | vLLM, on your Lambda box | `run_open_weight.sh` | `http://127.0.0.1:8000/v1` |
| EarthDial | vLLM, on your Lambda box | `run_open_weight.sh` | `http://127.0.0.1:8001/v1` |
| Detector reference | `serve_detector.py`, on your Lambda box | you, one command | `http://127.0.0.1:8010` |

**In the normal case you give me nothing.** Those URLs are already in
`configs/trace_run.frozen.yaml`, the scripts start the servers on exactly those
ports, and `127.0.0.1` is correct because the driver runs on the same machine.

You only need to tell me an address if you start something somewhere else — a
different port, or another host. Then export one variable instead of editing the
roster:

```bash
export CANYONBENCH_ENDPOINT__CANYONBENCH_INDEPENDENT_DETECTOR_V1=http://127.0.0.1:9100
```

The variable name is `CANYONBENCH_ENDPOINT__` followed by the model id
upper-cased with every non-alphanumeric character replaced by an underscore. Any
model can be redirected this way, and `trace run` reports which overrides it
applied.

### The non-language detector

`scripts/lambda/serve_detector.py` is a complete implementation of the Section 12
upper reference: a SegFormer semantic segmenter whose ADE20K labels are mapped
onto water, road, and field, answering the same structured schema every model
answers. It needs only the serving stack vLLM already installs, fits in <8 GB
beside the VLM session, and resolves its label mapping against the checkpoint's
own `id2label` at startup so a class can never be silently mis-mapped.

```bash
python scripts/lambda/serve_detector.py --port 8010 &
curl -s localhost:8010/health     # {"status": "ok", "labels": {...}}
```

### EarthDial

`run_open_weight.sh` starts it on port 8001 with plain vLLM. If it fails the
health check, EarthDial needs its own schema-enforcing wrapper: start that by
hand on any port, export the variable above, and rerun with
`--only-model akshaydudhane/EarthDial_4B_RGB`.

If either service cannot be stood up, the roster validator accepts 2–3
open-weight and 1–2 remote-sensing models, so the registered fallback is to drop
that entry and record the omission.

## Storage

All persistent state lives on the Lambda filesystem, so the instance type may
change between sessions without data migration.

```text
/lambda/canyonbench/
  hf/          HF_HOME: model weights, persisted across sessions
  dataset/     frozen scene bundles, uploaded once from Adroit
  results/     append-only predictions.jsonl
  logs/        per-session stdout, cost accounting, job state
  CanyonBench/ the project checkout
```

```bash
export HF_HOME=/lambda/canyonbench/hf
export HF_HUB_ENABLE_HF_TRANSFER=1
```

| Item | Size | Where |
|---|---|---|
| Source tiles for 120 sites | 100–300 GB | **Adroit only**, never transferred |
| Frozen dataset bundle (views, masks, interventions) | 20–40 GB | Uploaded to Lambda once |
| Model weights for the served roster | 100–150 GB | Lambda `hf/` |
| Results and logs | negligible | Lambda `results/`, `logs/` |
| **Lambda total** | **~200 GB** | |

## Session minimization

Four measures keep the number of GPU sessions small and cap the cost of an
interrupted session at minutes rather than hours.

- **C1 weight pre-retrieval** — `scripts/lambda/prefetch_weights.sh` downloads
  every served model to the persistent filesystem before any inference starts, so
  no GPU time is spent waiting on network transfer and later sessions incur no
  download cost.
- **C2 content-hash resumability** — every result is keyed by a hash of the model,
  view, prompt, intervention, and schema, and appended to a JSONL log. On restart
  the driver skips completed keys, so an interrupted session resumes at the point
  of failure.
- **C3 sequential model execution** — `scripts/lambda/run_open_weight.sh` loads and
  releases each model in turn inside one session, amortizing instance startup and
  filesystem mount across the whole roster.
- **C4 capability-adaptive configuration** — the driver reads the detected VRAM and
  selects a serving profile, so the same script runs unmodified on any admissible
  instance:

| Detected VRAM | `max_model_len` | `max_num_seqs` |
|---|---|---|
| < 30 GB | 8192 | 16 |
| 30–60 GB | 16384 | 64 |
| ≥ 60 GB | 32768 | 128 |

Inspect the selection with `canyonbench trace vllm-profile` (add
`--vram-gb 40` to preview a profile from a machine without a GPU).

## Verification before committing to either resource

Both checks together take about twenty minutes and eliminate the most disruptive
class of mid-project failure.

```bash
# Adroit: allocation live, partitions and wall time recorded, environment resolves
sbatch slurm/adroit_preflight.sbatch

# Lambda: filesystem mounted, GPU visible, bfloat16 native, vLLM importable
bash scripts/lambda/bootstrap.sh
```
