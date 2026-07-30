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

| Configuration | Verdict | Notes |
|---|---|---|
| 1× A100 40 GB SXM4 | **Primary** | Serves the 8B and the RS model with headroom; broadest availability |
| 1× H100 80 GB PCIe | **Secondary** | Required for the 32B model; preferred fallback when A100 capacity is gone |
| 1× H100 80 GB SXM5 | Substitute | Equivalent capability, higher interconnect bandwidth |
| 1× A10 24 GB PCIe | Fallback | 7–8B and detector only, with reduced batch concurrency |
| 2× H100 80 GB SXM5 | Not required | Only if a 70B-class model is added |
| 4× H100, 8× A100 (40 or 80 GB) | Not recommended | Exceeds requirements; cannot be used concurrently without restructuring |
| 8× Tesla V100 16 GB | **Excluded** | Volta lacks native bfloat16; violates the precision constraint |

The frozen roster needs **one 80 GB card** for a single-session run, because
`qwen/qwen3-vl-32b-instruct` will not fit 40 GB in bfloat16. If only a 40 GB
A100 is available, serve the 8B and EarthDial there and take the 32B in a second
short session on an H100.

`qwen/qwen3-vl-235b-a22b-instruct` needs roughly 470 GB in bfloat16, outside the
registered instance plan, so the largest open-weight size level is reached over
OpenRouter instead of self-served. Its published rates are about 2 percent of the
proprietary rates.

## Credit position

Projected GPU consumption is approximately 15 GPU-hours: 2 hours of setup and
weight retrieval, 6 hours for the roster across Tiers A–C, 2 hours of robustness
re-runs, and 5 hours of defect-recovery buffer. Against $400 of Lambda credit the
surplus is large, so the binding constraint is engineering time, not compute.

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
