# Budget, cost controls, kill criteria, and risk register

GPU compute and all persistent storage are covered by $400 of Lambda credit, and
CPU compute by the Adroit allocation, so the cash budget addresses proprietary
API inference alone. Coauthors perform the audits, so there is no annotation
line. The hardware split itself is documented in
[compute-and-storage.md](compute-and-storage.md).

## Call volume and predicted spend

Only API-reached models cost cash. Qwen Max and the 32B/235B Qwen VL models are
API-reached because they are hosted services or do not fit the 40 GB card
available in the filesystem's region; see
[compute-and-storage.md](compute-and-storage.md). Self-served models run on
credited GPU capacity, so they take the full three-operator Tier B; **metered models take the
primary operator only**, because three operators x six named cells x four
sequences is 72 paid calls per Tier-B view against the ~16 calls/query this
budget is registered at. O2/O3 evidence for the metered models comes from the
robustness re-run, and V3/V4 sweeps run on the credited models
(`sensitivity_on_metered_models: false`).

Calls per paid model, from `trace plan-run` against the full 960/240 lattice:

| Line | Calls |
|---|---|
| Tier A screening (960 clean) | 960 |
| Tier A degraded robustness | 80 |
| Tier B oracle + distractor (O1, 4 fractions x 2 sequences x 160 views) | 1,280 |
| V3 operator agreement (O2+O3 oracle+distractor on 40 views) | 640 |
| Tier B self-evidence + controls (4 sequences x K x 160 views, worst case K=6) | 3,840 |
| Tier C prompts and image controls (120 x 8) | 960 |
| Tier C CAVE stages (120 x 6) | 720 |
| **Per paid model** | **8,480** |

### Predicted OpenRouter spend

At 1,400 input and 80 output tokens per call:

| Model | $/M in | $/M out | $/call | Worst case (K=6) | Likely (K=3) |
|---|---|---|---|---|---|
| `openai/gpt-5.6-sol` | 5.00 | 30.00 | $0.0094 | $79.71 | $61.66 |
| `anthropic/claude-opus-5` | 5.00 | 25.00 | $0.0090 | $76.32 | $59.04 |
| `google/gemini-3.1-pro-preview` | 2.00 | 12.00 | $0.0038 | $31.88 | $24.67 |
| `qwen/qwen3.8-max` | 2.00 | 6.00 | $0.0033 | $27.81 | $21.52 |
| `qwen/qwen3-vl-32b-instruct` | 0.104 | 0.416 | $0.0002 | $1.52 | $1.17 |
| `qwen/qwen3-vl-235b-a22b-instruct` | 0.20 | 0.88 | $0.0004 | $2.97 | $2.30 |
| **Total** | | | | **$220.22** | **$170.36** |

The spread is driven entirely by how many evidence cells models actually name:
K=6 is the cap, K=3 a plausible average. The registered worst case leaves
$29.78 under the approved $250 cash allocation. The 50-call/model price pilot
still must authorize the production run; the harness aborts on the first
request that would breach the cap.

The V3 line exists because operator agreement correlates model *rankings* across
operators. Restricting metered models to O1 everywhere would leave that
correlation computed over three credited models, which is too few points to
support a required protocol element; 40 views of O2/O3 restores it to the full
roster for about $16.62.

**Two models carry 71 percent of the cost.** If the measured projection needs
to come down, in order of preference: drop one of the two $5/M models (descope
ladder step 6, saving about $76–80), or halve `causal_core_views` from 160 to 80 for metered
models only (saving ~$50 and widening the Tier B confidence intervals). Do not
reduce Tier A.

## Cost model

At a 768 px long edge a view costs roughly 1,400 input tokens plus about 80
output tokens. At mid-tier pricing near $1.50/M input and $6/M output that is
about $0.0026 per call (~$43 for the full volume); at premium pricing near $3/M
and $15/M it is about $0.0054 per call (~$90). `configs/trace_run.frozen.yaml`
carries the per-model rates frozen on 2026-08-06; Qwen Max uses the supplied
$2/M input and $6/M output rate pending the measured price pilot.

## Allocation

| Item | Cost | Covered by | Purpose |
|---|---|---|---|
| OpenRouter API plan | $220.22 worst case | Cash | All six API models across Tiers A–C plus robustness re-runs |
| Remaining headroom | $29.78 | Cash | The measured provider-token projection must still pass the price pilot |
| GPU compute | $0 | Lambda credit ($400) | 1x A100 40 GB in us-east-1 at $1.99/hr; ~15 GPU-hours ≈ $30 of the $340 |
| Persistent storage | $0 | Lambda credit ($400) | Weights, frozen dataset bundle, results and logs; ~200 GB |
| CPU compute and source tiles | $0 | Adroit allocation | Acquisition, generation, gates, interventions, analysis; tiles never leave Adroit |
| Tooling, hosting, DOI | $0 | Open access | vLLM, OpenCV, rasterio, GDAL, Hugging Face, Zenodo |
| **Cash total** | **$250** | | At the approved upper bound of the $200–250 target; $50 of headroom against the $300 cap |

`configs/trace_run.frozen.yaml` sets `budget.max_cost_usd: 250.0`, so the harness
aborts on the first request that would breach the cash allocation.

**Credit position.** Roughly 15 GPU-hours and ~200 GB against $400 of Lambda
credit leaves a large surplus. Scope expansions that consume only compute or disk
therefore carry no cash cost and are limited solely by the schedule — most
notably extending Tier B causal tracing from the 160-view subset to the full
960-view lattice for the self-served models.

## Cost controls, all enforced in the harness

- Every view is downscaled to `image_max_side` before encoding, and the token
  counts the provider actually returns are logged per prediction.
- `BudgetTracker` reserves each metered request against `max_requests` and prices
  each response against `max_cost_usd`, aborting on the first breach. A resumed
  run re-counts prior attempts and cost before issuing a new call.
- Responses are cached by content hash — model, source-image SHA-256, exact
  system/user prompt, temperature, token/image limits, and response schema — so
  an identical view is never paid for twice. Cache hits are recorded as
  zero-attempt, zero-cost predictions.
- `trace price-pilot --calls 50` runs the D1 pilot against real endpoints under a
  separate cap (5 percent of the production cost cap), records the observed mean
  tokens per call per model, and re-prices the whole plan from the measurement.
  Provider image-token accounting varies substantially, and this is the single
  most likely source of a budget surprise, so the pilot decides authorization:
  it exits non-zero when the measured projection does not fit the cap. Feed the
  result forward with `trace plan-run --price-pilot .../price_pilot.json`.
- If the bill runs low, spend the surplus on more sites (toward 240), never on
  more prompts. Because storage and GPU compute are credited rather than
  purchased, expansions that consume only those carry no cash cost.

## Kill criteria

Set before inference and honored without renegotiation. The machine-readable
copy is `configs/preregistration.yaml`.

| Trigger | Pre-committed response |
|---|---|
| More than 10 percent of audited sites show map-image contradiction | Narrow to two classes, tighten consensus and date gates, discard the weakest region, and report the audit rate |
| Rendered views obviously unlike the real footage | Recalibrate degradation and pose from the video; narrow the claim to controlled aerial evaluation; report the sim-to-real gap quantitatively |
| Target edits substantially easier to detect than distractor edits | Keep transparent suppression as primary, drop all photorealistic claims, keep untouched-view results as primary, report the failed inpainting experiment |
| All models respond identically to target and distractor | Run V6 and V1 before interpreting; never reinterpret generic edit sensitivity as grounding |
| CAVE does not improve reliability at useful coverage | Report as a negative result with the frontier; do not re-tune thresholds on the test split |
| Fewer than about 75 real frames pass strict registration | Remove the temporal tier from the main paper; use the video for calibration only; do not lower registration standards to inflate n |

## Descope ladder

Cut from the bottom, in this order:

1. Optional real-flight temporal tier.
2. Cross-biome group C (keep A and B).
3. CAVE component ablations, keeping full CAVE.
4. Oblique geometry, keeping near-nadir.
5. Tier C entirely, so Claim 4 is dropped and the paper becomes benchmark plus
   measurement.
6. Reduce proprietary models from three to two.

Tier A, the validation suite, and the extinction analysis are never cut. If they
cannot be completed the project converts to an extended abstract on the
generator and the extinction construct.

## Risk register

| Risk | Severity | Mitigation |
|---|---|---|
| Novelty challenged as incremental over MedGround-Bench | High | Cite prominently; anchor novelty on self-evidence causality, black-box operation, exact procedural masks, and controlled extinction |
| Sim-to-real gap questioned | High | Quantify relief displacement per view (`trace fidelity-report`, `d = r*dh/H`), calibrate degradations from the real video, scope the claim to controlled aerial evaluation, optional temporal tier |
| Deletion metrics operator- or k-dependent | High | V3 and V4 are required protocol; report ranking stability across O1–O3 rather than a single operator |
| Null causal result | Medium | V1 and V6 disambiguate instrument failure from model insensitivity; the outcome matrix makes either direction reportable |
| 21-day window overruns | High | Two hard freezes; Tier A first; descope ladder with pre-committed cuts; extended-abstract fallback |
| API cost overrun | Medium | D1 price pilot, per-model caps, content-hash caching, $40 contingency |
| Map incompleteness or temporal mismatch | Medium | Consensus and date gates, exclusion detector as removal-only, audited contradiction rate reported |
| Structured-output failures on some models | Medium | Retry up to three times, record `format_failure` per model, report it as a capability caveat |

## The single most important execution note

Tier A screening plus the validation suite plus the extinction analysis
constitute a complete, publishable paper on their own, and they are scheduled to
finish by D9. Everything after that adds strength but is severable. Build in that
order and the deadline stops being a risk to the paper's existence and becomes
only a constraint on its scope.
