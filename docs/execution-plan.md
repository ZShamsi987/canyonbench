# Frozen execution plan: 31 July–20 August 2026

The software is complete before D1. This schedule begins only after the source
rasters, independent feature layers, detector scores, and licensing decisions
can be supplied. Dataset and results freezes are hard boundaries: a generator
change after the dataset freeze invalidates downstream model runs.

| Date | Freeze-stage deliverable | Required command/output |
|---|---|---|
| Jul 31 | Both resources verified; one licensed pilot site; 50-call/model price pilot | `sbatch slurm/adroit_preflight.sbatch`; `scripts/lambda/bootstrap.sh`; `trace price-pilot --calls 50`; `trace plan-run --price-pilot` |
| Aug 1 | Pilot camera bundle | `trace build --smoke`; RGB/mask alignment check |
| Aug 2 | Pilot G1–G4 and extinction diagnostics | `trace validate`; gate results |
| Aug 3 | Independent 120-site cohort | `trace select-sites` |
| Aug 4 | 960 clean + 240 singly degraded views | `trace build`; validation report; `trace fidelity-report` |
| Aug 5 | Objective two-auditor packet | `trace audit-sample` |
| Aug 6 | O1–O4 curves and balance report | generated intervention manifests |
| Aug 7 | V1, V2, V6 complete; **dataset freeze** | instrument and audit reports |
| Aug 7 | Weights pre-retrieved; dataset pushed to Lambda | `scripts/lambda/prefetch_weights.sh`; `scripts/lambda/sync_dataset.sh` |
| Aug 8 | Final roster/config/cost preflight | `trace plan-run` |
| Aug 9 | Tier A complete for every model | resumable predictions + run manifest |
| Aug 10–11 | Tier B, self-served Lambda models before paid API models | causal traces |
| Aug 12 | Prompt suite and V5 controls | Tier C/control traces |
| Aug 13 | CAVE full/component runs | frozen dev thresholds and decisions |
| Aug 14 | V3 plus separate V4 grid/K run; **results freeze** | operator/sensitivity outputs |
| Aug 15 | Metrics, group/site intervals, paired/mixed/BH analysis | `trace score` |
| Aug 16 | Per-model/per-class acuity and extinction analysis | metrics/report artifacts; `trace audit-summary --dataset-dir` band validation |
| Aug 17 | Figures and main tables | `trace report --dataset-dir`; copy artifacts flat into `paper/` |
| Aug 18 | Methods/results filled only from frozen outputs | paper draft |
| Aug 19 | Related work, discussion, limitations, coauthor review | reviewed draft |
| Aug 20 | Release validation, anonymization, archive/DOI, final consistency | `trace release`; `scripts/check.sh` |

## Hard order

1. Freeze `configs/preregistration.yaml`, source manifests, dataset config, and
   real-flight calibration before full generation.
2. Resolve every audit or instrument failure before the dataset freeze.
3. Freeze model IDs, versions, providers, roles, prices, prompts, and request
   caps before inference. Authorize the paid run only from the measured price
   pilot, never from the registered token estimate; see
   [budget-and-risk.md](budget-and-risk.md).
4. Finish Tier A for all models before any Tier B/C claims.
5. Tune CAVE only on development; apply the frozen thresholds unchanged.
6. Inspect reserved test results once. Never use test outcomes to revise gates,
   prompts, operators, thresholds, or exclusions.

## What only the project lead must supply or authorize

- Keep paid API credentials private and set `OPENROUTER_API_KEY` only in the
  execution environment; never commit it.
- Launch the Lambda GPU instance (an 80 GB card serves the whole roster in one
  session) and provide SSH access; terminate it once results are fetched.
- Provide/authorize the EarthDial wrapper and the independent detector service,
  or approve their preregistered omission if they cannot be stood up.
- Report the Adroit wall-time limit from the preflight job so the long
  acquisition and API jobs can be re-chunked if the limit is under 12 hours.
- Start the registered model calls only after reviewing the cost preflight.
- Have Kunsh and Atharva complete the frozen objective audit independently and
  approve the documented resolution of any disagreements.
- Approve the final public release, archive/DOI deposit, and paper submission.

The source policy, 230-site candidate manifest, redistribution-compatible
public sources, real-flight quality calibration, model roster, prices, prompts,
and auditor identities are already prepared.

No model training is part of the critical path. CanyonBench-Trace is a
black-box evaluation benchmark; open-weight models are served as-is through
vLLM on Lambda in bfloat16, proprietary models are reached over the OpenRouter
API from an Adroit CPU node, and CAVE is training-free. The command sequence for
both hosts is in [../RUNBOOK.md](../RUNBOOK.md); the hardware contract is in
[compute-and-storage.md](compute-and-storage.md).
