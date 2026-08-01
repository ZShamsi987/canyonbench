# Changelog

## Unreleased

- Redirect the primary benchmark to CanyonBench-Trace v4: exact procedural
  virtual-camera projection over orthoimagery and three consensus feature masks.
- Add strict source, site, camera, gate, derived-view, intervention, response,
  audit, CAVE, and preregistration schemas.
- Add 120-site quota/split enforcement, G1–G4, exact RGB/mask/depth rendering,
  a 960-view lattice, and 240 single-degradation counterparts.
- Add O1–O4 interventions, matched distractors, balance/artifact diagnostics,
  model-self deletion/sufficiency, and random/texture controls.
- Add strict Tier A/B/C execution, text-only controls, non-language detector
  adapter, CAVE, registered v4 metrics/statistics, V1/V2/V3/V6 instruments, and
  public/escrow release construction.
- Add V4 grid/K/operator sensitivity, V5 deterministic and image controls,
  V6 detector suppression, dual-context CAVE, component ablations, and a
  development-only reliability/coverage/cost frontier with sequential cost.
- Add bounded source-window rendering, content-hash inference caching,
  site-safe resume keys, exact target-pixel localization, extinction gating,
  complete registered statistics/figures, and self-verifying release hashes.
- Freeze the cross-platform environment in `uv.lock` and validate wheel/sdist,
  schema examples, the paper build, and the full synthetic end-to-end path.
- Replace primary human annotation with a two-auditor, binary-only 5–10% audit.
- Retain the original balloon ingestion/annotation/registration implementation
  only as legacy calibration and optional real-flight analysis support.

- Accept World View-prefixed telemetry fields found in the real WORLD10 log.
- Add an optional reset-session audit CSV to make operational-flight selection reviewable.
- Measure clip duration from the final decodable frame in preallocated AVIs, and add explicit filename and relative-time ordering policies for cameras with invalid wall clocks.
- Add resumable per-clip extraction and hard-link frame materialization for low-disk, cloud-streamed ingestion.
- Add opt-in batched macOS File Provider cache eviction for oversized cloud-backed sources.
- Capture resumable per-clip source SHA-256 values during extraction without a second cloud download.
- Add bounded concurrent clip probing while preserving deterministic manifest order.
- Add an explicit unmatched-telemetry exclusion mode with a per-frame audit CSV.
- Add an explicit undecodable-clip exclusion mode with a separate audit CSV.
- Preserve verified relative camera-clock gaps with an explicit clip-end mtime timeline policy.
- Invalidate resumable extraction markers when the timeline or extraction contract changes.
- Bound trajectory-segment duration and refine segments at geographic split boundaries so long phases cannot collapse the dataset into one split.
- Mark constant-zero speed and vertical-velocity channels unavailable when the recovered GPS trajectory clearly moves.
- Quantify the sim-to-real relief-displacement gap (`d = r*dh/H`) per view, aggregate it with `trace fidelity-report`, and add an optional off-by-default DEM relief injection that shares one sampling grid with every mask.
- Project paid inference in dollars during preflight and add `trace price-pilot`, which measures real per-call tokens under a separate cap and re-prices the whole plan before production spend.
- Join the objective audit's `feature_resolvable` votes onto derived flags, closing the fourth extinction-band criterion with a per-class human confirmation and contradiction rate.
- Add the registered GSD/apparent-width extinction figure and geometric ladder table, and wire all six paper floats plus the main table to build products that degrade to a visible TBD when absent.
- Report O4 inpainting as a separate registered causal ablation instead of dropping it from scoring.
- Fix view manifests failing their own strict re-validation, which had caused every per-view hash, mask, gate, case-type, and intervention check to be skipped.
- Gate resolvability on interior reach rather than closest approach to the frame border, so features that cross the footprint are no longer rejected as unresolvable.
- Document the budget, cost controls, kill criteria, descope ladder, and risk register, and cite HalluSegBench alongside the nearest counterfactual-grounding work.
- Adopt the v4.2 two-resource split: Adroit CPU SLURM jobs for acquisition, generation, gates, interventions, API inference, and analysis; Lambda GPU scripts for the self-served roster.
- Encode the registered VRAM, instance, precision, and storage contract in `canyonbench.compute`, with `trace compute-check` as a per-host go/no-go preflight and `trace vllm-profile` as the capability-adaptive serving profile.
- Serve every reported model in bfloat16 and refuse hosts without native support; quantization is inadmissible for a faithfulness benchmark.
- Add weight pre-retrieval, sequential single-session serving, and `trace run --only-model/--dataset-dir/--output-dir` so one frozen roster drives both hosts and an interrupted GPU session resumes at the exact request.
- Add `trace merge-runs`, which joins the per-host prediction logs, refuses conflicting duplicates or mismatched dataset bundles, and carries the merged run manifest so benchmark roles survive into scoring.
- Move the largest open-weight model to the API because its bfloat16 weights exceed the registered single-card plan, and lower the cash cap to the $220 v4.2 allocation.
- Add RUNBOOK.md: the exact execution order across both hosts, with every blocking step and required human output marked.
- Standardize on the H100 80 GB PCIe at $3.29/hr, encode the hourly rates and the $40 storage / $340 GPU credit split, and record that the projection uses about a seventh of the GPU allocation.
- Ship the non-language detector reference as a runnable SegFormer service whose class mapping resolves against the checkpoint's own labels, so a mis-mapped class fails at startup rather than silently.
- Add `CANYONBENCH_ENDPOINT__<MODEL>` overrides so a hand-started service is pointed at without editing the frozen roster.
- Chunk source acquisition into resumable array tasks under four hours with a dependent cohort-freeze job, so the pipeline fits any Adroit wall-clock limit, and add `trace merge-sites`.
- Add docs/AUDITOR-GUIDE.md: standalone instructions for the two objective auditors, who work before any model runs.
- Reframe the paper for VLM4RWD: grounding, faithfulness, hallucination mitigation, and deployment reliability, with a compute and reproducibility appendix.
- Restrict Tier B to the primary operator for metered models: three operators by six cells by four sequences was 72 paid calls per view against a registered ~16, projecting $407 against a $220 cap. Self-served models keep all three operators, and V3/V4 sweeps run only where compute is credited.
- Update the 235B rate to $0.20/$0.88 per million and record the predicted OpenRouter spend, $176 worst case and about $133 at three named cells.
- Confine Adroit to two directories, refuse to operate on a checkout that is not this project, and never edit a shell profile.
- Assert the clean-view lattice per site and in total, and record the 120 x 4 x 2 = 960 arithmetic in the dataset manifest.
- Add a 40-view operator-agreement subset so metered models are also measured under O2 and O3: V3 correlates model rankings, and restricting paid models to O1 everywhere would have left that correlation over three credited models. Predicted spend $191 worst case, $148 at three named cells.

## 0.1.0 - 2026-07-21

- Initial complete pre-data implementation of the pipeline, registration, ground-truth, inference, and evaluation toolkit.
