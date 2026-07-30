# Evaluation protocol

Freeze `configs/preregistration.yaml`, the dataset manifest, prompt file, model
roster, provider prices, and code state before paid calls.

## Tier A — screening

Run one neutral structured query on all 960 clean views and every model. Report:

- balanced accuracy, FPR, FNR, abstention, macro F1, and per-class results;
- format failure rate;
- cell precision/recall/F1, mask-weighted overlap, evidence coverage, cell
  count, and area-penalized F1;
- extinction positive rate;
- P(detect)=0.5 apparent-width threshold from a psychometric curve;
- risk–coverage curves.

Tier A runs first and is never removed.

## Tier B — causal core

Use the frozen stratified 160-view subset. The runner consumes all accepted O1,
O2, and O3 oracle and distractor curves, then constructs model-specific
self-deletion, sufficiency, random-cell, and texture-cell edits from Tier A
rankings. O4 remains secondary until V2 passes.

- `OCRS = AUC(distractor) - AUC(target)`
- `SEN = AUC(random deletion) - AUC(self deletion)`
- `SES` is mean normalized recovery of the unedited Tier-A response probability
  when only claimed evidence remains clear
- `EFS = geometric_mean(max(0,SEN), max(0,SES))`
- `OSG = OCRS - EFS`

Curves use all registered steps and trapezoidal AUC, not a cherry-picked
deletion fraction.

## Tier C — prompts and CAVE

Use the frozen stratified 120-view subset. Run neutral, false-premise,
uncertainty-aware, evidence-first, and genuine no-image controls. Prompt prior
gap is false-premise yes rate minus neutral yes rate on negative examples.

CAVE:

1. obtains the initial answer and claimed cells;
2. suppresses those cells for necessity;
3. preserves them for sufficiency;
4. edits matched unselected cells for nuisance sensitivity;
5. retains a positive only if all development-tuned thresholds pass.

Report false-positive reduction, abstention increase, coverage, reliability, and
inference cost separately for neutral and false-premise prompts and in
aggregate. The operational policy stops after the first failed confidence,
necessity, or sufficiency check; accepted positives use four calls. Export the
non-dominated development reliability–coverage–cost frontier, freeze the
registered maximum-balanced-accuracy/coverage/minimum-cost choice, and never
re-tune on the reserved test split. Thresholds and component ablations are
frozen after development.

## Instrument validation

- **V1:** photometrically matched synthetic feature inserts spanning apparent
  widths through the extinction band.
- **V2:** group-cross-validated edit classifier for untouched, target-edited,
  and distractor-edited images.
- **V3:** Spearman agreement of model rankings across O1/O2/O3.
- **V4:** all grid sizes 4/6/8 and K values 3/6/10.
- **V5:** always yes/no, base rate, no image, blank, shuffled, unrelated, and
  coordinate-prior baselines.
- **V6:** independent-detector signal before and after suppression.

A null target effect is interpretable only beside V1 and V6.

## Statistical reporting

The site is the unit of independence. Report raw queries and effective sites.
Bootstrap groups and sites. Pair clean/degraded, target/distractor,
neutral/false-premise, direct/CAVE, and low/high altitude within site. Include
the registered mixed-effects model. Correct only the planned primary comparison
family with Benjamini–Hochberg; label other analyses secondary/exploratory.

## Required ablations

Run every item in `configs/preregistration.yaml`: source consensus, date gate,
road width, exclusion detector, O1/O2/O3/O4, target/random/texture, grid, K,
CAVE components, geographic groups, and clean/degraded conditions. Report
operator disagreement or sensitivity reversals as findings.

The gate-side variants come from `trace gate-ablations`. The operator variants
come from `trace score`: `causal_by_operator` carries O1–O3 and feeds V3 rank
agreement, while `causal_secondary_inpaint` carries O4 separately and never
enters the primary causal result. Add `analyses: [inpainting]` to the run config
to collect the O4 traces in the first place.
