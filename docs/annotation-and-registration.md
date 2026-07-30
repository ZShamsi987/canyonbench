# Objective audit (no semantic annotation)

The v4 primary benchmark does not use Label Studio, drawn masks, manual control
points, qualification tasks, a golden annotator, or adjudication. Exact masks
come from projected frozen source layers.

Human review is deliberately small and result-independent: two coauthors each
inspect the same stratified 5–10% sample (about 96 views at 10%) and independently
answer four yes/no questions:

| Field | Yes means |
|---|---|
| `overlay_aligned` | The displayed mask boundary visually follows the mapped feature. |
| `feature_resolvable` | The target can be judged at this output scale without zooming beyond native pixels. |
| `obvious_edit_artifact` | A causal edit has an obvious seam or corruption cue. |
| `source_mismatch` | The frozen map source visibly disagrees with the orthoimage. |

Notes are for objective failure details only. Auditors do not relabel the
feature, redraw a mask, or see model answers.

## Procedure

1. Generate the sample with `canyonbench trace audit-sample`.
   The command creates native-scale review sheets beside the CSV.
2. Give each auditor a separate copy or filtered view; they must not discuss
   decisions before both submit.
3. Open the clean RGB, target mask overlay, and representative O1/O2/O3 edits at
   native size.
4. Fill every binary cell with `yes` or `no`; add a short note only for a
   suspected failure.
5. Merge rows without changing either auditor's values.
6. Run `canyonbench trace audit-summary` with `--dataset-dir` so the
   `feature_resolvable` votes are joined onto the derived per-view flags.
7. Investigate systematic gate/source/operator failures before dataset freeze.
   Do not selectively delete difficult views based on model results.

Every sampled view must have exactly two records. The summary reports agreement,
prevalence, and failure votes rather than a subjective gold label.

## The audit validates the extinction band

`feature_resolvable` doubles as the fourth extinction criterion. The first three
are automatic — apparent width below the geometric threshold, local contrast below
the calibrated bound, and no signal from the exclusion-only detector — and the
audit closes the definition by confirming that humans see no trace either. The
summary reports, per class:

- the number of audited extinction views and the human confirmation rate;
- every contradicted view, where both auditors did see the feature;
- the resolvable-positive control rate, which must stay high or the audit itself
  is suspect rather than the band.

A contradicted extinction view is regenerated or excluded. It is never relabeled,
and the band's calibration is reported as a result.

## Legacy note

The older frame-annotation and homography modules remain available for an
optional real-flight temporal appendix. Their outputs must never be mixed into
the procedural primary labels or used to tune v4 test thresholds.
