# Audit assignment — Kunsh and Atharva

This is the what and when. [AUDITOR-GUIDE.md](AUDITOR-GUIDE.md) is the how, and
you should read it once before starting — it explains each of the four questions
with examples of the good and bad answer.

## The job in one paragraph

You will each get a CSV of about **96 rows** and a folder of review sheets, one
image per row. For each row you answer **four yes/no questions** by looking at
the image. You are not labelling anything, not drawing boundaries, and not
judging any model. There is no model output anywhere in this task.

**Time: 3–5 hours each**, roughly 2–3 minutes per view. You need no cluster
access, no software, and no setup — a CSV editor and an image viewer.

## Why this blocks the whole project

Nothing that costs money runs until both CSVs come back. The audit is the gate
between the built dataset and every model run, and it is there for a reason: if
the audit finds problems, sites get regenerated or dropped, and that cannot
happen after models have already been run against them.

It is also the only remaining step that runs on human time rather than compute.
Everything before it is automated and everything after it is queued behind you.

## The one rule that matters

**Work alone. Do not compare answers with each other until you have both
submitted.**

Two independent judgements are the entire point — the agreement rate between you
is a number reported in the paper. Comparing notes first destroys it. If a view
is genuinely ambiguous, write that in the `notes` column and move on.

## What you will receive

```
audit.csv                                  <- the file you fill in
audit.csv_assets/
  site_0041__view_c16km_oblique.png        <- one review sheet per row
  site_0057__view_a3km_nadir.png
  ...
```

Your auditor ID is already filled in on every row. You fill four columns with
`yes` or `no` and leave everything else alone.

## The four questions, in brief

| Column | What you are judging |
|---|---|
| `overlay_aligned` | Does the red overlay sit on the feature you can see, or is it plainly offset? |
| `feature_resolvable` | Can you see the feature at all, at 100% zoom, in the unmarked panel? |
| `obvious_edit_artifact` | Do the edited panels look pasted — a hard seam or rectangle — rather than smudged? |
| `source_mismatch` | Does the image plainly refute the map? |

Two of these check the pipeline. **`feature_resolvable` is different: it is
data, and both answers are equally correct.** The benchmark's central claim is
that some features become physically unresolvable from high altitude, and your
`no` is the human confirmation of that. A confident `no` on a 24 km view is a
correct and valuable answer, not a failure. Do not feel pressure toward `yes`.

Open each sheet at **100% zoom and no further**. Whether something is visible at
the image's own resolution is exactly what the question asks.

## When you finish

1. Save the CSV under the **same filename** with the **same columns** — do not
   reorder, rename, add, or delete any.
2. Send it back. Not to each other.
3. Once both arrive, disagreements are resolved by discussion and the agreement
   rate goes in the paper.

Never leave a cell blank. If you are stuck on one view, put a short note in
`notes` and answer with your best judgement.

## Timing

The submission deadline is **30 August 2026** (VLM4RWD, NeurIPS 2026). The
packet is generated automatically at the end of the dataset build; you will be
sent the CSV and the assets folder as soon as it exists.

Turning your CSVs around within a day of receiving them keeps the model runs,
analysis, and writing on schedule. If you hit a problem with the files
themselves — a missing sheet, a corrupt image, a row you cannot open — say so
immediately rather than working around it, because that is a defect in the
packet and it is faster to regenerate than to patch.
