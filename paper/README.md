# CanyonBench-Trace paper scaffold

Target venue: **2nd Workshop on Grounded and Faithful Vision-Language Models for
Real-World Deployment (VLM4RWD), NeurIPS 2026** — 8 pages excluding references
and appendices, NeurIPS 2026 format, double-blind, submitted via OpenReview as a
dataset/benchmark paper.

The scaffold follows spec v4.2: procedural benchmark construction,
extinction-scale measurement, causal evidence tracing, instrument validation, and
the training-free CAVE wrapper. It is framed against the workshop's themes —
visual grounding and evidence alignment, faithful reasoning, hallucination
mitigation, counterfactual and causal evaluation, and deployment reliability
under uncertainty.

`main.tex` currently stands in the NeurIPS layout with `geometry`. Drop
`neurips_2026.sty` beside it and switch the documentclass lines (see the comment
at the top of the file); nothing else changes.

Keep it anonymous: no author names, no institution, no cluster names, and no
repository URL anywhere in `paper/`.

It deliberately contains no fabricated model names, counts beyond the
preregistered design, results, p-values, citations, or audit outcomes. Replace a
`\TBD{}` only from a frozen artifact and cite its result manifest/hash in a
nearby source comment.

## Figures and the main table are build products

The six registered floats and the main table are never hand authored. Generate
them from the frozen run and copy the outputs **flat into this directory**, beside
`main.tex` (some engines refuse to resolve graphics from subdirectories):

```bash
canyonbench trace report <metrics.json> <metrics.rows.csv> out/report \
  --dataset-dir <dataset>          # adds the GSD/extinction instrument figure
cp out/report/{gsd_extinction,intervention_traces,accuracy_vs_efs,acuity_curves,\
extinction_by_altitude,cave_frontier}.pdf paper/
cp out/report/model_summary.tex paper/
```

`construction.pdf` (Figure 1, the pipeline schematic) is the only float without a
generator; draw it once and drop it in.

Any float whose artifact is absent renders as a red `TBD` rather than breaking the
build or silently appearing empty, so the draft always compiles from a clean
checkout. Do not commit the generated artifacts.

Primary claims remain reportable under every outcome:

1. the benchmark generator and extinction construct are valid;
2. target-versus-distractor reliance is measured by OCRS;
3. self-reported evidence faithfulness is measured by SEN/SES/EFS;
4. CAVE trades reliability for coverage/cost.

If an instrument fails, report that failure and restrict the corresponding
claim. Do not choose operators, K, grids, prompts, or exclusions based on which
version produces the preferred conclusion.
