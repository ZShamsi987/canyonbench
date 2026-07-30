# CanyonBench-Trace paper scaffold

The scaffold follows v4: procedural benchmark construction, extinction-scale
measurement, causal evidence tracing, instrument validation, and the
training-free CAVE wrapper.

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
