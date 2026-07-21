# Architecture and invariants

CanyonBench is a staged, fail-closed data system. Each stage writes a stable artifact that can be inspected independently.

```text
WORLD10.txt -> flight.csv -------------------------------+
AVI clips -> inventory -> sync -> 1 Hz crops -> names --+-> frames candidates
                                                            -> sample/segments/splits
human masks + presence + quality ------------------------+
control points -> homography -> held-out residuals -------+-> release/frames.csv
registered overlays -> human-confirmed 4x4 grids --------+

release + probe contract -> adapter -> predictions.jsonl -> deterministic metrics
                                                     \----> caption judge (secondary only)
```

## Invariants

1. `img_SSSSSS.jpg` joins to exactly one `elapsed_s` and zero-GPS rows are absent.
2. Only Launching and Floating enter scored releases; Floating is the core set.
3. The one-Hz extraction is an intermediate. Sampling enforces 30-120 second minimum spacing, movement or perceptual change, trajectory segments, and block-level splits.
4. Human visible-image labels are primary. NLCD, NHD, roads, and VARI are context or weak labels only.
5. Homographies map frame pixels to metric reference coordinates. Two points are held out. A frame exceeding one quarter of a ground-grid-cell width is not scored for grounding.
6. A 4x4 cell is positive at 1% final-mask coverage, subject to a logged human override.
7. Structured responses are validated before scoring. Invalid responses are reported and cannot silently disappear.
8. Confidence intervals resample segments. Reports always disclose frame and segment counts.
9. Altitude models are described as associations and include quality, prevalence, and phase controls.

All public schemas carry versions. A schema or metric change that can alter a score requires a minor or major release and a changelog entry.

