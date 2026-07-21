# Evaluation protocol

Freeze the frame release, probes, model configuration, and code commit before a reported run. The run manifest hashes the frames table and probe file. Resume keys include model, image, probe, variant, and prompt text, making interrupted runs idempotent.

Presence is exact structured classification over six fixed features. Human `uncertain` targets are excluded feature by feature and disclosed. Invalid model outputs count against exact accuracy and contribute to the invalid rate; FPR and F1 are computed only where a binary prediction exists.

Vegetation is predicted percent minus human-mask percent. Report signed error, MAE, and the proportion of valid estimates above truth. Grounding uses region precision/recall/F1 over 16 cells and only registration-reliable frames. False-premise compliance is `premise_correct=true`; the mitigation delta is leading compliance minus evidence-first compliance.

Free captions are secondary. The judge sees caption text and caption id only, never the image or evaluated model. Validate it against two independent annotators, report precision, recall, and agreement, and retain the fixed rule-based fallback.

Bootstrap confidence intervals sample trajectory segments with replacement. Never bootstrap individual frames. Calibration mechanisms—verbalized confidence, exposed token probability, and self-consistency—must remain separate.

The controlled ascent regression includes image-quality controls, feature prevalence, and phase, with segment-clustered covariance. Show stratified trends as a robustness check. Use “altitude-associated” or “across the ascent trajectory,” never causal language.

