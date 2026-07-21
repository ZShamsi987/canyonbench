# Private data ingestion

Keep recovered logs and videos outside Git. Back them up read-only and record checksums before parsing. `WORLD10.txt` is treated as untrusted tabular input because it contains power-cycle tests and embedded headers.

## Operational-flight recovery

The parser finds the header, canonicalizes known field aliases, detects elapsed-time resets, and considers only segments containing Launching, Floating, and Terminating in order. It chooses the longest such segment, preferring the later segment on a tie. Rows with absent or zero latitude/longitude are removed and seconds are unique.

Review the recovered phase boundaries against approximately 2742 (Launching), 6806 (Floating), and 25688 (Terminating). Differences should be investigated, not force-corrected.

## Video clock

`clips` prefers creation timestamps only if every clip has one; otherwise it orders by modification time then natural filename and records that fallback. Verify the order visually. Identify one exact event—launch or burst—in one clip and use `sync`. The resulting JSON records the original evidence and one offset.

`extract` removes the left third using `crop=2/3 width:x=1/3`, samples one frame per second, and preserves per-clip intermediate names. `name-frames` converts those into flight-second keys and writes a naming manifest. Inspect random frames around clip boundaries and the anchor before continuing.

## Sampling

Compute pHash on the cropped image. A frame cannot pass inside the configured minimum interval. Outside it, sufficient ground movement or perceptual change admits the frame. Segment breaks occur on phase changes, large time gaps, or implausible geographic jumps. Spatial blocks—not adjacent images—are assigned to splits, then whole segments are kept together.

The default values are starting points from the specification. Freeze final values before model evaluation and report them.

