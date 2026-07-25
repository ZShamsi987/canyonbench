# Private data ingestion

Keep recovered logs and videos outside Git. Back them up read-only and record checksums before parsing. `WORLD10.txt` is treated as untrusted tabular input because it contains power-cycle tests and embedded headers.

## Operational-flight recovery

The parser finds the header, canonicalizes known field aliases, detects elapsed-time resets, and considers only segments containing Launching, Floating, and Terminating in order. It chooses the longest such segment, preferring the later segment on a tie. Rows with absent or zero latitude/longitude are removed and seconds are unique. Pass `--audit-output work/flight-segments.csv` to preserve a compact record of every excluded reset session and the one selected operational segment.

Review the recovered phase boundaries against approximately 2742 (Launching), 6806 (Floating), and 25688 (Terminating). Differences should be investigated, not force-corrected.

## Video clock

`clips` measures duration from the final decodable video timestamp rather than trusting the AVI container's declared duration. This matters for cameras that preallocate fixed-size, nominally 60-second files even when a clip contains only a second of video. It prefers creation timestamps only if every clip has one; otherwise it orders by modification time then natural filename and records that fallback. When the camera's wall clock is known to be wrong but its numeric filenames form the capture sequence, pass `--order-by filename`. When only timestamp order is trustworthy, pass `--order-by relative_time`; this records the timestamps as relative evidence and makes no claim about their calendar dates. Verify the order visually. Identify one exact event—launch or burst—in one clip and use `sync`. The resulting JSON records the original evidence and one offset.

Undecodable clips fail inventory by default. After independently verifying that a file is an empty camera placeholder rather than recoverable footage, use `--exclude-undecodable --excluded-output work/excluded-clips.csv`. The usable manifest and exact exclusion reason remain separate, auditable artifacts.

Google Drive for desktop can stream cloud-only clips directly to `ffprobe` and `ffmpeg`; do not mark an oversized source folder “available offline.” Extraction reads one clip at a time; inventory can use a small bounded pool such as `--workers 4` while preserving deterministic output order. On macOS, pass `--evict-source-cache` to release File Provider range caches in bounded batches after successful reads; this removes only local cached bytes, never the cloud original. Confirm files remain cloud-only after a trial clip before processing the full collection, and retain enough local space for extracted JPEGs.

`extract` removes the left third using `crop=2/3 width:x=1/3`, samples one frame per second, and preserves per-clip intermediate names. Use `--resume` for long cloud-backed runs; a completion marker is written only after a clip produces frames, and a marker is trusted only while its recorded frame count still matches disk. Pass `--source-checksum-manifest work/source-checksums.json` to hash each source before its cache is evicted; hashes are persisted per clip so interrupted runs do not lose that work. `name-frames` converts extracted images into flight-second keys and writes a naming manifest. Use `--mode hardlink` when both directories are on the same filesystem to avoid storing a second physical JPEG copy. Inspect random frames around clip boundaries and the anchor before continuing.

## Sampling

The telemetry has occasional missing or invalid GPS seconds. `build-frames` fails on these by default. For a reviewed real-data run, pass `--drop-unmatched --unmatched-output work/unmatched-frames.csv`; only image seconds absent from the canonical telemetry table are excluded, and the complete exclusion list is retained.

Compute pHash on the cropped image. A frame cannot pass inside the configured minimum interval. Outside it, sufficient ground movement or perceptual change admits the frame. Segment breaks occur on phase changes, large time gaps, or implausible geographic jumps. Spatial blocks—not adjacent images—are assigned to splits, then whole segments are kept together.

The default values are starting points from the specification. Freeze final values before model evaluation and report them.
