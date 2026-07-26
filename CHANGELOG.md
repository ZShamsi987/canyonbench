# Changelog

## Unreleased

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

## 0.1.0 - 2026-07-21

- Initial complete pre-data implementation of the pipeline, registration, ground-truth, inference, and evaluation toolkit.
