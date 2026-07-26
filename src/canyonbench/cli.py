"""CanyonBench command-line interface."""

from __future__ import annotations

import importlib.util
import json
import shlex
import shutil
from pathlib import Path
from typing import Annotated, Literal

import pandas as pd
import typer

from canyonbench.eval.metrics import load_and_score
from canyonbench.eval.runner import load_run_config
from canyonbench.eval.runner import run as run_benchmark
from canyonbench.groundtruth.release import build_release as assemble_release
from canyonbench.io import write_json
from canyonbench.pipeline.clips import inventory_clips
from canyonbench.pipeline.extract import extract_clips
from canyonbench.pipeline.flight_log import audit_flight_segments, recover_operational_flight
from canyonbench.pipeline.join import build_frames_table, discover_frames
from canyonbench.pipeline.naming import materialize_frame_names, plan_frame_names
from canyonbench.pipeline.sampling import (
    add_perceptual_hashes,
    assign_geographic_splits,
    assign_segments,
    sample_frames,
)
from canyonbench.pipeline.sync import compute_anchor, load_anchor, save_anchor
from canyonbench.registration.batch import register_manifest
from canyonbench.registration.reference import ReferenceChipRequest, fetch_reference_chip
from canyonbench.validation import validate_release
from canyonbench.version import __version__

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)


@app.callback()
def main(
    version: Annotated[bool, typer.Option("--version", help="Show the package version.")] = False,
) -> None:
    """Prepare, register, validate, run, and score CanyonBench."""

    if version:
        typer.echo(__version__)
        raise typer.Exit()


@app.command()
def doctor() -> None:
    """Report optional executables and Python extras."""

    checks = {
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "ffprobe": shutil.which("ffprobe") is not None,
        "opencv": importlib.util.find_spec("cv2") is not None,
        "rasterio": importlib.util.find_spec("rasterio") is not None,
        "statsmodels": importlib.util.find_spec("statsmodels") is not None,
    }
    for name, available in checks.items():
        typer.echo(f"{name:12} {'ok' if available else 'missing (optional until used)'}")


@app.command("flight-log")
def flight_log(
    source: Path,
    output: Path,
    audit_output: Annotated[
        Path | None,
        typer.Option("--audit-output", help="Optional CSV summary of every reset session."),
    ] = None,
) -> None:
    """Recover and canonicalize the operational flight segment."""

    frame = recover_operational_flight(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    if audit_output is not None:
        audit = audit_flight_segments(source)
        audit_output.parent.mkdir(parents=True, exist_ok=True)
        audit.to_csv(audit_output, index=False)
        typer.echo(f"Wrote {len(audit)} reset-session audit rows to {audit_output}")
    typer.echo(f"Wrote {len(frame)} valid operational rows to {output}")


@app.command()
def clips(
    directory: Path,
    output: Path,
    order_by: Annotated[
        Literal["auto", "filename", "relative_time"],
        typer.Option(
            "--order-by",
            help="Clip order: auto, filename, or relative_time.",
        ),
    ] = "auto",
    timeline_by: Annotated[
        Literal["contiguous", "relative_mtime_end"],
        typer.Option(
            "--timeline-by",
            help="Video clock: concatenate clips or use relative mtime as each clip end.",
        ),
    ] = "contiguous",
    evict_source_cache: Annotated[
        bool,
        typer.Option(
            "--evict-source-cache",
            help="On macOS, return probed cloud files to placeholder state in batches.",
        ),
    ] = False,
    workers: Annotated[
        int,
        typer.Option(min=1, max=8, help="Concurrent ffprobe workers."),
    ] = 1,
    exclude_undecodable: Annotated[
        bool,
        typer.Option(
            "--exclude-undecodable",
            help="Audit and exclude clips with no decodable video instead of failing.",
        ),
    ] = False,
    excluded_output: Annotated[
        Path | None,
        typer.Option(
            "--excluded-output",
            help="Optional CSV audit of clips excluded as undecodable.",
        ),
    ] = None,
) -> None:
    """Inventory and deterministically order camera clips."""

    frame = inventory_clips(
        directory,
        order_by=order_by,
        timeline_by=timeline_by,
        evict_source_cache=evict_source_cache,
        workers=workers,
        exclude_undecodable=exclude_undecodable,
    )
    excluded = pd.DataFrame(frame.attrs.get("excluded_clips", []))
    if excluded_output is not None:
        excluded_output.parent.mkdir(parents=True, exist_ok=True)
        excluded.to_csv(excluded_output, index=False)
        typer.echo(f"Wrote {len(excluded)} excluded-clip audit rows to {excluded_output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    typer.echo(f"Wrote {len(frame)} usable clip records to {output}")


@app.command()
def sync(
    clips_csv: Path,
    output: Path,
    anchor_clip: Annotated[str, typer.Option("--anchor-clip")],
    anchor_offset_s: Annotated[float, typer.Option("--anchor-offset-s")],
    flight_elapsed_s: Annotated[int, typer.Option("--flight-elapsed-s")],
    event: Annotated[str, typer.Option()] = "verified_visual_anchor",
) -> None:
    """Create a single, auditable video-to-flight clock offset."""

    anchor = compute_anchor(
        pd.read_csv(clips_csv), anchor_clip, anchor_offset_s, flight_elapsed_s, event
    )
    save_anchor(output, anchor)
    typer.echo(f"Flight offset: {anchor.flight_offset_s:.3f} s")


@app.command()
def extract(
    clips_csv: Path,
    sync_json: Path,
    output_dir: Path,
    execute: Annotated[
        bool, typer.Option("--execute", help="Run ffmpeg; default prints a plan.")
    ] = False,
    resume: Annotated[
        bool,
        typer.Option("--resume", help="Skip clips with a verified completion marker."),
    ] = False,
    evict_source_cache: Annotated[
        bool,
        typer.Option(
            "--evict-source-cache",
            help="On macOS, release cloud source cache after each completed batch.",
        ),
    ] = False,
    source_checksum_manifest: Annotated[
        Path | None,
        typer.Option(
            "--source-checksum-manifest",
            help="Optional JSON manifest of source clip SHA-256 values.",
        ),
    ] = None,
) -> None:
    """Plan or run one-Hz extraction with left-third removal."""

    commands = extract_clips(
        pd.read_csv(clips_csv),
        load_anchor(sync_json),
        output_dir,
        execute=execute,
        resume=resume,
        evict_source_cache=evict_source_cache,
        checksum_manifest=source_checksum_manifest,
    )
    if execute:
        typer.echo(f"Extraction complete for {len(commands)} clip records")
    else:
        for command in commands:
            typer.echo(shlex.join(command))


@app.command("name-frames")
def name_frames(
    clips_csv: Path,
    sync_json: Path,
    extracted_dir: Path,
    output_dir: Path,
    mode: Annotated[
        Literal["copy", "hardlink", "move"],
        typer.Option(help="Materialize named frames by copy, hardlink, or move."),
    ] = "copy",
) -> None:
    """Materialize img_SSSSSS.jpg names keyed exactly to flight seconds."""

    plan = plan_frame_names(extracted_dir, pd.read_csv(clips_csv), load_anchor(sync_json))
    materialize_frame_names(plan, output_dir, mode=mode)
    plan.to_csv(output_dir / "naming_manifest.csv", index=False)
    typer.echo(f"Named {len(plan)} frames")


@app.command("build-frames")
def build_frames(
    images_dir: Path,
    flight_csv: Path,
    output: Path,
    drop_unmatched: Annotated[
        bool,
        typer.Option(
            "--drop-unmatched",
            help="Exclude image seconds without valid telemetry instead of failing.",
        ),
    ] = False,
    unmatched_output: Annotated[
        Path | None,
        typer.Option(
            "--unmatched-output",
            help="Optional CSV audit of image seconds absent from telemetry.",
        ),
    ] = None,
) -> None:
    """Join aerial frames to telemetry and compute objective quality controls."""

    images = discover_frames(images_dir)
    flight = pd.read_csv(flight_csv)
    unmatched = images.loc[~images["elapsed_s"].isin(flight["elapsed_s"])].copy()
    if unmatched_output is not None:
        unmatched_output.parent.mkdir(parents=True, exist_ok=True)
        unmatched.to_csv(unmatched_output, index=False)
        typer.echo(f"Wrote {len(unmatched)} unmatched-frame audit rows to {unmatched_output}")
    frame = build_frames_table(images, flight, drop_unmatched=drop_unmatched)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    typer.echo(f"Wrote {len(frame)} joined aerial frames")


@app.command()
def sample(
    frames_csv: Path,
    output: Path,
    image_root: Annotated[Path | None, typer.Option()] = None,
    min_interval_s: Annotated[int, typer.Option()] = 60,
    distance_m: Annotated[float, typer.Option()] = 500,
    phash_distance: Annotated[int, typer.Option()] = 8,
    max_interval_s: Annotated[int | None, typer.Option()] = None,
    segment_max_duration_s: Annotated[
        int,
        typer.Option(
            min=60,
            help="Maximum contiguous trajectory-segment duration in seconds.",
        ),
    ] = 600,
) -> None:
    """Deduplicate, assign segments, and create geographic splits."""

    frame = pd.read_csv(frames_csv)
    if "phash" not in frame:
        frame = add_perceptual_hashes(frame, image_root)
    raw_count = len(frame)
    frame = sample_frames(
        frame,
        distance_m=distance_m,
        phash_distance=phash_distance,
        min_interval_s=min_interval_s,
        max_interval_s=max_interval_s,
    )
    frame = assign_geographic_splits(assign_segments(frame, max_duration_s=segment_max_duration_s))
    frame.attrs["raw_frame_count"] = raw_count
    frame["sampling_raw_count"] = raw_count
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    typer.echo(f"Retained {len(frame)}/{raw_count} frames in {frame.segment_id.nunique()} segments")


@app.command()
def register(manifest_csv: Path, output_dir: Path, threshold_m: float) -> None:
    """Fit and validate a batch of frame-to-map homographies."""

    residuals = register_manifest(
        pd.read_csv(manifest_csv), output_dir=output_dir, default_threshold_m=threshold_m
    )
    typer.echo(f"Reliable registrations: {int(residuals.reliable.sum())}/{len(residuals)}")


@app.command("reference-chip")
def reference_chip(
    output: Path,
    west: Annotated[float, typer.Option(help="Western WGS84 longitude.")],
    south: Annotated[float, typer.Option(help="Southern WGS84 latitude.")],
    east: Annotated[float, typer.Option(help="Eastern WGS84 longitude.")],
    north: Annotated[float, typer.Option(help="Northern WGS84 latitude.")],
    width_px: Annotated[
        int,
        typer.Option(min=1, max=4000, help="Output width; USGS limits exports to 4000."),
    ] = 2000,
    height_px: Annotated[
        int,
        typer.Option(min=1, max=4000, help="Output height; USGS limits exports to 4000."),
    ] = 2000,
    year: Annotated[int, typer.Option(help="NAIP acquisition year to lock.")] = 2023,
    image_crs: Annotated[
        int,
        typer.Option(help="Metric output EPSG code; CanyonBench uses NAD83 / UTM zone 12N."),
    ] = 26912,
    force: Annotated[
        bool,
        typer.Option("--force", help="Replace an existing chip and provenance sidecar."),
    ] = False,
) -> None:
    """Fetch one bounded USGS NAIP GeoTIFF into the ignored local cache."""

    request = ReferenceChipRequest(
        west=west,
        south=south,
        east=east,
        north=north,
        width_px=width_px,
        height_px=height_px,
        year=year,
        image_crs=image_crs,
    )
    metadata = fetch_reference_chip(request, output, force=force)
    action = "Using cached" if metadata["cache_hit"] else "Downloaded"
    typer.echo(f"{action} {output}")
    typer.echo(f"SHA-256: {metadata['artifact']['sha256']}")
    typer.echo(f"Provenance: {output.with_suffix(output.suffix + '.reference.json')}")


@app.command("build-release")
def build_release(frames_csv: Path, data_repository: Path, output_dir: Path) -> None:
    """Merge adjudicated annotations, masks, grids, and registration."""

    release = assemble_release(pd.read_csv(frames_csv), data_repository, output_dir)
    typer.echo(f"Built release with {len(release)} frames")


@app.command("validate-release")
def validate_release_command(
    directory: Path,
    metadata_only: Annotated[bool, typer.Option("--metadata-only")] = False,
) -> None:
    """Check schemas, joins, masks, registration gates, and split leakage."""

    issues = validate_release(directory, require_files=not metadata_only)
    for issue in issues:
        suffix = f" [{issue.image}]" if issue.image else ""
        typer.echo(f"{issue.severity.upper()} {issue.code}: {issue.message}{suffix}")
    errors = [issue for issue in issues if issue.severity == "error"]
    if errors:
        raise typer.Exit(code=1)
    typer.echo("Release validation passed")


@app.command("run")
def run_command(config: Path) -> None:
    """Execute a resumable constrained-output probe run."""

    path = run_benchmark(load_run_config(config))
    typer.echo(f"Predictions: {path}")


@app.command()
def score(
    release_dir: Path,
    predictions: Path,
    output: Path,
    bootstrap_iterations: Annotated[int, typer.Option(min=100)] = 2000,
    seed: Annotated[int, typer.Option()] = 2026,
) -> None:
    """Compute core metrics with segment-bootstrap confidence intervals."""

    metrics = load_and_score(
        str(release_dir / "frames.csv"),
        str(predictions),
        bootstrap_iterations=bootstrap_iterations,
        seed=seed,
    )
    write_json(output, metrics)
    typer.echo(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    app()
