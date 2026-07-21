"""CanyonBench command-line interface."""

from __future__ import annotations

import importlib.util
import json
import shlex
import shutil
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from canyonbench.eval.metrics import load_and_score
from canyonbench.eval.runner import load_run_config
from canyonbench.eval.runner import run as run_benchmark
from canyonbench.groundtruth.release import build_release as assemble_release
from canyonbench.io import write_json
from canyonbench.pipeline.clips import inventory_clips
from canyonbench.pipeline.extract import extract_clips
from canyonbench.pipeline.flight_log import recover_operational_flight
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
def flight_log(source: Path, output: Path) -> None:
    """Recover and canonicalize the operational flight segment."""

    frame = recover_operational_flight(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    typer.echo(f"Wrote {len(frame)} valid operational rows to {output}")


@app.command()
def clips(directory: Path, output: Path) -> None:
    """Inventory and deterministically order camera clips."""

    frame = inventory_clips(directory)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    typer.echo(f"Wrote {len(frame)} clip records to {output}")


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
) -> None:
    """Plan or run one-Hz extraction with left-third removal."""

    commands = extract_clips(
        pd.read_csv(clips_csv), load_anchor(sync_json), output_dir, execute=execute
    )
    if execute:
        typer.echo(f"Executed {len(commands)} extraction commands")
    else:
        for command in commands:
            typer.echo(shlex.join(command))


@app.command("name-frames")
def name_frames(clips_csv: Path, sync_json: Path, extracted_dir: Path, output_dir: Path) -> None:
    """Materialize img_SSSSSS.jpg names keyed exactly to flight seconds."""

    plan = plan_frame_names(extracted_dir, pd.read_csv(clips_csv), load_anchor(sync_json))
    materialize_frame_names(plan, output_dir)
    plan.to_csv(output_dir / "naming_manifest.csv", index=False)
    typer.echo(f"Named {len(plan)} frames")


@app.command("build-frames")
def build_frames(images_dir: Path, flight_csv: Path, output: Path) -> None:
    """Join aerial frames to telemetry and compute objective quality controls."""

    frame = build_frames_table(discover_frames(images_dir), pd.read_csv(flight_csv))
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
    frame = assign_geographic_splits(assign_segments(frame))
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
