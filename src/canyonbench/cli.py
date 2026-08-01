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
from canyonbench.io import read_json, write_json
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
trace_app = typer.Typer(
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
    help="Build and evaluate the procedural CanyonBench-Trace v4 benchmark.",
)
app.add_typer(trace_app, name="trace")


def _version_callback(value: bool) -> bool:
    if value:
        typer.echo(__version__)
        raise typer.Exit()
    return value


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            help="Show the package version.",
            callback=_version_callback,
            is_eager=True,
        ),
    ] = False,
) -> None:
    """Prepare, register, validate, run, and score CanyonBench."""

    del version


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


@trace_app.command("discover-sites")
def trace_discover_sites(
    config: Path,
    output: Path,
    cache_dir: Annotated[
        Path,
        typer.Option(help="Reusable cache for low-resolution discovery products."),
    ] = Path("cache/site-discovery"),
) -> None:
    """Discover a deterministic, quota-balanced pool of candidate site centers."""

    from canyonbench.trace.acquisition import discover_candidates, write_candidate_manifest
    from canyonbench.trace.config import load_source_acquisition_config

    policy = load_source_acquisition_config(config)
    candidates = discover_candidates(policy, cache_dir.resolve())
    write_candidate_manifest(output.resolve(), candidates)
    typer.echo(f"Discovered {len(candidates)} candidate sites into {output.resolve()}")


@trace_app.command("acquire-sources")
def trace_acquire_sources(
    config: Path,
    candidates: Path,
    source_root: Path,
    output_manifest: Path,
    report: Path,
    flight_source: Annotated[
        Path | None,
        typer.Option(
            help="Optional flight log whose checksum is frozen into every source manifest."
        ),
    ] = None,
    start: Annotated[int, typer.Option(min=0)] = 0,
    limit: Annotated[int | None, typer.Option(min=1)] = None,
    fail_fast: Annotated[
        bool,
        typer.Option("--fail-fast", help="Stop at the first remote-source failure."),
    ] = False,
) -> None:
    """Materialize aligned imagery, dual masks, detector layers, and terrain."""

    from canyonbench.trace.acquisition import acquire_candidates
    from canyonbench.trace.config import (
        load_candidate_seeds,
        load_source_acquisition_config,
    )

    policy = load_source_acquisition_config(config)
    seeds = load_candidate_seeds(candidates)
    completed = acquire_candidates(
        seeds,
        policy,
        source_root.resolve(),
        output_manifest.resolve(),
        report.resolve(),
        flight_source_path=flight_source.resolve() if flight_source is not None else None,
        start=start,
        limit=limit,
        continue_on_error=not fail_fast,
        progress=typer.echo,
    )
    typer.echo(f"Materialized {len(completed)} source sites in this invocation")
    typer.echo(f"Prepared-site manifest: {output_manifest.resolve()}")
    typer.echo(f"Acquisition report: {report.resolve()}")


@trace_app.command("build")
def trace_build(
    config: Path,
    calibration: Annotated[
        Path | None,
        typer.Option(help="Override path to the frozen real-flight quality JSON."),
    ] = None,
    smoke: Annotated[
        bool,
        typer.Option(
            "--smoke", help="Allow a sub-quota site manifest for local integration tests."
        ),
    ] = False,
) -> None:
    """Generate aligned clean/degraded views and all causal interventions."""

    from canyonbench.trace.config import load_project_config, load_sites
    from canyonbench.trace.render import build_dataset

    project = load_project_config(config)
    sites = load_sites(project.dataset.site_manifest)
    output = build_dataset(
        project,
        sites,
        calibration_path=calibration,
        enforce_quota=not smoke,
    )
    typer.echo(f"Procedural dataset: {output}")


@trace_app.command("select-sites")
def trace_select_sites(
    candidates: Path,
    config: Path,
    output: Path,
    attempts: Annotated[int, typer.Option(min=1)] = 1000,
) -> None:
    """Gate candidates and freeze the independent quota-balanced 120-site cohort."""

    from canyonbench.trace.config import load_project_config, load_sites
    from canyonbench.trace.selection import select_sites, write_selection

    project = load_project_config(config)
    selected, gates = select_sites(
        load_sites(candidates),
        project.dataset,
        attempts=attempts,
    )
    write_selection(output, selected, gates)
    typer.echo(f"Selected {len(selected)} independent sites into {output}")
    typer.echo(f"Candidate gate report: {output.with_suffix('.gates.json')}")


@trace_app.command("merge-sites")
def trace_merge_sites(
    output: Path,
    chunks: Annotated[list[Path], typer.Argument(help="Per-chunk acquisition manifests.")],
) -> None:
    """Combine chunked acquisition manifests into one cohort manifest."""

    from canyonbench.trace.selection import merge_site_manifests

    report = merge_site_manifests(output, chunks)
    typer.echo(json.dumps(report, indent=2))


@trace_app.command("gate-ablations")
def trace_gate_ablations(config: Path, output: Path) -> None:
    """Run registered source/date/road-width/detector gate ablations."""

    from canyonbench.trace.ablations import write_gate_ablation_report
    from canyonbench.trace.config import load_project_config, load_sites

    project = load_project_config(config)
    sites = load_sites(project.dataset.site_manifest)
    write_gate_ablation_report(sites, project.dataset, output)
    typer.echo(f"Gate ablations: {output}")


@trace_app.command("instrument-v1")
def trace_instrument_v1(
    image: Path,
    output_dir: Path,
    widths: Annotated[
        str,
        typer.Option(help="Comma-separated apparent widths in pixels."),
    ] = "0.5,0.75,1,1.5,2,3,4,6",
) -> None:
    """Generate the V1 photometrically matched synthetic-width positive control."""

    from canyonbench.trace.instruments import build_synthetic_width_series

    parsed = [float(value.strip()) for value in widths.split(",") if value.strip()]
    rows = build_synthetic_width_series(image, output_dir, parsed)
    write_json(output_dir / "manifest.json", rows)
    typer.echo(f"V1 synthetic controls: {len(rows)} -> {output_dir}")


@trace_app.command("instrument-v1-run")
def trace_instrument_v1_run(
    config: Path,
    negative_image: Path,
    output_dir: Path,
    models: Annotated[
        list[str] | None,
        typer.Option("--model", help="Repeat to run a subset; default is the full roster."),
    ] = None,
) -> None:
    """Run and score V1 synthetic controls through the frozen model roster."""

    from canyonbench.trace.config import load_trace_run_config
    from canyonbench.trace.positive_control import (
        run_positive_control,
        score_positive_control,
    )

    predictions = run_positive_control(
        load_trace_run_config(config),
        negative_image,
        output_dir,
        model_ids=models,
    )
    result = score_positive_control(predictions, output_dir / "metrics.json")
    typer.echo(json.dumps(result, indent=2))


@trace_app.command("instrument-v2")
def trace_instrument_v2(dataset_dir: Path, output: Path) -> None:
    """Run the group-cross-validated O1-O4 edit-detectability audit."""

    from canyonbench.trace.instruments import dataset_edit_detectability_audit

    report = dataset_edit_detectability_audit(dataset_dir)
    write_json(output, report)
    typer.echo(json.dumps(report, indent=2))


@trace_app.command("compute-check")
def trace_compute_check(
    role: Annotated[
        Literal["adroit", "lambda", "openrouter"],
        typer.Option(help="Which host this is: adroit (CPU), lambda (GPU), or openrouter."),
    ],
    dataset_dir: Annotated[Path | None, typer.Option(help="Frozen dataset root to verify.")] = None,
    storage_root: Annotated[
        Path | None, typer.Option(help="Persistent Lambda filesystem root.")
    ] = None,
    output: Annotated[Path | None, typer.Option(help="Optional JSON report path.")] = None,
) -> None:
    """Go/no-go preflight for one host: credentials, storage, GPU, and dtype."""

    from canyonbench.compute import compute_check

    required = ("OPENROUTER_API_KEY",) if role == "openrouter" else ()
    report = compute_check(
        role=role,
        storage_root=storage_root,
        dataset_dir=dataset_dir,
        required_env=required,
    )
    for check in report["checks"]:
        mark = "ok  " if check["ok"] else ("FAIL" if check["blocking"] else "warn")
        typer.echo(f"{mark} {check['name']:34} {check['detail']}")
    if output:
        write_json(output, report)
    typer.echo("READY" if report["ready"] else f"BLOCKED: {report['blocking_failures']}")
    if not report["ready"]:
        raise typer.Exit(code=1)


@trace_app.command("vllm-profile")
def trace_vllm_profile(
    vram_gb: Annotated[
        float | None,
        typer.Option(help="Override detected device memory, in GB."),
    ] = None,
    server_args: Annotated[
        bool, typer.Option("--server-args", help="Emit api_server flags instead of JSON.")
    ] = False,
) -> None:
    """Emit the registered capability-adaptive vLLM serving profile."""

    from canyonbench.compute import serving_profile

    if vram_gb is None:
        from canyonbench.compute import _device_report

        devices = _device_report()["devices"]
        if not devices:
            typer.echo("No CUDA device detected; pass --vram-gb to compute a profile.", err=True)
            raise typer.Exit(code=1)
        vram_gb = min(float(device["vram_gb"]) for device in devices)
    profile = serving_profile(vram_gb)
    if server_args:
        typer.echo(shlex.join(profile.as_server_args()))
        return
    typer.echo(json.dumps({"vram_gb": profile.vram_gb, **profile.as_vllm_kwargs()}, indent=2))


@trace_app.command("merge-runs")
def trace_merge_runs(
    predictions: Annotated[list[Path], typer.Argument(help="Per-host predictions.jsonl files.")],
    output: Annotated[Path, typer.Option(help="Merged predictions.jsonl path.")],
) -> None:
    """Merge the Lambda and OpenRouter prediction logs into one scored input."""

    from canyonbench.trace.merge import merge_predictions

    report = merge_predictions(predictions, output)
    typer.echo(json.dumps(report, indent=2))


@trace_app.command("fidelity-report")
def trace_fidelity_report(
    dataset_dir: Path,
    output: Annotated[Path | None, typer.Option(help="Optional JSON report path.")] = None,
) -> None:
    """Quantify the reported sim-to-real relief-displacement gap (d = r*dh/H)."""

    from canyonbench.trace.fidelity import dataset_relief_report

    destination = output or dataset_dir / "relief_displacement.json"
    report = dataset_relief_report(dataset_dir, destination)
    summary = {key: value for key, value in report.items() if key != "views"}
    typer.echo(json.dumps(summary, indent=2))
    typer.echo(f"Relief-displacement report: {destination}")


@trace_app.command("calibrate-quality")
def trace_calibrate_quality(
    frames_dir: Path,
    output: Path,
    pattern: Annotated[str, typer.Option(help="Recursive frame glob.")] = "*.jpg",
) -> None:
    """Measure registered degradation proxies from real balloon frames."""

    from canyonbench.trace.degradation import calibrate_from_frames

    paths = sorted(path for path in frames_dir.rglob(pattern) if path.is_file())
    calibration = calibrate_from_frames(paths)
    write_json(output, calibration.model_dump(mode="json"))
    typer.echo(f"Calibrated {len(paths)} frames into {output}")


@trace_app.command("validate")
def trace_validate(
    directory: Path,
    config: Annotated[Path | None, typer.Option(help="Optional project config.")] = None,
    skip_interventions: Annotated[bool, typer.Option("--skip-interventions")] = False,
    output: Annotated[Path | None, typer.Option(help="Optional JSON report.")] = None,
) -> None:
    """Validate counts, schemas, hashes, masks, gates, and intervention coverage."""

    from canyonbench.trace.config import load_project_config
    from canyonbench.trace.validation import validate_dataset, validation_report

    project = load_project_config(config) if config else None
    report = validation_report(
        validate_dataset(
            directory,
            project=project,
            require_interventions=not skip_interventions,
        )
    )
    if output:
        write_json(output, report)
    typer.echo(json.dumps(report, indent=2))
    if not report["passed"]:
        raise typer.Exit(code=1)


@trace_app.command("audit-sample")
def trace_audit_sample(
    dataset_dir: Path,
    output: Path,
    auditor_1: Annotated[str, typer.Option()] = "auditor_1",
    auditor_2: Annotated[str, typer.Option()] = "auditor_2",
    fraction: Annotated[float, typer.Option(min=0.05, max=0.1)] = 0.1,
) -> None:
    """Create the objective two-auditor CSV; this is not semantic annotation."""

    from canyonbench.trace.audit import create_audit_sample

    path = create_audit_sample(
        dataset_dir,
        output,
        fraction=fraction,
        auditors=(auditor_1, auditor_2),
    )
    typer.echo(f"Audit sheet: {path}")


@trace_app.command("audit-summary")
def trace_audit_summary(
    source: Path,
    output: Path,
    dataset_dir: Annotated[
        Path | None,
        typer.Option(help="Frozen dataset root; adds the human extinction-band validation."),
    ] = None,
) -> None:
    """Validate completed audit rows and report agreement/failure votes."""

    from canyonbench.trace.audit import extinction_band_validation, load_audit, summarize_audit

    records = load_audit(source)
    summary = summarize_audit(records)
    if dataset_dir is not None:
        summary["extinction_band_validation"] = extinction_band_validation(records, dataset_dir)
    write_json(output, summary)
    typer.echo(json.dumps(summary, indent=2))


@trace_app.command("run")
def trace_run(
    config: Path,
    only_model: Annotated[
        list[str] | None,
        typer.Option(help="Restrict this invocation to named roster models (repeatable)."),
    ] = None,
    dataset_dir: Annotated[
        Path | None, typer.Option(help="Override the dataset root for this host.")
    ] = None,
    output_dir: Annotated[
        Path | None, typer.Option(help="Override the run output directory for this host.")
    ] = None,
) -> None:
    """Execute resumable Tier A/B/C structured black-box traces."""

    from canyonbench.trace.config import load_run_config_for_host
    from canyonbench.trace.runner import run_trace

    loaded = load_run_config_for_host(config, dataset_dir=dataset_dir, output_dir=output_dir)
    path = run_trace(loaded, only_models=only_model)
    typer.echo(f"Trace predictions: {path}")


@trace_app.command("plan-run")
def trace_plan_run(
    config: Path,
    output: Path | None = None,
    price_pilot: Annotated[
        Path | None,
        typer.Option(help="Optional price_pilot.json; re-prices the plan from measured tokens."),
    ] = None,
) -> None:
    """Calculate nominal and worst-case paid calls and dollars before inference."""

    from canyonbench.trace.config import load_trace_run_config
    from canyonbench.trace.planning import write_call_plan

    loaded = load_trace_run_config(config)
    destination = output or loaded.output_dir / "call_plan.json"
    observed = None
    if price_pilot is not None:
        measured = read_json(price_pilot)
        observed = measured.get("observed_tokens_per_call") or None
    plan = write_call_plan(loaded, destination, observed_tokens=observed)
    typer.echo(json.dumps(plan, indent=2))
    if not plan["nominal_fits_request_cap"]:
        raise typer.Exit(code=1)


@trace_app.command("price-pilot")
def trace_price_pilot(
    config: Path,
    calls: Annotated[int, typer.Option(min=1, help="Real calls per model (D1 uses 50).")] = 50,
    include_unmetered: Annotated[
        bool, typer.Option("--include-unmetered", help="Also smoke-test free endpoints.")
    ] = False,
    output_dir: Annotated[
        Path | None, typer.Option(help="Optional pilot output directory.")
    ] = None,
    dataset_dir: Annotated[
        Path | None, typer.Option(help="Override the dataset root for this host.")
    ] = None,
) -> None:
    """Measure real per-call tokens/cost and re-project the full run before spending."""

    from canyonbench.trace.config import load_run_config_for_host
    from canyonbench.trace.pilot import run_price_pilot

    loaded = load_run_config_for_host(config, dataset_dir=dataset_dir)
    report = run_price_pilot(
        loaded,
        calls_per_model=calls,
        include_unmetered=include_unmetered,
        output_dir=output_dir,
    )
    typer.echo(json.dumps(report, indent=2))
    if not report["authorized"]:
        raise typer.Exit(code=1)


@trace_app.command("score")
def trace_score(
    dataset_dir: Path,
    predictions: Path,
    output: Path,
    bootstrap_iterations: Annotated[int, typer.Option(min=0)] = 2000,
    seed: int = 2026,
    cave_decisions: Annotated[
        Path | None, typer.Option(help="Optional frozen CAVE decision JSONL.")
    ] = None,
    cave_ablations: Annotated[
        Path | None, typer.Option(help="Optional frozen CAVE component-ablation JSONL.")
    ] = None,
    cave_frontier: Annotated[
        Path | None,
        typer.Option(help="Optional development-only CAVE frontier JSON."),
    ] = None,
) -> None:
    """Compute v4 performance, localization, causal, and uncertainty metrics."""

    from canyonbench.trace.metrics import score_trace

    result = score_trace(
        dataset_dir,
        predictions,
        output,
        bootstrap_iterations=bootstrap_iterations,
        seed=seed,
        cave_decisions=cave_decisions,
        cave_ablations=cave_ablations,
        cave_frontier_path=cave_frontier,
    )
    typer.echo(json.dumps(result, indent=2))


@trace_app.command("cave-tune")
def trace_cave_tune(
    dataset_dir: Path,
    predictions: Path,
    output: Path,
) -> None:
    """Tune CAVE thresholds exclusively from development traces."""

    from canyonbench.trace.cave_pipeline import tune_from_run

    thresholds = tune_from_run(dataset_dir, predictions, output)
    typer.echo(json.dumps(thresholds.model_dump(mode="json"), indent=2))
    typer.echo(f"Frontier: {output.with_name(f'{output.stem}.frontier.json')}")


@trace_app.command("cave-apply")
def trace_cave_apply(
    dataset_dir: Path,
    predictions: Path,
    thresholds: Path,
    output: Path,
) -> None:
    """Apply frozen CAVE thresholds to all splits without re-tuning."""

    from canyonbench.io import read_json
    from canyonbench.trace.cave_pipeline import apply_from_run
    from canyonbench.trace.schemas import CaveThresholds

    config = CaveThresholds.model_validate(read_json(thresholds))
    decisions = apply_from_run(dataset_dir, predictions, config, output)
    typer.echo(f"CAVE decisions: {len(decisions)} -> {output}")


@trace_app.command("cave-ablations")
def trace_cave_ablations(
    dataset_dir: Path,
    predictions: Path,
    thresholds: Path,
    output: Path,
) -> None:
    """Apply necessity-only, sufficiency-only, nuisance-only, and full CAVE."""

    from canyonbench.io import read_json
    from canyonbench.trace.cave_pipeline import component_ablations_from_run
    from canyonbench.trace.schemas import CaveThresholds

    config = CaveThresholds.model_validate(read_json(thresholds))
    records = component_ablations_from_run(
        dataset_dir,
        predictions,
        config,
        output,
    )
    typer.echo(f"CAVE component-ablation records: {len(records)} -> {output}")


@trace_app.command("reference-baselines")
def trace_reference_baselines(dataset_dir: Path, output: Path) -> None:
    """Write always/base-rate/geographic-prior V5 predictions."""

    from canyonbench.io import read_json
    from canyonbench.trace.baselines import deterministic_baselines

    frame = deterministic_baselines(read_json(dataset_dir / "index.json"))
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    typer.echo(f"Reference baseline rows: {len(frame)} -> {output}")


@trace_app.command("extinction-figure")
def trace_extinction_figure(dataset_dir: Path, output_dir: Path) -> None:
    """Build the GSD/apparent-width extinction figure and ladder table."""

    from canyonbench.trace.reporting import build_extinction_figure

    outputs = build_extinction_figure(dataset_dir, output_dir)
    typer.echo(f"Extinction artifacts: {len(outputs)} -> {output_dir}")


@trace_app.command("report")
def trace_report(
    metrics: Path,
    rows: Path,
    output_dir: Path,
    dataset_dir: Annotated[
        Path | None,
        typer.Option(help="Frozen dataset root; adds the GSD/extinction instrument figure."),
    ] = None,
) -> None:
    """Create tidy/LaTeX tables and publication figures."""

    from canyonbench.trace.reporting import build_extinction_figure, build_report

    outputs = build_report(metrics, rows, output_dir)
    if dataset_dir is not None:
        outputs.extend(build_extinction_figure(dataset_dir, output_dir))
    typer.echo(f"Report artifacts: {len(outputs)} -> {output_dir}")


@trace_app.command("release")
def trace_release(dataset_dir: Path, output_dir: Path) -> None:
    """Build public dev/validation artifacts and hash-only test escrow."""

    from canyonbench.trace.release import build_release

    public, escrow = build_release(dataset_dir, output_dir)
    typer.echo(f"Public release: {public}")
    typer.echo(f"Private test escrow manifest: {escrow}")


@trace_app.command("release-validate")
def trace_release_validate(public_dir: Path, escrow_dir: Path) -> None:
    """Re-hash and validate a built public/escrow release pair."""

    from canyonbench.trace.release import validate_built_release

    result = validate_built_release(public_dir, escrow_dir)
    typer.echo(json.dumps(result, indent=2))


@trace_app.command("align-raster")
def trace_align_raster(
    source: Path,
    reference: Path,
    output: Path,
    continuous: Annotated[
        bool, typer.Option("--continuous", help="Use bilinear resampling for a DEM.")
    ] = False,
) -> None:
    """Reproject a mask, detector score, or DEM onto the exact imagery grid."""

    from canyonbench.trace.sources import align_raster_to_reference

    align_raster_to_reference(source, reference, output, categorical=not continuous)
    typer.echo(f"Aligned raster: {output}")


@trace_app.command("rasterize-geojson")
def trace_rasterize_geojson(
    source: Path,
    reference: Path,
    output: Path,
    all_touched: Annotated[bool, typer.Option("--all-touched")] = False,
) -> None:
    """Rasterize a GeoJSON feature layer onto the exact imagery grid."""

    from canyonbench.io import read_json
    from canyonbench.trace.sources import rasterize_geojson

    rasterize_geojson(read_json(source), reference, output, all_touched=all_touched)
    typer.echo(f"Rasterized mask: {output}")


if __name__ == "__main__":
    app()
