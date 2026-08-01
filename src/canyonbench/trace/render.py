"""End-to-end procedural view generation and immutable bundle writing."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import rasterio  # type: ignore[import-untyped]
from PIL import Image

from canyonbench.exceptions import DataValidationError
from canyonbench.io import read_json, sha256_file, write_json
from canyonbench.trace.config import load_source_manifest
from canyonbench.trace.degradation import apply_degradation, sample_degradations
from canyonbench.trace.derived import derive_feature
from canyonbench.trace.fidelity import view_relief_displacement
from canyonbench.trace.gates import evaluate_gates
from canyonbench.trace.geometry import camera_window, render_aligned_rasters
from canyonbench.trace.interventions import build_interventions_for_view
from canyonbench.trace.schemas import (
    CameraSpec,
    DatasetConfig,
    FeatureClass,
    GateResult,
    ProjectConfig,
    QualityCalibration,
    QualityParameters,
    SiteSpec,
    ViewManifest,
)
from canyonbench.trace.splits import (
    assign_splits,
    independence_issues,
    leakage_issues,
    validate_quota,
)


def view_id(altitude_m: float, geometry: str) -> str:
    altitude = f"{altitude_m / 1000:g}".replace(".", "p")
    return f"view_a{altitude}km_{geometry}"


def _camera(site: SiteSpec, config: DatasetConfig, altitude: float, geometry: str) -> CameraSpec:
    with rasterio.open(site.imagery_path) as source:
        source_crs = source.crs
    if source_crs is None or source_crs.to_epsg() is None:
        raise DataValidationError(f"Site imagery must have an EPSG CRS: {site.imagery_path}")
    return CameraSpec(
        longitude=site.longitude,
        latitude=site.latitude,
        altitude_agl_m=altitude,
        horizontal_fov_deg=config.horizontal_fov_deg,
        width_px=config.width_px,
        height_px=config.height_px,
        pitch_deg=config.oblique_pitch_deg if geometry == "oblique" else 0,
        yaw_deg=0,
        source_crs=f"EPSG:{source_crs.to_epsg()}",
    )


def _load_native(path: Path) -> np.ndarray:
    with rasterio.open(path) as source:
        array: np.ndarray = source.read([1])[0]
        return array


def _load_native_rgb(path: Path) -> np.ndarray:
    with rasterio.open(path) as source:
        bands = source.read(list(range(1, min(3, source.count) + 1)))
    if bands.shape[0] == 1:
        bands = np.repeat(bands, 3, axis=0)
    rgb = np.moveaxis(bands, 0, -1)
    if rgb.max(initial=0) > 255:
        rgb = rgb.astype(float) / max(float(rgb.max()), 1) * 255
    output: np.ndarray = np.clip(rgb, 0, 255).astype(np.uint8)
    return output


def _maximum_camera_footprint(
    site: SiteSpec,
    config: DatasetConfig,
    shape: tuple[int, int],
) -> np.ndarray:
    """Union the registered maximum-altitude nadir/oblique footprints."""

    footprint = np.zeros(shape, dtype=np.uint8)
    altitude = max(config.altitudes_agl_m)
    with rasterio.open(site.imagery_path) as source:
        for geometry in config.geometries:
            camera = _camera(site, config, altitude, geometry)
            points = np.rint(camera_window(source, camera).source_points).astype(np.int32)
            cv2.fillPoly(footprint, [points], 1)
    safety_px = max(
        1,
        round(config.gates.negative_safety_buffer_m / site.native_resolution_m),
    )
    return cv2.dilate(
        footprint,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (safety_px * 2 + 1, safety_px * 2 + 1),
        ),
    )


def evaluate_site(site: SiteSpec, config: DatasetConfig) -> list[GateResult]:
    """Apply G1-G4 inside the buffered maximum registered camera footprint."""

    image = _load_native_rgb(site.imagery_path)
    footprint = _maximum_camera_footprint(site, config, image.shape[:2]) > 0
    gated_image = np.where(footprint[..., None], image, 0).astype(np.uint8)
    results: list[GateResult] = []
    for feature in ("water", "road", "field"):
        detector = (
            _load_native(site.detector_score_paths[feature])
            if feature in site.detector_score_paths
            else None
        )
        if detector is not None:
            detector = np.where(footprint, detector, 0)
        case_type = site.case_type if feature == site.target_class else "positive"
        feature_site = site.model_copy(
            update={
                "target_class": feature,
                "case_type": case_type,
            }
        )
        primary = np.where(
            footprint,
            _load_native(site.primary_mask_paths[feature]),
            0,
        )
        secondary = np.where(
            footprint,
            _load_native(site.secondary_mask_paths[feature]),
            0,
        )
        if feature != site.target_class and not np.any(primary):
            feature_site = feature_site.model_copy(update={"case_type": "negative"})
        results.append(
            evaluate_gates(
                feature_site,
                feature,
                primary,
                secondary,
                gated_image,
                config=config.gates,
                native_resolution_m=site.native_resolution_m,
                detector_score=detector,
            )
        )
    return results


def _write_depth(path: Path, depth: np.ndarray) -> None:
    profile = {
        "driver": "GTiff",
        "height": depth.shape[0],
        "width": depth.shape[1],
        "count": 1,
        "dtype": "float32",
        "compress": "deflate",
        "nodata": np.nan,
    }
    with rasterio.open(path, "w", **profile) as destination:
        destination.write(depth.astype(np.float32), 1)


def _quality_assignments(
    sites: list[SiteSpec],
    config: DatasetConfig,
    *,
    calibration: QualityCalibration | None,
    calibration_source_sha256: str | None,
) -> dict[str, QualityParameters]:
    """Choose two views per site (240 total) with balanced lattice coverage."""

    all_keys: list[str] = []
    selected: list[str] = []
    lattice = [
        (altitude, geometry)
        for altitude in config.altitudes_agl_m
        for geometry in config.geometries
    ]
    for index, site in enumerate(sorted(sites, key=lambda value: value.site_id)):
        keys = [f"{site.site_id}/{view_id(altitude, geometry)}" for altitude, geometry in lattice]
        all_keys.extend(keys)
        # A rotating pair gives every altitude/geometry equal frequency over 120 sites.
        selected.extend([keys[(2 * index) % 8], keys[(2 * index + 3) % 8]])
    if config.degraded_view_count != len(selected):
        # V4 is two per site; custom smoke configurations use random selection.
        selected = sorted(
            np.random.default_rng(config.seed).choice(
                all_keys, size=config.degraded_view_count, replace=False
            )
        )
    sampled = sample_degradations(
        selected,
        count=len(selected),
        seed=config.seed,
        calibration=calibration,
        calibration_source_sha256=calibration_source_sha256,
    )
    return {key: sampled.get(key, QualityParameters()) for key in all_keys}


def _validate_inputs(sites: list[SiteSpec], config: DatasetConfig, *, enforce_quota: bool) -> None:
    if enforce_quota:
        validate_quota(sites, config)
    missing: list[Path] = []
    for site in sites:
        candidates = [
            site.imagery_path,
            site.source_manifest_path,
            *site.primary_mask_paths.values(),
            *site.secondary_mask_paths.values(),
            *site.detector_score_paths.values(),
        ]
        if site.dem_path:
            candidates.append(site.dem_path)
        missing.extend(path for path in candidates if not path.exists())
        if (
            enforce_quota
            and site.case_type != "negative"
            and site.target_class not in site.detector_score_paths
        ):
            raise DataValidationError(
                f"{site.site_id} requires an independent "
                f"{site.target_class} detector score raster for G4/extinction"
            )
    if missing:
        raise DataValidationError(f"Missing source files: {[str(path) for path in missing[:20]]}")
    for site in sites:
        manifest = load_source_manifest(site.source_manifest_path)
        hash_pairs: list[tuple[Path, str, str]] = [
            (site.imagery_path, manifest.imagery.sha256, "imagery")
        ]
        for feature in ("water", "road", "field"):
            records = manifest.feature_sources[feature]
            hash_pairs.extend(
                [
                    (
                        site.primary_mask_paths[feature],
                        records[0].sha256,
                        f"{feature} primary",
                    ),
                    (
                        site.secondary_mask_paths[feature],
                        records[1].sha256,
                        f"{feature} secondary",
                    ),
                ]
            )
            if feature in site.detector_score_paths:
                record = manifest.detector_sources.get(feature)
                if record is None:
                    raise DataValidationError(
                        f"{site.site_id} has a {feature} detector raster but no "
                        "detector source record"
                    )
                hash_pairs.append(
                    (
                        site.detector_score_paths[feature],
                        record.sha256,
                        f"{feature} detector",
                    )
                )
        if site.dem_path is not None:
            if manifest.terrain is None:
                raise DataValidationError(f"{site.site_id} has a DEM but no terrain source record")
            hash_pairs.append((site.dem_path, manifest.terrain.sha256, "terrain"))
        mismatches = [name for path, expected, name in hash_pairs if sha256_file(path) != expected]
        if mismatches:
            raise DataValidationError(
                f"{site.site_id} source hashes do not match its manifest: {mismatches}"
            )


def build_dataset(
    project: ProjectConfig,
    sites: list[SiteSpec],
    *,
    calibration_path: Path | None = None,
    enforce_quota: bool = True,
) -> Path:
    """Generate clean views plus the preregistered degraded counterparts."""

    config = project.dataset
    _validate_inputs(sites, config, enforce_quota=enforce_quota)
    if any(site.split is None for site in sites):
        sites = assign_splits(sites, seed=config.seed)
    leakage = leakage_issues(sites)
    if leakage:
        raise DataValidationError("Split leakage detected:\n" + "\n".join(leakage))
    independence = independence_issues(
        sites, minimum_site_separation_m=config.minimum_site_separation_m
    )
    if independence:
        raise DataValidationError(
            "Site independence violations detected:\n" + "\n".join(independence)
        )
    output = config.output_root
    output.mkdir(parents=True, exist_ok=True)
    calibration_source = calibration_path or config.quality_calibration
    calibration: QualityCalibration | None = None
    calibration_source_sha256: str | None = None
    if calibration_source is not None:
        if not calibration_source.exists():
            raise DataValidationError(f"Quality calibration does not exist: {calibration_source}")
        calibration = QualityCalibration.model_validate(read_json(calibration_source))
        calibration_source_sha256 = sha256_file(calibration_source)
    elif config.degraded_view_count and enforce_quota:
        raise DataValidationError(
            "A frozen real-flight quality calibration is required for degraded views"
        )
    qualities = _quality_assignments(
        sites,
        config,
        calibration=calibration,
        calibration_source_sha256=calibration_source_sha256,
    )
    index_rows: list[dict[str, object]] = []
    gate_counts: Counter[str] = Counter()

    # The clean-view lattice is exactly sites x altitudes x geometries. For the
    # frozen v4 design that is 120 x 4 x 2 = 960, and the loop below is the only
    # place views are created, so the count is a property of the loop rather
    # than a target to be met. `clean_views_per_site` is asserted per site and
    # the total is asserted once the loop finishes.
    clean_views_per_site = len(config.altitudes_agl_m) * len(config.geometries)
    expected_clean_views = len(sites) * clean_views_per_site

    for site in sorted(sites, key=lambda value: value.site_id):
        assert site.split is not None
        site_directory = output / site.site_id
        site_directory.mkdir(parents=True, exist_ok=True)
        gates = evaluate_site(site, config)
        target_gate = next(row for row in gates if row.feature == site.target_class)
        if not target_gate.accepted:
            raise DataValidationError(
                f"{site.site_id}/{site.target_class} failed automatic gates: {target_gate.reasons}"
            )
        for result in gates:
            gate_counts["accepted" if result.accepted else "excluded_non_target"] += 1
        write_json(
            site_directory / "gate_results.json",
            [row.model_dump(mode="json") for row in gates],
        )
        source_manifest = load_source_manifest(site.source_manifest_path)
        if source_manifest.site_id != site.site_id:
            raise DataValidationError(
                f"Source manifest site_id {source_manifest.site_id} does not match {site.site_id}"
            )
        write_json(
            site_directory / "source_manifest.json",
            source_manifest.model_dump(mode="json"),
        )
        source_manifest_hash = sha256_file(site_directory / "source_manifest.json")
        for altitude in config.altitudes_agl_m:
            for geometry in config.geometries:
                identifier = view_id(altitude, geometry)
                camera = _camera(site, config, altitude, geometry)
                rgb, masks, detector_scores, depth, _ = render_aligned_rasters(
                    site.imagery_path,
                    {str(key): path for key, path in site.primary_mask_paths.items()},
                    camera,
                    dem_path=site.dem_path,
                    continuous_paths={
                        str(key): path for key, path in site.detector_score_paths.items()
                    },
                    inject_relief=config.inject_relief_displacement,
                )
                relief = view_relief_displacement(camera, depth).model_copy(
                    update={"injected": config.inject_relief_displacement}
                )
                view_directory = site_directory / identifier
                view_directory.mkdir(parents=True, exist_ok=True)
                Image.fromarray(rgb).save(view_directory / "rgb.png", optimize=True)
                for feature, mask in masks.items():
                    Image.fromarray(mask * 255).save(view_directory / f"{feature}_mask.png")
                if depth is not None:
                    _write_depth(view_directory / "depth.tif", depth)
                clean_quality = QualityParameters()
                features = {}
                for feature, mask in masks.items():
                    if feature == site.target_class:
                        case_type = site.case_type
                    else:
                        case_type = "positive" if mask.any() else "negative"
                    features[feature] = derive_feature(
                        rgb,
                        mask,
                        case_type=case_type,
                        minimum_resolvable_width_px=(
                            config.gates.road_minimum_apparent_width_px
                            if feature == "road"
                            else config.gates.minimum_apparent_width_px
                        ),
                        extinction_width_px=config.gates.extinction_width_px,
                        minimum_component_px=config.gates.minimum_component_px,
                        minimum_boundary_distance_px=(config.gates.minimum_boundary_distance_px),
                        minimum_local_contrast=config.gates.minimum_local_contrast,
                        maximum_occlusion_fraction=(config.gates.maximum_occlusion_fraction),
                        detector_score=detector_scores.get(feature),
                        detector_min_score=config.gates.exclusion_detector_min_score,
                    )
                target_feature = features[site.target_class]
                if enforce_quota and site.case_type != "negative" and not target_feature.present:
                    raise DataValidationError(
                        f"{site.site_id}/{identifier} loses the target feature "
                        "outside the camera footprint"
                    )
                if (
                    enforce_quota
                    and site.case_type != "negative"
                    and not target_feature.resolvable
                    and not target_feature.extinction
                ):
                    raise DataValidationError(
                        f"{site.site_id}/{identifier} is neither resolvable nor "
                        "an empirically gated extinction view"
                    )
                write_json(view_directory / "camera.json", camera.model_dump(mode="json"))
                write_json(view_directory / "quality.json", clean_quality.model_dump(mode="json"))
                write_json(
                    view_directory / "derived.json",
                    {name: value.model_dump(mode="json") for name, value in features.items()},
                )
                manifest = ViewManifest(
                    site_id=site.site_id,
                    view_id=identifier,
                    split=site.split,
                    target_class=site.target_class,
                    geometry=geometry,
                    camera=camera,
                    quality=clean_quality,
                    features=cast(dict[FeatureClass, object], features),  # type: ignore[arg-type]
                    relief_displacement=relief,
                    rgb_sha256=sha256_file(view_directory / "rgb.png"),
                    mask_sha256=cast(
                        dict[FeatureClass, str],
                        {
                            feature: sha256_file(view_directory / f"{feature}_mask.png")
                            for feature in masks
                        },
                    ),
                    depth_sha256=(
                        sha256_file(view_directory / "depth.tif") if depth is not None else None
                    ),
                    source_manifest_sha256=source_manifest_hash,
                )
                write_json(view_directory / "view_manifest.json", manifest.model_dump(mode="json"))
                if site.case_type != "negative" and masks[str(site.target_class)].any():
                    build_interventions_for_view(
                        view_directory,
                        target_class=site.target_class,
                        config=project.interventions,
                        depth=depth,
                    )
                view_case_type = features[site.target_class].case_type
                index_rows.append(
                    {
                        "site_id": site.site_id,
                        "view_id": identifier,
                        "variant": "clean",
                        "split": site.split,
                        "group": site.group,
                        "target_class": site.target_class,
                        "case_type": view_case_type,
                        "longitude": site.longitude,
                        "latitude": site.latitude,
                        "image_path": str((view_directory / "rgb.png").relative_to(output)),
                        "manifest_path": str(
                            (view_directory / "view_manifest.json").relative_to(output)
                        ),
                    }
                )
                quality = qualities[f"{site.site_id}/{identifier}"]
                if quality.degradation != "none":
                    degraded_directory = site_directory / f"{identifier}_degraded"
                    degraded_directory.mkdir(parents=True, exist_ok=True)
                    degraded = apply_degradation(rgb, quality)
                    Image.fromarray(degraded).save(degraded_directory / "rgb.png", optimize=True)
                    write_json(degraded_directory / "quality.json", quality.model_dump(mode="json"))
                    write_json(
                        degraded_directory / "parent.json",
                        {
                            "clean_view": f"../{identifier}",
                            "clean_rgb_sha256": manifest.rgb_sha256,
                            "degradation_only": True,
                        },
                    )
                    index_rows.append(
                        {
                            "site_id": site.site_id,
                            "view_id": f"{identifier}_degraded",
                            "variant": "degraded",
                            "split": site.split,
                            "group": site.group,
                            "target_class": site.target_class,
                            "case_type": view_case_type,
                            "longitude": site.longitude,
                            "latitude": site.latitude,
                            "image_path": str((degraded_directory / "rgb.png").relative_to(output)),
                            "manifest_path": str(
                                (view_directory / "view_manifest.json").relative_to(output)
                            ),
                        }
                    )
        # Every site contributes the full lattice: 4 altitudes x 2 geometries.
        # A site that silently rendered fewer views would quietly shrink the
        # benchmark, so it is caught here rather than at validation time.
        rendered = sum(
            row["variant"] == "clean" and row["site_id"] == site.site_id for row in index_rows
        )
        if rendered != clean_views_per_site:
            raise DataValidationError(
                f"{site.site_id} produced {rendered} clean views; the registered "
                f"lattice is {len(config.altitudes_agl_m)} altitudes x "
                f"{len(config.geometries)} geometries = {clean_views_per_site}"
            )

    clean_view_count = sum(row["variant"] == "clean" for row in index_rows)
    degraded_view_count = sum(row["variant"] == "degraded" for row in index_rows)
    if clean_view_count != expected_clean_views:
        raise DataValidationError(
            f"generated {clean_view_count} clean views; expected "
            f"{len(sites)} sites x {clean_views_per_site} = {expected_clean_views}"
        )
    if enforce_quota and clean_view_count != config.clean_view_count:
        raise DataValidationError(
            f"generated {clean_view_count} clean views against the registered "
            f"{config.clean_view_count}; the site cohort and the lattice disagree"
        )
    write_json(output / "index.json", index_rows)
    write_json(
        output / "dataset_manifest.json",
        {
            "schema_version": "4.0.0",
            "site_count": len(sites),
            "clean_view_count": clean_view_count,
            "degraded_view_count": degraded_view_count,
            # The arithmetic behind the headline count, recorded so a reader of
            # the manifest can check it without rerunning anything.
            "lattice": {
                "sites": len(sites),
                "altitudes_agl_m": list(config.altitudes_agl_m),
                "geometries": list(config.geometries),
                "clean_views_per_site": clean_views_per_site,
                "clean_view_arithmetic": (
                    f"{len(sites)} sites x {len(config.altitudes_agl_m)} altitudes x "
                    f"{len(config.geometries)} geometries = {clean_view_count} clean views"
                ),
                "degraded_view_target": config.degraded_view_count,
                "degraded_views_per_site": (degraded_view_count / len(sites) if sites else 0),
            },
            "extinction_view_count": sum(
                row["variant"] == "clean" and row["case_type"] == "extinction" for row in index_rows
            ),
            "gate_counts": dict(gate_counts),
            "quality_calibration_sha256": calibration_source_sha256,
            "config": project.model_dump(mode="json"),
        },
    )
    return output
