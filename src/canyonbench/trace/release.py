"""Public-development release and reserved-test escrow construction."""

from __future__ import annotations

import shutil
from pathlib import Path

from canyonbench.exceptions import DataValidationError
from canyonbench.io import read_json, sha256_file, write_json
from canyonbench.trace.config import load_source_manifest
from canyonbench.trace.schemas import (
    PrivateTestEscrowManifest,
    PublicReleaseManifest,
    ReleaseFile,
)


def _copy(path: Path, source_root: Path, destination_root: Path) -> None:
    relative = path.relative_to(source_root)
    destination = destination_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)


def _inventory(root: Path) -> list[ReleaseFile]:
    return [
        ReleaseFile(
            path=str(path.relative_to(root)),
            sha256=sha256_file(path),
            size_bytes=path.stat().st_size,
        )
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "release_manifest.json"
    ]


def validate_built_release(public_root: Path, escrow_root: Path) -> dict[str, int]:
    """Validate release schemas, inventory hashes, and the private boundary."""

    public = PublicReleaseManifest.model_validate(read_json(public_root / "release_manifest.json"))
    escrow = PrivateTestEscrowManifest.model_validate(
        read_json(escrow_root / "held_out_hashes.json")
    )
    expected_paths = {record.path for record in public.files}
    actual_paths = {
        str(path.relative_to(public_root))
        for path in public_root.rglob("*")
        if path.is_file() and path.name != "release_manifest.json"
    }
    if expected_paths != actual_paths:
        raise DataValidationError(
            "Public release inventory differs from disk: "
            f"missing={sorted(expected_paths - actual_paths)}, "
            f"untracked={sorted(actual_paths - expected_paths)}"
        )
    for record in public.files:
        relative = Path(record.path)
        if relative.is_absolute() or ".." in relative.parts:
            raise DataValidationError(f"Unsafe path in public release inventory: {record.path}")
        path = public_root / relative
        if sha256_file(path) != record.sha256 or path.stat().st_size != record.size_bytes:
            raise DataValidationError(
                f"Public release artifact failed hash/size validation: {record.path}"
            )
    return {
        "public_files": len(public.files),
        "public_views": public.view_count,
        "public_sites": public.site_count,
        "escrow_records": escrow.record_count,
    }


def build_release(
    dataset_dir: Path,
    output_dir: Path,
    *,
    public_splits: tuple[str, ...] = ("development", "validation"),
) -> tuple[Path, Path]:
    """Copy releasable bundles and write hash-only escrow for held-out test views."""

    if output_dir.exists() and any(output_dir.iterdir()):
        raise DataValidationError(
            f"Release output must be absent or empty to prevent stale files: {output_dir}"
        )
    index = read_json(dataset_dir / "index.json")
    public_rows = [row for row in index if row["split"] in public_splits]
    held_out = [row for row in index if row["split"] not in public_splits]
    blocked_sources: list[str] = []
    for site in sorted({str(row["site_id"]) for row in public_rows}):
        manifest = load_source_manifest(dataset_dir / site / "source_manifest.json")
        records = [
            manifest.imagery,
            *[
                record
                for feature_records in manifest.feature_sources.values()
                for record in feature_records
            ],
            *manifest.detector_sources.values(),
        ]
        if manifest.terrain is not None:
            records.append(manifest.terrain)
        blocked_sources.extend(
            f"{site}/{record.source_id}:{record.redistribution}"
            for record in records
            if record.redistribution != "allowed"
        )
    if blocked_sources:
        raise DataValidationError(
            "Public release is blocked until every included source has "
            "redistribution=allowed: " + ", ".join(blocked_sources)
        )
    public_root = output_dir / "public"
    escrow_root = output_dir / "escrow"
    public_root.mkdir(parents=True, exist_ok=True)
    escrow_root.mkdir(parents=True, exist_ok=True)
    copied_sites: set[str] = set()
    for row in public_rows:
        image = dataset_dir / row["image_path"]
        view_dir = image.parent
        if row["variant"] == "clean":
            for path in view_dir.rglob("*"):
                if path.is_file():
                    _copy(path, dataset_dir, public_root)
        else:
            for path in view_dir.glob("*"):
                if path.is_file():
                    _copy(path, dataset_dir, public_root)
        site = str(row["site_id"])
        if site not in copied_sites:
            for name in ("source_manifest.json", "gate_results.json"):
                _copy(dataset_dir / site / name, dataset_dir, public_root)
            copied_sites.add(site)
    write_json(public_root / "index.json", public_rows)
    public_manifest = PublicReleaseManifest(
        splits=list(public_splits),  # type: ignore[arg-type]
        site_count=len({row["site_id"] for row in public_rows}),
        view_count=len(public_rows),
        input_index_sha256=sha256_file(dataset_dir / "index.json"),
        input_dataset_manifest_sha256=sha256_file(dataset_dir / "dataset_manifest.json"),
        files=_inventory(public_root),
    )
    write_json(
        public_root / "release_manifest.json",
        public_manifest.model_dump(mode="json"),
    )
    escrow_rows = []
    for row in held_out:
        image = dataset_dir / row["image_path"]
        escrow_rows.append(
            {
                "opaque_id": sha256_file(image)[:24],
                "variant": row["variant"],
                "image_sha256": sha256_file(image),
                "manifest_sha256": sha256_file(dataset_dir / row["manifest_path"]),
            }
        )
    escrow_manifest = PrivateTestEscrowManifest(
        record_count=len(escrow_rows),
        records=escrow_rows,  # type: ignore[arg-type]
    )
    write_json(
        escrow_root / "held_out_hashes.json",
        escrow_manifest.model_dump(mode="json"),
    )
    validate_built_release(public_root, escrow_root)
    return public_root, escrow_root
