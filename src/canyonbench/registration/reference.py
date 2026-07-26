"""Storage-bounded access to the public USGS NAIP reference service."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from canyonbench.exceptions import DataValidationError
from canyonbench.io import read_json, sha256_file, write_json

NAIP_IMAGE_SERVER = (
    "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer"
)
NAIP_EXPORT_ENDPOINT = f"{NAIP_IMAGE_SERVER}/exportImage"
NAIP_TERMS_URL = (
    "https://www.usgs.gov/faqs/what-are-terms-uselicensing-map-services-and-data-national-map"
)
DEFAULT_REFERENCE_CRS = 26912
DEFAULT_REFERENCE_YEAR = 2023
MAX_EXPORT_DIMENSION = 4000


@dataclass(frozen=True)
class ReferenceChipRequest:
    """One bounded WGS84 export request for a metric GeoTIFF."""

    west: float
    south: float
    east: float
    north: float
    width_px: int = 2000
    height_px: int = 2000
    year: int = DEFAULT_REFERENCE_YEAR
    image_crs: int = DEFAULT_REFERENCE_CRS

    def validate(self) -> None:
        """Reject invalid or unexpectedly large requests before network access."""

        if not (-180 <= self.west < self.east <= 180):
            raise DataValidationError("Reference bounds must satisfy -180 <= west < east <= 180")
        if not (-90 <= self.south < self.north <= 90):
            raise DataValidationError("Reference bounds must satisfy -90 <= south < north <= 90")
        if not (1 <= self.width_px <= MAX_EXPORT_DIMENSION):
            raise DataValidationError(
                f"Reference width must be between 1 and {MAX_EXPORT_DIMENSION} pixels"
            )
        if not (1 <= self.height_px <= MAX_EXPORT_DIMENSION):
            raise DataValidationError(
                f"Reference height must be between 1 and {MAX_EXPORT_DIMENSION} pixels"
            )
        if self.year < 2003:
            raise DataValidationError("NAIP reference year must be 2003 or later")
        if self.image_crs <= 0:
            raise DataValidationError("Reference CRS must be a positive EPSG code")

    def export_parameters(self) -> dict[str, str]:
        """Return stable ArcGIS ImageServer parameters for natural-color GeoTIFF."""

        self.validate()
        mosaic_rule = {
            "ascending": False,
            "mosaicMethod": "esriMosaicAttribute",
            "mosaicOperation": "MT_FIRST",
            "sortField": "Year",
            "sortValue": str(self.year),
            "where": f"Category = 1 AND Year = {self.year}",
        }
        return {
            "f": "json",
            "bbox": f"{self.west},{self.south},{self.east},{self.north}",
            "bboxSR": "4326",
            "imageSR": str(self.image_crs),
            "size": f"{self.width_px},{self.height_px}",
            "format": "tiff",
            "pixelType": "U8",
            "bandIds": "0,1,2",
            "interpolation": "RSP_BilinearInterpolation",
            "mosaicRule": json.dumps(mosaic_rule, separators=(",", ":"), sort_keys=True),
        }


def reference_sidecar_path(destination: str | Path) -> Path:
    """Return the provenance sidecar path for a downloaded reference chip."""

    output = Path(destination)
    return output.with_suffix(output.suffix + ".reference.json")


def _request_record(request: ReferenceChipRequest) -> dict[str, Any]:
    return {
        "bounds_crs": "EPSG:4326",
        "export_parameters": request.export_parameters(),
        **asdict(request),
    }


def _is_valid_cache(destination: Path, request: ReferenceChipRequest) -> bool:
    sidecar = reference_sidecar_path(destination)
    if not destination.is_file() or not sidecar.is_file():
        return False
    metadata = read_json(sidecar)
    if not isinstance(metadata, dict) or metadata.get("request") != _request_record(request):
        return False
    artifact = metadata.get("artifact")
    return (
        isinstance(artifact, dict)
        and artifact.get("sha256") == sha256_file(destination)
        and artifact.get("bytes") == destination.stat().st_size
    )


def _atomic_write_bytes(destination: Path, payload: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            handle.write(payload)
        os.replace(temporary_name, destination)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def fetch_reference_chip(
    request: ReferenceChipRequest,
    destination: str | Path,
    *,
    force: bool = False,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Download one bounded GeoTIFF and write a checksum/provenance sidecar."""

    output = Path(destination)
    if output.suffix.lower() not in {".tif", ".tiff"}:
        raise DataValidationError("Reference chip output must end in .tif or .tiff")
    request.validate()
    if not force and _is_valid_cache(output, request):
        cached = read_json(reference_sidecar_path(output))
        if not isinstance(cached, dict):
            raise DataValidationError("Reference cache sidecar must contain a JSON object")
        return {**cached, "cache_hit": True}
    if output.exists() and not force:
        raise DataValidationError(
            f"{output} exists but does not match this request; use --force to replace it"
        )

    owns_client = client is None
    active_client = client or httpx.Client(
        follow_redirects=True,
        timeout=httpx.Timeout(180.0, connect=30.0),
    )
    try:
        export_response = active_client.get(
            NAIP_EXPORT_ENDPOINT,
            params=request.export_parameters(),
        )
        export_response.raise_for_status()
        export_payload = export_response.json()
        if not isinstance(export_payload, dict):
            raise DataValidationError("USGS export response was not a JSON object")
        if "error" in export_payload:
            raise DataValidationError(f"USGS reference export failed: {export_payload['error']}")
        href = export_payload.get("href")
        if not isinstance(href, str) or not href.startswith("https://"):
            raise DataValidationError("USGS reference export did not return a download URL")

        image_response = active_client.get(href)
        image_response.raise_for_status()
        if not image_response.content:
            raise DataValidationError("USGS reference export returned an empty file")
        _atomic_write_bytes(output, image_response.content)
    except (httpx.HTTPError, json.JSONDecodeError) as exc:
        raise DataValidationError(f"USGS reference request failed: {exc}") from exc
    finally:
        if owns_client:
            active_client.close()

    metadata: dict[str, Any] = {
        "schema_version": "1.0.0",
        "source": {
            "provider": "U.S. Geological Survey, National Geospatial Program",
            "product": "USGS NAIP Imagery",
            "service_url": NAIP_IMAGE_SERVER,
            "terms_url": NAIP_TERMS_URL,
            "usage": "public domain",
        },
        "request": _request_record(request),
        "response": {
            "extent": export_payload.get("extent"),
            "height": export_payload.get("height"),
            "width": export_payload.get("width"),
        },
        "artifact": {
            "bytes": output.stat().st_size,
            "filename": output.name,
            "sha256": sha256_file(output),
        },
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
        "cache_hit": False,
    }
    write_json(reference_sidecar_path(output), metadata)
    return metadata
