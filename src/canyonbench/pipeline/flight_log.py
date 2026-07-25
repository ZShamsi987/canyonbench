"""Recover the operational flight from noisy repeated power-cycle logs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from canyonbench.exceptions import DataValidationError

CANONICAL_ALIASES: dict[str, tuple[str, ...]] = {
    "phase": ("flightphase", "phase"),
    "elapsed_s": ("elapsedtime", "elapsed", "second", "seconds"),
    "packets": ("packets", "packet"),
    "time": ("worldviewtime", "wvtime", "time", "timestamp"),
    "lat": ("worldviewlatitude", "wvlatitude", "latitude", "lat"),
    "lon": ("worldviewlongitude", "wvlongitude", "longitude", "lon", "long"),
    "alt_m": ("worldviewaltitude", "wvaltitude", "altitude", "altitudem", "altm"),
    "speed": ("wvspeed", "speed"),
    "heading": ("wvheading", "heading"),
    "velocity_down": ("wvvelocitydown", "velocitydown", "veldown"),
    "pressure": ("wvpressure", "pressure"),
    "temperature": ("wvtemperature", "temperature", "temp"),
    "violet": ("violet",),
    "blue": ("blue",),
    "green": ("green",),
    "yellow": ("yellow",),
    "orange": ("orange",),
    "red": ("red",),
}
PHASE_ORDER = ("Launching", "Floating", "Terminating")


def _normalise_header(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())


def _header_row(path: Path) -> int:
    with path.open(encoding="utf-8", errors="replace") as handle:
        for index, line in enumerate(handle):
            normalized = _normalise_header(line)
            if "elapsedtime" in normalized and "flightphase" in normalized:
                return index
            if index >= 200:
                break
    raise DataValidationError(f"Could not locate the flight-log header in {path}")


def _canonical_columns(columns: list[Any]) -> dict[Any, str]:
    normalized = {_normalise_header(column): column for column in columns}
    renames: dict[Any, str] = {}
    for target, aliases in CANONICAL_ALIASES.items():
        for alias in aliases:
            if alias in normalized:
                renames[normalized[alias]] = target
                break
    required = {"phase", "elapsed_s", "lat", "lon", "alt_m"}
    missing = sorted(required - set(renames.values()))
    if missing:
        raise DataValidationError(f"Flight log is missing required fields: {missing}")
    return renames


def _canonical_phase(value: Any) -> str:
    raw = str(value).strip().lower()
    mapping = {
        "ground": "Ground",
        "initializing": "Initializing",
        "initialising": "Initializing",
        "launching": "Launching",
        "floating": "Floating",
        "terminating": "Terminating",
    }
    return mapping.get(raw, str(value).strip())


def _segment_ids(elapsed: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(elapsed, errors="coerce")
    resets = numeric.diff().fillna(0).lt(0)
    return resets.cumsum().astype(int)


def _contains_phase_sequence(values: pd.Series) -> bool:
    positions: list[int] = []
    normalized = [_canonical_phase(value) for value in values]
    for phase in PHASE_ORDER:
        try:
            positions.append(normalized.index(phase))
        except ValueError:
            return False
    return positions == sorted(positions)


def _load_segmented_flight(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.is_file():
        raise DataValidationError(f"Flight log does not exist: {source}")
    frame = pd.read_csv(source, sep=None, engine="python", skiprows=_header_row(source))
    frame = frame.rename(columns=_canonical_columns(list(frame.columns)))
    frame["elapsed_s"] = pd.to_numeric(frame["elapsed_s"], errors="coerce")
    frame = frame.loc[frame["elapsed_s"].notna()].copy()  # drops embedded headers
    frame["segment_id"] = _segment_ids(frame["elapsed_s"])
    frame["phase"] = frame["phase"].map(_canonical_phase)
    return frame


def audit_flight_segments(path: str | Path) -> pd.DataFrame:
    """Summarize every reset-delimited session and identify the selected flight."""

    frame = _load_segmented_flight(path)
    candidate_ids = [
        int(str(segment_id))
        for segment_id, segment in frame.groupby("segment_id", sort=True)
        if _contains_phase_sequence(segment["phase"])
    ]
    selected_id = (
        max(
            candidate_ids,
            key=lambda segment_id: (
                int(frame["segment_id"].eq(segment_id).sum()),
                segment_id,
            ),
        )
        if candidate_ids
        else None
    )
    rows: list[dict[str, Any]] = []
    for segment_id, segment in frame.groupby("segment_id", sort=True):
        lat = pd.to_numeric(segment["lat"], errors="coerce")
        lon = pd.to_numeric(segment["lon"], errors="coerce")
        valid_gps = lat.notna() & lon.notna() & lat.ne(0) & lon.ne(0)
        phases = set(segment["phase"])
        rows.append(
            {
                "segment_id": int(str(segment_id)),
                "row_count": len(segment),
                "elapsed_start_s": int(segment["elapsed_s"].min()),
                "elapsed_end_s": int(segment["elapsed_s"].max()),
                "valid_gps_rows": int(valid_gps.sum()),
                "has_launching": "Launching" in phases,
                "has_floating": "Floating" in phases,
                "has_terminating": "Terminating" in phases,
                "has_operational_sequence": _contains_phase_sequence(segment["phase"]),
                "selected_operational": int(str(segment_id)) == selected_id,
            }
        )
    return pd.DataFrame(rows)


def recover_operational_flight(path: str | Path) -> pd.DataFrame:
    """Return the final/longest contiguous full-flight segment in canonical columns."""

    frame = _load_segmented_flight(path)
    candidates = [
        segment
        for _, segment in frame.groupby("segment_id", sort=True)
        if _contains_phase_sequence(segment["phase"])
    ]
    if not candidates:
        raise DataValidationError(
            "No contiguous log segment contains Launching, Floating, and Terminating in order"
        )
    operational = max(candidates, key=lambda value: (len(value), int(value["segment_id"].iloc[0])))
    operational = operational.drop(columns="segment_id").copy()

    numeric_columns = [name for name in operational.columns if name not in {"phase", "time"}]
    for name in numeric_columns:
        operational[name] = pd.to_numeric(operational[name], errors="coerce")
    operational = operational.loc[
        operational["lat"].notna()
        & operational["lon"].notna()
        & operational["lat"].ne(0)
        & operational["lon"].ne(0)
    ].copy()
    operational["elapsed_s"] = operational["elapsed_s"].round().astype(int)
    operational = operational.drop_duplicates("elapsed_s", keep="last").sort_values("elapsed_s")
    if operational.empty:
        raise DataValidationError("Operational segment contains no valid nonzero GPS rows")
    return operational.reset_index(drop=True)
