"""Convert per-clip extraction outputs into lossless flight-second filenames."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pandas as pd

from canyonbench.exceptions import DataValidationError
from canyonbench.pipeline.sync import SyncAnchor, clip_flight_start

EXTRACTED_PATTERN = re.compile(r"^clip_(\d{4})_(\d+)\.jpg$")


def plan_frame_names(
    extracted_dir: str | Path, clips: pd.DataFrame, anchor: SyncAnchor
) -> pd.DataFrame:
    root = Path(extracted_dir)
    rows: list[dict[str, object]] = []
    by_index = {int(row["clip_index"]): row for _, row in clips.iterrows()}
    for path in sorted(root.glob("clip_*.jpg")):
        match = EXTRACTED_PATTERN.match(path.name)
        if match is None:
            continue
        clip_index, output_index = (int(value) for value in match.groups())
        if clip_index not in by_index:
            raise DataValidationError(f"Extracted frame refers to unknown clip index: {path.name}")
        # ffmpeg image sequences begin at 1; the first sampled second is offset 0.
        flight_second = round(clip_flight_start(by_index[clip_index], anchor) + output_index - 1)
        if flight_second < 0:
            continue
        rows.append(
            {
                "source": str(path),
                "image": f"img_{flight_second:06d}.jpg",
                "elapsed_s": flight_second,
                "clip_index": clip_index,
                "clip_frame_index": output_index,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    duplicated = frame[frame.duplicated("elapsed_s", keep=False)]
    if not duplicated.empty:
        # Clip overlap is possible. Keep the later clip deterministically but expose the count.
        frame = frame.sort_values(["elapsed_s", "clip_index"]).drop_duplicates(
            "elapsed_s", keep="last"
        )
    return frame.sort_values("elapsed_s").reset_index(drop=True)


def materialize_frame_names(plan: pd.DataFrame, destination: str | Path) -> None:
    output = Path(destination)
    output.mkdir(parents=True, exist_ok=True)
    for row in plan.to_dict(orient="records"):
        target = output / str(row["image"])
        if target.exists():
            raise DataValidationError(f"Refusing to overwrite existing frame: {target}")
        shutil.copy2(str(row["source"]), target)
