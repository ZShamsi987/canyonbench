from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from PIL import Image

from canyonbench.eval.adapters import FixtureAdapter
from canyonbench.eval.runner import run
from canyonbench.io import iter_jsonl
from canyonbench.schemas import AdapterConfig, BudgetConfig, ModelConfig, RunConfig


def test_fixture_run_is_resumable(tmp_path: Path) -> None:
    release = tmp_path / "release"
    frames_dir = release / "frames"
    frames_dir.mkdir(parents=True)
    image_path = frames_dir / "img_006806.jpg"
    Image.new("RGB", (8, 8), "brown").save(image_path)
    pd.DataFrame(
        [
            {
                "image": image_path.name,
                "image_path": str(image_path),
                "elapsed_s": 6806,
                "segment_id": "s1",
                "split": "test",
                "registration_reliable": False,
            }
        ]
    ).to_csv(release / "frames.csv", index=False)
    probes = tmp_path / "probes.yaml"
    probes.write_text(
        "schema_version: 1\nprobes:\n"
        "  - name: presence\n    variant: neutral\n    system: inspect\n    prompt: classify\n",
        encoding="utf-8",
    )
    model = ModelConfig(id="fixture", adapter=AdapterConfig(kind="fixture"))
    config = RunConfig(
        release_dir=release,
        output_dir=tmp_path / "output",
        models=[model],
        probes_file=probes,
        probes=["presence"],
        budget=BudgetConfig(max_requests=2, max_cost_usd=0),
    )
    response = json.dumps(
        {"water": "no", "road": "no", "building": "no", "forest": "no", "snow": "no", "field": "no"}
    )
    adapter = FixtureAdapter({"classify": response})
    predictions = run(config, adapters={"fixture": adapter})
    assert len(list(iter_jsonl(predictions))) == 1
    run(config, adapters={"fixture": adapter})
    assert len(list(iter_jsonl(predictions))) == 1
    assert (config.output_dir / "run_manifest.json").exists()
