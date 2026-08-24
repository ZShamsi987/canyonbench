#!/usr/bin/env python3
"""Fan the metered roster out across processes without multiplying the spend cap.

WHY.  ``run_trace`` issues one request at a time. The metered roster is six
models, and at roughly three seconds a call the API half of the run is on the
order of a day of wall clock. The models are independent, so the fix does not
need a concurrency refactor of the paid execution path: ``--only-model`` and
``--output-dir`` already exist, the Lambda driver already uses them, and
``trace merge-runs`` already recombines the results. Each model becomes its own
process with its own predictions log.

THE HAZARD THIS EXISTS TO CLOSE.  ``BudgetTracker`` is constructed per process
from ``config.budget``, so six concurrent runs would each independently permit
the full ``max_cost_usd`` and ``max_requests``. A $250 cap silently becomes a
$1,500 ceiling. This script partitions the cap instead: every model gets its own
share, sized from the measured call plan, and the shares are asserted to sum to
no more than the global cap before a single process starts.

A model that exhausts its share stops, exactly as it would today; it cannot
borrow from another model's allowance. That is deliberate. A shared live
allowance across processes cannot be enforced without a shared lock, and a cap
that depends on cross-process coordination is not a cap.

    python scripts/adroit/run_openrouter_parallel.py configs/trace_run.frozen.yaml \\
        --output-root "$CANYONBENCH_DATA/runs/openrouter" --margin 1.35 --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from canyonbench.trace.config import load_trace_run_config
from canyonbench.trace.planning import estimate_call_plan


def _shares(config: Any, margin: float) -> dict[str, dict[str, float]]:
    """Split the global caps across metered models by their measured call plan."""

    plan = estimate_call_plan(config)
    calls = plan["calls_by_model"]
    metered = [model for model in config.models if model.metered]
    if not metered:
        raise SystemExit("roster contains no metered models")

    planned = {model.id: max(1, int(calls.get(model.id, 0))) for model in metered}

    # Cost share follows each model's own price, not its call count: one model at
    # $30/M output can cost more in a tenth of the calls.
    def unit_cost(model: Any) -> float:
        return float(model.input_per_million_usd or 0.0) + float(
            model.output_per_million_usd or 0.0
        )

    weights = {model.id: planned[model.id] * max(unit_cost(model), 1e-9) for model in metered}
    weight_total = sum(weights.values()) or 1.0

    out: dict[str, dict[str, float]] = {}
    for model in metered:
        out[model.id] = {
            "max_requests": int(planned[model.id] * margin) + 1,
            "max_cost_usd": config.budget.max_cost_usd * (weights[model.id] / weight_total),
            "planned_calls": planned[model.id],
        }

    request_sum = sum(int(v["max_requests"]) for v in out.values())
    cost_sum = sum(float(v["max_cost_usd"]) for v in out.values())
    # The whole point of this script. Refuse to launch if the partition would let
    # the fleet outspend the single-process cap it replaces.
    if cost_sum > config.budget.max_cost_usd + 1e-6:
        raise SystemExit(
            f"cost shares sum to ${cost_sum:.2f}, over the "
            f"${config.budget.max_cost_usd:.2f} global cap"
        )
    if request_sum > config.budget.max_requests:
        raise SystemExit(
            f"request shares sum to {request_sum} over cap {config.budget.max_requests}; "
            "lower --margin or raise the cap deliberately"
        )
    print(
        f"partitioned ${cost_sum:.2f} of ${config.budget.max_cost_usd:.2f} and "
        f"{request_sum} of {config.budget.max_requests} requests across {len(out)} models"
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--margin",
        type=float,
        default=1.35,
        help="Request headroom over the planned call count, for parse retries.",
    )
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()

    config = load_trace_run_config(arguments.config)
    shares = _shares(config, arguments.margin)
    raw = yaml.safe_load(arguments.config.read_text())
    arguments.output_root.mkdir(parents=True, exist_ok=True)

    commands: list[tuple[str, list[str]]] = []
    for model_id, share in sorted(shares.items()):
        slug = model_id.replace("/", "__").replace(".", "_")
        run_dir = arguments.output_root / slug
        run_dir.mkdir(parents=True, exist_ok=True)
        scoped = dict(raw)
        scoped["budget"] = dict(raw["budget"])
        scoped["budget"]["max_requests"] = int(share["max_requests"])
        scoped["budget"]["max_cost_usd"] = round(float(share["max_cost_usd"]), 4)
        config_path = run_dir / "run.yaml"
        config_path.write_text(yaml.safe_dump(scoped, sort_keys=False), encoding="utf-8")
        commands.append(
            (
                model_id,
                [
                    "canyonbench", "trace", "run", str(config_path),
                    "--only-model", model_id,
                    "--output-dir", str(run_dir),
                ],
            )
        )
        print(
            f"  {model_id:40s} {int(share['planned_calls']):6d} planned  "
            f"cap ${share['max_cost_usd']:7.2f} / {int(share['max_requests'])} req  -> {run_dir}"
        )

    if arguments.dry_run:
        print("\ndry run; nothing launched. Commands:")
        for _, command in commands:
            print("  " + " ".join(command))
        return

    processes = []
    for model_id, command in commands:
        log = arguments.output_root / f"{model_id.replace('/', '__')}.log"
        handle = log.open("w", encoding="utf-8")
        process = subprocess.Popen(command, stdout=handle, stderr=handle)
        processes.append((model_id, process, handle))
        print(f"launched {model_id}")

    failures = []
    for model_id, process, handle in processes:
        code = process.wait()
        handle.close()
        if code != 0:
            failures.append((model_id, code))
            print(f"FAILED {model_id} exit {code}", file=sys.stderr)

    print(f"\n{len(processes) - len(failures)}/{len(processes)} models completed")
    print(f"Merge with: canyonbench trace merge-runs {arguments.output_root}/*/predictions.jsonl")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
