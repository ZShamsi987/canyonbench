"""Registered compute and storage contract for the two-resource split.

Dataset construction is CPU-bound and runs on Adroit; only model inference needs
VRAM and runs on Lambda; proprietary inference needs neither and is driven over
the OpenRouter API from any host. This module encodes the registered hardware
requirements, the serving profile selection, and the storage layout so the same
driver script runs unmodified across the admissible instance types.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

LAMBDA_ROOT: Final[Path] = Path("/lambda/canyonbench")

# Quantization changes model behavior, and a benchmark whose subject is causal
# faithfulness cannot report a quantized model under the named model's identity.
REQUIRED_DTYPE: Final[str] = "bfloat16"
# Volta and older lack native bfloat16; Ampere (sm_80) is the floor.
MINIMUM_CUDA_CAPABILITY: Final[tuple[int, int]] = (8, 0)


@dataclass(frozen=True)
class ModelClass:
    """Registered VRAM and host-memory floor for one served model class."""

    name: str
    weights_bf16_gb: float
    minimum_vram_gb: int
    recommended_vram_gb: int
    system_ram_gb: int
    note: str
    required: bool = True


MODEL_CLASSES: Final[tuple[ModelClass, ...]] = (
    ModelClass(
        name="vlm_7_8b",
        weights_bf16_gb=16,
        minimum_vram_gb=24,
        recommended_vram_gb=40,
        system_ram_gb=64,
        note="40 GB permits a large KV cache and high batch concurrency",
    ),
    ModelClass(
        name="vlm_12_14b",
        weights_bf16_gb=28,
        minimum_vram_gb=40,
        recommended_vram_gb=80,
        system_ram_gb=64,
        note="fits 40 GB with reduced max_num_seqs",
    ),
    ModelClass(
        name="vlm_26_34b",
        weights_bf16_gb=68,
        minimum_vram_gb=80,
        recommended_vram_gb=80,
        system_ram_gb=128,
        note="requires an 80 GB card; will not fit 40 GB in bfloat16",
    ),
    ModelClass(
        name="vlm_70b_plus",
        weights_bf16_gb=140,
        minimum_vram_gb=160,
        recommended_vram_gb=160,
        system_ram_gb=256,
        note="tensor parallelism required; outside the planned model set",
        required=False,
    ),
    ModelClass(
        name="detector_or_segmenter",
        weights_bf16_gb=8,
        minimum_vram_gb=16,
        recommended_vram_gb=24,
        system_ram_gb=32,
        note="scheduled alongside VLM work rather than in a separate session",
    ),
)


@dataclass(frozen=True)
class Instance:
    """One offered Lambda configuration assessed against the requirements."""

    name: str
    gpus: int
    vram_gb_per_gpu: int
    vcpu: int
    ram_gib: int
    ssd_tib: float
    verdict: str
    assessment: str


INSTANCES: Final[tuple[Instance, ...]] = (
    Instance(
        name="1x A100 40 GB SXM4",
        gpus=1,
        vram_gb_per_gpu=40,
        vcpu=30,
        ram_gib=200,
        ssd_tib=0.5,
        verdict="primary",
        assessment=(
            "Sufficient for all 7-14B models with substantial KV cache headroom; "
            "broadest availability of the suitable options."
        ),
    ),
    Instance(
        name="1x H100 80 GB PCIe",
        gpus=1,
        vram_gb_per_gpu=80,
        vcpu=26,
        ram_gib=200,
        ssd_tib=1.0,
        verdict="secondary",
        assessment=(
            "Required for 26-34B models and the preferred fallback when 40 GB "
            "A100 capacity is unavailable."
        ),
    ),
    Instance(
        name="1x H100 80 GB SXM5",
        gpus=1,
        vram_gb_per_gpu=80,
        vcpu=26,
        ram_gib=225,
        ssd_tib=2.8,
        verdict="substitute",
        assessment="Equivalent capability to the PCIe variant; acceptable substitute.",
    ),
    Instance(
        name="1x A10 24 GB PCIe",
        gpus=1,
        vram_gb_per_gpu=24,
        vcpu=30,
        ram_gib=200,
        ssd_tib=1.4,
        verdict="fallback",
        assessment=(
            "Acceptable only for 7-8B models with reduced batch concurrency, and "
            "for the detector or segmentation workloads."
        ),
    ),
    Instance(
        name="2x H100 80 GB SXM5",
        gpus=2,
        vram_gb_per_gpu=80,
        vcpu=52,
        ram_gib=450,
        ssd_tib=5.5,
        verdict="not_required",
        assessment="Justified only if a 70B-class model is added to the roster.",
    ),
    Instance(
        name="4x H100 80 GB SXM5",
        gpus=4,
        vram_gb_per_gpu=80,
        vcpu=104,
        ram_gib=900,
        ssd_tib=11.0,
        verdict="not_recommended",
        assessment="Exceeds requirements.",
    ),
    Instance(
        name="8x A100 80 GB SXM4",
        gpus=8,
        vram_gb_per_gpu=80,
        vcpu=240,
        ram_gib=1800,
        ssd_tib=20.0,
        verdict="not_recommended",
        assessment="Exceeds requirements.",
    ),
    Instance(
        name="8x A100 40 GB SXM4",
        gpus=8,
        vram_gb_per_gpu=40,
        vcpu=124,
        ram_gib=1800,
        ssd_tib=6.0,
        verdict="not_recommended",
        assessment=(
            "Exceeds requirements; the workload is request-parallel and cannot use "
            "eight devices concurrently without restructuring."
        ),
    ),
    Instance(
        name="8x Tesla V100 16 GB",
        gpus=8,
        vram_gb_per_gpu=16,
        vcpu=88,
        ram_gib=448,
        ssd_tib=5.9,
        verdict="excluded",
        assessment=(
            "Volta lacks native bfloat16 and vLLM vision-model support is limited; "
            "violates the precision constraint regardless of nominal VRAM."
        ),
    ),
)


@dataclass(frozen=True)
class ServingProfile:
    """Capability-adaptive vLLM configuration for the detected device."""

    vram_gb: float
    max_model_len: int
    max_num_seqs: int
    dtype: str = REQUIRED_DTYPE
    gpu_memory_utilization: float = 0.90
    images_per_prompt: int = 1

    def as_vllm_kwargs(self, *, download_dir: str | None = None) -> dict[str, Any]:
        values: dict[str, Any] = {
            "dtype": self.dtype,
            "trust_remote_code": True,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "limit_mm_per_prompt": {"image": self.images_per_prompt},
            "max_model_len": self.max_model_len,
            "max_num_seqs": self.max_num_seqs,
        }
        if download_dir:
            values["download_dir"] = download_dir
        return values

    def as_server_args(self) -> list[str]:
        """Flags for `vllm.entrypoints.openai.api_server`, mirroring the kwargs."""

        return [
            "--dtype",
            self.dtype,
            "--trust-remote-code",
            "--gpu-memory-utilization",
            f"{self.gpu_memory_utilization}",
            "--limit-mm-per-prompt",
            f"image={self.images_per_prompt}",
            "--max-model-len",
            str(self.max_model_len),
            "--max-num-seqs",
            str(self.max_num_seqs),
        ]


def serving_profile(vram_gb: float) -> ServingProfile:
    """Select the registered serving profile from detected device memory."""

    if vram_gb <= 0:
        raise ValueError("device memory must be positive")
    if vram_gb < 30:
        return ServingProfile(vram_gb=vram_gb, max_model_len=8192, max_num_seqs=16)
    if vram_gb < 60:
        return ServingProfile(vram_gb=vram_gb, max_model_len=16384, max_num_seqs=64)
    return ServingProfile(vram_gb=vram_gb, max_model_len=32768, max_num_seqs=128)


def model_class(name: str) -> ModelClass:
    for entry in MODEL_CLASSES:
        if entry.name == name:
            return entry
    raise KeyError(f"unknown model class {name!r}")


def fits(class_name: str, vram_gb: float, *, gpus: int = 1) -> bool:
    """Whether one instance can serve a model class in bfloat16."""

    required = model_class(class_name).minimum_vram_gb
    # Only a model larger than a single card benefits from a second device.
    usable = vram_gb * gpus if required > vram_gb else vram_gb
    return usable >= required


def admissible_instances(class_names: list[str]) -> list[dict[str, Any]]:
    """Rank the offered instances against a set of model classes."""

    rows: list[dict[str, Any]] = []
    for instance in INSTANCES:
        supported = [
            name for name in class_names if fits(name, instance.vram_gb_per_gpu, gpus=instance.gpus)
        ]
        rows.append(
            {
                "instance": instance.name,
                "gpus": instance.gpus,
                "vram_gb_per_gpu": instance.vram_gb_per_gpu,
                "verdict": instance.verdict,
                "assessment": instance.assessment,
                "bfloat16_capable": instance.verdict != "excluded",
                "serves": supported if instance.verdict != "excluded" else [],
                "unserved": (
                    [name for name in class_names if name not in supported]
                    if instance.verdict != "excluded"
                    else list(class_names)
                ),
            }
        )
    return rows


@dataclass(frozen=True)
class StorageLayout:
    """Persistent Lambda filesystem layout, decoupled from instance lifetime."""

    root: Path = LAMBDA_ROOT

    @property
    def hf_home(self) -> Path:
        return self.root / "hf"

    @property
    def dataset(self) -> Path:
        return self.root / "dataset"

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def directories(self) -> tuple[Path, ...]:
        return (self.root, self.hf_home, self.dataset, self.results, self.logs)

    def environment(self) -> dict[str, str]:
        return {
            "HF_HOME": str(self.hf_home),
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "CANYONBENCH_DATASET_DIR": str(self.dataset),
            "CANYONBENCH_RESULTS_DIR": str(self.results),
        }

    def create(self) -> None:
        for directory in self.directories:
            directory.mkdir(parents=True, exist_ok=True)


# Registered capacity plan; source tiles stay on Adroit and are never transferred.
STORAGE_ESTIMATE_GB: Final[dict[str, tuple[int, int]]] = {
    "adroit_source_tiles": (100, 300),
    "lambda_dataset_bundle": (20, 40),
    "lambda_model_weights": (100, 150),
    "lambda_results_and_logs": (1, 1),
}
LAMBDA_TOTAL_ESTIMATE_GB: Final[int] = 200


def _device_report() -> dict[str, Any]:
    """Detect CUDA devices without making torch a hard import for CPU hosts."""

    try:
        import torch  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return {"torch_available": False, "cuda_available": False, "devices": []}
    if not torch.cuda.is_available():
        return {"torch_available": True, "cuda_available": False, "devices": []}
    devices: list[dict[str, Any]] = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        capability = (properties.major, properties.minor)
        devices.append(
            {
                "index": index,
                "name": properties.name,
                "vram_gb": round(properties.total_memory / 1e9, 1),
                "cuda_capability": f"{properties.major}.{properties.minor}",
                "bfloat16_native": capability >= MINIMUM_CUDA_CAPABILITY,
            }
        )
    return {"torch_available": True, "cuda_available": True, "devices": devices}


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    blocking: bool = True
    data: dict[str, Any] = field(default_factory=dict)


def compute_check(
    *,
    role: str,
    storage_root: Path | None = None,
    dataset_dir: Path | None = None,
    required_env: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Go/no-go preflight for an Adroit CPU host or a Lambda GPU instance.

    ``role`` is ``adroit``, ``lambda``, or ``openrouter``. Only the checks that
    apply to that role are blocking, so one command serves all three hosts.
    """

    if role not in {"adroit", "lambda", "openrouter"}:
        raise ValueError("role must be adroit, lambda, or openrouter")
    checks: list[CheckResult] = []

    for name in required_env:
        checks.append(
            CheckResult(
                name=f"env:{name}",
                ok=bool(os.environ.get(name)),
                detail=(
                    f"{name} is set" if os.environ.get(name) else f"{name} is not set in this shell"
                ),
            )
        )

    for executable in ("ffmpeg", "gdalinfo"):
        found = shutil.which(executable)
        checks.append(
            CheckResult(
                name=f"tool:{executable}",
                ok=found is not None,
                detail=found or f"{executable} not on PATH (optional)",
                blocking=False,
            )
        )

    if role == "lambda":
        layout = StorageLayout(storage_root or LAMBDA_ROOT)
        writable = True
        detail = f"{layout.root} is writable"
        try:
            layout.create()
            probe = layout.logs / ".compute_check"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
        except OSError as error:
            writable = False
            detail = f"{layout.root} is not writable: {error}"
        checks.append(
            CheckResult(
                name="storage:persistent_filesystem",
                ok=writable,
                detail=detail,
                data={
                    "directories": [str(path) for path in layout.directories],
                    "environment": layout.environment(),
                    "estimate_gb": LAMBDA_TOTAL_ESTIMATE_GB,
                },
            )
        )
        if writable:
            usage = shutil.disk_usage(layout.root)
            free_gb = usage.free / 1e9
            checks.append(
                CheckResult(
                    name="storage:free_capacity",
                    ok=free_gb >= LAMBDA_TOTAL_ESTIMATE_GB,
                    detail=(f"{free_gb:.0f} GB free against a {LAMBDA_TOTAL_ESTIMATE_GB} GB plan"),
                    data={"free_gb": round(free_gb, 1)},
                )
            )

        devices = _device_report()
        cuda_ok = bool(devices["devices"])
        checks.append(
            CheckResult(
                name="gpu:cuda_devices",
                ok=cuda_ok,
                detail=(
                    ", ".join(
                        f"{device['name']} ({device['vram_gb']} GB, cc {device['cuda_capability']})"
                        for device in devices["devices"]
                    )
                    if cuda_ok
                    else "no CUDA device visible"
                ),
                data=devices,
            )
        )
        if cuda_ok:
            native = all(device["bfloat16_native"] for device in devices["devices"])
            checks.append(
                CheckResult(
                    name="gpu:bfloat16_native",
                    ok=native,
                    detail=(
                        f"all devices support native {REQUIRED_DTYPE}"
                        if native
                        else "a device lacks native bfloat16; the precision "
                        "constraint forbids serving the roster here"
                    ),
                )
            )
            profile = serving_profile(min(device["vram_gb"] for device in devices["devices"]))
            checks.append(
                CheckResult(
                    name="gpu:serving_profile",
                    ok=True,
                    detail=(
                        f"max_model_len={profile.max_model_len} "
                        f"max_num_seqs={profile.max_num_seqs} dtype={profile.dtype}"
                    ),
                    data={
                        "profile": profile.as_vllm_kwargs(),
                        "server_args": profile.as_server_args(),
                    },
                )
            )
        try:
            import vllm  # type: ignore[import-not-found]  # noqa: F401

            vllm_ok, vllm_detail = True, "vllm importable"
        except ModuleNotFoundError:
            vllm_ok, vllm_detail = False, "vllm is not installed in this environment"
        checks.append(CheckResult(name="gpu:vllm", ok=vllm_ok, detail=vllm_detail))

    if dataset_dir is not None:
        index = dataset_dir / "index.json"
        checks.append(
            CheckResult(
                name="dataset:index",
                ok=index.is_file(),
                detail=str(index) if index.is_file() else f"missing {index}",
            )
        )

    blocking_failures = [check for check in checks if check.blocking and not check.ok]
    return {
        "schema_version": "4.2.0",
        "role": role,
        "ready": not blocking_failures,
        "blocking_failures": [check.name for check in blocking_failures],
        "checks": [
            {
                "name": check.name,
                "ok": check.ok,
                "blocking": check.blocking,
                "detail": check.detail,
                **({"data": check.data} if check.data else {}),
            }
            for check in checks
        ],
    }
