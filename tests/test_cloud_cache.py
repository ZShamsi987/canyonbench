from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import canyonbench.pipeline.cloud_cache as cache_module
from canyonbench.exceptions import ExternalToolError
from canyonbench.pipeline.cloud_cache import evict_cloud_files


def test_empty_eviction_is_a_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cache_module.sys, "platform", "linux")
    evict_cloud_files([])


def test_eviction_requires_macos_and_swift(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cache_module.sys, "platform", "linux")
    with pytest.raises(ExternalToolError, match="macOS"):
        evict_cloud_files(["clip.avi"])

    monkeypatch.setattr(cache_module.sys, "platform", "darwin")
    monkeypatch.setattr(cache_module.shutil, "which", lambda _: None)
    with pytest.raises(ExternalToolError, match="Swift"):
        evict_cloud_files(["clip.avi"])


def test_eviction_invokes_swift_for_each_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cache_module.sys, "platform", "darwin")
    monkeypatch.setattr(cache_module.shutil, "which", lambda _: "/usr/bin/swift")
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(cache_module.subprocess, "run", fake_run)
    paths = [tmp_path / "a.avi", tmp_path / "b.avi"]
    evict_cloud_files(paths)

    assert calls[0][0] == "/usr/bin/swift"
    assert calls[0][-2:] == [str(path) for path in paths]


def test_eviction_surfaces_provider_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cache_module.sys, "platform", "darwin")
    monkeypatch.setattr(cache_module.shutil, "which", lambda _: "/usr/bin/swift")

    def fail(command: list[str], **kwargs: object) -> None:
        raise subprocess.CalledProcessError(1, command, stderr="not cloud backed")

    monkeypatch.setattr(cache_module.subprocess, "run", fail)
    with pytest.raises(ExternalToolError, match="not cloud backed"):
        evict_cloud_files(["clip.avi"])
