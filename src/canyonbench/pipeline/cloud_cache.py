"""Optional macOS File Provider cache eviction for cloud-streamed sources."""

from __future__ import annotations

import shutil
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

from canyonbench.exceptions import ExternalToolError

SWIFT_EVICTION_PROGRAM = """
import Foundation
var failures = 0
for value in CommandLine.arguments.dropFirst() {
    do {
        try FileManager.default.evictUbiquitousItem(
            at: URL(fileURLWithPath: value)
        )
    } catch {
        failures += 1
        fputs("\\(value): \\(error)\\n", stderr)
    }
}
if failures > 0 {
    exit(1)
}
"""


def evict_cloud_files(paths: Iterable[str | Path], *, swift: str = "swift") -> None:
    """Return cloud-backed files to placeholder state without deleting originals."""

    values = [str(Path(path)) for path in paths]
    if not values:
        return
    if sys.platform != "darwin":
        raise ExternalToolError("Cloud cache eviction is currently supported only on macOS")
    executable = shutil.which(swift)
    if executable is None:
        raise ExternalToolError("Swift is required for macOS cloud cache eviction")
    try:
        subprocess.run(
            [executable, "-e", SWIFT_EVICTION_PROGRAM, *values],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or str(exc)
        raise ExternalToolError(f"Could not evict cloud source cache: {detail}") from exc
