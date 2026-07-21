from __future__ import annotations

from pathlib import Path

import pytest

from canyonbench.exceptions import DataValidationError
from canyonbench.io import iter_jsonl, read_json, sha256_file, write_json, write_jsonl


def test_json_helpers_and_hash(tmp_path: Path) -> None:
    path = tmp_path / "value.json"
    write_json(path, {"b": 2, "a": 1})
    assert read_json(path) == {"a": 1, "b": 2}
    assert len(sha256_file(path)) == 64
    lines = tmp_path / "rows.jsonl"
    write_jsonl(lines, [{"x": 1}, {"x": 2}])
    assert list(iter_jsonl(lines)) == [{"x": 1}, {"x": 2}]


def test_invalid_jsonl_reports_line(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("{}\n{broken}\n", encoding="utf-8")
    with pytest.raises(DataValidationError, match=":2"):
        list(iter_jsonl(path))
