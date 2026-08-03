#!/bin/sh
set -eu

uv lock --check
uv run --frozen --extra trace --extra dev ruff check .
uv run --frozen --extra trace --extra dev ruff format --check .
uv run --frozen --extra trace --extra dev mypy src
uv run --frozen --extra trace --extra dev pytest --cov=canyonbench --cov-report=term-missing
