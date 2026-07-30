#!/bin/sh
set -eu

uv lock --check
ruff check .
ruff format --check .
mypy src
pytest --cov=canyonbench --cov-report=term-missing
