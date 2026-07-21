#!/bin/sh
set -eu

ruff check .
ruff format --check .
mypy src
pytest --cov=canyonbench --cov-report=term-missing

