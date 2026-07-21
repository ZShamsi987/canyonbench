#!/bin/sh
set -eu

RELEASE_DIR="${1:?usage: scripts/release.sh RELEASE_DIR}"
canyonbench validate-release "$RELEASE_DIR"
python -m build
python -m twine check dist/*

