#!/bin/bash
# Cache official national CDL archives once, so source acquisition can window
# them locally instead of depending on the unstable per-site CropScape API.
set -euo pipefail

CANYONBENCH_DATA="${CANYONBENCH_DATA:-/scratch/network/$USER/canyonbench-trace-data}"
CACHE_DIR="${CANYONBENCH_CDL_CACHE_DIR:-$CANYONBENCH_DATA/cache/cdl}"
mkdir -p "$CACHE_DIR"

for year in "$@"; do
  case "$year" in
    2021|2022|2023) ;;
    *) echo "Only registered 2021–2023 CDL archives may be cached; got $year" >&2; exit 64 ;;
  esac
  archive="$CACHE_DIR/${year}_30m_cdls.zip"
  url="https://www.nass.usda.gov/Research_and_Science/Cropland/Release/datasets/${year}_30m_cdls.zip"
  if [ -f "$archive" ]; then
    unzip -tqq "$archive"
    echo "Already verified: $archive"
    continue
  fi
  temporary="$(mktemp "$CACHE_DIR/.${year}_30m_cdls.XXXXXX")"
  echo "Downloading official $year national CDL archive..."
  curl --fail --location --retry 5 --retry-all-errors --connect-timeout 30 \
    --output "$temporary" "$url"
  unzip -tqq "$temporary"
  mv "$temporary" "$archive"
  echo "Cached: $archive"
done
