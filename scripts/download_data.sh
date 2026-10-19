#!/usr/bin/env bash
# Downloads the raw UCI Heart Disease (Cleveland) file and writes the cleaned csv.
# Usage: bash scripts/download_data.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHONPATH=src python -m heart.data "$@"
