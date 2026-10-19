#!/usr/bin/env bash
set -euo pipefail
# Run from the repository root. Additional options are passed through unchanged.
python train.py "$@"
