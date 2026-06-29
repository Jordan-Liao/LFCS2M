#!/usr/bin/env bash
set -euo pipefail

# Inference-only LFCS2M script.


INPUT_DIR="${1:-LFCS2M/data/synthetic_test}"
OUTPUT_DIR="${2:-LFCS2M/results/translated}"
CHECKPOINT="${3:-LFCS2M/weights/lfcs2m.pth}"
DEVICE="${4:-cuda:0}"
STEPS="${5:-200}"

python LFCS2M/translate.py \
  --backend standalone \
  --input_dir "${INPUT_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --checkpoint "${CHECKPOINT}" \
  --device "${DEVICE}" \
  --steps "${STEPS}"
