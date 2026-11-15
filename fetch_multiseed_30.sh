#!/bin/bash
# Pull 30-seed multiseed aggregates from Triton into results/multiseed_30/
# (keeps the existing 5-seed results/multiseed/ untouched).
#
# Usage (from laptop repo root):
#   bash fetch_multiseed_30.sh
#   HOST=triton_via_kosh bash fetch_multiseed_30.sh
#
# Optional: also pull per-seed metrics JSON (small) for local re-aggregate:
#   WITH_METRICS=1 bash fetch_multiseed_30.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
HOST="${HOST:-triton_via_kosh}"
REMOTE_REPO="${REMOTE_REPO:-/scratch/work/soumir1/githubmultiomics/multiomics}"
REMOTE_SUMMARY="${REMOTE_REPO}/results/multiseed_30"
LOCAL_SUMMARY="${ROOT}/results/multiseed_30"

mkdir -p "${LOCAL_SUMMARY}"

echo "=== Fetch aggregates: ${HOST}:${REMOTE_SUMMARY}/ → ${LOCAL_SUMMARY}/ ==="
rsync -avz --progress \
  "${HOST}:${REMOTE_SUMMARY}/" \
  "${LOCAL_SUMMARY}/"

if [[ "${WITH_METRICS:-0}" == "1" ]]; then
  echo "=== Fetch per-seed metrics_multiseed.json under models_brca_* ==="
  # Only JSON metrics (not model weights).
  rsync -avz --progress \
    --include='*/' \
    --include='metrics_multiseed.json' \
    --exclude='*' \
    "${HOST}:${REMOTE_REPO}/results/" \
    "${ROOT}/results/" \
    --prune-empty-dirs
fi

echo "=== Done ==="
ls -la "${LOCAL_SUMMARY}"
