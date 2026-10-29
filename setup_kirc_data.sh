#!/bin/bash
# Unpack KIRC omics into data/KIRC/data/ (expected by code/loading_data.py).
# Safe to re-run. Does not touch Git.
#
# Usage (from repo root on Triton or laptop):
#   bash setup_kirc_data.sh
#
# Prerequisites if data/KIRC/data/gene1.csv is missing:
#   - data.zip at repo root (gene1.csv, methyl.csv, miRNA1.csv inside data/)
#   - label.csv at repo root OR already at data/KIRC/label.csv

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
TARGET="${ROOT}/data/KIRC"
OMICS="${TARGET}/data"
ZIP="${ROOT}/data.zip"
LABEL_DST="${TARGET}/label.csv"
LABEL_SRC="${ROOT}/label.csv"

mkdir -p "${TARGET}" "${ROOT}/plots/KIRC" "${ROOT}/logs"

need_omics=0
for f in gene1.csv methyl.csv miRNA1.csv; do
  if [[ ! -f "${OMICS}/${f}" ]]; then
    need_omics=1
    break
  fi
done

if [[ "${need_omics}" -eq 1 ]]; then
  if [[ ! -f "${ZIP}" ]]; then
    echo "ERROR: KIRC omics missing and ${ZIP} not found."
    echo "Copy data.zip to the repo root, then re-run:"
    echo "  scp data.zip soumir1@triton.aalto.fi:${ROOT}/"
    echo "  bash setup_kirc_data.sh"
    exit 1
  fi
  echo "=== Unzipping ${ZIP} -> ${TARGET}/ ==="
  unzip -o "${ZIP}" -d "${TARGET}"
  rm -rf "${TARGET}/__MACOSX" "${ROOT}/__MACOSX" 2>/dev/null || true
fi

if [[ ! -f "${LABEL_DST}" ]]; then
  if [[ -f "${LABEL_SRC}" ]]; then
    cp "${LABEL_SRC}" "${LABEL_DST}"
    echo "=== Copied label.csv -> ${LABEL_DST} ==="
  else
    echo "ERROR: ${LABEL_DST} missing (and no ${LABEL_SRC})."
    exit 1
  fi
fi

for f in gene1.csv methyl.csv miRNA1.csv; do
  if [[ ! -f "${OMICS}/${f}" ]]; then
    echo "ERROR: expected ${OMICS}/${f} after setup."
    exit 1
  fi
done

echo "=== KIRC data ready under ${TARGET} ==="
ls -lh "${LABEL_DST}" "${OMICS}"
