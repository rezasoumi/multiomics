#!/bin/bash
#SBATCH --job-name=multiseed30
#SBATCH --time=96:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
# torch 2.12+cu130 has no V100 (sm_70) kernels — require A100+
#SBATCH --partition=gpu-a100-80g
#SBATCH --gpus=a100:1
#SBATCH --constraint=a100
#SBATCH --output=logs/multiseed30_%j.out
#SBATCH --error=logs/multiseed30_%j.err

set -euo pipefail

module purge
module load scicomp-pytorch-env

SCRATCH=/scratch/work/soumir1
REPO=${SCRATCH}/githubmultiomics/multiomics
export PYTHONPATH=${SCRATCH}/githubmultiomics:${SCRATCH}
export PYTHONNOUSERSITE=1

source ${SCRATCH}/multiomics/venv/bin/activate

# cwd must be under code/* so loading_data.py ../../data resolves to repo/data
mkdir -p ${REPO}/results ${REPO}/logs ${REPO}/results/multiseed_30
cd ${REPO}/code/multiseed

# Same paper-priority BRCA configs as run_multiseed.slurm, but 30 seeds (21..50).
# --resume reuses finished seed_21..seed_25 dirs from the 5-seed run.
# Per-seed artifacts: results/<models_brca_*>/<base_config>/seed_<N>/
# Aggregate (does NOT overwrite results/multiseed/): results/multiseed_30/
SEEDS=${SEEDS:-21-50}
RUNS=${RUNS:-all}
EPOCHS=${EPOCHS:-100}
BATCH=${BATCH:-32}
SUMMARY_DIR=${SUMMARY_DIR:-${REPO}/results/multiseed_30}

echo "=== Host: $(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} ==="
python -u -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -u -m multiomics.code.multiseed.run --list-runs

echo "=== Multi-seed train+eval: runs=${RUNS} seeds=${SEEDS} epochs=${EPOCHS} ==="
echo "=== Summary dir: ${SUMMARY_DIR} ==="

python -u -m multiomics.code.multiseed.run \
  --runs "${RUNS}" \
  --seeds "${SEEDS}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH}" \
  --summary-dir "${SUMMARY_DIR}" \
  --resume

echo "=== Done ==="
echo "Fetch to laptop (from repo root):"
echo "  bash fetch_multiseed_30.sh"
echo "Or:"
echo "  rsync -avz triton_via_kosh:${SUMMARY_DIR}/ results/multiseed_30/"
echo "Server files:"
echo "  ${SUMMARY_DIR}/summary.json"
echo "  ${SUMMARY_DIR}/readme_snippets.md"
echo "  ${SUMMARY_DIR}/summary_flat.csv"
