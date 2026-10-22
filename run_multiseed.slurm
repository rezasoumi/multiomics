#!/bin/bash
#SBATCH --job-name=multiseed
#SBATCH --time=96:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
# torch 2.12+cu130 has no V100 (sm_70) kernels — require A100+
#SBATCH --partition=gpu-a100-80g
#SBATCH --gpus=a100:1
#SBATCH --constraint=a100
#SBATCH --output=logs/multiseed_%j.out
#SBATCH --error=logs/multiseed_%j.err

set -euo pipefail

module purge
module load scicomp-pytorch-env

SCRATCH=/scratch/work/soumir1
REPO=${SCRATCH}/githubmultiomics/multiomics
export PYTHONPATH=${SCRATCH}/githubmultiomics:${SCRATCH}

source ${SCRATCH}/multiomics/venv/bin/activate

# cwd must be under code/* so loading_data.py ../../data resolves to repo/data
mkdir -p ${REPO}/results ${REPO}/logs
cd ${REPO}/code/multiseed

# Paper-priority configs × seeds (mean ± std for README compact/summary/family/detail).
# Saves under results/<models_*>/<base_config>/seed_<N>/ (does NOT overwrite sweep folders).
SEEDS=${SEEDS:-21,22,23,24,25}
RUNS=${RUNS:-all}
EPOCHS=${EPOCHS:-100}
BATCH=${BATCH:-32}

echo "=== Host: $(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} ==="
python -u -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -u -m multiomics.code.multiseed.run --list-runs

echo "=== Multi-seed train+eval: runs=${RUNS} seeds=${SEEDS} epochs=${EPOCHS} ==="
python -u -m multiomics.code.multiseed.run \
  --runs "${RUNS}" \
  --seeds "${SEEDS}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH}" \
  --resume

echo "=== Done ==="
echo "Give Auto these files to update docs/README.md:"
echo "  ${REPO}/results/multiseed/summary.json"
echo "  ${REPO}/results/multiseed/readme_snippets.md"
echo "  ${REPO}/results/multiseed/summary_flat.csv"
