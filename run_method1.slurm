#!/bin/bash
#SBATCH --job-name=method1
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --output=logs/method1_%j.out
#SBATCH --error=logs/method1_%j.err

set -euo pipefail

module purge
module load scicomp-pytorch-env

SCRATCH=/scratch/work/soumir1
export PYTHONPATH=${SCRATCH}

source ${SCRATCH}/multiomics/venv/bin/activate

# Repo copy used for this project (same layout as run_omidient.slurm)
REPO=${SCRATCH}/githubmultiomics/multiomics
cd ${REPO}/code/method1_predictability
mkdir -p ${REPO}/results ${REPO}/logs

echo "=== Host: $(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} ==="
python -u -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo "=== Training Method 1 (predictability surrogate, MOCSS base) ==="
python -u hyperparam_tuning.py --epochs 100 --batch-size 32

echo "=== Evaluating best config ==="
python -u evaluate.py --disease brca

echo "=== Done ==="
