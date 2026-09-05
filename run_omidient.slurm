#!/bin/bash
#SBATCH --job-name=omidient
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --output=logs/omidient_%j.out
#SBATCH --error=logs/omidient_%j.err

set -euo pipefail

module purge
module load scicomp-pytorch-env

SCRATCH=/scratch/work/soumir1
export PYTHONPATH=${SCRATCH}

source ${SCRATCH}/multiomics/venv/bin/activate

cd ${SCRATCH}/githubmultiomics/code/prod_gamma_dirvae
mkdir -p ../../results ../../logs

echo "=== Training OMIDIENT (ProdGammaDirVae) ==="
python -u hyperparam_tuning_prod_gamma_dirvae_cancer.py --epochs 100 --batch-size 32

echo "=== Done training ==="