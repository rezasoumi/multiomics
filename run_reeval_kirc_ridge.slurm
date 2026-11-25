#!/bin/bash
#SBATCH --job-name=kirc_ridge
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu-a100-80g
#SBATCH --gpus=a100:1
#SBATCH --constraint=a100
#SBATCH --output=logs/reeval_kirc_ridge_%j.out
#SBATCH --error=logs/reeval_kirc_ridge_%j.err

# Re-evaluate Ridge Δ X R² / own-view R² for KIRC loss-best configs ONLY.
# No training. Requires existing model_kirc checkpoints under results/models_kirc_*.

set -euo pipefail

module purge
module load scicomp-pytorch-env

SCRATCH=/scratch/work/soumir1
REPO=${SCRATCH}/githubmultiomics/multiomics
export PYTHONPATH=${SCRATCH}/githubmultiomics:${SCRATCH}
export PYTHONNOUSERSITE=1

source ${SCRATCH}/multiomics/venv/bin/activate

cd ${REPO}
bash ${REPO}/setup_kirc_data.sh
mkdir -p ${REPO}/logs ${REPO}/results

# loading_data.py expects cwd under code/<pkg>/ (../../data → repo/data).
# reeval_kirc_ridge.py also chdirs there; keep this as a safe default.
cd ${REPO}/code/method1_predictability

echo "=== Re-eval KIRC tops (Ridge PCA-on-Y fix) ==="
python -u -m multiomics.code.reeval_kirc_ridge \
  --out ${REPO}/results/kirc_ridge_reeval.json

echo "=== Done ==="
echo "Summary: ${REPO}/results/kirc_ridge_reeval.json"
echo "Per-config: results/models_kirc_*/<config>/metrics_ridge_pca.json"
