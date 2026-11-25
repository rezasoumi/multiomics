#!/bin/bash
#SBATCH --job-name=viz_kirc
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
# torch 2.12+cu130 has no V100 (sm_70) kernels — require A100+
#SBATCH --partition=gpu-a100-80g
#SBATCH --gpus=a100:1
#SBATCH --constraint=a100
#SBATCH --output=logs/viz_kirc_%j.out
#SBATCH --error=logs/viz_kirc_%j.err

# Generate KIRC shared/private figures → docs/figures_kirc/
# Figs 1–4 from README numbers; 5–6 need models_kirc_* checkpoints / leaderboards.

set -euo pipefail

module purge
module load scicomp-pytorch-env

SCRATCH=/scratch/work/soumir1
REPO=${SCRATCH}/githubmultiomics/multiomics
export PYTHONPATH=${SCRATCH}/githubmultiomics:${SCRATCH}
export PYTHONNOUSERSITE=1

source ${SCRATCH}/multiomics/venv/bin/activate

cd ${REPO}
export MULTIOMICS_RESULTS=${REPO}/results
mkdir -p ${MULTIOMICS_RESULTS}/diagnostics ${REPO}/logs ${REPO}/docs/figures_kirc

# KIRC omics needed if test_data_*.npy were pruned (load_or_rebuild_test_split)
bash ${REPO}/setup_kirc_data.sh

echo "=== Host: $(hostname) ==="
echo "=== REPO=${REPO} ==="
ls -d ${MULTIOMICS_RESULTS}/models_kirc_* 2>/dev/null || echo "WARNING: no models_kirc_* under results"

python -u -c "import umap; print('umap-learn OK')" 2>/dev/null || echo "NOTE: umap-learn not installed; using t-SNE fallback"

python -u -m multiomics.code.viz.run_all \
  --mode all \
  --disease kirc \
  --reduce umap \
  --out-dir ${REPO}/docs/figures_kirc

echo "=== Done ==="
echo "Figures: ${REPO}/docs/figures_kirc/"
ls -1 ${REPO}/docs/figures_kirc/*.{png,pdf} 2>/dev/null | head -40
