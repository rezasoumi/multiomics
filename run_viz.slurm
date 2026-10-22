#!/bin/bash
#SBATCH --job-name=viz_sp
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/viz_shared_private_%j.out
#SBATCH --error=logs/viz_shared_private_%j.err

# Generate all six shared/private figures (README numbers + Triton checkpoints).
# No retraining — only loads existing models / leaderboard CSVs.

set -euo pipefail

module purge
module load scicomp-pytorch-env

SCRATCH=/scratch/work/soumir1
REPO=${SCRATCH}/githubmultiomics/multiomics
export PYTHONPATH=${SCRATCH}/githubmultiomics:${SCRATCH}

source ${SCRATCH}/multiomics/venv/bin/activate

cd ${REPO}
export MULTIOMICS_RESULTS=${REPO}/results
mkdir -p ${MULTIOMICS_RESULTS}/diagnostics ${REPO}/logs ${REPO}/docs/figures

echo "=== Host: $(hostname) ==="
echo "=== REPO=${REPO} ==="
echo "=== MULTIOMICS_RESULTS=${MULTIOMICS_RESULTS} ==="
ls -d ${MULTIOMICS_RESULTS}/models_brca_* 2>/dev/null || echo "WARNING: no models_brca_* under results"

# Optional: umap-learn if present; otherwise script falls back to sklearn t-SNE
python -u -c "import umap; print('umap-learn OK')" 2>/dev/null || echo "NOTE: umap-learn not installed; using t-SNE fallback"

python -u -m multiomics.code.viz.run_all \
  --mode all \
  --disease brca \
  --reduce umap \
  --out-dir ${REPO}/docs/figures

echo "=== Done ==="
echo "Figures: ${REPO}/docs/figures/"
ls -1 ${REPO}/docs/figures/*.{png,pdf} 2>/dev/null | head -40
