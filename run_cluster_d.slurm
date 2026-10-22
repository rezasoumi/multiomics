#!/bin/bash
#SBATCH --job-name=cluster_d
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --partition=batch
#SBATCH --output=logs/cluster_d_%j.out
#SBATCH --error=logs/cluster_d_%j.err

set -euo pipefail

module purge
module load scicomp-python-env || true

SCRATCH=/scratch/work/soumir1
REPO=${SCRATCH}/githubmultiomics/multiomics
export PYTHONPATH=${SCRATCH}/githubmultiomics:${SCRATCH}

# Prefer project venv (sklearn); fall back to module env
if [[ -f ${SCRATCH}/multiomics/venv/bin/activate ]]; then
  source ${SCRATCH}/multiomics/venv/bin/activate
fi

cd ${REPO}/code/cluster_d_consensus
mkdir -p ${REPO}/results ${REPO}/logs

echo "=== Host: $(hostname) ==="
echo "=== Cluster D: consensus co-association baseline (CPU, no neural nets) ==="
echo "=== Grid: d_pca×K×method = 3×3×2 = 18 ==="

python -u run_baseline.py --disease brca --grid

echo "=== Done ==="
python -u -c "
import json
print(open('../../results/models_brca_cluster_d/best_config.json').read())
"
echo "Leaderboard: ${REPO}/results/models_brca_cluster_d/leaderboard.csv"
