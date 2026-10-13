#!/bin/bash
#SBATCH --job-name=diag_sp
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --output=logs/diagnose_shared_private_%j.out
#SBATCH --error=logs/diagnose_shared_private_%j.err

set -euo pipefail

module purge
module load scicomp-pytorch-env

SCRATCH=/scratch/work/soumir1
REPO=${SCRATCH}/githubmultiomics/multiomics
export PYTHONPATH=${SCRATCH}/githubmultiomics:${SCRATCH}

source ${SCRATCH}/multiomics/venv/bin/activate

cd ${REPO}
mkdir -p ${REPO}/results/diagnostics ${REPO}/logs

echo "=== Host: $(hostname) ==="
echo "=== REPO=${REPO} ==="
echo "=== Diagnose shared vs private on existing checkpoints (no training) ==="
echo "=== Looking under: ${REPO}/results/models_brca_* ==="
ls -d ${REPO}/results/models_brca_* 2>/dev/null || echo "WARNING: no models_brca_* under ${REPO}/results"

echo "=== Panel: MOCSS + Method1 (loss-best & #3) + VICReg (loss-best & #10) + Method2 (loss-best & high-NMI) ==="

python -u code/diagnose_shared_private.py --disease brca

echo "=== Done ==="
echo "Summary: ${REPO}/results/diagnostics/shared_private_panel_brca.json"
echo "Per-config: <config_dir>/diagnostics_shared_private.json"
