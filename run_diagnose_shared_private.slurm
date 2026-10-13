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
export MULTIOMICS_RESULTS=${REPO}/results
mkdir -p ${MULTIOMICS_RESULTS}/diagnostics ${REPO}/logs

echo "=== Host: $(hostname) ==="
echo "=== REPO=${REPO} ==="
echo "=== MULTIOMICS_RESULTS=${MULTIOMICS_RESULTS} ==="
echo "=== script on disk: ==="
ls -l ${REPO}/code/diagnose_shared_private.py
grep -n "RESULTS_ROOT" ${REPO}/code/diagnose_shared_private.py | head -5
echo "=== Looking under: ${MULTIOMICS_RESULTS}/models_brca_* ==="
ls -d ${MULTIOMICS_RESULTS}/models_brca_* 2>/dev/null || echo "WARNING: no models_brca_* under ${MULTIOMICS_RESULTS}"
test -d ${MULTIOMICS_RESULTS}/models_brca_ae/0.0004_0.0007 && echo "OK: mocss checkpoint present" || echo "MISSING: mocss checkpoint"

echo "=== Panel: MOCSS + Method1 (loss-best & #3) + VICReg (loss-best & #10) + Method2 (loss-best & high-NMI) ==="

python -u ${REPO}/code/diagnose_shared_private.py --disease brca

echo "=== Done ==="
echo "Summary: ${REPO}/results/diagnostics/shared_private_panel_brca.json"
echo "Per-config: <config_dir>/diagnostics_shared_private.json"
