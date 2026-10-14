#!/bin/bash
#SBATCH --job-name=omidient_eval
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --output=logs/omidient_eval_%j.out
#SBATCH --error=logs/omidient_eval_%j.err

set -euo pipefail

module purge
module load scicomp-pytorch-env

SCRATCH=/scratch/work/soumir1
REPO=${SCRATCH}/githubmultiomics/multiomics
export PYTHONPATH=${SCRATCH}/githubmultiomics:${SCRATCH}

source ${SCRATCH}/multiomics/venv/bin/activate

cd ${REPO}/code/prod_gamma_dirvae
mkdir -p ${REPO}/results ${REPO}/logs

DISEASE=brca
# Repo default best OMIDIENT config for BRCA (same as classifier.py / prod_gamma_dirvae_cancer.py)
CONFIG=0.0003_0.0005_4

echo "=== Host: $(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} ==="
python -u -c "import torch; import multiomics.code.prod_gamma_dirvae.prod_gamma_dirvae_cancer as m; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'omidient_ok', m.__file__)"

echo "=== Evaluate existing OMIDIENT checkpoint with shared_rep_metrics ==="
echo "=== disease=${DISEASE} config=${CONFIG} ==="
echo "=== Metrics: concat clustering/kNN + shared-only + probes + alignment ==="

python -u evaluate.py --disease "${DISEASE}" --config "${CONFIG}" --plot

echo "=== Done ==="
echo "Metrics JSON: ${REPO}/results/models_${DISEASE}_ProdGammaDirVae/${CONFIG}/metrics_shared_rep.json"
