#!/bin/bash
#SBATCH --job-name=omidient_ortho
#SBATCH --time=96:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
# torch 2.12+cu130 has no V100 (sm_70) kernels — require A100+
#SBATCH --partition=gpu-a100-80g
#SBATCH --gpus=a100:1
#SBATCH --constraint=a100
#SBATCH --output=logs/omidient_ortho_%j.out
#SBATCH --error=logs/omidient_ortho_%j.err

set -euo pipefail

module purge
module load scicomp-pytorch-env

SCRATCH=/scratch/work/soumir1
REPO=${SCRATCH}/githubmultiomics/multiomics
export PYTHONPATH=${SCRATCH}/githubmultiomics:${SCRATCH}

source ${SCRATCH}/multiomics/venv/bin/activate

cd ${REPO}/code/prod_gamma_dirvae
mkdir -p ${REPO}/results ${REPO}/logs

# Fixed optimizer / K at pre-ortho BRCA best: lr=0.0003, wd=0.0005, K=4
# Grid loss weights WITH L_ortho:
#   L = λ_ctr L_ctr + λ_recon L_recon + λ_kl KL + λ_ortho L_ortho
#   5×5×5×5 = 625 configs  (c/r/k/o grids)
# Saves under 0.0003_0.0005_4_ortho_c*_r*_k*_o* (does NOT overwrite no-ortho)
# Rank by unweighted eval_loss = L_ctr + L_recon + KL (λ and L_ortho excluded)
PARENT=${REPO}/results/models_brca_ProdGammaDirVae

echo "=== Host: $(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} ==="
python -u -c "import torch; import multiomics.code.prod_gamma_dirvae.prod_gamma_dirvae_cancer as m; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'omidient_ok', m.__file__)"

echo "=== Train OMIDIENT + L_ortho: λ grid (lr/wd/K fixed) ==="
echo "=== --resume safe to requeue ==="

python -u hyperparam_tuning_prod_gamma_dirvae_cancer.py \
  --epochs 100 \
  --batch-size 32 \
  --use-orthogonal \
  --grid-search \
  --resume

echo "=== Summarize best by eval_loss among *_ortho_* ==="
python -u hyperparam_tuning_prod_gamma_dirvae_cancer.py --summarize-only --use-orthogonal

BEST=$(python -u -c "import json; print(json.load(open('${PARENT}/best_config_ortho.json'))['config'])")
echo "=== Re-evaluate best ortho config: ${BEST} ==="
python -u evaluate.py --disease brca --config "${BEST}" --plot

echo "=== Done ==="
echo "Leaderboard: ${PARENT}/leaderboard_ortho.csv"
echo "Best JSON:   ${PARENT}/best_config_ortho.json"
echo "Metrics:     ${PARENT}/${BEST}/metrics_shared_rep.json"
echo "Baseline (no ortho) remains at: ${PARENT}/0.0003_0.0005_4/"
