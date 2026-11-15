#!/bin/bash
#SBATCH --job-name=omi_kirc
#SBATCH --time=96:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
# torch 2.12+cu130 has no V100 (sm_70) kernels — require A100+
#SBATCH --partition=gpu-a100-80g
#SBATCH --gpus=a100:1
#SBATCH --constraint=a100
#SBATCH --output=logs/omidient_kirc_%j.out
#SBATCH --error=logs/omidient_kirc_%j.err

set -euo pipefail

module purge
module load scicomp-pytorch-env

SCRATCH=/scratch/work/soumir1
REPO=${SCRATCH}/githubmultiomics/multiomics
export PYTHONPATH=${SCRATCH}/githubmultiomics:${SCRATCH}
export PYTHONNOUSERSITE=1

source ${SCRATCH}/multiomics/venv/bin/activate

cd ${REPO}/code/prod_gamma_dirvae
mkdir -p ${REPO}/results ${REPO}/logs

# Unpack data.zip into data/KIRC/ if omics CSVs are missing (see setup_kirc_data.sh)
bash ${REPO}/setup_kirc_data.sh

# Fixed optimizer / K at pre-ortho BRCA best: lr=0.0003, wd=0.0005, K=4
# Coarse λ grid WITHOUT L_ortho:
#   L = λ_ctr L_ctr + λ_recon L_recon + λ_kl KL
#   coarse 3×3×3 = 27
# Saves under 0.0003_0.0005_4_c*_r*_k* (does NOT overwrite +ortho)
# Rank by unweighted eval_loss = L_ctr + L_recon + KL
# Space: skip omics dumps; keep only top-5 model_* by eval_loss
PARENT=${REPO}/results/models_kirc_ProdGammaDirVae

echo "=== Host: $(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} ==="
python -u -c "import torch; import multiomics.code.prod_gamma_dirvae.prod_gamma_dirvae_cancer as m; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'omidient_ok', m.__file__)"

echo "=== Train OMIDIENT (no ortho) on KIRC: coarse λ grid ==="
echo "=== --resume safe to requeue ==="

python -u hyperparam_tuning_prod_gamma_dirvae_cancer.py \
  --disease kirc \
  --epochs 100 \
  --batch-size 32 \
  --grid-search coarse \
  --resume

echo "=== Summarize best by eval_loss among no-ortho configs ==="
python -u hyperparam_tuning_prod_gamma_dirvae_cancer.py \
  --disease kirc \
  --summarize-only

BEST=$(python -u -c "import json; print(json.load(open('${PARENT}/best_config_default.json'))['config'])")
echo "=== Re-evaluate best no-ortho config: ${BEST} ==="
python -u evaluate.py --disease kirc --config "${BEST}" --plot

echo "=== Done ==="
echo "Leaderboard: ${PARENT}/leaderboard_default.csv"
echo "Best JSON:   ${PARENT}/best_config_default.json"
echo "Metrics:     ${PARENT}/${BEST}/metrics_shared_rep.json"
