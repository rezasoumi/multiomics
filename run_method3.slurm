#!/bin/bash
#SBATCH --job-name=method3_spectral
#SBATCH --time=96:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
# torch 2.12+cu130 has no V100 (sm_70) kernels — require A100+
#SBATCH --partition=gpu-a100-80g
#SBATCH --gpus=a100:1
#SBATCH --constraint=a100
#SBATCH --output=logs/method3_spectral_%j.out
#SBATCH --error=logs/method3_spectral_%j.err

set -euo pipefail

module purge
module load scicomp-pytorch-env

SCRATCH=/scratch/work/soumir1
REPO=${SCRATCH}/githubmultiomics/multiomics
export PYTHONPATH=${SCRATCH}/githubmultiomics:${SCRATCH}

source ${SCRATCH}/multiomics/venv/bin/activate

cd ${REPO}/code/method3_spectral_graph
mkdir -p ${REPO}/results ${REPO}/logs ${REPO}/results/spectral_targets

echo "=== Host: $(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} ==="
python -u -c "import torch; import multiomics.code.method3_spectral_graph.model as m; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'model_ok', m.__file__)"

echo "=== Method 3: spectral / graph shared + residual private (hybrid) ==="
echo "=== Loss: λ_spec L_spec + λ_res L_res + λ_ctr L_ctr + λ_vic L_vic + λ_view L_view + λ_leak L_leak ==="
echo "=== Grid default: coarse 4x3x3=36 (fixed vic=0.1 view=0.5 leak=0.5) ==="
echo "=== Rank: eval_loss = L_spec + L_res + L_ctr ==="
echo "=== First run builds results/spectral_targets/brca_U.npy (no labels) ==="

python -u hyperparam_tuning.py \
  --epochs 100 \
  --batch-size 32 \
  --grid-search coarse \
  --resume \
  --rank-metric eval_loss

echo "=== Final summarize ==="
python -u hyperparam_tuning.py --summarize-only --rank-metric eval_loss

echo "=== Best config ==="
python -u -c "
import json
print(open('../../results/models_brca_method3/best_config.json').read())
"

BEST=$(python -u -c "import json; print(json.load(open('../../results/models_brca_method3/best_config.json'))['config'])")
echo "=== Re-evaluate best with plots: ${BEST} ==="
python -u evaluate.py --disease brca --config "${BEST}"

echo "=== Done ==="
echo "Leaderboard: ${REPO}/results/models_brca_method3/leaderboard.csv"
echo "Best JSON: ${REPO}/results/models_brca_method3/best_config.json"
echo "Spectral U: ${REPO}/results/spectral_targets/brca_U.npy"
