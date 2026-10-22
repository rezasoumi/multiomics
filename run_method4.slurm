#!/bin/bash
#SBATCH --job-name=method4_invariant
#SBATCH --time=96:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
# torch 2.12+cu130 has no V100 (sm_70) kernels — require A100+
#SBATCH --partition=gpu-a100-80g
#SBATCH --gpus=a100:1
#SBATCH --constraint=a100
#SBATCH --output=logs/method4_invariant_%j.out
#SBATCH --error=logs/method4_invariant_%j.err

set -euo pipefail

module purge
module load scicomp-pytorch-env

SCRATCH=/scratch/work/soumir1
REPO=${SCRATCH}/githubmultiomics/multiomics
export PYTHONPATH=${SCRATCH}/githubmultiomics:${SCRATCH}

source ${SCRATCH}/multiomics/venv/bin/activate

cd ${REPO}/code/method4_invariant_cluster
mkdir -p ${REPO}/results ${REPO}/logs

echo "=== Host: $(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} ==="
python -u -c "import torch; import multiomics.code.method4_invariant_cluster.model as m; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'model_ok', m.__file__)"

echo "=== Method 4: invariant DEC + cross-view agree + residual private ==="
echo "=== Warm-up 15 epochs (ctr+res+view), then KMeans init centroids, then full loss ==="
echo "=== Grid default: coarse 3x3x3x3=81 (fixed view=0.5 ctr=0.5 vic=0.1) ==="
echo "=== Rank: eval_loss = L_dec + L_agree + L_res + L_ctr ==="

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
print(open('../../results/models_brca_method4/best_config.json').read())
"

BEST=$(python -u -c "import json; print(json.load(open('../../results/models_brca_method4/best_config.json'))['config'])")
echo "=== Re-evaluate best with plots: ${BEST} ==="
python -u evaluate.py --disease brca --config "${BEST}"

echo "=== Done ==="
echo "Leaderboard: ${REPO}/results/models_brca_method4/leaderboard.csv"
echo "Best JSON: ${REPO}/results/models_brca_method4/best_config.json"
