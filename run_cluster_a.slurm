#!/bin/bash
#SBATCH --job-name=cluster_a
#SBATCH --time=96:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu-a100-80g
#SBATCH --gpus=a100:1
#SBATCH --constraint=a100
#SBATCH --output=logs/cluster_a_%j.out
#SBATCH --error=logs/cluster_a_%j.err

set -euo pipefail

module purge
module load scicomp-pytorch-env

SCRATCH=/scratch/work/soumir1
REPO=${SCRATCH}/githubmultiomics/multiomics
export PYTHONPATH=${SCRATCH}/githubmultiomics:${SCRATCH}

source ${SCRATCH}/multiomics/venv/bin/activate

cd ${REPO}/code/cluster_a_contrastive_dec
mkdir -p ${REPO}/results ${REPO}/logs

echo "=== Host: $(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} ==="
python -u -c "import torch; import multiomics.code.cluster_a_contrastive_dec.model as m; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'model_ok', m.__file__)"

echo "=== Cluster A: contrastive + DEC on fused Z (no S/P) ==="
echo "=== Warm-up 20 epochs (ctr±rec), then KMeans centroids, then full loss ==="
echo "=== Grid: λ_ctr×λ_dec×λ_rec = 4×4×3 = 48 ==="
echo "=== Rank: eval_loss = L_ctr + L_dec ==="

python -u hyperparam_tuning.py \
  --epochs 100 \
  --batch-size 32 \
  --resume \
  --rank-metric eval_loss

echo "=== Final summarize ==="
python -u hyperparam_tuning.py --summarize-only --rank-metric eval_loss

echo "=== Best config ==="
python -u -c "
import json
print(open('../../results/models_brca_cluster_a/best_config.json').read())
"

BEST=$(python -u -c "import json; print(json.load(open('../../results/models_brca_cluster_a/best_config.json'))['config'])")
echo "=== Re-evaluate best: ${BEST} ==="
python -u evaluate.py --disease brca --config "${BEST}"

echo "=== Done ==="
echo "Leaderboard: ${REPO}/results/models_brca_cluster_a/leaderboard.csv"
echo "Best JSON: ${REPO}/results/models_brca_cluster_a/best_config.json"
