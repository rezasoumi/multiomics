#!/bin/bash
#SBATCH --job-name=cluster_c
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu-a100-80g
#SBATCH --gpus=a100:1
#SBATCH --constraint=a100
#SBATCH --output=logs/cluster_c_%j.out
#SBATCH --error=logs/cluster_c_%j.err

set -euo pipefail

module purge
module load scicomp-pytorch-env

SCRATCH=/scratch/work/soumir1
REPO=${SCRATCH}/githubmultiomics/multiomics
export PYTHONPATH=${SCRATCH}/githubmultiomics:${SCRATCH}

source ${SCRATCH}/multiomics/venv/bin/activate

cd ${REPO}/code/cluster_c_gnn
mkdir -p ${REPO}/results ${REPO}/logs

echo "=== Host: $(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} ==="
python -u -c "import torch; import multiomics.code.cluster_c_gnn.model as m; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'model_ok', m.__file__)"

echo "=== Cluster C: pure-PyTorch GCN + DEC on consensus patient graph ==="
echo "=== Grid: k×hidden×out×lr = 3×2×2×2 = 24 ==="
echo "=== Rank: eval_loss = val KL(p‖q) on train-masked DEC ==="
echo "=== Gate: run if A stalls below D / no-ortho (see adr-single-z-clustering.md) ==="

python -u hyperparam_tuning.py \
  --epochs 200 \
  --resume \
  --rank-metric eval_loss

echo "=== Final summarize ==="
python -u hyperparam_tuning.py --summarize-only --rank-metric eval_loss

BEST=$(python -u -c "import json; print(json.load(open('../../results/models_brca_cluster_c/best_config.json'))['config'])")
echo "=== Re-evaluate best: ${BEST} ==="
python -u evaluate.py --disease brca --config "${BEST}"

echo "=== Done ==="
echo "Leaderboard: ${REPO}/results/models_brca_cluster_c/leaderboard.csv"
echo "Best JSON: ${REPO}/results/models_brca_cluster_c/best_config.json"
