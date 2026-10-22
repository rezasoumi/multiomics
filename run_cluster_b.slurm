#!/bin/bash
#SBATCH --job-name=cluster_b
#SBATCH --time=48:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu-a100-80g
#SBATCH --gpus=a100:1
#SBATCH --constraint=a100
#SBATCH --output=logs/cluster_b_%j.out
#SBATCH --error=logs/cluster_b_%j.err

set -euo pipefail

module purge
module load scicomp-pytorch-env

SCRATCH=/scratch/work/soumir1
REPO=${SCRATCH}/githubmultiomics/multiomics
export PYTHONPATH=${SCRATCH}/githubmultiomics:${SCRATCH}

source ${SCRATCH}/multiomics/venv/bin/activate

cd ${REPO}/code/cluster_b_scan
mkdir -p ${REPO}/results ${REPO}/logs

echo "=== Host: $(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} ==="
python -u -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# Requires Cluster A best_config.json (or pass PARENT_CONFIG)
PARENT_CONFIG="${PARENT_CONFIG:-}"
if [[ -z "${PARENT_CONFIG}" ]]; then
  if [[ ! -f ${REPO}/results/models_brca_cluster_a/best_config.json ]]; then
    echo "ERROR: Cluster A best_config.json missing. Run run_cluster_a.slurm first,"
    echo "       or: sbatch --export=ALL,PARENT_CONFIG=<a_config> run_cluster_b.slurm"
    exit 1
  fi
  echo "=== Using parent from Cluster A best_config.json ==="
  EXTRA_ARGS=()
else
  echo "=== Using PARENT_CONFIG=${PARENT_CONFIG} ==="
  EXTRA_ARGS=(--parent-config "${PARENT_CONFIG}")
fi

echo "=== Cluster B: SCAN neighbor fine-tune on Cluster A checkpoint ==="
echo "=== Grid: k_emb×λ_ent×lr_ft = 3×3×2 = 18 ==="

python -u hyperparam_tuning.py \
  --epochs 50 \
  --resume \
  "${EXTRA_ARGS[@]}"

BEST=$(python -u -c "import json; print(json.load(open('../../results/models_brca_cluster_b/best_config.json'))['config'])")
echo "=== Re-evaluate best: ${BEST} ==="
python -u evaluate.py --disease brca --config "${BEST}"

echo "=== Done ==="
echo "Leaderboard: ${REPO}/results/models_brca_cluster_b/leaderboard.csv"
echo "Best JSON: ${REPO}/results/models_brca_cluster_b/best_config.json"
