#!/bin/bash
#SBATCH --job-name=method1_sweep
#SBATCH --time=48:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --output=logs/method1_sweep_%j.out
#SBATCH --error=logs/method1_sweep_%j.err

set -euo pipefail

module purge
module load scicomp-pytorch-env

SCRATCH=/scratch/work/soumir1
REPO=${SCRATCH}/githubmultiomics/multiomics

# Prefer the synced github copy (has method1_predictability)
export PYTHONPATH=${SCRATCH}/githubmultiomics:${SCRATCH}

source ${SCRATCH}/multiomics/venv/bin/activate

cd ${REPO}/code/method1_predictability
mkdir -p ${REPO}/results ${REPO}/logs

echo "=== Host: $(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} ==="
echo "=== PYTHONPATH=${PYTHONPATH} ==="
python -u -c "import torch; import multiomics.code.method1_predictability.model as m; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'model_ok', m.__file__)"

echo "=== Method 1 sweep: 5x5x5 = 125 predictability-weight configs ==="
echo "=== Grid: alpha={0.3,0.5,0.7,1.0,1.5} beta={0.5,0.7,1.0,1.5,2.0} gamma={0.7,1.0,1.5,2.0,2.5} ==="
echo "=== Fixed: lr/wd=MOCSS-brca defaults, lambda_ctr=1.0, epochs=100 ==="
echo "=== Each config: train -> evaluate -> append leaderboard.csv; --resume skips done ==="

# --resume makes the job safe to requeue if the walltime is hit
python -u hyperparam_tuning.py --epochs 100 --batch-size 32 --resume --rank-metric knn_acc

echo "=== Final summarize (best by kNN acc) ==="
python -u hyperparam_tuning.py --summarize-only --rank-metric knn_acc

echo "=== Best config details ==="
python -u -c "
import json
p='../../results/models_brca_method1/best_config.json'
print(open(p).read())
"

BEST=$(python -u -c "import json; print(json.load(open('../../results/models_brca_method1/best_config.json'))['config'])")
echo "=== Re-evaluate best with plots: ${BEST} ==="
python -u evaluate.py --disease brca --config "${BEST}"

echo "=== Done. Leaderboard: ${REPO}/results/models_brca_method1/leaderboard.csv ==="
echo "=== Best JSON: ${REPO}/results/models_brca_method1/best_config.json ==="
