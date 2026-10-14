#!/bin/bash
#SBATCH --job-name=method1_sweep
#SBATCH --time=96:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
# torch 2.12+cu130 has no V100 (sm_70) kernels — require A100+
#SBATCH --partition=gpu-a100-80g
#SBATCH --gpus=a100:1
#SBATCH --constraint=a100
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

echo "=== Method 1 sweep: 5x5x5x4 = 500 configs (includes private-sufficiency b / λ_priv) ==="
echo "=== Grid: alpha beta gamma lambda_priv={0.1,0.3,0.5,1.0} ==="
echo "=== Fixed: lr/wd=MOCSS-brca defaults, lambda_ctr=1.0, epochs=100 ==="
echo "=== Checkpoint/early-stop: constructive = L_std + α L_shared + λ_priv L_priv + λ_ctr L_ctr ==="
echo "=== Rank across configs: eval_loss = L_std + L_shared + L_priv + L_ctr (unweighted) ==="
echo "=== Config names include _p{lambda_priv}; archive old leaderboard.csv if mixing with pre-b runs ==="
echo "=== Each config: train -> evaluate -> append leaderboard.csv; --resume skips done ==="

# --resume makes the job safe to requeue if the walltime is hit
python -u hyperparam_tuning.py --epochs 100 --batch-size 32 --resume --rank-metric eval_loss

echo "=== Final summarize (best by fixed eval_loss) ==="
python -u hyperparam_tuning.py --summarize-only --rank-metric eval_loss

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
