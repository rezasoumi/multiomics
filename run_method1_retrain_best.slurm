#!/bin/bash
#SBATCH --job-name=method1_retrain
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --output=logs/method1_retrain_%j.out
#SBATCH --error=logs/method1_retrain_%j.err

set -euo pipefail

module purge
module load scicomp-pytorch-env

SCRATCH=/scratch/work/soumir1
REPO=${SCRATCH}/githubmultiomics/multiomics
export PYTHONPATH=${SCRATCH}/githubmultiomics:${SCRATCH}

source ${SCRATCH}/multiomics/venv/bin/activate

cd ${REPO}/code/method1_predictability
mkdir -p ${REPO}/results ${REPO}/logs

echo "=== Host: $(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} ==="
python -u -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo "=== Retrain best sweep config WITHOUT early stopping ==="
echo "=== alpha=0.5 beta=0.5 gamma=2.5 lambda_ctr=1.0 epochs=100 ==="
echo "=== Saves LAST epoch (+ best-by-constructive for comparison) ==="

python -u retrain_config.py \
  --disease brca \
  --epochs 100 \
  --batch-size 32 \
  --lr 0.0004 \
  --weight-decay 0.0007 \
  --alpha 1.0 \
  --beta 1.0 \
  --gamma 1.0 \
  --lambda-ctr 1.0

OUTDIR=${REPO}/results/models_brca_method1/0.0004_0.0007_a0.5_b0.5_g2.5_l1.0_full100
echo "=== Done ==="
echo "Results: ${OUTDIR}"
echo "History: ${OUTDIR}/epoch_history.csv"
echo "Metrics last: ${OUTDIR}/metrics_last.json"
echo "Metrics constructive-best: ${OUTDIR}/metrics_best_constructive.json"
