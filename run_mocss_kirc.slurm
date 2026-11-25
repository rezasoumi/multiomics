#!/bin/bash
#SBATCH --job-name=mocss_kirc
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
# torch 2.12+cu130 has no V100 (sm_70) kernels — require A100+
#SBATCH --partition=gpu-a100-80g
#SBATCH --gpus=a100:1
#SBATCH --constraint=a100
#SBATCH --output=logs/mocss_kirc_%j.out
#SBATCH --error=logs/mocss_kirc_%j.err

set -euo pipefail

module purge
module load scicomp-pytorch-env

SCRATCH=/scratch/work/soumir1
REPO=${SCRATCH}/githubmultiomics/multiomics
export PYTHONPATH=${SCRATCH}/githubmultiomics:${SCRATCH}
export PYTHONNOUSERSITE=1

source ${SCRATCH}/multiomics/venv/bin/activate

cd ${REPO}/code/ae
mkdir -p ${REPO}/results ${REPO}/logs ${REPO}/results/models_kirc_ae

bash ${REPO}/setup_kirc_data.sh

# MOCSS AE baseline for KIRC (single published lr/wd; not a big grid).
# Config folder: 0.0002_0.0007 → results/models_kirc_ae/0.0002_0.0007/
echo "=== Host: $(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} ==="
python -u -c "import torch; import multiomics.code.ae.mocss_original_refactored as m; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'ae_ok', m.__file__)"

echo "=== Train + evaluate MOCSS (AE) on KIRC ==="
python -u hyperparam_tuning_mocss.py \
  --disease kirc \
  --epochs 100 \
  --batch-size 32 \
  --resume \
  --plot

echo "=== Done ==="
echo "Checkpoint: ${REPO}/results/models_kirc_ae/0.0002_0.0007/model_kirc"
echo "Metrics:    ${REPO}/results/models_kirc_ae/0.0002_0.0007/metrics_shared_rep.json"
