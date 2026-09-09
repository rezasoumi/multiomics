#!/bin/bash
#SBATCH --job-name=method2
#SBATCH --time=48:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --output=logs/method2_%j.out
#SBATCH --error=logs/method2_%j.err

set -euo pipefail

module purge
module load scicomp-pytorch-env

SCRATCH=/scratch/work/soumir1
REPO=${SCRATCH}/githubmultiomics/multiomics
export PYTHONPATH=${SCRATCH}/githubmultiomics:${SCRATCH}

source ${SCRATCH}/multiomics/venv/bin/activate

cd ${REPO}/code/method2_joint_vicreg
mkdir -p ${REPO}/results ${REPO}/logs

echo "=== Host: $(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} ==="
python -u -c "import torch; import multiomics.code.method2_joint_vicreg.model as m; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'model_ok', m.__file__)"

echo "=== Method 2 sweep: 5x5x5 = 125 configs ==="
echo "=== Joint D(P,S)->X + VICReg P⊥S + contrastive (no solo shared/specific recon) ==="
echo "=== Grid: lambda_rec={0.3,0.5,0.7,1.0,1.5} lambda_ctr={0.5,0.7,1.0,1.5,2.0} lambda_vic={0.1,0.5,1.0,2.0,5.0} ==="
echo "=== Fixed: lr/wd=MOCSS-brca, epochs=100; rank by NMI ==="
echo "=== Checkpoint/early-stop on full val loss (all positive terms) ==="

python -u hyperparam_tuning.py --epochs 100 --batch-size 32 --resume --rank-metric nmi

echo "=== Final summarize (best by NMI) ==="
python -u hyperparam_tuning.py --summarize-only --rank-metric nmi

echo "=== Best config ==="
python -u -c "
import json
print(open('../../results/models_brca_method2/best_config.json').read())
"

BEST=$(python -u -c "import json; print(json.load(open('../../results/models_brca_method2/best_config.json'))['config'])")
echo "=== Re-evaluate best with plots: ${BEST} ==="
python -u evaluate.py --disease brca --config "${BEST}"

echo "=== Done ==="
echo "Leaderboard: ${REPO}/results/models_brca_method2/leaderboard.csv"
echo "Best JSON: ${REPO}/results/models_brca_method2/best_config.json"
