#!/bin/bash
#SBATCH --job-name=m2_omidient
#SBATCH --time=96:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --output=logs/method2_omidient_%j.out
#SBATCH --error=logs/method2_omidient_%j.err

set -euo pipefail

module purge
module load scicomp-pytorch-env

SCRATCH=/scratch/work/soumir1
REPO=${SCRATCH}/githubmultiomics/multiomics
export PYTHONPATH=${SCRATCH}/githubmultiomics:${SCRATCH}

source ${SCRATCH}/multiomics/venv/bin/activate

cd ${REPO}/code/method2_omidient_private
mkdir -p ${REPO}/results ${REPO}/logs

echo "=== Host: $(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} ==="
python -u -c "import torch; import multiomics.code.method2_omidient_private.model as m; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'model_ok', m.__file__)"

echo "=== Method 2 + OMIDIENT private: joint train ==="
echo "=== Shared: Method-2 AE + contrastive; Private: product-Dirichlet (K=4) ==="
echo "=== Loss: λ_rec L_recon + λ_ctr L_ctr + λ_vic L_vic + λ_kl L_kl (λ_kl=1.0 fixed) ==="
echo "=== Grid: same Method-2 5x5x5 weights; config token _k4 ==="
echo "=== Rank: eval_loss = L_recon + L_ctr (L_vic/L_kl excluded) ==="
echo "=== Note: Dirichlet sampling is slower than AE private; --resume safe to requeue ==="

python -u hyperparam_tuning.py --epochs 100 --batch-size 32 --k-groups 4 --resume --rank-metric eval_loss

echo "=== Final summarize (best by fixed eval_loss) ==="
python -u hyperparam_tuning.py --summarize-only --rank-metric eval_loss

echo "=== Best config ==="
python -u -c "
import json
print(open('../../results/models_brca_method2_omidient/best_config.json').read())
"

BEST=$(python -u -c "import json; print(json.load(open('../../results/models_brca_method2_omidient/best_config.json'))['config'])")
echo "=== Re-evaluate best with plots: ${BEST} ==="
python -u evaluate.py --disease brca --config "${BEST}"

echo "=== Done ==="
echo "Leaderboard: ${REPO}/results/models_brca_method2_omidient/leaderboard.csv"
echo "Best JSON: ${REPO}/results/models_brca_method2_omidient/best_config.json"
