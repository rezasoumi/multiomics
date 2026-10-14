"""Evaluate a trained OMIDIENT (ProdGammaDirVae) checkpoint with shared_rep_metrics.

Same metric suite as MOCSS AE / Method 1 / Method 2:
  - concat(P1,P2,P3, mean(S)) clustering + kNN (KMeans n_init=30)
  - shared-only clustering / kNN
  - cross-view probes
  - S-alignment (cosine / CCA / CKA)
"""

import os
import json
import argparse
import warnings

import numpy as np
import torch

from multiomics.code.loading_data import load_data
from multiomics.code import util
from multiomics.code.prod_gamma_dirvae.prod_gamma_dirvae_cancer import (
    SharedAndSpecificEmbedding,
    setup_seed,
)
from multiomics.code.shared_rep_metrics import (
    evaluate_shared_and_integration,
    print_metrics_report,
)

warnings.filterwarnings('ignore')

NUM_CLUST = {'lihc': 2, 'coad': 4, 'kirc': 2, 'brca': 5}

# Same hardcoded “best” configs used in classifier.py / prod_gamma_dirvae_cancer.py
DEFAULT_OMIDIENT_CONFIG = {
    'brca': '0.0003_0.0005_4',
    'lihc': '0.0005_0.0007_4',
    'kirc': '0.0003_0.0005_4',
    'coad': '0.0002_0.0003_5',
}


def _truth_from_label(disease, label):
    if disease == 'coad':
        return label.flatten().astype('int')
    if disease == 'lihc':
        return label[:, 1].flatten().astype('int')
    if disease == 'kirc':
        return label[:, 1].flatten().astype('int')
    return label.flatten()


def parse_k_from_config(folder):
    """Folder names are {lr}_{wd}_{K}, e.g. 0.0003_0.0005_4."""
    return int(str(folder).rstrip('/').split('_')[-1])


def results_parent(disease):
    return '../../results/models_{}_ProdGammaDirVae'.format(disease)


def evaluate_config(disease, config_dir, folder_name, plot=False):
    num_clust = NUM_CLUST[disease]
    view1_data, view2_data, view3_data, _, _ = load_data(disease)
    k_groups = parse_k_from_config(folder_name)

    data = np.load(os.path.join(config_dir, 'test_data_{}.npy'.format(disease)))
    label = np.load(
        os.path.join(config_dir, 'test_label_{}.npy'.format(disease)),
        allow_pickle=True,
    )

    model = SharedAndSpecificEmbedding(
        'ProdGamDirVae',
        k_groups,
        view_size=[view1_data.shape[1], view2_data.shape[1], view3_data.shape[1]],
        n_units_1=[512, 256, 128, 8],
        n_units_2=[512, 256, 128, 8],
        n_units_3=[256, 128, 64, 8],
        mlp_size=[32, 8],
    )
    model_path = os.path.join(config_dir, 'model_{}'.format(disease))
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    setup_seed(2)

    d1, d2 = view1_data.shape[1], view2_data.shape[1]
    view1 = torch.tensor(data[:, :d1], dtype=torch.float32)
    view2 = torch.tensor(data[:, d1:d1 + d2], dtype=torch.float32)
    view3 = torch.tensor(data[:, d1 + d2:], dtype=torch.float32)

    with torch.no_grad():
        (
            s1, _a1, sh1,
            s2, _a2, sh2,
            s3, _a3, sh3,
            *_rest,
        ) = model(view1, view2, view3)

    specific = {
        1: s1.detach().numpy(),
        2: s2.detach().numpy(),
        3: s3.detach().numpy(),
    }
    shared = {
        1: sh1.detach().numpy(),
        2: sh2.detach().numpy(),
        3: sh3.detach().numpy(),
    }
    views = {
        1: view1.numpy(),
        2: view2.numpy(),
        3: view3.numpy(),
    }
    truth = _truth_from_label(disease, label)

    metrics, final_embedding, _ = evaluate_shared_and_integration(
        specific, shared, views, truth, num_clust
    )

    if plot:
        util.plot_with_path(
            final_embedding,
            truth,
            os.path.join(config_dir, 'final_em_shared_rep'),
            'OMIDIENT',
        )

    return metrics


def main(args):
    disease = args.disease
    parent = results_parent(disease)

    if args.config:
        folder = args.config
    else:
        folder = DEFAULT_OMIDIENT_CONFIG[disease]

    desired_path = os.path.join(parent, folder)
    if not os.path.isdir(desired_path):
        raise FileNotFoundError(
            'OMIDIENT config not found: {}. Train ProdGammaDirVae first '
            '(code/prod_gamma_dirvae/).'.format(desired_path)
        )

    print('Using OMIDIENT config:', folder, '| K=', parse_k_from_config(folder))
    print('Path:', desired_path)
    metrics = evaluate_config(disease, desired_path, folder, plot=args.plot)
    print_metrics_report(metrics)

    out = {
        'method': 'omidient',
        'disease': disease,
        'config': folder,
        'K': parse_k_from_config(folder),
        **metrics,
    }
    metrics_path = os.path.join(desired_path, 'metrics_shared_rep.json')
    with open(metrics_path, 'w') as f:
        json.dump(out, f, indent=2)
    print('Wrote', metrics_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate OMIDIENT with shared_rep_metrics')
    parser.add_argument('--disease', type=str, default='brca')
    parser.add_argument(
        '--config',
        type=str,
        default='',
        help='Subfolder under results/models_*_ProdGammaDirVae (default: repo best)',
    )
    parser.add_argument('--plot', action='store_true', help='Write embedding plots')
    args, unknown = parser.parse_known_args()
    if unknown:
        raise ValueError(f'Unkown args: {unknown}')
    main(args)
