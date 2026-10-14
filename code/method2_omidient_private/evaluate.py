"""Evaluate Method 2 + OMIDIENT-private models (integration + shared-quality metrics)."""

import os
import argparse
import warnings

import numpy as np
import torch

from multiomics.code.loading_data import load_data
from multiomics.code import util
from multiomics.code.method2_omidient_private.model import SharedAndSpecificEmbedding
from multiomics.code.shared_rep_metrics import (
    evaluate_shared_and_integration,
    print_metrics_report,
)

warnings.filterwarnings('ignore')

NUM_CLUST = {'lihc': 2, 'coad': 4, 'kirc': 2, 'brca': 5}
DEFAULT_K = 4


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def _truth_from_label(disease, label):
    if disease == 'coad':
        return label.flatten().astype('int')
    if disease == 'lihc':
        return label[:, 1].flatten().astype('int')
    if disease == 'kirc':
        return label[:, 1].flatten().astype('int')
    return label.flatten()


def parse_k_from_config(folder, default=DEFAULT_K):
    """Prefer `_k{K}_` token in config name; else default."""
    parts = str(folder).split('_')
    for p in parts:
        if p.startswith('k') and p[1:].isdigit():
            return int(p[1:])
    return default


def build_model(view_size, k_groups):
    return SharedAndSpecificEmbedding(
        k_groups=k_groups,
        view_size=view_size,
        n_units_shared_1=[512, 256, 128, 32],
        n_units_shared_2=[512, 256, 128, 32],
        n_units_shared_3=[256, 128, 64, 32],
        n_units_priv_1=[512, 256, 128, 8],
        n_units_priv_2=[512, 256, 128, 8],
        n_units_priv_3=[256, 128, 64, 8],
        mlp_size=[32, 8],
    )


def evaluate_config(disease, config_dir, plot=False, k_groups=None):
    num_clust = NUM_CLUST[disease]
    view1_data, view2_data, view3_data, _, _ = load_data(disease)
    folder = os.path.basename(os.path.abspath(config_dir))
    if k_groups is None:
        k_groups = parse_k_from_config(folder)

    data = np.load(os.path.join(config_dir, 'test_data_{}.npy'.format(disease)))
    label = np.load(
        os.path.join(config_dir, 'test_label_{}.npy'.format(disease)),
        allow_pickle=True,
    )

    model = build_model(
        [view1_data.shape[1], view2_data.shape[1], view3_data.shape[1]],
        k_groups,
    )
    model.load_state_dict(
        torch.load(os.path.join(config_dir, 'model_{}'.format(disease)), map_location='cpu')
    )
    model.eval()
    setup_seed(2)

    d1, d2 = view1_data.shape[1], view2_data.shape[1]
    view1 = torch.tensor(data[:, :d1], dtype=torch.float32)
    view2 = torch.tensor(data[:, d1:d1 + d2], dtype=torch.float32)
    view3 = torch.tensor(data[:, d1 + d2:], dtype=torch.float32)

    with torch.no_grad():
        out = model(view1, view2, view3)

    specific = {i: out['specific'][i].numpy() for i in (1, 2, 3)}
    shared = {i: out['shared'][i].numpy() for i in (1, 2, 3)}
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
            os.path.join(config_dir, 'final_em'),
            'Method2OmidientPrivate',
        )

    return metrics


def main(args):
    disease = args.disease
    parent = '../../results/models_{}_method2_omidient'.format(disease)

    if args.config:
        folder = args.config
    else:
        ls = [{'loss': 1e12, 'config': None}]
        if os.path.isdir(parent):
            for config in os.listdir(parent):
                f = os.path.join(parent, config, 'loss.npy')
                if os.path.exists(f):
                    ls.append({'loss': float(np.load(f)), 'config': config})
        best = min(ls, key=lambda x: x['loss'])
        if best['config'] is None:
            raise FileNotFoundError(
                'No trained Method2+OMIDIENT models under {}'.format(parent)
            )
        folder = best['config']

    desired_path = os.path.join(parent, folder)
    print('Using config:', folder)
    metrics = evaluate_config(
        disease, desired_path, plot=True, k_groups=args.k_groups or None
    )
    print_metrics_report(metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Method 2 + OMIDIENT private')
    parser.add_argument('--disease', type=str, default='brca')
    parser.add_argument('--config', type=str, default='')
    parser.add_argument('--k-groups', type=int, default=0, help='0 = parse from config name')
    args, unknown = parser.parse_known_args()
    if unknown:
        raise ValueError(f'Unkown args: {unknown}')
    if args.k_groups == 0:
        args.k_groups = None
    main(args)
