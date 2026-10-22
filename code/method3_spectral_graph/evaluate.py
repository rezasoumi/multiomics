"""Evaluate Method 3 models (integration + shared-quality metrics)."""

import os
import argparse
import warnings

import numpy as np
import torch

from multiomics.code.loading_data import load_data
from multiomics.code import util
from multiomics.code.method3_spectral_graph.model import SharedAndSpecificEmbedding
from multiomics.code.method3_spectral_graph.graph_spectral import (
    targets_paths,
    nmi_on_spectral_U,
)
from multiomics.code.shared_rep_metrics import (
    evaluate_shared_and_integration,
    print_metrics_report,
)

warnings.filterwarnings('ignore')

NUM_CLUST = {'lihc': 2, 'coad': 4, 'kirc': 2, 'brca': 5}
SPECTRAL_DIM = 16


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def _truth_from_label(disease, label):
    # load_data('brca') returns a pandas DataFrame; npy loads return ndarray
    label = np.asarray(label)
    if disease == 'coad':
        return label.flatten().astype('int')
    if disease == 'lihc':
        lst = label[:, 0].flatten()
        unique_vals = list(set(lst))
        mapping = {val: idx for idx, val in enumerate(unique_vals)}
        return np.asarray([mapping[val] for val in lst])
    if disease == 'kirc':
        return label[:, 1].flatten().astype('int')
    return label.flatten()


def evaluate_config(disease, config_dir, plot=False):
    num_clust = NUM_CLUST[disease]
    view1_data, view2_data, view3_data, _, _ = load_data(disease)

    data = np.load(os.path.join(config_dir, 'test_data_{}.npy'.format(disease)))
    label = np.load(
        os.path.join(config_dir, 'test_label_{}.npy'.format(disease)),
        allow_pickle=True,
    )

    model = SharedAndSpecificEmbedding(
        view_size=[view1_data.shape[1], view2_data.shape[1], view3_data.shape[1]],
        n_units_1=[512, 256, 128, 32],
        n_units_2=[512, 256, 128, 32],
        n_units_3=[256, 128, 64, 32],
        mlp_size=[32, 8],
        spectral_dim=SPECTRAL_DIM,
        n_clusters=num_clust,
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
    views = {1: view1.numpy(), 2: view2.numpy(), 3: view3.numpy()}
    truth = _truth_from_label(disease, label)

    metrics, final_embedding, _ = evaluate_shared_and_integration(
        specific, shared, views, truth, num_clust
    )

    # Spectral ceiling (full-cohort U), if available
    paths = targets_paths(disease)
    if os.path.exists(paths['U']):
        U = np.load(paths['U'])
        # Full-cohort labels for ceiling (same order as load_data)
        _, _, _, _, y_all = load_data(disease)
        truth_all = _truth_from_label(disease, y_all)
        try:
            metrics['nmi_spectral_U'] = nmi_on_spectral_U(U, truth_all, num_clust)
        except Exception as exc:
            print('Warning: nmi_spectral_U failed:', exc)
            metrics['nmi_spectral_U'] = float('nan')
    else:
        metrics['nmi_spectral_U'] = float('nan')

    if plot:
        util.plot_with_path(
            final_embedding,
            truth,
            os.path.join(config_dir, 'final_em'),
            'Method3SpectralGraph',
        )

    return metrics


def main(args):
    disease = args.disease
    parent = '../../results/models_{}_method3'.format(disease)

    if args.config:
        folder = args.config
    else:
        ls = [{'loss': 1e12, 'config': None}]
        if os.path.isdir(parent):
            for config in os.listdir(parent):
                f = os.path.join(parent, config, 'eval_loss.npy')
                if os.path.exists(f):
                    ls.append({'loss': float(np.load(f)), 'config': config})
        best = min(ls, key=lambda x: x['loss'])
        if best['config'] is None:
            raise FileNotFoundError('No trained Method 3 models under {}'.format(parent))
        folder = best['config']

    desired_path = os.path.join(parent, folder)
    print('Using config:', folder)
    metrics = evaluate_config(disease, desired_path, plot=True)
    print_metrics_report(metrics)
    if 'nmi_spectral_U' in metrics:
        print('nmi_spectral_U (ceiling proxy): {:.4f}'.format(metrics['nmi_spectral_U']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Method 3')
    parser.add_argument('--disease', type=str, default='brca')
    parser.add_argument('--config', type=str, default='')
    args, unknown = parser.parse_known_args()
    if unknown:
        raise ValueError(f'Unkown args: {unknown}')
    main(args)
