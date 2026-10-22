"""Evaluate Method 4 models."""

import os
import argparse
import warnings

import numpy as np
import torch

from multiomics.code.loading_data import load_data
from multiomics.code import util
from multiomics.code.method4_invariant_cluster.model import SharedAndSpecificEmbedding
from multiomics.code.shared_rep_metrics import (
    evaluate_shared_and_integration,
    print_metrics_report,
)

warnings.filterwarnings('ignore')

NUM_CLUST = {'lihc': 2, 'coad': 4, 'kirc': 2, 'brca': 5}


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def _truth_from_label(disease, label):
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

    # Cross-view soft-label agreement (JS-like via mean symmetric KL)
    q = {i: out['q'][i].numpy() for i in (1, 2, 3)}
    def _sym_kl(a, b):
        eps = 1e-8
        a = np.clip(a, eps, 1)
        b = np.clip(b, eps, 1)
        kl_ab = np.mean(np.sum(a * (np.log(a) - np.log(b)), axis=1))
        kl_ba = np.mean(np.sum(b * (np.log(b) - np.log(a)), axis=1))
        return 0.5 * (kl_ab + kl_ba)

    metrics['agree_symkl_mean'] = float(
        (_sym_kl(q[1], q[2]) + _sym_kl(q[1], q[3]) + _sym_kl(q[2], q[3])) / 3.0
    )
    # Entropy of q_bar (collapse detector)
    q_bar = out['q_bar'].numpy()
    ent = -np.sum(q_bar * np.log(np.clip(q_bar, 1e-8, 1.0)), axis=1).mean()
    metrics['entropy_q_bar'] = float(ent)

    if plot:
        util.plot_with_path(
            final_embedding,
            truth,
            os.path.join(config_dir, 'final_em'),
            'Method4InvariantCluster',
        )

    return metrics


def main(args):
    disease = args.disease
    parent = '../../results/models_{}_method4'.format(disease)

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
            raise FileNotFoundError('No trained Method 4 models under {}'.format(parent))
        folder = best['config']

    desired_path = os.path.join(parent, folder)
    print('Using config:', folder)
    metrics = evaluate_config(disease, desired_path, plot=True)
    print_metrics_report(metrics)
    print('agree_symkl_mean={:.4f} entropy_q_bar={:.4f}'.format(
        metrics.get('agree_symkl_mean', float('nan')),
        metrics.get('entropy_q_bar', float('nan')),
    ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Method 4')
    parser.add_argument('--disease', type=str, default='brca')
    parser.add_argument('--config', type=str, default='')
    args, unknown = parser.parse_known_args()
    if unknown:
        raise ValueError(f'Unkown args: {unknown}')
    main(args)
