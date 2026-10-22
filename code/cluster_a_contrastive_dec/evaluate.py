"""Evaluate Cluster A: KMeans on Z + DEC assignment metrics."""

import argparse
import os
import warnings

import numpy as np
import torch

from multiomics.code.loading_data import load_data
from multiomics.code.cluster_common import NUM_CLUST, truth_from_label, cluster_metrics, assignment_metrics
from multiomics.code.cluster_a_contrastive_dec.model import MultiViewClusterModel

warnings.filterwarnings('ignore')


def evaluate_config(disease, config_dir, plot=False):
    n_clust = NUM_CLUST[disease]
    view1, view2, view3, _, _ = load_data(disease)
    data = np.load(os.path.join(config_dir, 'test_data_{}.npy'.format(disease)))
    label = np.load(os.path.join(config_dir, 'test_label_{}.npy'.format(disease)), allow_pickle=True)
    truth = truth_from_label(disease, label)

    model = MultiViewClusterModel(
        view_size=[view1.shape[1], view2.shape[1], view3.shape[1]],
        n_units_1=[512, 256, 128, 32],
        n_units_2=[512, 256, 128, 32],
        n_units_3=[256, 128, 64, 32],
        mlp_size=[32, 8],
        n_clusters=n_clust,
    )
    model.load_state_dict(
        torch.load(os.path.join(config_dir, 'model_{}'.format(disease)), map_location='cpu')
    )
    model.eval()

    d1, d2 = view1.shape[1], view2.shape[1]
    v1 = torch.tensor(data[:, :d1], dtype=torch.float32)
    v2 = torch.tensor(data[:, d1:d1 + d2], dtype=torch.float32)
    v3 = torch.tensor(data[:, d1 + d2:], dtype=torch.float32)
    with torch.no_grad():
        out = model(v1, v2, v3)
        z = out['z'].numpy()
        q = out['q'].numpy()
        pred_assign = q.argmax(axis=1)

    metrics = cluster_metrics(z, truth, n_clust)
    metrics.update(assignment_metrics(pred_assign, truth))
    metrics['entropy_q'] = float(
        (-np.sum(q * np.log(np.clip(q, 1e-8, 1.0)), axis=1)).mean()
    )
    return metrics


def main(args):
    disease = args.disease
    parent = '../../results/models_{}_cluster_a'.format(disease)
    if args.config:
        folder = args.config
    else:
        best = None
        best_loss = 1e12
        if os.path.isdir(parent):
            for c in os.listdir(parent):
                f = os.path.join(parent, c, 'eval_loss.npy')
                if os.path.exists(f):
                    loss = float(np.load(f))
                    if loss < best_loss:
                        best_loss = loss
                        best = c
        if best is None:
            raise FileNotFoundError('No Cluster A models under {}'.format(parent))
        folder = best
    path = os.path.join(parent, folder)
    print('Using config:', folder)
    metrics = evaluate_config(disease, path)
    for k, v in metrics.items():
        print('  {}: {}'.format(k, v))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--disease', default='brca')
    p.add_argument('--config', default='')
    args, unknown = p.parse_known_args()
    if unknown:
        raise ValueError(unknown)
    main(args)
