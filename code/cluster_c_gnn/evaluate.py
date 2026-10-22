"""Evaluate Cluster C: KMeans on Z + DEC assignment metrics."""

import argparse
import os
import warnings

import numpy as np
import torch

from multiomics.code.loading_data import load_data
from multiomics.code.cluster_common import (
    NUM_CLUST, truth_from_label, cluster_metrics, assignment_metrics, index_split,
)
from multiomics.code.cluster_c_gnn.model import PatientGCN

warnings.filterwarnings('ignore')


def evaluate_config(disease, config_dir):
    n_clust = NUM_CLUST[disease]
    view1, view2, view3, _, y_true = load_data(disease)
    truth = truth_from_label(disease, y_true)
    _, _, idx_te = index_split(len(truth))

    ckpt_path = os.path.join(config_dir, 'model_{}.pt'.format(disease))
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    cfg = ckpt['config']
    A_hat = ckpt['A_hat']
    feats = ckpt['feats']

    model = PatientGCN(
        feats.shape[1],
        hidden=cfg['hidden'],
        out_dim=cfg['out_dim'],
        n_clusters=n_clust,
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    with torch.no_grad():
        out = model(
            torch.tensor(feats, dtype=torch.float32),
            torch.tensor(A_hat, dtype=torch.float32),
        )
        z = out['z'].numpy()
        q = out['q'].numpy()
        pred = q.argmax(axis=1)

    metrics = cluster_metrics(z[idx_te], truth[idx_te], n_clust)
    metrics.update(assignment_metrics(pred[idx_te], truth[idx_te]))
    metrics['entropy_q'] = float(
        (-np.sum(q * np.log(np.clip(q, 1e-8, 1.0)), axis=1)).mean()
    )
    metrics['nmi_all'] = assignment_metrics(pred, truth)['nmi_assign']
    return metrics


def main(args):
    disease = args.disease
    parent = '../../results/models_{}_cluster_c'.format(disease)
    if args.config:
        folder = args.config
    else:
        best_path = os.path.join(parent, 'best_config.json')
        if not os.path.exists(best_path):
            raise FileNotFoundError(best_path)
        import json
        folder = json.load(open(best_path))['config']
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
