"""Hyperparameter sweep for Cluster C (patient-graph GCN + DEC)."""

import argparse
import csv
import json
import os
import warnings
from itertools import product

import numpy as np
import torch

from multiomics.code.loading_data import load_data
from multiomics.code.cluster_common import NUM_CLUST, index_split, truth_from_label
from multiomics.code.cluster_c_gnn.graph_build import build_consensus_graph
from multiomics.code.cluster_c_gnn.model import PatientGCN
from multiomics.code.cluster_c_gnn.training import (
    EarlyStopper, init_centroids, train_epoch, val_loss,
)
from multiomics.code.cluster_c_gnn.evaluate import evaluate_config

warnings.filterwarnings('ignore')

disease = 'brca'
EPOCHS = 200
USE_GPU = torch.cuda.is_available()
SEED = 21
N_CLUST = NUM_CLUST[disease]
WD = 1e-4

K_GRID = [10, 15, 20]
HIDDEN_GRID = [64, 128]
OUT_GRID = [32, 64]
LR_GRID = [1e-3, 5e-4]


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def results_parent():
    return '../../results/models_{}_cluster_c'.format(disease)


def config_name(k_graph, hidden, out_dim, lr):
    return 'k{}_h{}_z{}_lr{}'.format(k_graph, hidden, out_dim, lr)


def append_leaderboard(row, fields):
    path = os.path.join(results_parent(), 'leaderboard.csv')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        if write_header:
            w.writeheader()
        w.writerow(row)


def load_done():
    path = os.path.join(results_parent(), 'leaderboard.csv')
    if not os.path.exists(path):
        return set()
    with open(path, 'r', newline='') as f:
        return {r['config'] for r in csv.DictReader(f) if r.get('config')}


def summarize(metric='eval_loss'):
    path = os.path.join(results_parent(), 'leaderboard.csv')
    with open(path, 'r', newline='') as f:
        rows = list(csv.DictReader(f))
    minimize = metric == 'eval_loss'
    key = (lambda r: float(r[metric]))
    best = min(rows, key=key) if minimize else max(rows, key=key)
    ranked = sorted(rows, key=key, reverse=not minimize)
    print('\n========== TOP 10 by {} =========='.format(metric))
    for i, r in enumerate(ranked[:10], 1):
        print(
            '{:2d}. {} | nmi={:.4f} nmi_assign={:.4f} eval={:.4f}'.format(
                i, r['config'], float(r['nmi']), float(r.get('nmi_assign', 'nan')),
                float(r['eval_loss']),
            )
        )
    out = {
        'rank_metric': metric,
        'config': best['config'],
        'eval_loss': float(best['eval_loss']),
        'nmi': float(best['nmi']),
        'nmi_assign': float(best.get('nmi_assign', 'nan')),
        'ari': float(best['ari']),
        'acc': float(best['acc']),
        'knn_acc': float(best.get('knn_acc', 'nan')),
    }
    with open(os.path.join(results_parent(), 'best_config.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print('\nBEST:', json.dumps(out, indent=2))
    return out


def train_one(k_graph, hidden, out_dim, lr, epochs):
    setup_seed(SEED)
    view1, view2, view3, _, y_true = load_data(disease)
    idx_tr, idx_va, idx_te = index_split(len(y_true))
    A_hat, feats = build_consensus_graph(
        view1, view2, view3, k=k_graph, d_pca=50, random_state=SEED,
    )
    x = torch.tensor(feats, dtype=torch.float32)
    a = torch.tensor(A_hat, dtype=torch.float32)
    if USE_GPU:
        x, a = x.cuda(), a.cuda()

    model = PatientGCN(feats.shape[1], hidden=hidden, out_dim=out_dim, n_clusters=N_CLUST)
    if USE_GPU:
        model = model.cuda()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WD)

    init_centroids(model, x, a, idx_tr, N_CLUST, USE_GPU, seed=SEED)

    name = config_name(k_graph, hidden, out_dim, lr)
    path = os.path.join(results_parent(), name)
    os.makedirs(path, exist_ok=True)

    stopper = EarlyStopper()
    best_score, best_state = np.inf, None
    hist = os.path.join(path, 'epoch_history.csv')
    with open(hist, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'val_loss']).writeheader()

    for epoch in range(epochs):
        tr = train_epoch(model, x, a, idx_tr, opt)
        va = val_loss(model, x, a, idx_va)
        with open(hist, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'val_loss']).writerow({
                'epoch': epoch + 1, 'train_loss': tr, 'val_loss': va,
            })
        if epoch == 0 or (epoch + 1) % 20 == 0:
            print('[C] epoch {}/{} train={:.4f} val={:.4f}'.format(epoch + 1, epochs, tr, va))
        if va < best_score:
            best_score = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if stopper.early_stop(va):
            print('Early stop at', epoch)
            break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    torch.save({
        'model_state_dict': best_state,
        'A_hat': A_hat,
        'feats': feats,
        'config': {
            'k_graph': k_graph, 'hidden': hidden, 'out_dim': out_dim,
            'lr': lr, 'n_clusters': N_CLUST, 'seed': SEED,
        },
        'train_idx': idx_tr, 'val_idx': idx_va, 'test_idx': idx_te,
        'eval_loss': float(best_score),
    }, os.path.join(path, 'model_{}.pt'.format(disease)))
    np.save(os.path.join(path, 'eval_loss.npy'), best_score)
    np.save(os.path.join(path, 'loss.npy'), best_score)
    return path, name, float(best_score)


def run_sweep(args):
    if args.quick:
        grid = [(15, 128, 32, 1e-3), (20, 128, 64, 5e-4)]
    else:
        grid = list(product(K_GRID, HIDDEN_GRID, OUT_GRID, LR_GRID))
    print('Cluster C sweep: {} configs | USE_GPU={}'.format(len(grid), USE_GPU))
    done = load_done() if args.resume else set()
    fields = [
        'config', 'k_graph', 'hidden', 'out_dim', 'lr',
        'eval_loss', 'nmi', 'ari', 'acc', 'knn_acc', 'nmi_assign', 'entropy_q', 'nmi_all',
    ]
    for i, (k_graph, hidden, out_dim, lr) in enumerate(grid, 1):
        name = config_name(k_graph, hidden, out_dim, lr)
        print('\n===== [{}/{}] {} ====='.format(i, len(grid), name))
        if name in done:
            print('Skip', name)
            continue
        model_file = os.path.join(results_parent(), name, 'model_{}.pt'.format(disease))
        if args.resume and os.path.exists(model_file) and not args.retrain:
            path = os.path.join(results_parent(), name)
            eval_loss = float(np.load(os.path.join(path, 'eval_loss.npy')))
        else:
            path, name, eval_loss = train_one(k_graph, hidden, out_dim, lr, args.epochs)
        metrics = evaluate_config(disease, path)
        print('RESULT nmi={:.4f} nmi_assign={:.4f} eval={:.4f}'.format(
            metrics['nmi'], metrics['nmi_assign'], eval_loss))
        row = {
            'config': name, 'k_graph': k_graph, 'hidden': hidden, 'out_dim': out_dim,
            'lr': lr, 'eval_loss': eval_loss, **metrics,
        }
        append_leaderboard(row, fields)
        with open(os.path.join(path, 'metrics.json'), 'w') as f:
            json.dump(row, f, indent=2)
    summarize(args.rank_metric)


def main(args):
    if args.summarize_only:
        summarize(args.rank_metric)
        return
    run_sweep(args)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=EPOCHS)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--retrain', action='store_true')
    p.add_argument('--quick', action='store_true')
    p.add_argument('--summarize-only', action='store_true')
    p.add_argument('--rank-metric', default='eval_loss', choices=['eval_loss', 'nmi', 'nmi_assign'])
    args, unknown = p.parse_known_args()
    if unknown:
        raise ValueError(unknown)
    main(args)
