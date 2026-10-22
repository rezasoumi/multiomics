"""Hyperparameter sweep for Cluster A (contrastive + DEC on fused Z)."""

import argparse
import csv
import json
import os
import warnings
from itertools import product

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from multiomics.code.loading_data import load_data
from multiomics.code.cluster_common import NUM_CLUST, index_split
from multiomics.code.cluster_a_contrastive_dec.model import MultiViewClusterModel, ClusterALoss
from multiomics.code.cluster_a_contrastive_dec.training import (
    train_epoch, val_epoch, EarlyStopper, init_centroids_kmeans,
)
from multiomics.code.cluster_a_contrastive_dec.evaluate import evaluate_config

warnings.filterwarnings('ignore')

disease = 'brca'
EPOCHS = 100
BATCH_SIZE = 32
USE_GPU = torch.cuda.is_available()
SEED = 21
TEMPERATURE = 0.4
WARMUP = 20
N_CLUST = NUM_CLUST[disease]
LR = 0.0004
WD = 0.0007

LAMBDA_CTR_GRID = [0.5, 1.0, 1.5, 2.0]
LAMBDA_DEC_GRID = [0.5, 1.0, 1.5, 2.0]
LAMBDA_REC_GRID = [0.0, 0.3, 0.7]


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def config_name(lr, wd, lc, ld, lr_):
    return '{}_{}_c{}_d{}_r{}'.format(lr, wd, lc, ld, lr_)


def results_parent():
    return '../../results/models_{}_cluster_a'.format(disease)


def fixed_eval_loss(terms):
    return float(terms['l_ctr'] + terms['l_dec'])


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
            '{:2d}. {} | nmi={:.4f} nmi_assign={:.4f} ari={:.4f} acc={:.4f} eval={:.4f}'.format(
                i, r['config'], float(r['nmi']), float(r.get('nmi_assign', 'nan')),
                float(r['ari']), float(r['acc']), float(r['eval_loss']),
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
        'knn_acc': float(best['knn_acc']),
        'lambda_ctr': float(best['lambda_ctr']),
        'lambda_dec': float(best['lambda_dec']),
        'lambda_rec': float(best['lambda_rec']),
    }
    with open(os.path.join(results_parent(), 'best_config.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print('\nBEST:', json.dumps(out, indent=2))
    return out


def train_one(batch, epochs, lr, wd, lambda_ctr, lambda_dec, lambda_rec):
    setup_seed(SEED)
    view1, view2, view3, concat, y_true = load_data(disease)
    d1, d2 = view1.shape[1], view2.shape[1]
    idx_tr, idx_va, idx_te = index_split(len(concat))
    X_tr, X_va, X_te = concat[idx_tr], concat[idx_va], concat[idx_te]
    y_tr = np.asarray(y_true)[idx_tr]
    y_va = np.asarray(y_true)[idx_va]
    y_te = np.asarray(y_true)[idx_te]

    model = MultiViewClusterModel(
        view_size=[view1.shape[1], view2.shape[1], view3.shape[1]],
        n_units_1=[512, 256, 128, 32],
        n_units_2=[512, 256, 128, 32],
        n_units_3=[256, 128, 64, 32],
        mlp_size=[32, 8],
        n_clusters=N_CLUST,
    )
    if USE_GPU:
        model = model.cuda()

    name = config_name(lr, wd, lambda_ctr, lambda_dec, lambda_rec)
    path = os.path.join(results_parent(), name)
    os.makedirs(path, exist_ok=True)

    train_loader = torch.utils.data.DataLoader(X_tr, batch_size=batch, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(X_va, batch_size=batch, shuffle=False, drop_last=True)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = ClusterALoss(lambda_ctr, lambda_dec, lambda_rec, sharpen=2.5, warm_up=True)
    stopper = EarlyStopper(patience=20, min_delta=0.005)
    best_score, best_state, best_terms = np.inf, None, None
    centroids_inited = False

    hist = os.path.join(path, 'epoch_history.csv')
    fields = ['epoch', 'train_loss', 'val_loss', 'warm_up', 'val_ctr', 'val_dec', 'val_rec', 'eval_loss']
    with open(hist, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    for epoch in range(epochs):
        if epoch < WARMUP:
            loss_fn.warm_up = True
        else:
            if not centroids_inited:
                init_centroids_kmeans(model, train_loader, d1, d2, USE_GPU, N_CLUST)
                centroids_inited = True
            loss_fn.warm_up = False

        tr = train_epoch(train_loader, d1, d2, model, loss_fn, TEMPERATURE, opt, epoch, USE_GPU, epochs)
        # val with warm_up flag matching train for fair score after W
        vs, avg = val_epoch(val_loader, d1, d2, model, loss_fn, TEMPERATURE, USE_GPU, epoch, epochs)
        ev = fixed_eval_loss(avg)
        with open(hist, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=fields).writerow({
                'epoch': epoch + 1, 'train_loss': tr, 'val_loss': vs,
                'warm_up': int(loss_fn.warm_up),
                'val_ctr': avg['l_ctr'], 'val_dec': avg['l_dec'], 'val_rec': avg['l_rec'],
                'eval_loss': ev,
            })

        if not loss_fn.warm_up:
            if vs < best_score:
                best_score = vs
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_terms = dict(avg)
            if stopper.early_stop(vs):
                print('Early stop at', epoch)
                break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        loss_fn.warm_up = False
        _, best_terms = val_epoch(val_loader, d1, d2, model, loss_fn, TEMPERATURE, USE_GPU, 0, 1)
        best_score = float(best_terms['val_score'])

    eval_loss = fixed_eval_loss(best_terms)
    torch.save(best_state, os.path.join(path, 'model_{}'.format(disease)))
    np.save(os.path.join(path, 'train_data_{}'.format(disease)), X_tr)
    np.save(os.path.join(path, 'train_label_{}'.format(disease)), y_tr)
    np.save(os.path.join(path, 'val_data_{}'.format(disease)), X_va)
    np.save(os.path.join(path, 'val_label_{}'.format(disease)), y_va)
    np.save(os.path.join(path, 'test_data_{}'.format(disease)), X_te)
    np.save(os.path.join(path, 'test_label_{}'.format(disease)), y_te)
    np.save(os.path.join(path, 'loss.npy'), best_score)
    np.save(os.path.join(path, 'eval_loss.npy'), eval_loss)
    with open(os.path.join(path, 'val_terms.json'), 'w') as f:
        json.dump({k: float(v) for k, v in best_terms.items()}, f, indent=2)
    return path, name, float(best_score), float(eval_loss)


def run_sweep(args):
    grid = list(product(LAMBDA_CTR_GRID, LAMBDA_DEC_GRID, LAMBDA_REC_GRID))
    print('Cluster A sweep: {} configs | USE_GPU={}'.format(len(grid), USE_GPU))
    done = load_done() if args.resume else set()
    fields = [
        'config', 'lambda_ctr', 'lambda_dec', 'lambda_rec', 'lr', 'weight_decay',
        'eval_loss', 'val_loss', 'nmi', 'ari', 'acc', 'knn_acc', 'nmi_assign', 'ari_assign',
        'acc_assign', 'entropy_q',
    ]
    for i, (lc, ld, lr_) in enumerate(grid, 1):
        name = config_name(LR, WD, lc, ld, lr_)
        print('\n===== [{}/{}] {} ====='.format(i, len(grid), name))
        if name in done:
            print('Skip', name)
            continue
        model_file = os.path.join(results_parent(), name, 'model_{}'.format(disease))
        if args.resume and os.path.exists(model_file) and not args.retrain:
            path = os.path.join(results_parent(), name)
            val_loss = float(np.load(os.path.join(path, 'loss.npy')))
            with open(os.path.join(path, 'val_terms.json')) as f:
                eval_loss = fixed_eval_loss(json.load(f))
        else:
            path, name, val_loss, eval_loss = train_one(
                args.batch_size, args.epochs, LR, WD, lc, ld, lr_
            )
        metrics = evaluate_config(disease, path)
        print('RESULT nmi={:.4f} nmi_assign={:.4f} eval={:.4f}'.format(
            metrics['nmi'], metrics['nmi_assign'], eval_loss))
        row = {
            'config': name, 'lambda_ctr': lc, 'lambda_dec': ld, 'lambda_rec': lr_,
            'lr': LR, 'weight_decay': WD, 'eval_loss': eval_loss, 'val_loss': val_loss,
            **metrics,
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
    p.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--retrain', action='store_true')
    p.add_argument('--summarize-only', action='store_true')
    p.add_argument('--rank-metric', default='eval_loss', choices=['eval_loss', 'nmi', 'nmi_assign'])
    args, unknown = p.parse_known_args()
    if unknown:
        raise ValueError(unknown)
    main(args)
