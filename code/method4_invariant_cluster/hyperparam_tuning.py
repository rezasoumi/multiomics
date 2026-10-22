"""Hyperparameter sweep for Method 4 v2 (invariant DEC + residual private).

v2 vs v1:
  - stronger DEC (higher λ_dec, sharpen=2.5), milder agree
  - weaker λ_res, stronger λ_leak
  - results under models_{disease}_method4_v2/

Warm-up: first WARMUP_EPOCHS use contrastive + residual + view only, then
init DEC centroids via KMeans on mean(S), then full loss.

Checkpoint: val_score (excludes L_leak)
Rank: eval_loss = L_dec + L_agree + L_res + L_ctr
"""

import os
import csv
import json
import argparse
import warnings
from itertools import product

import numpy as np
import torch
import torch.optim
from sklearn.model_selection import train_test_split

from multiomics.code.loading_data import load_data
from multiomics.code.method4_invariant_cluster.model import (
    SharedAndSpecificEmbedding,
    SharedAndSpecificLoss,
)
from multiomics.code.method4_invariant_cluster.training import (
    train, validation, EarlyStopper, init_centroids_kmeans,
)
from multiomics.code.method4_invariant_cluster.evaluate import evaluate_config
from multiomics.code.shared_rep_metrics import LEADERBOARD_METRIC_KEYS

warnings.filterwarnings('ignore')

disease = 'brca'
EPOCHS = 100
BATCH_SIZE = 32
USE_GPU = torch.cuda.is_available()
SEED = 21
TEMPERATURE = 0.4
WARMUP_EPOCHS = 15
NUM_CLUST = {'lihc': 2, 'coad': 4, 'kirc': 2, 'brca': 5}[disease]

LR = {'brca': 0.0004, 'kirc': 0.0002, 'coad': 0.0002, 'lihc': 0.0002}[disease]
WEIGHT_DECAY = {'brca': 0.0007, 'kirc': 0.0007, 'coad': 0.0007, 'lihc': 0.0007}[disease]

# v2: stronger clustering, milder agree, weaker residual, stronger leak
LAMBDA_DEC_GRID = [1.0, 1.5, 2.0]
LAMBDA_AGREE_GRID = [0.5, 1.0, 1.5]
LAMBDA_RES_GRID = [0.1, 0.3, 0.5]
LAMBDA_LEAK_GRID = [1.0, 2.0, 3.0]
LAMBDA_VIEW_FIXED = 0.5
LAMBDA_CTR_FIXED = 0.5
LAMBDA_VIC_FIXED = 0.1
SHARPEN_FIXED = 2.5

LAMBDA_VIEW_GRID = [0.3, 0.5, 1.0]
LAMBDA_CTR_GRID = [0.3, 0.5, 1.0]
LAMBDA_VIC_GRID = [0.0, 0.1]

MINIMIZE_METRICS = frozenset({'eval_loss', 'val_loss', 'agree_symkl_mean'})
DEFAULT_RANK_METRIC = 'eval_loss'
RANK_CHOICES = [
    'nmi', 's_nmi', 'ari', 'acc', 'knn_acc', 's_knn_acc', 'f_score',
    'eval_loss', 'probe_subtype_from_S_other_acc', 'align_cka_mean',
    'agree_symkl_mean', 'entropy_q_bar',
]


def fixed_eval_loss(avg_terms):
    return float(
        avg_terms['l_dec'] + avg_terms['l_agree'] + avg_terms['l_res'] + avg_terms['l_ctr']
    )


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def config_name(lr, wd, lambda_dec, lambda_agree, lambda_res, lambda_view, lambda_leak, lambda_ctr, lambda_vic):
    return '{}_{}_d{}_a{}_r{}_vw{}_lk{}_c{}_v{}'.format(
        lr, wd, lambda_dec, lambda_agree, lambda_res, lambda_view, lambda_leak, lambda_ctr, lambda_vic
    )


def results_parent():
    return '../../results/models_{}_method4_v2'.format(disease)


def leaderboard_path():
    return os.path.join(results_parent(), 'leaderboard.csv')


def best_json_path():
    return os.path.join(results_parent(), 'best_config.json')


def append_leaderboard(row, fieldnames):
    path = leaderboard_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def load_leaderboard_configs():
    path = leaderboard_path()
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path, 'r', newline='') as f:
        for row in csv.DictReader(f):
            if row.get('config'):
                done.add(row['config'])
    return done


def summarize_best(metric=DEFAULT_RANK_METRIC):
    path = leaderboard_path()
    if not os.path.exists(path):
        raise FileNotFoundError('No leaderboard at {}'.format(path))
    with open(path, 'r', newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError('Leaderboard is empty')
    if metric not in rows[0]:
        raise KeyError('Leaderboard has no column {!r}'.format(metric))

    minimize = metric in MINIMIZE_METRICS

    def key(r):
        return float(r[metric])

    best = min(rows, key=key) if minimize else max(rows, key=key)
    out = {
        'rank_metric': metric,
        'rank_direction': 'min' if minimize else 'max',
        'eval_loss_definition': 'L_dec + L_agree + L_res + L_ctr (L_vic/L_view/L_leak excluded)',
        'config': best['config'],
        'eval_loss': float(best['eval_loss']),
        'val_loss': float(best['val_loss']),
        'nmi': float(best['nmi']),
        'ari': float(best['ari']),
        'acc': float(best['acc']),
        'knn_acc': float(best['knn_acc']),
    }
    for k in (
        'lambda_dec', 'lambda_agree', 'lambda_res', 'lambda_view', 'lambda_leak',
        'lambda_ctr', 'lambda_vic',
        's_nmi', 's_ari', 's_acc', 's_knn_acc', 'p_nmi',
        'probe_subtype_from_S_other_acc',
        'probe_XA_from_S_other_r2', 'probe_XA_from_P_other_r2', 'probe_XA_S_minus_P_r2',
        'align_cosine_mean', 'align_cca_mean', 'align_cka_mean',
        'agree_symkl_mean', 'entropy_q_bar',
    ):
        if k in best and best[k] not in (None, ''):
            out[k] = float(best[k])
    with open(best_json_path(), 'w') as f:
        json.dump(out, f, indent=2)

    ranked = sorted(rows, key=key, reverse=not minimize)
    print('\n========== TOP 10 by {} ({}) =========='.format(
        metric, 'lower better' if minimize else 'higher better'
    ))
    for i, r in enumerate(ranked[:10], 1):
        s_nmi = float(r['s_nmi']) if r.get('s_nmi') not in (None, '') else float('nan')
        print(
            '{:2d}. {} | nmi={:.4f} s_nmi={:.4f} ari={:.4f} acc={:.4f} knn={:.2f} eval={:.4f}'.format(
                i, r['config'], float(r['nmi']), s_nmi,
                float(r['ari']), float(r['acc']), float(r['knn_acc']), float(r['eval_loss']),
            )
        )
    print('\nBEST: {}'.format(best['config']))
    print(json.dumps(out, indent=2))
    return out


def train_one(
    batch, epochs, lr, wd,
    lambda_dec, lambda_agree, lambda_res, lambda_view, lambda_leak, lambda_ctr, lambda_vic,
):
    setup_seed(SEED)
    loss_best = np.inf
    best_state = None
    best_terms = None
    early_stopper = EarlyStopper(patience=20, min_delta=0.005)

    view1_data, view2_data, view3_data, view_train_concatenate, y_true = load_data(disease)

    model = SharedAndSpecificEmbedding(
        view_size=[view1_data.shape[1], view2_data.shape[1], view3_data.shape[1]],
        n_units_1=[512, 256, 128, 32],
        n_units_2=[512, 256, 128, 32],
        n_units_3=[256, 128, 64, 32],
        mlp_size=[32, 8],
        n_clusters=NUM_CLUST,
    )
    if USE_GPU:
        model = model.cuda()

    name = config_name(
        lr, wd, lambda_dec, lambda_agree, lambda_res, lambda_view, lambda_leak, lambda_ctr, lambda_vic
    )
    path = os.path.join(results_parent(), name)
    os.makedirs(path, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(
        view_train_concatenate, y_true, test_size=0.2, random_state=1
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=1
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=X_train, batch_size=batch, shuffle=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=X_val, batch_size=batch, shuffle=False, drop_last=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_function = SharedAndSpecificLoss(
        lambda_dec=lambda_dec,
        lambda_agree=lambda_agree,
        lambda_res=lambda_res,
        lambda_view=lambda_view,
        lambda_leak=lambda_leak,
        lambda_ctr=lambda_ctr,
        lambda_vic=lambda_vic,
        warm_up=True,
        sharpen=SHARPEN_FIXED,
    )

    history_path = os.path.join(path, 'epoch_history.csv')
    hist_fields = [
        'epoch', 'train_loss', 'val_loss', 'warm_up',
        'val_dec', 'val_agree', 'val_res', 'val_view', 'val_leak', 'val_ctr', 'val_vic',
        'eval_loss',
    ]
    with open(history_path, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=hist_fields).writeheader()

    centroids_inited = False
    for epoch in range(epochs):
        if epoch < WARMUP_EPOCHS:
            loss_function.warm_up = True
        else:
            if not centroids_inited:
                init_centroids_kmeans(
                    model, train_loader, view1_data, view2_data, USE_GPU, NUM_CLUST
                )
                centroids_inited = True
            loss_function.warm_up = False

        grl_lambda = 0.0 if loss_function.warm_up else min(1.0, (epoch - WARMUP_EPOCHS + 1) / 20.0)
        train_loss = train(
            train_loader, view1_data, view2_data, model, loss_function,
            TEMPERATURE, optimizer, epoch, USE_GPU, epochs, grl_lambda=grl_lambda,
        )
        val_loss, avg_terms = validation(
            val_loader, view1_data, view2_data, model, TEMPERATURE, loss_function,
            USE_GPU, epoch, epochs, grl_lambda=grl_lambda,
        )
        epoch_eval = fixed_eval_loss(avg_terms)
        print('epoch {0} done (warm_up={1})'.format(epoch, loss_function.warm_up))

        with open(history_path, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=hist_fields).writerow({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'warm_up': int(loss_function.warm_up),
                'val_dec': avg_terms['l_dec'],
                'val_agree': avg_terms['l_agree'],
                'val_res': avg_terms['l_res'],
                'val_view': avg_terms['l_view'],
                'val_leak': avg_terms['l_leak'],
                'val_ctr': avg_terms['l_ctr'],
                'val_vic': avg_terms['l_vic'],
                'eval_loss': epoch_eval,
            })

        # Only early-stop / checkpoint after warm-up (DEC phase)
        if not loss_function.warm_up:
            if val_loss < loss_best:
                loss_best = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_terms = dict(avg_terms)
            if early_stopper.early_stop(val_loss):
                print('Early stopping at epoch {}'.format(epoch))
                break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        loss_function.warm_up = False
        _, best_terms = validation(
            val_loader, view1_data, view2_data, model, TEMPERATURE, loss_function,
            USE_GPU, 0, epochs,
        )
        loss_best = float(best_terms['val_score'])

    eval_loss = fixed_eval_loss(best_terms)
    torch.save(best_state, os.path.join(path, 'model_{}'.format(disease)))
    np.save(os.path.join(path, 'train_data_{}'.format(disease)), X_train)
    np.save(os.path.join(path, 'train_label_{}'.format(disease)), y_train)
    np.save(os.path.join(path, 'val_data_{}'.format(disease)), X_val)
    np.save(os.path.join(path, 'val_label_{}'.format(disease)), y_val)
    np.save(os.path.join(path, 'test_data_{}'.format(disease)), X_test)
    np.save(os.path.join(path, 'test_label_{}'.format(disease)), y_test)
    np.save(os.path.join(path, 'loss.npy'), loss_best)
    np.save(os.path.join(path, 'eval_loss.npy'), eval_loss)
    np.save(
        os.path.join(path, 'hparams.npy'),
        np.array(
            [lr, wd, lambda_dec, lambda_agree, lambda_res, lambda_view, lambda_leak, lambda_ctr, lambda_vic],
            dtype=object,
        ),
    )
    with open(os.path.join(path, 'val_terms.json'), 'w') as f:
        json.dump({k: float(v) for k, v in best_terms.items()}, f, indent=2)
    return path, name, float(loss_best), float(eval_loss)


def build_grid(mode='coarse'):
    if mode == 'coarse':
        # 3 x 3 x 3 x 3 = 81
        return [
            (d, a, r, LAMBDA_VIEW_FIXED, lk, LAMBDA_CTR_FIXED, LAMBDA_VIC_FIXED)
            for d, a, r, lk in product(
                LAMBDA_DEC_GRID, LAMBDA_AGREE_GRID, LAMBDA_RES_GRID, LAMBDA_LEAK_GRID
            )
        ]
    return list(product(
        LAMBDA_DEC_GRID, LAMBDA_AGREE_GRID, LAMBDA_RES_GRID,
        LAMBDA_VIEW_GRID, LAMBDA_LEAK_GRID, LAMBDA_CTR_GRID, LAMBDA_VIC_GRID,
    ))


def run_sweep(args):
    batch, epochs = args.batch_size, args.epochs
    lr, wd = LR, WEIGHT_DECAY
    grid = build_grid(args.grid_search)
    total = len(grid)
    print(
        'Sweep size: {} ({}) | USE_GPU={} | warmup={} | lr={} wd={}'.format(
            total, args.grid_search, USE_GPU, WARMUP_EPOCHS, lr, wd
        )
    )

    already = load_leaderboard_configs() if args.resume else set()
    if already:
        print('Resume: {} configs already in leaderboard, will skip.'.format(len(already)))

    fieldnames = [
        'config', 'lambda_dec', 'lambda_agree', 'lambda_res', 'lambda_view',
        'lambda_leak', 'lambda_ctr', 'lambda_vic',
        'lr', 'weight_decay', 'eval_loss', 'val_loss',
        'agree_symkl_mean', 'entropy_q_bar',
    ] + list(LEADERBOARD_METRIC_KEYS)

    for idx, (lambda_dec, lambda_agree, lambda_res, lambda_view, lambda_leak, lambda_ctr, lambda_vic) in enumerate(grid, 1):
        name = config_name(
            lr, wd, lambda_dec, lambda_agree, lambda_res, lambda_view, lambda_leak, lambda_ctr, lambda_vic
        )
        print('\n===== [{}/{}] {} ====='.format(idx, total, name))
        if name in already:
            print('Skip (already evaluated): {}'.format(name))
            continue

        model_file = os.path.join(results_parent(), name, 'model_{}'.format(disease))
        if args.resume and os.path.exists(model_file) and not args.retrain:
            print('Found existing model, evaluating only: {}'.format(name))
            path = os.path.join(results_parent(), name)
            val_loss = float(np.load(os.path.join(path, 'loss.npy')))
            terms_path = os.path.join(path, 'val_terms.json')
            if os.path.exists(terms_path):
                with open(terms_path, 'r') as f:
                    avg_terms = json.load(f)
                eval_loss = fixed_eval_loss(avg_terms)
            else:
                eval_loss = float(np.load(os.path.join(path, 'eval_loss.npy')))
        else:
            path, name, val_loss, eval_loss = train_one(
                batch, epochs, lr, wd,
                lambda_dec, lambda_agree, lambda_res, lambda_view, lambda_leak, lambda_ctr, lambda_vic,
            )

        metrics = evaluate_config(disease, path, plot=False)
        print(
            'RESULT {} | nmi={:.4f} s_nmi={:.4f} p_nmi={:.4f} agree={:.4f} ent={:.4f} eval={:.4f}'.format(
                name, metrics['nmi'], metrics['s_nmi'], metrics.get('p_nmi', float('nan')),
                metrics.get('agree_symkl_mean', float('nan')),
                metrics.get('entropy_q_bar', float('nan')),
                eval_loss,
            )
        )

        row = {
            'config': name,
            'lambda_dec': lambda_dec,
            'lambda_agree': lambda_agree,
            'lambda_res': lambda_res,
            'lambda_view': lambda_view,
            'lambda_leak': lambda_leak,
            'lambda_ctr': lambda_ctr,
            'lambda_vic': lambda_vic,
            'lr': lr,
            'weight_decay': wd,
            'eval_loss': eval_loss,
            'val_loss': val_loss,
            **metrics,
        }
        append_leaderboard(row, fieldnames)
        with open(os.path.join(path, 'metrics.json'), 'w') as f:
            json.dump(row, f, indent=2)

    summarize_best(metric=args.rank_metric)


def main(args):
    if args.summarize_only:
        summarize_best(metric=args.rank_metric)
        return
    run_sweep(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Method 4 hyperparameter sweep')
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--retrain', action='store_true')
    parser.add_argument('--summarize-only', action='store_true')
    parser.add_argument(
        '--grid-search', type=str, default='coarse', choices=['coarse', 'full'],
    )
    parser.add_argument(
        '--rank-metric', type=str, default=DEFAULT_RANK_METRIC, choices=RANK_CHOICES,
    )
    args, unknown = parser.parse_known_args()
    if unknown:
        raise ValueError(f'Unkown args: {unknown}')
    main(args)
