"""Hyperparameter sweep for Method 2 + OMIDIENT Dirichlet private.

Within each config: checkpoint / early-stop on weighted val loss
  L = λ_rec L_recon + λ_ctr L_ctr + λ_vic L_vic + λ_kl L_kl

Across configs: rank by fixed core validation loss:
  eval_loss = L_recon + L_ctr
(L_vic and L_kl are training regularizers only; not used for ranking).

Private: product-Dirichlet (OMIDIENT), K fixed (default 4 for BRCA).
Shared: Method-2 AE + contrastive + joint D(concat(P,S)).
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
from multiomics.code.method2_omidient_private.model import SharedAndSpecificLoss
from multiomics.code.method2_omidient_private.training import train, validation, EarlyStopper
from multiomics.code.method2_omidient_private.evaluate import (
    evaluate_config,
    build_model,
)
from multiomics.code.shared_rep_metrics import LEADERBOARD_METRIC_KEYS

warnings.filterwarnings('ignore')

disease = 'brca'
EPOCHS = 100
BATCH_SIZE = 32
USE_GPU = False
SEED = 21
TEMPERATURE = 0.4
K_GROUPS = 4
LAMBDA_KL = 1.0
PRIOR_ALPHA = 0.5

LR = {'brca': 0.0004, 'kirc': 0.0002, 'coad': 0.0002, 'lihc': 0.0002}[disease]
WEIGHT_DECAY = {'brca': 0.0007, 'kirc': 0.0007, 'coad': 0.0007, 'lihc': 0.0007}[disease]

# Same Method-2 weight grid; K / λ_kl fixed to OMIDIENT BRCA defaults
LAMBDA_REC_GRID = [0.3, 0.5, 0.7, 1.0, 1.5]
LAMBDA_CTR_GRID = [0.5, 0.7, 1.0, 1.5, 2.0]
LAMBDA_VIC_GRID = [0.1, 0.5, 1.0, 2.0, 5.0]

MINIMIZE_METRICS = frozenset({'eval_loss', 'val_loss'})
DEFAULT_RANK_METRIC = 'eval_loss'
RANK_CHOICES = [
    'nmi',
    's_nmi',
    'ari',
    'acc',
    'knn_acc',
    's_knn_acc',
    'f_score',
    'eval_loss',
    'probe_subtype_from_S_other_acc',
    'align_cka_mean',
]


def fixed_eval_loss(avg_terms):
    """Config-comparable core val score (no L_vic / L_kl)."""
    return float(avg_terms['l_recon'] + avg_terms['l_ctr'])


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def config_name(lr, wd, lambda_rec, lambda_ctr, lambda_vic, k_groups):
    return '{}_{}_r{}_c{}_v{}_k{}'.format(
        lr, wd, lambda_rec, lambda_ctr, lambda_vic, k_groups
    )


def results_parent():
    return '../../results/models_{}_method2_omidient'.format(disease)


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
        raise KeyError(
            'Leaderboard has no column {!r}; re-run sweep or re-evaluate configs.'.format(
                metric
            )
        )

    minimize = metric in MINIMIZE_METRICS

    def key(r):
        return float(r[metric])

    best = min(rows, key=key) if minimize else max(rows, key=key)
    out = {
        'rank_metric': metric,
        'rank_direction': 'min' if minimize else 'max',
        'eval_loss_definition': (
            'L_recon + L_ctr (unweighted; ranking; L_vic and L_kl excluded)'
        ),
        'config': best['config'],
        'lambda_rec': float(best['lambda_rec']),
        'lambda_ctr': float(best['lambda_ctr']),
        'lambda_vic': float(best['lambda_vic']),
        'k_groups': int(float(best['k_groups'])),
        'lambda_kl': float(best.get('lambda_kl', LAMBDA_KL)),
        'eval_loss': float(best['eval_loss']),
        'val_loss': float(best['val_loss']),
        'nmi': float(best['nmi']),
        'ari': float(best['ari']),
        'f_score': float(best['f_score']),
        'acc': float(best['acc']),
        'knn_acc': float(best['knn_acc']),
        'ch_index': float(best['ch_index']),
    }
    for k in (
        's_nmi',
        's_ari',
        's_acc',
        's_knn_acc',
        'probe_subtype_from_S_other_acc',
        'probe_XA_from_S_other_r2',
        'probe_XA_from_P_other_r2',
        'probe_XA_S_minus_P_r2',
        'align_cosine_mean',
        'align_cca_mean',
        'align_cka_mean',
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
                i,
                r['config'],
                float(r['nmi']),
                s_nmi,
                float(r['ari']),
                float(r['acc']),
                float(r['knn_acc']),
                float(r['eval_loss']),
            )
        )
    print('\nBEST: {}'.format(best['config']))
    print(json.dumps(out, indent=2))
    return out


def train_one(batch, epochs, lr, wd, lambda_rec, lambda_ctr, lambda_vic, k_groups):
    setup_seed(SEED)
    loss_best = np.inf
    best_state = None
    best_terms = None
    early_stopper = EarlyStopper(patience=20, min_delta=0.005)
    temperature = TEMPERATURE

    view1_data, view2_data, view3_data, view_train_concatenate, y_true = load_data(disease)

    model = build_model(
        [view1_data.shape[1], view2_data.shape[1], view3_data.shape[1]],
        k_groups,
    )
    if USE_GPU:
        model = model.cuda()

    name = config_name(lr, wd, lambda_rec, lambda_ctr, lambda_vic, k_groups)
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
        k_groups=k_groups,
        lambda_rec=lambda_rec,
        lambda_ctr=lambda_ctr,
        lambda_vic=lambda_vic,
        lambda_kl=LAMBDA_KL,
        prior_alpha=PRIOR_ALPHA,
    )

    history_path = os.path.join(path, 'epoch_history.csv')
    hist_fields = [
        'epoch',
        'train_loss',
        'val_loss',
        'val_recon',
        'val_ctr',
        'val_vic',
        'val_kl',
        'eval_loss',
    ]
    with open(history_path, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=hist_fields).writeheader()

    for epoch in range(epochs):
        train_loss = train(
            train_loader,
            view1_data,
            view2_data,
            model,
            loss_function,
            temperature,
            optimizer,
            epoch,
            USE_GPU,
            epochs,
        )
        val_loss, avg_terms = validation(
            val_loader,
            view1_data,
            view2_data,
            model,
            temperature,
            loss_function,
            USE_GPU,
            epoch,
            epochs,
        )
        epoch_eval = fixed_eval_loss(avg_terms)
        print('epoch {0} done'.format(epoch))

        with open(history_path, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=hist_fields).writerow({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_recon': avg_terms['l_recon'],
                'val_ctr': avg_terms['l_ctr'],
                'val_vic': avg_terms['l_vic'],
                'val_kl': avg_terms['l_kl'],
                'eval_loss': epoch_eval,
            })

        if val_loss < loss_best:
            loss_best = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_terms = dict(avg_terms)

        if early_stopper.early_stop(val_loss):
            print('Early stopping at epoch {}'.format(epoch))
            break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_terms is None:
        _, best_terms = validation(
            val_loader,
            view1_data,
            view2_data,
            model,
            temperature,
            loss_function,
            USE_GPU,
            0,
            epochs,
        )

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
            [lr, wd, lambda_rec, lambda_ctr, lambda_vic, k_groups, LAMBDA_KL],
            dtype=object,
        ),
    )
    with open(os.path.join(path, 'val_terms.json'), 'w') as f:
        json.dump({k: float(v) for k, v in best_terms.items()}, f, indent=2)
    return path, name, float(loss_best), float(eval_loss)


def load_hparams(path):
    h = np.load(os.path.join(path, 'hparams.npy'), allow_pickle=True)
    return {
        'lr': float(h[0]),
        'wd': float(h[1]),
        'lambda_rec': float(h[2]),
        'lambda_ctr': float(h[3]),
        'lambda_vic': float(h[4]),
        'k_groups': int(h[5]),
        'lambda_kl': float(h[6]) if len(h) > 6 else LAMBDA_KL,
    }


def load_or_rebuild_val_data(path):
    candidates = [
        os.path.join(path, 'val_data_{}.npy'.format(disease)),
        os.path.join(path, 'val_data_{}'.format(disease)),
    ]
    for cand in candidates:
        if os.path.exists(cand):
            return np.load(cand)

    print('val_data not found under {}; rebuilding train/val/test split'.format(path))
    _, _, _, view_train_concatenate, y_true = load_data(disease)
    X_train, X_test, y_train, y_test = train_test_split(
        view_train_concatenate, y_true, test_size=0.2, random_state=1
    )
    _, X_val, _, _ = train_test_split(
        X_train, y_train, test_size=0.25, random_state=1
    )
    return X_val


def compute_eval_loss_from_checkpoint(path, batch=BATCH_SIZE, hparams=None):
    eval_path = os.path.join(path, 'eval_loss.npy')
    terms_path = os.path.join(path, 'val_terms.json')
    if os.path.exists(terms_path):
        with open(terms_path, 'r') as f:
            avg_terms = json.load(f)
        eval_loss = fixed_eval_loss(avg_terms)
        np.save(eval_path, eval_loss)
        return eval_loss

    history_path = os.path.join(path, 'epoch_history.csv')
    if os.path.exists(history_path):
        with open(history_path, 'r', newline='') as f:
            hist = list(csv.DictReader(f))
        if hist and all(k in hist[0] for k in ('val_loss', 'val_recon', 'val_ctr')):
            best = min(hist, key=lambda r: float(r['val_loss']))
            avg_terms = {
                'l_recon': float(best['val_recon']),
                'l_ctr': float(best['val_ctr']),
                'l_vic': float(best.get('val_vic', 0.0)),
                'l_kl': float(best.get('val_kl', 0.0)),
            }
            eval_loss = fixed_eval_loss(avg_terms)
            np.save(eval_path, eval_loss)
            with open(os.path.join(path, 'val_terms.json'), 'w') as f:
                json.dump(avg_terms, f, indent=2)
            return eval_loss

    hp = hparams if hparams is not None else load_hparams(path)
    view1_data, view2_data, view3_data, _, _ = load_data(disease)
    X_val = load_or_rebuild_val_data(path)
    val_loader = torch.utils.data.DataLoader(
        dataset=X_val, batch_size=batch, shuffle=False, drop_last=True
    )
    model = build_model(
        [view1_data.shape[1], view2_data.shape[1], view3_data.shape[1]],
        hp['k_groups'],
    )
    state = torch.load(
        os.path.join(path, 'model_{}'.format(disease)),
        map_location='cuda' if USE_GPU else 'cpu',
    )
    model.load_state_dict(state)
    if USE_GPU:
        model = model.cuda()
    loss_function = SharedAndSpecificLoss(
        k_groups=hp['k_groups'],
        lambda_rec=hp['lambda_rec'],
        lambda_ctr=hp['lambda_ctr'],
        lambda_vic=hp['lambda_vic'],
        lambda_kl=hp.get('lambda_kl', LAMBDA_KL),
        prior_alpha=PRIOR_ALPHA,
    )
    _, avg_terms = validation(
        val_loader,
        view1_data,
        view2_data,
        model,
        TEMPERATURE,
        loss_function,
        USE_GPU,
        0,
        1,
    )
    eval_loss = fixed_eval_loss(avg_terms)
    np.save(eval_path, eval_loss)
    with open(os.path.join(path, 'val_terms.json'), 'w') as f:
        json.dump({k: float(v) for k, v in avg_terms.items()}, f, indent=2)
    return eval_loss


def build_grid():
    return list(product(LAMBDA_REC_GRID, LAMBDA_CTR_GRID, LAMBDA_VIC_GRID))


def run_sweep(args):
    batch, epochs = args.batch_size, args.epochs
    lr, wd = LR, WEIGHT_DECAY
    k_groups = args.k_groups

    grid = build_grid()
    total = len(grid)
    print(
        'Sweep size: {} | lambda_rec={} lambda_ctr={} lambda_vic={} | '
        'K={} lambda_kl={} | lr={} wd={}'.format(
            total,
            LAMBDA_REC_GRID,
            LAMBDA_CTR_GRID,
            LAMBDA_VIC_GRID,
            k_groups,
            LAMBDA_KL,
            lr,
            wd,
        )
    )
    print(
        'Rank metric default: {} = L_recon + L_ctr (unweighted; no L_vic/L_kl); '
        'checkpoint/early-stop uses full weighted val loss.'.format(DEFAULT_RANK_METRIC)
    )

    already = load_leaderboard_configs() if args.resume else set()
    if already:
        print('Resume: {} configs already in leaderboard, will skip.'.format(len(already)))

    fieldnames = [
        'config',
        'lambda_rec',
        'lambda_ctr',
        'lambda_vic',
        'k_groups',
        'lambda_kl',
        'lr',
        'weight_decay',
        'eval_loss',
        'val_loss',
    ] + list(LEADERBOARD_METRIC_KEYS)

    for idx, (lambda_rec, lambda_ctr, lambda_vic) in enumerate(grid, 1):
        name = config_name(lr, wd, lambda_rec, lambda_ctr, lambda_vic, k_groups)
        print('\n===== [{}/{}] {} ====='.format(idx, total, name))

        if name in already:
            print('Skip (already evaluated): {}'.format(name))
            continue

        model_file = os.path.join(results_parent(), name, 'model_{}'.format(disease))
        if args.resume and os.path.exists(model_file) and not args.retrain:
            print('Found existing model, evaluating only: {}'.format(name))
            path = os.path.join(results_parent(), name)
            val_loss = float(np.load(os.path.join(path, 'loss.npy')))
            eval_loss = compute_eval_loss_from_checkpoint(
                path,
                batch=batch,
                hparams={
                    'lambda_rec': lambda_rec,
                    'lambda_ctr': lambda_ctr,
                    'lambda_vic': lambda_vic,
                    'k_groups': k_groups,
                    'lambda_kl': LAMBDA_KL,
                },
            )
        else:
            path, name, val_loss, eval_loss = train_one(
                batch, epochs, lr, wd, lambda_rec, lambda_ctr, lambda_vic, k_groups
            )

        metrics = evaluate_config(disease, path, plot=False, k_groups=k_groups)
        print(
            'RESULT {} | nmi={:.4f} s_nmi={:.4f} ari={:.4f} acc={:.4f} knn={:.2f} '
            'probe_S={:.4f} cka={:.4f}'.format(
                name,
                metrics['nmi'],
                metrics['s_nmi'],
                metrics['ari'],
                metrics['acc'],
                metrics['knn_acc'],
                metrics['probe_subtype_from_S_other_acc'],
                metrics['align_cka_mean'],
            )
        )

        row = {
            'config': name,
            'lambda_rec': lambda_rec,
            'lambda_ctr': lambda_ctr,
            'lambda_vic': lambda_vic,
            'k_groups': k_groups,
            'lambda_kl': LAMBDA_KL,
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
    parser = argparse.ArgumentParser(
        description='Method 2 + OMIDIENT private hyperparameter sweep'
    )
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--k-groups', type=int, default=K_GROUPS)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--retrain', action='store_true')
    parser.add_argument('--summarize-only', action='store_true')
    parser.add_argument(
        '--rank-metric',
        type=str,
        default=DEFAULT_RANK_METRIC,
        choices=RANK_CHOICES,
        help='Metric used to pick the best config (default: core eval_loss)',
    )
    args, unknown = parser.parse_known_args()
    if unknown:
        raise ValueError(f'Unkown args: {unknown}')
    main(args)
