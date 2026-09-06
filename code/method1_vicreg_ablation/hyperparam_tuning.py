"""Hyperparameter sweep for Method 1 VICReg ablation (alpha, beta, lambda_vic)."""

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
from multiomics.code.method1_vicreg_ablation.model import (
    SharedAndSpecificEmbedding,
    SharedAndSpecificLoss,
)
from multiomics.code.method1_vicreg_ablation.training import (
    train,
    validation,
    EarlyStopper,
)
from multiomics.code.method1_vicreg_ablation.evaluate import evaluate_config

warnings.filterwarnings('ignore')

disease = 'brca'
EPOCHS = 100
BATCH_SIZE = 32
USE_GPU = True
SEED = 21

LR = {'brca': 0.0004, 'kirc': 0.0002, 'coad': 0.0002, 'lihc': 0.0002}[disease]
WEIGHT_DECAY = {'brca': 0.0007, 'kirc': 0.0007, 'coad': 0.0007, 'lihc': 0.0007}[disease]
LAMBDA_CTR = 1.0

# 5 x 5 x 5 = 125 configs
ALPHA_GRID = [0.3, 0.5, 0.7, 1.0, 1.5]
BETA_GRID = [0.5, 0.7, 1.0, 1.5, 2.0]
LAMBDA_VIC_GRID = [0.1, 0.5, 1.0, 2.0, 5.0]


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def config_name(lr, wd, alpha, beta, lambda_vic, lambda_ctr):
    return '{}_{}_a{}_b{}_v{}_l{}'.format(lr, wd, alpha, beta, lambda_vic, lambda_ctr)


def results_parent():
    return '../../results/models_{}_method1_vicreg'.format(disease)


def leaderboard_path():
    return os.path.join(results_parent(), 'leaderboard.csv')


def best_json_path():
    return os.path.join(results_parent(), 'best_config.json')


def append_leaderboard(row, fieldnames):
    path = leaderboard_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
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


def summarize_best(metric='nmi'):
    path = leaderboard_path()
    if not os.path.exists(path):
        raise FileNotFoundError('No leaderboard at {}'.format(path))
    with open(path, 'r', newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError('Leaderboard is empty')

    best = max(rows, key=lambda r: float(r[metric]))
    out = {
        'rank_metric': metric,
        'config': best['config'],
        'alpha': float(best['alpha']),
        'beta': float(best['beta']),
        'lambda_vic': float(best['lambda_vic']),
        'lambda_ctr': float(best['lambda_ctr']),
        'val_constructive': float(best['val_loss']),
        'nmi': float(best['nmi']),
        'ari': float(best['ari']),
        'f_score': float(best['f_score']),
        'acc': float(best['acc']),
        'knn_acc': float(best['knn_acc']),
        'ch_index': float(best['ch_index']),
    }
    with open(best_json_path(), 'w') as f:
        json.dump(out, f, indent=2)

    ranked = sorted(rows, key=lambda r: float(r[metric]), reverse=True)
    print('\n========== TOP 10 by {} =========='.format(metric))
    for i, r in enumerate(ranked[:10], 1):
        print(
            '{:2d}. {} | nmi={:.4f} ari={:.4f} acc={:.4f} knn={:.2f} constructive={:.4f}'.format(
                i,
                r['config'],
                float(r['nmi']),
                float(r['ari']),
                float(r['acc']),
                float(r['knn_acc']),
                float(r['val_loss']),
            )
        )
    print('\nBEST: {}'.format(best['config']))
    print(json.dumps(out, indent=2))
    return out


def train_one(batch, epochs, lr, wd, alpha, beta, lambda_vic, lambda_ctr):
    setup_seed(SEED)
    loss_best = np.inf
    best_state = None
    early_stopper = EarlyStopper(patience=20, min_delta=0.005)
    temperature = 0.4

    view1_data, view2_data, view3_data, view_train_concatenate, y_true = load_data(disease)

    model = SharedAndSpecificEmbedding(
        view_size=[view1_data.shape[1], view2_data.shape[1], view3_data.shape[1]],
        n_units_1=[512, 256, 128, 32],
        n_units_2=[512, 256, 128, 32],
        n_units_3=[256, 128, 64, 32],
        mlp_size=[32, 8],
    )
    if USE_GPU:
        model = model.cuda()

    name = config_name(lr, wd, alpha, beta, lambda_vic, lambda_ctr)
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

    optimizer_main = torch.optim.Adam(model.main_parameters(), lr=lr, weight_decay=wd)
    optimizer_adv = torch.optim.Adam(model.adversary_parameters(), lr=lr, weight_decay=wd)
    loss_function = SharedAndSpecificLoss(
        alpha=alpha, beta=beta, lambda_ctr=lambda_ctr, lambda_vic=lambda_vic
    )

    for epoch in range(epochs):
        train(
            train_loader,
            view1_data,
            view2_data,
            model,
            loss_function,
            temperature,
            optimizer_main,
            optimizer_adv,
            epoch,
            USE_GPU,
            epochs,
        )
        _, constructive, _ = validation(
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
        print('epoch {0} done'.format(epoch))

        if constructive < loss_best:
            loss_best = constructive
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if early_stopper.early_stop(constructive):
            print('Early stopping at epoch {}'.format(epoch))
            break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    torch.save(best_state, os.path.join(path, 'model_{}'.format(disease)))
    np.save(os.path.join(path, 'train_data_{}'.format(disease)), X_train)
    np.save(os.path.join(path, 'train_label_{}'.format(disease)), y_train)
    np.save(os.path.join(path, 'val_data_{}'.format(disease)), X_val)
    np.save(os.path.join(path, 'val_label_{}'.format(disease)), y_val)
    np.save(os.path.join(path, 'test_data_{}'.format(disease)), X_test)
    np.save(os.path.join(path, 'test_label_{}'.format(disease)), y_test)
    np.save(os.path.join(path, 'loss.npy'), loss_best)
    np.save(
        os.path.join(path, 'hparams.npy'),
        np.array([lr, wd, alpha, beta, lambda_vic, lambda_ctr], dtype=object),
    )
    return path, name, float(loss_best)


def build_grid():
    return list(product(ALPHA_GRID, BETA_GRID, LAMBDA_VIC_GRID))


def run_sweep(args):
    batch = args.batch_size
    epochs = args.epochs
    lr, wd, lambda_ctr = LR, WEIGHT_DECAY, LAMBDA_CTR

    grid = build_grid()
    total = len(grid)
    print(
        'Sweep size: {} | alpha={} beta={} lambda_vic={} | lr={} wd={} lambda_ctr={}'.format(
            total, ALPHA_GRID, BETA_GRID, LAMBDA_VIC_GRID, lr, wd, lambda_ctr
        )
    )

    already = load_leaderboard_configs() if args.resume else set()
    if already:
        print('Resume: {} configs already in leaderboard, will skip.'.format(len(already)))

    fieldnames = [
        'config',
        'alpha',
        'beta',
        'lambda_vic',
        'lambda_ctr',
        'lr',
        'weight_decay',
        'val_loss',
        'nmi',
        'ari',
        'f_score',
        'acc',
        'v_measure',
        'ch_index',
        'knn_acc',
    ]

    for idx, (alpha, beta, lambda_vic) in enumerate(grid, 1):
        name = config_name(lr, wd, alpha, beta, lambda_vic, lambda_ctr)
        print('\n===== [{}/{}] {} ====='.format(idx, total, name))

        if name in already:
            print('Skip (already evaluated): {}'.format(name))
            continue

        model_file = os.path.join(results_parent(), name, 'model_{}'.format(disease))
        if args.resume and os.path.exists(model_file) and not args.retrain:
            print('Found existing model, evaluating only: {}'.format(name))
            path = os.path.join(results_parent(), name)
            val_loss = float(np.load(os.path.join(path, 'loss.npy')))
        else:
            path, name, val_loss = train_one(
                batch, epochs, lr, wd, alpha, beta, lambda_vic, lambda_ctr
            )

        metrics = evaluate_config(disease, path, plot=False)
        print(
            'RESULT {} | constructive={:.4f} nmi={:.4f} ari={:.4f} acc={:.4f} knn={:.2f}'.format(
                name,
                val_loss,
                metrics['nmi'],
                metrics['ari'],
                metrics['acc'],
                metrics['knn_acc'],
            )
        )

        row = {
            'config': name,
            'alpha': alpha,
            'beta': beta,
            'lambda_vic': lambda_vic,
            'lambda_ctr': lambda_ctr,
            'lr': lr,
            'weight_decay': wd,
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
    parser = argparse.ArgumentParser(description='Method 1 VICReg ablation sweep')
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--retrain', action='store_true')
    parser.add_argument('--summarize-only', action='store_true')
    parser.add_argument(
        '--rank-metric',
        type=str,
        default='nmi',
        choices=['nmi', 'ari', 'acc', 'knn_acc', 'f_score'],
    )
    args, unknown = parser.parse_known_args()
    if unknown:
        raise ValueError(f'Unkown args: {unknown}')
    main(args)
