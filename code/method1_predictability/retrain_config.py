"""Retrain one Method 1 config for full epochs (no early stopping), then evaluate."""

import os
import csv
import json
import argparse
import warnings

import numpy as np
import torch
import torch.optim
from sklearn.model_selection import train_test_split

from multiomics.code.loading_data import load_data
from multiomics.code.method1_predictability.model import (
    SharedAndSpecificEmbedding,
    SharedAndSpecificLoss,
)
from multiomics.code.method1_predictability.training import train, validation
from multiomics.code.method1_predictability.evaluate import evaluate_config

warnings.filterwarnings('ignore')

DISEASE = 'brca'
USE_GPU = True
SEED = 21


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def main(args):
    setup_seed(SEED)
    disease = args.disease
    alpha, beta, gamma = args.alpha, args.beta, args.gamma
    lambda_ctr = args.lambda_ctr
    lr, wd = args.lr, args.weight_decay
    epochs, batch = args.epochs, args.batch_size

    # Separate folder so we do not overwrite the broken epoch-1 sweep checkpoint
    name = '{}_{}_a{}_b{}_g{}_l{}_full{}'.format(
        lr, wd, alpha, beta, gamma, lambda_ctr, epochs
    )
    parent = '../../results/models_{}_method1'.format(disease)
    path = os.path.join(parent, name)
    os.makedirs(path, exist_ok=True)

    print('Retrain config:', name)
    print('No early stopping; run all {} epochs; save LAST epoch as model_{}'.format(
        epochs, disease
    ))

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
        alpha=alpha, beta=beta, gamma=gamma, lambda_ctr=lambda_ctr
    )
    temperature = 0.4

    history_path = os.path.join(path, 'epoch_history.csv')
    hist_fields = [
        'epoch', 'train_loss', 'val_loss', 'val_constructive',
        'val_std', 'val_shared', 'val_adv', 'val_own', 'val_ctr',
    ]
    with open(history_path, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=hist_fields).writeheader()

    best_constructive = float('inf')
    best_constructive_state = None
    last_val_loss = None
    last_constructive = None

    for epoch in range(epochs):
        train_loss = train(
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
        val_loss, constructive, avg_terms = validation(
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
        last_val_loss = val_loss
        last_constructive = constructive

        row = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_constructive': constructive,
            'val_std': avg_terms['l_std'],
            'val_shared': avg_terms['l_shared'],
            'val_adv': avg_terms['l_adv'],
            'val_own': avg_terms['l_own'],
            'val_ctr': avg_terms['l_ctr'],
        }
        with open(history_path, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=hist_fields).writerow(row)

        if constructive < best_constructive:
            best_constructive = constructive
            best_constructive_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

        print('epoch {0} done'.format(epoch))

    # Primary checkpoint: LAST epoch (no early stop)
    last_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save(last_state, os.path.join(path, 'model_{}'.format(disease)))
    # Also keep best-by-constructive for comparison
    if best_constructive_state is not None:
        torch.save(
            best_constructive_state,
            os.path.join(path, 'model_{}_best_constructive'.format(disease)),
        )

    np.save(os.path.join(path, 'train_data_{}'.format(disease)), X_train)
    np.save(os.path.join(path, 'train_label_{}'.format(disease)), y_train)
    np.save(os.path.join(path, 'val_data_{}'.format(disease)), X_val)
    np.save(os.path.join(path, 'val_label_{}'.format(disease)), y_val)
    np.save(os.path.join(path, 'test_data_{}'.format(disease)), X_test)
    np.save(os.path.join(path, 'test_label_{}'.format(disease)), y_test)
    np.save(os.path.join(path, 'loss.npy'), last_val_loss)
    np.save(
        os.path.join(path, 'hparams.npy'),
        np.array([lr, wd, alpha, beta, gamma, lambda_ctr, epochs], dtype=object),
    )

    meta = {
        'config': name,
        'alpha': alpha,
        'beta': beta,
        'gamma': gamma,
        'lambda_ctr': lambda_ctr,
        'lr': lr,
        'weight_decay': wd,
        'epochs': epochs,
        'early_stop': False,
        'checkpoint': 'last_epoch',
        'last_val_loss': float(last_val_loss),
        'last_val_constructive': float(last_constructive),
        'best_val_constructive': float(best_constructive),
    }
    with open(os.path.join(path, 'retrain_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print('\n=== Evaluate LAST-epoch checkpoint ===')
    metrics_last = evaluate_config(disease, path, plot=True)
    print(
        'LAST | nmi={:.4f} ari={:.4f} acc={:.4f} knn={:.2f}'.format(
            metrics_last['nmi'],
            metrics_last['ari'],
            metrics_last['acc'],
            metrics_last['knn_acc'],
        )
    )
    with open(os.path.join(path, 'metrics_last.json'), 'w') as f:
        json.dump({**meta, **metrics_last}, f, indent=2)

    # Temporarily swap constructive-best weights into model_* for evaluate_config
    constructive_path = os.path.join(path, 'model_{}_best_constructive'.format(disease))
    main_path = os.path.join(path, 'model_{}'.format(disease))
    if os.path.exists(constructive_path):
        print('\n=== Evaluate BEST-constructive checkpoint ===')
        backup = main_path + '.last_bak'
        os.rename(main_path, backup)
        os.rename(constructive_path, main_path)
        metrics_c = evaluate_config(disease, path, plot=False)
        print(
            'CONSTRUCTIVE-BEST | nmi={:.4f} ari={:.4f} acc={:.4f} knn={:.2f}'.format(
                metrics_c['nmi'],
                metrics_c['ari'],
                metrics_c['acc'],
                metrics_c['knn_acc'],
            )
        )
        with open(os.path.join(path, 'metrics_best_constructive.json'), 'w') as f:
            json.dump({**meta, 'checkpoint': 'best_constructive', **metrics_c}, f, indent=2)
        # Restore: main = last, constructive file restored
        os.rename(main_path, constructive_path)
        os.rename(backup, main_path)

    print('\nDone. Results in:', path)
    print('Epoch history:', history_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Retrain one Method 1 config (no early stop)')
    parser.add_argument('--disease', type=str, default=DISEASE)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0004)
    parser.add_argument('--weight-decay', type=float, default=0.0007)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--gamma', type=float, default=2.5)
    parser.add_argument('--lambda-ctr', type=float, default=1.0)
    args, unknown = parser.parse_known_args()
    if unknown:
        raise ValueError(f'Unkown args: {unknown}')
    main(args)
