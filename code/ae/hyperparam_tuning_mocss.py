"""MOCSS (AE) training for a disease (single published lr/wd config).

BRCA default: lr=4e-4, wd=7e-4 → results/models_brca_ae/0.0004_0.0007
KIRC default: lr=2e-4, wd=7e-4 → results/models_kirc_ae/0.0002_0.0007
"""

import argparse
import json
import os
import warnings

import numpy as np
import torch
import torch.optim
from sklearn.model_selection import train_test_split

from multiomics.code.ae.evaluate import evaluate_config
from multiomics.code.ae.mocss_original_refactored import (
    SharedAndSpecificEmbedding,
    SharedAndSpecificLoss,
    setup_seed,
)
from multiomics.code.ae.training import EarlyStopper, train, validation
from multiomics.code.loading_data import load_data
from multiomics.code.shared_rep_metrics import print_metrics_report

warnings.filterwarnings('ignore')

disease = 'brca'
EPOCHS = 100
BATCH_SIZE = 32
USE_GPU = torch.cuda.is_available()
SEED = 21
SUPPORTED_DISEASES = ('brca', 'kirc', 'coad', 'lihc')
LR_BY_DISEASE = {
    'brca': 0.0004,
    'kirc': 0.0002,
    'coad': 0.0002,
    'lihc': 0.0002,
}
WD_BY_DISEASE = {
    'brca': 0.0007,
    'kirc': 0.0007,
    'coad': 0.0007,
    'lihc': 0.0007,
}


def configure_disease(name):
    global disease
    if name not in SUPPORTED_DISEASES:
        raise ValueError('Unknown disease {!r}; expected one of {}'.format(name, SUPPORTED_DISEASES))
    disease = name


def results_parent():
    return '../../results/models_{}_ae'.format(disease)


def config_directory(lr, wd):
    return '{}_{}'.format(lr, wd)


def work(batch, epochs, lr, wd, resume=False):
    setup_seed(SEED)
    loss_best = np.inf
    model_path = ''
    # Plateau stop (old min_delta=10 never fired usefully on modern losses)
    early_stopper = EarlyStopper(patience=30, min_delta=0.002)
    temperature = 0.4

    directory = config_directory(lr, wd)
    parent_dir = results_parent()
    path = os.path.join(parent_dir, directory)
    os.makedirs(path, exist_ok=True)

    final_model = os.path.join(path, 'model_{}'.format(disease))
    loss_file = os.path.join(path, 'loss.npy')
    if resume and os.path.isfile(final_model) and os.path.isfile(loss_file):
        print('Skip (resume): {} already finished'.format(directory))
        return path

    print('Saving to {}'.format(path))
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

    train_loader = torch.utils.data.DataLoader(dataset=X_train, batch_size=batch, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=X_val, batch_size=batch, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_function = SharedAndSpecificLoss()

    for epoch in range(epochs):
        train(
            train_loader, view1_data, view2_data, model, loss_function,
            temperature, optimizer, epoch, USE_GPU, epochs,
        )
        loss_val = validation(
            val_loader, view1_data, view2_data, model, temperature,
            loss_function, USE_GPU, epoch, epochs,
        )
        print('epoch {0} done'.format(epoch))

        if loss_val < loss_best:
            loss_best = loss_val
            model_path = '{}/model_{}_epoch_{}'.format(path, disease, epoch)

        if early_stopper.early_stop(loss_val):
            break

    if not model_path:
        model_path = '{}/model_{}_epoch_last'.format(path, disease)
    torch.save(model.state_dict(), model_path)
    torch.save(model.state_dict(), '{}/model_{}'.format(path, disease))
    # Skip omics dumps (especially huge on KIRC); evaluate rebuilds the test split.
    np.save('{}/loss'.format(path), loss_best)
    return path


def main(args):
    configure_disease(args.disease)
    lr = float(args.lr) if args.lr is not None else LR_BY_DISEASE[disease]
    wd = float(args.weight_decay) if args.weight_decay is not None else WD_BY_DISEASE[disease]
    print(
        'disease={} | lr={} wd={} | epochs={} batch={} | USE_GPU={}'.format(
            disease, lr, wd, args.epochs, args.batch_size, USE_GPU
        )
    )
    path = work(args.batch_size, args.epochs, lr, wd, resume=bool(args.resume))
    folder = config_directory(lr, wd)
    print('=== Evaluate {} ==='.format(folder))
    metrics = evaluate_config(disease, path, plot=bool(args.plot))
    print_metrics_report(metrics)
    out = {
        'method': 'mocss_ae',
        'disease': disease,
        'config': folder,
        'lr': lr,
        'weight_decay': wd,
        **metrics,
    }
    metrics_path = os.path.join(path, 'metrics_shared_rep.json')
    with open(metrics_path, 'w') as f:
        json.dump(out, f, indent=2)
    print('Wrote', metrics_path)
    print('Best JSON-ish path: {}/{}'.format(results_parent(), folder))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MOCSS AE train (+ evaluate)')
    parser.add_argument(
        '--disease',
        type=str,
        default=disease,
        choices=SUPPORTED_DISEASES,
    )
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=None, help='Override disease default lr')
    parser.add_argument(
        '--weight-decay', type=float, default=None, help='Override disease default wd'
    )
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--plot', action='store_true')
    args, unknown = parser.parse_known_args()
    if unknown:
        raise ValueError('Unkown args: {}'.format(unknown))
    main(args)
