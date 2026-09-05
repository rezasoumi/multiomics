import os
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
from multiomics.code.method1_predictability.training import (
    train,
    validation,
    EarlyStopper,
)

warnings.filterwarnings("ignore")

disease = 'brca'  # 'kirc' 'coad' 'lihc'
EPOCHS = 100
LR = {'brca': [.0004], 'kirc': [.0002], 'coad': [0.0002], 'lihc': [0.0002]}[disease]
BATCH_SIZE = 32
USE_GPU = False
parallel = False
WEIGHT_DECAY = {'brca': [.0007], 'kirc': [.0007], 'coad': [0.0007], 'lihc': [0.0007]}[disease]
SEED = 21

# Predictability weights (search grids can replace these)
ALPHA = 1.0
BETA = 1.0
GAMMA = 0.7
LAMBDA_CTR = 1.0

# use the below instead if you want to search over hyperparameters
# LR = [.0003, .0002, .0001, .0004, .0005, .0006, .0007]
# WEIGHT_DECAY = [5e-4, 4e-4, 3e-4, 6e-4, 7e-4]
# ALPHA_GRID = [0.4, 0.7, 1.0, 1.5, 2.0]
# BETA_GRID = [0.4, 0.7, 1.0, 1.5, 2.0]
# GAMMA_GRID = [0.4, 0.7, 1.0]


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def work(p):
    setup_seed(SEED)
    batch, epochs, lr, wd, alpha, beta, gamma, lambda_ctr = p
    loss_best = np.inf
    model_path = ''
    early_stopper = EarlyStopper(patience=30, min_delta=10)
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

    directory = '{}_{}_a{}_b{}_g{}'.format(lr, wd, alpha, beta, gamma)
    parent_dir = '../../results/models_{}_method1'.format(disease)
    path = os.path.join(parent_dir, directory)
    os.makedirs(path, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(
        view_train_concatenate, y_true, test_size=0.2, random_state=1
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=1
    )

    # drop_last=True: InstanceLoss builds a mask sized to the nominal batch
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
        loss_val = validation(
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

        if loss_val < loss_best:
            loss_best = loss_val
            model_path = '{}/model_{}_epoch_{}'.format(path, disease, epoch)

        if early_stopper.early_stop(loss_val):
            break

    torch.save(model.state_dict(), model_path)
    torch.save(model.state_dict(), '{}/model_{}'.format(path, disease))
    np.save('{}/train_data_{}'.format(path, disease), X_train)
    np.save('{}/train_label_{}'.format(path, disease), y_train)
    np.save('{}/val_data_{}'.format(path, disease), X_val)
    np.save('{}/val_label_{}'.format(path, disease), y_val)
    np.save('{}/test_data_{}'.format(path, disease), X_test)
    np.save('{}/test_label_{}'.format(path, disease), y_test)
    np.save('{}/loss'.format(path), loss_best)
    np.save(
        '{}/hparams'.format(path),
        np.array([lr, wd, alpha, beta, gamma, lambda_ctr], dtype=object),
    )


def main(args):
    batch = args.batch_size
    epochs = args.epochs
    configs = [
        (batch, epochs, lr, wd, ALPHA, BETA, GAMMA, LAMBDA_CTR)
        for lr in LR
        for wd in WEIGHT_DECAY
    ]

    if parallel:
        pool = torch.multiprocessing.Pool(10)
        pool.map(work, configs)
        pool.close()
    else:
        for p in configs:
            work(p)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Method 1 Hyperparameter Tuning')
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    args, unknown = parser.parse_known_args()
    if unknown:
        raise ValueError(f'Unkown args: {unknown}')
    main(args)
