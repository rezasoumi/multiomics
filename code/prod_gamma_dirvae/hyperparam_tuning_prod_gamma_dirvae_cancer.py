"""OMIDIENT ProdGammaDirVae hyperparameter tuning.

Optimizer / architecture knobs are fixed at the BRCA published best:
  lr=0.0003, wd=0.0005, K=4

Optional --grid-search sweeps loss weights:
  L = λ_ctr L_ctr + λ_recon L_recon + λ_kl KL  (+ λ_ortho L_ortho if --use-orthogonal)

Configs are ranked by unweighted eval_loss = L_ctr + L_recon + KL
(λ weights and L_ortho excluded from ranking, same spirit as Method 2).
"""

import torch.optim
import numpy as np
import argparse
import json
from itertools import product
from sklearn.model_selection import train_test_split
from multiomics.code.prod_gamma_dirvae.training import train, validation, EarlyStopper
from multiomics.code.loading_data import load_data
from multiomics.code.prod_gamma_dirvae.prod_gamma_dirvae_cancer import (
    setup_seed,
    SharedAndSpecificLoss,
    SharedAndSpecificEmbedding,
)
import os
import warnings

warnings.filterwarnings("ignore")

disease = 'brca'
EPOCHS = 100
# Fixed at pre-ortho BRCA best (do not re-sweep these for λ grids)
LR = 0.0003
WEIGHT_DECAY = 5e-4
K_GROUPS = 4
BATCH_SIZE = 32
USE_GPU = True
SEED = 21

# Default λ matching original hardcoded objective: 1·L_ctr + 0.7·L_recon + 1·KL (+ 1·L_ortho)
DEFAULT_LAMBDA_CTR = 1.0
DEFAULT_LAMBDA_RECON = 0.7
DEFAULT_LAMBDA_KL = 1.0
DEFAULT_LAMBDA_ORTHO = 1.0

# Loss-weight grids (Method-2-style spans; always include the original defaults)
LAMBDA_CTR_GRID_FULL = [0.5, 0.7, 1.0, 1.5, 2.0]
LAMBDA_RECON_GRID_FULL = [0.3, 0.5, 0.7, 1.0, 1.5]
LAMBDA_KL_GRID_FULL = [0.5, 0.7, 1.0, 1.5, 2.0]
LAMBDA_ORTHO_GRID_FULL = [0.5, 0.7, 1.0, 1.5, 2.0]
# Coarse: keep low / default-ish / high → 3^4 = 81 (ortho) instead of 625
LAMBDA_CTR_GRID_COARSE = [0.5, 1.0, 2.0]
LAMBDA_RECON_GRID_COARSE = [0.3, 0.7, 1.5]
LAMBDA_KL_GRID_COARSE = [0.5, 1.0, 2.0]
LAMBDA_ORTHO_GRID_COARSE = [0.5, 1.0, 2.0]
# Back-compat aliases (full)
LAMBDA_CTR_GRID = LAMBDA_CTR_GRID_FULL
LAMBDA_RECON_GRID = LAMBDA_RECON_GRID_FULL
LAMBDA_KL_GRID = LAMBDA_KL_GRID_FULL
LAMBDA_ORTHO_GRID = LAMBDA_ORTHO_GRID_FULL


def _fmt(x):
    """Stable token for folder names (1.0 not 1)."""
    s = '{:.4g}'.format(float(x))
    return s


def config_directory(use_orthogonal, lambda_ctr, lambda_recon, lambda_kl, lambda_ortho):
    # Base: fixed lr_wd_K; λ tokens; optional _ortho before λ so evaluate can still parse K
    base = '{}_{}_{}'.format(LR, WEIGHT_DECAY, K_GROUPS)
    tokens = [
        'c{}'.format(_fmt(lambda_ctr)),
        'r{}'.format(_fmt(lambda_recon)),
        'k{}'.format(_fmt(lambda_kl)),
    ]
    if use_orthogonal:
        tokens.append('o{}'.format(_fmt(lambda_ortho)))
        return '{}_ortho_{}'.format(base, '_'.join(tokens))
    return '{}_{}'.format(base, '_'.join(tokens))


def results_parent():
    return '../../results/models_{}_ProdGammaDirVae'.format(disease)


def _grids_for_mode(grid_mode):
    """Return (ctr, recon, kl, ortho) lists for the requested search mode."""
    if grid_mode in (False, None, 'off', 'single'):
        return None
    if grid_mode in (True, 'full'):
        return (
            LAMBDA_CTR_GRID_FULL,
            LAMBDA_RECON_GRID_FULL,
            LAMBDA_KL_GRID_FULL,
            LAMBDA_ORTHO_GRID_FULL,
        )
    if grid_mode == 'coarse':
        return (
            LAMBDA_CTR_GRID_COARSE,
            LAMBDA_RECON_GRID_COARSE,
            LAMBDA_KL_GRID_COARSE,
            LAMBDA_ORTHO_GRID_COARSE,
        )
    if grid_mode == 'ortho-only':
        # Fix published defaults; only sweep λ_ortho (fastest meaningful ortho screen)
        return (
            [DEFAULT_LAMBDA_CTR],
            [DEFAULT_LAMBDA_RECON],
            [DEFAULT_LAMBDA_KL],
            LAMBDA_ORTHO_GRID_FULL,
        )
    raise ValueError(
        "Unknown grid mode {!r}; use off|single|coarse|full|ortho-only".format(grid_mode)
    )


def build_lambda_grid(use_orthogonal, grid_mode):
    grids = _grids_for_mode(grid_mode)
    if grids is None:
        return [(
            DEFAULT_LAMBDA_CTR,
            DEFAULT_LAMBDA_RECON,
            DEFAULT_LAMBDA_KL,
            DEFAULT_LAMBDA_ORTHO,
        )]
    ctr_g, recon_g, kl_g, ortho_g = grids
    if use_orthogonal:
        return list(product(ctr_g, recon_g, kl_g, ortho_g))
    return [
        (c, r, k, DEFAULT_LAMBDA_ORTHO)
        for c, r, k in product(ctr_g, recon_g, kl_g)
    ]


def summarize_runs(use_orthogonal=True):
    """Rank finished λ-configs by saved unweighted eval_loss."""
    parent = results_parent()
    rows = []
    if not os.path.isdir(parent):
        print('No results parent:', parent)
        return None
    for name in sorted(os.listdir(parent)):
        path = os.path.join(parent, name)
        if not os.path.isdir(path):
            continue
        if use_orthogonal and '_ortho_' not in name and not name.endswith('_ortho'):
            continue
        if (not use_orthogonal) and ('_ortho_' in name or name.endswith('_ortho')):
            continue
        # Prefer unweighted eval_loss for cross-λ ranking; fall back to weighted val loss
        eval_path = os.path.join(path, 'eval_loss.npy')
        loss_path = os.path.join(path, 'loss.npy')
        model_path = os.path.join(path, 'model_{}'.format(disease))
        if not os.path.isfile(model_path):
            continue
        if os.path.isfile(eval_path):
            score = float(np.load(eval_path))
            score_name = 'eval_loss'
        elif os.path.isfile(loss_path):
            score = float(np.load(loss_path))
            score_name = 'val_loss'
        else:
            continue
        rows.append({'config': name, 'score': score, 'score_name': score_name})
    if not rows:
        print('No finished configs to summarize under', parent)
        return None
    rows.sort(key=lambda r: r['score'])
    tag = 'ortho' if use_orthogonal else 'default'
    board_path = os.path.join(parent, 'leaderboard_{}.csv'.format(tag))
    best_path = os.path.join(parent, 'best_config_{}.json'.format(tag))
    with open(board_path, 'w') as f:
        f.write('rank,config,{},score_name\n'.format('score'))
        for i, r in enumerate(rows, 1):
            f.write('{},{},{},{}\n'.format(i, r['config'], r['score'], r['score_name']))
    best = {
        'config': rows[0]['config'],
        'eval_loss': rows[0]['score'],
        'score_name': rows[0]['score_name'],
        'use_orthogonal': use_orthogonal,
        'fixed': {'lr': LR, 'weight_decay': WEIGHT_DECAY, 'K': K_GROUPS},
    }
    with open(best_path, 'w') as f:
        json.dump(best, f, indent=2)
    print('Wrote', board_path)
    print('Best:', best)
    print('Wrote', best_path)
    return best


def work(p):
    (
        batch, epochs, use_orthogonal, resume,
        lambda_ctr, lambda_recon, lambda_kl, lambda_ortho,
    ) = p
    setup_seed(SEED)
    loss_best = np.inf
    eval_best = np.inf
    model_path = ''
    # Plateau stop: no improvement of min_delta for `patience` epochs (old min_delta=10 never fired)
    early_stopper = EarlyStopper(patience=15, min_delta=0.002)
    temperature = 0.4
    method = 'prod_gamma_dirvae'

    directory = config_directory(
        use_orthogonal, lambda_ctr, lambda_recon, lambda_kl, lambda_ortho
    )
    parent_dir = results_parent()
    path = os.path.join(parent_dir, directory)
    os.makedirs(path, exist_ok=True)

    final_model = os.path.join(path, 'model_{}'.format(disease))
    if resume and os.path.isfile(final_model) and os.path.isfile(os.path.join(path, 'eval_loss.npy')):
        print('Skip (resume): {} already finished'.format(directory))
        return

    print(
        'Saving to {} | ortho={} | λ_ctr={} λ_recon={} λ_kl={} λ_ortho={}'.format(
            path, use_orthogonal, lambda_ctr, lambda_recon, lambda_kl, lambda_ortho
        )
    )

    view1_data, view2_data, view3_data, view_train_concatenate, y_true = load_data(disease)

    model = SharedAndSpecificEmbedding(
        method, K_GROUPS,
        view_size=[view1_data.shape[1], view2_data.shape[1], view3_data.shape[1]],
        n_units_1=[512, 256, 128, 8], n_units_2=[512, 256, 128, 8],
        n_units_3=[256, 128, 64, 8], mlp_size=[32, 8],
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_function = SharedAndSpecificLoss(
        K=K_GROUPS,
        use_orthogonal=use_orthogonal,
        lambda_ctr=lambda_ctr,
        lambda_recon=lambda_recon,
        lambda_kl=lambda_kl,
        lambda_ortho=lambda_ortho,
    )

    for epoch in range(epochs):
        train(
            train_loader, view1_data, view2_data, model, loss_function,
            temperature, optimizer, epoch, USE_GPU, epochs,
        )
        loss_val, eval_val = validation(
            val_loader, view1_data, view2_data, model, temperature,
            loss_function, USE_GPU, epoch, epochs,
        )
        print('epoch {0} done'.format(epoch))

        # Checkpoint by weighted training objective; track unweighted for ranking
        if loss_val < loss_best:
            loss_best = loss_val
            eval_best = eval_val
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
    np.save('{}/eval_loss'.format(path), eval_best)
    with open('{}/train_flags.txt'.format(path), 'w') as f:
        f.write('use_orthogonal={}\n'.format(use_orthogonal))
        f.write('lr={}\nwd={}\nK={}\n'.format(LR, WEIGHT_DECAY, K_GROUPS))
        f.write('lambda_ctr={}\n'.format(lambda_ctr))
        f.write('lambda_recon={}\n'.format(lambda_recon))
        f.write('lambda_kl={}\n'.format(lambda_kl))
        f.write('lambda_ortho={}\n'.format(lambda_ortho))


def main(args):
    if args.summarize_only:
        summarize_runs(use_orthogonal=bool(args.use_orthogonal))
        return

    batch = args.batch_size
    epochs = args.epochs
    use_orthogonal = bool(args.use_orthogonal)
    resume = bool(args.resume)
    grid_mode = args.grid_search
    # argparse store_true legacy: True → full
    if grid_mode is True:
        grid_mode = 'full'
    elif grid_mode is False or grid_mode is None:
        grid_mode = 'off'
    grid = build_lambda_grid(use_orthogonal, grid_mode)
    grids = _grids_for_mode(grid_mode)

    print(
        'Fixed: lr={} wd={} K={} | use_orthogonal={} | grid={} | {} λ-config(s)'.format(
            LR, WEIGHT_DECAY, K_GROUPS, use_orthogonal, grid_mode, len(grid)
        )
    )
    if grids is not None:
        ctr_g, recon_g, kl_g, ortho_g = grids
        print(
            'λ grids: ctr={} recon={} kl={}{}'.format(
                ctr_g,
                recon_g,
                kl_g,
                ' ortho={}'.format(ortho_g) if use_orthogonal else '',
            )
        )

    for idx, (lambda_ctr, lambda_recon, lambda_kl, lambda_ortho) in enumerate(grid, 1):
        print('=== Config {}/{} ==='.format(idx, len(grid)))
        work((
            batch, epochs, use_orthogonal, resume,
            lambda_ctr, lambda_recon, lambda_kl, lambda_ortho,
        ))

    summarize_runs(use_orthogonal=use_orthogonal)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OMIDIENT λ hyperparameter tuning')
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument(
        '--use-orthogonal',
        action='store_true',
        help='Include λ_ortho L_ortho; save under {lr}_{wd}_{K}_ortho_c*_r*_k*_o*',
    )
    parser.add_argument(
        '--grid-search',
        nargs='?',
        const='full',
        default='off',
        choices=['off', 'single', 'coarse', 'full', 'ortho-only'],
        help=(
            'λ sweep mode (lr/wd/K fixed). '
            'off/single=1 default; '
            'ortho-only=5 (fix c/r/k, sweep o); '
            'coarse=3^4=81; full=5^4=625. '
            'Bare --grid-search means full.'
        ),
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Skip configs that already have model_* and eval_loss.npy',
    )
    parser.add_argument(
        '--summarize-only',
        action='store_true',
        help='Only rank finished configs by eval_loss (no training)',
    )

    args, unknown = parser.parse_known_args()
    if unknown:
        raise ValueError(f'Unkown args: {unknown}')
    main(args)
