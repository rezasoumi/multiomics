"""Train + evaluate one paper config at one seed into .../<config>/seed_<N>/."""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict

import numpy as np
import torch
import torch.optim
from sklearn.model_selection import train_test_split

from multiomics.code.loading_data import load_data
from multiomics.code.multiseed.paths import metrics_path, model_path, seed_run_dir

DISEASE = 'brca'
TEMPERATURE = 0.4


def _setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def _use_gpu() -> bool:
    return torch.cuda.is_available()


def _split_data(view_train_concatenate, y_true):
    X_train, X_test, y_train, y_test = train_test_split(
        view_train_concatenate, y_true, test_size=0.2, random_state=1
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=1
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def _save_splits(path, X_train, X_val, X_test, y_train, y_val, y_test):
    np.save(os.path.join(path, 'train_data_{}'.format(DISEASE)), X_train)
    np.save(os.path.join(path, 'train_label_{}'.format(DISEASE)), y_train)
    np.save(os.path.join(path, 'val_data_{}'.format(DISEASE)), X_val)
    np.save(os.path.join(path, 'val_label_{}'.format(DISEASE)), y_val)
    np.save(os.path.join(path, 'test_data_{}'.format(DISEASE)), X_test)
    np.save(os.path.join(path, 'test_label_{}'.format(DISEASE)), y_test)


def _enrich_metrics(metrics: Dict[str, Any], eval_loss=None) -> Dict[str, Any]:
    out = dict(metrics)
    if 's_nmi' in out and 'nmi' in out:
        out['gap_S_minus_PS'] = float(out['s_nmi']) - float(out['nmi'])
    if eval_loss is not None:
        out['eval_loss'] = float(eval_loss)
    return out


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _write_metrics(path: str, payload: dict):
    with open(metrics_path(path), 'w') as f:
        json.dump(_jsonable(payload), f, indent=2)


def _already_done(path: str, resume: bool) -> bool:
    if not resume:
        return False
    return os.path.isfile(model_path(path)) and os.path.isfile(metrics_path(path))


def train_mocss(run: dict, seed: int, epochs: int, batch_size: int, resume: bool) -> str:
    from multiomics.code.ae.mocss_original_refactored import (
        SharedAndSpecificEmbedding,
        SharedAndSpecificLoss,
    )
    from multiomics.code.ae.training import EarlyStopper, train, validation
    from multiomics.code.ae.evaluate import evaluate_config

    hp = run['hparams']
    path = seed_run_dir(run['results_subdir'], run['base_config'], seed)
    os.makedirs(path, exist_ok=True)
    if _already_done(path, resume):
        print('Skip (resume): {}'.format(path))
        return path

    use_gpu = _use_gpu()
    _setup_seed(seed)
    view1_data, view2_data, view3_data, view_train_concatenate, y_true = load_data(DISEASE)
    model = SharedAndSpecificEmbedding(
        view_size=[view1_data.shape[1], view2_data.shape[1], view3_data.shape[1]],
        n_units_1=[512, 256, 128, 32],
        n_units_2=[512, 256, 128, 32],
        n_units_3=[256, 128, 64, 32],
        mlp_size=[32, 8],
    )
    if use_gpu:
        model = model.cuda()

    X_train, X_val, X_test, y_train, y_val, y_test = _split_data(
        view_train_concatenate, y_true
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=X_train, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=X_val, batch_size=batch_size, shuffle=False
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=hp['lr'], weight_decay=hp['weight_decay']
    )
    loss_function = SharedAndSpecificLoss()
    early_stopper = EarlyStopper(patience=30, min_delta=10)
    loss_best = np.inf
    best_state = None

    for epoch in range(epochs):
        train(
            train_loader, view1_data, view2_data, model, loss_function,
            TEMPERATURE, optimizer, epoch, use_gpu, epochs,
        )
        loss_val = validation(
            val_loader, view1_data, view2_data, model, TEMPERATURE,
            loss_function, use_gpu, epoch, epochs,
        )
        print('epoch {0} done'.format(epoch))
        if loss_val < loss_best:
            loss_best = loss_val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if early_stopper.early_stop(loss_val):
            break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save(best_state, model_path(path))
    _save_splits(path, X_train, X_val, X_test, y_train, y_val, y_test)
    np.save(os.path.join(path, 'loss.npy'), loss_best)

    metrics = evaluate_config(DISEASE, path, plot=False)
    payload = _enrich_metrics(metrics, eval_loss=None)
    payload.update({
        'run_id': run['id'],
        'method': run['method'],
        'base_config': run['base_config'],
        'seed': int(seed),
        'val_loss': float(loss_best),
    })
    _write_metrics(path, payload)
    return path


def train_method1(run: dict, seed: int, epochs: int, batch_size: int, resume: bool) -> str:
    from multiomics.code.method1_predictability.model import (
        SharedAndSpecificEmbedding,
        SharedAndSpecificLoss,
    )
    from multiomics.code.method1_predictability.training import (
        EarlyStopper,
        train,
        validation,
    )
    from multiomics.code.method1_predictability.evaluate import evaluate_config
    from multiomics.code.method1_predictability.hyperparam_tuning import fixed_eval_loss

    hp = run['hparams']
    path = seed_run_dir(run['results_subdir'], run['base_config'], seed)
    os.makedirs(path, exist_ok=True)
    if _already_done(path, resume):
        print('Skip (resume): {}'.format(path))
        return path

    use_gpu = _use_gpu()
    _setup_seed(seed)
    view1_data, view2_data, view3_data, view_train_concatenate, y_true = load_data(DISEASE)
    model = SharedAndSpecificEmbedding(
        view_size=[view1_data.shape[1], view2_data.shape[1], view3_data.shape[1]],
        n_units_1=[512, 256, 128, 32],
        n_units_2=[512, 256, 128, 32],
        n_units_3=[256, 128, 64, 32],
        mlp_size=[32, 8],
    )
    if use_gpu:
        model = model.cuda()

    X_train, X_val, X_test, y_train, y_val, y_test = _split_data(
        view_train_concatenate, y_true
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=X_train, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=X_val, batch_size=batch_size, shuffle=False, drop_last=True
    )
    optimizer_main = torch.optim.Adam(
        model.main_parameters(), lr=hp['lr'], weight_decay=hp['weight_decay']
    )
    optimizer_adv = torch.optim.Adam(
        model.adversary_parameters(), lr=hp['lr'], weight_decay=hp['weight_decay']
    )
    loss_function = SharedAndSpecificLoss(
        alpha=hp['alpha'],
        beta=hp['beta'],
        gamma=hp['gamma'],
        lambda_ctr=hp['lambda_ctr'],
        lambda_priv=hp['lambda_priv'],
    )
    early_stopper = EarlyStopper(patience=20, min_delta=0.005)
    loss_best = np.inf
    best_state = None
    best_terms = None

    for epoch in range(epochs):
        train(
            train_loader, view1_data, view2_data, model, loss_function,
            TEMPERATURE, optimizer_main, optimizer_adv, epoch, use_gpu, epochs,
        )
        _, constructive, avg_terms = validation(
            val_loader, view1_data, view2_data, model, TEMPERATURE,
            loss_function, use_gpu, epoch, epochs,
        )
        print('epoch {0} done'.format(epoch))
        if constructive < loss_best:
            loss_best = constructive
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_terms = dict(avg_terms)
        if early_stopper.early_stop(constructive):
            print('Early stopping at epoch {}'.format(epoch))
            break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_terms is None:
        _, _, best_terms = validation(
            val_loader, view1_data, view2_data, model, TEMPERATURE,
            loss_function, use_gpu, 0, epochs,
        )
    eval_loss = fixed_eval_loss(best_terms)
    torch.save(best_state, model_path(path))
    _save_splits(path, X_train, X_val, X_test, y_train, y_val, y_test)
    np.save(os.path.join(path, 'loss.npy'), loss_best)
    np.save(os.path.join(path, 'eval_loss.npy'), eval_loss)

    metrics = evaluate_config(DISEASE, path, plot=False)
    payload = _enrich_metrics(metrics, eval_loss=eval_loss)
    payload.update({
        'run_id': run['id'],
        'method': run['method'],
        'base_config': run['base_config'],
        'seed': int(seed),
        'val_loss': float(loss_best),
    })
    _write_metrics(path, payload)
    return path


def train_method1_vicreg(
    run: dict, seed: int, epochs: int, batch_size: int, resume: bool
) -> str:
    from multiomics.code.method1_vicreg_ablation.model import (
        SharedAndSpecificEmbedding,
        SharedAndSpecificLoss,
    )
    from multiomics.code.method1_vicreg_ablation.training import (
        EarlyStopper,
        train,
        validation,
    )
    from multiomics.code.method1_vicreg_ablation.evaluate import evaluate_config
    from multiomics.code.method1_vicreg_ablation.hyperparam_tuning import fixed_eval_loss

    hp = run['hparams']
    path = seed_run_dir(run['results_subdir'], run['base_config'], seed)
    os.makedirs(path, exist_ok=True)
    if _already_done(path, resume):
        print('Skip (resume): {}'.format(path))
        return path

    use_gpu = _use_gpu()
    _setup_seed(seed)
    view1_data, view2_data, view3_data, view_train_concatenate, y_true = load_data(DISEASE)
    model = SharedAndSpecificEmbedding(
        view_size=[view1_data.shape[1], view2_data.shape[1], view3_data.shape[1]],
        n_units_1=[512, 256, 128, 32],
        n_units_2=[512, 256, 128, 32],
        n_units_3=[256, 128, 64, 32],
        mlp_size=[32, 8],
    )
    if use_gpu:
        model = model.cuda()

    X_train, X_val, X_test, y_train, y_val, y_test = _split_data(
        view_train_concatenate, y_true
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=X_train, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=X_val, batch_size=batch_size, shuffle=False, drop_last=True
    )
    optimizer_main = torch.optim.Adam(
        model.main_parameters(), lr=hp['lr'], weight_decay=hp['weight_decay']
    )
    optimizer_adv = torch.optim.Adam(
        model.adversary_parameters(), lr=hp['lr'], weight_decay=hp['weight_decay']
    )
    loss_function = SharedAndSpecificLoss(
        alpha=hp['alpha'],
        beta=hp['beta'],
        lambda_ctr=hp['lambda_ctr'],
        lambda_vic=hp['lambda_vic'],
        lambda_priv=hp['lambda_priv'],
    )
    early_stopper = EarlyStopper(patience=20, min_delta=0.005)
    loss_best = np.inf
    best_state = None
    best_terms = None

    for epoch in range(epochs):
        train(
            train_loader, view1_data, view2_data, model, loss_function,
            TEMPERATURE, optimizer_main, optimizer_adv, epoch, use_gpu, epochs,
        )
        _, constructive, avg_terms = validation(
            val_loader, view1_data, view2_data, model, TEMPERATURE,
            loss_function, use_gpu, epoch, epochs,
        )
        print('epoch {0} done'.format(epoch))
        if constructive < loss_best:
            loss_best = constructive
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_terms = dict(avg_terms)
        if early_stopper.early_stop(constructive):
            print('Early stopping at epoch {}'.format(epoch))
            break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_terms is None:
        _, _, best_terms = validation(
            val_loader, view1_data, view2_data, model, TEMPERATURE,
            loss_function, use_gpu, 0, epochs,
        )
    eval_loss = fixed_eval_loss(best_terms)
    torch.save(best_state, model_path(path))
    _save_splits(path, X_train, X_val, X_test, y_train, y_val, y_test)
    np.save(os.path.join(path, 'loss.npy'), loss_best)
    np.save(os.path.join(path, 'eval_loss.npy'), eval_loss)

    metrics = evaluate_config(DISEASE, path, plot=False)
    payload = _enrich_metrics(metrics, eval_loss=eval_loss)
    payload.update({
        'run_id': run['id'],
        'method': run['method'],
        'base_config': run['base_config'],
        'seed': int(seed),
        'val_loss': float(loss_best),
    })
    _write_metrics(path, payload)
    return path


def train_method2(run: dict, seed: int, epochs: int, batch_size: int, resume: bool) -> str:
    from multiomics.code.method2_joint_vicreg.model import (
        SharedAndSpecificEmbedding,
        SharedAndSpecificLoss,
    )
    from multiomics.code.method2_joint_vicreg.training import (
        EarlyStopper,
        train,
        validation,
    )
    from multiomics.code.method2_joint_vicreg.evaluate import evaluate_config
    from multiomics.code.method2_joint_vicreg.hyperparam_tuning import fixed_eval_loss

    hp = run['hparams']
    path = seed_run_dir(run['results_subdir'], run['base_config'], seed)
    os.makedirs(path, exist_ok=True)
    if _already_done(path, resume):
        print('Skip (resume): {}'.format(path))
        return path

    use_gpu = _use_gpu()
    _setup_seed(seed)
    view1_data, view2_data, view3_data, view_train_concatenate, y_true = load_data(DISEASE)
    model = SharedAndSpecificEmbedding(
        view_size=[view1_data.shape[1], view2_data.shape[1], view3_data.shape[1]],
        n_units_1=[512, 256, 128, 32],
        n_units_2=[512, 256, 128, 32],
        n_units_3=[256, 128, 64, 32],
        mlp_size=[32, 8],
    )
    if use_gpu:
        model = model.cuda()

    X_train, X_val, X_test, y_train, y_val, y_test = _split_data(
        view_train_concatenate, y_true
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=X_train, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=X_val, batch_size=batch_size, shuffle=False, drop_last=True
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=hp['lr'], weight_decay=hp['weight_decay']
    )
    loss_function = SharedAndSpecificLoss(
        lambda_rec=hp['lambda_rec'],
        lambda_ctr=hp['lambda_ctr'],
        lambda_vic=hp['lambda_vic'],
    )
    early_stopper = EarlyStopper(patience=20, min_delta=0.005)
    loss_best = np.inf
    best_state = None
    best_terms = None

    for epoch in range(epochs):
        train(
            train_loader, view1_data, view2_data, model, loss_function,
            TEMPERATURE, optimizer, epoch, use_gpu, epochs,
        )
        val_loss, avg_terms = validation(
            val_loader, view1_data, view2_data, model, TEMPERATURE,
            loss_function, use_gpu, epoch, epochs,
        )
        print('epoch {0} done'.format(epoch))
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
            val_loader, view1_data, view2_data, model, TEMPERATURE,
            loss_function, use_gpu, 0, epochs,
        )
    eval_loss = fixed_eval_loss(best_terms)
    torch.save(best_state, model_path(path))
    _save_splits(path, X_train, X_val, X_test, y_train, y_val, y_test)
    np.save(os.path.join(path, 'loss.npy'), loss_best)
    np.save(os.path.join(path, 'eval_loss.npy'), eval_loss)

    metrics = evaluate_config(DISEASE, path, plot=False)
    payload = _enrich_metrics(metrics, eval_loss=eval_loss)
    payload.update({
        'run_id': run['id'],
        'method': run['method'],
        'base_config': run['base_config'],
        'seed': int(seed),
        'val_loss': float(loss_best),
    })
    _write_metrics(path, payload)
    return path


def train_method2_omidient(
    run: dict, seed: int, epochs: int, batch_size: int, resume: bool
) -> str:
    from multiomics.code.method2_omidient_private.model import SharedAndSpecificLoss
    from multiomics.code.method2_omidient_private.training import (
        EarlyStopper,
        train,
        validation,
    )
    from multiomics.code.method2_omidient_private.evaluate import (
        build_model,
        evaluate_config,
    )
    from multiomics.code.method2_omidient_private.hyperparam_tuning import (
        LAMBDA_KL,
        PRIOR_ALPHA,
        fixed_eval_loss,
    )

    hp = run['hparams']
    path = seed_run_dir(run['results_subdir'], run['base_config'], seed)
    os.makedirs(path, exist_ok=True)
    if _already_done(path, resume):
        print('Skip (resume): {}'.format(path))
        return path

    use_gpu = _use_gpu()
    _setup_seed(seed)
    k_groups = int(hp['k_groups'])
    view1_data, view2_data, view3_data, view_train_concatenate, y_true = load_data(DISEASE)
    model = build_model(
        [view1_data.shape[1], view2_data.shape[1], view3_data.shape[1]],
        k_groups,
    )
    if use_gpu:
        model = model.cuda()

    X_train, X_val, X_test, y_train, y_val, y_test = _split_data(
        view_train_concatenate, y_true
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=X_train, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=X_val, batch_size=batch_size, shuffle=False, drop_last=True
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=hp['lr'], weight_decay=hp['weight_decay']
    )
    loss_function = SharedAndSpecificLoss(
        k_groups=k_groups,
        lambda_rec=hp['lambda_rec'],
        lambda_ctr=hp['lambda_ctr'],
        lambda_vic=hp['lambda_vic'],
        lambda_kl=LAMBDA_KL,
        prior_alpha=PRIOR_ALPHA,
    )
    early_stopper = EarlyStopper(patience=20, min_delta=0.005)
    loss_best = np.inf
    best_state = None
    best_terms = None

    for epoch in range(epochs):
        train(
            train_loader, view1_data, view2_data, model, loss_function,
            TEMPERATURE, optimizer, epoch, use_gpu, epochs,
        )
        val_loss, avg_terms = validation(
            val_loader, view1_data, view2_data, model, TEMPERATURE,
            loss_function, use_gpu, epoch, epochs,
        )
        print('epoch {0} done'.format(epoch))
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
            val_loader, view1_data, view2_data, model, TEMPERATURE,
            loss_function, use_gpu, 0, epochs,
        )
    eval_loss = fixed_eval_loss(best_terms)
    torch.save(best_state, model_path(path))
    _save_splits(path, X_train, X_val, X_test, y_train, y_val, y_test)
    np.save(os.path.join(path, 'loss.npy'), loss_best)
    np.save(os.path.join(path, 'eval_loss.npy'), eval_loss)

    metrics = evaluate_config(DISEASE, path, plot=False, k_groups=k_groups)
    payload = _enrich_metrics(metrics, eval_loss=eval_loss)
    payload.update({
        'run_id': run['id'],
        'method': run['method'],
        'base_config': run['base_config'],
        'seed': int(seed),
        'val_loss': float(loss_best),
    })
    _write_metrics(path, payload)
    return path


def train_omidient(run: dict, seed: int, epochs: int, batch_size: int, resume: bool) -> str:
    from multiomics.code.prod_gamma_dirvae.prod_gamma_dirvae_cancer import (
        SharedAndSpecificEmbedding,
        SharedAndSpecificLoss,
    )
    from multiomics.code.prod_gamma_dirvae.training import EarlyStopper, train, validation
    from multiomics.code.prod_gamma_dirvae.evaluate import evaluate_config

    hp = run['hparams']
    path = seed_run_dir(run['results_subdir'], run['base_config'], seed)
    os.makedirs(path, exist_ok=True)
    if _already_done(path, resume):
        print('Skip (resume): {}'.format(path))
        return path

    use_gpu = _use_gpu()
    _setup_seed(seed)
    k_groups = int(hp['k_groups'])
    use_orthogonal = bool(hp['use_orthogonal'])
    view1_data, view2_data, view3_data, view_train_concatenate, y_true = load_data(DISEASE)
    model = SharedAndSpecificEmbedding(
        'prod_gamma_dirvae',
        k_groups,
        view_size=[view1_data.shape[1], view2_data.shape[1], view3_data.shape[1]],
        n_units_1=[512, 256, 128, 8],
        n_units_2=[512, 256, 128, 8],
        n_units_3=[256, 128, 64, 8],
        mlp_size=[32, 8],
    )
    if use_gpu:
        model = model.cuda()

    X_train, X_val, X_test, y_train, y_val, y_test = _split_data(
        view_train_concatenate, y_true
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=X_train, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=X_val, batch_size=batch_size, shuffle=False
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=hp['lr'], weight_decay=hp['weight_decay']
    )
    loss_function = SharedAndSpecificLoss(
        K=k_groups,
        use_orthogonal=use_orthogonal,
        lambda_ctr=hp['lambda_ctr'],
        lambda_recon=hp['lambda_recon'],
        lambda_kl=hp['lambda_kl'],
        lambda_ortho=hp['lambda_ortho'],
    )
    early_stopper = EarlyStopper(patience=15, min_delta=0.002)
    loss_best = np.inf
    eval_best = np.inf
    best_state = None

    for epoch in range(epochs):
        train(
            train_loader, view1_data, view2_data, model, loss_function,
            TEMPERATURE, optimizer, epoch, use_gpu, epochs,
        )
        loss_val, eval_val = validation(
            val_loader, view1_data, view2_data, model, TEMPERATURE,
            loss_function, use_gpu, epoch, epochs,
        )
        print('epoch {0} done'.format(epoch))
        if loss_val < loss_best:
            loss_best = loss_val
            eval_best = eval_val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if early_stopper.early_stop(loss_val):
            break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save(best_state, model_path(path))
    _save_splits(path, X_train, X_val, X_test, y_train, y_val, y_test)
    np.save(os.path.join(path, 'loss.npy'), loss_best)
    np.save(os.path.join(path, 'eval_loss.npy'), eval_best)

    # Pass base_config as folder_name so K parsing ignores seed_* parent layout.
    metrics = evaluate_config(DISEASE, path, run['base_config'], plot=False)
    payload = _enrich_metrics(metrics, eval_loss=eval_best)
    payload.update({
        'run_id': run['id'],
        'method': run['method'],
        'base_config': run['base_config'],
        'seed': int(seed),
        'val_loss': float(loss_best),
        'use_orthogonal': use_orthogonal,
    })
    _write_metrics(path, payload)
    return path


def train_method3(run: dict, seed: int, epochs: int, batch_size: int, resume: bool) -> str:
    from torch.utils.data import DataLoader

    from multiomics.code.method3_spectral_graph.model import (
        SharedAndSpecificEmbedding,
        SharedAndSpecificLoss,
    )
    from multiomics.code.method3_spectral_graph.training import (
        EarlyStopper,
        train,
        validation,
    )
    from multiomics.code.method3_spectral_graph.evaluate import (
        SPECTRAL_DIM,
        evaluate_config,
        _truth_from_label,
    )
    from multiomics.code.method3_spectral_graph.graph_spectral import (
        compute_and_save_targets,
        nmi_on_spectral_U,
    )
    from multiomics.code.method3_spectral_graph.hyperparam_tuning import (
        SpectralOmicsDataset,
        fixed_eval_loss,
        _index_split,
    )
    from multiomics.code.multiseed.paths import repo_results_root

    hp = run['hparams']
    path = seed_run_dir(run['results_subdir'], run['base_config'], seed)
    os.makedirs(path, exist_ok=True)
    if _already_done(path, resume):
        print('Skip (resume): {}'.format(path))
        return path

    use_gpu = _use_gpu()
    _setup_seed(seed)
    num_clust = 5
    k_nn, d_pca, r_spec = 15, 50, SPECTRAL_DIM

    view1_data, view2_data, view3_data, view_train_concatenate, y_true = load_data(DISEASE)
    # Spectral targets are deterministic (random_state=21); reuse shared cache under results/
    U_all, cluster_all, meta = compute_and_save_targets(
        DISEASE,
        view1_data,
        view2_data,
        view3_data,
        n_clusters=num_clust,
        k_nn=k_nn,
        d_pca=d_pca,
        r=r_spec,
        random_state=21,
        repo_results_root=repo_results_root(),
        force=False,
    )
    nmi_u = nmi_on_spectral_U(
        U_all, _truth_from_label(DISEASE, y_true), num_clust
    )
    print('Spectral ceiling nmi_spectral_U = {:.4f} | meta={}'.format(nmi_u, meta))

    idx_train, idx_val, idx_test = _index_split(len(view_train_concatenate))
    X_train = view_train_concatenate[idx_train]
    X_val = view_train_concatenate[idx_val]
    X_test = view_train_concatenate[idx_test]
    y_arr = np.asarray(y_true)
    y_train, y_val, y_test = y_arr[idx_train], y_arr[idx_val], y_arr[idx_test]
    U_train, U_val = U_all[idx_train], U_all[idx_val]
    c_train, c_val = cluster_all[idx_train], cluster_all[idx_val]

    model = SharedAndSpecificEmbedding(
        view_size=[view1_data.shape[1], view2_data.shape[1], view3_data.shape[1]],
        n_units_1=[512, 256, 128, 32],
        n_units_2=[512, 256, 128, 32],
        n_units_3=[256, 128, 64, 32],
        mlp_size=[32, 8],
        spectral_dim=r_spec,
        n_clusters=num_clust,
    )
    if use_gpu:
        model = model.cuda()

    train_loader = DataLoader(
        SpectralOmicsDataset(X_train, U_train, c_train),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        SpectralOmicsDataset(X_val, U_val, c_val),
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=hp['lr'], weight_decay=hp['weight_decay']
    )
    loss_function = SharedAndSpecificLoss(
        lambda_spec=hp['lambda_spec'],
        lambda_res=hp['lambda_res'],
        lambda_ctr=hp['lambda_ctr'],
        lambda_vic=hp['lambda_vic'],
        lambda_view=hp['lambda_view'],
        lambda_leak=hp['lambda_leak'],
    )
    early_stopper = EarlyStopper(patience=20, min_delta=0.005)
    loss_best = np.inf
    best_state = None
    best_terms = None

    for epoch in range(epochs):
        grl_lambda = min(1.0, (epoch + 1) / 20.0)
        train(
            train_loader, view1_data, view2_data, model, loss_function,
            TEMPERATURE, optimizer, epoch, use_gpu, epochs, grl_lambda=grl_lambda,
        )
        val_loss, avg_terms = validation(
            val_loader, view1_data, view2_data, model, TEMPERATURE, loss_function,
            use_gpu, epoch, epochs, grl_lambda=grl_lambda,
        )
        print('epoch {0} done'.format(epoch))
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
            val_loader, view1_data, view2_data, model, TEMPERATURE, loss_function,
            use_gpu, 0, epochs,
        )
    eval_loss = fixed_eval_loss(best_terms)
    torch.save(best_state, model_path(path))
    _save_splits(path, X_train, X_val, X_test, y_train, y_val, y_test)
    np.save(os.path.join(path, 'loss.npy'), loss_best)
    np.save(os.path.join(path, 'eval_loss.npy'), eval_loss)

    metrics = evaluate_config(DISEASE, path, plot=False)
    metrics['nmi_spectral_U'] = float(nmi_u)
    payload = _enrich_metrics(metrics, eval_loss=eval_loss)
    payload.update({
        'run_id': run['id'],
        'method': run['method'],
        'base_config': run['base_config'],
        'seed': int(seed),
        'val_loss': float(loss_best),
    })
    _write_metrics(path, payload)
    return path


def train_method4(run: dict, seed: int, epochs: int, batch_size: int, resume: bool) -> str:
    from multiomics.code.method4_invariant_cluster.model import (
        SharedAndSpecificEmbedding,
        SharedAndSpecificLoss,
    )
    from multiomics.code.method4_invariant_cluster.training import (
        EarlyStopper,
        train,
        validation,
        init_centroids_kmeans,
    )
    from multiomics.code.method4_invariant_cluster.evaluate import evaluate_config
    from multiomics.code.method4_invariant_cluster.hyperparam_tuning import (
        WARMUP_EPOCHS,
        fixed_eval_loss,
    )

    hp = run['hparams']
    path = seed_run_dir(run['results_subdir'], run['base_config'], seed)
    os.makedirs(path, exist_ok=True)
    if _already_done(path, resume):
        print('Skip (resume): {}'.format(path))
        return path

    use_gpu = _use_gpu()
    _setup_seed(seed)
    num_clust = 5

    view1_data, view2_data, view3_data, view_train_concatenate, y_true = load_data(DISEASE)
    model = SharedAndSpecificEmbedding(
        view_size=[view1_data.shape[1], view2_data.shape[1], view3_data.shape[1]],
        n_units_1=[512, 256, 128, 32],
        n_units_2=[512, 256, 128, 32],
        n_units_3=[256, 128, 64, 32],
        mlp_size=[32, 8],
        n_clusters=num_clust,
    )
    if use_gpu:
        model = model.cuda()

    X_train, X_val, X_test, y_train, y_val, y_test = _split_data(
        view_train_concatenate, y_true
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=X_train, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=X_val, batch_size=batch_size, shuffle=False, drop_last=True
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=hp['lr'], weight_decay=hp['weight_decay']
    )
    loss_function = SharedAndSpecificLoss(
        lambda_dec=hp['lambda_dec'],
        lambda_agree=hp['lambda_agree'],
        lambda_res=hp['lambda_res'],
        lambda_view=hp['lambda_view'],
        lambda_leak=hp['lambda_leak'],
        lambda_ctr=hp['lambda_ctr'],
        lambda_vic=hp['lambda_vic'],
        warm_up=True,
    )
    early_stopper = EarlyStopper(patience=20, min_delta=0.005)
    loss_best = np.inf
    best_state = None
    best_terms = None
    centroids_inited = False

    for epoch in range(epochs):
        if epoch < WARMUP_EPOCHS:
            loss_function.warm_up = True
        else:
            if not centroids_inited:
                init_centroids_kmeans(
                    model, train_loader, view1_data, view2_data, use_gpu, num_clust
                )
                centroids_inited = True
            loss_function.warm_up = False

        grl_lambda = (
            0.0
            if loss_function.warm_up
            else min(1.0, (epoch - WARMUP_EPOCHS + 1) / 20.0)
        )
        train(
            train_loader, view1_data, view2_data, model, loss_function,
            TEMPERATURE, optimizer, epoch, use_gpu, epochs, grl_lambda=grl_lambda,
        )
        val_loss, avg_terms = validation(
            val_loader, view1_data, view2_data, model, TEMPERATURE, loss_function,
            use_gpu, epoch, epochs, grl_lambda=grl_lambda,
        )
        print('epoch {0} done (warm_up={1})'.format(epoch, loss_function.warm_up))

        if not loss_function.warm_up:
            if val_loss < loss_best:
                loss_best = val_loss
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                best_terms = dict(avg_terms)
            if early_stopper.early_stop(val_loss):
                print('Early stopping at epoch {}'.format(epoch))
                break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        loss_function.warm_up = False
        _, best_terms = validation(
            val_loader, view1_data, view2_data, model, TEMPERATURE, loss_function,
            use_gpu, 0, epochs,
        )
        loss_best = float(best_terms.get('val_score', 0.0))

    eval_loss = fixed_eval_loss(best_terms)
    torch.save(best_state, model_path(path))
    _save_splits(path, X_train, X_val, X_test, y_train, y_val, y_test)
    np.save(os.path.join(path, 'loss.npy'), loss_best)
    np.save(os.path.join(path, 'eval_loss.npy'), eval_loss)

    metrics = evaluate_config(DISEASE, path, plot=False)
    payload = _enrich_metrics(metrics, eval_loss=eval_loss)
    payload.update({
        'run_id': run['id'],
        'method': run['method'],
        'base_config': run['base_config'],
        'seed': int(seed),
        'val_loss': float(loss_best),
    })
    _write_metrics(path, payload)
    return path


TRAINERS: Dict[str, Callable[..., str]] = {
    'mocss': train_mocss,
    'method1': train_method1,
    'method1_vicreg': train_method1_vicreg,
    'method2': train_method2,
    'method2_omidient': train_method2_omidient,
    'omidient': train_omidient,
    'method3': train_method3,
    'method4': train_method4,
}


def run_one(run: dict, seed: int, epochs: int, batch_size: int, resume: bool) -> str:
    trainer = TRAINERS.get(run['method'])
    if trainer is None:
        raise KeyError('No trainer for method={!r}'.format(run['method']))
    print(
        '\n===== {} | config={} | seed={} ====='.format(
            run['id'], run['base_config'], seed
        )
    )
    return trainer(run, seed, epochs, batch_size, resume)
