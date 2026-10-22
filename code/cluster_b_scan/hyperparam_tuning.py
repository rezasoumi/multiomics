"""Cluster B: SCAN fine-tune on a Cluster A checkpoint."""

import argparse
import csv
import json
import os
import warnings
from itertools import product

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from multiomics.code.loading_data import load_data
from multiomics.code.cluster_common import NUM_CLUST, index_split, truth_from_label
from multiomics.code.cluster_a_contrastive_dec.model import MultiViewClusterModel
from multiomics.code.cluster_a_contrastive_dec.evaluate import evaluate_config as eval_a
from multiomics.code.cluster_b_scan.mine_neighbors import mine_neighbors
from multiomics.code.cluster_b_scan.training import scan_loss, EarlyStopper, _split

warnings.filterwarnings('ignore')

disease = 'brca'
USE_GPU = torch.cuda.is_available()
SEED = 21
N_CLUST = NUM_CLUST[disease]
EPOCHS_FT = 50
BATCH = 32


class IndexedDataset(Dataset):
    def __init__(self, X, indices):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.indices = np.asarray(indices)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], int(self.indices[i])


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def results_parent():
    return '../../results/models_{}_cluster_b'.format(disease)


def parent_a_dir(config):
    return os.path.join('../../results/models_{}_cluster_a'.format(disease), config)


def config_name(parent, k_emb, lambda_ent, lr_ft):
    return '{}_k{}_e{}_lr{}'.format(parent, k_emb, lambda_ent, lr_ft)


@torch.no_grad()
def encode_all(model, concat, d1, d2, use_gpu, batch=256):
    model.eval()
    zs, qs = [], []
    for start in range(0, len(concat), batch):
        chunk = concat[start:start + batch]
        v1, v2, v3 = _split(chunk, d1, d2, use_gpu)
        out = model(v1, v2, v3)
        zs.append(out['z'].cpu().numpy())
        qs.append(out['q'].cpu().numpy())
    return np.concatenate(zs, 0), np.concatenate(qs, 0)


def train_scan(parent_config, k_emb, lambda_ent, lr_ft, epochs=EPOCHS_FT, freeze_encoder=False):
    setup_seed(SEED)
    view1, view2, view3, concat, y_true = load_data(disease)
    d1, d2 = view1.shape[1], view2.shape[1]
    idx_tr, idx_va, idx_te = index_split(len(concat))

    a_path = parent_a_dir(parent_config)
    model = MultiViewClusterModel(
        view_size=[view1.shape[1], view2.shape[1], view3.shape[1]],
        n_units_1=[512, 256, 128, 32],
        n_units_2=[512, 256, 128, 32],
        n_units_3=[256, 128, 64, 32],
        mlp_size=[32, 8],
        n_clusters=N_CLUST,
    )
    state = torch.load(os.path.join(a_path, 'model_{}'.format(disease)), map_location='cpu')
    model.load_state_dict(state)
    if USE_GPU:
        model = model.cuda()

    # Mine neighbors on TRAIN patients only (indices into full cohort)
    Z_all, _ = encode_all(model, concat, d1, d2, USE_GPU)
    # Build neighbor lists in full space then restrict edges to train
    nbrs_full = mine_neighbors(
        Z_all, view1, view2, view3, k_emb=k_emb, k_raw=15, d_pca=50
    )
    train_set = set(idx_tr.tolist())
    nbrs_train = {}
    for i in idx_tr:
        nbrs_train[int(i)] = [j for j in nbrs_full[int(i)] if j in train_set][:k_emb]

    name = config_name(parent_config, k_emb, lambda_ent, lr_ft)
    out_path = os.path.join(results_parent(), name)
    os.makedirs(out_path, exist_ok=True)

    if freeze_encoder:
        for p in model.enc.parameters():
            p.requires_grad = False
        params = [model.centroids]
    else:
        params = model.parameters()

    opt = torch.optim.Adam(params, lr=lr_ft, weight_decay=1e-5)
    loader = DataLoader(
        IndexedDataset(concat[idx_tr], idx_tr),
        batch_size=BATCH, shuffle=True, drop_last=True,
    )
    # For neighbor soft labels, re-encode train each epoch (or use memory bank). v1: memory bank refreshed each epoch.
    stopper = EarlyStopper()
    best_state, best_loss = None, np.inf

    for epoch in range(epochs):
        model.train()
        # refresh soft assignment bank on train
        with torch.no_grad():
            model.eval()
            Z_tr, Q_tr = encode_all(model, concat[idx_tr], d1, d2, USE_GPU)
            model.train()
            # map global idx -> row in train bank
            glob_to_row = {int(g): r for r, g in enumerate(idx_tr)}
            Q_bank = torch.tensor(Q_tr, dtype=torch.float32)
            if USE_GPU:
                Q_bank = Q_bank.cuda()

        total = 0.0
        nobs = 0
        for xb, gidx in loader:
            v1, v2, v3 = _split(xb, d1, d2, USE_GPU)
            out = model(v1, v2, v3)
            q = out['q']
            # gather neighbor q's
            neigh_qs = []
            max_k = k_emb
            B = q.shape[0]
            # build tensor (max_k, B, K) with zeros if missing
            for slot in range(max_k):
                buf = torch.zeros_like(q)
                for b in range(B):
                    g = int(gidx[b])
                    nlist = nbrs_train.get(g, [])
                    if slot < len(nlist):
                        row = glob_to_row[nlist[slot]]
                        buf[b] = Q_bank[row]
                    else:
                        buf[b] = q[b].detach()  # self fallback (no force)
                neigh_qs.append(buf)

            loss, terms = scan_loss(q, neigh_qs, lambda_ent=lambda_ent, lambda_sharp=1.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * B
            nobs += B

        avg = total / max(nobs, 1)
        print('[SCAN epoch {}/{}] loss={:.4f}'.format(epoch + 1, epochs, avg))
        if avg < best_loss:
            best_loss = avg
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if stopper.early_stop(avg):
            print('Early stop SCAN')
            break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # Save in Cluster-A-compatible layout for evaluate reuse
    torch.save(best_state, os.path.join(out_path, 'model_{}'.format(disease)))
    for fname in (
        'train_data_{}.npy', 'train_label_{}.npy',
        'val_data_{}.npy', 'val_label_{}.npy',
        'test_data_{}.npy', 'test_label_{}.npy',
    ):
        src = os.path.join(a_path, fname.format(disease))
        dst = os.path.join(out_path, fname.format(disease))
        if os.path.exists(src):
            np.save(dst, np.load(src, allow_pickle=True))
    if not os.path.exists(os.path.join(out_path, 'test_data_{}.npy'.format(disease))):
        np.save(os.path.join(out_path, 'test_data_{}'.format(disease)), concat[idx_te])
        np.save(os.path.join(out_path, 'test_label_{}'.format(disease)), np.asarray(y_true)[idx_te])
        np.save(os.path.join(out_path, 'train_data_{}'.format(disease)), concat[idx_tr])
        np.save(os.path.join(out_path, 'train_label_{}'.format(disease)), np.asarray(y_true)[idx_tr])
        np.save(os.path.join(out_path, 'val_data_{}'.format(disease)), concat[idx_va])
        np.save(os.path.join(out_path, 'val_label_{}'.format(disease)), np.asarray(y_true)[idx_va])

    np.save(os.path.join(out_path, 'loss.npy'), best_loss)
    np.save(os.path.join(out_path, 'eval_loss.npy'), best_loss)
    with open(os.path.join(out_path, 'parent.json'), 'w') as f:
        json.dump({
            'parent_config': parent_config,
            'k_emb': k_emb,
            'lambda_ent': lambda_ent,
            'lr_ft': lr_ft,
            'freeze_encoder': freeze_encoder,
        }, f, indent=2)
    return out_path, name, float(best_loss)


def resolve_parent(args):
    if args.parent_config:
        return args.parent_config
    best = os.path.join('../../results/models_{}_cluster_a'.format(disease), 'best_config.json')
    if not os.path.exists(best):
        raise FileNotFoundError('No Cluster A best_config.json; pass --parent-config')
    with open(best) as f:
        return json.load(f)['config']


def run_sweep(args):
    parent = resolve_parent(args)
    print('SCAN parent A config:', parent)
    grid = list(product([5, 10, 15], [1.0, 2.0, 5.0], [1e-4, 3e-4]))
    fields = [
        'config', 'parent_config', 'k_emb', 'lambda_ent', 'lr_ft',
        'eval_loss', 'nmi', 'ari', 'acc', 'knn_acc', 'nmi_assign', 'entropy_q',
    ]
    lb = os.path.join(results_parent(), 'leaderboard.csv')
    os.makedirs(results_parent(), exist_ok=True)
    done = set()
    if args.resume and os.path.exists(lb):
        with open(lb) as f:
            done = {r['config'] for r in csv.DictReader(f)}

    for i, (k_emb, lent, lr_ft) in enumerate(grid, 1):
        name = config_name(parent, k_emb, lent, lr_ft)
        print('\n===== [{}/{}] {} ====='.format(i, len(grid), name))
        if name in done:
            print('Skip')
            continue
        path, name, loss = train_scan(parent, k_emb, lent, lr_ft, epochs=args.epochs)
        metrics = eval_a(disease, path)
        row = {
            'config': name, 'parent_config': parent, 'k_emb': k_emb,
            'lambda_ent': lent, 'lr_ft': lr_ft, 'eval_loss': loss, **metrics,
        }
        write_header = not os.path.exists(lb) or os.path.getsize(lb) == 0
        with open(lb, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            if write_header:
                w.writeheader()
            w.writerow(row)
        with open(os.path.join(path, 'metrics.json'), 'w') as f:
            json.dump(row, f, indent=2)
        print('RESULT nmi={:.4f} nmi_assign={:.4f}'.format(metrics['nmi'], metrics['nmi_assign']))

    # summarize by nmi (analysis) but also keep best eval_loss
    with open(lb) as f:
        rows = list(csv.DictReader(f))
    best = max(rows, key=lambda r: float(r['nmi']))
    out = {'rank_metric': 'nmi', 'config': best['config'], **{k: best[k] for k in best}}
    with open(os.path.join(results_parent(), 'best_config.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print('BEST by nmi:', best['config'], best['nmi'])


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--parent-config', type=str, default='')
    p.add_argument('--epochs', type=int, default=EPOCHS_FT)
    p.add_argument('--resume', action='store_true')
    args, unknown = p.parse_known_args()
    if unknown:
        raise ValueError(unknown)
    run_sweep(args)
