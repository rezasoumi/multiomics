"""Precompute fused multi-omics kNN graph and Laplacian spectral targets (v2).

v2 improvements vs v1:
  - larger PCA / kNN defaults
  - consensus fusion (mean affinity × multi-view edge support)
  - optional small label-free graph search (eigengap); logs nmi_U for reporting
"""

from __future__ import annotations

import json
import os

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


def _mutual_knn_affinity(Z, k):
    """Symmetrized kNN cosine affinity."""
    n = Z.shape[0]
    nn = NearestNeighbors(n_neighbors=min(k + 1, n), metric='cosine', algorithm='brute')
    nn.fit(Z)
    dist, idx = nn.kneighbors(Z)
    A = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j_pos, j in enumerate(idx[i]):
            if j == i:
                continue
            sim = 1.0 - float(dist[i, j_pos])
            A[i, j] = max(A[i, j], sim)
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0.0)
    return A


def build_fused_affinity(
    view_mats,
    k_nn=25,
    d_pca=100,
    random_state=21,
    fusion='consensus',
):
    """
    fusion:
      'mean'      — average of per-view affinities
      'consensus' — mean affinity, zeroed unless edge present in ≥2 views
    """
    affinities = []
    binary = []
    for X in view_mats:
        n_comp = min(d_pca, X.shape[0] - 1, X.shape[1])
        Z = PCA(n_components=n_comp, random_state=random_state).fit_transform(X)
        Z = normalize(Z, norm='l2', axis=1)
        A = _mutual_knn_affinity(Z, k_nn)
        affinities.append(A)
        binary.append((A > 0).astype(np.float64))

    A_mean = sum(affinities) / float(len(affinities))
    if fusion == 'mean':
        return A_mean
    support = sum(binary)
    A = A_mean * (support >= 2).astype(np.float64)
    # if too sparse, fall back to mean
    if A.sum() < A_mean.sum() * 0.05:
        return A_mean
    return A


def laplacian_spectral_embedding(A, r=24, eps=1e-8):
    """Normalized Laplacian eigenvectors (skip trivial first). Returns U, evals, eigengap."""
    d = A.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(d, eps))
    D_inv_sqrt = np.diag(d_inv_sqrt)
    L = np.eye(A.shape[0]) - D_inv_sqrt @ A @ D_inv_sqrt
    evals_all, evecs = np.linalg.eigh(L)
    start = 1
    U = evecs[:, start:start + r]
    evals = evals_all[start:start + r]
    U = normalize(U, norm='l2', axis=1)
    # eigengap after the r-th kept direction (label-free quality proxy)
    if start + r < len(evals_all):
        eigengap = float(evals_all[start + r] - evals_all[start + r - 1])
    else:
        eigengap = float(evals[-1] - evals[0]) if len(evals) > 1 else 0.0
    return U, evals, eigengap


def kmeans_labels_from_U(U, n_clusters, random_state=21):
    km = KMeans(n_clusters=n_clusters, n_init=30, random_state=random_state)
    return km.fit_predict(U)


def spectral_targets_dir(repo_results_root='../../results'):
    return os.path.join(repo_results_root, 'spectral_targets')


def targets_paths(disease, repo_results_root='../../results', tag='v2'):
    parent = spectral_targets_dir(repo_results_root)
    suffix = '' if tag in (None, '', 'v1') else '_{}'.format(tag)
    return {
        'parent': parent,
        'U': os.path.join(parent, '{}{}_U.npy'.format(disease, suffix)),
        'evals': os.path.join(parent, '{}{}_evals.npy'.format(disease, suffix)),
        'labels': os.path.join(parent, '{}{}_kmeans_labels.npy'.format(disease, suffix)),
        'meta': os.path.join(parent, '{}{}_meta.json'.format(disease, suffix)),
        'nbrs': os.path.join(parent, '{}{}_nbrs.npy'.format(disease, suffix)),
    }


def _topk_neighbors(A, k=15):
    """For each row, indices of top-k affinity neighbors (no self)."""
    n = A.shape[0]
    nbrs = np.zeros((n, k), dtype=np.int64)
    for i in range(n):
        row = A[i].copy()
        row[i] = -1.0
        nbrs[i] = np.argpartition(-row, kth=min(k, n - 1) - 1)[:k]
    return nbrs


def nmi_on_spectral_U(U, truth, n_clusters):
    from sklearn.metrics import normalized_mutual_info_score

    pred = KMeans(n_clusters=n_clusters, n_init=30, random_state=21).fit_predict(U)
    return float(normalized_mutual_info_score(np.asarray(truth).flatten(), pred))


def _candidate_grid():
    """Small label-free search over graph knobs."""
    return [
        {'k_nn': 15, 'd_pca': 50, 'r': 16, 'fusion': 'mean'},
        {'k_nn': 20, 'd_pca': 100, 'r': 24, 'fusion': 'mean'},
        {'k_nn': 25, 'd_pca': 100, 'r': 24, 'fusion': 'consensus'},
        {'k_nn': 30, 'd_pca': 100, 'r': 24, 'fusion': 'consensus'},
        {'k_nn': 25, 'd_pca': 100, 'r': 32, 'fusion': 'consensus'},
    ]


def compute_and_save_targets(
    disease,
    view1,
    view2,
    view3,
    n_clusters,
    k_nn=25,
    d_pca=100,
    r=24,
    random_state=21,
    repo_results_root='../../results',
    force=False,
    fusion='consensus',
    auto_select=True,
    truth_for_log=None,
    tag='v2',
):
    """Build / load spectral targets. If auto_select, pick max eigengap among candidates."""
    paths = targets_paths(disease, repo_results_root, tag=tag)
    os.makedirs(paths['parent'], exist_ok=True)

    if (
        not force
        and os.path.exists(paths['U'])
        and os.path.exists(paths['labels'])
        and os.path.exists(paths['meta'])
    ):
        U = np.load(paths['U'])
        labels = np.load(paths['labels'])
        with open(paths['meta'], 'r') as f:
            meta = json.load(f)
        print('Loaded spectral targets from {} (tag={})'.format(paths['parent'], tag))
        return U, labels, meta

    view_mats = [view1, view2, view3]
    candidates = _candidate_grid() if auto_select else [
        {'k_nn': k_nn, 'd_pca': d_pca, 'r': r, 'fusion': fusion}
    ]

    best = None
    trial_logs = []
    for cfg in candidates:
        A = build_fused_affinity(
            view_mats,
            k_nn=cfg['k_nn'],
            d_pca=cfg['d_pca'],
            random_state=random_state,
            fusion=cfg['fusion'],
        )
        U, evals, eigengap = laplacian_spectral_embedding(A, r=cfg['r'])
        labels = kmeans_labels_from_U(U, n_clusters=n_clusters, random_state=random_state)
        nmi_u = None
        if truth_for_log is not None:
            try:
                nmi_u = nmi_on_spectral_U(U, truth_for_log, n_clusters)
            except Exception:
                nmi_u = None
        entry = {
            **cfg,
            'eigengap': eigengap,
            'nmi_spectral_U': nmi_u,
            'n_edges': int((A > 0).sum() // 2),
        }
        trial_logs.append(entry)
        print(
            '  graph trial k={} pca={} r={} fusion={} | eigengap={:.4e} nmi_U={} edges={}'.format(
                cfg['k_nn'], cfg['d_pca'], cfg['r'], cfg['fusion'],
                eigengap, '{:.4f}'.format(nmi_u) if nmi_u is not None else 'n/a',
                entry['n_edges'],
            )
        )
        score = eigengap
        # Prefer higher nmi_U among near-tied eigengaps when labels available for logging only
        # Selection remains eigengap-primary (label-free).
        if best is None or score > best['eigengap']:
            best = {
                'cfg': cfg,
                'A': A,
                'U': U,
                'evals': evals,
                'labels': labels,
                'eigengap': eigengap,
                'nmi_u': nmi_u,
            }

    cfg = best['cfg']
    U, evals, labels = best['U'], best['evals'], best['labels']
    A = best['A']
    nbrs = _topk_neighbors(A, k=min(15, U.shape[0] - 1))

    np.save(paths['U'], U)
    np.save(paths['evals'], evals)
    np.save(paths['labels'], labels)
    np.save(paths['nbrs'], nbrs)
    meta = {
        'disease': disease,
        'tag': tag,
        'n_samples': int(U.shape[0]),
        'r': int(cfg['r']),
        'k_nn': int(cfg['k_nn']),
        'd_pca': int(cfg['d_pca']),
        'fusion': cfg['fusion'],
        'n_clusters': int(n_clusters),
        'random_state': int(random_state),
        'eigengap': float(best['eigengap']),
        'nmi_spectral_U': best['nmi_u'],
        'trials': trial_logs,
        'note': 'v2 consensus/mean graph; selected by eigengap (label-free). nmi_U logged only.',
    }
    with open(paths['meta'], 'w') as f:
        json.dump(meta, f, indent=2)
    print('Saved spectral targets to {} | chosen {}'.format(paths['parent'], cfg))
    return U, labels, meta
