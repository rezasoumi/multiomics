"""Precompute fused multi-omics kNN graph and Laplacian spectral targets.

No subtype labels are used. Targets are saved under results/spectral_targets/.
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
    """Symmetrized mutual-ish kNN cosine affinity (edge if either is in other's kNN)."""
    n = Z.shape[0]
    nn = NearestNeighbors(n_neighbors=min(k + 1, n), metric='cosine', algorithm='brute')
    nn.fit(Z)
    dist, idx = nn.kneighbors(Z)
    # drop self (first neighbor under cosine with identical rows)
    A = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j_pos, j in enumerate(idx[i]):
            if j == i:
                continue
            # convert cosine distance -> similarity in [0, 1]
            sim = 1.0 - float(dist[i, j_pos])
            A[i, j] = max(A[i, j], sim)
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0.0)
    return A


def build_fused_affinity(view_mats, k_nn=15, d_pca=50, random_state=21):
    """
    view_mats: list/tuple of (N, D_m) arrays (already scaled like load_data).
    Returns A_fused (N, N).
    """
    affinities = []
    for X in view_mats:
        n_comp = min(d_pca, X.shape[0] - 1, X.shape[1])
        Z = PCA(n_components=n_comp, random_state=random_state).fit_transform(X)
        Z = normalize(Z, norm='l2', axis=1)
        affinities.append(_mutual_knn_affinity(Z, k_nn))
    A = sum(affinities) / float(len(affinities))
    return A


def laplacian_spectral_embedding(A, r=16, eps=1e-8):
    """
    Normalized Laplacian eigenvectors (skip trivial first).
    Returns U (N, r) row-L2-normalized.
    """
    d = A.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(d, eps))
    D_inv_sqrt = np.diag(d_inv_sqrt)
    L = np.eye(A.shape[0]) - D_inv_sqrt @ A @ D_inv_sqrt
    # Symmetric eigh: eigenvalues ascending
    evals, evecs = np.linalg.eigh(L)
    # skip the smallest (near-zero / constant)
    start = 1
    U = evecs[:, start:start + r]
    U = normalize(U, norm='l2', axis=1)
    return U, evals[start:start + r]


def kmeans_labels_from_U(U, n_clusters, random_state=21):
    km = KMeans(n_clusters=n_clusters, n_init=30, random_state=random_state)
    return km.fit_predict(U)


def spectral_targets_dir(repo_results_root='../../results'):
    return os.path.join(repo_results_root, 'spectral_targets')


def targets_paths(disease, repo_results_root='../../results'):
    parent = spectral_targets_dir(repo_results_root)
    return {
        'parent': parent,
        'U': os.path.join(parent, '{}_U.npy'.format(disease)),
        'evals': os.path.join(parent, '{}_evals.npy'.format(disease)),
        'labels': os.path.join(parent, '{}_kmeans_labels.npy'.format(disease)),
        'meta': os.path.join(parent, '{}_meta.json'.format(disease)),
    }


def compute_and_save_targets(
    disease,
    view1,
    view2,
    view3,
    n_clusters,
    k_nn=15,
    d_pca=50,
    r=16,
    random_state=21,
    repo_results_root='../../results',
    force=False,
):
    """Build / load spectral targets aligned with load_data sample order."""
    paths = targets_paths(disease, repo_results_root)
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
        print('Loaded spectral targets from {}'.format(paths['parent']))
        return U, labels, meta

    print('Computing fused spectral targets for {} (k={}, pca={}, r={}) ...'.format(
        disease, k_nn, d_pca, r
    ))
    A = build_fused_affinity(
        [view1, view2, view3], k_nn=k_nn, d_pca=d_pca, random_state=random_state
    )
    U, evals = laplacian_spectral_embedding(A, r=r)
    labels = kmeans_labels_from_U(U, n_clusters=n_clusters, random_state=random_state)

    np.save(paths['U'], U)
    np.save(paths['evals'], evals)
    np.save(paths['labels'], labels)
    meta = {
        'disease': disease,
        'n_samples': int(U.shape[0]),
        'r': int(r),
        'k_nn': int(k_nn),
        'd_pca': int(d_pca),
        'n_clusters': int(n_clusters),
        'random_state': int(random_state),
        'note': 'Graph built on all unlabeled features; no subtype labels used.',
    }
    with open(paths['meta'], 'w') as f:
        json.dump(meta, f, indent=2)
    print('Saved spectral targets to {}'.format(paths['parent']))
    return U, labels, meta


def nmi_on_spectral_U(U, truth, n_clusters):
    """Ceiling proxy: KMeans on U vs subtype labels."""
    from sklearn.metrics import normalized_mutual_info_score
    from sklearn.cluster import KMeans as _KM

    pred = _KM(n_clusters=n_clusters, n_init=30, random_state=21).fit_predict(U)
    return float(normalized_mutual_info_score(truth.flatten(), pred))
