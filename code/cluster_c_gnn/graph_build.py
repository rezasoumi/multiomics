"""Build consensus multi-omics patient graph (pure numpy)."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


def _knn_binary(X, k):
    X = normalize(X, norm='l2', axis=1)
    n = X.shape[0]
    nn = NearestNeighbors(n_neighbors=min(k + 1, n), metric='cosine', algorithm='brute')
    nn.fit(X)
    _, idx = nn.kneighbors(X)
    A = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in idx[i]:
            if j != i:
                A[i, j] = 1.0
    A = np.maximum(A, A.T)
    return A


def build_consensus_graph(view1, view2, view3, k=15, d_pca=50, random_state=21):
    binaries = []
    node_feats = []
    for X in (view1, view2, view3):
        n_comp = min(d_pca, X.shape[0] - 1, X.shape[1])
        Z = PCA(n_components=n_comp, random_state=random_state).fit_transform(X)
        Z = normalize(Z, norm='l2', axis=1)
        node_feats.append(Z)
        binaries.append(_knn_binary(Z, k))
    support = sum(binaries)
    A = (support >= 2).astype(np.float32)
    if A.sum() < 2 * view1.shape[0]:
        A = (sum(binaries) / 3.0).astype(np.float32)
        A = (A > 0).astype(np.float32)
    np.fill_diagonal(A, 0.0)
    # add self-loops for GCN
    A = A + np.eye(A.shape[0], dtype=np.float32)
    feats = np.concatenate(node_feats, axis=1).astype(np.float32)
    # symmetric normalized adjacency
    deg = A.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-8))
    D = np.diag(d_inv_sqrt)
    A_hat = D @ A @ D
    return A_hat.astype(np.float32), feats
