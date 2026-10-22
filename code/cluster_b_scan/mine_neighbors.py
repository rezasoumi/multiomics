"""Mine must-link neighbors in embedding space ∩ multi-omics raw kNN."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


def _knn_indices(X, k):
    X = normalize(X, norm='l2', axis=1)
    nn = NearestNeighbors(n_neighbors=min(k + 1, X.shape[0]), metric='cosine', algorithm='brute')
    nn.fit(X)
    _, idx = nn.kneighbors(X)
    # drop self
    out = []
    for i in range(X.shape[0]):
        row = [int(j) for j in idx[i] if j != i][:k]
        out.append(row)
    return out


def mine_neighbors(Z, view1, view2, view3, k_emb=10, k_raw=15, d_pca=50, random_state=21):
    """
    N(i) = kNN_Z(i) ∩ (union of patients that co-appear as raw kNN in ≥2 views).
    Falls back to kNN_Z if intersection empty.
    Returns list of neighbor index lists (variable length).
    """
    n = Z.shape[0]
    emb_nn = _knn_indices(Z, k_emb)

    raw_sets = []
    for X in (view1, view2, view3):
        n_comp = min(d_pca, X.shape[0] - 1, X.shape[1])
        Xp = PCA(n_components=n_comp, random_state=random_state).fit_transform(X)
        raw_sets.append(_knn_indices(Xp, k_raw))

    # support[i][j] = number of views where j is among kNN of i
    support = [np.zeros(n, dtype=np.int32) for _ in range(n)]
    for view_nn in raw_sets:
        for i, nbrs in enumerate(view_nn):
            for j in nbrs:
                support[i][j] += 1

    neighbors = []
    for i in range(n):
        emb = set(emb_nn[i])
        raw_ok = {j for j in range(n) if j != i and support[i][j] >= 2}
        inter = sorted(emb & raw_ok)
        if not inter:
            inter = emb_nn[i]
        neighbors.append(inter)
    return neighbors
