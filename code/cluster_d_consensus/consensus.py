"""Co-association consensus clustering (no neural nets)."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize


def _pca_features(X, d_pca, random_state=21):
    n_comp = min(d_pca, X.shape[0] - 1, X.shape[1])
    Z = PCA(n_components=n_comp, random_state=random_state).fit_transform(X)
    return normalize(Z, norm='l2', axis=1)


def base_kmeans_labels(feature_list, n_clusters, random_state=21):
    labels = []
    for Z in feature_list:
        km = KMeans(n_clusters=n_clusters, n_init=30, random_state=random_state)
        labels.append(km.fit_predict(Z))
    return labels


def coassociation_matrix(label_list):
    n = len(label_list[0])
    S = np.zeros((n, n), dtype=np.float64)
    for lab in label_list:
        lab = np.asarray(lab)
        for k in np.unique(lab):
            idx = np.where(lab == k)[0]
            S[np.ix_(idx, idx)] += 1.0
    S /= float(len(label_list))
    np.fill_diagonal(S, 1.0)
    return S


def final_from_coassociation(S, n_clusters, method='kmeans', random_state=21):
    if method == 'spectral':
        sc = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            assign_labels='kmeans',
            random_state=random_state,
            n_init=20,
        )
        return sc.fit_predict(S)
    # embed via top eigenvectors of S then KMeans (stable)
    evals, evecs = np.linalg.eigh(S)
    r = min(n_clusters, S.shape[0] - 1)
    U = evecs[:, -r:]
    U = normalize(U, norm='l2', axis=1)
    return KMeans(n_clusters=n_clusters, n_init=30, random_state=random_state).fit_predict(U)


def run_consensus(
    view1,
    view2,
    view3,
    n_clusters,
    d_pca=50,
    final_method='kmeans',
    random_state=21,
    include_concat=True,
):
    feats = [
        _pca_features(view1, d_pca, random_state),
        _pca_features(view2, d_pca, random_state),
        _pca_features(view3, d_pca, random_state),
    ]
    if include_concat:
        concat = np.concatenate([view1, view2, view3], axis=1)
        feats.append(_pca_features(concat, d_pca, random_state))

    base = base_kmeans_labels(feats, n_clusters, random_state)
    S = coassociation_matrix(base)
    final = final_from_coassociation(S, n_clusters, method=final_method, random_state=random_state)
    return final, S, base
