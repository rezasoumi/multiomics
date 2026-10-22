"""Shared helpers for the single-Z clustering track (Methods A/B/C/D)."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, normalized_mutual_info_score, adjusted_rand_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from multiomics.code import evaluation

NUM_CLUST = {'lihc': 2, 'coad': 4, 'kirc': 2, 'brca': 5}


def truth_from_label(disease, label):
    label = np.asarray(label)
    if disease == 'coad':
        return label.flatten().astype('int')
    if disease == 'lihc':
        lst = label[:, 0].flatten()
        unique_vals = list(set(lst))
        mapping = {val: idx for idx, val in enumerate(unique_vals)}
        return np.asarray([mapping[val] for val in lst])
    if disease == 'kirc':
        return label[:, 1].flatten().astype('int')
    return label.flatten()


def cluster_metrics(X, y, n_clusters, random_state=21):
    """NMI/ARI/Acc/f/v/ch + kNN on embedding X."""
    y = np.asarray(y).flatten()
    pred = KMeans(n_clusters=n_clusters, n_init=30, random_state=random_state).fit_predict(X)
    nmi, ari, f_score, acc, v_measure, ch_index = evaluation.evaluate(y, pred)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=n_clusters)
    knn.fit(X_tr, y_tr)
    knn_acc = float(accuracy_score(y_te, knn.predict(X_te)))
    return {
        'nmi': float(nmi),
        'ari': float(ari),
        'f_score': float(f_score),
        'acc': float(acc),
        'v_measure': float(v_measure),
        'ch_index': float(ch_index),
        'knn_acc': knn_acc,
    }


def assignment_metrics(pred, y):
    """Metrics when hard labels are already given (DEC argmax / consensus)."""
    y = np.asarray(y).flatten()
    pred = np.asarray(pred).flatten()
    nmi, ari, f_score, acc, v_measure, ch_index = evaluation.evaluate(y, pred)
    return {
        'nmi_assign': float(nmi),
        'ari_assign': float(ari),
        'f_score_assign': float(f_score),
        'acc_assign': float(acc),
        'v_measure_assign': float(v_measure),
        'ch_index_assign': float(ch_index),
    }


def index_split(n, test_size=0.2, val_size=0.25, random_state=1):
    """Same 60/20/20 index split as Methods 1–2 (train_test_split random_state=1)."""
    idx = np.arange(n)
    idx_train, idx_test = train_test_split(idx, test_size=test_size, random_state=random_state)
    idx_train, idx_val = train_test_split(idx_train, test_size=val_size, random_state=random_state)
    return idx_train, idx_val, idx_test
