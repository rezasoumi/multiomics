"""Shared-representation evaluation helpers for Method 1 / ablation / Method 2.

Primary integration metrics use concat(P1,P2,P3, mean(S)) with KMeans n_init=30.
Additional diagnostics for shared quality:
  1) clustering + kNN on mean(S) (and optional per-view S_m)
  2) cross-view probes (subtype from S_other; X_A from S_other vs P_other)
  3) alignment of shared codes (cosine / CCA / linear CKA)
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.cross_decomposition import CCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

import multiomics.code.evaluation as evaluation

KMEANS_N_INIT = 30
KMEANS_RANDOM_STATE = 42
PROBE_RANDOM_STATE = 12
PROBE_TEST_SIZE = 0.25


def kmeans_predict(X, n_clusters, n_init=KMEANS_N_INIT, random_state=KMEANS_RANDOM_STATE):
    """KMeans with multiple inits (aligned with OMIDIENT/MOCSS LIHC protocol)."""
    km = KMeans(
        n_clusters=n_clusters,
        init='k-means++',
        n_init=n_init,
        random_state=random_state,
    )
    return km.fit_predict(X)


def clustering_and_knn(X, y, n_clusters):
    """NMI/ARI/… + kNN accuracy on embedding X (CH via shared evaluation.evaluate)."""
    y_pred = kmeans_predict(X, n_clusters)
    nmi, ari, f_score, acc, v_measure, ch_index = evaluation.evaluate(y, y_pred)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=PROBE_TEST_SIZE, random_state=PROBE_RANDOM_STATE
    )
    knn = KNeighborsClassifier(n_neighbors=n_clusters)
    knn.fit(X_tr, y_tr)
    knn_acc = float(accuracy_score(y_te, knn.predict(X_te)))

    return {
        'nmi': float(nmi),
        'ari': float(ari),
        'f_score': float(f_score),
        'acc': float(acc),
        'v_measure': float(v_measure),
        'ch_index': ch_index,
        'knn_acc': knn_acc,
    }


def _center_gram(K):
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def linear_cka(X, Y):
    """Linear Centered Kernel Alignment between two (n, d) matrices."""
    X = X.astype(np.float64, copy=False)
    Y = Y.astype(np.float64, copy=False)
    K = _center_gram(X @ X.T)
    L = _center_gram(Y @ Y.T)
    hsic_kl = float(np.sum(K * L))
    hsic_kk = float(np.sum(K * K))
    hsic_ll = float(np.sum(L * L))
    denom = np.sqrt(hsic_kk * hsic_ll)
    if denom < 1e-12:
        return 0.0
    return hsic_kl / denom


def mean_pairwise_cosine(X, Y):
    """Mean cosine similarity of paired rows after L2 normalization."""
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    Yn = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12)
    return float(np.mean(np.sum(Xn * Yn, axis=1)))


def first_cca_corr(X, Y, n_components=1):
    """Absolute correlation of the first canonical pair (or mean of a few)."""
    n = X.shape[0]
    max_comp = max(1, min(n_components, X.shape[1], Y.shape[1], max(1, n // 5)))
    try:
        Xs = StandardScaler().fit_transform(X)
        Ys = StandardScaler().fit_transform(Y)
        cca = CCA(n_components=max_comp, max_iter=500)
        Xc, Yc = cca.fit_transform(Xs, Ys)
        corrs = []
        for i in range(Xc.shape[1]):
            c = np.corrcoef(Xc[:, i], Yc[:, i])[0, 1]
            if np.isfinite(c):
                corrs.append(abs(float(c)))
        return float(np.mean(corrs)) if corrs else 0.0
    except Exception:
        return 0.0


def alignment_metrics(shared_by_view):
    """
    shared_by_view: dict {1: S1, 2: S2, 3: S3} numpy arrays (n, d).
    Returns mean pairwise cosine / CCA / CKA over the three view pairs.
    """
    pairs = [(1, 2), (1, 3), (2, 3)]
    cosines, ccas, ckas = [], [], []
    out = {}
    for a, b in pairs:
        cos = mean_pairwise_cosine(shared_by_view[a], shared_by_view[b])
        cca = first_cca_corr(shared_by_view[a], shared_by_view[b], n_components=1)
        cka = linear_cka(shared_by_view[a], shared_by_view[b])
        cosines.append(cos)
        ccas.append(cca)
        ckas.append(cka)
        out['align_cosine_{}_{}'.format(a, b)] = cos
        out['align_cca_{}_{}'.format(a, b)] = cca
        out['align_cka_{}_{}'.format(a, b)] = cka
    out['align_cosine_mean'] = float(np.mean(cosines))
    out['align_cca_mean'] = float(np.mean(ccas))
    out['align_cka_mean'] = float(np.mean(ckas))
    return out


def _stratified_or_plain_split(X, y, test_size=PROBE_TEST_SIZE, random_state=PROBE_RANDOM_STATE):
    try:
        return train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    except ValueError:
        return train_test_split(X, y, test_size=test_size, random_state=random_state)


def probe_subtype_from_S_other(shared_by_view, y):
    """
    For each target view A: predict subtype from concat(S_B, S_C) with a linear probe.
    Returns mean accuracy and per-target accuracies.
    """
    others = {1: (2, 3), 2: (1, 3), 3: (1, 2)}
    accs = []
    out = {}
    for a, (b, c) in others.items():
        X = np.concatenate([shared_by_view[b], shared_by_view[c]], axis=1)
        X_tr, X_te, y_tr, y_te = _stratified_or_plain_split(X, y)
        clf = LogisticRegression(max_iter=2000)
        clf.fit(X_tr, y_tr)
        acc = float(accuracy_score(y_te, clf.predict(X_te)))
        accs.append(acc)
        out['probe_subtype_from_S_other_acc_{}'.format(a)] = acc
    out['probe_subtype_from_S_other_acc'] = float(np.mean(accs))
    return out


def _ridge_r2(X, Y):
    """Train/test Ridge R² for multi-output regression Y ~ X."""
    n = X.shape[0]
    if n < 8:
        return 0.0
    idx = np.arange(n)
    tr, te = train_test_split(idx, test_size=PROBE_TEST_SIZE, random_state=PROBE_RANDOM_STATE)
    scaler_x = StandardScaler()
    X_tr = scaler_x.fit_transform(X[tr])
    X_te = scaler_x.transform(X[te])
    # Standardize targets per feature for stabler multioutput R²
    scaler_y = StandardScaler()
    Y_tr = scaler_y.fit_transform(Y[tr])
    Y_te = scaler_y.transform(Y[te])
    model = Ridge(alpha=1.0)
    model.fit(X_tr, Y_tr)
    pred = model.predict(X_te)
    try:
        return float(r2_score(Y_te, pred, multioutput='uniform_average'))
    except Exception:
        return 0.0


def probe_XA_from_other(views_by_id, shared_by_view, specific_by_view):
    """
    For each target A: Ridge R² predicting X_A from (S_B,S_C) vs (P_B,P_C).
    Higher S-probe and lower P-probe support cleaner shared codes.
    """
    others = {1: (2, 3), 2: (1, 3), 3: (1, 2)}
    s_r2s, p_r2s = [], []
    out = {}
    for a, (b, c) in others.items():
        X_s = np.concatenate([shared_by_view[b], shared_by_view[c]], axis=1)
        X_p = np.concatenate([specific_by_view[b], specific_by_view[c]], axis=1)
        Y = views_by_id[a]
        r2_s = _ridge_r2(X_s, Y)
        r2_p = _ridge_r2(X_p, Y)
        s_r2s.append(r2_s)
        p_r2s.append(r2_p)
        out['probe_XA_from_S_other_r2_{}'.format(a)] = r2_s
        out['probe_XA_from_P_other_r2_{}'.format(a)] = r2_p
        out['probe_XA_S_minus_P_r2_{}'.format(a)] = float(r2_s - r2_p)
    out['probe_XA_from_S_other_r2'] = float(np.mean(s_r2s))
    out['probe_XA_from_P_other_r2'] = float(np.mean(p_r2s))
    out['probe_XA_S_minus_P_r2'] = float(np.mean(s_r2s) - np.mean(p_r2s))
    return out


def evaluate_shared_and_integration(
    specific_by_view,
    shared_by_view,
    views_by_id,
    y,
    n_clusters,
):
    """
    Full metric suite.

    specific_by_view / shared_by_view / views_by_id: dicts keyed 1..3 with numpy arrays.
    y: subtype labels (1d).

    Returns a flat dict. Primary ranking key remains 'nmi' (concat integration).
    Shared-only metrics are prefixed with 's_'.
    """
    s_mean = (shared_by_view[1] + shared_by_view[2] + shared_by_view[3]) / 3.0
    final_embedding = np.concatenate(
        [
            specific_by_view[1],
            specific_by_view[2],
            specific_by_view[3],
            s_mean,
        ],
        axis=1,
    )

    integration = clustering_and_knn(final_embedding, y, n_clusters)
    shared_only = clustering_and_knn(s_mean, y, n_clusters)
    metrics = dict(integration)
    metrics.update({
        's_nmi': shared_only['nmi'],
        's_ari': shared_only['ari'],
        's_f_score': shared_only['f_score'],
        's_acc': shared_only['acc'],
        's_v_measure': shared_only['v_measure'],
        's_ch_index': shared_only['ch_index'],
        's_knn_acc': shared_only['knn_acc'],
    })

    # Per-view shared clustering (diagnostic)
    for i in (1, 2, 3):
        m_i = clustering_and_knn(shared_by_view[i], y, n_clusters)
        metrics['s{}_nmi'.format(i)] = m_i['nmi']
        metrics['s{}_knn_acc'.format(i)] = m_i['knn_acc']

    metrics.update(probe_subtype_from_S_other(shared_by_view, y))
    metrics.update(probe_XA_from_other(views_by_id, shared_by_view, specific_by_view))
    metrics.update(alignment_metrics(shared_by_view))
    return metrics, final_embedding, s_mean


# Flat keys written to leaderboard.csv / metrics.json
LEADERBOARD_METRIC_KEYS = [
    'nmi',
    'ari',
    'f_score',
    'acc',
    'v_measure',
    'ch_index',
    'knn_acc',
    's_nmi',
    's_ari',
    's_f_score',
    's_acc',
    's_v_measure',
    's_ch_index',
    's_knn_acc',
    's1_nmi',
    's1_knn_acc',
    's2_nmi',
    's2_knn_acc',
    's3_nmi',
    's3_knn_acc',
    'probe_subtype_from_S_other_acc',
    'probe_subtype_from_S_other_acc_1',
    'probe_subtype_from_S_other_acc_2',
    'probe_subtype_from_S_other_acc_3',
    'probe_XA_from_S_other_r2',
    'probe_XA_from_P_other_r2',
    'probe_XA_S_minus_P_r2',
    'probe_XA_from_S_other_r2_1',
    'probe_XA_from_S_other_r2_2',
    'probe_XA_from_S_other_r2_3',
    'probe_XA_from_P_other_r2_1',
    'probe_XA_from_P_other_r2_2',
    'probe_XA_from_P_other_r2_3',
    'align_cosine_mean',
    'align_cca_mean',
    'align_cka_mean',
    'align_cosine_1_2',
    'align_cosine_1_3',
    'align_cosine_2_3',
    'align_cca_1_2',
    'align_cca_1_3',
    'align_cca_2_3',
    'align_cka_1_2',
    'align_cka_1_3',
    'align_cka_2_3',
]


def print_metrics_report(metrics):
    print(
        '\n' + ' ' * 8
        + '|==>  [concat P+S] nmi: %.4f,  ari: %.4f,  f_score: %.4f,  acc: %.4f,  '
          'v_measure: %.4f,  ch_index: %.4f  <==|'
        % (
            metrics['nmi'],
            metrics['ari'],
            metrics['f_score'],
            metrics['acc'],
            metrics['v_measure'],
            metrics['ch_index'],
        )
    )
    print('kNN acc (concat): {:.2f}'.format(metrics['knn_acc']))
    print(
        ' ' * 8
        + '|==>  [shared mean S] nmi: %.4f,  ari: %.4f,  acc: %.4f,  knn: %.2f  <==|'
        % (metrics['s_nmi'], metrics['s_ari'], metrics['s_acc'], metrics['s_knn_acc'])
    )
    print(
        'Probes: subtype←S_other acc={:.4f} | X←S_other R²={:.4f} | X←P_other R²={:.4f} | '
        'Δ(S−P)={:.4f}'.format(
            metrics['probe_subtype_from_S_other_acc'],
            metrics['probe_XA_from_S_other_r2'],
            metrics['probe_XA_from_P_other_r2'],
            metrics['probe_XA_S_minus_P_r2'],
        )
    )
    print(
        'Alignment (mean over pairs): cosine={:.4f}  cca={:.4f}  cka={:.4f}'.format(
            metrics['align_cosine_mean'],
            metrics['align_cca_mean'],
            metrics['align_cka_mean'],
        )
    )
