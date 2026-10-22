"""CLI: multi-omics consensus clustering baseline (Cluster D)."""

import argparse
import csv
import json
import os
import warnings
from itertools import product

import numpy as np

from multiomics.code.loading_data import load_data
from multiomics.code.cluster_common import (
    NUM_CLUST,
    truth_from_label,
    assignment_metrics,
    cluster_metrics,
    index_split,
)
from multiomics.code.cluster_d_consensus.consensus import run_consensus

warnings.filterwarnings('ignore')


def results_parent(disease):
    return '../../results/models_{}_cluster_d'.format(disease)


def config_name(d_pca, k_c, final_method):
    return 'pca{}_k{}_{}'.format(d_pca, k_c, final_method)


def run_one(disease, d_pca, k_c, final_method, report_test_only=False):
    view1, view2, view3, _, y_true = load_data(disease)
    truth = truth_from_label(disease, y_true)
    n_clust_eval = NUM_CLUST[disease]

    final, S, base = run_consensus(
        view1, view2, view3,
        n_clusters=k_c,
        d_pca=d_pca,
        final_method=final_method,
    )

    # Full-cohort assignment metrics (standard for consensus)
    metrics = assignment_metrics(final, truth)
    # Also KMeans on co-association embedding is already in final; add embedding-style
    # by treating one-hot of final as trivial — instead report assign as primary nmi
    metrics['nmi'] = metrics['nmi_assign']
    metrics['ari'] = metrics['ari_assign']
    metrics['acc'] = metrics['acc_assign']
    metrics['f_score'] = metrics['f_score_assign']
    metrics['knn_acc'] = float('nan')  # not defined the same way

    if report_test_only:
        idx_train, idx_val, idx_test = index_split(len(truth))
        m_te = assignment_metrics(final[idx_test], truth[idx_test])
        metrics['nmi_test'] = m_te['nmi_assign']
        metrics['ari_test'] = m_te['ari_assign']
        metrics['acc_test'] = m_te['acc_assign']

    # Per-view base NMI for diagnostics
    for i, lab in enumerate(base):
        metrics['base{}_nmi'.format(i + 1)] = float(
            __import__('sklearn.metrics', fromlist=['normalized_mutual_info_score'])
            .normalized_mutual_info_score(truth, lab)
        )

    name = config_name(d_pca, k_c, final_method)
    path = os.path.join(results_parent(disease), name)
    os.makedirs(path, exist_ok=True)
    np.save(os.path.join(path, 'labels_final.npy'), final)
    with open(os.path.join(path, 'metrics.json'), 'w') as f:
        json.dump({
            'config': name,
            'd_pca': d_pca,
            'k_c': k_c,
            'final_method': final_method,
            'n_clust_eval': n_clust_eval,
            **metrics,
        }, f, indent=2)
    return name, metrics


def main(args):
    disease = args.disease
    parent = results_parent(disease)
    os.makedirs(parent, exist_ok=True)

    if args.grid:
        d_pca_grid = [30, 50, 100]
        k_grid = [4, 5, 6]
        methods = ['kmeans', 'spectral']
    else:
        d_pca_grid = [args.d_pca]
        k_grid = [args.k_clusters]
        methods = [args.final_method]

    fieldnames = [
        'config', 'd_pca', 'k_c', 'final_method',
        'nmi', 'ari', 'acc', 'f_score', 'nmi_test', 'ari_test', 'acc_test',
        'base1_nmi', 'base2_nmi', 'base3_nmi', 'base4_nmi',
    ]
    lb_path = os.path.join(parent, 'leaderboard.csv')
    # rewrite leaderboard each full run
    rows = []
    for d_pca, k_c, method in product(d_pca_grid, k_grid, methods):
        # evaluate K should match disease subtypes for fair headline when k_c == NUM_CLUST
        name, metrics = run_one(
            disease, d_pca, k_c, method, report_test_only=True
        )
        print(
            'D {} | nmi={:.4f} ari={:.4f} acc={:.4f} | test_nmi={}'.format(
                name, metrics['nmi'], metrics['ari'], metrics['acc'],
                '{:.4f}'.format(metrics['nmi_test']) if 'nmi_test' in metrics else 'n/a',
            )
        )
        row = {
            'config': name,
            'd_pca': d_pca,
            'k_c': k_c,
            'final_method': method,
            'nmi': metrics['nmi'],
            'ari': metrics['ari'],
            'acc': metrics['acc'],
            'f_score': metrics['f_score'],
            'nmi_test': metrics.get('nmi_test', ''),
            'ari_test': metrics.get('ari_test', ''),
            'acc_test': metrics.get('acc_test', ''),
            'base1_nmi': metrics.get('base1_nmi', ''),
            'base2_nmi': metrics.get('base2_nmi', ''),
            'base3_nmi': metrics.get('base3_nmi', ''),
            'base4_nmi': metrics.get('base4_nmi', ''),
        }
        rows.append(row)

    with open(lb_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    # Headline: k_c == NUM_CLUST, d_pca=50, kmeans if present
    target_k = NUM_CLUST[disease]
    candidates = [r for r in rows if int(r['k_c']) == target_k]
    if not candidates:
        candidates = rows
    best = max(candidates, key=lambda r: float(r['nmi']))
    # Prefer default pca50_kmeans among same k
    defaults = [
        r for r in candidates
        if int(r['d_pca']) == 50 and r['final_method'] == 'kmeans'
    ]
    headline = defaults[0] if defaults else best
    out = {
        'headline_config': headline['config'],
        'headline_nmi': float(headline['nmi']),
        'best_nmi_config': best['config'],
        'best_nmi': float(best['nmi']),
        'note': 'Full-cohort NMI; nmi_test is consensus labels restricted to Method1/2 test indices.',
    }
    with open(os.path.join(parent, 'best_config.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print('\nHEADLINE:', json.dumps(out, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Cluster D consensus baseline')
    p.add_argument('--disease', type=str, default='brca')
    p.add_argument('--grid', action='store_true', help='Sweep d_pca × K × method')
    p.add_argument('--d-pca', type=int, default=50)
    p.add_argument('--k-clusters', type=int, default=5)
    p.add_argument('--final-method', type=str, default='kmeans', choices=['kmeans', 'spectral'])
    args, unknown = p.parse_known_args()
    if unknown:
        raise ValueError('Unknown args: {}'.format(unknown))
    main(args)
