"""Re-evaluate KIRC loss-best configs only (Ridge probes after PCA-on-Y fix).

Does NOT retrain. Loads existing model_* checkpoints and prints Δ X R² /
own-view R² with the fixed shared_rep_metrics._ridge_r2.

Usage (from a code/* cwd so ../../data resolves, or any cwd with PYTHONPATH):

  cd code/method1_predictability   # any package under code/ is fine via -m
  python -m multiomics.code.reeval_kirc_ridge

  # or only some runs:
  python -m multiomics.code.reeval_kirc_ridge --runs method1,method2,omidient_reported
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Top configs from docs/README_kirc.md (seed-21 loss winners).
KIRC_TOPS = [
    {
        'id': 'method1',
        'label': 'Method 1 + b_m',
        'pkg': 'method1_predictability',
        'results_subdir': 'models_kirc_method1',
        'config': '0.0002_0.0007_a1.5_b0.7_g1.0_p1.0_l1.0',
    },
    {
        'id': 'method1_vicreg',
        'label': 'Method 1 VICReg + b_m',
        'pkg': 'method1_vicreg_ablation',
        'results_subdir': 'models_kirc_method1_vicreg',
        'config': '0.0002_0.0007_a1.0_b0.5_v1.0_p1.0_l1.0',
    },
    {
        'id': 'method2',
        'label': 'Method 2 (AE)',
        'pkg': 'method2_joint_vicreg',
        'results_subdir': 'models_kirc_method2',
        'config': '0.0002_0.0007_r1.0_c0.7_v0.5',
    },
    {
        'id': 'method3',
        'label': 'Method 3 spectral',
        'pkg': 'method3_spectral_graph',
        'results_subdir': 'models_kirc_method3',
        'config': '0.0002_0.0007_s2.0_r1.0_c0.0_v0.1_vw0.5_lk0.5',
    },
    {
        'id': 'method2_omidient',
        'label': 'Method 2 + OMIDIENT P',
        'pkg': 'method2_omidient_private',
        'results_subdir': 'models_kirc_method2_omidient',
        'config': '0.0002_0.0007_r1.5_c0.5_v1.0_k4',
    },
    {
        'id': 'omidient_reported',
        'label': 'OMIDIENT (no ortho)',
        'pkg': 'prod_gamma_dirvae',
        'results_subdir': 'models_kirc_ProdGammaDirVae',
        'config': '0.0003_0.0005_4_c2_r1.5_k1',
        'omidient': True,
    },
    {
        'id': 'omidient_ortho',
        'label': 'OMIDIENT (+ortho)',
        'pkg': 'prod_gamma_dirvae',
        'results_subdir': 'models_kirc_ProdGammaDirVae',
        'config': '0.0003_0.0005_4_ortho_c2_r1.5_k0.5_o0.5',
        'omidient': True,
    },
]

RIDGE_KEYS = [
    'probe_XA_from_S_other_r2',
    'probe_XA_from_P_other_r2',
    'probe_XA_S_minus_P_r2',
    'probe_Xm_from_S_own_r2',
    'probe_Xm_from_P_own_r2',
    'probe_Xm_from_PS_own_r2',
]


def _repo_results_root():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, '..', 'results'))


def _evaluate(run):
    disease = 'kirc'
    config_dir = os.path.join(_repo_results_root(), run['results_subdir'], run['config'])
    model_path = os.path.join(config_dir, 'model_{}'.format(disease))
    if not os.path.isfile(model_path):
        raise FileNotFoundError('Missing checkpoint: {}'.format(model_path))

    if run.get('omidient'):
        from multiomics.code.prod_gamma_dirvae.evaluate import evaluate_config

        metrics = evaluate_config(disease, config_dir, run['config'], plot=False)
    else:
        mod = __import__(
            'multiomics.code.{}.evaluate'.format(run['pkg']),
            fromlist=['evaluate_config'],
        )
        metrics = mod.evaluate_config(disease, config_dir, plot=False)
    return config_dir, metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description='Re-eval KIRC tops for Ridge probes')
    parser.add_argument(
        '--runs',
        type=str,
        default='all',
        help='Comma-separated run ids or "all"',
    )
    parser.add_argument(
        '--out',
        type=str,
        default='',
        help='Write JSON summary (default: results/kirc_ridge_reeval.json)',
    )
    args = parser.parse_args(argv)

    if args.runs.strip().lower() in ('all', '*'):
        selected = KIRC_TOPS
    else:
        want = {p.strip() for p in args.runs.split(',') if p.strip()}
        selected = [r for r in KIRC_TOPS if r['id'] in want]
        missing = want - {r['id'] for r in selected}
        if missing:
            raise SystemExit('Unknown run id(s): {}'.format(sorted(missing)))

    out_path = args.out.strip() or os.path.join(_repo_results_root(), 'kirc_ridge_reeval.json')
    summary = {'note': 'Ridge Y PCA when dim>512 (see shared_rep_metrics._ridge_r2)', 'runs': {}}

    print('Ridge fix: PCA target to ≤50 comps when Y width > 512')
    print('Evaluating {} config(s)\n'.format(len(selected)))

    for run in selected:
        print('===== {} | {} ====='.format(run['id'], run['config']))
        try:
            config_dir, metrics = _evaluate(run)
        except Exception as e:
            print('FAILED:', e)
            summary['runs'][run['id']] = {'error': str(e), 'config': run['config']}
            continue
        row = {k: metrics.get(k) for k in RIDGE_KEYS}
        row.update({
            'config': run['config'],
            'label': run['label'],
            's_nmi': metrics.get('s_nmi'),
            'p_nmi': metrics.get('p_nmi'),
            'nmi': metrics.get('nmi'),
            'align_cka_mean': metrics.get('align_cka_mean'),
            'probe_subtype_from_S_other_acc': metrics.get('probe_subtype_from_S_other_acc'),
        })
        summary['runs'][run['id']] = row
        print(
            '  Δ X R² = {delta:.4f}  '
            '(S_other={s:.4f}, P_other={p:.4f})'.format(
                delta=row['probe_XA_S_minus_P_r2'],
                s=row['probe_XA_from_S_other_r2'],
                p=row['probe_XA_from_P_other_r2'],
            )
        )
        print(
            '  Own-view: X←S={xs:.4f}  X←P={xp:.4f}  X←[P;S]={xps:.4f}'.format(
                xs=row['probe_Xm_from_S_own_r2'],
                xp=row['probe_Xm_from_P_own_r2'],
                xps=row['probe_Xm_from_PS_own_r2'],
            )
        )
        # Persist next to the checkpoint for later README / viz updates
        metrics_path = os.path.join(config_dir, 'metrics_ridge_pca.json')
        with open(metrics_path, 'w') as f:
            json.dump({**row, **{k: metrics.get(k) for k in (
                'nmi', 'ari', 'acc', 'knn_acc', 's_nmi', 'p_nmi', 'align_cka_mean',
                'probe_subtype_from_S_other_acc',
            )}}, f, indent=2)
        print('  Wrote', metrics_path)
        print()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print('Summary:', out_path)

    # Compact table for paste into README
    print('\n| Method | Δ X R² | X←S own | X←P own | X←[P;S] own |')
    print('| ------ | ---: | ---: | ---: | ---: |')
    for run in selected:
        row = summary['runs'].get(run['id'], {})
        if 'error' in row:
            print('| {} | ERR | — | — | — |'.format(run['label']))
            continue
        print(
            '| {lab} | {d:.4f} | {xs:.4f} | {xp:.4f} | {xps:.4f} |'.format(
                lab=run['label'],
                d=row['probe_XA_S_minus_P_r2'],
                xs=row['probe_Xm_from_S_own_r2'],
                xp=row['probe_Xm_from_P_own_r2'],
                xps=row['probe_Xm_from_PS_own_r2'],
            )
        )
    return 0


if __name__ == '__main__':
    sys.exit(main())
