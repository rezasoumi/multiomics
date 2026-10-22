"""Aggregate per-seed metrics_multiseed.json → mean ± std for README tables."""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np

from multiomics.code.multiseed.paper_configs import (
    COMPACT_KEYS,
    DETAIL_KEYS,
    PAPER_RUNS,
    SUMMARY_KEYS,
)
from multiomics.code.multiseed.paths import metrics_path, seed_run_dir, summary_dir


def _fmt_pm(mean: float, std: float, digits: int = 4) -> str:
    return '{:.{d}f} ± {:.{d}f}'.format(mean, std, d=digits)


def _load_seed_metrics(run: dict, seeds: List[int]) -> List[dict]:
    rows = []
    for seed in seeds:
        path = seed_run_dir(run['results_subdir'], run['base_config'], seed)
        mp = metrics_path(path)
        if not os.path.isfile(mp):
            print('Missing metrics for {} seed={}: {}'.format(run['id'], seed, mp))
            continue
        with open(mp, 'r') as f:
            rows.append(json.load(f))
    return rows


def _aggregate_keys(rows: List[dict], keys: List[str]) -> Dict[str, Any]:
    out = {}
    for key in keys:
        vals = []
        for r in rows:
            if key in r and r[key] is not None:
                vals.append(float(r[key]))
        if not vals:
            out[key] = None
            continue
        arr = np.asarray(vals, dtype=float)
        mean = float(arr.mean())
        std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        digits = 2 if key == 'knn_acc' else 4
        out[key] = {
            'mean': mean,
            'std': std,
            'n': len(arr),
            'values': vals,
            'display': _fmt_pm(mean, std, digits=digits),
        }
    return out


def aggregate_runs(
    seeds: List[int],
    run_ids: Optional[List[str]] = None,
) -> dict:
    selected = PAPER_RUNS
    if run_ids:
        id_set = set(run_ids)
        selected = [r for r in PAPER_RUNS if r['id'] in id_set]

    summary = {
        'seeds': list(seeds),
        'n_seeds_requested': len(seeds),
        'note': (
            'mean ± sample std (ddof=1) over training seeds; '
            'data splits fixed at random_state=1'
        ),
        'runs': {},
        'tables': {
            'compact': {},
            'summary': {},
            'family': {},
            'detail': {},
        },
    }

    for run in selected:
        rows = _load_seed_metrics(run, seeds)
        found_seeds = [int(r['seed']) for r in rows]
        block = {
            'id': run['id'],
            'label': run['label'],
            'base_config': run['base_config'],
            'tables': run['tables'],
            'seeds_found': found_seeds,
            'n_seeds_found': len(found_seeds),
            'per_seed': [
                {
                    'seed': int(r['seed']),
                    **{k: r.get(k) for k in DETAIL_KEYS if k in r},
                }
                for r in rows
            ],
            'metrics': _aggregate_keys(rows, DETAIL_KEYS),
        }
        summary['runs'][run['id']] = block

        table_keys = {
            'compact': COMPACT_KEYS,
            'summary': SUMMARY_KEYS,
            'family': COMPACT_KEYS + ['eval_loss'],
            'detail': DETAIL_KEYS,
        }
        for table, key_list in table_keys.items():
            if table not in run['tables']:
                continue
            agg = _aggregate_keys(rows, key_list)
            display = {
                k: (None if v is None else v['display']) for k, v in agg.items()
            }
            summary['tables'][table][run['id']] = {
                'label': run['label'],
                'base_config': run['base_config'],
                'n': len(rows),
                'display': display,
                'raw': agg,
            }

    return summary


def write_summary(summary: dict, out_dir: Optional[str] = None) -> str:
    out_dir = out_dir or summary_dir()
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, 'summary.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Flat CSV: one row per run × metric
    csv_path = os.path.join(out_dir, 'summary_flat.csv')
    fields = [
        'run_id',
        'label',
        'base_config',
        'metric',
        'mean',
        'std',
        'n',
        'display',
        'values',
    ]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for run_id, block in summary['runs'].items():
            for metric, stats in block['metrics'].items():
                if stats is None:
                    continue
                writer.writerow({
                    'run_id': run_id,
                    'label': block['label'],
                    'base_config': block['base_config'],
                    'metric': metric,
                    'mean': stats['mean'],
                    'std': stats['std'],
                    'n': stats['n'],
                    'display': stats['display'],
                    'values': ';'.join('{:.6g}'.format(v) for v in stats['values']),
                })

    # Markdown-ish display blocks for easy paste review
    md_path = os.path.join(out_dir, 'readme_snippets.md')
    lines = [
        '# Multi-seed README snippets',
        '',
        'Seeds: {}'.format(summary['seeds']),
        '',
        '## Compact comparison',
        '',
    ]
    for run_id, block in summary['tables']['compact'].items():
        d = block['display']
        lines.append(
            '- **{label}** (`{cfg}`, n={n}): '
            'mean(S) NMI={s_nmi} | concat(P)={p_nmi} | concat(P,S)={nmi} | '
            'gap={gap} | ΔX R²={dx} | X←S={xs} | X←P={xp} | X←[P;S]={xps} | CKA={cka}'.format(
                label=block['label'],
                cfg=block['base_config'],
                n=block['n'],
                s_nmi=d.get('s_nmi'),
                p_nmi=d.get('p_nmi'),
                nmi=d.get('nmi'),
                gap=d.get('gap_S_minus_PS'),
                dx=d.get('probe_XA_S_minus_P_r2'),
                xs=d.get('probe_Xm_from_S_own_r2'),
                xp=d.get('probe_Xm_from_P_own_r2'),
                xps=d.get('probe_Xm_from_PS_own_r2'),
                cka=d.get('align_cka_mean'),
            )
        )
    lines.extend(['', '## Summary (best by eval_loss)', ''])
    for run_id, block in summary['tables']['summary'].items():
        d = block['display']
        lines.append(
            '- **{label}**: concat NMI={nmi} | s_nmi={s_nmi} | ARI={ari} | '
            'Acc={acc} | kNN={knn} | eval_loss={ev}'.format(
                label=block['label'],
                nmi=d.get('nmi'),
                s_nmi=d.get('s_nmi'),
                ari=d.get('ari'),
                acc=d.get('acc'),
                knn=d.get('knn_acc'),
                ev=d.get('eval_loss'),
            )
        )
    lines.extend(['', '## Method 2 / OMIDIENT family', ''])
    for run_id, block in summary['tables']['family'].items():
        d = block['display']
        lines.append(
            '- **{label}**: mean(S)={s_nmi} | concat(P)={p_nmi} | concat(P,S)={nmi} | '
            'ΔX={dx} | X←S={xs} | X←P={xp} | X←[P;S]={xps} | CKA={cka} | eval={ev}'.format(
                label=block['label'],
                s_nmi=d.get('s_nmi'),
                p_nmi=d.get('p_nmi'),
                nmi=d.get('nmi'),
                dx=d.get('probe_XA_S_minus_P_r2'),
                xs=d.get('probe_Xm_from_S_own_r2'),
                xp=d.get('probe_Xm_from_P_own_r2'),
                xps=d.get('probe_Xm_from_PS_own_r2'),
                cka=d.get('align_cka_mean'),
                ev=d.get('eval_loss'),
            )
        )
    with open(md_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print('Wrote', json_path)
    print('Wrote', csv_path)
    print('Wrote', md_path)
    return json_path
