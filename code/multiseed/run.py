"""CLI: multi-seed retrain of paper-priority configs + aggregate mean ± std.

Examples:
  python -m multiomics.code.multiseed.run --seeds 21,22,23,24,25 --resume
  python -m multiomics.code.multiseed.run --runs method1,mocss --seeds 21,22 --resume
  python -m multiomics.code.multiseed.run --aggregate-only --seeds 21,22,23,24,25
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

from multiomics.code.multiseed.aggregate import aggregate_runs, write_summary
from multiomics.code.multiseed.paper_configs import (
    DEFAULT_SEEDS,
    PAPER_RUNS,
    get_run,
    list_run_ids,
)
from multiomics.code.multiseed.trainers import run_one


def _parse_seeds(text: str):
    seeds = []
    for part in text.split(','):
        part = part.strip()
        if not part:
            continue
        seeds.append(int(part))
    if not seeds:
        raise ValueError('No seeds parsed from {!r}'.format(text))
    return seeds


def _parse_runs(text: str):
    if not text or text.strip().lower() in ('all', '*'):
        return list_run_ids()
    ids = [p.strip() for p in text.split(',') if p.strip()]
    known = set(list_run_ids())
    bad = [i for i in ids if i not in known]
    if bad:
        raise ValueError(
            'Unknown run id(s): {}. Choose from: {}'.format(bad, list_run_ids())
        )
    return ids


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Multi-seed paper-config retrain + aggregate for README ±'
    )
    parser.add_argument(
        '--seeds',
        type=str,
        default=','.join(str(s) for s in DEFAULT_SEEDS),
        help='Comma-separated training seeds (default: {})'.format(DEFAULT_SEEDS),
    )
    parser.add_argument(
        '--runs',
        type=str,
        default='all',
        help='Comma-separated run ids or "all" (default). Options: {}'.format(
            list_run_ids()
        ),
    )
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Skip seed dirs that already have model_* + metrics_multiseed.json',
    )
    parser.add_argument(
        '--aggregate-only',
        action='store_true',
        help='Only build results/multiseed/summary*.{json,csv,md} from existing runs',
    )
    parser.add_argument(
        '--list-runs',
        action='store_true',
        help='Print paper run registry and exit',
    )
    args = parser.parse_args(argv)

    if args.list_runs:
        for run in PAPER_RUNS:
            print(
                '{id}: {label} | {cfg} | tables={tables}'.format(
                    id=run['id'],
                    label=run['label'],
                    cfg=run['base_config'],
                    tables=','.join(run['tables']),
                )
            )
        return 0

    seeds = _parse_seeds(args.seeds)
    run_ids = _parse_runs(args.runs)

    print('USE_GPU={} | device={}'.format(
        torch.cuda.is_available(),
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
    ))
    print('Seeds:', seeds)
    print('Runs:', run_ids)

    if not args.aggregate_only:
        for run_id in run_ids:
            run = get_run(run_id)
            for seed in seeds:
                run_one(
                    run,
                    seed=seed,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    resume=args.resume,
                )

    summary = aggregate_runs(seeds=seeds, run_ids=run_ids)
    out = write_summary(summary)
    print('\nDone. Provide this file for README updates:')
    print(' ', out)
    print('Also useful:')
    print(' ', os.path.join(os.path.dirname(out), 'readme_snippets.md'))
    print(' ', os.path.join(os.path.dirname(out), 'summary_flat.csv'))
    # Print a compact status of missing seeds
    for run_id, block in summary['runs'].items():
        missing = sorted(set(seeds) - set(block['seeds_found']))
        if missing:
            print('WARNING: {} missing seeds {}'.format(run_id, missing))
        else:
            print('OK: {} n={}'.format(run_id, block['n_seeds_found']))
    return 0


if __name__ == '__main__':
    sys.exit(main())
