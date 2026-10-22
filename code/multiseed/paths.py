"""Paths for multi-seed runs: results/<subdir>/<base_config>/seed_<N>/."""

from __future__ import annotations

import os


def repo_results_root():
    """Absolute path to multiomics/results (works from any cwd)."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, '..', '..', 'results'))


def seed_run_dir(results_subdir: str, base_config: str, seed: int) -> str:
    return os.path.join(
        repo_results_root(),
        results_subdir,
        base_config,
        'seed_{}'.format(int(seed)),
    )


def metrics_path(run_dir: str) -> str:
    return os.path.join(run_dir, 'metrics_multiseed.json')


def model_path(run_dir: str, disease: str = 'brca') -> str:
    return os.path.join(run_dir, 'model_{}'.format(disease))


def summary_dir() -> str:
    return os.path.join(repo_results_root(), 'multiseed')
