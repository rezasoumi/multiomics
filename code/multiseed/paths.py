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


def ensure_dir(path: str) -> str:
    """Create ``path`` (and parents). Remove broken symlinks that block makedirs."""
    path = os.path.abspath(path)
    built = os.sep if path.startswith(os.sep) else ''
    parts = path.strip(os.sep).split(os.sep) if path.startswith(os.sep) else path.split(os.sep)
    for part in parts:
        if not part:
            continue
        built = os.path.join(built, part) if built not in ('', os.sep) else (
            os.path.join(os.sep, part) if path.startswith(os.sep) else part
        )
        if os.path.islink(built) and not os.path.exists(built):
            os.unlink(built)
        elif os.path.isfile(built):
            raise NotADirectoryError(
                'Cannot create directory {!r}: {!r} is a file'.format(path, built)
            )
    os.makedirs(path, exist_ok=True)
    return path


def ensure_seed_run_dir(results_subdir: str, base_config: str, seed: int) -> str:
    return ensure_dir(seed_run_dir(results_subdir, base_config, seed))


def metrics_path(run_dir: str) -> str:
    return os.path.join(run_dir, 'metrics_multiseed.json')


def model_path(run_dir: str, disease: str = 'brca') -> str:
    return os.path.join(run_dir, 'model_{}'.format(disease))


def summary_dir() -> str:
    return os.path.join(repo_results_root(), 'multiseed')
