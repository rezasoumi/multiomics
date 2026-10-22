"""CLI to generate shared/private visualization figures.

Examples (from anywhere, with PYTHONPATH set to the github parent):

  # Figs 1–4 only (README numbers; no checkpoints)
  python -m multiomics.code.viz.run_all --mode readme

  # All six figures (needs MULTIOMICS_RESULTS with checkpoints + leaderboards)
  python -m multiomics.code.viz.run_all --mode all

  # Individual groups
  python -m multiomics.code.viz.run_all --mode umap
  python -m multiomics.code.viz.run_all --mode sweeps

Outputs:
  docs/figures/*.png and *.pdf   (default)
  Override with --out-dir
"""

from __future__ import annotations

import argparse
import os
import sys


def default_out_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, '..', '..', 'docs', 'figures'))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Generate shared/private visualization figures for docs/README.md'
    )
    parser.add_argument(
        '--mode',
        choices=['readme', 'umap', 'sweeps', 'all'],
        default='all',
        help='readme=figs 1–4 from README numbers; umap=fig 5; sweeps=fig 6; all=1–6',
    )
    parser.add_argument(
        '--out-dir',
        type=str,
        default='',
        help='Output directory (default: <repo>/docs/figures)',
    )
    parser.add_argument('--disease', type=str, default='brca')
    parser.add_argument(
        '--reduce',
        choices=['umap', 'tsne'],
        default='umap',
        help='2D reducer for fig 5 (falls back to t-SNE if umap-learn missing)',
    )
    parser.add_argument(
        '--results-root',
        type=str,
        default='',
        help='Override MULTIOMICS_RESULTS / results root for checkpoints & CSVs',
    )
    args = parser.parse_args(argv)

    if args.results_root:
        os.environ['MULTIOMICS_RESULTS'] = os.path.abspath(args.results_root)

    out_dir = os.path.abspath(args.out_dir or default_out_dir())
    os.makedirs(out_dir, exist_ok=True)

    # Import after env override so diagnose_shared_private picks it up
    from multiomics.code.viz import plots

    print('out_dir:', out_dir)
    if args.mode in ('readme', 'all'):
        print('--- figs 1–4 (README numbers) ---')
        plots.plot_readme_only(out_dir)
    if args.mode in ('umap', 'all'):
        print('--- fig 5 (embeddings) ---')
        plots.plot_umap(out_dir, disease=args.disease, reduce=args.reduce)
    if args.mode in ('sweeps', 'all'):
        print('--- fig 6 (sweep heatmaps) ---')
        plots.plot_sweep_heatmaps(out_dir)

    print('\nDone. Figures are in:', out_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())
