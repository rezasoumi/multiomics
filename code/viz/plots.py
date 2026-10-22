"""Generate the six README visualization figures."""

from __future__ import annotations

import csv
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from multiomics.code.viz.panel_data import (
    AE_PANEL,
    OMI_PANEL,
    SWEEP_SPECS,
    UMAP_JOBS,
)
from multiomics.code.viz.style import (
    COLOR_GRID,
    COLOR_P,
    COLOR_PS,
    COLOR_S,
    METHOD_MARKERS,
    apply_style,
)

warnings.filterwarnings('ignore')


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def _save(fig, out_dir, stem):
    _ensure_dir(out_dir)
    png = os.path.join(out_dir, '{}.png'.format(stem))
    pdf = os.path.join(out_dir, '{}.pdf'.format(stem))
    fig.savefig(png)
    fig.savefig(pdf)
    plt.close(fig)
    print('Wrote', png)
    return png


# ---------------------------------------------------------------------------
# Fig 1 — S / P / [P;S] NMI grouped bars
# ---------------------------------------------------------------------------

def plot_nmi_bars(out_dir):
    apply_style()
    panels = [
        ('AE-private methods', AE_PANEL, '01_nmi_bars_ae'),
        ('Method 2 / OMIDIENT family', OMI_PANEL, '01_nmi_bars_omidient'),
    ]
    paths = []
    for title, rows, stem in panels:
        labels = [r['short'] for r in rows]
        x = np.arange(len(labels))
        width = 0.25
        fig_w = max(9.5, 1.55 * len(labels) + 2.5)
        fig, ax = plt.subplots(figsize=(fig_w, 4.2))
        ax.bar(x - width, [r['s_nmi'] for r in rows], width, label='mean(S)', color=COLOR_S)
        ax.bar(x, [r['p_nmi'] for r in rows], width, label='concat(P)', color=COLOR_P)
        ax.bar(x + width, [r['ps_nmi'] for r in rows], width, label='concat(P,S)', color=COLOR_PS)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha='right')
        ax.set_ylabel('NMI')
        ax.set_ylim(0, max(0.55, max(r['ps_nmi'] for r in rows) + 0.08))
        ax.set_title('Shared / private clustering NMI — {}'.format(title))
        ax.legend(frameon=False, loc='upper right')
        ax.axhline(0, color='#888888', lw=0.5)
        ax.yaxis.grid(True, color=COLOR_GRID, lw=0.6)
        ax.set_axisbelow(True)
        paths.append(_save(fig, out_dir, stem))
    return paths


# ---------------------------------------------------------------------------
# Fig 2 — scatter: subtype separation vs Δ X R²
# ---------------------------------------------------------------------------

def plot_scatter(out_dir):
    apply_style()
    rows = AE_PANEL + [r for r in OMI_PANEL if r['label'] != 'M2 (AE)']
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    for r in rows:
        sep = r['s_nmi'] - r['p_nmi']
        ax.scatter(
            sep,
            r['delta_x_r2'],
            s=110,
            marker=METHOD_MARKERS.get(r['label'], 'o'),
            color=COLOR_S if r['family'] == 'ae' else COLOR_P,
            edgecolors='black',
            linewidths=0.6,
            zorder=3,
            label=r['label'],
        )
        ax.annotate(
            r['short'],
            (sep, r['delta_x_r2']),
            textcoords='offset points',
            xytext=(6, 5),
            fontsize=8,
        )
    ax.axhline(0, color='#888888', lw=0.8, ls='--')
    ax.axvline(0, color='#888888', lw=0.8, ls='--')
    ax.set_xlabel('Subtype separation: mean(S) NMI − concat(P) NMI')
    ax.set_ylabel('Cross-view reconstructiveness: Δ X R²  (S_other − P_other)')
    ax.set_title('Decomposition axes: subtype coding ≠ reconstruction coding')
    ax.legend(frameon=False, loc='best', fontsize=8)
    ax.grid(True, color=COLOR_GRID, lw=0.6)
    return [_save(fig, out_dir, '02_scatter_separation_vs_delta_xr2')]


# ---------------------------------------------------------------------------
# Fig 3 — color-coded metric heatmap (compact panel)
# ---------------------------------------------------------------------------

def plot_metric_heatmap(out_dir):
    apply_style()
    panels = [
        ('AE-private compact panel', AE_PANEL, '03_metric_heatmap_ae'),
        ('Method 2 / OMIDIENT family', OMI_PANEL, '03_metric_heatmap_omidient'),
    ]
    metric_keys = [
        ('s_nmi', 'mean(S) NMI'),
        ('p_nmi', 'concat(P) NMI'),
        ('ps_nmi', 'concat(P,S) NMI'),
        ('gap_s_ps', 'gap S−PS'),
        ('delta_x_r2', 'Δ X R²'),
        ('x_s_own', 'X←S own'),
        ('x_p_own', 'X←P own'),
        ('cka', 'CKA'),
    ]
    diverging = {'gap_s_ps', 'delta_x_r2'}
    paths = []
    for title, rows, stem in panels:
        mat = np.array([[r[k] for k, _ in metric_keys] for r in rows], dtype=float)
        fig, ax = plt.subplots(figsize=(10.5, 0.9 + 0.55 * len(rows)))
        # Draw column-by-column so diverging cols use a different cmap
        for j, (key, _) in enumerate(metric_keys):
            col = mat[:, j:j + 1]
            if key in diverging:
                vmax = max(0.05, np.nanmax(np.abs(col)))
                norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
                ax.imshow(
                    col,
                    aspect='auto',
                    cmap='RdYlGn',
                    norm=norm,
                    extent=(j - 0.5, j + 0.5, len(rows) - 0.5, -0.5),
                    origin='upper',
                )
            else:
                # Higher is better for most; for p_nmi lower can be "cleaner" but
                # we still show raw value intensity in Blues for consistency.
                cmap = 'Oranges' if key == 'p_nmi' else 'Blues'
                ax.imshow(
                    col,
                    aspect='auto',
                    cmap=cmap,
                    extent=(j - 0.5, j + 0.5, len(rows) - 0.5, -0.5),
                    origin='upper',
                )
            for i in range(len(rows)):
                val = mat[i, j]
                ax.text(j, i, '{:.2f}'.format(val), ha='center', va='center', fontsize=8)
        ax.set_xticks(range(len(metric_keys)))
        ax.set_xticklabels([lab for _, lab in metric_keys], rotation=30, ha='right')
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r['short'] for r in rows])
        ax.set_xlim(-0.5, len(metric_keys) - 0.5)
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.set_title('Color-coded metrics — {}'.format(title))
        # Legend strip for color code
        ax.text(
            0.0,
            -0.35,
            'Blue = S / higher shared · Orange = P · Green↔Red = signed gap / Δ around 0',
            transform=ax.transAxes,
            fontsize=8,
            color='#555555',
        )
        paths.append(_save(fig, out_dir, stem))
    return paths


# ---------------------------------------------------------------------------
# Fig 4 — method schematic (visual color code of the process)
# ---------------------------------------------------------------------------

def _box(ax, xy, w, h, text, facecolor, fontsize=9):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle='round,pad=0.02,rounding_size=0.04',
        linewidth=1.2,
        edgecolor='#333333',
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fontsize, color='#111111')


def _arrow(ax, p0, p1, color='#444444'):
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle='-|>',
            mutation_scale=12,
            linewidth=1.2,
            color=color,
        )
    )


def plot_schematic(out_dir):
    apply_style()
    fig, ax = plt.subplots(figsize=(12.5, 6.0))
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 6.0)
    ax.axis('off')
    ax.set_title('Shared / private process (visual color code)', pad=12)

    # Inputs
    _box(ax, (0.3, 4.2), 1.6, 0.7, 'X₁ (omics 1)', '#F0F0F0')
    _box(ax, (0.3, 3.0), 1.6, 0.7, 'X₂ (omics 2)', '#F0F0F0')
    _box(ax, (0.3, 1.8), 1.6, 0.7, 'X₃ (omics 3)', '#F0F0F0')

    # Encoders
    _box(ax, (2.4, 2.6), 1.5, 1.2, 'Encoders\nE_m', '#E8E8E8', fontsize=8)

    # Latents
    _box(ax, (4.6, 4.0), 1.7, 0.85, 'S_m  shared', COLOR_S)
    _box(ax, (4.6, 1.8), 1.7, 0.85, 'P_m  private', COLOR_P)

    # Downstream (shared-side objectives)
    _box(ax, (7.1, 5.0), 2.0, 0.65, 'Contrastive on S\n(M1/M2/OMI)', '#BDD7E7', fontsize=7.5)
    _box(ax, (7.1, 4.15), 2.0, 0.65, 'M3: L_spec → fused\nLaplacian U', '#C7E9C0', fontsize=7.5)
    _box(ax, (7.1, 3.3), 2.0, 0.65, 'M4: DEC + agree\non soft clusters', '#A1D99B', fontsize=7.5)
    _box(ax, (7.1, 2.45), 2.0, 0.65, 'Recon / f,g,k,b\nor residual D', '#FEE6CE', fontsize=7.5)
    _box(ax, (7.1, 1.6), 2.0, 0.65, 'VICReg / ortho /\nview-ID + leak GRL', '#FDD0A2', fontsize=7.5)
    _box(ax, (7.1, 0.7), 2.0, 0.65, 'Eval: KMeans on\nmean(S), P, [P;S]', COLOR_PS, fontsize=7.5)

    # Method callouts
    _box(ax, (9.6, 4.15), 2.5, 0.65, 'M3 shared = spectral\ngraph geometry', '#E5F5E0', fontsize=7.5)
    _box(ax, (9.6, 3.3), 2.5, 0.65, 'M4 shared = invariant\ncross-omics clustering', '#E5F5E0', fontsize=7.5)
    _box(ax, (9.6, 1.6), 2.5, 0.65, 'M3/M4 private =\nresidual + view − leak', '#FFF5EB', fontsize=7.5)

    _arrow(ax, (1.9, 4.55), (2.4, 3.5))
    _arrow(ax, (1.9, 3.35), (2.4, 3.2))
    _arrow(ax, (1.9, 2.15), (2.4, 2.9))
    _arrow(ax, (3.9, 3.4), (4.6, 4.35), COLOR_S)
    _arrow(ax, (3.9, 2.9), (4.6, 2.3), COLOR_P)
    _arrow(ax, (6.3, 4.45), (7.1, 5.3), COLOR_S)
    _arrow(ax, (6.3, 4.4), (7.1, 4.45), COLOR_S)
    _arrow(ax, (6.3, 4.3), (7.1, 3.6), COLOR_S)
    _arrow(ax, (6.3, 4.1), (7.1, 2.75), COLOR_S)
    _arrow(ax, (6.3, 2.2), (7.1, 2.75), COLOR_P)
    _arrow(ax, (6.3, 2.1), (7.1, 1.9), COLOR_P)
    _arrow(ax, (6.3, 4.3), (7.1, 1.0), COLOR_PS)

    ax.text(
        0.3,
        0.2,
        'Blue = shared S · Orange = private P · Purple = fused eval · '
        'M1: f/g/k/b · M2: joint recon + VICReg · M3: spectral U · M4: DEC/agree · '
        'OMIDIENT: Dirichlet on P.',
        fontsize=8,
        color='#444444',
    )
    return [_save(fig, out_dir, '04_method_schematic')]


# ---------------------------------------------------------------------------
# Fig 5 — UMAP / t-SNE of mean(S) and concat(P)
# ---------------------------------------------------------------------------

def _reduce_2d(X, method='umap', random_state=42):
    X = np.asarray(X, dtype=np.float64)
    # Light PCA pre-step for stability when dim is large
    if X.shape[1] > 32:
        from sklearn.decomposition import PCA

        X = PCA(n_components=min(32, X.shape[0] - 1), random_state=random_state).fit_transform(X)
    if method == 'umap':
        try:
            import umap

            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=random_state)
            return reducer.fit_transform(X), 'UMAP'
        except Exception:
            method = 'tsne'
    from sklearn.manifold import TSNE

    perplexity = max(5, min(30, X.shape[0] // 4))
    emb = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        init='pca',
        learning_rate='auto',
    ).fit_transform(X)
    return emb, 't-SNE'


def plot_umap(out_dir, disease='brca', reduce='umap'):
    apply_style()
    from multiomics.code.diagnose_shared_private import load_latents, METHOD_DIRS, RESULTS_ROOT

    jobs = []
    for method, config, label in UMAP_JOBS:
        parent = os.path.join(RESULTS_ROOT, METHOD_DIRS[method].format(disease))
        config_dir = os.path.join(parent, config)
        if os.path.isdir(config_dir) and os.path.isfile(
            os.path.join(config_dir, 'model_{}'.format(disease))
        ):
            jobs.append((method, config, label, config_dir))
        else:
            print('SKIP UMAP (missing):', config_dir)

    if not jobs:
        print('No checkpoints found for UMAP under', RESULTS_ROOT)
        return []

    n = len(jobs)
    fig, axes = plt.subplots(n, 2, figsize=(8.5, 2.6 * n), squeeze=False)
    used_name = 'UMAP'
    last_sc = None
    for row_i, (method, config, label, config_dir) in enumerate(jobs):
        try:
            specific, shared, _views, y, _model, _out = load_latents(
                method, disease, config_dir
            )
        except Exception as e:
            print('SKIP UMAP error {} / {}: {}'.format(method, config, e))
            for c in (0, 1):
                axes[row_i, c].set_visible(False)
            continue
        s_mean = (shared[1] + shared[2] + shared[3]) / 3.0
        p_cat = np.concatenate([specific[1], specific[2], specific[3]], axis=1)
        for col, (Z, title, color_edge) in enumerate(
            (
                (s_mean, 'mean(S)', COLOR_S),
                (p_cat, 'concat(P)', COLOR_P),
            )
        ):
            emb, used_name = _reduce_2d(Z, method=reduce)
            ax = axes[row_i, col]
            last_sc = ax.scatter(
                emb[:, 0],
                emb[:, 1],
                c=y,
                cmap='tab10',
                s=12,
                alpha=0.85,
                edgecolors='none',
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title('{} — {}'.format(label, title), fontsize=10, color=color_edge)
            for spine in ax.spines.values():
                spine.set_edgecolor(color_edge)
                spine.set_linewidth(1.5)

    if last_sc is not None:
        fig.colorbar(last_sc, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01, label='subtype')
    fig.suptitle(
        '{} of shared vs private latents (colored by BRCA subtype)'.format(used_name),
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()
    return [_save(fig, out_dir, '05_embedding_{}'.format(used_name.lower().replace('-', '')))]


# ---------------------------------------------------------------------------
# Fig 6 — hyperparameter sweep heatmaps from leaderboard CSVs
# ---------------------------------------------------------------------------

def _read_leaderboard(path):
    with open(path, 'r', newline='') as f:
        return list(csv.DictReader(f))


def _to_float(x):
    if x is None or x == '':
        return float('nan')
    return float(x)


def _pivot(rows, x_key, y_key, z_key):
    xs, ys, zs = [], [], []
    for r in rows:
        if r.get(x_key) in (None, '') or r.get(y_key) in (None, ''):
            continue
        if r.get(z_key) in (None, ''):
            continue
        xs.append(_to_float(r[x_key]))
        ys.append(_to_float(r[y_key]))
        zs.append(_to_float(r[z_key]))
    if not xs:
        return None
    x_vals = sorted(set(xs))
    y_vals = sorted(set(ys))
    # Aggregate duplicates by mean
    grid = {(xv, yv): [] for xv in x_vals for yv in y_vals}
    for x, y, z in zip(xs, ys, zs):
        grid[(x, y)].append(z)
    mat = np.full((len(y_vals), len(x_vals)), np.nan)
    for i, yv in enumerate(y_vals):
        for j, xv in enumerate(x_vals):
            vals = grid[(xv, yv)]
            if vals:
                mat[i, j] = float(np.mean(vals))
    return x_vals, y_vals, mat


def plot_sweep_heatmaps(out_dir, results_root=None):
    apply_style()
    if results_root is None:
        from multiomics.code.diagnose_shared_private import RESULTS_ROOT

        results_root = RESULTS_ROOT

    paths = []
    for spec in SWEEP_SPECS:
        csv_path = os.path.join(results_root, spec['csv'])
        if not os.path.isfile(csv_path):
            print('SKIP sweep (missing CSV):', csv_path)
            continue
        rows = _read_leaderboard(csv_path)
        if spec.get('filter_priv'):
            rows = [r for r in rows if r.get('lambda_priv') not in (None, '')]
        if not rows:
            print('SKIP sweep (empty):', csv_path)
            continue

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for ax, z_key, z_title, cmap, invert in (
            (axes[0], 's_nmi', 'mean(S) NMI', 'Blues', False),
            (axes[1], 'eval_loss', 'eval_loss (lower better)', 'YlOrRd_r', True),
        ):
            pivoted = _pivot(rows, spec['x'], spec['y'], z_key)
            if pivoted is None:
                ax.set_visible(False)
                continue
            x_vals, y_vals, mat = pivoted
            im = ax.imshow(mat, aspect='auto', cmap=cmap, origin='lower')
            ax.set_xticks(range(len(x_vals)))
            ax.set_xticklabels(['{:g}'.format(v) for v in x_vals], rotation=45, ha='right')
            ax.set_yticks(range(len(y_vals)))
            ax.set_yticklabels(['{:g}'.format(v) for v in y_vals])
            ax.set_xlabel(spec['x'])
            ax.set_ylabel(spec['y'])
            ax.set_title(z_title)
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    if np.isfinite(mat[i, j]):
                        ax.text(
                            j,
                            i,
                            '{:.2f}'.format(mat[i, j]),
                            ha='center',
                            va='center',
                            fontsize=7,
                            color='black',
                        )
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            _ = invert  # cmap maps already encode direction
        fig.suptitle(spec['title'] + '  (n={})'.format(len(rows)), fontsize=12)
        fig.tight_layout()
        paths.append(_save(fig, out_dir, '06_sweep_{}'.format(spec['name'])))
    return paths


def plot_readme_only(out_dir):
    """Figs 1–4: no checkpoints required."""
    paths = []
    paths += plot_nmi_bars(out_dir)
    paths += plot_scatter(out_dir)
    paths += plot_metric_heatmap(out_dir)
    paths += plot_schematic(out_dir)
    return paths


def plot_all(out_dir, disease='brca', reduce='umap'):
    paths = plot_readme_only(out_dir)
    paths += plot_umap(out_dir, disease=disease, reduce=reduce)
    paths += plot_sweep_heatmaps(out_dir)
    return paths
