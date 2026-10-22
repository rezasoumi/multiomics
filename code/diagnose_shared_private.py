"""Diagnostics: why concat(P,S) can be worse than mean(S), and why Δ(S−P) can go negative.

Loads an existing checkpoint (no training). Prints / saves:
  - clustering on mean(S), concat(P), concat(P,S), each S_m / P_m
  - subtype linear probes from S_other vs P_other
  - X←S_other vs X←P_other Ridge R² (same as shared_rep_metrics)
  - latent norms / stds (collapse checks)
  - optional Method-1 decoder probe: recon error of h/f/g/k/b on test (if available)

Usage examples:
  python diagnose_shared_private.py --method method1 --config 0.0004_0.0007_a0.7_b0.7_g0.7_l1.0
  python diagnose_shared_private.py --method mocss --config 0.0004_0.0007
"""

from __future__ import annotations

import argparse
import json
import os
import warnings

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from multiomics.code.shared_rep_metrics import (
    clustering_and_knn,
    probe_XA_from_other,
    alignment_metrics,
)

warnings.filterwarnings('ignore')

NUM_CLUST = {'lihc': 2, 'coad': 4, 'kirc': 2, 'brca': 5}

# This file lives in <repo>/code/ → results are <repo>/results/
# Override with env MULTIOMICS_RESULTS if checkpoints live elsewhere.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RESULTS_ROOT = os.path.abspath(
    os.environ.get('MULTIOMICS_RESULTS', os.path.join(REPO_ROOT, 'results'))
)

METHOD_DIRS = {
    'method1': 'models_{}_method1',
    'method1_vicreg': 'models_{}_method1_vicreg',
    'method2': 'models_{}_method2',
    'method2_omidient': 'models_{}_method2_omidient',
    'mocss': 'models_{}_ae',
    'omidient': 'models_{}_ProdGammaDirVae',
}


def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def truth_from_label(disease, label):
    if disease == 'coad':
        return label.flatten().astype('int')
    if disease == 'lihc':
        return label[:, 1].flatten().astype('int')
    if disease == 'kirc':
        return label[:, 1].flatten().astype('int')
    return label.flatten()


def _latent_stats(name, Z):
    return {
        '{}_mean_norm'.format(name): float(np.linalg.norm(Z, axis=1).mean()),
        '{}_dim_std_mean'.format(name): float(Z.std(axis=0).mean()),
        '{}_dim_std_min'.format(name): float(Z.std(axis=0).min()),
        '{}_frac_std_lt_0.1'.format(name): float(np.mean(Z.std(axis=0) < 0.1)),
    }


def _subtype_probe(X, y):
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.25, random_state=12, stratify=y
        )
    except ValueError:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.25, random_state=12
        )
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_tr, y_tr)
    return float(accuracy_score(y_te, clf.predict(X_te)))


def infer_view_sizes(method, state):
    """Read input dims from checkpoint weights (no CSV / load_data needed)."""
    if method in ('mocss', 'omidient'):
        d1 = int(state['shared1_l1.weight'].shape[1])
        d2 = int(state['shared2_l1.weight'].shape[1])
        d3 = int(state['shared3_l1.weight'].shape[1])
        return [d1, d2, d3]
    # Method1 / VICReg / Method2 / Method2+OMIDIENT ModuleDict encoders
    d1 = int(state['shared_enc.1.0.weight'].shape[1])
    d2 = int(state['shared_enc.2.0.weight'].shape[1])
    d3 = int(state['shared_enc.3.0.weight'].shape[1])
    return [d1, d2, d3]


def load_latents(method, disease, config_dir):
    """Load frozen latents from an existing checkpoint + saved test split.

    Does not call loading_data.load_data (avoids fragile ../../data CWD paths).
    View sizes are inferred from the checkpoint; inputs come from test_data_*.npy.
    """
    model_path = os.path.join(config_dir, 'model_{}'.format(disease))
    data_path = os.path.join(config_dir, 'test_data_{}.npy'.format(disease))
    label_path = os.path.join(config_dir, 'test_label_{}.npy'.format(disease))
    if not os.path.isfile(model_path):
        raise FileNotFoundError(model_path)
    if not os.path.isfile(data_path):
        raise FileNotFoundError(data_path)

    state = torch.load(model_path, map_location='cpu')
    view_size = infer_view_sizes(method, state)
    d1, d2, d3 = view_size
    data = np.load(data_path)
    label = np.load(label_path, allow_pickle=True)
    if data.shape[1] != d1 + d2 + d3:
        raise ValueError(
            'test_data width {} != sum(view_size) {} for {}'.format(
                data.shape[1], d1 + d2 + d3, config_dir
            )
        )

    view1 = torch.tensor(data[:, :d1], dtype=torch.float32)
    view2 = torch.tensor(data[:, d1:d1 + d2], dtype=torch.float32)
    view3 = torch.tensor(data[:, d1 + d2:], dtype=torch.float32)
    views = {1: view1.numpy(), 2: view2.numpy(), 3: view3.numpy()}
    y = truth_from_label(disease, label)
    setup_seed(2)
    print(
        'Loaded {} | view_size={} | n_test={}'.format(
            config_dir, view_size, data.shape[0]
        )
    )

    if method == 'mocss':
        from multiomics.code.ae.mocss_original_refactored import SharedAndSpecificEmbedding

        model = SharedAndSpecificEmbedding(
            view_size=view_size,
            n_units_1=[512, 256, 128, 32],
            n_units_2=[512, 256, 128, 32],
            n_units_3=[256, 128, 64, 32],
            mlp_size=[32, 8],
        )
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            s1, sh1, s2, sh2, s3, sh3, *_ = model(view1, view2, view3)
        specific = {1: s1.numpy(), 2: s2.numpy(), 3: s3.numpy()}
        shared = {1: sh1.numpy(), 2: sh2.numpy(), 3: sh3.numpy()}
        return specific, shared, views, y, model, None

    if method == 'omidient':
        from multiomics.code.prod_gamma_dirvae.evaluate import parse_k_from_config
        from multiomics.code.prod_gamma_dirvae.prod_gamma_dirvae_cancer import (
            SharedAndSpecificEmbedding,
        )

        # Folder may be lr_wd_K or lr_wd_K_ortho_c…_o… — never take the last token.
        k_groups = parse_k_from_config(os.path.basename(config_dir.rstrip('/')))
        model = SharedAndSpecificEmbedding(
            'ProdGamDirVae',
            k_groups,
            view_size=view_size,
            n_units_1=[512, 256, 128, 8],
            n_units_2=[512, 256, 128, 8],
            n_units_3=[256, 128, 64, 8],
            mlp_size=[32, 8],
        )
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            s1, _a1, sh1, s2, _a2, sh2, s3, _a3, sh3, *_ = model(view1, view2, view3)
        specific = {1: s1.numpy(), 2: s2.numpy(), 3: s3.numpy()}
        shared = {1: sh1.numpy(), 2: sh2.numpy(), 3: sh3.numpy()}
        return specific, shared, views, y, model, None

    if method == 'method1':
        from multiomics.code.method1_predictability.model import SharedAndSpecificEmbedding

        model = SharedAndSpecificEmbedding(
            view_size=view_size,
            n_units_1=[512, 256, 128, 32],
            n_units_2=[512, 256, 128, 32],
            n_units_3=[256, 128, 64, 32],
            mlp_size=[32, 8],
        )
    elif method == 'method1_vicreg':
        from multiomics.code.method1_vicreg_ablation.model import SharedAndSpecificEmbedding

        model = SharedAndSpecificEmbedding(
            view_size=view_size,
            n_units_1=[512, 256, 128, 32],
            n_units_2=[512, 256, 128, 32],
            n_units_3=[256, 128, 64, 32],
            mlp_size=[32, 8],
        )
    elif method == 'method2_omidient':
        from multiomics.code.method2_omidient_private.evaluate import (
            build_model,
            parse_k_from_config,
        )

        folder = os.path.basename(config_dir.rstrip('/'))
        model = build_model(view_size, parse_k_from_config(folder))
    else:
        from multiomics.code.method2_joint_vicreg.model import SharedAndSpecificEmbedding

        model = SharedAndSpecificEmbedding(
            view_size=view_size,
            n_units_1=[512, 256, 128, 32],
            n_units_2=[512, 256, 128, 32],
            n_units_3=[256, 128, 64, 32],
            mlp_size=[32, 8],
        )

    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        out = model(view1, view2, view3)
    specific = {i: out['specific'][i].numpy() for i in (1, 2, 3)}
    shared = {i: out['shared'][i].numpy() for i in (1, 2, 3)}
    return specific, shared, views, y, model, out


def method1_decoder_errors(model, out, views):
    """Mean MOCSS-style matrix-norm recon error for h/f/g(/k)/b on the test batch."""
    import torch.nn.functional as F

    def recon_err(rec, ori):
        rec = torch.tensor(rec) if not torch.is_tensor(rec) else rec
        ori = torch.tensor(ori) if not torch.is_tensor(ori) else ori
        rec = rec - rec.mean()
        ori = ori - ori.mean()
        rec = F.normalize(rec, p=2, dim=1)
        ori = F.normalize(ori, p=2, dim=1)
        return float(torch.linalg.matrix_norm(rec - ori).item())

    # out already has predictions if model forward computed them
    errs = {}
    if out is None:
        return errs
    for key in ('h_rec', 'f_rec', 'g_rec', 'k_rec', 'b_rec', 'joint_rec'):
        if key not in out:
            continue
        total = 0.0
        for a in (1, 2, 3):
            total += recon_err(out[key][a], views[a])
        errs['decoder_{}_err'.format(key)] = total / 3.0
    return errs


def diagnose(method, disease, config):
    parent_tmpl = METHOD_DIRS[method]
    parent = os.path.join(RESULTS_ROOT, parent_tmpl.format(disease))
    config_dir = os.path.join(parent, config)
    if not os.path.isdir(config_dir):
        raise FileNotFoundError(os.path.abspath(config_dir))

    n_clusters = NUM_CLUST[disease]
    specific, shared, views, y, model, out = load_latents(method, disease, config_dir)

    s_mean = (shared[1] + shared[2] + shared[3]) / 3.0
    p_cat = np.concatenate([specific[1], specific[2], specific[3]], axis=1)
    z = np.concatenate([p_cat, s_mean], axis=1)

    report = {
        'method': method,
        'disease': disease,
        'config': config,
    }

    for name, X in (
        ('mean_S', s_mean),
        ('concat_P', p_cat),
        ('concat_PS', z),
    ):
        m = clustering_and_knn(X, y, n_clusters)
        report['{}_nmi'.format(name)] = m['nmi']
        report['{}_ari'.format(name)] = m['ari']
        report['{}_acc'.format(name)] = m['acc']
        report['{}_knn'.format(name)] = m['knn_acc']

    for i in (1, 2, 3):
        ms = clustering_and_knn(shared[i], y, n_clusters)
        mp = clustering_and_knn(specific[i], y, n_clusters)
        report['S{}_nmi'.format(i)] = ms['nmi']
        report['P{}_nmi'.format(i)] = mp['nmi']
        report.update(_latent_stats('S{}'.format(i), shared[i]))
        report.update(_latent_stats('P{}'.format(i), specific[i]))

    # subtype from other-view S vs P
    others = {1: (2, 3), 2: (1, 3), 3: (1, 2)}
    s_accs, p_accs = [], []
    for a, (b, c) in others.items():
        Xs = np.concatenate([shared[b], shared[c]], axis=1)
        Xp = np.concatenate([specific[b], specific[c]], axis=1)
        as_ = _subtype_probe(Xs, y)
        ap_ = _subtype_probe(Xp, y)
        s_accs.append(as_)
        p_accs.append(ap_)
        report['subtype_from_S_other_acc_{}'.format(a)] = as_
        report['subtype_from_P_other_acc_{}'.format(a)] = ap_
    report['subtype_from_S_other_acc'] = float(np.mean(s_accs))
    report['subtype_from_P_other_acc'] = float(np.mean(p_accs))
    report['subtype_S_minus_P_acc'] = float(np.mean(s_accs) - np.mean(p_accs))

    report.update(probe_XA_from_other(views, shared, specific))
    report.update(alignment_metrics(shared))

    # Gap that explains "P hurts"
    report['gap_meanS_minus_concatPS_nmi'] = report['mean_S_nmi'] - report['concat_PS_nmi']
    report['gap_meanS_minus_concatP_nmi'] = report['mean_S_nmi'] - report['concat_P_nmi']

    if method in ('method1', 'method1_vicreg', 'method2'):
        # Need fresh forward with predictions for method1-style outs
        if method != 'method2' and out is not None:
            # method1 forward may already include h_rec etc.
            report.update(method1_decoder_errors(model, out, views))
        elif method == 'method2' and out is not None:
            report.update(method1_decoder_errors(model, out, views))

    return report, config_dir


def print_report(r):
    print('\n======== {} | {} ========'.format(r['method'], r['config']))
    print(
        'Clustering NMI:  mean(S)={:.4f}  concat(P)={:.4f}  concat(P,S)={:.4f}'.format(
            r['mean_S_nmi'], r['concat_P_nmi'], r['concat_PS_nmi']
        )
    )
    print(
        '  gap mean(S)−concat(P,S) NMI = {:.4f}  (large positive ⇒ P hurts joint clustering)'.format(
            r['gap_meanS_minus_concatPS_nmi']
        )
    )
    print(
        '  per-view S NMI: {:.3f}/{:.3f}/{:.3f}   P NMI: {:.3f}/{:.3f}/{:.3f}'.format(
            r['S1_nmi'], r['S2_nmi'], r['S3_nmi'], r['P1_nmi'], r['P2_nmi'], r['P3_nmi']
        )
    )
    print(
        'Subtype probes: S_other={:.4f}  P_other={:.4f}  Δ={:.4f}'.format(
            r['subtype_from_S_other_acc'],
            r['subtype_from_P_other_acc'],
            r['subtype_S_minus_P_acc'],
        )
    )
    print(
        'X probes R²: S_other={:.4f}  P_other={:.4f}  Δ(S−P)={:.4f}'.format(
            r['probe_XA_from_S_other_r2'],
            r['probe_XA_from_P_other_r2'],
            r['probe_XA_S_minus_P_r2'],
        )
    )
    print(
        'Alignment: cosine={:.4f}  cca={:.4f}  cka={:.4f}'.format(
            r['align_cosine_mean'], r['align_cca_mean'], r['align_cka_mean']
        )
    )
    print(
        'Latent std (mean over dims): '
        'S1={:.3f} S2={:.3f} S3={:.3f} | P1={:.3f} P2={:.3f} P3={:.3f}'.format(
            r['S1_dim_std_mean'],
            r['S2_dim_std_mean'],
            r['S3_dim_std_mean'],
            r['P1_dim_std_mean'],
            r['P2_dim_std_mean'],
            r['P3_dim_std_mean'],
        )
    )
    for k, v in sorted(r.items()):
        if k.startswith('decoder_'):
            print('  {}: {:.4f}'.format(k, v))


def main(args):
    print('diagnose_shared_private.py file:', os.path.abspath(__file__))
    print('RESULTS_ROOT:', RESULTS_ROOT)
    jobs = []
    if args.config:
        jobs.append((args.method, args.disease, args.config))
    else:
        # Default diagnostic panel from recent sweeps
        jobs = [
            ('mocss', args.disease, '0.0004_0.0007'),
            ('omidient', args.disease, '0.0003_0.0005_4'),
            ('method1', args.disease, '0.0004_0.0007_a0.7_b0.7_g0.7_l1.0'),
            ('method1', args.disease, '0.0004_0.0007_a1.0_b1.0_g0.7_l1.0'),
            ('method1_vicreg', args.disease, '0.0004_0.0007_a1.0_b0.5_v0.1_l1.0'),
            ('method1_vicreg', args.disease, '0.0004_0.0007_a1.0_b1.5_v0.1_l1.0'),
            ('method2', args.disease, '0.0004_0.0007_r1.5_c0.5_v0.1'),
            ('method2', args.disease, '0.0004_0.0007_r0.5_c0.5_v0.1'),
        ]

    all_rows = []
    for method, disease, config in jobs:
        try:
            report, config_dir = diagnose(method, disease, config)
        except FileNotFoundError as e:
            print('SKIP missing:', e)
            continue
        except Exception as e:
            print('SKIP error for {} / {}: {}: {}'.format(method, config, type(e).__name__, e))
            continue
        print_report(report)
        out_path = os.path.join(config_dir, 'diagnostics_shared_private.json')
        with open(out_path, 'w') as f:
            json.dump(report, f, indent=2)
        print('Wrote', out_path)
        all_rows.append(report)

    summary_dir = os.path.join(RESULTS_ROOT, 'diagnostics')
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, 'shared_private_panel_{}.json'.format(args.disease))
    with open(summary_path, 'w') as f:
        json.dump(all_rows, f, indent=2)
    print('\nRESULTS_ROOT:', RESULTS_ROOT)
    print('Panel summary:', summary_path)
    if not all_rows:
        print(
            'WARNING: no configs found. Check that trained folders exist under RESULTS_ROOT, e.g.\n'
            '  {}/models_brca_ae/0.0004_0.0007'.format(RESULTS_ROOT)
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Diagnose shared vs private embeddings')
    parser.add_argument(
        '--method',
        type=str,
        default='',
        choices=[
            '',
            'method1',
            'method1_vicreg',
            'method2',
            'method2_omidient',
            'mocss',
            'omidient',
        ],
    )
    parser.add_argument('--disease', type=str, default='brca')
    parser.add_argument('--config', type=str, default='', help='If empty, run default panel')
    args, unknown = parser.parse_known_args()
    if unknown:
        raise ValueError(f'Unkown args: {unknown}')
    if args.config and not args.method:
        raise ValueError('--method is required when --config is set')
    main(args)
