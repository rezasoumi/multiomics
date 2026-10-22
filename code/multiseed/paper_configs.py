"""Paper-priority configs to retrain across seeds (README compact / summary / family).

Hyperparameters are fixed from the seed-21 sweeps; only the training seed changes.
Data splits stay at random_state=1 (same as the original sweeps).
"""

from __future__ import annotations

# Default seeds for mean ± std (include original sweep seed 21).
DEFAULT_SEEDS = [21, 22, 23, 24, 25]

# Metrics written into multiseed summary / used for README ± columns.
COMPACT_KEYS = [
    's_nmi',
    'p_nmi',
    'nmi',
    'gap_S_minus_PS',
    'probe_XA_S_minus_P_r2',
    'probe_Xm_from_S_own_r2',
    'probe_Xm_from_P_own_r2',
    'probe_Xm_from_PS_own_r2',
    'align_cka_mean',
]

SUMMARY_KEYS = [
    'nmi',
    's_nmi',
    'ari',
    'acc',
    'knn_acc',
    'eval_loss',
]

DETAIL_KEYS = [
    'nmi',
    'ari',
    'f_score',
    'acc',
    'knn_acc',
    'p_nmi',
    'p_ari',
    'p_acc',
    'p_knn_acc',
    's_nmi',
    's_ari',
    's_acc',
    's_knn_acc',
    'probe_subtype_from_S_other_acc',
    'probe_XA_from_S_other_r2',
    'probe_XA_from_P_other_r2',
    'probe_XA_S_minus_P_r2',
    'probe_Xm_from_S_own_r2',
    'probe_Xm_from_P_own_r2',
    'probe_Xm_from_PS_own_r2',
    'align_cosine_mean',
    'align_cca_mean',
    'align_cka_mean',
    'eval_loss',
    'gap_S_minus_PS',
    # Method 3 / 4 extras (ignored if absent)
    'nmi_spectral_U',
    'agree_symkl_mean',
    'entropy_q_bar',
]

# method: trainer dispatch key
# base_config: original sweep folder name (no seed suffix)
# results_subdir: under results/
PAPER_RUNS = [
    {
        'id': 'mocss',
        'method': 'mocss',
        'label': 'MOCSS (AE)',
        'base_config': '0.0004_0.0007',
        'results_subdir': 'models_brca_ae',
        'tables': ['compact', 'summary', 'detail'],
        'hparams': {'lr': 0.0004, 'weight_decay': 0.0007},
    },
    {
        'id': 'method1',
        'method': 'method1',
        'label': 'Method 1 (+b_m)',
        'base_config': '0.0004_0.0007_a1.5_b0.7_g0.7_p0.5_l1.0',
        'results_subdir': 'models_brca_method1',
        'tables': ['compact', 'summary', 'detail'],
        'hparams': {
            'lr': 0.0004,
            'weight_decay': 0.0007,
            'alpha': 1.5,
            'beta': 0.7,
            'gamma': 0.7,
            'lambda_priv': 0.5,
            'lambda_ctr': 1.0,
        },
    },
    {
        'id': 'method1_vicreg',
        'method': 'method1_vicreg',
        'label': 'Method 1 VICReg (+b_m)',
        'base_config': '0.0004_0.0007_a0.7_b0.5_v0.1_p1.0_l1.0',
        'results_subdir': 'models_brca_method1_vicreg',
        'tables': ['compact', 'summary', 'detail'],
        'hparams': {
            'lr': 0.0004,
            'weight_decay': 0.0007,
            'alpha': 0.7,
            'beta': 0.5,
            'lambda_vic': 0.1,
            'lambda_priv': 1.0,
            'lambda_ctr': 1.0,
        },
    },
    {
        'id': 'method2',
        'method': 'method2',
        'label': 'Method 2',
        'base_config': '0.0004_0.0007_r1.5_c0.5_v0.1',
        'results_subdir': 'models_brca_method2',
        'tables': ['compact', 'summary', 'family', 'detail'],
        'hparams': {
            'lr': 0.0004,
            'weight_decay': 0.0007,
            'lambda_rec': 1.5,
            'lambda_ctr': 0.5,
            'lambda_vic': 0.1,
        },
    },
    {
        'id': 'method2_high_nmi',
        'method': 'method2',
        'label': 'Method 2 high-NMI',
        'base_config': '0.0004_0.0007_r0.5_c0.5_v0.1',
        'results_subdir': 'models_brca_method2',
        'tables': ['compact'],
        'hparams': {
            'lr': 0.0004,
            'weight_decay': 0.0007,
            'lambda_rec': 0.5,
            'lambda_ctr': 0.5,
            'lambda_vic': 0.1,
        },
    },
    {
        'id': 'method2_omidient',
        'method': 'method2_omidient',
        'label': 'Method 2 + OMIDIENT private',
        'base_config': '0.0004_0.0007_r1.5_c0.5_v0.1_k4',
        'results_subdir': 'models_brca_method2_omidient',
        'tables': ['family', 'detail'],
        'hparams': {
            'lr': 0.0004,
            'weight_decay': 0.0007,
            'lambda_rec': 1.5,
            'lambda_ctr': 0.5,
            'lambda_vic': 0.1,
            'k_groups': 4,
        },
    },
    {
        'id': 'omidient_reported',
        'method': 'omidient',
        'label': 'OMIDIENT (reported / no L_ortho)',
        'base_config': '0.0003_0.0005_4',
        'results_subdir': 'models_brca_ProdGammaDirVae',
        'tables': ['family', 'detail'],
        'hparams': {
            'lr': 0.0003,
            'weight_decay': 0.0005,
            'k_groups': 4,
            'use_orthogonal': False,
            'lambda_ctr': 1.0,
            'lambda_recon': 0.7,
            'lambda_kl': 1.0,
            'lambda_ortho': 1.0,
        },
    },
    {
        'id': 'omidient_ortho',
        'method': 'omidient',
        'label': 'OMIDIENT (with L_ortho, λ-best)',
        'base_config': '0.0003_0.0005_4_ortho_c2_r1.5_k1_o0.5',
        'results_subdir': 'models_brca_ProdGammaDirVae',
        'tables': ['family', 'detail'],
        'hparams': {
            'lr': 0.0003,
            'weight_decay': 0.0005,
            'k_groups': 4,
            'use_orthogonal': True,
            'lambda_ctr': 2.0,
            'lambda_recon': 1.5,
            'lambda_kl': 1.0,
            'lambda_ortho': 0.5,
        },
    },
    {
        'id': 'method3',
        'method': 'method3',
        'label': 'Method 3 (spectral)',
        'base_config': '0.0004_0.0007_s2.0_r1.0_c0.0_v0.1_vw0.5_lk0.5',
        'results_subdir': 'models_brca_method3',
        'tables': ['compact', 'summary', 'detail'],
        'hparams': {
            'lr': 0.0004,
            'weight_decay': 0.0007,
            'lambda_spec': 2.0,
            'lambda_res': 1.0,
            'lambda_ctr': 0.0,
            'lambda_vic': 0.1,
            'lambda_view': 0.5,
            'lambda_leak': 0.5,
        },
    },
    {
        'id': 'method3_high_nmi',
        'method': 'method3',
        'label': 'Method 3 high-NMI',
        'base_config': '0.0004_0.0007_s1.0_r0.5_c0.0_v0.1_vw0.5_lk0.5',
        'results_subdir': 'models_brca_method3',
        'tables': ['compact'],
        'hparams': {
            'lr': 0.0004,
            'weight_decay': 0.0007,
            'lambda_spec': 1.0,
            'lambda_res': 0.5,
            'lambda_ctr': 0.0,
            'lambda_vic': 0.1,
            'lambda_view': 0.5,
            'lambda_leak': 0.5,
        },
    },
    {
        'id': 'method4',
        'method': 'method4',
        'label': 'Method 4 (invariant)',
        'base_config': '0.0004_0.0007_d0.5_a2.0_r1.0_vw0.5_lk0.5_c0.5_v0.1',
        'results_subdir': 'models_brca_method4',
        'tables': ['compact', 'summary', 'detail'],
        'hparams': {
            'lr': 0.0004,
            'weight_decay': 0.0007,
            'lambda_dec': 0.5,
            'lambda_agree': 2.0,
            'lambda_res': 1.0,
            'lambda_view': 0.5,
            'lambda_leak': 0.5,
            'lambda_ctr': 0.5,
            'lambda_vic': 0.1,
        },
    },
]


def get_run(run_id: str) -> dict:
    for run in PAPER_RUNS:
        if run['id'] == run_id:
            return run
    raise KeyError('Unknown paper run id: {}'.format(run_id))


def list_run_ids():
    return [r['id'] for r in PAPER_RUNS]
