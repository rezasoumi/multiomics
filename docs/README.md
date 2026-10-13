# Shared Representation Training Results

Training and evaluation results for the methods described in [`shared-representation-method.md`](./shared-representation-method.md). Metrics are reported on the BRCA multi-omics setting used in this repository.

**Private branch (important):** Method 1, Method 1 VICReg ablation, and Method 2 all use a **MOCSS-style plain AE** for private codes `P_m`. They do **not** use OMIDIENT’s Dirichlet private VAE. Fair comparisons in this document are therefore vs **MOCSS (AE)** only. OMIDIENT will be mixed in later once shared methods are settled.

**Ranking:** configs are ranked by **core validation loss** (MOCSS-style; label-free): Method 1 / Method 1 VICReg → `L_std + L_shared + L_ctr`; Method 2 → `L_recon + L_ctr`. VICReg (`L_vic`) and adversarial terms are **not** used for ranking. Clustering/kNN/shared probes are reported on the selected config (KMeans `n_init=30`).

**Shared-quality metrics** (implemented in `code/shared_rep_metrics.py`; fill after re-evaluation):

1. Clustering + kNN on `mean(S)` (`s_nmi`, `s_knn_acc`, …)
2. Cross-view probes: subtype ← `S_other`; Ridge R² for `X_A ← (S_B,S_C)` vs `X_A ← (P_B,P_C)`
3. Alignment of shared codes: mean pairwise cosine / CCA / linear CKA

Config name prefixes like `0.0004_0.0007` are learning-rate / weight-decay settings; the remaining tokens encode loss weights (`a`/`b`/`g` = α/β/γ, `l` = λ_ctr, `v` = λ_vic, `r`/`c` = λ_rec / λ_ctr for Method 2).

> **Note:** Tables below still reflect an earlier eval pass (single KMeans init; no shared-only / probe / alignment columns). Re-run evaluation or sweeps to refresh NMI under `n_init=30` and to populate shared-quality metrics.

---

## Summary (best by NMI)

| Method | Best config | NMI | ARI | Acc | kNN |
| ------ | ----------- | ---: | ---: | ---: | ---: |
| MOCSS (AE) | — | 0.4550 | 0.3730 | 0.6343 | 0.73 |
| Method 1 (4 networks) | `0.0004_0.0007_a0.5_b0.5_g2.5_l1.0` | 0.4438 | 0.3466 | 0.5943 | 0.70 |
| Method 1 VICReg ablation | `0.0004_0.0007_a0.5_b1.5_v0.5_l1.0` | 0.4839 | 0.5083 | 0.6857 | 0.66 |
| Method 2 | `0.0004_0.0007_r0.5_c0.5_v0.1` | 0.4680 | 0.3820 | 0.6743 | 0.75 |

---

## Baseline: MOCSS (AE)

Plain multi-omics contrastive shared–specific autoencoder (same private AE family as the methods below).

| nmi | ari | f_score | acc | v_measure | knn_acc |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.4550 | 0.3730 | 0.5349 | 0.6343 | 0.4550 | 0.73 |

---

## Method 1 (4 networks)

Package: `code/method1_predictability/` (full predictability with own-network term). Private encoders: plain AE (MOCSS-style).

### Top 10 by NMI

| Rank | Config | NMI | ARI | Acc | kNN | val_loss |
| ---: | ------ | ---: | ---: | ---: | ---: | ---: |
| 1 | `0.0004_0.0007_a0.5_b0.5_g2.5_l1.0` | 0.4438 | 0.3466 | 0.5943 | 0.70 | −0.6724 |
| 2 | `0.0004_0.0007_a0.3_b0.5_g2.0_l1.0` | 0.4292 | 0.2878 | 0.5657 | 0.70 | −0.4779 |
| 3 | `0.0004_0.0007_a0.5_b0.5_g1.5_l1.0` | 0.4157 | 0.3023 | 0.6000 | 0.68 | −0.0215 |
| 4 | `0.0004_0.0007_a0.3_b0.5_g1.5_l1.0` | 0.4130 | 0.2853 | 0.5257 | 0.73 | −0.1398 |
| 5 | `0.0004_0.0007_a0.3_b0.5_g2.5_l1.0` | 0.4119 | 0.3402 | 0.6400 | 0.73 | −0.7940 |
| 6 | `0.0004_0.0007_a1.0_b1.0_g0.7_l1.0` | 0.4110 | 0.3323 | 0.6343 | 0.70 | 0.1715 |
| 7 | `0.0004_0.0007_a0.7_b0.5_g2.0_l1.0` | 0.4083 | 0.3225 | 0.6514 | 0.64 | −0.2322 |
| 8 | `0.0004_0.0007_a0.7_b0.5_g2.5_l1.0` | 0.4027 | 0.3015 | 0.6343 | 0.68 | −0.5600 |
| 9 | `0.0004_0.0007_a0.5_b0.5_g2.0_l1.0` | 0.3990 | 0.3461 | 0.6457 | 0.73 | −0.3449 |
| 10 | `0.0004_0.0007_a1.5_b1.0_g1.0_l1.0` | 0.3947 | 0.2940 | 0.5543 | 0.70 | 0.2315 |

**Best (NMI):** `0.0004_0.0007_a0.5_b0.5_g2.5_l1.0`

```json
{
  "rank_metric": "nmi",
  "config": "0.0004_0.0007_a0.5_b0.5_g2.5_l1.0",
  "alpha": 0.5,
  "beta": 0.5,
  "gamma": 2.5,
  "lambda_ctr": 1.0,
  "val_loss": -0.672432792186737,
  "nmi": 0.4437890490849838,
  "ari": 0.34655194079713575,
  "f_score": 0.5181688693772225,
  "acc": 0.5942857142857143,
  "knn_acc": 0.7045454545454546,
  "ch_index": 33.50923298570472
}
```

---

## Method 1 VICReg ablation (no own network)

Package: `code/method1_vicreg_ablation/`. Results directory: `results/models_brca_method1_vicreg/`. Private encoders: plain AE (MOCSS-style).

### Top 10 by NMI

| Rank | Config | NMI | ARI | Acc | kNN | constructive |
| ---: | ------ | ---: | ---: | ---: | ---: | ---: |
| 1 | `0.0004_0.0007_a0.5_b1.5_v0.5_l1.0` | 0.4839 | 0.5083 | 0.6857 | 0.66 | 0.7057 |
| 2 | `0.0004_0.0007_a0.5_b2.0_v0.5_l1.0` | 0.4672 | 0.3402 | 0.6686 | 0.64 | 0.7065 |
| 3 | `0.0004_0.0007_a0.3_b2.0_v0.1_l1.0` | 0.4672 | 0.3217 | 0.5829 | 0.70 | 0.5923 |
| 4 | `0.0004_0.0007_a0.7_b1.5_v0.1_l1.0` | 0.4657 | 0.3263 | 0.5714 | 0.70 | 0.7088 |
| 5 | `0.0004_0.0007_a1.0_b1.5_v0.1_l1.0` | 0.4636 | 0.4626 | 0.6629 | 0.64 | 0.7943 |
| 6 | `0.0004_0.0007_a0.5_b1.5_v1.0_l1.0` | 0.4623 | 0.3811 | 0.6857 | 0.68 | 0.7654 |
| 7 | `0.0004_0.0007_a1.5_b2.0_v0.1_l1.0` | 0.4595 | 0.4509 | 0.6800 | 0.73 | 0.9440 |
| 8 | `0.0004_0.0007_a0.7_b2.0_v0.1_l1.0` | 0.4540 | 0.4437 | 0.6571 | 0.68 | 0.7083 |
| 9 | `0.0004_0.0007_a0.5_b2.0_v0.1_l1.0` | 0.4521 | 0.4385 | 0.6743 | 0.70 | 0.6502 |
| 10 | `0.0004_0.0007_a0.7_b2.0_v1.0_l1.0` | 0.4442 | 0.3183 | 0.5771 | 0.68 | 0.8263 |

**Best (NMI):** `0.0004_0.0007_a0.5_b1.5_v0.5_l1.0`

```json
{
  "rank_metric": "nmi",
  "config": "0.0004_0.0007_a0.5_b1.5_v0.5_l1.0",
  "alpha": 0.5,
  "beta": 1.5,
  "lambda_vic": 0.5,
  "lambda_ctr": 1.0,
  "val_constructive": 0.7056930460035801,
  "nmi": 0.48387043844661665,
  "ari": 0.5083233906216597,
  "f_score": 0.6462308635553173,
  "acc": 0.6857142857142857,
  "knn_acc": 0.6590909090909091,
  "ch_index": 37.396597252809066
}
```

---

## Method 2 (joint recon + VICReg)

Package: `code/method2_joint_vicreg/`. Private encoders: plain AE (MOCSS-style).

### Top 10 by NMI

| Rank | Config | NMI | ARI | Acc | kNN | val |
| ---: | ------ | ---: | ---: | ---: | ---: | ---: |
| 1 | `0.0004_0.0007_r0.5_c0.5_v0.1` | 0.4680 | 0.3820 | 0.6743 | 0.75 | 0.2479 |
| 2 | `0.0004_0.0007_r1.5_c0.7_v0.5` | 0.4623 | 0.3068 | 0.5829 | 0.84 | 0.5617 |
| 3 | `0.0004_0.0007_r0.7_c2.0_v0.1` | 0.4619 | 0.4464 | 0.6743 | 0.70 | 0.6774 |
| 4 | `0.0004_0.0007_r0.5_c1.5_v1.0` | 0.4605 | 0.3580 | 0.6629 | 0.70 | 0.5652 |
| 5 | `0.0004_0.0007_r1.5_c0.5_v0.5` | 0.4604 | 0.3723 | 0.6629 | 0.77 | 0.5120 |
| 6 | `0.0004_0.0007_r0.7_c0.7_v0.1` | 0.4551 | 0.3172 | 0.5771 | 0.73 | 0.3444 |
| 7 | `0.0004_0.0007_r1.5_c1.0_v0.5` | 0.4528 | 0.3136 | 0.6171 | 0.80 | 0.6380 |
| 8 | `0.0004_0.0007_r1.0_c0.7_v0.1` | 0.4481 | 0.3770 | 0.6514 | 0.75 | 0.4096 |
| 9 | `0.0004_0.0007_r0.3_c1.5_v0.1` | 0.4478 | 0.3748 | 0.6857 | 0.75 | 0.4545 |
| 10 | `0.0004_0.0007_r0.5_c2.0_v0.1` | 0.4463 | 0.3605 | 0.6743 | 0.73 | 0.6260 |

**Best (NMI):** `0.0004_0.0007_r0.5_c0.5_v0.1`

```json
{
  "rank_metric": "nmi",
  "config": "0.0004_0.0007_r0.5_c0.5_v0.1",
  "lambda_rec": 0.5,
  "lambda_ctr": 0.5,
  "lambda_vic": 0.1,
  "val_loss": 0.24788183271884917,
  "nmi": 0.4679629033414979,
  "ari": 0.3819508426638335,
  "f_score": 0.5407825816252462,
  "acc": 0.6742857142857143,
  "knn_acc": 0.75,
  "ch_index": 42.750245230677
}
```

---

## Shared-quality metrics (to fill after re-eval)

After re-running `evaluate.py` / sweeps, report for each method’s best-by-NMI config:

| Method | s_nmi | s_knn | subtype←S_other | X←S R² | X←P R² | Δ(S−P) | cosine | CCA | CKA |
| ------ | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MOCSS (AE) | — | — | — | — | — | — | — | — | — |
| Method 1 | — | — | — | — | — | — | — | — | — |
| Method 1 VICReg | — | — | — | — | — | — | — | — | — |
| Method 2 | — | — | — | — | — | — | — | — | — |
