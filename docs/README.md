# Shared Representation Training Results

Training and evaluation results for the methods described in [`shared-representation-method.md`](./shared-representation-method.md). Metrics are reported on the BRCA multi-omics setting used in this repository.

**Private branch (important):** Method 1, Method 1 VICReg ablation, and Method 2 (AE private) use a **MOCSS-style plain AE** for `P_m`. Fair AE-private comparisons are vs **MOCSS (AE)**. **OMIDIENT** (Dirichlet private + MOCSS shared) is benchmarked with the same `shared_rep_metrics` suite (`code/prod_gamma_dirvae/evaluate.py`, `run_omidient_eval.slurm`). **Method 2 + OMIDIENT private** jointly trains Method-2 shared/joint-decode with OMIDIENT product-Dirichlet private (`code/method2_omidient_private/`, `run_method2_omidient.slurm`); results TBD.

**Ranking:** configs are ranked by **core validation loss** (MOCSS-style; label-free). For the tables below (pre–private-sufficiency sweeps): Method 1 / Method 1 VICReg → `L_std + L_shared + L_ctr`; Method 2 → `L_recon + L_ctr`. VICReg (`L_vic`) and adversarial terms are **not** used for ranking. Clustering/kNN/shared probes are reported on the selected config (KMeans `n_init=30`).

**`b_m(P_m) → X_m` (private sufficiency):** Method 1 and Method 1 VICReg numbers in this README are **without** the private-sufficiency head `b_m`. Results **with** `b_m(P_m) → X_m` (and `λ_priv` in ranking) will be added after the new sweeps finish.

**Shared-quality metrics** (implemented in `code/shared_rep_metrics.py`):

1. Clustering + kNN on `mean(S)` (`s_nmi`, `s_knn_acc`, …)
2. Cross-view probes: subtype ← `S_other`; Ridge R² for `X_A ← (S_B,S_C)` vs `X_A ← (P_B,P_C)`
3. Alignment of shared codes: mean pairwise cosine / CCA / linear CKA

Config name prefixes like `0.0004_0.0007` are learning-rate / weight-decay settings; the remaining tokens encode loss weights (`a`/`b`/`g` = α/β/γ, `p` = λ_priv for private-sufficiency `b(P)→X` when present, `l` = λ_ctr, `v` = λ_vic, `r`/`c` = λ_rec / λ_ctr for Method 2).

---

## Compact comparison

Shared / private diagnostic panel (BRCA). `gap S−PS` = mean(S) NMI − concat(P,S) NMI (large positive ⇒ private hurts integration). `subtype S−P` = subtype←S_other − subtype←P_other. `Δ X R²` = R²(X←S_other) − R²(X←P_other).

| Config | mean(S) NMI | concat(P) NMI | concat(P,S) NMI | gap S−PS | subtype S−P | Δ X R² | CKA |
| ------ | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MOCSS | 0.422 | 0.448 | 0.450 | −0.03 | +0.02 | +0.06 | 0.75 |
| M1 loss-best (a0.7 b0.7 g0.7) | 0.442 | 0.109 | 0.163 | +0.28 | +0.14 | +0.12 | 0.40 |
| M1 #3 (a1.0 b1.0 g0.7) | 0.451 | 0.188 | 0.395 | +0.06 | +0.23 | −0.06 | 0.70 |
| VICReg loss-best | 0.452 | 0.243 | 0.336 | +0.12 | +0.21 | −0.34 | 0.77 |
| VICReg #10 | 0.421 | 0.236 | 0.421 | ~0 | +0.20 | −0.21 | 0.73 |
| M2 loss-best | 0.394 | 0.408 | 0.420 | −0.03 | +0.02 | +0.10 | 0.79 |
| M2 high-NMI (r0.5 c0.5 v0.1) | 0.412 | 0.382 | 0.456 | −0.04 | ~0 | +0.09 | 0.82 |

---

## Summary (best by eval_loss)

| Method | Best config | concat NMI | s_nmi | ARI | Acc | kNN | eval_loss |
| ------ | ----------- | ---: | ---: | ---: | ---: | ---: | ---: |
| MOCSS (AE) | `0.0004_0.0007` | 0.4496 | 0.4219 | 0.3034 | 0.5429 | 0.73 | — |
| Method 1 (pre-`b_m`) | `0.0004_0.0007_a0.7_b0.7_g0.7_l1.0` | 0.1633 | 0.4424 | 0.0735 | 0.3886 | 0.68 | 0.8118 |
| Method 1 VICReg (pre-`b_m`) | `0.0004_0.0007_a1.0_b0.5_v0.1_l1.0` | 0.3357 | 0.4517 | 0.2031 | 0.4800 | 0.73 | 0.7728 |
| Method 2 | `0.0004_0.0007_r1.5_c0.5_v0.1` | 0.4199 | 0.3944 | 0.2806 | 0.5200 | 0.70 | 0.4693 |

---

## Baseline: MOCSS (AE)

Plain multi-omics contrastive shared–specific autoencoder (same private AE family as the methods below). Config: `0.0004_0.0007`.

| | nmi | ari | f_score | acc | knn |
| --- | ---: | ---: | ---: | ---: | ---: |
| concat(P,S) | 0.4496 | 0.3034 | 0.4783 | 0.5429 | 0.73 |
| mean(S) | 0.4219 | 0.3362 | — | 0.5943 | 0.73 |

Probes: subtype←S_other = 0.7424 | X←S R² = −0.0742 | X←P R² = −0.1314 | Δ(S−P) = 0.0573  
Alignment: cosine = −0.0350 | CCA = 0.9730 | CKA = 0.7531

---

## Method 1 (4 networks) — without `b_m`

Package: `code/method1_predictability/` (full predictability with own-network term). Private encoders: plain AE (MOCSS-style).

**Note:** These results are **before** adding private-sufficiency `b_m(P_m) → X_m`. Post-`b_m` sweeps will be reported later.

### Top 10 by eval_loss

| Rank | Config | NMI | s_nmi | ARI | Acc | kNN | eval_loss |
| ---: | ------ | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `0.0004_0.0007_a0.7_b0.7_g0.7_l1.0` | 0.1633 | 0.4424 | 0.0735 | 0.3886 | 0.68 | 0.8118 |
| 2 | `0.0004_0.0007_a1.5_b1.5_g0.7_l1.0` | 0.3181 | 0.4371 | 0.2089 | 0.4971 | 0.68 | 0.8141 |
| 3 | `0.0004_0.0007_a1.0_b1.0_g0.7_l1.0` | 0.3951 | 0.4507 | 0.3425 | 0.6229 | 0.70 | 0.8170 |
| 4 | `0.0004_0.0007_a1.5_b0.7_g0.7_l1.0` | 0.3009 | 0.3844 | 0.1761 | 0.4686 | 0.66 | 0.8196 |
| 5 | `0.0004_0.0007_a1.0_b0.5_g0.7_l1.0` | 0.2393 | 0.3987 | 0.1195 | 0.4629 | 0.66 | 0.8227 |
| 6 | `0.0004_0.0007_a1.0_b1.5_g0.7_l1.0` | 0.3266 | 0.4351 | 0.1885 | 0.4800 | 0.73 | 0.8267 |
| 7 | `0.0004_0.0007_a1.5_b2.0_g0.7_l1.0` | 0.3319 | 0.4214 | 0.1814 | 0.5029 | 0.68 | 0.8300 |
| 8 | `0.0004_0.0007_a1.5_b1.0_g0.7_l1.0` | 0.3333 | 0.3867 | 0.1831 | 0.4571 | 0.66 | 0.8312 |
| 9 | `0.0004_0.0007_a1.5_b0.5_g0.7_l1.0` | 0.2775 | 0.3773 | 0.1680 | 0.4629 | 0.59 | 0.8350 |
| 10 | `0.0004_0.0007_a1.5_b0.7_g1.0_l1.0` | 0.2759 | 0.4100 | 0.1601 | 0.4857 | 0.64 | 0.8386 |

**Best (eval_loss):** `0.0004_0.0007_a0.7_b0.7_g0.7_l1.0`

| | nmi | ari | f_score | acc | knn |
| --- | ---: | ---: | ---: | ---: | ---: |
| concat(P,S) | 0.1633 | 0.0735 | 0.2992 | 0.3886 | 0.68 |
| mean(S) | 0.4424 | 0.3296 | — | 0.5829 | 0.66 |

Probes: subtype←S_other = 0.7652 | X←S R² = 0.0806 | X←P R² = −0.0360 | Δ(S−P) = 0.1166  
Alignment: cosine = −0.0778 | CCA = 0.9777 | CKA = 0.3997

---

## Method 1 VICReg ablation (no own network) — without `b_m`

Package: `code/method1_vicreg_ablation/`. Results directory: `results/models_brca_method1_vicreg/`. Private encoders: plain AE (MOCSS-style).

**Note:** These results are **before** adding private-sufficiency `b_m(P_m) → X_m`. Post-`b_m` sweeps will be reported later.

### Top 10 by eval_loss

| Rank | Config | NMI | s_nmi | ARI | Acc | kNN | eval_loss |
| ---: | ------ | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `0.0004_0.0007_a1.0_b0.5_v0.1_l1.0` | 0.3357 | 0.4517 | 0.2031 | 0.4800 | 0.73 | 0.7728 |
| 2 | `0.0004_0.0007_a1.5_b0.7_v0.1_l1.0` | 0.3634 | 0.4111 | 0.2031 | 0.4800 | 0.68 | 0.7754 |
| 3 | `0.0004_0.0007_a0.5_b0.5_v0.1_l1.0` | 0.3112 | 0.4437 | 0.1748 | 0.4857 | 0.66 | 0.7766 |
| 4 | `0.0004_0.0007_a1.5_b0.5_v0.1_l1.0` | 0.3130 | 0.4136 | 0.1760 | 0.4914 | 0.70 | 0.7767 |
| 5 | `0.0004_0.0007_a0.5_b0.7_v0.1_l1.0` | 0.3604 | 0.4363 | 0.2185 | 0.5143 | 0.77 | 0.7772 |
| 6 | `0.0004_0.0007_a0.7_b0.5_v0.1_l1.0` | 0.3185 | 0.4283 | 0.1962 | 0.5143 | 0.64 | 0.7778 |
| 7 | `0.0004_0.0007_a1.0_b1.0_v0.1_l1.0` | 0.3165 | 0.4128 | 0.1804 | 0.4800 | 0.70 | 0.7783 |
| 8 | `0.0004_0.0007_a0.7_b0.7_v0.1_l1.0` | 0.3238 | 0.4181 | 0.1843 | 0.4857 | 0.75 | 0.7790 |
| 9 | `0.0004_0.0007_a1.0_b0.7_v0.1_l1.0` | 0.2984 | 0.4398 | 0.1913 | 0.4914 | 0.73 | 0.7794 |
| 10 | `0.0004_0.0007_a1.0_b1.5_v0.1_l1.0` | 0.4206 | 0.4212 | 0.2790 | 0.5314 | 0.64 | 0.7802 |

**Best (eval_loss):** `0.0004_0.0007_a1.0_b0.5_v0.1_l1.0`

| | nmi | ari | f_score | acc | knn |
| --- | ---: | ---: | ---: | ---: | ---: |
| concat(P,S) | 0.3357 | 0.2031 | 0.4051 | 0.4800 | 0.73 |
| mean(S) | 0.4517 | 0.3090 | — | 0.5314 | 0.66 |

Probes: subtype←S_other = 0.7576 | X←S R² = −0.2093 | X←P R² = 0.1270 | Δ(S−P) = −0.3363  
Alignment: cosine = −0.0764 | CCA = 0.9807 | CKA = 0.7689

---

## Method 2 (joint recon + VICReg)

Package: `code/method2_joint_vicreg/`. Private encoders: plain AE (MOCSS-style).

### Top 10 by eval_loss

| Rank | Config | NMI | s_nmi | ARI | Acc | kNN | eval_loss |
| ---: | ------ | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `0.0004_0.0007_r1.5_c0.5_v0.1` | 0.4199 | 0.3944 | 0.2806 | 0.5200 | 0.70 | 0.4693 |
| 2 | `0.0004_0.0007_r1.5_c0.7_v0.1` | 0.4115 | 0.4304 | 0.2650 | 0.5143 | 0.68 | 0.4725 |
| 3 | `0.0004_0.0007_r1.0_c0.7_v0.1` | 0.4120 | 0.3956 | 0.2687 | 0.5200 | 0.75 | 0.4756 |
| 4 | `0.0004_0.0007_r1.5_c1.0_v0.1` | 0.4208 | 0.3809 | 0.2814 | 0.5486 | 0.75 | 0.4761 |
| 5 | `0.0004_0.0007_r0.7_c0.5_v0.1` | 0.4348 | 0.3960 | 0.2962 | 0.5543 | 0.70 | 0.4772 |
| 6 | `0.0004_0.0007_r1.0_c0.5_v0.1` | 0.4427 | 0.4177 | 0.3011 | 0.5371 | 0.68 | 0.4774 |
| 7 | `0.0004_0.0007_r1.0_c1.0_v0.1` | 0.4478 | 0.4485 | 0.3582 | 0.6400 | 0.75 | 0.4786 |
| 8 | `0.0004_0.0007_r0.5_c0.5_v0.1` | 0.4562 | 0.4117 | 0.3489 | 0.6400 | 0.75 | 0.4790 |
| 9 | `0.0004_0.0007_r0.7_c0.7_v0.1` | 0.4551 | 0.4116 | 0.3172 | 0.5771 | 0.73 | 0.4793 |
| 10 | `0.0004_0.0007_r0.5_c2.0_v0.1` | 0.4551 | 0.4456 | 0.3052 | 0.5714 | 0.73 | 0.4794 |

**Best (eval_loss):** `0.0004_0.0007_r1.5_c0.5_v0.1`

| | nmi | ari | f_score | acc | knn |
| --- | ---: | ---: | ---: | ---: | ---: |
| concat(P,S) | 0.4199 | 0.2806 | 0.4611 | 0.5200 | 0.70 |
| mean(S) | 0.3944 | 0.2644 | — | 0.5314 | 0.73 |

Probes: subtype←S_other = 0.7121 | X←S R² = −0.0126 | X←P R² = −0.1082 | Δ(S−P) = 0.0956  
Alignment: cosine = 0.0794 | CCA = 0.9796 | CKA = 0.7914

---

## OMIDIENT baseline (to fill)

Same metrics as MOCSS AE. Evaluate an existing checkpoint:

```bash
sbatch run_omidient_eval.slurm
# or: cd code/prod_gamma_dirvae && python evaluate.py --disease brca --config 0.0003_0.0005_4 --plot
```

Results write to `results/models_brca_ProdGammaDirVae/<config>/metrics_shared_rep.json`.

---

## Method 2 + OMIDIENT private (joint train; results TBD)

Package: `code/method2_omidient_private/`. Shared = Method-2 AE + contrastive; private = product-Dirichlet (K=4); recon = joint `D(concat(P,S))` + VICReg + KL. Sweep via `run_method2_omidient.slurm`. **Not comparable to Method 1/2 AE-private tables above** until reported here; compare to **OMIDIENT** and to Method 2 AE as ablations.

---

## Shared-quality metrics

See **Compact comparison** above (diagnostic panel with mean(S) / concat(P) / concat(P,S), probes, CKA). Per-method probe/alignment numbers for the eval_loss winners are also listed under each method section.
