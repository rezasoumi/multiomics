# Shared Representation Training Results

Training and evaluation results for the methods described in [`shared-representation-method.md`](./shared-representation-method.md). Metrics are reported on the BRCA multi-omics setting used in this repository.

**Multi-seed (headline tables):** MOCSS, Method 1, Method 1 VICReg, Method 2 (incl. high-NMI), Method 2 + OMIDIENT private, and both OMIDIENT rows report **mean ± sample std** over training seeds `{21, 22, 23, 24, 25}` (n=5). Data splits stay fixed (`random_state=1`). Source: `results/multiseed/summary.json` (`run_multiseed.slurm`). Method 3 / Method 4, Top-10 leaderboards, and the without-`b_m` appendix remain **single-seed** (seed 21) unless noted.

**Private branch (important):** Method 1, Method 1 VICReg ablation, and Method 2 (AE private) use a **MOCSS-style plain AE** for `P_m`. Fair AE-private comparisons are vs **MOCSS (AE)**. **OMIDIENT** uses product-Dirichlet private + MOCSS-style shared. In this repo the published checkpoint `0.0003_0.0005_4` was trained **without** `L_ortho` in `loss_total` (implementation gap); that row is labeled **OMIDIENT (reported / no \(L_\mathrm{ortho}\))**. **OMIDIENT (with \(L_\mathrm{ortho}\))** is the intended objective (λ-sweep below). **Method 2 + OMIDIENT private** mixes Method-2 shared with Dirichlet private — see the separate family table.

**Ranking:** configs are ranked by **core validation loss** (MOCSS-style; label-free). Method 1 / Method 1 VICReg → `L_std + L_shared + L_priv + L_ctr` (with `b_m`; `L_vic` excluded for VICReg). Method 2 / Method 2+OMIDIENT → `L_recon + L_ctr` (`L_vic` / `L_kl` excluded). Method 3 → `L_spec + L_res + L_ctr`. Method 4 → `L_dec + L_agree + L_res + L_ctr`. OMIDIENT with \(L_\mathrm{ortho}\) λ-sweep → `L_ctr + L_recon + KL` (λ and `L_ortho` excluded from ranking). Clustering/kNN/shared probes are reported on the selected config (KMeans `n_init=30`).

**`b_m(P_m) → X_m` (private sufficiency):** Method 1 and Method 1 VICReg numbers below are **with** `b_m` and `λ_priv` (`p` token in config names).

**Note on `p_nmi`:** high `concat(P)` NMI does **not** by itself prove \(S\)/\(P\) overlap. Private codes can carry **complementary** cluster-useful biology (assay-specific subtype correlates) while still being disjoint from \(S\). Prefer probes (Δ X R², subtype←S_other, leak of shared soft-labels from \(P\)) over “`p_nmi` must be near 0.”

**Shared-quality metrics** (implemented in `code/shared_rep_metrics.py`):

1. Clustering + kNN on `mean(S)` (`s_nmi`, `s_knn_acc`, …)
2. Cross-view probes: subtype ← `S_other`; Ridge R² for `X_A ← (S_B,S_C)` vs `X_A ← (P_B,P_C)`
3. Own-view reconstructiveness: Ridge R² for `X_m ← S_m`, `X_m ← P_m`, `X_m ← [P_m;S_m]` (mean over modalities)
4. Alignment of shared codes: mean pairwise cosine / CCA / linear CKA

Config name prefixes like `0.0004_0.0007` are learning-rate / weight-decay settings; the remaining tokens encode loss weights (`a`/`b`/`g` = α/β/γ, `p` = λ_priv, `l` = λ_ctr, `v` = λ_vic, `r`/`c` = λ_rec / λ_ctr for Method 2; Method 3: `s`/`r`/`c`/`v`/`vw`/`lk` = λ_spec / λ_res / λ_ctr / λ_vic / λ_view / λ_leak; Method 4: `d`/`a`/`r`/`vw`/`lk`/`c`/`v` = λ_dec / λ_agree / λ_res / λ_view / λ_leak / λ_ctr / λ_vic; OMIDIENT ortho: `c`/`r`/`k`/`o` = λ_ctr / λ_recon / λ_kl / λ_ortho).

---

## Figures (visual color code)

**Color language (same in every plot):** blue = shared \(S\); orange = private \(P\); purple/gray = concat \([P;S]\); green↔red = signed gaps / \(\Delta\) around 0.

| # | File | What it shows | Needs checkpoints? |
| - | ---- | ------------- | ------------------ |
| 1 | [`figures/01_nmi_bars_ae.png`](./figures/01_nmi_bars_ae.png), [`01_nmi_bars_omidient.png`](./figures/01_nmi_bars_omidient.png) | mean(S) / concat(P) / concat(P,S) NMI bars | No (README numbers) |
| 2 | [`figures/02_scatter_separation_vs_delta_xr2.png`](./figures/02_scatter_separation_vs_delta_xr2.png) | subtype separation vs \(\Delta\) X R² | No |
| 3 | [`figures/03_metric_heatmap_ae.png`](./figures/03_metric_heatmap_ae.png), [`03_metric_heatmap_omidient.png`](./figures/03_metric_heatmap_omidient.png) | color-coded compact metrics | No |
| 4 | [`figures/04_method_schematic.png`](./figures/04_method_schematic.png) | process schematic (\(S\)/\(P\)/losses) | No |
| 5 | [`figures/05_embedding_umap.png`](./figures/05_embedding_umap.png) (or `…_tsne`) | UMAP/t-SNE of mean(S) vs concat(P), colored by subtype | Yes (existing models) |
| 6 | [`figures/06_sweep_*.png`](./figures/) | hyperparameter heatmaps from leaderboard CSVs | Yes (existing CSVs) |

**Regenerate on Triton** (no retraining):

```bash
cd /scratch/work/soumir1/githubmultiomics/multiomics
sbatch run_viz.slurm
# outputs → docs/figures/*.png and *.pdf
```

**Or locally / interactively** (set `PYTHONPATH` to the parent of `multiomics`):

```bash
export MULTIOMICS_RESULTS=/path/to/multiomics/results   # Triton results root
python -m multiomics.code.viz.run_all --mode readme     # figs 1–4 only
python -m multiomics.code.viz.run_all --mode all        # all six
```

Code: `code/viz/` (`run_all.py`, `plots.py`). Fig 5 reuses `code/diagnose_shared_private.py::load_latents`.

### Fig 1 — NMI bars (S / P / [P;S])

![NMI bars AE](./figures/01_nmi_bars_ae.png)

![NMI bars OMIDIENT family](./figures/01_nmi_bars_omidient.png)

### Fig 2 — Separation vs Δ X R²

![Scatter separation vs delta X R2](./figures/02_scatter_separation_vs_delta_xr2.png)

### Fig 3 — Color-coded metrics

![Metric heatmap AE](./figures/03_metric_heatmap_ae.png)

![Metric heatmap OMIDIENT](./figures/03_metric_heatmap_omidient.png)

### Fig 4 — Process schematic

![Method schematic](./figures/04_method_schematic.png)

### Fig 5–6 — embeddings & sweeps

Generate on Triton with `sbatch run_viz.slurm` (needs checkpoints / leaderboard CSVs under `results/`). After that, this section will show:

- `figures/05_embedding_umap.png` (or `05_embedding_tsne.png`) — includes Method 3 / Method 4 winners
- `figures/06_sweep_method1.png`, `06_sweep_method1_vicreg.png`, `06_sweep_method2.png`, `06_sweep_method3.png`, `06_sweep_method4.png`, `06_sweep_method2_omidient.png`, `06_sweep_omidient_ortho.png`

---

## Compact comparison (AE private)

Shared / private diagnostic panel (BRCA). `gap S−PS` = mean(S) NMI − concat(P,S) NMI (large positive ⇒ private hurts integration). `Δ X R²` = R²(X←S_other) − R²(X←P_other). Own-view columns: Ridge R² for \(X_m \leftarrow S_m\) / \(P_m\) / \([P_m;S_m]\) (mean over \(m\)). MOCSS–M2 rows are mean ± std over 5 training seeds; M3/M4 remain single-seed.

| Config | mean(S) NMI | concat(P) NMI | concat(P,S) NMI | gap S−PS | Δ X R² | X←S own | X←P own | X←[P;S] own | CKA |
| ------ | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MOCSS | 0.4181 ± 0.0138 | 0.4656 ± 0.0168 | 0.4607 ± 0.0083 | −0.0427 ± 0.0162 | 0.0445 ± 0.0234 | 0.4035 ± 0.0126 | 0.4812 ± 0.0061 | 0.3626 ± 0.0147 | 0.7605 ± 0.0240 |
| M1 + `b_m` loss-best (a1.5 b0.7 g0.7 p0.5) | 0.4142 ± 0.0303 | 0.2365 ± 0.0986 | 0.3234 ± 0.0632 | 0.0908 ± 0.0544 | −0.0383 ± 0.0951 | 0.2936 ± 0.0320 | 0.4138 ± 0.0284 | 0.3379 ± 0.0303 | 0.5891 ± 0.1029 |
| VICReg + `b_m` loss-best (a0.7 b0.5 v0.1 p1.0) | 0.4232 ± 0.0207 | 0.2539 ± 0.0514 | 0.3942 ± 0.0352 | 0.0290 ± 0.0383 | −0.0610 ± 0.0709 | 0.3467 ± 0.0396 | 0.4267 ± 0.0198 | 0.3456 ± 0.0279 | 0.7875 ± 0.0133 |
| M2 loss-best | 0.4171 ± 0.0280 | 0.3893 ± 0.0336 | 0.4338 ± 0.0328 | −0.0166 ± 0.0299 | 0.0507 ± 0.0444 | 0.3759 ± 0.0128 | 0.4415 ± 0.0035 | 0.3769 ± 0.0034 | 0.7703 ± 0.0233 |
| M2 high-NMI (r0.5 c0.5 v0.1) | 0.4079 ± 0.0191 | 0.4094 ± 0.0261 | 0.4293 ± 0.0300 | −0.0214 ± 0.0277 | 0.1030 ± 0.0251 | 0.3362 ± 0.0077 | 0.4490 ± 0.0054 | 0.3779 ± 0.0092 | 0.7985 ± 0.0244 |
| M3 spectral loss-best (s2 r1 c0) | 0.397 | 0.432 | 0.426 | −0.03 | **+0.26** | 0.406 | 0.407 | 0.371 | 0.54 |
| M3 high-NMI (s1.0 r0.5 c0) | 0.370 | — | **0.497** | — | — | — | — | — | — |
| M4 invariant loss-best | 0.345 | 0.357 | 0.368 | −0.02 | **+0.23** | 0.330 | 0.419 | 0.340 | **0.82** |

---

## Summary (best by eval_loss) — AE private

| Method | Best config | concat NMI | s_nmi | ARI | Acc | kNN | eval_loss |
| ------ | ----------- | ---: | ---: | ---: | ---: | ---: | ---: |
| MOCSS (AE) | `0.0004_0.0007` | 0.4607 ± 0.0083 | 0.4181 ± 0.0138 | 0.3604 ± 0.0320 | 0.6434 ± 0.0595 | 0.71 ± 0.03 | — |
| Method 1 (+`b_m`) | `0.0004_0.0007_a1.5_b0.7_g0.7_p0.5_l1.0` | 0.3234 ± 0.0632 | 0.4142 ± 0.0303 | 0.1904 ± 0.0624 | 0.4846 ± 0.0604 | 0.72 ± 0.03 | 1.0825 ± 0.0189 |
| Method 1 VICReg (+`b_m`) | `0.0004_0.0007_a0.7_b0.5_v0.1_p1.0_l1.0` | 0.3942 ± 0.0352 | 0.4232 ± 0.0207 | 0.2733 ± 0.0695 | 0.5794 ± 0.0726 | 0.73 ± 0.03 | 1.0256 ± 0.0019 |
| Method 2 | `0.0004_0.0007_r1.5_c0.5_v0.1` | 0.4338 ± 0.0328 | 0.4171 ± 0.0280 | 0.3200 ± 0.0493 | 0.6206 ± 0.0704 | 0.72 ± 0.03 | 0.4726 ± 0.0028 |
| Method 3 (spectral) | `0.0004_0.0007_s2.0_r1.0_c0.0_v0.1_vw0.5_lk0.5` | 0.4256 | 0.3968 | 0.2623 | 0.5600 | 0.77 | 0.2330 |
| Method 4 (invariant) | `0.0004_0.0007_d0.5_a2.0_r1.0_vw0.5_lk0.5_c0.5_v0.1` | 0.3682 | 0.3445 | 0.2319 | 0.5543 | 0.73 | 0.4918 |

---

## Baseline: MOCSS (AE)

Plain multi-omics contrastive shared–specific autoencoder (same private AE family as Method 1 / Method 2 AE). Config: `0.0004_0.0007`. Winner detail = mean ± std over seeds `{21…25}`.

| | nmi | ari | f_score | acc | knn |
| --- | ---: | ---: | ---: | ---: | ---: |
| concat(P,S) | 0.4607 ± 0.0083 | 0.3604 ± 0.0320 | 0.5238 ± 0.0257 | 0.6434 ± 0.0595 | 0.71 ± 0.03 |
| mean(S) | 0.4181 ± 0.0138 | 0.3138 ± 0.0321 | — | 0.5794 ± 0.0276 | 0.73 ± 0.03 |

Probes: subtype←S_other = 0.7470 ± 0.0086 | X←S R² = −0.1058 ± 0.0377 | X←P R² = −0.1503 ± 0.0294 | Δ(S−P) = 0.0445 ± 0.0234  
Own-view recon: X←S = 0.4035 ± 0.0126 | X←P = 0.4812 ± 0.0061 | X←[P;S] = 0.3626 ± 0.0147  
Alignment: cosine = 0.0254 ± 0.0589 | CCA = 0.9778 ± 0.0030 | CKA = 0.7605 ± 0.0240

---

## Method 1 (4 networks) — with `b_m`

Package: `code/method1_predictability/` (full predictability with own-network term + private-sufficiency `b_m`). Private encoders: plain AE (MOCSS-style). Rank: `eval_loss = L_std + L_shared + L_priv + L_ctr`.

### Top 10 by eval_loss

| Rank | Config | NMI | s_nmi | ARI | Acc | kNN | eval_loss |
| ---: | ------ | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `0.0004_0.0007_a1.5_b0.7_g0.7_p0.5_l1.0` | 0.3981 | 0.4532 | 0.2815 | 0.5714 | 0.68 | 1.0528 |
| 2 | `0.0004_0.0007_a1.5_b0.5_g0.7_p0.5_l1.0` | 0.4079 | 0.4329 | 0.2511 | 0.5429 | 0.70 | 1.0545 |
| 3 | `0.0004_0.0007_a1.5_b0.7_g0.7_p1.0_l1.0` | 0.4165 | 0.4539 | 0.2811 | 0.5714 | 0.68 | 1.0559 |
| 4 | `0.0004_0.0007_a0.5_b0.7_g0.7_p1.0_l1.0` | 0.3537 | 0.4055 | 0.2113 | 0.5543 | 0.70 | 1.0589 |
| 5 | `0.0004_0.0007_a1.5_b0.5_g0.7_p1.0_l1.0` | 0.3269 | 0.4133 | 0.2164 | 0.5200 | 0.68 | 1.0594 |
| 6 | `0.0004_0.0007_a0.3_b0.7_g0.7_p1.0_l1.0` | 0.4241 | 0.4349 | 0.2660 | 0.5257 | 0.73 | 1.0604 |
| 7 | `0.0004_0.0007_a1.0_b0.5_g0.7_p0.5_l1.0` | 0.4156 | 0.4863 | 0.2845 | 0.5371 | 0.70 | 1.0622 |
| 8 | `0.0004_0.0007_a1.0_b0.5_g0.7_p1.0_l1.0` | 0.3888 | 0.4338 | 0.2805 | 0.6000 | 0.70 | 1.0632 |
| 9 | `0.0004_0.0007_a1.5_b0.5_g0.7_p0.3_l1.0` | 0.3955 | 0.4725 | 0.2601 | 0.5886 | 0.73 | 1.0656 |
| 10 | `0.0004_0.0007_a1.0_b0.7_g0.7_p1.0_l1.0` | 0.3938 | 0.4311 | 0.2707 | 0.5771 | 0.66 | 1.0661 |

**Best (eval_loss):** `0.0004_0.0007_a1.5_b0.7_g0.7_p0.5_l1.0` (α=1.5, β=0.7, γ=0.7, λ_priv=0.5, λ_ctr=1.0). Winner detail = mean ± std over seeds `{21…25}` (Top-10 above still single-seed sweep).

| | nmi | ari | f_score | acc | knn |
| --- | ---: | ---: | ---: | ---: | ---: |
| concat(P,S) | 0.3234 ± 0.0632 | 0.1904 ± 0.0624 | 0.3956 ± 0.0451 | 0.4846 ± 0.0604 | 0.72 ± 0.03 |
| concat(P) | 0.2365 ± 0.0986 | 0.1405 ± 0.0617 | — | 0.4229 ± 0.0658 | 0.66 ± 0.07 |
| mean(S) | 0.4142 ± 0.0303 | 0.2899 ± 0.0502 | — | 0.5611 ± 0.0401 | 0.71 ± 0.02 |

Probes: subtype←S_other = 0.7455 ± 0.0358 | X←S R² = 0.0403 ± 0.0860 | X←P R² = 0.0786 ± 0.0669 | Δ(S−P) = −0.0383 ± 0.0951  
Own-view recon: X←S = 0.2936 ± 0.0320 | X←P = 0.4138 ± 0.0284 | X←[P;S] = 0.3379 ± 0.0303  
Alignment: cosine = −0.0265 ± 0.0410 | CCA = 0.9735 ± 0.0029 | CKA = 0.5891 ± 0.1029

---

## Method 1 VICReg ablation (no own network) — with `b_m`

Package: `code/method1_vicreg_ablation/`. Results: `results/models_brca_method1_vicreg/`. Private encoders: plain AE (MOCSS-style). Rank: `eval_loss = L_std + L_shared + L_priv + L_ctr` (`L_vic` excluded).

### Top 10 by eval_loss

| Rank | Config | NMI | s_nmi | ARI | Acc | kNN | eval_loss |
| ---: | ------ | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `0.0004_0.0007_a0.7_b0.5_v0.1_p1.0_l1.0` | 0.4374 | 0.4468 | 0.3341 | 0.6571 | 0.75 | 1.0249 |
| 2 | `0.0004_0.0007_a0.3_b0.5_v0.1_p1.0_l1.0` | 0.4344 | 0.4421 | 0.2889 | 0.5486 | 0.70 | 1.0250 |
| 3 | `0.0004_0.0007_a1.5_b0.5_v0.1_p1.0_l1.0` | 0.4513 | 0.4248 | 0.4437 | 0.6857 | 0.70 | 1.0254 |
| 4 | `0.0004_0.0007_a1.0_b0.5_v0.1_p1.0_l1.0` | 0.4018 | 0.4272 | 0.2617 | 0.5600 | 0.68 | 1.0265 |
| 5 | `0.0004_0.0007_a0.5_b0.5_v0.1_p1.0_l1.0` | 0.4148 | 0.4623 | 0.2639 | 0.5429 | 0.68 | 1.0289 |
| 6 | `0.0004_0.0007_a0.5_b0.7_v0.1_p1.0_l1.0` | 0.4720 | 0.4515 | 0.3131 | 0.5543 | 0.73 | 1.0314 |
| 7 | `0.0004_0.0007_a0.7_b0.7_v0.1_p1.0_l1.0` | 0.4485 | 0.4261 | 0.3269 | 0.6229 | 0.70 | 1.0357 |
| 8 | `0.0004_0.0007_a1.0_b0.5_v0.5_p1.0_l1.0` | 0.2151 | 0.4297 | 0.1178 | 0.4571 | 0.66 | 1.0369 |
| 9 | `0.0004_0.0007_a0.3_b0.7_v0.1_p1.0_l1.0` | 0.4558 | 0.4704 | 0.3045 | 0.5429 | 0.68 | 1.0374 |
| 10 | `0.0004_0.0007_a0.5_b0.5_v0.1_p0.5_l1.0` | 0.3687 | 0.3848 | 0.2038 | 0.4857 | 0.66 | 1.0382 |

**Best (eval_loss):** `0.0004_0.0007_a0.7_b0.5_v0.1_p1.0_l1.0` (α=0.7, β=0.5, λ_vic=0.1, λ_priv=1.0, λ_ctr=1.0). Winner detail = mean ± std over seeds `{21…25}`.

| | nmi | ari | f_score | acc | knn |
| --- | ---: | ---: | ---: | ---: | ---: |
| concat(P,S) | 0.3942 ± 0.0352 | 0.2733 ± 0.0695 | 0.4595 ± 0.0590 | 0.5794 ± 0.0726 | 0.73 ± 0.03 |
| concat(P) | 0.2539 ± 0.0514 | 0.1307 ± 0.0526 | — | 0.4194 ± 0.0710 | 0.69 ± 0.06 |
| mean(S) | 0.4232 ± 0.0207 | 0.3175 ± 0.0316 | — | 0.6057 ± 0.0210 | 0.71 ± 0.03 |

Probes: subtype←S_other = 0.7303 ± 0.0231 | X←S R² = −0.0351 ± 0.0539 | X←P R² = 0.0259 ± 0.0286 | Δ(S−P) = −0.0610 ± 0.0709  
Own-view recon: X←S = 0.3467 ± 0.0396 | X←P = 0.4267 ± 0.0198 | X←[P;S] = 0.3456 ± 0.0279  
Alignment: cosine = −0.0057 ± 0.0618 | CCA = 0.9800 ± 0.0022 | CKA = 0.7875 ± 0.0133

---

## Method 2 (joint recon + VICReg) — AE private

Package: `code/method2_joint_vicreg/`. Private encoders: plain AE (MOCSS-style). Rank: `eval_loss = L_recon + L_ctr`.

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

**Best (eval_loss):** `0.0004_0.0007_r1.5_c0.5_v0.1`. Winner detail = mean ± std over seeds `{21…25}`.

| | nmi | ari | f_score | acc | knn |
| --- | ---: | ---: | ---: | ---: | ---: |
| concat(P,S) | 0.4338 ± 0.0328 | 0.3200 ± 0.0493 | 0.4926 ± 0.0387 | 0.6206 ± 0.0704 | 0.72 ± 0.03 |
| mean(S) | 0.4171 ± 0.0280 | 0.3331 ± 0.0739 | — | 0.5909 ± 0.0578 | 0.72 ± 0.06 |

Probes: subtype←S_other = 0.7303 ± 0.0190 | X←S R² = −0.0450 ± 0.0345 | X←P R² = −0.0958 ± 0.0101 | Δ(S−P) = 0.0507 ± 0.0444  
Own-view recon: X←S = 0.3759 ± 0.0128 | X←P = 0.4415 ± 0.0035 | X←[P;S] = 0.3769 ± 0.0034  
Alignment: cosine = −0.0120 ± 0.0690 | CCA = 0.9772 ± 0.0026 | CKA = 0.7703 ± 0.0233

---

## Method 3 (spectral / graph shared) — v1 sweep

Package: `code/method3_spectral_graph/` (default code = **v1**; reproduces this table via `sbatch run_method3.slurm`). Design: [`adr-clustering-aligned-shared.md`](./adr-clustering-aligned-shared.md). Shared target = fused multi-omics Laplacian embedding \(U\); private = residual recon + view-ID + GRL leak. Rank: `eval_loss = L_spec + L_res + L_ctr`. Spectral ceiling on this run: `nmi_spectral_U = 0.3873`.

> **v2 ablation (not default):** stronger leak / true residual / λ_ctr floor / graph search did **not** raise `nmi_spectral_U` and lost Δ X R² (+0.26 → −0.05). Kept only as a negative result under `results/models_brca_method3_v2/` if present; do not use for re-runs.

### Top 10 by eval_loss

| Rank | Config | NMI | s_nmi | ARI | Acc | kNN | eval_loss |
| ---: | ------ | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `0.0004_0.0007_s2.0_r1.0_c0.0_v0.1_vw0.5_lk0.5` | 0.4256 | 0.3968 | 0.2623 | 0.5600 | 0.77 | 0.2330 |
| 2 | `0.0004_0.0007_s1.5_r1.0_c0.0_v0.1_vw0.5_lk0.5` | 0.4023 | 0.3913 | 0.2668 | 0.5771 | 0.70 | 0.2345 |
| 3 | `0.0004_0.0007_s1.0_r1.0_c0.0_v0.1_vw0.5_lk0.5` | 0.4557 | 0.3757 | 0.3507 | 0.6571 | 0.75 | 0.2365 |
| 4 | `0.0004_0.0007_s2.0_r0.5_c0.0_v0.1_vw0.5_lk0.5` | 0.4712 | 0.3822 | 0.3000 | 0.5771 | 0.70 | 0.2367 |
| 5 | `0.0004_0.0007_s1.5_r0.5_c0.0_v0.1_vw0.5_lk0.5` | 0.4838 | 0.3656 | 0.3808 | 0.6914 | 0.70 | 0.2377 |
| 6 | `0.0004_0.0007_s0.5_r1.0_c0.0_v0.1_vw0.5_lk0.5` | 0.4610 | 0.3933 | 0.3339 | 0.6514 | 0.77 | 0.2386 |
| 7 | `0.0004_0.0007_s1.0_r0.5_c0.0_v0.1_vw0.5_lk0.5` | 0.4967 | 0.3698 | 0.4641 | 0.7429 | 0.75 | 0.2394 |
| 8 | `0.0004_0.0007_s2.0_r0.3_c0.0_v0.1_vw0.5_lk0.5` | 0.4343 | 0.3547 | 0.3054 | 0.6057 | 0.70 | 0.2410 |
| 9 | `0.0004_0.0007_s0.5_r0.5_c0.0_v0.1_vw0.5_lk0.5` | 0.4425 | 0.4229 | 0.3045 | 0.5771 | 0.73 | 0.2422 |
| 10 | `0.0004_0.0007_s1.5_r0.3_c0.0_v0.1_vw0.5_lk0.5` | 0.4480 | 0.3695 | 0.3310 | 0.6457 | 0.68 | 0.2438 |

**Best (eval_loss):** `0.0004_0.0007_s2.0_r1.0_c0.0_v0.1_vw0.5_lk0.5` (λ_spec=2.0, λ_res=1.0, λ_ctr=0, λ_vic=0.1, λ_view=0.5, λ_leak=0.5)

| | nmi | ari | f_score | acc | knn |
| --- | ---: | ---: | ---: | ---: | ---: |
| concat(P,S) | 0.4256 | 0.2623 | 0.4480 | 0.5600 | 0.77 |
| concat(P) | 0.4323 | 0.3295 | — | 0.6629 | 0.77 |
| mean(S) | 0.3968 | 0.2539 | — | 0.5486 | 0.66 |

Probes: subtype←S_other = 0.7197 | X←S R² = 0.1444 | X←P R² = −0.1159 | Δ(S−P) = **0.2602**  
Own-view recon: X←S = 0.4059 | X←P = 0.4071 | X←[P;S] = 0.3710  
Alignment: cosine = −0.0441 | CCA = 0.9580 | CKA = 0.5359  
`nmi_spectral_U` = 0.3873

**Secondary (high concat NMI, not eval_loss-best):** `…_s1.0_r0.5_c0.0_…` → concat NMI 0.4967 / Acc 0.7429 / ARI 0.4641 (`s_nmi` 0.3698).

---

## Method 4 (invariant DEC cluster) — v1 sweep

Package: `code/method4_invariant_cluster/` (default code = **v1**; reproduces this table via `sbatch run_method4.slurm`). Shared = DEC + cross-view soft-label agreement; private = residual + view-ID + GRL leak. Rank: `eval_loss = L_dec + L_agree + L_res + L_ctr`.

> **v2 ablation (not default):** sharper DEC / stronger leak did not improve NMI (concat ~0.35). Optional artifacts under `results/models_brca_method4_v2/` only.

### Top 10 by eval_loss

| Rank | Config | NMI | s_nmi | ARI | Acc | kNN | eval_loss |
| ---: | ------ | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `0.0004_0.0007_d0.5_a2.0_r1.0_vw0.5_lk0.5_c0.5_v0.1` | 0.3682 | 0.3445 | 0.2319 | 0.5543 | 0.73 | 0.4918 |
| 2 | `0.0004_0.0007_d0.5_a1.0_r1.0_vw0.5_lk0.5_c0.5_v0.1` | 0.3632 | 0.3363 | 0.2028 | 0.5086 | 0.75 | 0.4961 |
| 3 | `0.0004_0.0007_d0.5_a2.0_r1.0_vw0.5_lk1.0_c0.5_v0.1` | 0.3450 | 0.3451 | 0.2021 | 0.5029 | 0.75 | 0.4980 |
| 4 | `0.0004_0.0007_d0.5_a2.0_r0.5_vw0.5_lk0.5_c0.5_v0.1` | 0.3749 | 0.3446 | 0.2171 | 0.5200 | 0.77 | 0.4988 |
| 5 | `0.0004_0.0007_d0.5_a0.5_r1.0_vw0.5_lk0.5_c0.5_v0.1` | 0.3646 | 0.3463 | 0.2045 | 0.4971 | 0.73 | 0.5012 |
| 6 | `0.0004_0.0007_d0.5_a1.0_r0.5_vw0.5_lk0.5_c0.5_v0.1` | 0.3458 | 0.3608 | 0.1986 | 0.4971 | 0.70 | 0.5026 |
| 7 | `0.0004_0.0007_d0.5_a1.0_r1.0_vw0.5_lk1.0_c0.5_v0.1` | 0.3627 | 0.3334 | 0.2119 | 0.5200 | 0.77 | 0.5034 |
| 8 | `0.0004_0.0007_d0.5_a1.0_r0.3_vw0.5_lk0.5_c0.5_v0.1` | 0.3668 | 0.3455 | 0.2109 | 0.5257 | 0.70 | 0.5038 |
| 9 | `0.0004_0.0007_d0.5_a2.0_r0.3_vw0.5_lk0.5_c0.5_v0.1` | 0.3676 | 0.3252 | 0.2190 | 0.5257 | 0.73 | 0.5047 |
| 10 | `0.0004_0.0007_d0.5_a0.5_r0.5_vw0.5_lk0.5_c0.5_v0.1` | 0.3459 | 0.3547 | 0.1984 | 0.4914 | 0.66 | 0.5065 |

**Best (eval_loss):** `0.0004_0.0007_d0.5_a2.0_r1.0_vw0.5_lk0.5_c0.5_v0.1` (λ_dec=0.5, λ_agree=2.0, λ_res=1.0, …)

| | nmi | ari | f_score | acc | knn |
| --- | ---: | ---: | ---: | ---: | ---: |
| concat(P,S) | 0.3682 | 0.2319 | 0.4244 | 0.5543 | 0.73 |
| concat(P) | 0.3570 | 0.2257 | — | 0.5429 | 0.73 |
| mean(S) | 0.3445 | 0.2433 | — | 0.5429 | 0.64 |

Probes: subtype←S_other = 0.7121 | X←S R² = 0.0932 | X←P R² = −0.1352 | Δ(S−P) = **0.2283**  
Own-view recon: X←S = 0.3295 | X←P = 0.4194 | X←[P;S] = 0.3398  
Alignment: cosine = 0.1983 | CCA = 0.9810 | CKA = **0.8234**  
Agreement diagnostics: `agree_symkl_mean` = 0.0076 | `entropy_q_bar` = 1.4233 (vs \(\ln 5 ≈ 1.61\); not collapsed)

---

## Method 2 / OMIDIENT family (separate comparison)

Dirichlet private (`P`) vs Method-2 AE private. Both OMIDIENT rows keep MOCSS-style shared; Method 2 + OMIDIENT uses Method-2 joint decode + VICReg with product-Dirichlet private. Not directly comparable to the AE-private compact panel above on private geometry. Own-view columns: Ridge R² \(X_m \leftarrow S_m\) / \(P_m\) / \([P_m;S_m]\) (mean over \(m\)).

| Method | Best config | mean(S) NMI | concat(P) NMI | concat(P,S) NMI | Δ X R² | X←S own | X←P own | X←[P;S] own | CKA | eval_loss |
| ------ | ----------- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Method 2 (AE private) | `0.0004_0.0007_r1.5_c0.5_v0.1` | 0.4171 ± 0.0280 | 0.3893 ± 0.0336 | 0.4338 ± 0.0328 | 0.0507 ± 0.0444 | 0.3759 ± 0.0128 | 0.4415 ± 0.0035 | 0.3769 ± 0.0034 | 0.7703 ± 0.0233 | 0.4726 ± 0.0028 |
| Method 2 + OMIDIENT private | `0.0004_0.0007_r1.5_c0.5_v0.1_k4` | 0.4097 ± 0.0181 | 0.0430 ± 0.0021 | 0.4089 ± 0.0141 | 0.7704 ± 0.0547 | 0.4466 ± 0.0109 | −0.3413 ± 0.0001 | 0.2582 ± 0.0138 | 0.7232 ± 0.0236 | 0.4799 ± 0.0018 |
| OMIDIENT (reported / no \(L_\mathrm{ortho}\)) | `0.0003_0.0005_4` (K=4) | 0.4262 ± 0.0463 | 0.0390 ± 0.0093 | 0.4297 ± 0.0536 | 1.1137 ± 0.0131 | 0.4270 ± 0.0083 | −0.3416 ± 0.0003 | 0.2430 ± 0.0113 | 0.7572 ± 0.0367 | 0.9642 ± 0.0042 |
| OMIDIENT (with \(L_\mathrm{ortho}\), λ-best) | `0.0003_0.0005_4_ortho_c2_r1.5_k1_o0.5` | 0.4290 ± 0.0344 | 0.0430 ± 0.0046 | 0.4290 ± 0.0344 | 1.1111 ± 0.0124 | 0.4251 ± 0.0043 | −0.3416 ± 0.0003 | 0.2440 ± 0.0071 | 0.7525 ± 0.0307 | 0.9619 ± 0.0048 |

### Method 2 + OMIDIENT private — top 10 / detail

Package: `code/method2_omidient_private/`. Rank: `eval_loss = L_recon + L_ctr` (`L_vic` / `L_kl` excluded).

| Rank | Config | NMI | s_nmi | ARI | Acc | kNN | eval_loss |
| ---: | ------ | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `0.0004_0.0007_r1.5_c0.5_v0.1_k4` | 0.3902 | 0.3925 | 0.2694 | 0.5543 | 0.73 | 0.4827 |
| 2 | `0.0004_0.0007_r0.7_c0.5_v0.1_k4` | 0.4238 | 0.4238 | 0.2789 | 0.5257 | 0.70 | 0.4830 |
| 3 | `0.0004_0.0007_r1.5_c1.0_v0.1_k4` | 0.4459 | 0.4459 | 0.3781 | 0.6914 | 0.68 | 0.4844 |
| 4 | `0.0004_0.0007_r1.5_c2.0_v0.1_k4` | 0.4026 | 0.4026 | 0.2541 | 0.5486 | 0.70 | 0.4900 |
| 5 | `0.0004_0.0007_r0.3_c0.5_v0.1_k4` | 0.4258 | 0.4285 | 0.2883 | 0.5486 | 0.75 | 0.4904 |
| 6 | `0.0004_0.0007_r0.7_c1.0_v0.1_k4` | 0.3985 | 0.3985 | 0.2469 | 0.5143 | 0.77 | 0.4922 |
| 7 | `0.0004_0.0007_r1.5_c2.0_v1.0_k4` | 0.4396 | 0.4396 | 0.2729 | 0.5314 | 0.66 | 0.4943 |
| 8 | `0.0004_0.0007_r1.5_c0.5_v1.0_k4` | 0.4389 | 0.4584 | 0.2773 | 0.5486 | 0.77 | 0.4960 |
| 9 | `0.0004_0.0007_r1.5_c1.0_v1.0_k4` | 0.3946 | 0.3946 | 0.2719 | 0.5657 | 0.80 | 0.4961 |
| 10 | `0.0004_0.0007_r0.3_c0.7_v0.1_k4` | 0.4661 | 0.4661 | 0.3677 | 0.6343 | 0.68 | 0.4978 |

**Best (eval_loss):** `0.0004_0.0007_r1.5_c0.5_v0.1_k4`. Winner detail = mean ± std over seeds `{21…25}`.

| | nmi | ari | f_score | acc | knn |
| --- | ---: | ---: | ---: | ---: | ---: |
| concat(P,S) | 0.4089 ± 0.0141 | 0.2860 ± 0.0235 | 0.4669 ± 0.0171 | 0.5886 ± 0.0361 | 0.73 ± 0.03 |
| concat(P) | 0.0430 ± 0.0021 | 0.0120 ± 0.0018 | — | 0.2891 ± 0.0051 | 0.50 ± 0.00 |
| mean(S) | 0.4097 ± 0.0181 | 0.2954 ± 0.0333 | — | 0.5966 ± 0.0433 | 0.73 ± 0.04 |

Probes: subtype←S_other = 0.7394 ± 0.0324 | X←S R² = −0.0987 ± 0.0547 | X←P R² = −0.8691 ± 0.0003 | Δ(S−P) = 0.7704 ± 0.0547  
Own-view recon: X←S = 0.4466 ± 0.0109 | X←P = −0.3413 ± 0.0001 | X←[P;S] = 0.2582 ± 0.0138  
Alignment: cosine = −0.0057 ± 0.0212 | CCA = 0.9729 ± 0.0027 | CKA = 0.7232 ± 0.0236

### OMIDIENT (reported / no \(L_\mathrm{ortho}\))

Config `0.0003_0.0005_4`. Trained with `L = L_ctr + 0.7 L_recon + KL` only — `orthogonal_loss` was **not** in `loss_total` (gap vs intended OMIDIENT / MOCSS-style coupling). Numbers below = mean ± std over seeds `{21…25}` (n=5), not the single published checkpoint.

| | nmi | ari | f_score | acc | knn |
| --- | ---: | ---: | ---: | ---: | ---: |
| concat(P,S) | 0.4297 ± 0.0536 | 0.3360 ± 0.0305 | 0.5074 ± 0.0220 | 0.5989 ± 0.0219 | 0.72 ± 0.02 |
| concat(P) | 0.0390 ± 0.0093 | 0.0085 ± 0.0085 | — | 0.2846 ± 0.0102 | 0.50 ± 0.00 |
| mean(S) | 0.4262 ± 0.0463 | 0.3281 ± 0.0299 | — | 0.5943 ± 0.0277 | 0.73 ± 0.04 |

Probes: subtype←S_other = 0.7364 ± 0.0196 | X←S R² = 0.2444 ± 0.0130 | X←P R² = −0.8693 ± 0.0005 | Δ(S−P) = 1.1137 ± 0.0131  
Own-view recon: X←S = 0.4270 ± 0.0083 | X←P = −0.3416 ± 0.0003 | X←[P;S] = 0.2430 ± 0.0113  
Alignment: cosine = −0.0442 ± 0.1202 | CCA = 0.9385 ± 0.0084 | CKA = 0.7572 ± 0.0367

### OMIDIENT (with \(L_\mathrm{ortho}\), λ-sweep best)

Intended OMIDIENT-style objective: `L = λ_ctr L_ctr + λ_recon L_recon + λ_kl KL + λ_ortho L_ortho`. Fixed `lr=0.0003`, `wd=0.0005`, `K=4`; sweep λ. Rank: unweighted `L_ctr + L_recon + KL`.

**Best (eval_loss):** `0.0003_0.0005_4_ortho_c2_r1.5_k1_o0.5` (λ_ctr=2.0, λ_recon=1.5, λ_kl=1.0, λ_ortho=0.5). Winner detail = mean ± std over seeds `{21…25}` (Top-10 below still single-seed λ-sweep).

| | nmi | ari | f_score | acc | knn |
| --- | ---: | ---: | ---: | ---: | ---: |
| concat(P,S) | 0.4290 ± 0.0344 | 0.3074 ± 0.0424 | 0.4824 ± 0.0323 | 0.5760 ± 0.0260 | 0.70 ± 0.03 |
| concat(P) | 0.0430 ± 0.0046 | 0.0119 ± 0.0044 | — | 0.2891 ± 0.0051 | 0.50 ± 0.00 |
| mean(S) | 0.4290 ± 0.0344 | 0.3074 ± 0.0424 | — | 0.5760 ± 0.0260 | 0.71 ± 0.05 |

| Rank | Config | NMI | s_nmi | ARI | Acc | kNN | eval_loss |
| ---: | ------ | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `…_ortho_c2_r1.5_k1_o0.5` | 0.4470 | 0.4537 | 0.3498 | 0.5943 | 0.73 | 0.9568 |
| 2 | `…_ortho_c2_r1.5_k0.5_o0.5` | 0.4470 | 0.4540 | 0.3498 | 0.5943 | 0.73 | 0.9568 |
| 3 | `…_ortho_c2_r1.5_k2_o0.5` | 0.4350 | 0.4537 | 0.3381 | 0.6000 | 0.73 | 0.9569 |
| 4 | `…_ortho_c2_r1.5_k2_o1` | 0.4357 | 0.4504 | 0.3123 | 0.5543 | 0.68 | 0.9582 |
| 5 | `…_ortho_c2_r1.5_k1_o1` | 0.4357 | 0.4453 | 0.3123 | 0.5543 | 0.70 | 0.9591 |
| 6 | `…_ortho_c2_r1.5_k0.5_o1` | 0.4783 | 0.4689 | 0.3969 | 0.6400 | 0.68 | 0.9591 |
| 7 | `…_ortho_c2_r1.5_k2_o2` | 0.4098 | 0.4161 | 0.3386 | 0.6286 | 0.73 | 0.9592 |
| 8 | `…_ortho_c2_r1.5_k0.5_o2` | 0.4051 | 0.3989 | 0.3214 | 0.6171 | 0.70 | 0.9598 |
| 9 | `…_ortho_c1_r0.7_k2_o1` | 0.4086 | 0.4241 | 0.3339 | 0.6343 | 0.75 | 0.9604 |
| 10 | `…_ortho_c2_r1.5_k1_o2` | 0.4609 | 0.4609 | 0.3761 | 0.6343 | 0.68 | 0.9612 |

Probes (best, multi-seed): subtype←S_other = 0.7500 ± 0.0186 | X←S R² = 0.2418 ± 0.0123 | X←P R² = −0.8693 ± 0.0004 | Δ(S−P) = 1.1111 ± 0.0124  
Own-view recon: X←S = 0.4251 ± 0.0043 | X←P = −0.3416 ± 0.0003 | X←[P;S] = 0.2440 ± 0.0071  
Alignment: cosine = −0.0361 ± 0.1128 | CCA = 0.9374 ± 0.0086 | CKA = 0.7525 ± 0.0307

---

## Analysis

Reading the BRCA numbers as a shared/private **decomposition** problem (not just “higher NMI”). Unless noted, AE / OMIDIENT figures below are **multi-seed means** (n=5).

**MOCSS — private steals shared.**  
`concat(P) ≈ concat(P,S) ≈ 0.46` while `mean(S) ≈ 0.42`. Under dual solo reconstruction, the AE private branch is strong enough to absorb subtype/shared biology. That is leakage into \(P\), not a clean split.

**VICReg + `b_m` — best AE-private compromise, still not a clear win over MOCSS.**  
Best AE-private control of the split (gap ~0.03, strong CKA ~0.79, usable concat). But it does **not** beat MOCSS on `concat(P,S)` NMI (0.394 ± 0.035 vs 0.461 ± 0.008) and is only tied/slightly ahead on `mean(S)` (0.423 vs 0.418). Better hygiene / less collapse than Method 1; not a decisive clustering upgrade over MOCSS.

**Method 1 + `b_m` — subtype coding ≠ reconstruction coding; high seed variance.**  
M1+`b_m` has `mean(S)` NMI (0.41) &gt; `concat(P)` NMI (0.24 ± 0.10), so \(S\) carries more **subtype** structure than \(P\) on average. Multi-seed Δ X R² is near a wash (−0.04 ± 0.10), not the old single-seed −0.19; concat NMI is also weaker and noisier (0.32 ± 0.06). Clustering NMI and Ridge R² remain different axes. VICReg+`b_m` is stabler (Δ ≈ −0.06 ± 0.07).

**Method 2 — not working as intended.**  
It behaves like a **cleaner MOCSS**: joint \(D([P,S])\) + VICReg improve training and alignment, but `concat(P)` stays high (~0.39) and the gap stays near zero/negative. Why: VICReg only soft-penalizes linear \(P\)–\(S\) dependence, and joint recon still allows both codes to carry overlapping content as long as the concat decoder succeeds. There is no operational definition that forces cross-view content into \(S\) and residual into \(P\) (unlike Method 1’s `f`/`g`/`k`/`b`). So independence + joint recon do not really separate shared from private. (M2’s *positive* Δ X R² is mostly foreign \(P\) failing at cross-view \(X\), not foreign \(S\) succeeding.)

**OMIDIENT (reported / no \(L_\mathrm{ortho}\)) — uncoupled paths; single-seed “0.50 NMI” does not hold.**  
Training omits `L_ortho`, so \(P\) and \(S\) have **no direct coupling**. Across seeds, `concat(P)≈0.04` and `mean(S)≈0.43±0.05` (not the optimistic single-checkpoint ~0.50). That is still a strong private-weak / shared-carrying pattern from **path separation**, not a verified shared/private theory win. Treat \(S\) as a contrastive multi-view code that can mix private detail.

**OMIDIENT (with \(L_\mathrm{ortho}\)) — intended-style; similar multi-seed level to no-ortho.**  
With ortho (λ-best), multi-seed `concat(P,S)≈mean(S)≈0.43` and `concat(P)≈0.04` — essentially the same shared NMI band as no-ortho once variance is included. Dirichlet still keeps \(P\) subtype-weak; the dramatic single-seed gap between “0.50 no-ortho” and “0.45 ortho” shrinks under re-seeding.

**Realization — reconstruction training ≠ clustering objective.**  
In MOCSS/OMIDIENT, reconstruction (+ ortho / contrastive / …) is a **self-supervised proxy** used to shape \(S\) and \(P\); the paper-level end task is **unsupervised subtyping** (KMeans on fused latents → NMI). Those two are **not automatically coordinated**: embeddings that are good for reconstructing \(X\) need not be geometry that KMeans likes, and vice versa. Diagnostics make that mismatch concrete, e.g.:

- **VICReg + `b_m` (best AE split so far):** lowest `concat(P)` NMI among AE methods (~0.25) with still-high `mean(S)` NMI (~0.42) and solid own-view recon (X←S ≈ 0.35, X←P ≈ 0.43, X←[P;S] ≈ 0.35) — closest to the intended definition (\(S\) subtype-rich, \(P\) less so, both still reconstructive within view). **Even this run lacks cross-view reconstruction**: Δ X R² ≈ −0.06, so foreign codes barely help predict \(X_A\).
- **M1 + `b_m`:** `concat(P)` NMI ~0.24 ± 0.10, own-view X←P R² ≈ 0.41 — \(P\) is **recon-friendly, not subtype-cluster-friendly**; own-shared X←S ≈ 0.29.
- **MOCSS:** high `concat(P)` / `concat(P,S)` NMI, yet cross-view X←S / X←P R² both **negative**, while own-view X←P R² ≈ 0.48 is the strongest AE private recon — **cluster-friendly codes need not be cross-view reconstructive**.
- **Dirichlet \(P\)** (OMIDIENT / M2+OMI): own-view X←P R² ≈ −0.34 with `concat(P)` NMI ≈ 0.04 — private is neither subtype- nor (linearly) recon-useful; \(S\) carries almost all own-view recon.

So the open design question is not only “better AE losses,” but **how to define tasks or design a system under the hood which is aligned with the final clustering/subtyping goal** (and, for this thesis, with a true shared/private split). Reconstruction alone does not guarantee that — and even the best current AE split (VICReg+`b_m`) still fails the cross-view recon story that would make \(S\) “shared” in the Method-1 sense.

**Method 3 (spectral v1) — cross-view \(X\) win; shared ceiling limited by \(U\).**  
First AE-family run with a clear positive Δ X R² (**+0.26**): foreign \(S\) predicts \(X_A\) better than foreign \(P\). `mean(S)` NMI (~0.40) sits at the spectral ceiling (`nmi_spectral_U≈0.39`), so the graph target—not the regressor—is the bottleneck. Top eval_loss configs all used `λ_ctr=0` → weak CKA (0.54). High `concat(P)` NMI (~0.43) does **not** prove \(S\)/\(P\) are the same code: \(P\) can hold complementary cluster-useful private biology; the Δ X R² / subtype←S_other probes are the better “is \(S\) shared?” tests. Secondary high-NMI configs reach concat NMI ~0.50 / Acc ~0.74 without purifying `mean(S)`.

**Method 4 (invariant v1) — agreement OK, clustering weak.**  
Cross-view soft labels agree (`symKL≈0.008`) and CKA is strong (0.82); entropy shows no hard collapse. But concat / `s_nmi` lag MOCSS (~0.37 / 0.34). Leaderboard prefers **weak DEC + strong agree** (`d0.5`, `a2.0`) — invariance without sharp subtype geometry. Δ X R² still positive (+0.23).

**Takeaway.**  
Under multi-seed reporting, the uncoupled Dirichlet “0.50 NMI” free lunch shrinks; no-ortho and ortho OMIDIENT land in a similar ~0.43 shared-NMI band with near-zero `concat(P)`. Among AE-private methods, VICReg+`b_m` remains the most coherent *subtype* split so far but still does not beat MOCSS on concat NMI; Method 3 is the first clear *cross-view recon* win for \(S\) (single-seed so far). Method 2 needs a stronger shared definition; Methods 3–4 need multi-seed checks plus a stronger graph / sharper clustering objective (v2). The deeper issue remains **proxy–target misalignment**: recon-trained \(S\)/\(P\) are not necessarily clustering-optimal (or decomposition-optimal).

---

## Shared-quality metrics

See **Compact comparison** (AE private), **Method 3 / 4** sections, and **Method 2 / OMIDIENT family** above. Per-method probe/alignment numbers for the eval_loss winners are also listed under each method section.

---

## Single-\(Z\) clustering track (no \(S\)/\(P\))

Separate from Methods 1–4. Goal: maximize BRCA subtype NMI with one patient embedding \(Z\) (beat / contextualize OMIDIENT no-ortho ~0.50). Design: [`adr-single-z-clustering.md`](./adr-single-z-clustering.md).

| ID | Package | Role | Results dir |
| -- | ------- | ---- | ----------- |
| **A** | `code/cluster_a_contrastive_dec/` | Contrastive + DEC on \(Z=\mathrm{mean}(h_m)\) | `results/models_brca_cluster_a/` |
| **D** | `code/cluster_d_consensus/` | Consensus co-association baseline (CPU) | `results/models_brca_cluster_d/` |
| **B** | `code/cluster_b_scan/` | SCAN fine-tune on A’s checkpoint | `results/models_brca_cluster_b/` |
| **C** | `code/cluster_c_gnn/` | Pure-PyTorch GCN + DEC (gated) | `results/models_brca_cluster_c/` |

**Ranking:** A/C by label-free `eval_loss` (A: \(L_\mathrm{ctr}+L_\mathrm{dec}\); C: val DEC KL). D/B may report NMI for analysis; fill the comparison table after Triton runs.

| Method | Config | NMI (KMeans on \(Z\) / assign) | Notes |
| ------ | ------ | ---: | ----- |
| OMIDIENT no-ortho (bar) | `0.0003_0.0005_4` | ~0.50 (mean(S) ≈ concat) | free lunch; no true \(S\)/\(P\) |
| Cluster D (headline) | *TBD* | — | full-cohort consensus |
| Cluster A (eval_loss) | *TBD* | — | test split |
| Cluster B (best NMI) | *TBD* | — | needs A |
| Cluster C (eval_loss) | *TBD* | — | gated |

### Run on Triton

```bash
cd /scratch/work/soumir1/githubmultiomics/multiomics
git pull   # after pushing this branch

# 1) Consensus baseline (CPU, ~minutes) — can run in parallel with A
sbatch run_cluster_d.slurm

# 2) Primary neural method (A100, ~48 configs)
sbatch run_cluster_a.slurm

# 3) After A finishes (needs results/models_brca_cluster_a/best_config.json)
sbatch run_cluster_b.slurm
# optional: pin parent A config
# sbatch --export=ALL,PARENT_CONFIG=0.0004_0.0007_c1.0_d1.0_r0.0 run_cluster_b.slurm

# 4) Only if A is below D and/or clearly below ~0.50 after B
sbatch run_cluster_c.slurm
```

**Interactive / debug** (same `PYTHONPATH` as other jobs):

```bash
export PYTHONPATH=/scratch/work/soumir1/githubmultiomics:/scratch/work/soumir1
source /scratch/work/soumir1/multiomics/venv/bin/activate
cd /scratch/work/soumir1/githubmultiomics/multiomics

# D
cd code/cluster_d_consensus && python -u run_baseline.py --disease brca --grid

# A
cd ../cluster_a_contrastive_dec && python -u hyperparam_tuning.py --epochs 100 --resume

# B (after A)
cd ../cluster_b_scan && python -u hyperparam_tuning.py --epochs 50 --resume

# C (gated)
cd ../cluster_c_gnn && python -u hyperparam_tuning.py --epochs 200 --resume
# quick smoke: add --quick
```

Artifacts: `leaderboard.csv` + `best_config.json` under each `results/models_brca_cluster_{a,b,c,d}/`.

---

## Appendix: Method 1 / VICReg without `b_m` (pre–private-sufficiency)

Earlier sweeps **before** adding `b_m(P_m) → X_m`. Rank was `eval_loss = L_std + L_shared + L_ctr` (no `L_priv`). Kept for comparison with the `+b_m` tables above; not used in the compact panel.

### Method 1 (4 networks) — without `b_m`

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

### Method 1 VICReg ablation — without `b_m`

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

