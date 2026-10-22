# ADR: Clustering-Aligned Shared / Private Representation Learning

> **Status:** Accepted for implementation (design freeze for v1)  
> **Date:** 2026-10-22  
> **Related:** [`shared-representation-method.md`](./shared-representation-method.md), [`README.md`](./README.md), single-\(Z\) clustering track: [`adr-single-z-clustering.md`](./adr-single-z-clustering.md)  
> **Note on math:** plain text so it renders in any Markdown preview.

This ADR records the decision to add two **clustering-aligned** methods that replace (or demote) feature reconstruction as the definition of shared codes. It is the implementation contract for:

| Working name | Package (planned) | Shared defined by |
| ------------ | ----------------- | ----------------- |
| **Method 3** (Idea 3) | `code/method3_spectral_graph/` | Multi-omics fused graph / spectral geometry |
| **Method 4** (Idea 4) | `code/method4_invariant_cluster/` | Cross-omics invariant soft clustering rule |
| **Method 3+4 hybrid** (preferred v1 ship) | `code/method3_spectral_graph/` with private block from Method 4 | Spectral shared + residual / non-leakage private |

---

## 1. Context

### 1.1 What we already know from BRCA results

From `docs/README.md` (Analysis):

- Reconstruction (+ contrastive / VICReg / adversaries) is a **self-supervised proxy**.
- The paper-level end task is **unsupervised subtyping** (KMeans on fused latents → NMI / ARI / Acc).
- Those objectives are **not automatically coordinated**.
- Best AE-private split so far: **Method 1 VICReg + `b_m`** (hygiene), but still weak **cross-view** \(X\)-recon and no decisive NMI win over MOCSS.
- **OMIDIENT (no \(L_\mathrm{ortho}\))** reaches `mean(S) ≈ concat(P,S) ≈ 0.50` largely because \(P\) and \(S\) were **uncoupled** — a free lunch, not a verified shared/private theory win.
- Fair scientific bar: beat / match **MOCSS** and **OMIDIENT with \(L_\mathrm{ortho}\)** (~0.45) **with** a real \(S/P\) split. Beating no-ortho 0.50 is aspirational.

### 1.2 Decision drivers

1. Align the **training target** with clustering geometry (not raw \(X\) fidelity).
2. Keep a **defensible shared/private split** for the thesis (disjoint \(S\) vs \(P\) is success even if NMI ≤ 0.50).
3. Reuse MOCSS-aligned plumbing (encoders, batching, `shared_rep_metrics`, Slurm sweep pattern).
4. Avoid recreating MOCSS’s failure mode: **solo \(P \to X\) full reconstruction**.

### 1.3 Decision

| Priority | Choice | Rationale |
| -------- | ------ | --------- |
| **Implement first** | Method 3 (spectral / graph shared) | Closest proxy to KMeans; clearer ceiling (quality of fused graph) |
| **Private block** | Method 4 residual + view-ID + non-leakage | Fixes “why would \(P\) reconstruct all of \(X\)?” |
| **Implement second** | Standalone Method 4 | Stronger invariance / thesis chapter; harder to tune |
| **Do not chase as primary KPI** | Beat OMIDIENT no-ortho 0.50 NMI | Unfair uncoupled baseline; report as aspirational |

**Preferred v1 system:** Method 3 shared objective + Method 4 private objectives (hybrid). Standalone Method 4 follows once Method 3 is green.

---

## 2. Shared definitions (both methods)

Operational definitions used in this ADR:

```text
S  = information that induces the SAME patient grouping across modalities
P  = within-modality residual / assay-specific structure that is USELESS for that grouping
```

**Not** used as the definition of shared:

```text
✗  Dec_S(S_m) → X_m          (MOCSS solo shared recon)
✗  Dec_P(P_m) → X_m alone    (MOCSS solo private recon / over-strong b_m)
```

Optional light joint recon may remain as scaffolding:

```text
✓  Dec(P_m, stopgrad(S_m)) → X_m     # residual sufficiency
✓  tiny λ_alive · variance hinge on P # anti-collapse only
```

### 2.1 Success criteria (v1)

**Primary (decomposition) — must improve vs MOCSS AE:**

| Metric | Desired direction |
| ------ | ----------------- |
| `mean(S)` NMI | ↑ (competitive with MOCSS ~0.42 / OMI-ortho ~0.45) |
| `concat(P)` NMI | ↓ (clearly below `mean(S)`) |
| gap `mean(S) − concat(P,S)` | ≥ 0 preferred (private should not hurt / steal) |
| Own-view: X←[P;S] vs X←S / X←P | Joint best; P alone **not** full-\(X\) strong |
| Cluster soft-label leak from P | Low (Method 4 / hybrid) |

**Secondary (integration):**

| Metric | Fair bar | Aspirational |
| ------ | -------- | ------------ |
| `concat(P,S)` NMI | ≥ MOCSS (~0.45) / OMI-ortho (~0.45) | ≥ OMI no-ortho (~0.50) |

**Ranking for sweeps (label-free):** core positive val terms only (see each method). Do **not** rank by NMI during HPO.

### 2.2 Shared plumbing (reuse from Methods 1–2)

Unless a section overrides:

- BRCA, batch 32, 100 epochs, early stop patience 20 / `min_delta=0.005`, seed 21.
- Encoders: 4-layer MLP `tanh`, widths `[512, 256, 128, 32]` (emb = 32).
- Decoders: 4-layer MLP, `tanh` then `sigmoid` last.
- Init: Kaiming normal Linear; zero bias.
- MOCSS-style recon metric when any recon term is used (center → row L2 → `matrix_norm`).
- Optional VICReg `vicreg_ps_independence(P,S)` with single `λ_vic` (same as Method 2).
- Optional MOCSS-style `L_ctr` on MLP(`S`) — **demoted** (small `λ_ctr` or off) so spectral / invariance dominate shared.
- Eval: `code/shared_rep_metrics.py` + same KMeans `n_init=30`.
- Results under `results/models_brca_method3/` / `results/models_brca_method4/`.
- Private branch v1: **plain AE** (fair vs MOCSS). Optional later: Dirichlet private (OMIDIENT mix-in).

---

## 3. Method 3 — Spectral / graph shared (Idea 3)

### 3.1 One-sentence summary

Train \(S\) to match **multi-omics neighborhood / spectral geometry**; train \(P\) as **residual + non-leakage**, not as a full autoencoder of \(X\).

### 3.2 Motivation

Spectral embeddings of a patient graph are the continuous relaxation of clustering. Optimizing \(S\) toward fused-graph geometry aligns the proxy with KMeans much more than reconstructing assay intensities.

### 3.3 Offline graph + spectral target (precompute once per disease)

**Inputs:** training-set feature matrices \(X_1, X_2, X_3\) (same sample order / patient IDs as the rest of the repo).

**Steps:**

```text
For each modality m:
  1. Optional PCA to d_pca dimensions (default d_pca=50) on train only;
     apply same projection to val/test rows.
  2. Row L2-normalize PCA features.
  3. Affinity A_m[i,j] = cosine_sim(z_m_i, z_m_j) for j among kNN of i
     (mutual kNN recommended: edge if i∈kNN(j) or j∈kNN(i); default k=15).
  4. Symmetrize: A_m ← max(A_m, A_mᵀ); zero diagonal.

Fuse:
  A_fused = (A_1 + A_2 + A_3) / 3
  # v1 alternative (ablation): A_fused[i,j] = 1 if edge in ≥2 modalities else 0

Laplacian (normalized):
  D = diag(sum_j A_fused[i,j])
  L = I − D^{-1/2} A_fused D^{-1/2}

Eigendecomposition:
  L U = U Λ
  Keep U = [u_1, …, u_r] corresponding to the r smallest eigenvalues
  AFTER the trivial u_0 (constant / near-zero eigenvalue).
  Default r = 16 (≤ emb dim 32).

Row L2-normalize U (standard spectral clustering practice).
Save: results/.../spectral_targets/{disease}_U_train.npy (+ val/test aligned rows
      via out-of-sample Nyström OR restrict spectral loss to train indices only).
```

**v1 decision on val/test spectral targets:**

- **Train loss:** only on patients that appear in the precomputed train graph.
- **Val checkpoint score:** use train-graph \(U\) only for the spectral term on train batches; for val, score `L_res + L_ctr + L_vic + L_leak` **without** requiring val \(U\), **or** build graph on full unlabeled cohort (train+val features, no labels) then split rows — acceptable because graph uses no subtype labels.
- **Preferred v1:** build \(A_fused\) on **all unlabeled samples** used in the dataset split files (still no labels), compute \(U\) once, index by sample id. Simplest and avoids Nyström.

**Leakage note:** using features from val/test to build the graph is standard for unsupervised multi-omics integration papers (no label use). Document it explicitly in the thesis. Do **not** use subtype labels anywhere in graph construction.

### 3.4 Architecture

```text
X_m → Enc_m → S_m ∈ R^{32}, P_m ∈ R^{32}

Shared:
  S̄ = mean(S_1, S_2, S_3)          # v1 fusion
  # ablation: S̄ = MLP([S_1;S_2;S_3])
  f_spec(S̄) → Û ∈ R^{r}           # 2-layer MLP tanh → linear

Private / residual:
  Dec_m( concat(P_m, stopgrad(S_m)) ) → X̂_m

Optional:
  MLP_proj(S_m) → contrastive (small λ_ctr)
  vicreg_ps_independence(P_m, S_m)
  Adv_cluster(P_m) → soft cluster from stopgrad(q_S)   # from Method 4 private block
```

### 3.5 Losses

**A. Spectral regression (default shared loss)**

```text
L_spec = mean_i ‖ f_spec(S̄_i) − U_i ‖₂²
```

**B. Neighborhood contrastive (ablation / optional add-on)**

For each patient \(i\) in batch, positives = fused-graph neighbors of \(i\) that also fall in the batch (or a memory bank of embeddings). v1 keep it simple:

```text
# If too few in-batch neighbors, skip or use spectral-only.
L_graph = InfoNCE(S̄_i, S̄_j for j ∈ N_fused(i) ∩ batch)
```

**Default v1:** `L_spec` only. Add `L_graph` later if spectral regression underperforms.

**C. Residual reconstruction (private sufficiency — NOT solo P→X)**

```text
L_res = Σ_m recon( Dec_m(concat(P_m, stopgrad(S_m))), X_m )
```

MOCSS-style recon metric. `stopgrad(S)` so residual path does not redefine shared.

**D. Optional anti-collapse / independence**

```text
L_vic = Σ_m vicreg_ps_independence(P_m, S_m)
```

**E. Optional light contrastive (demoted)**

```text
L_ctr = InfoNCE pairs on MLP(S_m)     # same as MOCSS InstanceLoss
```

**F. Non-leakage (from Method 4 private block — include in hybrid v1)**

```text
q = softmax( h_cluster(stopgrad(S̄)) / τ )   # or KMeans soft labels on S̄, refreshed
L_leak_step1 = CE( Adv(P_m), q )            # train Adv only; P detached
L_leak_step2 = − β_leak · CE( Adv(P_m), q ) # encoder: make P uninformative for q
# alternative: gradient reversal layer on Adv — single optimizer
```

v1 recommendation: **GRL** (simpler than Method 1 two-opt) with weight `λ_leak`.

**Total (hybrid Method 3 v1):**

```text
L = λ_spec · L_spec
  + λ_res  · L_res
  + λ_ctr  · L_ctr
  + λ_vic  · L_vic
  + λ_leak · L_leak_grl
```

All terms handled so checkpointing stays on a **non-negative** score:

```text
val_score = λ_spec·L_spec + λ_res·L_res + λ_ctr·L_ctr + λ_vic·L_vic
# exclude GRL adversarial magnitude from early-stop if unstable; log it separately
```

**Rank configs by:**

```text
eval_loss = L_spec + L_res + L_ctr
# exclude L_vic, L_leak from ranking (regularizers)
```

### 3.6 Hyperparameter grids (v1)

```text
Fixed:
  k_nn = 15, d_pca = 50, r = 16
  emb = 32, batch = 32, epochs = 100, seed = 21
  lr / wd = MOCSS BRCA defaults (0.0004 / 0.0007) unless sweep says otherwise

Sweep:
  λ_spec ∈ {0.5, 1.0, 1.5, 2.0}
  λ_res  ∈ {0.3, 0.5, 1.0}
  λ_ctr  ∈ {0.0, 0.3, 0.5}      # include 0 = spectral-only shared
  λ_vic  ∈ {0.0, 0.1, 0.5}
  λ_leak ∈ {0.0, 0.5, 1.0}

→ up to 4×3×3×3×3 = 324 configs; if too heavy, freeze λ_vic=0.1, λ_leak=0.5
  and sweep λ_spec × λ_res × λ_ctr (4×3×3 = 36) first.
```

Config folder token example:

```text
{lr}_{wd}_s{λ_spec}_r{λ_res}_c{λ_ctr}_v{λ_vic}_lk{λ_leak}
```

### 3.7 Collapse modes and mitigations

| Failure | Symptom | Mitigation |
| ------- | ------- | ---------- |
| \(S\) ignores \(U\) | high `L_spec`, low `s_nmi` | ↑ `λ_spec`; check \(U\) quality (KMeans on \(U\) NMI baseline) |
| Bad graph | KMeans on \(U\) itself has low NMI | retune `k`, PCA dim, fusion rule **before** blaming the net |
| \(P\) steals subtypes | high `concat(P)` NMI | ↑ `λ_leak`; keep `stopgrad(S)` on residual; do not add solo \(P\to X\) |
| \(P\) dies | near-zero P variance / useless residual | `L_vic` var hinge; modest `λ_res` |
| \(S\) = copy of \(U\) only on train | overfit spectral | early stop on val `eval_loss`; optional tiny weight decay already present |

**Mandatory sanity check before big sweeps:**  
Run KMeans (`n_init=30`) on precomputed \(U\) alone and log NMI/ARI. This is the **ceiling proxy** for Method 3. If `NMI(U)` ≪ MOCSS, fix the graph first.

### 3.8 Package layout (planned)

```text
code/method3_spectral_graph/
  __init__.py
  graph_spectral.py      # PCA, kNN, fuse, Laplacian, save/load U
  model.py               # Enc, Dec residual, f_spec, optional Adv/GRL
  losses.py              # L_spec, L_res, L_ctr wrap, L_vic wrap, L_leak
  training.py            # one optimizer; val_score; logging
  hyperparam_tuning.py
  evaluate.py            # shared_rep_metrics + extra: nmi_on_U baseline
run_method3.slurm
```

### 3.9 Extra metrics to log (beyond `shared_rep_metrics`)

```text
nmi_spectral_U          # KMeans on U (baseline ceiling)
l_spec, l_res, l_ctr, l_vic, l_leak
mean_std_P, mean_std_S  # split variance (avoid Method-2 logging gap)
probe_q_from_P_acc      # how well Adv predicts cluster soft-label from P
```

---

## 4. Method 4 — Invariant cluster shared (Idea 4)

### 4.1 One-sentence summary

Treat each omics as an **environment**. \(S\) must support **one** grouping head \(h\) that produces the **same** soft subtypes from every modality; \(P\) holds environment-specific residual and must **not** predict that grouping.

### 4.2 Motivation

IRM / domain-invariance: features that only work in one assay are spurious for true shared biology. Cross-view agreement of a clustering rule is a direct operationalization of “shared,” without requiring \(S\) to reconstruct \(X\).

### 4.3 Honest limitation (document in thesis)

Without labels, the “task” is **invented** (DEC / prototypes / spectral soft-assignment). Invariance is w.r.t. that task, not guaranteed w.r.t. true subtypes. That is why Method 3’s graph target is a good partner: it supplies a clustering-shaped task.

### 4.4 Architecture

```text
X_m → Enc_m → S_m, P_m

Shared head (parameters shared across modalities):
  h(S_m) → logits_m ∈ R^{K_c}
  q_m = softmax(logits_m / τ)

Fusion for eval / leak target:
  S̄ = mean(S_1,S_2,S_3)
  q̄ = softmax(h(S̄) / τ)

Private:
  Dec_m(concat(P_m, stopgrad(S_m))) → X̂_m
  view_head(P_m) → modality_id ∈ {1,2,3}
  Adv(P_m) → q̄   (GRL or two-step)
```

Default \(K_c\): number of BRCA subtypes used in eval (typically 5), or treat \(K_c\) as a hyperparameter `{3,4,5,6}`.

### 4.5 Losses

**A. Per-view clustering loss (unsupervised)**

v1 pick **DEC-style** (stable, well-known):

```text
# Soft assignment (Student-t), centroids μ_k learned
q_m[i,k] ∝ (1 + ‖S_m_i − μ_k‖²/α)^{-(α+1)/2}
p_m[i,k] ∝ q_m[i,k]² / Σ_i q_m[i,k]      # target distribution (standard DEC)
L_dec^m = KL(p_m ‖ q_m)
L_dec = Σ_m L_dec^m
```

Warm-up: 10–20 epochs contrastive or spectral init (if hybrid) before enabling `L_dec`.

**Alternative (ablation):** SwAV-style Sinkhorn prototypes on \(S_m\) with cross-view swapped prediction.

**B. Cross-view agreement (invariance heart)**

```text
L_agree = KL(q_1 ‖ q_2) + KL(q_1 ‖ q_3) + KL(q_2 ‖ q_3)
# use symmetric KL or JS; v1: mean of KL(q_a‖q_b)+KL(q_b‖q_a) per pair
```

**C. Optional weak IRM penalty (v1.1, not required for first run)**

```text
L_irm = Var_m ( ‖∇_θ L_dec^m‖² )   # expensive / noisy — defer
```

**D. Residual recon (same as Method 3)**

```text
L_res = Σ_m recon( Dec_m(concat(P_m, stopgrad(S_m))), X_m )
```

**E. View identity**

```text
L_view = Σ_m CE( view_head(P_m), m )
```

**F. Non-leakage**

```text
L_leak via GRL: Adv(P_m) predicts stopgrad(q̄)
```

**G. Optional VICReg + tiny contrastive**

```text
L_vic, L_ctr as in Method 3
```

**Total (standalone Method 4 v1):**

```text
L = λ_dec · L_dec
  + λ_agree · L_agree
  + λ_res · L_res
  + λ_view · L_view
  + λ_leak · L_leak
  + λ_ctr · L_ctr
  + λ_vic · L_vic
```

**Checkpoint:**

```text
val_score = λ_dec·L_dec + λ_agree·L_agree + λ_res·L_res + λ_view·L_view + λ_ctr·L_ctr + λ_vic·L_vic
```

**Rank:**

```text
eval_loss = L_dec + L_agree + L_res + L_ctr
# exclude L_vic, L_leak, L_view from ranking (or include L_view — prefer exclude)
```

### 4.6 Why not `recon(P → X)` alone

| Loss | Effect |
| ---- | ------ |
| `recon(P → X)` strong | \(P\) becomes full AE → subtypes leak into private (MOCSS / strong `b_m`) |
| `recon([P, sg(S)] → X)` | \(P\) only needs residual → correct private meaning |
| `L_view` | Forces \(P\) to keep assay-specific signal even if residual is small |
| `L_leak` | Explicitly removes grouping information from \(P\) |

### 4.7 Hyperparameter grids (v1)

```text
Fixed:
  K_c = 5, τ = 1.0, DEC α = 1.0
  warm-up_epochs = 15 (L_ctr or frozen random centroids → then full L)

Sweep:
  λ_dec   ∈ {0.5, 1.0, 1.5}
  λ_agree ∈ {0.5, 1.0, 2.0}
  λ_res   ∈ {0.3, 0.5, 1.0}
  λ_view  ∈ {0.3, 0.5, 1.0}
  λ_leak  ∈ {0.5, 1.0, 2.0}
  λ_ctr   ∈ {0.0, 0.5}
  λ_vic   ∈ {0.0, 0.1}

Start smaller: freeze λ_view=0.5, λ_vic=0.1, λ_ctr=0.5;
sweep λ_dec × λ_agree × λ_res × λ_leak (3×3×3×3 = 81).
```

Config token:

```text
{lr}_{wd}_d{λ_dec}_a{λ_agree}_r{λ_res}_vw{λ_view}_lk{λ_leak}_c{λ_ctr}_v{λ_vic}
```

### 4.8 Collapse modes and mitigations

| Failure | Symptom | Mitigation |
| ------- | ------- | ---------- |
| One-cluster collapse | entropy(\(q\)) → 0 | DEC target sharpening + balance; Sinkhorn; ↑ warm-up |
| Agreement by trivial constant \(q\) | low `L_agree`, useless NMI | cluster loss + entropy floor on \(q\) |
| \(S\) ignores some modality | one `L_dec^m` high | per-view `L_dec` must all be active; monitor per-m |
| \(P\) predicts subtypes | high `probe_q_from_P` | ↑ `λ_leak` |
| \(P\) empty | low `L_view` accuracy | ↑ `λ_view` / `λ_res` |

### 4.9 Package layout (planned)

```text
code/method4_invariant_cluster/
  __init__.py
  model.py               # Enc, Dec residual, h, view_head, Adv/GRL, DEC centroids
  losses.py
  training.py            # warm-up schedule; one optimizer + GRL
  hyperparam_tuning.py
  evaluate.py
run_method4.slurm
```

### 4.10 Extra metrics

```text
mean_pairwise_JS(q_m, q_n)     # agreement quality
entropy_q                      # collapse detector
view_id_acc_from_P
probe_q_from_P_acc
l_dec, l_agree, l_res, l_view, l_leak, …
```

---

## 5. Preferred v1 ship: Method 3 shared + Method 4 private (hybrid)

### 5.1 Why hybrid

- Method 3 supplies a **clustering-shaped shared target** (best NMI odds).
- Method 4 private block supplies **disjoint \(P\)** without full \(P\to X\).
- Standalone Method 4 can wait; its `L_dec`/`L_agree` can later replace or sit beside `L_spec`.

### 5.2 Hybrid loss (implementation default)

```text
L = λ_spec · L_spec
  + λ_res  · L_res          # Dec(P, sg(S)) → X
  + λ_view · L_view         # modality ID from P
  + λ_leak · L_leak_grl     # P ↛ cluster(S̄)
  + λ_ctr  · L_ctr          # optional, small
  + λ_vic  · L_vic          # optional
```

Optional later:

```text
+ λ_agree · L_agree(q(S_1), q(S_2), q(S_3))   # soft labels from h(S_m)
```

### 5.3 Implementation order

1. `graph_spectral.py` + sanity `nmi_spectral_U` on BRCA.  
2. Method 3 model + `L_spec` + `L_res` only (no leak/view). Smoke train 1 config.  
3. Add `L_view` + `L_leak` (Method 4 private).  
4. Small sweep (36 configs).  
5. Full eval vs MOCSS / VICReg+`b_m` / OMI-ortho / OMI no-ortho.  
6. Only then: standalone Method 4 package.

---

## 6. Comparison to existing methods

| | M1 / VICReg | M2 | M3 spectral | M4 invariant | Hybrid 3+4 |
| - | ----------- | -- | ----------- | ------------ | ---------- |
| Shared defined by | predictability of \(X\) | joint recon + ctr | fused spectrum / graph | invariant \(q\) | spectrum |
| Private defined by | `b` / adv / VICReg | VICReg + joint | residual (+ leak) | residual + view + leak | same as M4 priv |
| Solo \(P\to X\) | yes (`b_m`) | via joint | **no** | **no** | **no** |
| Two-step adv | yes | no | GRL optional | GRL | GRL |
| Proxy≈KMeans | weak | weak | **strong** | medium | **strong** |
| Expected NMI vs no-ortho 0.50 | unlikely | unlikely | possible / match | unlikely | best bet of set |
| Expected split quality | medium | weak | medium | **strong** | **strong** |

---

## 7. Explicit non-goals (v1)

- Do **not** add Method 1’s full `h/f/g/k/b` stack into M3/M4.  
- Do **not** use subtype labels in graph construction, DEC, or losses.  
- Do **not** rank HPO by NMI.  
- Do **not** treat beating OMIDIENT no-ortho as a merge gate.  
- Do **not** implement full IRM gradient-penalty until DEC + agree works.  
- Do **not** make solo `Dec(P)→X` the main private loss.

---

## 8. Evaluation protocol (same suite + additions)

Use existing `code/shared_rep_metrics.py` for all methods. Additionally report:

1. **`nmi_spectral_U`** (Method 3 / hybrid) — unsupervised ceiling of the target.  
2. **`gap S−PS`**, **`concat(P)` NMI**, own-view Ridge panel (already in README).  
3. **`view_id_acc_from_P`**, **`probe_q_from_P_acc`**.  
4. Multi-seed later via `code/multiseed/` once a winner config exists.

Fair comparison tables (to add to `docs/README.md` after runs):

- AE-private panel: MOCSS, M1+`b_m`, VICReg+`b_m`, M2, **M3 hybrid**.  
- OMIDIENT family panel: OMI no-ortho, OMI ortho, M2+OMI, **M3 hybrid** (and optional Dirichlet-P variant later).

---

## 9. Risks and open implementation choices

| Topic | v1 default | Ablation later |
| ----- | ---------- | -------------- |
| Graph fusion | mean of affinities | ≥2-view consensus edges |
| Spectral on train-only vs all unlabeled | **all unlabeled features** | train-only + Nyström |
| `S̄` fusion | mean | concat MLP |
| Leak mechanism | GRL | two-step adv (Method 1 style) |
| Cluster target for leak | `h(S̄)` soft / DEC \(q\) | KMeans labels refreshed every \(T\) epochs |
| Neighborhood InfoNCE | off | on if `L_spec` weak |
| Dirichlet private | off (AE private) | Method 2+OMI style mix-in |

---

## 10. References (inspiration)

- Spectral clustering / SpectralNet-style latent maps  
- DEC / IDEC / VaDE (deep clustering)  
- SwAV / DeepCluster / SCAN (prototype & neighbor pseudo-labels)  
- VICReg (variance–invariance–covariance; already in-repo)  
- IRM / domain-invariant representation learning  
- Multi-omics graph integration (e.g. GLUE-style graph coupling — conceptual only)  
- In-repo: MOCSS, OMIDIENT, Methods 1–2 (`shared-representation-method.md`)

---

## 11. ADR outcome checklist

Before calling Method 3 “implemented”:

- [x] `graph_spectral.py` produces `U` with logged `nmi_spectral_U`  
- [x] Training runs with `L_spec + L_res`  
- [x] Hybrid private: `L_view + L_leak`  
- [x] `evaluate.py` + leaderboard metrics (README tables after Triton runs)  
- [x] Slurm sweep (`run_method3.slurm`) + top-10 by `eval_loss`  
- [ ] Short analysis paragraph added to `docs/README.md` (after results)  
- [x] Update code location table in `shared-representation-method.md`

Before calling Method 4 “implemented”:

- [x] DEC warm-up schedule in `hyperparam_tuning.py`  
- [x] `L_agree` / `entropy_q_bar` logged in evaluate  
- [x] Residual private (no solo `P→X`)  
- [ ] README row after Triton sweep  

---

## 12. Summary decision

**We will implement clustering-aligned shared learning.**  

- **Method 3** replaces recon-as-shared-definition with **fused spectral / graph geometry**.  
- **Method 4** defines shared as an **invariant cross-omics clustering rule**, with private as **residual + view-ID − leak**.  
- **v1 ship** is the hybrid: Method 3 shared + Method 4 private.  
- Success is primarily a **clean \(S/P\) split** with competitive NMI vs MOCSS / OMIDIENT-with-ortho; beating no-ortho ~0.50 is optional upside.

---

## 13. Implementation status + Triton runbook

**Code (implemented):**

| Method | Package | Slurm |
| ------ | ------- | ----- |
| Method 3 (hybrid) | `code/method3_spectral_graph/` | `run_method3.slurm` |
| Method 4 | `code/method4_invariant_cluster/` | `run_method4.slurm` |

**Results dirs:**

- `results/models_brca_method3/`
- `results/models_brca_method4/`
- `results/spectral_targets/` (Method 3 fused `U` + kmeans labels)

### Sync code to Triton

From your laptop (adjust paths if needed):

```bash
# Example: rsync local repo into the githubmultiomics tree on scratch
rsync -av --exclude '.git' --exclude 'results' --exclude 'logs' \
  /Users/rezasoumi/Projects/multiomics/ \
  triton_via_kosh:/scratch/work/soumir1/githubmultiomics/multiomics/
```

Or commit + `git pull` on Triton inside `${SCRATCH}/githubmultiomics/multiomics`.

### Submit Method 3 (recommended first)

```bash
ssh triton_via_kosh
cd /scratch/work/soumir1/githubmultiomics/multiomics
mkdir -p logs results
sbatch run_method3.slurm
squeue -u $USER
tail -f logs/method3_spectral_<jobid>.out
```

Coarse grid = **36** configs. Full grid:

```bash
# edit run_method3.slurm: --grid-search full
# or interactive / override:
cd code/method3_spectral_graph
python -u hyperparam_tuning.py --epochs 100 --batch-size 32 --grid-search full --resume
```

Recompute spectral targets only if you change `k_nn` / PCA / `r`:

```bash
python -u hyperparam_tuning.py --recompute-spectral --epochs 100 --batch-size 32 --grid-search coarse --resume
```

### Submit Method 4

```bash
cd /scratch/work/soumir1/githubmultiomics/multiomics
sbatch run_method4.slurm
tail -f logs/method4_invariant_<jobid>.out
```

Coarse grid = **81** configs (15-epoch contrastive warm-up, then DEC).

### Smoke test (1 config, before full sweep)

```bash
cd /scratch/work/soumir1/githubmultiomics/multiomics/code/method3_spectral_graph
export PYTHONPATH=/scratch/work/soumir1/githubmultiomics:/scratch/work/soumir1
source /scratch/work/soumir1/multiomics/venv/bin/activate
# Temporarily shrink grids in a one-off python, or run and kill after first config appears under results/
python -u -c "
from multiomics.code.method3_spectral_graph import hyperparam_tuning as h
h.LAMBDA_SPEC_GRID=[1.0]; h.LAMBDA_RES_GRID=[0.5]; h.LAMBDA_CTR_GRID=[0.3]
class A: pass
a=A(); a.epochs=5; a.batch_size=32; a.resume=False; a.retrain=True
a.summarize_only=False; a.recompute_spectral=False; a.grid_search='coarse'; a.rank_metric='eval_loss'
h.run_sweep(a)
"
```

### What to check after a run

1. `results/spectral_targets/brca_meta.json` + printed `nmi_spectral_U` (ceiling).  
2. `results/models_brca_method3/leaderboard.csv` / `best_config.json`.  
3. Compare `s_nmi`, `p_nmi`, `probe_*`, `align_cka_mean` to MOCSS / VICReg+`b_m` / OMI-ortho in `docs/README.md`.  
4. Method 4: watch `entropy_q_bar` (collapse if near 0) and `agree_symkl_mean`.
