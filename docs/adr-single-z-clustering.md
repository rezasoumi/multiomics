# ADR: Single-Embedding Multi-Omics Clustering (No Shared / Private Split)

> **Status:** Accepted for implementation (design freeze for v1)  
> **Date:** 2026-10-23  
> **Related:** [`shared-representation-method.md`](./shared-representation-method.md), [`README.md`](./README.md), [`adr-clustering-aligned-shared.md`](./adr-clustering-aligned-shared.md)  
> **Note on math:** plain text so it renders in any Markdown preview.

This ADR defines a **separate experimental track** from Methods 1–4. Here we **ignore** the shared/private (\(S\)/\(P\)) decomposition and ask only:

> Can a **single** patient embedding \(Z\) beat unsupervised subtyping benchmarks (esp. OMIDIENT no-ortho ~0.50 NMI on BRCA)?

---

## 1. Context

### 1.1 What OMIDIENT (no \(L_\mathrm{ortho}\)) really is

From `docs/README.md`:

- Reported BRCA checkpoint `0.0003_0.0005_4` was trained **without** `L_ortho` in `loss_total`.
- Result: `concat(P) ≈ 0.05`, `mean(S) ≈ concat(P,S) ≈ 0.50`.
- Interpretation: \(P\) and \(S\) were **uncoupled**; private carries almost no subtype signal. The strong number is effectively a **multi-view contrastive autoencoder** on the shared path — not a verified shared/private theory win.

So the clustering bar (~0.50 NMI) is already a **single-code** (or single-path) result. Methods 3–4 tried to improve clustering *while* keeping an \(S\)/\(P\) story; spectral ceiling `nmi_spectral_U ≈ 0.39` limited purified shared NMI. This track drops that constraint.

### 1.2 Relation to the thesis

| Track | Goal | Docs / code |
| ----- | ---- | ----------- |
| **A — Decomposition** (Methods 1–4) | True \(S\)/\(P\) split; competitive NMI is secondary | `adr-clustering-aligned-shared.md` |
| **B — Clustering** (this ADR) | Maximize subtype clustering with one \(Z\); no \(S\)/\(P\) | this file |

Both can appear in the thesis: Track B answers “what is the unsupervised multi-omics NMI ceiling with modern deep clustering?”; Track A answers “can we get a clean shared/private split?”

### 1.3 Decision

Implement four items in this **priority order**:

| Priority | ID | Name | Package (planned) | Role |
| -------- | -- | ---- | ----------------- | ---- |
| 1 | **A** | Contrastive + prototype / DEC on fused \(Z\) | `code/cluster_a_contrastive_dec/` | Primary neural upgrade of no-ortho |
| 2 | **D** | Consensus clustering baseline | `code/cluster_d_consensus/` | Non-neural bar (must know) |
| 3 | **B** | SCAN-style neighbor bootstrapping | `code/cluster_b_scan/` | Second stage on A’s checkpoint |
| 4 | **C** | Patient-graph GNN | `code/cluster_c_gnn/` | Only if A stalls and graph can improve |

**Gate:** Implement A and D first (can run in parallel). Start B after a trained A checkpoint exists. Start C only if A’s best concat/mean NMI is clearly below D **and** below no-ortho ~0.50, *or* if A matches no-ortho but D is higher (then GNN is an attempt to beat the consensus bar).

---

## 2. Success criteria

**Primary KPI (BRCA, same split / KMeans `n_init=30` as existing eval):**

| Metric | Fair bar | Aspirational |
| ------ | -------- | ------------ |
| NMI of KMeans on \(Z\) (or prototype argmax) | ≥ OMIDIENT-with-ortho / MOCSS (~0.45) | ≥ OMIDIENT no-ortho (~0.50) |
| ARI / Acc / kNN | report alongside NMI | — |

**Secondary:**

- Compare to **D** (consensus). Neural methods must beat D to claim a deep-learning win.
- Stability: multi-seed later via `code/multiseed/` once a winner exists.
- Label-free HPO: do **not** rank sweeps by NMI; use a constructive val loss / clustering surrogate (see each method). Optionally log NMI for analysis only.

**Non-goals for this track:**

- No private encoders, no VICReg \(P\!\perp\!S\), no Method-1 adversaries.
- No claim that \(Z\) is a “purified shared factor.”
- Do not change Methods 1–4 defaults (v1 remains the \(S\)/\(P\) story).

---

## 3. Shared plumbing

Reuse where possible:

- Data: `load_data(disease)` (BRCA first), same 60/20/20 split `random_state=1` as Methods 1–2 for fair test-set comparison when reporting neural methods.
- Encoders: 4-layer MLP `tanh`, widths aligned with MOCSS (`[512,256,128,32]` per view; view3 may use `[256,128,64,32]` as elsewhere).
- Contrastive: MOCSS `InstanceLoss` on projected embeddings (temperature 0.4).
- Results under `results/models_brca_cluster_{a,b,c,d}/`.
- Slurm: `run_cluster_a.slurm`, `run_cluster_d.slurm`, … (A100, same env as other jobs).
- Eval script: clustering metrics on \(Z\) (NMI/ARI/Acc/kNN). Reuse pieces of `shared_rep_metrics` where helpful, but **no** \(S\)/\(P\) probe panel is required for the main table (optional: run probes on \(Z\) as “single code”).

**Fusion (default for A/B/C):**

```text
h_m = Enc_m(X_m)           # m ∈ {1,2,3}
Z   = mean(h_1, h_2, h_3)  # v1; ablation: concat→MLP or attention
```

---

## 4. Method A — Contrastive + prototype / DEC on fused \(Z\)

### 4.1 One-sentence summary

Train a multi-view contrastive AE (no-ortho style) and add an explicit **cluster / prototype** head on fused \(Z\) so training targets subtype geometry, not only instance alignment.

### 4.2 Architecture

```text
X_m → Enc_m → h_m ∈ R^{32}
Z = mean(h_1, h_2, h_3)

Instance path:
  p_m = MLP_proj(h_m) → InfoNCE across modality pairs

Cluster path (choose one for v1; other as ablation):
  (DEC)  soft q = Student-t(Z; centroids μ_k), K_c = 5 (or swept)
  (SwAV) prototypes C ∈ R^{K×d}, Sinkhorn codes, swapped prediction

Optional light recon:
  Dec_m(Z) → X̂_m   or   Dec(Z) → concat(X)
```

**No private branch.**

### 4.3 Losses (v1 default = DEC)

```text
L_ctr = InfoNCE(p_1,p_2) + InfoNCE(p_1,p_3) + InfoNCE(p_2,p_3)

# DEC
q_ik ∝ (1 + ‖Z_i − μ_k‖² / α)^{−(α+1)/2}
p_ik ∝ q_ik^sharpen / Σ_i q_ik          # sharpen ∈ {2.0, 2.5}
L_dec = KL(p ‖ q)

L_rec = Σ_m recon_MOCSS(Dec_m(Z), X_m)   # optional, small λ

L = λ_ctr·L_ctr + λ_dec·L_dec + λ_rec·L_rec
```

**Warm-up:** epochs `0 … W−1` with `λ_dec = 0` (contrastive ± recon only); at epoch \(W\) init centroids via KMeans on train \(Z\); then full loss. Default `W = 20`.

### 4.4 Checkpoint / rank

```text
val_score = λ_ctr·L_ctr + λ_dec·L_dec + λ_rec·L_rec   # all ≥ 0
eval_loss = L_ctr + L_dec                             # rank; L_rec excluded
```

Also save **prototype/hard assignment NMI** at eval (argmax \(q\)) in addition to KMeans on \(Z\).

### 4.5 Hyperparameter grids (v1)

```text
Fixed: lr/wd = MOCSS BRCA (0.0004 / 0.0007), batch=32, epochs=100,
       K_c=5, α=1.0, sharpen=2.5, W=20, seed=21

Coarse sweep:
  λ_ctr ∈ {0.5, 1.0, 1.5, 2.0}
  λ_dec ∈ {0.5, 1.0, 1.5, 2.0}
  λ_rec ∈ {0.0, 0.3, 0.7}
→ 4×4×3 = 48 configs

Ablation later: SwAV instead of DEC; attention fusion; K_c ∈ {4,5,6}
```

Config token: `{lr}_{wd}_c{λ_ctr}_d{λ_dec}_r{λ_rec}`

### 4.6 Collapse modes

| Failure | Symptom | Mitigation |
| ------- | ------- | ---------- |
| One-cluster | entropy(\(q\)) → 0 | sharpen + balance; longer warm-up |
| Ignores DEC | high `L_dec`, NMI≈contrastive-only | ↑ `λ_dec` after warm-up |
| Over-recon | good recon, flat clusters | keep `λ_rec` small / 0 |

### 4.7 Package layout

```text
code/cluster_a_contrastive_dec/
  __init__.py
  model.py
  training.py
  hyperparam_tuning.py
  evaluate.py
run_cluster_a.slurm
```

---

## 5. Method D — Consensus clustering baseline

### 5.1 One-sentence summary

Non-neural multi-omics consensus clustering to establish the **classical bar** neural methods must beat.

### 5.2 Procedure (v1)

```text
For each modality m (and optional fused feature concat):
  1. PCA to d_pca (default 50)
  2. Row L2-normalize
  3. KMeans (K_c=5, n_init=30) → labels ℓ_m
  Optional: also spectral clustering on per-view / fused kNN graphs

Build co-association matrix S ∈ [0,1]^{N×N}:
  S_ij = (# of base clusterings where i and j co-cluster) / (# base clusterings)

Final:
  KMeans or spectral clustering on S (or on eigenvectors of S) → ℓ_final
```

**Base clusterings (v1 set):**

1. KMeans on PCA(view1)  
2. KMeans on PCA(view2)  
3. KMeans on PCA(view3)  
4. KMeans on PCA(concat of three views)  
5. (optional) KMeans on fused spectral \(U\) from Method 3’s `graph_spectral` (reuse if present)

No labels used in construction. Report metrics vs subtypes on **all** samples and, for fair compare to neural test metrics, also on the same test indices if desired (document which).

### 5.3 “Hyperparameters”

Small grid, no GPU required:

```text
d_pca ∈ {30, 50, 100}
K_c ∈ {4, 5, 6}          # final K; base K same as final for v1
graph_k ∈ {10, 15, 20}   # if spectral bases enabled
```

Rank by — for baselines only — we **may** report NMI directly (no learned val loss). Still freeze a default `(d_pca=50, K_c=5)` as the headline before cherry-picking.

### 5.4 Package layout

```text
code/cluster_d_consensus/
  __init__.py
  consensus.py           # co-association + final cluster
  run_baseline.py        # CLI: disease, grid, save metrics json/csv
run_cluster_d.slurm      # CPU partition OK (no GPU needed)
```

Results: `results/models_brca_cluster_d/leaderboard.csv`

---

## 6. Method B — SCAN-style neighbor bootstrapping

### 6.1 One-sentence summary

Second stage on a frozen or fine-tuned **Method A** checkpoint: mine must-link neighbors and train cluster consistency.

### 6.2 Procedure

```text
Stage 1: Train Method A → embedding Z (train+val+test encode once)

Stage 2a — Mine neighbors (label-free):
  For each i:
    N_emb(i)  = kNN in Z-space (cosine, k=10)
    N_raw(i)  = patients that are kNN of i in ≥2 omics (PCA-cosine)
    N(i)      = N_emb(i) ∩ N_raw(i)   # if empty, fall back to N_emb only

Stage 2b — Train cluster head g(Z) → q ∈ Δ^{K−1}:
  L_scan = mean_{j∈N(i)} CE(q_i, stopgrad(q_j))   # or KL
         + λ_ent · (−entropy of mean q)             # balance
         + λ_sharp · (−entropy of q_i)              # confidence
  Optional: continue small λ_ctr
```

Default: **fine-tune** encoders + head with small LR; ablation: freeze encoders, train head only.

### 6.3 Inputs / outputs

- Requires: `results/models_brca_cluster_a/<best_config>/model_*`
- Writes: `results/models_brca_cluster_b/<parentA>_<scan_hparams>/`

### 6.4 Grids (v1)

```text
k_emb ∈ {5, 10, 15}
λ_scan = 1.0 (fixed), λ_ent ∈ {1.0, 2.0, 5.0}, lr_ft ∈ {1e-4, 3e-4}
epochs_ft = 50
```

### 6.5 Package layout

```text
code/cluster_b_scan/
  __init__.py
  mine_neighbors.py
  model_head.py          # or reuse A’s DEC head
  training.py
  hyperparam_tuning.py
  evaluate.py
run_cluster_b.slurm      # takes --parent-config from A
```

---

## 7. Method C — Patient-graph GNN

### 7.1 One-sentence summary

Learn \(Z\) with a GNN on a multi-omics patient graph + clustering loss, instead of regressing fixed Laplacian eigenvectors (Method 3’s weak \(U\)).

### 7.2 When to run

Only if:

- Method A best NMI **<** Method D headline, or  
- Method A ≈ no-ortho (~0.50) but **D > A**, or  
- A clearly plateaus below 0.50 after B.

### 7.3 Architecture (v1)

```text
Node features: concat(PCA(X_1), PCA(X_2), PCA(X_3)) ∈ R^{3·d_pca}
              or learned Enc_m then concat (heavier)

Graph: consensus multi-omics kNN (reuse Method 3 graph_spectral fusion;
       edges if neighbor in ≥2 views; k ∈ {10,15,20})

GNN: 2-layer GCN or GAT → Z_i ∈ R^{32}
L = λ_dec·L_dec(Z) + λ_edge·edge_prediction (optional)
    + λ_ctr·InfoNCE on Z for same-patient? (N/A — one node per patient)
```

Self-supervision options: DEC/SwAV on \(Z\); or predict masked node features.

### 7.4 Package layout

```text
code/cluster_c_gnn/
  __init__.py
  graph_build.py         # thin wrapper on method3 graph_spectral or local copy
  model.py               # GCN/GAT
  training.py
  hyperparam_tuning.py
  evaluate.py
run_cluster_c.slurm
```

Dependency: PyTorch Geometric **or** a minimal pure-PyTorch GCN (prefer minimal to avoid env pain on Triton). **v1 decision:** implement a **2-layer GCN in pure PyTorch** (sparse `index_add_` message passing) so `scicomp-pytorch-env` + venv stay unchanged.

---

## 8. Evaluation protocol

For neural methods (A/B/C), on the **held-out test split** (same as Methods 1–2):

1. Encode → \(Z\)  
2. KMeans(`n_clusters=NUM_CLUST[disease]`, `n_init=30`) → NMI, ARI, Acc, kNN  
3. If DEC/SwAV: also metrics from **argmax soft assignment** (no KMeans)  
4. Log embedding dim, entropy of \(q\), and for A: contrastive-only warmup checkpoint NMI (ablation)

For D: report on full cohort (standard for consensus) **and** note N; optionally restrict to test IDs for a side-by-side column.

**Leaderboard columns (minimal):**

```text
config, nmi, ari, acc, knn_acc, nmi_assign (if any), eval_loss (neural), method
```

Compare in README under a new section: **Single-\(Z\) clustering track** vs OMIDIENT no-ortho / with-ortho / MOCSS concat NMI.

---

## 9. Implementation order (checklist)

1. [x] **D** — `cluster_d_consensus` CPU baseline + CSV (`run_cluster_d.slurm`)  
2. [x] **A** — `cluster_a_contrastive_dec` + Slurm coarse 48 + evaluate (`run_cluster_a.slurm`)  
3. [ ] README row: A vs D vs OMI no-ortho *(fill after Triton runs)*  
4. [x] **B** — SCAN stage on A’s eval_loss winner (`run_cluster_b.slurm`)  
5. [x] **C** — GNN implemented (`run_cluster_c.slurm`); run only if gated by A/D results  
6. [ ] Multi-seed on the final winner only  

---

## 10. Comparison to existing methods

| | MOCSS / OMI shared path | Methods 3–4 | This track (A/B/C/D) |
| - | ----------------------- | ----------- | -------------------- |
| Output | \(S\), \(P\) | \(S\), \(P\) | single \(Z\) |
| Private | yes | yes | **no** |
| Clustering in train | indirect (ctr/recon) | spectral / DEC on \(S\) | **explicit on \(Z\)** |
| Goal | integration + split | split + probes | **max NMI** |

Method A is the closest “honest” upgrade of OMIDIENT no-ortho: same multi-view contrastive idea, plus a cluster head, without a fake private path.

---

## 11. Risks

| Risk | Mitigation |
| ---- | ---------- |
| Ranking by NMI during HPO (label leakage into selection) | Rank by `eval_loss`; report NMI after |
| DEC collapse | Warm-up + sharpen + entropy monitoring |
| D already ≥ 0.50 | Thesis reports classical win; neural must beat D |
| GNN env / complexity | Pure PyTorch GCN; run only if gated |
| Split mismatch vs D full-cohort | Document evaluation cohort clearly |

---

## 12. References (inspiration)

- SwAV; SCAN; Contrastive Clustering (Li et al.); DEC / IDEC / VaDE  
- Consensus clustering / co-association (Monti et al.; multi-omics CSPA-style)  
- SpectralNet / GCN node clustering  
- In-repo: MOCSS contrastive, OMIDIENT no-ortho analysis, Method 3 `graph_spectral.py`

---

## 13. ADR outcome

**We will implement a single-\(Z\) clustering track** with priority **A → D (parallel) → B → (gated) C**, aiming to beat or contextualize OMIDIENT no-ortho ~0.50 NMI without any shared/private split. Methods 1–4 remain the decomposition track; this ADR does not replace them.
