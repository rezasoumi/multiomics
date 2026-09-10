# Shared Representation Learning: Problem Statement and Proposed Methods

> **Note on math in this file:** Equations are written in plain text so they render in any Markdown preview.

This document summarizes the limitations of the current OMIDIENT/MOCSS shared-representation pipeline in this repository, and compares the implemented extensions for the thesis goal of learning **truly shared** latent codes across multi-omics modalities.

**Code locations:**

| Method | Package |
| ------ | ------- |
| Method 1 (full predictability) | `code/method1_predictability/` |
| Method 1 ablation (no `k` / `L_own` + VICReg) | `code/method1_vicreg_ablation/` |
| Method 2 (joint recon + VICReg) | `code/method2_joint_vicreg/` |

**Context:** OMIDIENT (Negar et al.) improves multi-omics integration by incorporating Dirichlet distributions into the **private/specific** reconstruction networks. The **shared** branch still follows MOCSS. This thesis aims to extend OMIDIENT by improving the shared representation learning component—potentially via Dirichlet-based VAEs or other methods.

---



## The Core Problem

In MOCSS/OMIDIENT, each modality `m ∈ {1, 2, 3}` is split into:

- **S_m** — intended shared latent code
- **P_m** — intended private/specific latent code

The current pipeline (see `code/ae/mocss_original_refactored.py` and `code/prod_gamma_dirvae/prod_gamma_dirvae_cancer.py`) does the following:

1. **Orthogonality** between `S_m` and `P_m` per modality
2. **Reconstruction** where **both** `S_m` and `P_m` independently reconstruct the **full** input `X_m`
3. **Contrastive loss** on MLP-projected shared embeddings across modalities



### Why This Fails to Guarantee "Pure" Shared Representations


| Force                                        | Effect on S_m                                                                               |
| -------------------------------------------- | ------------------------------------------------------------------------------------------- |
| Shared reconstruction: X̂_m^(S) = Dec_S(S_m) | Pushes S_m to encode anything needed to reconstruct X_m, including modality-specific detail |
| Contrastive on projected S_m                 | Pushes S_m to align across modalities                                                       |
| Orthogonality vs P_m                         | Only weakly discourages linear overlap                                                      |


**Orthogonality in the current code is not true orthogonality.** It is the mean element-wise product of L2-normalized vectors (see `orthogonal_loss` in `mocss_original_refactored.py`). That penalizes per-dimension co-activation, not subspace orthogonality or statistical independence. Nonlinear redundancy can remain, and S_m can still carry private information if it helps reconstruction.

OMIDIENT adds Dirichlet structure only to the **private** branch (`prodDirVae.variational_encoder_decoder` for specific; plain AE for shared). The shared branch remains unchanged MOCSS—making improved shared representation learning a well-motivated thesis direction.

---



## Shared implementation conventions (all three methods)

These details match the code and are shared unless a method section says otherwise.

### Reconstruction metric (MOCSS-style)

Not literal ‖·‖². For every reconstruction term (`h`, `f`, `g`, `k`, or joint `D`):

```text
1. Center:  rec ← rec − mean(rec);  ori ← ori − mean(ori)
2. Row L2-normalize both
3. Loss = ‖ rec − ori ‖_matrix   (torch.linalg.matrix_norm)
```

Used for fair comparison with MOCSS/OMIDIENT.

### Contrastive term

Per batch, after MLP projection of each `S_m`:

```text
L_ctr = InfoNCE(MLP(S_1), MLP(S_2))
      + InfoNCE(MLP(S_1), MLP(S_3))
      + InfoNCE(MLP(S_2), MLP(S_3))
```

Each pair is centered and L2-normalized before `InstanceLoss` (temperature as in MOCSS). Default sweep weight: `λ_ctr = 1.0` (Method 2 also sweeps `λ_ctr`).

### VICReg-style P ⊥ S independence (`vicreg_ps_independence`)

Used by **Method 1 ablation** and **Method 2**. For one modality, with batch-centered `P` and `S` (`eps = 1e-4`):

```text
σ_P = sqrt( Var_batch(P) + eps )     # per latent dim
σ_S = sqrt( Var_batch(S) + eps )

var_loss = mean( ReLU(1 − σ_P) ) + mean( ReLU(1 − σ_S) )
           # two separate hinges, then summed (not pooled into one mean)

Cov_P = (Pᵀ P) / (B−1);   Cov_S = (Sᵀ S) / (B−1)
cov_loss = mean( offdiag(Cov_P)² ) + mean( offdiag(Cov_S)² )

cross = (Pᵀ S) / (B−1)
cross_loss = mean( cross² )          # ≈ ‖Cov(P,S)‖_F² / (#entries)

L_vic^m = var_loss + cov_loss + cross_loss
L_vic   = Σ_m L_vic^m
```

| Sub-term | Role |
| -------- | ---- |
| `var_loss` on P and on S | Anti-collapse: each dim’s std should reach ≥ 1 |
| within off-diag `cov_loss` | Whitening / reduce redundant dims inside P or inside S |
| `cross_loss` | Main **P ⊥ S** (linear) independence term |

**Notes from code:**

- Target std `1.0` is hardcoded in the hinge.
- `var_loss` already adds **separate** means for P and S. If S collapses and P does not, the S hinge still increases `var_loss`. For **logging**, the current code only records total `l_vic` (var + cov + cross over all modalities), so S-vs-P variance is not visible in CSV logs unless split later.
- Only a single outer weight `λ_vic` scales the whole sum; there are no separate `λ_var_P` / `λ_var_S` / `λ_cross` yet.
- MOCSS `orthogonal_loss` is **not** used alongside this term.

### Network plumbing (MOCSS-aligned)

- Encoders: 4-layer MLP with `tanh`, widths e.g. `[512, 256, 128, 32]` (emb dim = last unit).
- Decoders / predictors: 4-layer MLP, `tanh` then `sigmoid` on the last layer.
- Shared MLP projection: 2-layer `tanh` head on `S_m` before contrastive.
- Init: Kaiming normal on Linear weights; zero biases.
- Typical sweep: BRCA, batch 32, 100 epochs, early stop patience 20 / `min_delta=0.005`, seed 21, MOCSS default `lr` / `weight_decay` for the disease.
- Results under repo-root `results/` (e.g. `results/models_brca_method1/`), not under `code/results/`.
- Ranking default: **NMI** (also log ARI, ACC, kNN).

---



## Method 1: Cross-Reconstruction + Predictability Surrogate

**Code:** `code/method1_predictability/`

### Architecture

```
X_m → Enc → P_m, S_m   (per modality)

Predictors for target modality m (separate networks h, f, g, k):
  h_m(P_m, S_m)       → X̂_m   (should predict)       # within-view recon
  f_m(S_m', S_m'')    → X̂_m   (should predict)       # cross-shared sufficiency
  g_m(P_m', P_m'')    → X̂_m   (should NOT predict)   # cross-private adversary
  k_m(S_m)            → X̂_m   (should NOT predict)   # own-shared insufficiency

Keep shared embedding + MLP projection + contrastive (per modality):
  S_1, S_2, S_3 → MLP → contrastive loss across the 3 projected rows
```

The key change vs MOCSS: drop solo `shared_rec` / `specific_rec`. Instead, each modality is explained by four predictability constraints—joint within-view reconstruction (`h`), cross-view shared sufficiency (`f`), cross-view private privacy (`g`), and own-shared insufficiency (`k`).

**No VICReg and no MOCSS orthogonality** in this variant.

### Predictability Surrogate

For target modality `A`, define four predictors:


| Network | Input      | Target | Role                                                                                                                                                     |
| ------- | ---------- | ------ | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| h_A     | (P_A, S_A) | X_A    | Standard within-view reconstruction                                                                                                                      |
| f_A     | (S_B, S_C) | X_A    | **Shared sufficiency**: other views' shared codes should predict A (how strongly is controlled by α)                                                     |
| g_A     | (P_B, P_C) | X_A    | **Private adversary**: other views' private codes should **not** predict A                                                                               |
| k_A     | S_A        | X_A    | **Own-shared insufficiency**: own shared code alone should **not** fully reconstruct A (private residual must live in P_A so that h_A can still succeed) |


Together: shared content is pushed into `S` and out of foreign `P` (`f` + `g`), while private-to-A content is discouraged from sitting in `S_A` alone (`k`) and recovered via `P_A` in `h_A`.

### Loss Terms

Each of `L_std`, `L_shared`, `L_adv`, `L_own` is the sum over modalities `A ∈ {1,2,3}` of the MOCSS-style reconstruction metric above.

```text
L_std^A    = recon( h_A(P_A, S_A), X_A )
L_shared^A = recon( f_A(S_B, S_C), X_A )
L_adv^A    = recon( g_A(P_B, P_C), X_A )
L_own^A    = recon( k_A(S_A), X_A )
```

Do **not** optimize a single combined scalar with one joint `backward()` on adversaries. A naive `L = … − β·L_adv − γ·L_own` would also update `g`/`k` to increase their own error. Use the two-step procedure below.

### Two-Step Training (per batch)

Implemented with **two optimizers** (`optimizer_adv` for `g`/`k`, `optimizer_main` for encoders / `h` / `f` / MLP).

**Step 1 — train adversaries only** (`g` and `k`; codes `P`/`S` **detached** so encoders get no gradient):

```text
L_step1 = Σ_A [ L_adv^A + L_own^A ]
```

**Step 2 — train encoders and constructive predictors** (`g`/`k` parameters frozen via `requires_grad=False`; fresh encode so the graph is clean):

```text
L_step2 = Σ_A [ L_std^A + α·L_shared^A − β·L_adv^A − γ·L_own^A ] + λ_ctr·L_ctr
```

Interpretation of the minus signs in step 2: encoders try to make `(P_B, P_C)` uninformative about `X_A` and make `S_A` alone insufficient for full `X_A`, while step 1 keeps `g`/`k` competent predictors. Keep `γ` modest so `k` removes private bleed from `S` without wiping the shared signal that `f` and contrastive need.

### Checkpointing (important)

Signed `L_step2` can go **negative** and is a bad early-stop / checkpoint signal (high-γ runs often saved near epoch 1). The code therefore checkpoints and early-stops on a **constructive** validation score:

```text
constructive = L_std + α·L_shared + λ_ctr·L_ctr
```

(no `−β L_adv` / `−γ L_own`). Logged terms: `l_std`, `l_shared`, `l_adv`, `l_own`, `l_ctr`.

### Hyperparameter grids (as implemented)

```text
α ∈ {0.3, 0.5, 0.7, 1.0, 1.5}
β ∈ {0.5, 0.7, 1.0, 1.5, 2.0}
γ ∈ {0.7, 1.0, 1.5, 2.0, 2.5}
λ_ctr = 1.0 (fixed)
→ 125 configs; results under results/models_{disease}_method1/
```

Optional full retrain without early stop: `retrain_config.py` (saves last-epoch and best-constructive checkpoints).

### Information-Theoretic Interpretation

- **Minimize** L_shared → maximize I(X_A; S_B, S_C) → S captures cross-view predictable content
- **Maximize** L_adv (via −β in step 2) → minimize I(X_A; P_B, P_C) → private codes of other views don't leak shared signal
- **Maximize** L_own (via −γ in step 2) → minimize sufficiency of S_A alone for full X_A → private residual is pushed into P_A
- **Minimize** L_std → (P_A, S_A) remain jointly sufficient for X_A
- **Cross-view shared decoder** `f` forces S to be useful *collectively*, not a per-view full copy of X_m

### Advantages

1. **Operational definition of "shared"**: what is predictable from other modalities' S, not what reconstructs locally.
2. **Explicit anti-leakage** for private codes across views via `g`.
3. **Explicit own-shared insufficiency** via `k`: unlike MOCSS's solo `shared_rec`, own `S_A` is discouraged from fully reconstructing `X_A`.
4. **Cross-reconstruction** (`f`) reduces incentive for S_m to duplicate only local detail.
5. **Theoretically defensible** in a thesis (sufficiency, privacy, minimal shared representation).

### Concerns


| Issue                        | Detail                                                                                                                                       |
| ---------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| **12 predictors**            | 3 modalities × (h, f, g, k) = 12 networks on top of encoders—more parameters; tune α, β, γ                                                   |
| **Minimax / two-step**       | Needs the two-step procedure above; naive single backward on −L_adv/−L_own trains adversaries incorrectly; β, γ can still oscillate |
| **Over-strong k**            | Large γ can strip shared content from S_A, not only private residual—monitor L_own vs L_shared and keep γ modest                             |
| **Weak f or g**              | If f is weak, shared may underfit; if g is weak, −L_adv is vacuous                                                                           |
| **Signed val loss**          | Must use constructive checkpointing; do not early-stop on raw L_step2                                                                        |
| **Contrastive / projection** | MLP projection can still hide some private dims in S; k mitigates but does not prove purity                                                  |


---



## Method 1 Ablation: Predictability without `k` + VICReg P⊥S

**Code:** `code/method1_vicreg_ablation/`

This is Method 1 with:

1. **`k` / `L_own` removed** (no own-shared insufficiency adversary).
2. **VICReg P⊥S** added per modality (same `vicreg_ps_independence` as Method 2).
3. Predictors kept: **`h`, `f`, `g` only** (9 predictors instead of 12).

### Architecture

```
X_m → Enc → P_m, S_m

h_m(P_m, S_m)     → X̂_m   minimize
f_m(S_m', S_m'')  → X̂_m   minimize (α)
g_m(P_m', P_m'')  → X̂_m   maximize via −β (adversary)
+ L_vic(P_m, S_m)           minimize (λ_vic)
+ L_ctr on MLP(S)           minimize (λ_ctr)
```

### Two-Step Training

Same pattern as Method 1, but step 1 trains **`g` only**:

```text
L_step1 = Σ_A L_adv^A

L_step2 = Σ_A [ L_std^A + α·L_shared^A − β·L_adv^A ]
        + λ_ctr·L_ctr + λ_vic·L_vic
```

### Checkpointing

Constructive (positive) validation score:

```text
constructive = L_std + α·L_shared + λ_ctr·L_ctr + λ_vic·L_vic
```

(excludes `−β L_adv`). Logged: `l_std`, `l_shared`, `l_adv`, `l_ctr`, `l_vic` (total VICReg; not split into var/cov/cross).

### Hyperparameter grids (as implemented)

```text
α ∈ {0.3, 0.5, 0.7, 1.0, 1.5}
β ∈ {0.5, 0.7, 1.0, 1.5, 2.0}
λ_vic ∈ {0.1, 0.5, 1.0, 2.0, 5.0}
λ_ctr = 1.0 (fixed)
→ 125 configs; results under results/models_{disease}_method1_vicreg/
```

### Why this ablation

- Isolates whether **cross-view predictability (`h`/`f`/`g`)** plus **within-view linear independence (VICReg)** is enough without the tricky `k` / `γ` term.
- VICReg’s variance hinges act as the day-1 anti-collapse for both P and S (see shared VICReg section).

### Concerns

| Issue | Detail |
| ----- | ------ |
| No `k` | Own `S_A` is not explicitly forced to be insufficient for full `X_A`; private bleed into S may remain if `f`/`g`/VICReg do not catch it |
| Single `λ_vic` | Weights var + within-cov + cross together; cannot upweight S variance alone without a code change |
| Still adversarial | `−β L_adv` needs two-step training and constructive checkpointing |

---



## Method 2: Joint Reconstruction + VICReg Disentanglement

**Code:** `code/method2_joint_vicreg/`

### Architecture

Method 2 does **not** use cross-reconstruction or adversaries. Each modality is reconstructed only from **its own** private and shared codes via one joint decoder:

```
X_m → Enc → P_m, S_m   (per modality)

Decoder_m:  D_m(concat(P_m, S_m)) → X̂_m   (joint within-view reconstruction)

L_vic(P_m, S_m)   (VICReg; replaces MOCSS orthogonality)
L_ctr on MLP(S)   (unchanged MOCSS-style contrastive)
```

|                 | Current MOCSS for the shared reconstruction part | Method 2                                   |
| --------------- | ------------------------------------------------ | ------------------------------------------ |
| Reconstruction  | `Dec_S(S_m) → X_m` and `Dec_P(P_m) → X_m`      | `D_m(concat(P_m, S_m)) → X_m` only         |
| Disentanglement | Weak elementwise orthogonality                   | VICReg `var + within-cov + cross-cov`      |
| Training        | (varies)                                         | **Single** optimizer; all loss terms ≥ 0   |


With a single joint decoder, S_m no longer must **alone** reconstruct X_m. It only needs to contribute the shared part together with P_m. VICReg then pushes P_m and S_m to carry non-overlapping linear information and not collapse.

### Loss (as in code)

```text
L_recon = Σ_m recon( D_m(concat(P_m, S_m)), X_m )   # MOCSS-style metric

L_vic   = Σ_m vicreg_ps_independence(P_m, S_m)       # see shared section

L = λ_rec·L_recon + λ_ctr·L_ctr + λ_vic·L_vic
```

All terms are positive → **raw validation loss is safe** for checkpointing / early stop (no constructive workaround).

Logged per epoch: `l_recon`, `l_ctr`, `l_vic` (again total VICReg, not split).

**Not used:** MOCSS `orthogonal_loss`, solo shared/specific decoders, Method 1 predictors.

### Hyperparameter grids (as implemented)

Intended full sweep (125 configs):

```text
λ_rec ∈ {0.3, 0.5, 0.7, 1.0, 1.5}
λ_ctr ∈ {0.5, 0.7, 1.0, 1.5, 2.0}
λ_vic ∈ {0.1, 0.5, 1.0, 2.0, 5.0}
```

Results under `results/models_{disease}_method2/` with folder names `{lr}_{wd}_r{}_c{}_v{}`. Rank by NMI by default. (`hyperparam_tuning.py` may temporarily shrink grids for single-config runs.)

### Advantages

1. **Simpler than Method 1** — no cross-reconstruction, no adversarial predictors; only 3 joint decoders
2. **Fixes a MOCSS flaw directly** — removes the separate `shared_rec` path that forces S_m to solo-reconstruct X_m
3. **Clear disentanglement story** — P_m and S_m partition information under a single reconstruction objective, with VICReg preventing duplication / collapse
4. **Easier to integrate** with OMIDIENT Dirichlet on P_m only; minimal architectural change from current code
5. **Stable checkpointing** — all-positive objective

### Concerns

**Collapse modes:**


| Collapse               | Mechanism                                                          |
| ---------------------- | ------------------------------------------------------------------ |
| S_m → 0                | Strong independence + joint decoder lets P_m alone satisfy L_recon |
| P_m absorbs everything | Weak contrastive or weak λ_ctr; S_m not needed for alignment       |


**Mitigations (in code today):**

1. **Variance hinges** inside `L_vic` on **both** P and S from day 1 (`mean(ReLU(1−σ))` each)
2. **Cross-covariance** term as the main P⊥S force
3. **Within off-diag cov** whitening (optional in spirit; currently always on and tied to `λ_vic`)
4. **Contrastive weight** `λ_ctr` swept so S stays aligned across views

**Optional plan B (not implemented):** light shared-only auxiliary or a small amount of Method 1-style cross-view predictability if S still collapses.

**Monitoring gap:** because `l_vic` is logged as one scalar, prefer splitting logs into `var_P`, `var_S`, `cov_within`, `cross` if diagnosing asymmetric collapse.

---



## Method comparison (quick)


| | Method 1 | Method 1 ablation | Method 2 |
| --- | --- | --- | --- |
| Joint / solo recon | `h(P,S)`; no solo MOCSS recon | same | `D(concat(P,S))` only |
| Cross-shared `f` | yes | yes | no |
| Private adversary `g` | yes | yes | no |
| Own-shared `k` | yes | **no** | no |
| VICReg P⊥S | no | yes | yes |
| MOCSS ortho | no | no | no |
| Training | 2-step, 2 opts | 2-step, 2 opts | 1-step, 1 opt |
| Checkpoint | constructive | constructive | raw val loss |
| Sweep knobs | α, β, γ | α, β, λ_vic | λ_rec, λ_ctr, λ_vic |

---



## References

- OMIDIENT paper: *OMIDIENT: Multiomics Integration for Cancer by Dirichlet Auto-Encoder Networks*
- Repository: [Zenodo record](https://zenodo.org/records/20273710)
- Baseline shared/specific model: MOCSS (Multi-Omics Contrastive Shared-Specific)
