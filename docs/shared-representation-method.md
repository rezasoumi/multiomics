# Shared Representation Learning: Problem Statement and Proposed Methods

> **Note on math in this file:** Equations are written in plain text so they render in any Markdown preview. LaTeX versions for your thesis are collected at the [end of this document](#latex-appendix-for-thesis).

This document summarizes the limitations of the current OMIDIENT/MOCSS shared-representation pipeline in this repository, and compares two proposed extensions for the thesis goal of learning **truly shared** latent codes across multi-omics modalities.

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



## Method 1: Cross-Reconstruction + Predictability Surrogate



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

```text
L_std^A    = ‖ h_A(P_A, S_A) − X_A ‖
L_shared^A = ‖ f_A(S_B, S_C) − X_A ‖
L_adv^A    = ‖ g_A(P_B, P_C) − X_A ‖
L_own^A    = ‖ k_A(S_A) − X_A ‖
```

Do **not** optimize a single combined scalar with one joint `backward()` on adversaries. A naive `L = … − β·L_adv − γ·L_own` would also update `g`/`k` to increase their own error. Use the two-step procedure below.

### Two-Step Training (per batch)

**Step 1 — train adversaries only** (`g` and `k`; encoders / `h` / `f` frozen, stop-grad on their inputs `P` and `S`):

```text
L_step1 = Σ_A [ L_adv^A + L_own^A ]
```

Minimize `L_step1` w.r.t. `g` and `k` only.

**Step 2 — train encoders and constructive predictors** (`h`, `f`, shared/private encoders, MLP heads; stop-grad through `g` and `k` parameters / detach adversary weights):

```text
L_step2 = Σ_A [ L_std^A + α·L_shared^A − β·L_adv^A − γ·L_own^A ] + λ_ctr·L_contrastive
```

Minimize `L_step2` w.r.t. encoders, `h`, and `f` only.

Interpretation of the minus signs in step 2: encoders try to make `(P_B, P_C)` uninformative about `X_A` and make `S_A` alone insufficient for full `X_A`, while step 1 keeps `g`/`k` competent predictors. Keep `γ` modest so `k` removes private bleed from `S` without wiping the shared signal that `f` and contrastive need.

Hyperparameters `α`, `β`, `γ` (and `λ_ctr`) should be chosen by search. A simple grid for the predictability weights is e.g. `[0.4, 0.7, 1, 1.5, 2]`.

### Information-Theoretic Interpretation

- **Minimize** L_shared → maximize I(X_A; S_B, S_C) → S captures cross-view predictable content
- **Maximize** L_adv (via −β in step 2) → minimize I(X_A; P_B, P_C) → private codes of other views don't leak shared signal
- **Maximize** L_own (via −γ in step 2) → minimize sufficiency of S_A alone for full X_A → private residual is pushed into P_A
- **Minimize** L_std → (P_A, S_A) remain jointly sufficient for X_A
- **Cross-view shared decoder** `f` forces S to be useful *collectively*, not a per-view full copy of X_m

This is closer to **multi-view sufficiency + privacy + own-shared insufficiency** than orthogonality alone.

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
| **Minimax / two-step**       | Needs the two-step (or GRL) procedure above; naive single backward on −L_adv/−L_own trains adversaries incorrectly; β, γ can still oscillate |
| **Over-strong k**            | Large γ can strip shared content from S_A, not only private residual—monitor L_own vs L_shared and keep γ modest                             |
| **Weak f or g**              | If f is weak, shared may underfit; if g is weak, −L_adv is vacuous                                                                           |
| **Contrastive / projection** | MLP projection can still hide some private dims in S; k mitigates but does not prove purity                                                  |




### Implementation Sketch

1. **Base on MOCSS** (`SharedAndSpecificEmbedding` / `SharedAndSpecificLoss`): keep separate encoders for P_m and S_m, and MLP projection heads + contrastive loss.
2. **Replace** solo `shared_rec` / `specific_rec` decoders with four predictors per modality: `h_A`, `f_A`, `g_A`, `k_A` (inputs as in the table above).
3. **Loss / training**: implement the two-step procedure (`L_step1` for `g`/`k`, `L_step2` for encoders/`h`/`f` + contrastive); log each term separately.
4. **Keep** contrastive on projected shared embeddings as in MOCSS.
5. **Evaluation**: same `evaluation.py` pipeline on concatenated [P_1, P_2, P_3, S̄].

---



## Method 2: Joint Reconstruction + Orthogonality Disentanglement



### Architecture

Method 2 does **not** use cross-reconstruction. Each modality is reconstructed only from **its own** private and shared codes:

```
X_m → Enc → P_m, S_m   (per modality)

Decoder_m:  D_m(P_m, S_m) → X̂_m   (joint within-view reconstruction)

Orthogonality loss between P_m and S_m (per modality)

Keep shared embedding + MLP projection + contrastive (per modality):
  S_1, S_2, S_3 → MLP → contrastive loss across the 3 projected rows
```

The key change vs current MOCSS is **how** reconstruction works:


|                 | Current MOCSS for the shared reconstruction part | Method 2                                   |
| --------------- | ------------------------------------------------ | ------------------------------------------ |
| Reconstruction  | `Dec_S(S_m) → X_m`                               | `D_m(P_m, S_m) → X_m`                      |
| Disentanglement | Weak elementwise orthogonality                   | Stronger P_m ⊥ S_m constraint per modality |


With a single joint decoder, S_m no longer must **alone** reconstruct X_m (as in MOCSS's shared branch). It only needs to contribute the shared part together with P_m. Independence constraint then pushes P_m and S_m to carry non-overlapping information.

### Mathematics

**Reconstruction (joint within-view decoder):**

```text
X̂_m = D_m(P_m, S_m)

L_recon = Σ_m ‖ D_m(P_m, S_m) − X_m ‖²
```

**Independence** between P_m and S_m within each modality (stronger options than current MOCSS):


| Variant           | Formula                                                   | What it enforces                                              |
| ----------------- | --------------------------------------------------------- | ------------------------------------------------------------- |
| Current (MOCSS)   | E[S̃ ⊙ P̃]                                                | Dim-wise co-activation ↓                                      |
| VICReg covariance | Σ_{i≠j} Cov(S)*ij² + Σ*{i≠j} Cov(P)_ij² + ‖ Cov(S,P) ‖_F² | Decorrelated dimensions; penalizes redundancy between P and S |


```text
L_independence = VIC-Reg decorrelation
```

**Contrastive** on projected shared codes (unchanged from MOCSS pipeline):

```text
L_ctr = Σ_{m < m'} InfoNCE( MLP(S_m), MLP(S_m') )
```

**Full objective:**

```text
L = L_recon + λ_ctr·L_ctr + + λ_vic·L_vic
```



Note: Don’t keep MOCSS `E[S⊙P]` alongside VICReg.

Note: Use existing MOCSS `reconstruction_loss` (center → normalize → matrix norm), not literal ‖·‖², for fair comparison.

### Advantages

1. **Simpler than Method 1** — no cross-reconstruction, no adversarial predictors; only 3 joint decoders
2. **Fixes a MOCSS flaw directly** — removes the separate `shared_rec` path that forces S_m to solo-reconstruct X_m
3. **Clear disentanglement story** — P_m and S_m partition information under a single reconstruction objective, with independence preventing duplication
4. **Easier to integrate** with OMIDIENT Dirichlet on P_m only; minimal architectural change from current code



### Concerns

**Collapse modes:**


| Collapse               | Mechanism                                                          |
| ---------------------- | ------------------------------------------------------------------ |
| S_m → 0                | Strong independence + joint decoder lets P_m alone satisfy L_recon |
| P_m absorbs everything | Weak contrastive or weak λ_ctr; S_m not needed for alignment       |


**Mitigations:**

1. **Capacity balancing (variance hinge)**: prevent variance of S to drop by introducing a loss term with a weight.
2. **Stronger orthogonality**: VICReg cross-covariance instead of MOCSS elementwise product
3. **Contrastive weight**: keep λ_ctr high enough so S_m stays meaningful across views
4. **Optional shared-only auxiliary**: light penalty ‖D_m^S(0, S_m) − X_m‖ or cross-view predictability (borrow lightly from Method 1) if S still collapses (It's a plan B if nothing else worked for this method)



### Implementation Sketch

1. **Keep** separate encoders for P_m and S_m per modality
2. **Replace** independent `shared_rec` with one joint decoder: `D_m(concat(P_m, S_m))`
3. **Upgrade** `orthogonal_loss` to VICReg cross-covariance per modality
4. **Keep** shared information matrices C^m, MLP projection heads, and contrastive loss as in MOCSS
5. **Add** variance hinge on S_m if collapse is observed during training. Ship **S variance from day 1**.

---

---



## References

- OMIDIENT paper: *OMIDIENT: Multiomics Integration for Cancer by Dirichlet Auto-Encoder Networks*
- Repository: [Zenodo record](https://zenodo.org/records/20273710)
- Baseline shared/specific model: MOCSS (Multi-Omics Contrastive Shared-Specific)

