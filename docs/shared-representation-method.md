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

Decoder_m:  
D_m(P_m, S_m) → X̂_m (should predict)
D_m(P_m', P_m'') → X̂_m (should not predict)
D_m(S_m', S_m'') → X̂_m (should predict)

Build shared-info matrices C_m from {S_m} (per modality)
Take selected rows (3 modalities, so 3 rows) → MLP (projection function) → contrastive loss (between these 3 rows)
```

Cross-reconstruction is the key structural change: modality `m` is rebuilt from **its own private and shared code plus the other modalities' shared codes**, not from `S_m` alone. Also, another decoder to reconstruct each modality from the private codes of other modality exists with a reverse loss term so that each modality won't be reconstructable using the other modalities' private codes.

### In another language: Predictability Surrogate

For target modality `A`, define three predictors:


| Network | Input      | Target | Role                                                                                                                                             |
| ------- | ---------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| h_A     | (P_A, S_A) | X_A    | Standard within-view reconstruction                                                                                                              |
| f_A     | (S_B, S_C) | X_A    | **Shared sufficiency**: other views' shared codes should predict A (not as close as h_A though, this is handled via the hyperparameters α and β) |
| g_A     | (P_B, P_C) | X_A    | **Private adversary**: other views' private codes should **not** predict A                                                                       |




### Loss Function

**Total loss (per sample, summed over modalities) (for training all the networks except g_A):**

```text
L = Σ_A [ L_std^A + α·L_shared^A − β·L_adv^A ] + λ_ctr·L_contrastive
```

where typically:

```text
L_std    = ‖ h_A(P_A, S_A) − X_A ‖
L_shared = ‖ f_A(S_B, S_C) − X_A ‖
L_adv    = ‖ g_A(P_B, P_C) − X_A ‖
```

Loss function of g_A:

```text
L = Σ_A L_adv^A
```

This separete g_A loss makes the minimax game situation so that the encoders try to put the informations into private codes that the decoder is not able to reconstruct the modality from that info.

The **minus sign on L_adv** makes training a minimax-style game: encoders try to make cross-view private codes uninformative about X_A, while g_A tries to predict anyway (gradient reversal or alternating updates).

The best α and β should be obtained in the hyperparamter search. (For the sake of simplicity, we can also set a set for each, such as [0.4, ,0.7, 1, 1.5, 2])

### Information-Theoretic Interpretation

- **Minimize** L_shared → maximize I(X_A; S_B, S_C) → S captures cross-view predictable content
- **Maximize** L_adv (via subtraction) → minimize I(X_A; P_B, P_C) → private codes of other views don't leak into A
- **Cross-decoder** forces S_m to be useful *collectively*, not per-view redundant copies of full X_m

This is closer to **multi-view sufficiency + privacy** than orthogonality alone.

### Advantages

1. **Operational definition of "shared"**: what is predictable from other modalities' S, not what reconstructs locally.
2. **Explicit anti-leakage** for private codes across views via adversarial term.
3. **Cross-reconstruction** reduces incentive for S_m to duplicate P_m's job.
4. **Theoretically defensible** in a thesis (sufficiency, minimal sufficient representation, adversarial privacy).



### Concerns


| Issue                               | Detail                                                                                                                 |
| ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| **9 extra networks**                | 3 modalities × (f, g, h) = 9 decoders/predictors on top of encoders—more parameters, tuning α, β, training instability |
| **Minimax optimization**            | −β·L_adv can oscillate or collapse without GRL, careful β schedule, or stop-gradient on encoders                       |
| **Shared may still encode private** | If f_A is weak, encoders can hide cross-view info in P_A and still satisfy h_A                                         |
| **Matrix C^m design**               | Not in current code; must be defined and ablated                                                                       |




### Implementation Sketch

1. **Keep** specific represenation's encoder and decoder producing P_m as is, and MLP projection heads from `SharedAndSpecificEmbedding`
2. Extend `SharedAndSpecificEmbedding` so that for shared representation there will be two encoders and three decoders for each modality -> one encoder for shared representation and one encoder for private representation. One decoder for h_A, one for f_A, and one for g_A.
3. **Extend** `SharedAndSpecificLoss` with L_std, α·L_shared, and −β·L_adv; drop solo `shared_rec` / `specific_rec` losses.
4. **Keep** C^m, MLP projection heads, and contrastive loss as in MOCSS
5. **Evaluation**: same `evaluation.py` pipeline on concatenated [P_1, P_2, P_3, S̄] as OMIDIENT does today

---



## Method 2: Joint Reconstruction + Orthogonality Disentanglement



### Architecture

Method 2 does **not** use cross-reconstruction. Each modality is reconstructed only from **its own** private and shared codes:

```
X_m → Enc → P_m, S_m   (per modality)

Decoder_m:  D_m(P_m, S_m) → X̂_m   (joint within-view reconstruction)

Orthogonality loss between P_m and S_m (per modality)

Build shared-info matrices C_ from {S_m}
Take selected rows (3 modalities, so 3 rows) → MLP (projection function) → contrastive loss (between these 3 rows)
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
L = L_recon + λ_ctr·L_ctr + VICReg independence loss
```



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
5. **Add** variance hinge on S_m if collapse is observed during training

---

---



## References

- OMIDIENT paper: *OMIDIENT: Multiomics Integration for Cancer by Dirichlet Auto-Encoder Networks*
- Repository: [Zenodo record](https://zenodo.org/records/20273710)
- Baseline shared/specific model: MOCSS (Multi-Omics Contrastive Shared-Specific)

