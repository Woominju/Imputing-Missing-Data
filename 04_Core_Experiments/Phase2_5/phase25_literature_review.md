# Literature review: delay-embedding imputation with no fully-observed variable

Date: 2026-10-27. Sources were read from the papers themselves: full text extracted from arXiv/PLoS PDFs, plus the ICLR text mirror for LCCM. Anything I could not verify from primary text is marked **[UNVERIFIED]**.

Project recap (the yardstick for "overlap"): MCAR missingness hits every variable, so no variable is fully observed. Each variable has its own delay-window encoder (zeros at masked entries plus a mask channel). The encoders fuse into one shared latent ζ(t). Decoders output current values (Model C, mean fusion) or the full delay vectors of every variable (FDR). The planned probabilistic version uses PoE fusion and a variational objective. Baseline: FPRM (Nat. Commun. 2026), which runs GP regression from the delay embedding of one fully observed variable, plus a "lag FPRM" variant for synchronous gaps.

---

## 1. Bakarji, Champion, Kutz, Brunton: "Discovering governing equations from partial measurements with deep delay autoencoders"

- **Venue:** Proc. R. Soc. A 479(2276):20230422, 2023. DOI [10.1098/rspa.2023.0422](https://doi.org/10.1098/rspa.2023.0422). Preprint [arXiv:2201.05136](https://arxiv.org/abs/2201.05136) (Jan 2022). The journal DOI and volume come from search metadata; I read the arXiv v1 text.
- **Problem:** Learn a closed-form ODE (SINDy) plus a coordinate system from a single measured scalar of a chaotic system, i.e. recover "hidden variables". Tested on Lorenz, Rössler, Lotka–Volterra, and a chaotic waterwheel video.
- **Observation conditions:** A single scalar y(t), fully and regularly observed. The text says "we will assume scalar measurements y(t), although the following arguments generalize to multi-dimensional vectors". The multivariate case is not developed. Missing data is not treated at all.
- **Model:** Delay vector h(t) = [y(t), y(t+τ), …, y(t+(n−1)τ)], i.e. the columns of a Hankel matrix. An optional linear SVD projection onto p modes comes first, then encoder φ to z, decoder ψ, and SINDy dynamics ż = Θ(z)Ξ. The losses are:
  - L_recon = ||h − ψ(φ(hᵀU_p))U_pᵀ||²
  - SINDy losses in ḣ and ż
  - **L_z1 = ||h_{i1} − z_{i1}||²**, which forces the first latent coordinate to equal the measurement
  - **L_cons**, which integrates the SINDy model (RK4 layer) and requires the first latent component to reproduce the *whole delay vector* h_i
  - L1 sparsity on Ξ
- **Delay embedding:** Yes. A univariate Hankel/delay vector, with Takens n > 2m cited as the condition for a diffeomorphism.
- **Shared latent:** z is a learned coordinate system meant to be diffeomorphic to the original attractor, with sparse dynamics. There is only one view, so nothing is shared across views.
- **Theory:** None proven. The premises are (1) Takens says a diffeomorphism exists and (2) universal approximation says a network can learn it. The authors state: "It is unknown whether or not there exists a loss function for which the original system… can be exactly recovered." They suggest invertible networks and probabilistic SINDy as future work.
- **Overlap with us:** (a) An autoencoder on delay vectors. (b) A decoder that reconstructs the *whole delay vector*, which is conceptually close to FDR. (c) L_cons is a temporal-consistency loss across delays.
- **Differentiation:** They have one fully observed scalar and do no imputation, no multiple views, no missingness, and no fusion. Our setting is multi-view (one embedding per variable), has MCAR in every view, and has imputation as the goal. FDR reconstructs the delay vectors of *all* variables from one fused latent. Bakarji reconstructs only the input's own Hankel vector.

## 2. De Brouwer, Arany, Simm, Moreau: "Latent Convergent Cross Mapping" (ICLR 2021)

- **Links:** [ICLR 2021 poster](https://iclr.cc/virtual/2021/poster/2939), [mlanthology](https://mlanthology.org/iclr/2021/brouwer2021iclr-latent). I did not find an arXiv ID or OpenReview ID; the text was read from the ICLR proceedings text mirror.
- **Problem:** *Causal discovery* (CCM) between chaotic time series that are "short, noisy and sporadic (incomplete observations at infrequent and irregular intervals)". The goal is not imputation.
- **Observation conditions:** Sporadic data: "irregular in time and across dimensions… multiple missing observations at any given time". No variable needs to be fully observed. Observations come as many short segments.
- **Model:** **One GRU-ODE-Bayes model per time series** (X and Y each get a separate model). Each learns a continuous-time latent H(t) with dH/dt = f(H,t) and X = g(H). It is trained "to minimize forecasting error" and uses a filtering approach, so no future information leaks. Causality X→Y is then tested by kNN-regressing H_X from H_Y and checking correlation and its convergence (Sc = Ccm_full − Ccm_0).
- **Delay embedding:** **No.** The paper explicitly replaces delay embeddings: "without computing delay embeddings… eliminating the need for delay-embedding".
- **Latent meaning:** H_X is claimed to be an embedding of the attractor (Lemma 3.1). The condition is that, for some observation function and a valid Takens (k, τ), the map Φ_{k,τ}(g(φ_H), H(H(t))) is injective. They also say: "The requirement of … being injective is not enforced in our architecture. However, with sufficient regularization… it is satisfied in practice." The latents are **per series and not fused or shared.**
- **Missing data handling:** Handled natively by the GRU-ODE-Bayes filter (Bayesian updates at observation times, ODE between them). Baselines include multispatial CCM on linear or GP/MVGP-interpolated data.
- **Overlap:** Uses a learned latent instead of explicit delay coordinates under sporadic, all-variable missingness, and justifies it with Takens. Cross-mapping between latents (kNN from one latent to another) is similar in spirit to cross-variable prediction in FPRM.
- **Differentiation:** Their task is causal inference, not reconstruction. The latents are per series with no fusion, and imputation accuracy is not evaluated. We fuse per-variable delay views into *one* latent and decode missing values. Our explicit delay windows connect to Deyle–Sugihara mixed-lag embeddings, whereas their ODE-filter latent has an injectivity assumption the architecture does not enforce.

## 3. Nguyen, Karaman, Kim, Wang, Liu, Sabuncu: "Knockout: A simple way to handle missing inputs"

- **Links:** [arXiv:2405.20448](https://arxiv.org/abs/2405.20448) (v1 May 2024, v3 Jul 2025). The arXiv comment says **"Accepted at TMLR"**.
- **Problem:** Supervised prediction p(Y|X) where some inputs or modalities may be missing at inference. Applications include multimodal medical imaging, tabular data, and tumour segmentation.
- **Method:** During training, randomly replace features with **constant placeholder values**: X′ = M⊙x̄ + (1−M)⊙X, with M sampled independently of X and Y. For theory, x̄ is out of support. In practice: z-score then 0 (the mean) for structured or latent inputs, 0 or the mean image for images. "Structured Knockout" knocks out whole modalities or blocks.
- **Theory:** If x̄_M ∉ Support(X_M) and M ⊥ (X, Y), then p(Y | X′_M = x̄_M, X′_{−M} = x_{−M}) = p(Y | X_{−M} = x_{−M}). In other words, the placeholder implements the correct marginal ("implicit marginalization"). The training loss equals Σ_m p(M=m) E[ℓ(Y; f(X′(m,X)))], a multi-task mix of all marginal predictors. They discuss MCAR training data (merge the natural mask with the knockout mask using the same placeholder). For MAR/MNAR they use a *different* placeholder.
- **Delay embedding / latent / dynamics:** None. The method is generic.
- **Overlap:** This is exactly the mechanism our encoders use: zero-filling masked entries, a mask indicator, and random artificial masking during training. Knockout gives the formal justification that a network trained with random masking, independent of the data, learns E[target | visible inputs] for every mask pattern. The caveat is that the paper's theorem uses an out-of-support placeholder; the mask indicator plays that role for us.
- **Differentiation:** Knockout is a training trick for discriminative prediction. It does not do imputation of the inputs themselves, and has no time-series, embedding, or latent fusion structure. It should be cited as justification for our masking scheme, not as a competitor.

## 4. Multimodal VAEs

### 4a. Wu & Goodman, "Multimodal Generative Models for Scalable Weakly-Supervised Learning" (MVAE), NeurIPS 2018. [arXiv:1802.05335](https://arxiv.org/abs/1802.05335)
- **Generative model:** p(x_1..x_N, z) = p(z) Π_i p(x_i | z), with modalities conditionally independent given z.
- **Inference:** Starting from p(z|x_1..N) ∝ Π p(z|x_i) / Π^{N−1} p(z), they approximate p(z|x_i) ≈ q̃(z|x_i)p(z), which gives **PoE with a prior expert**: q(z|X) ∝ p(z) Π_{x_i∈X} q̃(z|x_i), with Gaussian closed form T = Σ T_i, μ = (Σ μ_i T_i)(Σ T_i)⁻¹.
- **Objective (Eq. 2):** ELBO(X) = E_{q(z|X)}[Σ_{x_i∈X} λ_i log p(x_i|z)] − β KL(q(z|X) || p(z)), which lower-bounds **log p(X)** for the *present* subset X. Unobserved modalities are dropped from the likelihood.
- **Sub-sampled training (Eq. 5):** ELBO(x_1..N) + Σ_i ELBO(x_i) + Σ_j ELBO(X_j) for k random subsets. Each ELBO term encodes **and** reconstructs the *same* subset. Daunhawer et al. state this explicitly: MVAE "does not reconstruct the missing modalities from the set of sub-sampled modalities." The paper's stated motivation is that a PoE does not uniquely determine its factors, so without sub-sampling the unimodal experts never learn to work alone.
- **Overlap:** The PoE fusion over views is exactly the fusion we plan. In our setting, views are per-variable delay windows (for us, PoE also gives automatic down-weighting of heavily masked views, if the experts' variances depend on the mask).

### 4b. Shi, Siddharth, Paige, Torr, "Variational Mixture-of-Experts Autoencoders for Multi-Modal Deep Generative Models" (MMVAE), NeurIPS 2019. [arXiv:1911.03393](https://arxiv.org/abs/1911.03393)
- **Fusion:** Mixture of experts, q(z|x_1:M) = Σ_m (1/M) q(z|x_m).
- **Objective (Eq. 3):** Stratified IWAE (DReG estimator), (1/M) Σ_m E_{z_m^{1:K}∼q(z|x_m)} [log (1/K) Σ_k p(z_m^k, x_{1:M}) / q(z_m^k | x_{1:M})]. Each **unimodal** sample must reconstruct **all** modalities, so the objective has cross-reconstruction built in. It is a bound on log p(x_{1:M}) and needs **all modalities present during training**. Missing modalities are handled only at test time.
- **Critique of PoE:** "experts with greater precision will have more influence… When the precisions are miscalibrated… [PoE is] undesirable for learning factored latent representations."
- **Relevance:** In our setting, MCAR mask-dependent precision miscalibration is a real risk for PoE. MoE avoids overconfident experts but cannot sharpen the posterior as views are added. Our Model C currently uses mean fusion, which is closer to MoE and averaging.

### 4c. Sutter, Daunhawer, Vogt, "Generalized Multimodal ELBO" (MoPoE-VAE), ICLR 2021. [arXiv:2105.02470](https://arxiv.org/abs/2105.02470)
- **Fusion:** Posterior is a mixture over all subsets in the powerset P(X) of the PoE of each subset.
- **Key equation (Eq. 4):** L_k(θ, φ_k; X) := E_{q̃(z|X_k)}[log p_θ(X|z)] − KL(q̃(z|X_k) || p_θ(z)), i.e. **encode the subset X_k, decode everything X**. The authors note this "still yields a valid lower bound on p_θ(X), whereas [encoding and decoding X_k] forms a lower bound on log p_θ(X_k)". Lemma 1 relates the average over P(X) to log p_θ(X).
- **Relevance:** Structurally this is the closest published objective to our held-out ELBO (see §8).

### 4d. Daunhawer, Sutter, Chin-Cheong, Palumbo, Vogt, "On the Limitations of Multimodal VAEs", ICLR 2022. [arXiv:2110.04121](https://arxiv.org/abs/2110.04121)
- **Mixture-based family (Eq. 2):** L_S = Σ_{A∈S} ω_A {E_{p(x)p(z|x_A)}[log q(x|z)] − E KL(p(z|x_A) || q(z))}. This family includes MMVAE, MoPoE, and MVAE *without* ELBO sub-sampling.
- **Theorem 1:** E_{p(x)}[log p(x)] ≥ L_S + Δ(X,S), where **Δ(X,S) = Σ_A ω_A H(X_{\A} | X_A)**. The gap is independent of θ and φ, so "cross-modal prediction cannot recover information that is specific to the target modalities". Corollary 1: Δ = 0 without modality sub-sampling. Corollary 2: Δ grows with the number of modalities for MMVAE and MoPoE. MVAE's ELBO sub-sampling is analysed separately in their App. C.
- **Implication for us (my inference, not stated by the paper):** An objective that encodes a subset x_in and decodes the held-out x_H is a bad *density model* of the joint, with an irreducible gap H(X_H | X_in). But **for imputation the target is p(x_H | x_in)**, so this gap is exactly the irreducible imputation uncertainty. For deterministic chaotic systems where x_in contains an embedding (Deyle–Sugihara), this conditional entropy is small apart from noise. That is a useful framing argument for us: the Takens/Deyle–Sugihara condition is precisely what makes this "generative discrepancy" vanish.

## 5. Deyle & Sugihara, "Generalized Theorems for Nonlinear State Space Reconstruction", PLoS ONE 6(3):e18295, 2011. DOI [10.1371/journal.pone.0018295](https://doi.org/10.1371/journal.pone.0018295)

These statements come from the extracted PDF text. Symbols are partly garbled; m denotes the manifold dimension.

- **Theorem 1 (no lags):** M is compact, m-dimensional; there are 2m+1 C² observation functions y_k: M→R. "It is a generic property of all possible [maps] Φ = (y_1, …, y_{2m+1})" that Φ is an embedding. This is Whitney-like: genericity is over *all* maps.
- **Setup for Theorem 2:**
  - y_q is a "lag" of y if y_q = y∘φ^b for b > 0. Negative lags are "noted that the results apply equally".
  - The unlagged observables are y_r for r in a set **r**. For each r, C_r contains y_r and all its lags. **"The lags b_q are distinct for fixed r."**
  - The pair C = {C_r}, b = {b_k : k = 1..2m+1} defines the subset Ỹ^{2m+1}_{C,b} ⊂ Y^{2m+1} "containing all choices of 2m+1 observables {y_k} which obey the correct lag relationships under a dynamical system φ". Each element is identified with (φ, {y_r}).
- **Theorem 2:**
  - Assumptions: φ: M→M is a diffeomorphism on a compact m-manifold, and there are 2m+1 C² observation functions restricted to lag relationships (C, b).
  - Generic properties imposed on φ: (1) the set A of periodic points with period p < max(b_k) is finite; (2) the eigenvalues of (Dφ^b)_x on a compact neighbourhood of A are distinct and ≠ 1.
  - Conclusion: "Then, for generic {y_k} ∈ Ỹ^{2m+1}_{C,b}, the mapping Φ(φ,{y_r}) = (y_1(x), …, y_{2m+1}(x)) is an embedding."
- **Corollary 3 (flows):** Same, with φ_τ the time-τ flow of a smooth vector field X. Conditions: (1) at zeros of X, the eigenvalues of (Dφ_τ) are distinct and ≠ 1; (2) "No periodic integral curve of X has integer period ≤ 2m+1."
- **Theorem 7 (prevalence, after Sauer–Yorke–Casdagli):**
  - Setup: φ is a diffeomorphism on an open U ⊂ R^m, and A ⊂ U is compact with boxdim(A) = d. C and b are as above, "such that **n = Σ_r n_r > 2d**". Here n_r is the number of coordinates in group C_r, i.e. the total number of coordinates across all variables.
  - Periodic-point condition: for every p ≤ max(b), 2·boxdim(A_p) < n − B_p, and Dφ^p has distinct eigenvalues on A_p. B_p counts lag coincidences: B_{pr} is the number of y_q ∈ C_r whose lag b_q = m·p + b_{q′} for some other b_{q′}, and B_p = Σ_r B_{pr}.
  - Conclusion: "for almost every set of n observation functions {y_k} satisfying the given lag relationships, the map … is an embedding on A."
- **Answers to the specific questions:**
  - **Genericity is over:** the *observation functions* (the unlagged y_r, which then determine their lags). φ is fixed and required to satisfy generic periodic-point conditions. In Thm 2 the proof perturbs the y_r with φ fixed; formally the elements of Ỹ are written (φ, {y_r}). Thm 7 is "almost every" (prevalence) over observation functions with φ fixed. **Important for us:** the theorems do *not* guarantee that a specific choice like "the Lorenz x and z coordinates" works. The paper's own Fig. 1C shows lags of Lorenz z giving only an immersion (a symmetry, non-generic), while Fig. 1D, (y(t), y(t−τ), z(t)), is an embedding.
  - **Dimension condition:** total coordinates 2m+1 (Thm 1, 2, Cor 3), with m the manifold dimension; or n > 2·boxdim(A) (Thm 7). The count is over *all* coordinates pooled across variables.
  - **Arbitrary mixtures of lags across variables:** Yes, within the framework. Any collection of unlagged observables, each with any set of distinct non-negative lags, is allowed, *including non-consecutive lags*. The paper says it "extends the generalization to non-consecutive lags", i.e. Sauer et al.'s Remark 2.9. The price is the periodic-orbit condition through B_p.
  - **Number of lags per variable:** **No minimum is stated.** A variable may contribute only its unlagged value (Fig. 1D uses z(t) alone), and n_r can differ across variables. Each C_r is defined to contain y_r itself, so implicitly each variable that is used has its lag-0 term; lags are relative to that "unlagged" member. This is a relative-time convention, not a requirement that the present value be observed. The paper does **not** discuss missing coordinates. It says nothing about which subsets of lags remain embeddings when some coordinates are deleted (i.e. MCAR masks).
  - **Caveats in the paper:** "generic" exceptions need not have measure zero (hence Thm 7); the results are a "proof of principle"; noise benefits are conjectured, not proven.
- **Use for us:** Under MCAR with every variable missing, the visible coordinates at time t form a random *mixed-lag multivariate* coordinate set. Thm 2/7 say that, generically, such a set is an embedding provided the *total number of visible coordinates* exceeds 2d (plus the periodic-orbit technicalities). This is the theoretical hook for "no fully observed variable is needed". Two caveats must be stated honestly: (i) generic does not mean guaranteed for the specific coordinate functions (the Lorenz z example); (ii) the lag set must be fixed in advance in the theorem, while a mask-dependent subset means a union over finitely many lag patterns. Each pattern is generic, and a finite intersection of open dense or prevalent sets is still open dense or prevalent, so the argument extends to all masks with enough coordinates. That last step is my own inference and is not in the paper.

## 6. Missing-data deep latent variable models

### MIWAE: Mattei & Frellsen, "MIWAE: Deep Generative Modelling and Imputation of Incomplete Data Sets", ICML 2019 (PMLR 97). [arXiv:1812.02633](https://arxiv.org/abs/1812.02633)
- **Setting:** i.i.d. tabular or image data, with MAR missingness in *training* data. Not time series.
- **Encoder:** q_γ(z|x^o) = Ψ(z | g_γ(ι(x^o))), where ι is a fixed imputation (e.g. zeros) completing x^o.
- **Objective (Eq. 4):** L_K = Σ_i E_{z_1..K ∼ q(z|x^o_i)} [log (1/K) Σ_k p_θ(x^o_i | z_k) p(z_k) / q(z_k | x^o_i)] ≤ ℓ(θ) = Σ log p_θ(x^o_i), the **observed-data log-likelihood**. With K=1 this is the "MVAE bound" (no relation to Wu & Goodman), derived concurrently by Nazábal et al. (HI-VAE).
- **Imputation:** Single imputation via self-normalised importance sampling of E[x^m | x^o]; multiple imputation via SIR. No artificial held-out entries.

### Partial VAE / EDDI: Ma, Tschiatschek, Palla, Hernández-Lobato, Nowozin, Zhang, "EDDI: Efficient Dynamic Discovery of High-Value Information with Partial VAE", ICML 2019. [arXiv:1809.11142](https://arxiv.org/abs/1809.11142)
- **Problem:** Active feature acquisition. The Partial VAE predicts p(x_U | x_O) for any observed subset.
- **Partial inference net:** q(z|x_O) uses a **permutation-invariant set encoder** c(x_O) = g(h(s_1), …, h(s_|O|)), with s_d = [e_d, x_d] (PointNet) or e_d·x_d (PNP; generalises zero-imputation). e_d is a learned per-variable ID embedding and g is sum or max pooling.
- **Objective (Eq. 4):** log p(x_O) ≥ E_{q(z|x_O)}[log p(x_O|z) + log p(z) − log q(z|x_O)], a lower bound on the **observed-part marginal** log p(x_O). Imputation uses p(x_U|x_O) ≈ ∫ p(x_U|z) q(z|x_O) dz.
- **Training masks:** App. B: "We first draw a missing rate parameter from U(0, 0.7) and randomly choose variables as unobserved. This step is repeated at each iteration." The held-out variables are *removed* from the input. As far as I can read, the likelihood is still only on x_O, so the paper does not state any loss on the artificially removed entries **[partially unverified: the main text shows the likelihood only on x_O]**.
- **Overlap:** A per-variable encoding with a learned ID plus aggregation into one latent is architecturally close to our per-variable encoders plus fusion. Their aggregation is sum or max pooling of deterministic features, not PoE.

### GP-VAE: Fortuin, Baranchuk, Rätsch, Mandt, "GP-VAE: Deep Probabilistic Time Series Imputation", AISTATS 2020 (PMLR 108). [arXiv:1907.04155](https://arxiv.org/abs/1907.04155)
- **Model:** A latent z_t per time step. Each latent channel gets an independent GP prior over time (Cauchy kernel). The encoder is a CNN over the zero-filled sequence, giving a structured Gaussian q(z_{1:T,j} | x^o_{1:T}) with banded precision. Decoding is per time step: p(x_t | z_t).
- **Objective:** log p(X^o) ≥ Σ_t E_q[log p(x^o_t | z_t)] − β KL(q(z_{1:T} | x^o) || p(z_{1:T})). The ELBO is evaluated only on observed features (following HI-VAE); missing inputs are zero-filled.
- **Delay embedding:** No. Temporal structure comes from the GP prior and the CNN, with smoothness as the inductive bias. Attractor or embedding theory is not invoked.
- **Overlap:** Probabilistic imputation of multivariate time series with all channels missing and no fully observed channel, which is our *setting*. The *mechanism* differs: we use a dynamical embedding, they use a smooth GP latent.

### One-liners for other time-series imputers
- **BRITS** (Cao et al., NeurIPS 2018, [arXiv:1805.10572](https://arxiv.org/abs/1805.10572)): a bidirectional RNN with temporal decay. Imputations are treated as RNN graph variables, with feature regression and a forward/backward consistency loss. Missingness is allowed in all variables, and no dynamical-systems theory is used.
- **SAITS** (Du, Côté, Liu, Expert Systems with Applications 2023, doi:10.1016/j.eswa.2023.119619, [arXiv:2202.08516](https://arxiv.org/abs/2202.08516)): a self-attention imputer trained jointly on the **Masked Imputation Task** (artificially mask observed values and compute the loss on them) and the **Observed Reconstruction Task**. MIT is a deterministic analogue of our x_H loss.
- **CSDI** (Tashiro, Song, Song, Ermon, NeurIPS 2021, [arXiv:2107.03502](https://arxiv.org/abs/2107.03502)): a conditional score-based diffusion model. Self-supervised training "separates observed values into conditional information and imputation targets" (as in masked language modelling). This is exactly the x_in / x_H split, but with a diffusion model rather than an ELBO over a latent.

## 7. Prior work on delay-embedding imputation without a fully observed variable

- **Ye & Sugihara, "Information leverage in interconnected ecosystems: Overcoming the curse of dimensionality", Science 353:922–925 (2016). DOI [10.1126/science.aag0863](https://doi.org/10.1126/science.aag0863).** Multiview embedding (MVE) builds many attractor reconstructions from different combinations of variables and lags, ranks them by in-sample forecast skill, and averages nearest-neighbour forecasts from the top k (k = √(number of views) by default in rEDM). The goal is **forecasting** short, noisy series, not imputation. Missing data is not the stated focus. **[UNVERIFIED from the primary text: Science full text was not accessible. Details come from the abstract and rEDM documentation.]**
  - Overlap: the "multiple views from different variable/lag combinations, then combine" idea is the conceptual ancestor of our per-variable views plus fusion.
  - Differentiation: we learn the fusion end-to-end, the views are fixed per variable, and the masks vary per time step.
- **Ma, Leng, Aihara, Lin, Chen, "Randomly distributed embedding making short-term high-dimensional data predictable", PNAS 115(43):E9994 (2018).** Random low-dimensional *non-delay* (cross-variable) embeddings are mapped to a delay embedding of a target variable, and the predictions are aggregated as a distribution. This is forecasting with the target fully observed in the training window. It is the same "multi-view fusion" spirit but not imputation. [Abstract-level only.]
- **Paparella, "Filling gaps in chaotic time series", Phys. Lett. A 346:47–53 (2005), [arXiv:nlin/0502044](https://arxiv.org/abs/nlin/0502044).** Fills wide gaps in a (apparently scalar) deterministic series by minimising a functional measuring consistency with Takens-reconstructed dynamics. Univariate, gap-structured, and non-learned. [Abstract-level only.]
- **Lv, Mao, Wang, Yao, "Reconstruction and Prediction of Chaotic Time Series with Missing Data: Leveraging Dynamical Correlations Between Variables", Mathematics 13(1):152 (2025), DOI 10.3390/math13010152.** LightGBM models inter-variable coupling and a Non-Stationary Transformer handles prediction. Tested on chaotic systems at 30–50% missing rates. **[UNVERIFIED: the MDPI page returned 403. I could not confirm whether all variables are missing or whether delay embeddings are used.]** This is the closest application-level overlap I found and should be checked manually.
- **Young & Graham, "Deep learning delay coordinate dynamics for chaotic attractors from partial observable data", Phys. Rev. E 107:034215 (2023), [arXiv:2211.11061](https://arxiv.org/abs/2211.11061).** Uses deep nets on delay coordinates of partial observations to forecast. No missing data.
- **LCCM (§2)** and **multispatial CCM** (Clark et al. 2015, cited by LCCM) handle sporadic or short data for causal inference, not imputation.
- **Search result:** I found **no paper** that does imputation with per-variable delay embeddings of *all* variables, a learned fused latent, and MCAR on every variable with no fully observed variable. Searches covered "delay embedding imputation all variables missing", "cross mapping imputation", "multi-view embedding missing data Takens", and "PoE VAE time-series per-variable encoders". This is a negative search result, not proof of absence. The Lv et al. 2025 paper and possible follow-ups citing FPRM (2026) should be checked by hand.

## 8. The held-out ELBO question

Proposed objective: **J = E_{q(ζ|x_in)}[log p(x_H|ζ)] − KL(q(ζ|x_in) || p(ζ))**, where x_in are the visible inputs and x_H the artificially held-out entries.

- **Is it used verbatim in any reviewed paper?** **No.**
  - MVAE: encodes and decodes the same subset (Eq. 2/5).
  - MIWAE, Partial VAE, GP-VAE: encode x^o and decode x^o (observed-only likelihood).
  - MMVAE: encodes one modality and decodes *all* modalities x_{1:M}, which must be fully present.
  - **MoPoE Eq. (4)** comes closest: L_k = E_{q(z|X_k)}[log p(X|z)] − KL(q(z|X_k) || p(z)), i.e. encode a subset, decode *all*, *including* the encoded subset.
  - Daunhawer Eq. (2) defines the general family (encode x_A, decode the full x).
  - Deterministic analogues of "loss on artificially held-out entries" are SAITS (MIT) and CSDI's conditional/target split. EDDI uses random artificial removal but, as far as I could verify, keeps the likelihood on x_O.
- **What J bounds:** Because the ELBO holds for any q, including one conditioned on x_in, **J ≤ log p_θ(x_H)**, the *marginal* of the held-out entries under the unconditional prior. It is **not** a bound on the imputation target log p(x_H | x_in). Two variants:
  - Decode x_in ∪ x_H (all observed entries): J′ = E_{q(ζ|x_in)}[log p(x_in, x_H | ζ)] − KL(q || p(ζ)) ≤ log p(x_obs). This is exactly MoPoE's Eq. 4 restricted to observed entries, a MIWAE/HI-VAE-style observed-data bound with a sub-sampled encoder. It is the most defensible "proper" objective, and Daunhawer's Theorem 1 says its gap is H(X_H | X_in) per sub-sampled set.
  - Use a conditional prior p(ζ | x_in) (CVAE-style): E_{q(ζ|x_in, x_H)}[log p(x_H|ζ)] − KL(q(ζ|x_in, x_H) || p(ζ|x_in)) ≤ log p(x_H | x_in). This is the direct bound on the imputation likelihood.
  - With J as written, the KL to an *unconditional* prior penalises ζ for carrying the very information about x_in needed to predict x_H. For imputation it acts as an information bottleneck (β-VAE / VIB view), not as a likelihood bound on the conditional. This is worth stating explicitly in the paper. Daunhawer et al. note that the multimodal ELBO can be viewed through the VIB lens.
- **How the papers define their missing-modality objectives (which quantity is bounded):**
  - **MVAE:** log p(X_present) via ELBO(X) over present modalities, with a sub-sampled sum of ELBOs over subsets, each a bound on that subset's marginal.
  - **MMVAE:** log p(x_{1:M}) (complete data) via a stratified MoE-IWAE.
  - **MoPoE:** log p(X) via the average over subsets of E_{q(z|X_k)}[log p(X|z)] − KL.
  - **MIWAE:** Σ_i log p(x^o_i) (observed-data likelihood, valid under MAR) via IWAE with q(z | ι(x^o)).
  - **Partial VAE:** log p(x_O) via the ELBO with a set-encoder q(z|x_O).
  - **GP-VAE:** log p(X^o_{1:T}) via a structured-q ELBO evaluated on observed features only.
  - **Knockout:** not a VAE. It minimises E_M E_{X,Y} ℓ(Y; f(X′(M,X))), equal to a p(M)-weighted sum of risks of all marginal predictors p(Y | X_{−M}).

## Differentiation summary (my synthesis)
1. **Setting:** every variable has MCAR missingness and no variable is fully observed. This is unlike FPRM (needs a fully observed driver), Bakarji (one fully observed scalar), and MVE/RDE (forecasting with full data). GP-VAE, BRITS, SAITS, and CSDI share the setting but use no dynamical embedding theory.
2. **Theory hook:** Deyle–Sugihara Thm 2/7 (pooled mixed-lag coordinates, n > 2d, non-consecutive lags allowed, no per-variable minimum) justifies reconstructing the state from whatever mixed-lag coordinates survive the mask. Combined with Knockout's implicit-marginalisation argument for random masking, it justifies a single network for all masks. Daunhawer's Δ = H(X_H | X_in) gives a clean way to say the cross-reconstruction gap vanishes (up to noise) once the visible coordinates embed the attractor.
3. **Architecture:** per-variable delay-view encoders, then PoE/mean fusion into one ζ, then FDR decoding of the delay vectors of *all* variables. Nearest relatives: MVAE (PoE over modalities), Partial VAE (per-variable ID encoders plus pooling), and Bakarji (decoding full delay vectors, consistency across delays). None combines them for dynamical imputation.
4. **Risks flagged by the literature:** PoE precision miscalibration (Shi et al.), generative-quality gaps from sub-sampling (Daunhawer), and genericity not guaranteeing specific coordinates (Deyle–Sugihara's Lorenz-z example).

## Verification notes
- Read from primary full text: Bakarji (arXiv v1), LCCM (ICLR text), Knockout (arXiv v3), MVAE, MMVAE, MoPoE, Daunhawer, MIWAE, EDDI, GP-VAE, SAITS, CSDI, BRITS (abstract and method portions), Deyle–Sugihara (PLoS PDF; some math glyphs garbled, but the theorem text was readable).
- Not verified from primary text: Ye & Sugihara 2016 methods, RDE, Paparella, Lv et al. 2025 (abstract or secondary sources only), and FPRM (Nature page behind redirect; taken from the user's description).
