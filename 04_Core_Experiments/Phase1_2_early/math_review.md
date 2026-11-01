# Math review for interpreting the current experiments

Tags: **[Proposal]** stated in the official proposal · **[Literature]** existing results, not re-proved
here · **[Verified here]** computed in this repo (file given) · **[Ours]** our design choice ·
**[Open]** not established.

Setting: ds/dt = f(s), s in R^d (Lorenz d = 3), x_i(t) = h_i(s(t)),
F_i(s) = [x_i(t), x_i(t - tau), ..., x_i(t - (m-1) tau)]. In code tau is in samples; Lorenz dt = 0.01,
so tau = 1 sample = 0.01 time units and the main runs (m = 8, tau = 1) span 0.07 time units.

## 1. When does G_{j<-i} = F_j o F_i^{-1} exist? [Literature]
- Takens (1981): M compact smooth manifold, dim d; for generic (f, h) and m >= 2d + 1, F is an embedding.
- Sauer, Yorke & Casdagli (1991): compact invariant set A with box-counting dimension d_A; for
  prevalent h and m > 2 d_A, F is injective on A and an immersion on compact pieces of smooth
  manifolds in A; conditions on periodic orbits of period tau, 2 tau, ... are needed.
- Conclusion if F_i and F_j are both injective on A: G_{j<-i} is a well-defined continuous bijection
  F_i(A) -> F_j(A). "Diffeomorphism between images" needs embeddings of a manifold containing A.
- Scope for Lorenz: d_A ~ 2.06, so m >= 5 meets the counting condition. **Genericity can fail for
  a specific observable**, see 2(b).
- [Proposal] uses this correspondence as motivation. [Ours] no part of the code enforces or checks it.

## 2. Lorenz-specific facts [Verified here: src/observability.py, results/observability_results.csv]
(a) O_x(s) = [x, x', x''] has det D O_x = -sigma^2 x (SymPy; numeric max error 5e-12).
    O_x is not an immersion on the plane x = 0, which the attractor crosses.
(b) z-only observation is **never injective** on the attractor: Gamma(x, y, z) = (-x, -y, z) commutes with
    the flow (SymPy) and leaves z unchanged, so F_z(s) = F_z(Gamma s) for every m and tau.
    Numerically z(t; s) and z(t; Gamma s) agree exactly over 5 time units while |s - Gamma s| = 7.23.
    Hence G_{x<-z} does not exist. This is a failure of the genericity assumption in 1, not of Takens.
(c) Small tau: det D F_{x,tau} = sigma^2 tau^3 x + O(tau^4) (SymPy, 2nd-order Taylor). Numerically
    corr(det, x) = 0.9986 at tau = dt, and the approximation degrades as tau grows
    (median relative error 0.12 at dt, 0.74 at 5 dt, 12 at 20 dt). Delay maps at small tau are
    ill-conditioned (median condition number 855 at tau = dt).
(d) Multi-view [x(t), y(t - tau), z(t - 2 tau)]: det = 1 + (1 + 2 beta) tau + O(tau^2) > 0 for small tau
    (SymPy, first order). Median condition number 1.2 at tau = dt versus 855 for x-only.
(e) Exact algebraic identities of the Lorenz equations: y = x + x'/sigma, and z = rho - (y' + y)/x
    (singular at x = 0). Their link to which variable is easiest to impute is **[Open]**.

## 3. Partial (masked) delay vectors
- Q: does the embedding theorem carry over to missing entries? **No.**
  O_B = Pi_B o F is a coordinate projection of the full delay map. Projection can destroy
  injectivity and immersivity.
  - Necessary for a local immersion: at least d observed rows and rank J_{O_B}(s) = d.
  - Global injectivity on A is not implied by any rank condition.
  - Generic results for irregular delay sets may exist in the literature; **[Open]**, not checked.
- Local condition [Verified here, numerically]: rank J_{O_B}(s) = d, via flow Jacobians
  d x_j(t - k tau)/d s(t) = e_j^T DPhi_{-k tau}(s(t)). Checked against finite differences of solve_ivp
  (tests/test_integrity.py, error < 1e-5).
  - Counts: rank < 3 occurs in 0 % of C's windows at p = 0.3 and 1.5 % at p = 0.7.
  - For B (own variable only): 2 % at p = 0.3 and 65 % at p = 0.7.
- What the numbers mean:
  - rank: local identifiability yes or no.
  - sigma_min: worst-case state sensitivity, |ds| <= |do| / sigma_min. It depends on the state
    coordinates.
  - amp = ||c^T R^+||, with c = d x_i(t)/d s(t): sensitivity of the imputed target to observation
    perturbations. It is invariant to reparametrising the state when R has full column rank, so it
    is the appropriate local quantity for imputing x_i(t).
  - All three are linearised and local. They say nothing about global ambiguity such as 2(b), or
    about model error.

## 4. Observability vs a learned inverse
- Observability means an inverse map exists (information is present).
- A network trained with masked MSE estimates E[x_i(t) | observed inputs, mask] under the training
  mask distribution. That equals the true value only if all of the following hold:
  - the target is identifiable from those inputs
  - there is no noise
  - capacity, optimisation and data are sufficient
  - the test mask patterns resemble training
- So observable does not imply learned, and a low error does not imply the inverse was learned.
- Evidence from these experiments:
  - 19 of 241 main runs hit the 150-epoch cap.
  - Exp B (gaps) is a shift in the mask distribution.

## 5. Why Model C's accuracy is not evidence of a learned diffeomorphic structure
[Ours] C = per-view encoders, observation-count-weighted mean, shared z (dim 16), per-variable decoders.
1. Nothing makes E_i injective or immersive on F_i(A), or makes z_i agree across views (no
   alignment loss, by design). z is an average whose meaning changes with the mask.
2. With tau = 1 the lag-1 autocorrelation is about 0.996 to 0.999. Accuracy can come from local
   smoothness: linear interpolation is competitive on Lorenz. Exp A varies tau to test this.
3. Exp C shows C relies on the other variables (dependence). Dependence is not structure, and
   input removal shifts the input distribution.
4. A diffeomorphism claim would need, at minimum: an injective E_i with full-rank differential on
   F_i(A); and decoders composing to approximately F_j o F_i^{-1} on held-out states. **Not tested.**
