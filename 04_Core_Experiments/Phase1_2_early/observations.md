# Observations for interpretation (written 2026-10-24 ~21:25; Exp A was still running)

Three categories, kept separate:
- **Verified**: reproducible computation; file given.
- **Empirical**: measured over 3 seeds; no claim beyond these settings.
- **Hypothesis**: not tested.

All RMSE values below are in train-z-score units, on artificially hidden test entries.

## Verified
- All 499 completed main rows (as of 20:40) were recomputed from their best checkpoints on CPU.
  - Max relative difference: 6e-6.
  - Masks identical; checkpoint configs and n_eval match.
  - File: results/verification/recompute_check.csv.
  - The queue repeats this after all runs finish.
- Leakage tests pass on the actual training and inference paths (tests/test_smoke.py, tests/test_integrity.py).
- Information sets per model, measured by perturbation:
  - A: time t only
  - B: own variable, lags t-7..t
  - C: all variables, lags t-7..t
  - D: all variables, t-7..t+7, i.e. **uses future steps**
  - linear: both sides of t
- Lorenz z-only delay maps are not injective (Z2 symmetry): math_review.md 2(b).

## Empirical
- **Lorenz, MCAR (main):**
  - C <= linear interpolation at p = 0.1 to 0.5 (e.g. p = 0.5: 0.025 vs 0.029).
  - linear is better at p = 0.7 (0.077 vs 0.125).
  - Differences at p = 0.1 are within one seed standard deviation.
  - B degrades sharply with p (p = 0.7: 0.34). A (no lags) is far worse at every p (0.44 to 0.90).
- **Exp B, contiguous gaps (eval-only generalisation, models trained on MCAR):**
  - At gap positions >= m steps into a gap, B and C have no own-variable observation in their window.
    - B ~ mean imputation (about 1.0), as it must be.
    - C: x 0.25, y 0.09, z 0.34 to 0.38 (p = 0.3); 0.31 to 0.46 (p = 0.7).
    - linear: 0.38 to 0.70 for 25-step gaps, about 1.0 to 1.3 for 100-step gaps.
    - A: 0.62 to 0.97.
  - So C recovers values from the other variables' past observations where its own are absent.
    Here C is better than linear for long gaps and worse for short gaps.
  - File: results/extra/expB_contiguous/overnight/.
- **Exp C, input ablation (same eval positions; inputs shift distribution, so not causal attribution):**
  - C at p = 0.3: full 0.010 to 0.013; others-only 0.09 to 0.38; own-only 0.20 to 0.28.
  - Either view alone is far worse than both together.
  - y is best recovered from the other variables (0.09).
  - File: results/extra/expC_crossview/overnight/.
- **Exp D, observability vs error:**
  - Spearman(amp, |err|) per target and seed:
    - C, p = 0.7: mean 0.21, 9/9 block-bootstrap CIs exclude 0.
    - C, p = 0.3: mean 0.05, 4/9.
    - B: 0.14 to 0.15, 8/9.
  - Rank-deficient windows have 4 to 7 times larger mean error (results/observability_audit_summary.csv).
  - Point estimates do not depend on the bootstrap block length.
  - With 400-step blocks there are only about 7 blocks, so those CIs are unreliable. The spread
    across seeds is the more conservative indicator.
- **Real data:**
  - EEG: A best (cross-sectional correlation); linear poor (0.8 to 1.2).
  - ENSO and Exchange: linear best.
  - Exchange: 62 % of variables leave the training range by > 5 sd in the test split. Models
    normalised on train extrapolate poorly. Pooled original-unit RMSE mixes currency scales; do not use it.
- 19/241 main model runs stopped at the 150-epoch cap (not converged by the early-stopping rule).

## Hypotheses (untested)
- C's advantage on y from others may reflect the identity y = x + x'/sigma. The weaker z recovery
  may relate to z = rho - (y' + y)/x being singular at x = 0, and to the Z2 symmetry.
- Good MCAR performance at tau = 1 is largely due to adjacent-sample autocorrelation. Exp A tests
  this; see results/extra/expA_tau/.
- The amp-error association at p = 0.7 means information availability limits C at high missingness.
  An association is not identifiability or diffeomorphism.
