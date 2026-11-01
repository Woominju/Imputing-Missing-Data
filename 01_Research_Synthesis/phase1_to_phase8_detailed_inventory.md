# Inventory — Phases 1–8 + FPRM (Imputing_Missing_Data)

Read-only inventory, 2026-10-31. Paths are relative to `/home/minju/faicpaston/Imputing_Missing_Data/`; `notes/` = `results/notes/`. Sources: `research_handover_latest.md` (§P8.Z, P8.*, P7.0, P6.*, §1–§10), notes, CSVs (primary numbers for Phases 5–8 were re-read from the CSVs cited). Phase 9 is documented in `notes/phase9_*.md` (main_results, mathematical_foundations/conclusions, model_design, preregistered_protocol, pilot_report, independent_verification, test_log.txt), `results/extra/phase9/`, `results/audit/phase9*/`, and is not inventoried here.

**Naming:** no record uses "Phase 1"; here it is everything before the explicitly-labelled Phase 2 documents (10-27). "Gate A–F" in the handover §4 are **Phase 5** gates. Phase 6–9 also use gate names (G0–G3, P7-A/B, Gate A–E) per phase.

**Labels:** INVALIDATED = number/claim later corrected (correcting doc given); UNVERIFIED = not confirmed in a file.

## Phase 1 — preliminary implementation + aim_autonomous (10-24 → 10-27 04:31)
- **Research question:** Can multi-view delay-embedding models recover MCAR-missing values when no variable is fully observed? Also, how do they compare with the FPRM premise of one fully observed reference variable?
- **Mathematical basis:**
  - `notes/math_review.md`: §1 Takens/SYC [Literature]; §2(b) z-only Lorenz delay map is never injective (Γ(x,y,z)=(−x,−y,z), SymPy); §3 masked delay vectors do not inherit embedding.
  - `notes/aim_autonomous_20261027.md` §C.2: Prop 1 (z-reference Bayes bound MSE ≥ E[x²]); Prop 2 (forced alignment loss erases sign); Prop 3 (mean fusion); coupled κ=0 product-space non-injectivity.
- **Models:** A (current-step MLP), B (per-variable delay AE), C (per-view encoders + count-weighted mean, shared z, dim 16), D (latent ODE, uses future) (`README.md` "Models"). Model E = C + view-dropout; Model F = learned mask-aware fusion (`experiments/aim_autonomous/`).
- **Conditions:**
  - Lorenz with 20000 points, dt=0.01 (also 0.05 and 0.1); coupled Lorenz with k∈{0,0.5,2}; real data EEG, ENSO, Exchange, Wind, Transportation.
  - MCAR p∈{0.1,0.3,0.5,0.7}; chronological 70/15/15 split.
  - Seeds 0–2. **Replicate unit = seed on one trajectory (mask/init), not independent trajectories.**
  - Baselines: mean, LOCF, linear (uses future). FPRM-T/FPRM-I and Adapted FPRM ran on the Python port.
- **Main results:**
  - Lorenz MCAR dt=0.01: C ≤ linear for p=0.1–0.5 (p=0.5: 0.025 vs 0.029); linear is better at p=0.7 (0.077 vs 0.125) (`notes/observations.md:27–28`). Unit: RMSE in train z-score, 3-seed mean.
  - Exp D: Spearman(amp,|err|) for C at p=0.7 = 0.21, 9/9 CIs exclude 0 (`observations.md:47`).
  - Condition B (no fully observed variable): C vs Adapted FPRM. At dt=0.05, C 0.0091/0.0179/0.0387/0.1424 vs linear 0.1472/0.2130/0.3517/0.5713 (`aim_autonomous_20261027.md:100–120`, table (1)).
  - Condition A, z-reference FPRM-T: 1.006–1.069 (symmetry failure, as Prop 1 predicts) (`:122–135`).
  - Coupled, x1-reference FPRM-T on subsystem 2: nRMSE 0.913–1.076 (`:167–186`).
  - F q=0 − C at p=0.7: −0.012…−0.019, 3/3 seeds (`:151`).
  - Exchange: 42.6% of test entries fall outside the train range (`:204`).
- **Negative results:**
  - The single-view path of C is poor: x view → others 0.23–0.46 vs FPRM-GP 0.053–0.059.
  - View-dropout E trades off multivariate accuracy.
  - ENSO and Exchange: linear interpolation is best.
  - The original FPRM paper numbers were not reproduced (§Summary 1).
  - 19/241 runs hit the epoch cap (`observations.md`).
- **AIM contribution:** Showed that the FPRM "one complete reference" premise fails (z symmetry; coupled κ=0). Showed that C works under all-variable MCAR. Shared-latent consistency is not achieved (§D table).
- **Evidence:**
  - Code: `src/`, `experiments/{run_lorenz,run_coupled_lorenz,run_real_data,run_observability,audit_observability,recompute_from_checkpoints,verify_results}.py`, `experiments/extra/{exp_b_contiguous,exp_c_crossview,exp_d_observability,aim_crossview,aim_latent,aim_multiview_rescue,aim_window_audit,aim_abc_compare}.py`, `experiments/aim_autonomous/`.
  - Results: `results/{lorenz,coupled_lorenz,real_data}_results.csv`, `results/extra/{expA_tau,expB_contiguous,expC_crossview,expD_observability,aim_crossview/v1,aim_latent/v1,aim_rescue/v1,aim_rescue/v1_abc,aim_autonomous_20261027}/`.
  - Ops: `results/extra/aim_autonomous_20261027/ops/FINAL_OPS_SUMMARY.md` (OK).
- **Verification:**
  - All 499 main rows were recomputed from checkpoints (max relative difference 6e-6; `results/verification/recompute_check.csv`, OK).
  - `notes/code_issues.md` items #1–#6 are fixed. #2 means the original "rho ~ 0" is not interpretable; v1 is kept as `observability_results_v1_before_audit.csv`. #3 means the i.i.d. bootstrap CIs were about 19% too narrow; superseded by block bootstrap.
  - Leakage and past-only behaviour were later confirmed by perturbation tests (`notes/phase25_mathematical_audit.md` §0, outputs differ by 0.0).
  - Phase 2.5 audit judged aim_autonomous §C claims (P-numbering in Phase 2 docs).
  - The aim_rescue, aim_latent and aim_crossview 10-25 analyses have no dedicated report. They are cited only in `notes/fprm_comparison_protocol.md` §3.1 and §7. Their numbers are **not inventoried (UNVERIFIED beyond CSV existence).**

## Phase 2 — mathematical foundation, architecture candidates, FPRM verification (10-27 10:10–10:29)
- **Research question:** Formalise identifiability under partial observation and design an architecture for a consistent shared latent.
- **Mathematical basis:** `notes/aim_mathematical_foundation.md` P0–P11 (P2 Γ non-identifiability; P4 joint-only identification; P7 point-consistency collapse; P8 mean-fusion dilemma; P9 MCAR→PoE; P10 tower; P11 injective decoder ⇒ consistency).
- **Models:** Design only. `notes/aim_architecture_candidates.md` lists FDR-AE (1a/1b), XV-AE (2) and PoE-FDR (3). **Recommended: PoE-FDR (§7). Not implemented.**
- **FPRM verification:** `notes/fprm_verification_phase2.md`.
  - Tests: 13/13 rerun on code snapshot `b5b8754f` (10:11), plus 3/3 new D8 jitter tests (§1).
  - **GP plateau defect:** 33/1,152 condition-A test rows have σ_L>100 with success=True (§0, §3.3). In the x→y example, RMSE was about 13× worse (0.021 vs 0.0016).
  - 45/576 condition-A fits ended with success=False.
  - The condition A/B comparison conclusions are unchanged, but FPRM x→y accuracy is underestimated in some seeds.
  - Lag-FPRM has no official code (§6).
- **Negative results:** None experimental.
- **Verification:** Audited in Phase 2.5. **Corrections:**
  - P9 (MCAR ⇒ Gaussian PoE exact) is **Overstated/withdrawn** (`phase25_mathematical_audit.md:237–251, :332`). This is handover §5 "P9 error".
  - The "ELBO optimum ⇒ q = true posterior" claim (AC §3.1) is **incorrect**: L_old is not a conditional ELBO (`phase25_objective_redesign.md:53`). This is handover §5 "held-out ELBO not conditional". It was design-only, so no numbers are affected.
  - P3 and P8 are Overstated (`:320–333`).
  - The PoE-FDR recommendation was withdrawn (`phase25_architecture_decision.md:137–141`).
  - The "FPRM tests not rerun" issue was corrected by `fprm_verification_phase2.md` §1.

## Phase 2.5 — audit, objective redesign, architecture decision (10-27 11:29–11:32)
- **Research question:** Which Phase 2 claims hold, and which objective and architecture should be tested first?
- **Mathematical basis:**
  - `phase25_mathematical_audit.md`: verdict table at :320–333; open problems U1–U8.
  - `phase25_objective_redesign.md`: method A, the observed-data ELBO.
  - `phase25_literature_review.md` §5 (Deyle–Sugihara).
- **Diagnostics (CPU):**
  - `results/extra/phase25_diagnostics/{leakage_audit,pair_consistency,elbo_toy}`.
  - Leakage perturbation tests L1–L3: output difference 0.0.
  - C latent mismatch for non-Γ pairs is 0.47–1.38× the time scale (`phase25_research_roadmap.md` F6).
- **Decision:** Implement deterministic FDR (Architecture A) first. Probabilistic C′ is conditional (`phase25_architecture_decision.md` §8).
- **AIM contribution:** Showed that unconditional point consistency collapses information (P7). Identifiability is a property of the (pattern, state) pair (P6).
- **Verification:** Self-audit only. No external verification is recorded.

## Phase 3A — decoder fibre diagnostics + 2×2 pilot (10-27 11:38–11:47)
- **Research question:** Is C's latent inconsistency caused by decoder non-injectivity, missing view information, or OOD masks?
- **Basis:** P11′ (`phase25_mathematical_audit.md`).
- **Diagnostics:** 12 existing checkpoints (C p∈{0.3,0.7} × 3 seeds; E Q=0.5), CPU only (`phase3a_decoder_diagnostics.md`).
- **Pilot:**
  - Design: k∈{3,16} × {C,E}, Lorenz dt=0.01, p=0.7, **single seed, 4 runs**.
  - **Results:** official test RMSE C16 0.116, C3 0.152, E16 0.117, E3 0.151. Decoder σ_min median: C3 0.047 vs C16 0.50 (`phase3a_pilot_results.md` §3.1, §3.4).
  - **Negative result:** k=3 does not give an injective decoder, and output agreement worsens (d_R 0.121→0.180).
  - Decoder diagnostics: the full-observation pattern does worse than the natural mask (C p=0.7: 0.22 vs 0.095). This points to an OOD cause (`phase3a_decoder_diagnostics.md` §6).
- **Plan:** `phase3a_fdr_implementation_plan.md` (design only).
- **Novelty:** Lv et al. 2025 full text not obtained. `third_party/lv2025/` is **MISSING** (`phase3a_novelty_update.md`).
- **Evidence:** `results/extra/phase3a_decoder_diagnostics/` (fiber_diag.py, patterns.csv, pilot/pilot_runs.csv). Pilot C16/E16 matched the archived checkpoints to the last digit.

## Phase 3B — FDR vs C pilot, fully observed Lorenz dt=0.05 (10-27 12:05–12:11)
- **Model:** FDR = C + per-view full delay-vector decoder, with 1/c(u) multiplicity weighting (`experiments/fdr/{model,loss,impute}.py`). Tests 7/7 (`results/extra/phase3b_fdr_pilot/test_fdr.log`).
- **Conditions:** p=0.7, seed 0, Q∈{0,0.15}, 4 runs. C0 and FDR0 hit the 150-epoch cap.
- **Results:** test RMSE C0 0.1544, C15 0.1442, FDR0 0.1526, FDR15 0.1493. All differences fall within the seed range of 0.023 (`phase3b_pilot_results.md` §2). Latent d_z q50 0.39→0.34, but d_R is not improved (0.149 vs 0.154) (§3, §7).
- **Negative result:** No coordinate-invariant consistency gain.
- **Verification:** `phase3b_implementation_verification.md`. C0 reproduced the archived checkpoint bit-for-bit.

## Phase 3C — partial observation (x,z observed, y hidden), Lorenz dt=0.05, p=0.3 (10-27 12:28–12:46)
- **Research question:** Do the latents encode the unobserved y consistently, without using y?
- **Basis:** `phase3c_observability_audit.md` (Γ, y = x + ẋ/σ). AE is trained without y; y is used only in probes.
- **Models:** C-xz and FDR-xz (seeds 0–2); C-x and FDR-x (seed 0). 8 runs, capped at 300 epochs.
- **Unit:** Seed on one trajectory, with block bootstrap on the test segment (100 steps × 30).
- **Results:**
  - Imputation C-xz 0.0218/0.0294/0.0364 vs FDR-xz 0.0315/0.0329/0.0433. FDR is worse in 3/3 seeds (`phase3c_experiment_results.md:39–47`).
  - y-probe MLP: FDR 0.035/0.040/0.050 vs C 0.045/0.061/0.061. kNN FDR−C is −0.021/−0.032/−0.034, CIs exclude 0. Ridge goes the opposite way (`:52–73`).
  - Adding z roughly halves y error (`:79–100`).
- **INVALIDATED (scope):**
  - The "FDR better same-state consistency" claim (d_R 0.035–0.043 vs 0.046–0.054, `phase3c_state_consistency.md` §1 and §5(a); `phase3c_research_decision.md` §2 item 3) is invalidated as stated.
  - The phrase "first quantitative evidence for AIM shared latent" is also invalidated.
  - Reason: the metric mixed observed coordinates, which are unsupervised, with missing coordinates.
  - On missing coordinates, C is better. **Corrected in** `phase3d1_evaluation_integrity.md` §5 and `phase3d1_error_decomposition.md` §1–2.
- **INVALIDATED rows:** 18 `*_zremoved_refit` rows in `eval_probe.csv` are actually x-removed results. Use `eval_probe_zremoved.csv` or `phase3d1_audit/probe_summary_safe.csv` (`phase3c_experiment_results.md` §7.1; `phase3d1_evaluation_integrity.md` §2). The `analysis.log` z-removed lines are invalid.
- **Future-y boundary leak:** Affects at most D rows; value impact ≤ 2.3e-4; conclusions unchanged (`phase3d1_evaluation_integrity.md` §3; fixed values in `eval_dynamics_boundary_fixed.csv`).
- **Evidence:** `results/extra/phase3c_hidden_state/` (train.py, evaluate.py, runs.csv, eval_*.csv, recovery/RECOVERY_MANIFEST.md); `experiments/fdr/xz_data.py`.

## Phase 3D-1 — CPU audit of 3C (10-27 15:54–15:57)
- **Results:** On missing coordinates, C 0.022/0.029/0.036 vs FDR 0.032/0.033/0.043. C−FDR CIs are below 0 in 3/3 seeds. On observed coordinates, C's error is 2–3× larger, which reverses the comparison (`phase3d1_error_decomposition.md:33`).
- On coordinates missing under both masks, C is more accurate and more consistent (§2).
- Analytic lag-0 loss share is 13.7% (effective about 11–12%).
- The MLP probe was single-init. This was resolved in 3D-2.
- **Evidence:** `results/extra/phase3d1_audit/` (safe_aggregate.py, obs_miss_errors.csv, samestate_decomp.csv, eval_dynamics_boundary_fixed.csv).
- **Verification:** CPU recomputation matched within 3.2e-6 (outputs) and 5e-8 (RMSE). Ridge and kNN probes are bit-identical.

## Phase 3D-2 — FDR lag-weight ablation (10-27 ~16:53)
- **Research question:** Can raising the lag-0 weight α give both good imputation and y information?
- **Conditions:** α∈{1/8, 0.35, 0.65, 1.0} plus a capacity-matched control Ccap. Seeds 0–4 on the same trajectory. 25 new runs plus reused 3C checkpoints.
- **Results:**
  - Missing-position RMSE: C 0.0299, Ccap 0.0311, FDR1/8 0.0361, FDR0.35 0.0396, FDR0.65 0.0384, FDR1.0 0.0359.
  - y MLP: C 0.056, FDR1/8 0.042, FDR1.0 0.063.
  - FDR1/8 vs C: imputation worse in 5/5 seeds; y better in 5/5 (`phase3d2_fdr_lag_weight_results.md:55–74`).
- **Negative result:** The trade-off is not resolved, and the dilution hypothesis H3 is not supported (§3, §5). Seed 3 stopped early.
- **Finding:** The hidden-state advantage comes from the delay targets, not from capacity or output dimension (FDR1.0 and Ccap lose it).
- **Evidence:** `results/extra/phase3d2_lag_weight/` (paired_contrasts.csv); `experiments/fdr/lagweight.py`; tests 8/8 (`experiments/fdr/tests/test_lagweight.py`).

## Phase 4 — coupled Lorenz pilot (10-27 16:22–16:55)
- **Research question:** Does combining incomplete multi-view delays (x1,z1,x2,z2; y1,y2 hidden) improve identifiability and recovery over single views?
- **Basis:** `phase4_coupled_observability.md`. At κ=0 a subsystem view has rank 3 and cannot determine s2. At κ=1 the rank is 6 but y2 amplification is in the hundreds. h± is a local-rank-6 but globally non-injective counterexample.
- **Design:** decoder {C, FDR} × fusion {Scalar, Learned}; κ∈{0,1}; p∈{0.3,0.7}; seeds 0–2 on trajectory 0; 48 runs; dt=0.05.
- **Results:**
  - Decoder effect flips sign with p. At p=0.7, FDR − C = −0.022…−0.035, 6/6 seed-conditions significant; at p=0.3, C is better (`phase4_coupled_pilot_results.md` §2).
  - At p=0.3, FDR-Learned y1 0.055–0.057 and y2 0.041–0.043, better than the raw own-subsystem window 0.078–0.099 (§3).
  - Learned fusion improves the y kNN probe in 12/12 cases.
- **Negative results:**
  - At p=0.7, imputation is 0.33–0.37 and the latent has no advantage over raw windows.
  - At κ=0 and at p=0.7, combining views hurts own-subsystem y (§5, §8).
- **Limitation / correction:**
  - The view-removal comparison was OOD (§5, §7.5). Handover §5 states it was replaced by the Phase 5 Arm A/B design.
  - Only one trajectory, so the replicates are not independent.
  - FPRM was not run.
- **Evidence:** `results/extra/phase4_coupled_pilot/` (factorial_contrasts_pilot.csv, view_removal_mlp.csv, KAPPA_SELECTION.md); `experiments/coupled/{data.py,tests/test_leakage.py}` (3/3).

## Phase 5 — Controlled multi-view experiment (H1/H2/H3) + Gate F real data

- **Dates:** 2026-10-28 (protocol frozen 10:50; Gate F §10 frozen 10:46; Gate E done 12:21, Gate F 12:18; results doc 12:55; review package 13:11). Source: `research_handover_latest.md` §4, §10.
- **Research question (AIM):** does feeding the other subsystem's incomplete delay views into the AE latent (Arm B) improve decoding of the own hidden y_j over an own-subsystem-only AE (Arm A), with identical reconstruction targets? H1 = observation-level complementarity; H2 (primary) = learned use; H3 = one shared readout is accurate/stable across admissible masks (L_j ≥ 2). Negative control κ=0. (`results/notes/phase5_hypothesis_protocol.md` §1)
- **Mathematical basis:** `results/notes/phase5_mathematical_foundations.md` — Prop. 1 (pointwise non-identifiability, l.87), Lemma S (Γ×Γ and P symmetric for all κ; Γ×Id only κ=0, l.107), Cor. 1 (z-only masks globally 2:1, l.121), Cor. 2/3 (κ=0, x_j unobserved: sign not identifiable; Bayes estimate 0), Prop. 2 (L² projection; dependence ≠ risk reduction), Prop. 3 (κ=0 other-view Bayes gain 0, l.205), Prop. 5 (y_j = x_j + (ẋ_j − κ(x_o − x_j))/σ: κ=1 instantaneous-identity gain, l.270), Prop. 6 (DPI: latent ≤ raw, l.293). Takens/SYC table §4 (l.249). Gate C numerics: `results/notes/phase5_observability_information_audit.md`.
- **Models:** FDR-Learned ArmModel (`experiments/phase5/core.py`), Arms A (own), Aslot, B (own+other), Bcur (other current value only), Sham (other view from donor trajectory), C (all-view target); params A 31.5k / B 46.3k / C 57.8k (handover §5).
- **Conditions:** synthetic coupled Lorenz κ∈{0,1}, MCAR p∈{0.3,0.7} on full grid then windowed, m=8, τ=0.05; replicates r=0..7 per κ (train id 0–7, val 100–107); **test ids 200–209 shared by all replicates and Arms** (protocol l.38); stat unit = replicate (n=8, paired t, df=7) + crossed bootstrap. Model selection uses masked observed val entries (no ground truth). Gate F: Beijing, Jena × p{0.3,0.7} × 3 seeds × (MV, MV-shift, SV_k) + LOCF/linear; unit = seed (1-week block bootstrap).
- **Main results** (`results/notes/phase5_controlled_experiment_results.md`; CSV `results/extra/phase5/analysis/contrasts_probe_mlp.csv`, checked):
  - H2 primary B−A (κ=1,p=0.7) **+0.0137** [−0.0015, +0.0288], 1/8 → inconclusive, not supported (l.62).
  - κ=1,p=0.3: −0.0128 [−0.0158,−0.0098], 8/8, < δ=0.02 (l.63). κ=0,p=0.7 control: +0.0234 (OPPOSITE) (l.64).
  - Sham−A (p=0.7): +0.0200 (κ=0), +0.0286 (κ=1) = input-column cost (l.109).
  - Post-hoc B−Sham κ=1: −0.0133 (p=0.3), −0.0149 (p=0.7), 8/8 (l.127) — exploratory.
  - L0 stratum (κ=1,p=0.7) B−A −0.0277, 8/8 (l.84), but RMSE ≈1.0.
  - H3: strong form rejected; y_j shared readout ≈0.04 at p=0.3; s_j ridge error 0.44–0.68 (l.197–205).
  - Gate F: Beijing p=0.7 MV−MV-shift **−0.137** (−0.141~−0.130), 3/3 seeds CI<0; g≥8: −0.386; Jena all negative (l.245–255; `results/extra/phase5/external/analysis/summary.csv`).
  - Gate C: κ=1 identity x_o(t) cuts y error 0.113→0.061; L0 y1 amplification 168 → 5.4 (MCAR)/0.53 (full) (handover §4; audit l.100–103).
- **Negative results:** H2 not supported; H1 not detected (raw_joint worse than own — estimator failure); H3 strong form rejected; seed variability ≈±0.01 ≈ effect size (results l.230).
- **AIM contribution:** shows input-cost dominates other-view benefit in a shared-latent AE; aligned multi-view helps prediction on real data (not hidden-state recovery).
- **Evidence:** code `experiments/phase5/` (gen_data, obs_audit, core, train, evaluate, eval_scheduler, h1_capacity, analyze, analyze_posthoc, external, analyze_external, tests/); results `results/extra/phase5/{analysis,eval,h1,gateC,external,runs,data}`, `results/extra/phase5_external_audit/`; package `results/review_packages/phase5_20261028/{phase5_review_documents.zip, phase5_verification_bundle.zip (1,016 files, MANIFEST.json), SHA256SUMS, build_bundle.py}` (all present).
- **Verification / errors:** bundle re-run byte-identical, tests 11/11 (handover §10). Later errata (`results/notes/phase6_mathematical_errata.md`): **E6** Sham donors = other replicates' real trajectories → B−Sham t-CI not independent (ρ̂₁=+0.43 at κ=1,p=0.7; CI upper −0.0006 at ρ=0.5) → **post-hoc B−Sham CIs INVALID as confirmatory** (numbers kept); **E7** Gate F MV-shift is not past-only (25–74% of other-view inputs are future in Beijing) → "alignment gain" narrowed to "better than an alignment-broken stress control", MV-shift−SV interpretation suspended; **E8** decide() checks TOST before OPPOSITE (κ=0,p=0.3 labelled EQUIV with CI>0); E1 "exactly 2:1" → "at least 2:1"; E2 domain of D; E3 Bayes props ≠ estimator expectations ("κ=0 B−A=0 expected" retracted); E4 fixed-t vs time-mix; E9 "sufficient" withdrawn. Also: shared test set across replicates (protocol l.38) — replicate CI conditional on that test set (not stated as an error in records; flag). Bugs fixed during pilot: A2 output index, lane rc logging (handover §5).

## Phase 6 — Mathematical integrity; separate H1 (information), H2 (use), H3 (coordinates)

- **Dates:** 2026-10-28 (G0 14:58; G1 frozen 15:08, done 15:27; G2/G3 frozen 15:04, done 19:24; diag J/S 20:07; S1R frozen 20:10, done 20:48; final report 20:50; review package 21:16).
- **Research question:** does the other view carry Bayes information about y_j (H1); does the model use it (Real vs Sham, H2); does the latent form a mask-invariant 6-D coordinate (H3)?
- **Mathematical basis:** `results/notes/phase6_mathematical_errata.md` (E1–E11), `results/notes/phase6_g1_protocol.md` §2 (G1.1 symmetry of generator law, no SRB; G1.2 L0: E[y_j|U_j]=0, R*(U|L0)=E[y²], κ=0 Δ*_L0=0; G1.3 Δ* ≥ p⁸Δ*_L0; G1.4 witness bound; G1.5–6 cross-fit bias = product of errors); Prop. E3.1 (κ=0, independent donor ⇒ E[B−Sham]=0 at estimator level). Counterexamples: handover §P6.1 table.
- **Models:** supervised G1 estimators (oracle diagnostics); AEs A, Aslot, B, Bcur, Sham × j (`experiments/phase6/core6.py`, `train6.py`); shared 6-D MLP readout for G3.
- **Conditions:** new trajectories seed base 30261029 (`gen6.py`); 24 replicates/cell, each with independent train/val/test (8 test) and donors; κ{0,1} × p{0.3,0.7}; 960 G2 trainings + 192 diag + 192 S1R; G1: κ×p{0,0.3,0.7}, 32 test trajectories, cluster bootstrap. Fixed sequence S1 (B−Sham κ=1,p=0.7) → S2 DiD → S3 (κ=1,p=0.3); δ=0.02; negative control κ=0.
- **Main results:**
  - G1: Δ*_L0 LCB **0.0322** (κ=1,p=0.7), Δ* ≥ p⁸×0.032 = **0.0019** (`phase6_g1_results.md` l.22,31); L0c0 0.0353 [LCB 0.0274] (l.46); R*(U) ≥ 0.0576 (l.80). CSV `results/extra/phase6/g1/analysis_g1.csv`.
  - G2 S1 B−Sham **−0.0070** [−0.0116,−0.0024] 19/24; S2 −0.0132; S3 **−0.0151** [−0.0162,−0.0139] 24/24 (`phase6_g2_g3_results.md` l.23–25; `results/extra/phase6/analysis/g2_fixed_sequence.csv`, checked).
  - **Negative control κ=0,p=0.7 FAILED:** +0.0062 [+0.0022,+0.0101] (l.36) → S1 held.
  - S1R (κ=1) **−0.0104** [−0.0141,−0.0068] 23/24; κ=0 +0.0009 [−0.0034,+0.0051] (l.187–188; `results/extra/phase6/s1r/s1r_results.csv`, checked).
  - Original H2 B−A (κ=1,p=0.7) **+0.0123** [0.0085,0.0161] — B worse (l.53).
  - G3: no AE latent met RMSE_s6 ≤ 0.2 ∧ SD ≤ 0.1 ∧ better than raw; raw joint readout 0.08 (p=0.3)/0.37 (p=0.7) (`phase6_final_report.md` Q4 l.84–89).
  - Diag J −0.0007, S −0.0003 (numerical coupling rejected; handover §P6.7).
- **Negative results:** H2 (B>A) unsupported in two samples; G3 no shared coordinates; κ=0 control failure unexplained (Unresolved, still open in Phase 8/9).
- **AIM contribution:** first rigorous (symmetry-based) evidence that incomplete other-view delays carry hidden-state information; separates information vs use vs coordinates.
- **Evidence:** code `experiments/phase6/` (g0_checks, g1_info, analyze_g1, gen6, core6, train6, evaluate6, eval_sched6, jac6, analyze6, run_lane6.sh, diag/, s1r/, tests/); results `results/extra/phase6/{g0,g1,analysis,eval,jac,diag_k0,s1r,runs,data}`, freezes `PROTOCOL_FREEZE_g1.txt`, `PROTOCOL_FREEZE_g2g3.txt`, `diag_k0/FREEZE.txt`, `s1r/FREEZE.txt`; package `results/review_packages/phase6_20261028/{phase6_review_documents.zip, phase6_verification_bundle.zip (3,423 files), _checks.md, SHA256SUMS, build_bundle.py}`.
- **Verification / corrections:** bundle re-analysis byte-identical, tests 9/9+8/8+3/3 (handover §P6.8). Independent review → `results/notes/phase7_mathematical_corrections.md` §A5: (i) "Δ*>0 Verified (math)+LCB, no estimation-error assumption" **DOWNGRADED** to Empirical (approximate cluster-bootstrap LCB, finite-sample coverage unproven); (ii) L0c0 "from delayed observations" → Prop. P7.1, mechanism Unresolved; (iii) "S1R → S1 independently confirmed" **WITHDRAWN** as confirmation of original family (S1R post-hoc-designed new replication); (iv) S3 "pre-registered confirmed" → individually significant, **not FWER-confirmed**; S2 descriptive; (v) G3 conclusion restricted to self-target AEs; raw-superiority not a necessary condition (DPI). Execution errors (seed overlap with Phase 5; S1R SyntaxError) caught before training — no effect (handover §P6.3).

## Phase 7 — P7-A corrections + P7-B B-full vs B-self supervision

- **Dates:** P7-A 2026-10-28 21:38; P7-B frozen 21:40, run 10-28 21:40 → analyze7 23:06; final report 2026-10-29 09:10.
- **Research question:** does full multi-view reconstruction supervision (B-full) make the latent preserve the 6-D coupled state better than self-subsystem supervision (B-self)? (`results/notes/phase7_final_report.md` §1 l.18–19)
- **Mathematical basis:** `results/notes/phase7_mathematical_corrections.md` — A1 (Δ* ≥ p⁸ G_L0(f) Verified math; Δ*>0 Empirical), A2 Prop. P7.1 (E[y_j|U_j,V_cur,L0c0]=0), A3 (G2 inference scope), A4 (DPI: latent ≤ raw). Recompute `experiments/phase7/p7a_recompute.py` → `results/extra/phase7/p7a/g1_recompute.csv` (L0 0.041065 / LCB 0.032181; handover §P7.A).
- **Models:** Bself1, Bself2, Bfull — same 4-view input, same 4-view decoder, 57,840 params, same init/masks; only loss-active columns differ (`experiments/phase7/core7.py`, `train7.py`).
- **Conditions:** new ids 40000+; 24 replicates × κ{0,1} × p{0.3,0.7}; 288 trainings; per replicate train 1 (20k), val 1, test 8 × 5 masks; primary Q = RMSE_s6 via shared 2×128 MLP readout; Δ = Bfull − mean_j Bself_j; fixed sequence S1(1,0.7)→S2(1,0.3)→S3(0,0.7)→S4(0,0.3), δ₆=0.05; unit = replicate (n=24).
- **Main results** (`phase7_final_report.md` §4.1 l.85–88; `results/extra/phase7/analysis/p7_fixed_sequence.csv`, checked): S1 **−0.3702** [−0.3755,−0.3649]; S2 −0.2552; S3 −0.4092; S4 −0.6617; all 24/24, CI upper < −δ₆. Own subsystem worse under Bfull at p=0.7 by +0.022–0.037 (24/24) (§4.2 l.110–112). Bfull − raw_joint −0.035…−0.061 (§4.4). Mask SD 0.27–0.28 ≈ RMSE 0.31–0.33 at p=0.7 (§4.5). Own-L0 y_j Bfull 1.019 (κ=1,p=0.7) — worse than zero prediction (§4.10 l.194–198).
- **Negative results:** gain identical at κ=0 ⇒ not evidence of coupling use (M2); p=0.7 H3 (G3 criteria) fails all strata; L0-certified information not realized by shared readout; imputation accuracy not evaluated (§4.9).
- **AIM contribution:** single-variable (loss column) pre-registered evidence that self-AE failure to encode the other subsystem is caused by the objective; gain = supervision coverage, not coupling.
- **Evidence:** code `experiments/phase7/` (core7, train7, gen7, jac7, evaluate7, eval_sched7, analyze7, run_lane7.sh, auto_analyze.sh, p7a_recompute, verify7, tests/); results `results/extra/phase7/{analysis,verify,p7a,eval,jac,runs,data}`, `PROTOCOL_FREEZE_p7b.txt`; packages `results/review_packages/phase7_20261029/{phase7_review_documents.zip, phase7_verification_bundle.zip}`.
- **Verification:** independent reimplementation `verify7.py` reproduces primary means/CI exactly (|diff|=0.0, report §3 row 10); tests 6/6. Deviations disclosed: run_lane7.sh/eval_sched7.py hashes added after first training start but before first result; "p difference" auxiliary implemented post hoc (§3). No later invalidation found in records.

## Phase 8 — Information vs model limit; coupling-specific information access (CSIA) vs factorized baseline

- **Dates:** protocol v1 2026-10-29 (self-review 10:05), v2 Gate A, Gate B 18/18→21/21 tests, Gate C freeze 11:16 (813 hashes, `PROTOCOL_FREEZE_p8.txt` SHA-256 86243a77…), run 10-29 11:17 → paused 13:19 (432/1276) → resumed 10-31 16:23/16:26 (GPU 0–3) → done 18:19:44; post-run audit 2026-10-31 (handover §P8.Z.1).
- **Research question:** (I) is the B-full failure on hard stratum H an information limit or a model limit? (II) does the coupled B-full encoder access coupling-specific information better than factorized encoders (Fact8/Fact16)? Plus imputation (III) and latent diagnostics (IV). (`results/notes/phase8_mathematical_protocol_v2.md` §0)
- **Mathematical basis:** `phase8_mathematical_protocol_v2.md` §2–§5: B1 (E_A(h) ≥ R_A(h) − R_A(f_oracle)), B3 (latent−raw readout gap ≠ Bayes loss), B4, §2.3 identity (MSE = mask variance + bias), L1–L3 (symmetry group, absorbing ball/flow diffeomorphism, P ≠ ν), N1 symmetric lower bounds (X0 ≈ 4/6, κ=0 L_j0 ≈ 2/6; exact only under variance match), counterexamples C1–C5, P8.1–P8.4. Corrections pre-freeze: `phase8_protocol_review_notes.md` R1–R5, `phase8_revision_crosswalk.md`. Conclusions: `phase8_mathematical_conclusions_20261031.md`.
- **Models:** Bfull (57,840; latent 16), BfullSham_j (donor other view), Fact8 (2×27,944 = 55,888; latent 8+8), Fact16 (latent 32; 31,552/encoder per audit B2); oracles O_raw, O_imp, O_own_j, O_lat (4-layer MLP), k-NN; readout R1 (2×128). Code `experiments/phase8/{core8,gen8,train8,feats8,oracle8,knn8,evaluate8,eval_p7imp8,analyze8,sched8}.py`, `launch_p8.sh`.
- **Conditions:** κ{0,1} × p{0.3,0.7}; 24 replicates; ids train 50000+r, test 50200+10r+i (8 test), donors 51000+, oracle pool 60000+, k-NN ref 70000+ (protocol v2 l.293–298); 1,276 tasks (864 trainings); unit = replicate r (trajectories/masks pooled); FWER: Family I α=0.025, IIa/IIb 0.0125 each, fixed order; δ_I=0.02 (MSE), δ_y=0.02 (RMSE), ε=0.01 TOST; oracle-pool conditional.
- **Main results** (`results/notes/phase8_postrun_independent_audit_20261031.md` §0 l.27–37, §D1 l.218–221, §E l.305–311, §F l.345–347; CSVs `results/extra/phase8/analysis/complete_20261031-181946_0f246de749/p8_familyI_sequence.csv`, `p8_familyII_sequence.csv`, `p8_gates.csv` — checked):
  - **Family I:** I-1 G_tot **0.04412**, LCB 0.04117 > δ_I → PASS; I-2 G_lat **0.01898** [LCB 0.01746, UCB 0.02050] → FAIL (indifference zone); I-3/I-4 untested (κ=0 descriptive 0.0542, 0.0270). X0: ≥83% of B-full risk is information limit (LB 0.6679).
  - **Gates** IIa/IIb (κ=0,p=0.7): PASS/PASS, TOST 90% CI ⊂ ±0.01, ε/SD 1.96/1.79.
  - **Family IIa:** IIa-1 CG_Bfull **−0.00863** [UCB −0.00548], 21/24 → PASS but < δ_y; IIa-2 C_Bfull **0.03337** [LCB 0.02447], 24/24 → PASS (P8.4b certified).
  - **Family IIb:** IIb-1 Δ_JF **−0.00278** [UCB +0.0000286] → not detected (borderline); IIb-2 untested, descriptive RMSE_s6(Bfull)−(Fact8) **+0.01764** [0.01241,0.02287] (B-full worse; vs Fact16 +0.0248).
  - **Family III (descriptive):** missing-position RMSE κ=1,p=0.7 Bfull 0.3121 vs Fact8 0.2959, Fact16 0.2932, LOCF 0.9303, zero 0.9979, O_imp (supervised) 0.2595; Fact8−Bfull −0.0161 [−0.0199,−0.0124] 23/24; κ=1,p=0.3 +0.0120 (B-full better) (audit §G l.398–410).
  - **Family IV (exploratory):** ρ_φ Bfull−Fact8 κ=1 −0.0205 [−0.0245,−0.0164], κ=0 −0.0224; readout mask variance R1:Bfull−Fact8 +0.0050 [0.0038,0.0062] (audit §H l.438–440).
- **Negative results:** coupled encoder superiority over factorization not confirmed; B-full worse in 6-D decodability and p=0.7 imputation; GSR (b),(c) unmet, (d) also at κ=0; O_raw unsaturated (R*_H location unknown); IIa-1 effect 43% of δ_y; Phase 6 κ=0 failure still Unresolved (handover §P8.Z.3).
- **AIM contribution:** establishes "incomplete other view carries coupling-specific information and the pipeline uses part of it"; does not establish shared-latent integration superiority or state correspondence (`phase8_mathematical_conclusions_20261031.md` §5).
- **Evidence:** results `results/extra/phase8/{analysis/complete_20261031-181946_0f246de749, eval, oracle, runs, data, logs, failed, FREEZE_CHANGES.md, RESUME.md, PROTOCOL_FREEZE_p8.txt}`; audit `results/audit/phase8_postrun_20261031/` (verify_p8_independent.py, audit_execution.py, verify_p8_supplement.py, verify_p8_oown.py, posthoc_driver_sensitivity.py, out/, logs/, freeze_evidence/, README.md); bundle `results/audit/phase8_postrun_verification_bundle_20261031.zip` (10.7 MB) + `_FULL_DATA_LIST.txt`; review packages `results/review_packages/phase8_protocol_review_20261029{/,.zip}`, `phase8_final_review_20261029.zip`; Gate B logs `results/notes/phase8_gateB_test_log{,_v2,_v3}.txt`.
- **Verification:** independent recomputation 54/54 checks True (`results/audit/phase8_postrun_20261031/out/verification_checks.csv`, verified count); masks regenerated (368,640 strata counts exact), O_raw forward pass re-implemented (max rel err 5.5e-6). **UNVERIFIED (L1 only):** R1:Bfull and O_lat SSE (readouts not stored; audit §C1 l.200–203, F11); M-1 Bernstein auxiliary not implemented (F1). Documentation defect F3 (second launcher change missing from v2 §14). Driver change 580.105.08→580.196.07 effect unverifiable; IIb-1 flips to "detected" when mixed-driver replicates dropped (post hoc, decision unchanged; audit §B3). ChatGPT re-audit pending (handover §P8.Z.7).

## FPRM work (10-25 protocol → 10-27 verification)
- **Protocol:** `notes/fprm_comparison_protocol.md` (10-25). Condition A: one fully observed reference. Condition B: all variables MCAR, where FPRM is structurally inapplicable and only Adapted FPRM is used. Lag-FPRM is excluded (no specification).
- **Port:** `baselines/fprm/{delay,gp,fprm,ode45}.py`, 13 tests (`notes/fprm_python_port_report.md`). MATLAB was never run, so there is no direct numerical cross-check.
- **Paper reproduction:**
  - Repro grid: 32,400 runs; 0 errors. On the official ode45 data, ρ_y 0.997–0.999 vs 0.90 in the paper.
  - Uniform dt=0.15 gives ρ_y 0.899 and ρ_z 0.777 (paper 0.90 and 0.81).
  - The cause of the gap is a hypothesis only (`aim_autonomous_20261027.md` §B.3(5)).
- **Comparisons:** Condition A, B, coupled and real-data results are given in Phase 1 above. Result dirs: `results/extra/fprm_comparison/{v1,v2/{repro_paper,repro_sensitivity,cond_a,cond_b,coupled,real}}` (all OK).
- **Verification:** `fprm_verification_phase2.md` (tests rerun, plateau, jitter); outputs in `results/extra/fprm_verification_phase2/` (census.csv, plateau_check.csv).
- **Unresolved** (§7 of that report):
  - No MATLAB cross-check.
  - The optimisation state was not stored for condition B, coupled and real runs.
  - The handover §P9.B still lists "FPRM fair comparison not performed" (no Lag-FPRM; the D6′ defect awaits a user decision).
- **Licences:**
  - `third_party/fprm_official/PROVENANCE.md`: Figshare 10.6084/m9.figshare.30446765 v3, **CC BY 4.0**.
  - `third_party/fprm_paper/PROVENANCE.md:8`: "Kept for reference only; **do not redistribute**."

## Summary
- Phases 1–4 (10-24 → 10-27): exploratory, single-trajectory seed replicates (not independent). Established that the FPRM fully-observed-reference premise fails under symmetry/κ=0, and that FDR delay targets trade imputation for hidden-y information. Phase 3C "consistency" evidence is INVALIDATED (3D-1).
- Phases 5–8 (10-28 → 10-31): pre-registered, independent-replicate designs. Main confirmed facts: other-view information exists (Phase 6 G1, now Empirical; Phase 8 IIa-2). The model uses it only weakly (S1R −0.0104; IIa-1 −0.0086, both < δ). Full-view supervision is what makes the latent encode the other subsystem (Phase 7, equal at κ=0). Model-limit gap ≥0.041 MSE (Phase 8 I-1). The coupled encoder is not shown better than factorized encoders (IIb-1); descriptively it is worse in 6-D decodability and in p=0.7 imputation.
- Open across phases: Phase 6 κ=0 control failure, shared-latent state correspondence, injectivity/diffeomorphism, FPRM fair comparison (never done).
