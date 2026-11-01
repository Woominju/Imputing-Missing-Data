# Phase 3D-2 + Phase 4 pilot — completion summary (2026-10-27)

| item | status | evidence |
|---|---|---|
| Track A training (25 runs) | 25/25 completed, rc=0, patience stop, no NaN | ops/PILOT_STATUS.md, phase3d2_lag_weight/rows, lanes |
| Track A evaluation + analysis | done (rc=0) | phase3d2_lag_weight/eval_trackA.csv, paired_contrasts.csv, ops/analyze_A.log |
| Track A reproduction check | FDR alpha=1/8 seed 0 == Phase 3C FDR-xz seed 0 (max param diff 0.0) | Document A §1 |
| Track B training (48 runs) | 48/48 completed, rc=0, patience stop (max 227 epochs < 300), no NaN | ops/PILOT_STATUS.md |
| Track B evaluation + analysis | done (rc=0); recorded vs recomputed test RMSE <= 2.5e-8 | eval_models_pilot.csv, eval_controls_pilot.csv, factorial_contrasts_pilot.csv |
| Post-pilot CPU checks | classical baselines, within-model view removal (kNN + MLP) | classical_baselines_pilot.csv, view_removal_*.csv |
| Leakage | tests passed before training; kappa=0 cross-subsystem probe ~1.0 (mean level) | tests_leakage.log, Document C §0 |
| Expansion (64 runs) | NOT run (awaiting decision) | Document D §8 |
| Reports | results/notes/phase3d2_fdr_lag_weight_results.md, phase4_coupled_observability.md, phase4_coupled_pilot_results.md, phase4_research_interpretation.md | |

Idempotency: ops/*.done markers; evaluate.py skips evaluated exp_ids; nothing earlier than Phase 3D-2 was modified.
