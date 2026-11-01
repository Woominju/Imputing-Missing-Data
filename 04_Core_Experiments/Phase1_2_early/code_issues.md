# Code issues found, fixes, impact (2026-10-24)

| # | Where | Problem | Severity | Impact on results | Status |
|---|---|---|---|---|---|
| 1 | experiments/common.py append_row | header from the first (baseline) row; model rows carried extra columns, so header and fields were misaligned | high (parsing) | none: values written in fixed order; read_part repairs them; all 499 rows recomputed from checkpoints match (results/verification/) | fixed (fixed-column writer). Running main processes keep the old writer; read_part handles both formats |
| 2 | experiments/run_observability.py pattern analysis | past-window all-variable Jacobian applied to B, D and linear; x, y, z pooled; target-agnostic sigma_min | medium (interpretation) | original "rho ~ 0" not interpretable; v1 kept as observability_results_v1_before_audit.csv | fixed: restricted to C; per-target amp in audit_observability.py and Exp D |
| 3 | audit_observability.py | i.i.d. bootstrap on time-dependent samples | medium (statistics) | CIs about 19 % too narrow; point estimates unchanged | fixed: block bootstrap; v1 kept as *_v1_iid_bootstrap.csv |
| 4 | src/observability.py | multi-view det labelled first order but contained incomplete tau^2 terms | low | label only | fixed (series to O(tau)) |
| 5 | README | D's use of future steps and the train vs test input-missingness difference not stated | medium (fair comparison) | none on numbers | fixed (information table) |
| 6 | configs/overnight_watchdog.sh (pre-deployment) | process matching by cmdline substring could hit shells containing the string | high (operational) | found in an isolated test; deployed version requires a python executable, argv[1] = experiment script and cwd = project | fixed before deployment |

Operational incidents (no effect on results):
- A watchdog test was given a real training PID. The deadline branch did not run. Processes verified alive.
- `pkill -f` patterns matched the operator's own shell twice. Only test processes were affected.

Not bugs, but limitations to keep in mind:
- 19 runs at the epoch cap.
- Wind orientation (rows = time) is assumed.
- Exchange is non-stationary.
- Only 3 seeds.
- FPRM, Lag-FPRM and MVCM are not reproduced (no implementation available).

Re-runs required for correctness: none. Incomplete grids: see results/logs/overnight_status.txt.
