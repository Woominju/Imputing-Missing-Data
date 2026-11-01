# Imputing Missing Data in Complex Dynamical Systems using Delay Embedding

Preliminary implementation. Run commands from this directory.

```sh
export PYTHONPATH=.pydeps CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<reserved gpu>
PY=~/.conda/envs/reef_gpu/bin/python
$PY experiments/inspect_data.py                      # results/dataset_summary.csv
$PY tests/test_smoke.py                              # shapes, leakage, alignment, gradients, learning
$PY experiments/run_lorenz.py --models A,B,C,D --tau 1 --epochs 150 --patience 15
$PY experiments/run_coupled_lorenz.py --models C,D --kappas 0.0,0.5,2.0 --tau 1
$PY experiments/run_real_data.py --models A,B,C --tau 1
$PY experiments/run_observability.py                 # needs Lorenz imputations for p=0.3, seed 0
$PY experiments/make_figures.py                      # merges results/parts/*, summaries, figures
```
`configs/launch_all.sh` runs the main grids in parallel on GPUs 0/1. All CLI options: `--help`
(dataset/model/rates/m/tau/latent/batch/lr/epochs/seeds/device, ...).

## Protocol
- M_orig: originally observed entries. M = M_orig * Bernoulli(1-p) (MCAR, all variables).
  Evaluation targets: M_orig=1, M=0 (ground truth known, never fed to models).
- Chronological 70/15/15 split; normalisation from observed train entries only.
- Training: segments from the train split; H ~ Bernoulli(hide_rate) on M; input X*B, B=M-H;
  delay vectors are built after masking; loss = L_mask (only objective, no extra losses).
- Model selection / early stopping: RMSE on validation-split targets. Reported: test-split targets.
- Same masks for every model at a given (p, seed). Masks saved in results/imputations/*_masks.npz.
- Delay vectors are past-only: v_i(t) = [x_i(t), x_i(t-tau), ..., x_i(t-(m-1)tau)].
  Baseline `linear` interpolation uses past and future observations; `locf` past only.

## Information used to impute x_i(t) (measured in tests/test_integrity.py)
| method | observed values used |
|---|---|
| mean | train mean only |
| locf | last observation of variable i at or before t |
| linear | nearest observations of variable i before and after t (uses future) |
| A | all variables at time t |
| B | variable i at t, t-tau, ..., t-(m-1)tau (past only) |
| C | all variables at t, t-tau, ..., t-(m-1)tau (past only) |
| D | all variables from t-(m-1)tau to t+window-1 (uses up to window-1 future steps) |

Training inputs are missing at rate about p + hide_rate*(1-p) (extra H hiding); at evaluation
they are missing at rate p.

## Resuming an interrupted grid
Completed rows are written after each run. To resume without mixing CSV formats, write to a new
part and skip ids already recorded, e.g.
`run_real_data.py --datasets wind --out parts/real_wind_r2.csv --skip_from parts/real_wind.csv` (same other args).
`make_figures.py` merges all parts with the same prefix and de-duplicates by exp_id.

## Verification
- `experiments/verify_results.py`: exp_id/field consistency, epoch logs vs epochs_run/best_val_rmse.
- `experiments/recompute_from_checkpoints.py`: regenerates masks and recomputes every row's
  metrics from the saved best checkpoint (CPU) -> results/verification/.
- `experiments/audit_observability.py`: per-model information sets, per-variable,
  target-specific amplification ||c^T R^+|| -> results/observability_audit*.csv.

## Models
A: mask-aware MLP on current time step. B: per-variable delay AE. C: per-variable encoders +
observation-count-weighted mean fusion -> shared z -> per-variable decoders.
D: C's encoder gives z(t0); RK4 rollout of dz/dt = f_theta(z) over a short window
(default backward from the window end, window 8); imputations average the window predictions.

## Not done
FPRM / Lag-FPRM / MVCM: no implementation found on this server; not reproduced, no numbers reported.
