"""Experiment A analysis (CPU): delay-step sensitivity of Models B and C on Lorenz.

Question: is the good performance at tau=1 mainly due to strong autocorrelation of adjacent
samples, and how do B (single view) and C (multi view) respond as the delay window spans a
longer physical time?
Training runs: experiments/run_lorenz.py with IMD_RESULTS_DIR=results/extra/expA_tau,
tau in {5,10,20}; tau=1 rows are taken from the main Lorenz runs (same masks, split, m, epochs).
This script
  - checks the expA masks are identical to the main masks
  - recomputes every expA row from its checkpoint (integrity)
  - tabulates RMSE/MAE (mean, sd over seeds), epochs, epoch-cap hits, non-finite steps
  - reports what changes with tau: window span (m-1)*tau*dt, autocorrelation at lag tau and
    (m-1)tau, and the observability amplification ||c^T R^+|| of C's observation window
Output: results/extra/expA_tau/expA_summary.csv, expA_information.csv, expA_recompute.csv
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from experiments.common import read_part  # noqa: E402
from experiments.run_lorenz import get_lorenz  # noqa: E402
from src.evaluate import impute, metrics  # noqa: E402
from src.missingness import prepare  # noqa: E402
from src.models import build_model  # noqa: E402
from src.observability import step_jacobians, backward_flow_jacobians  # noqa: E402

RES = os.path.join(ROOT, "results")
EA = os.environ.get("IMD_EXPA_DIR", os.path.join(RES, "extra", "expA_tau"))
DT, M_DIM = 0.01, 8


def main():
    torch.set_num_threads(8)
    X, _ = get_lorenz()
    parts = sorted(glob.glob(os.path.join(EA, "lorenz_tau*.csv")))
    if not parts:
        print("no expA results yet"); return
    ea = pd.concat([read_part(p) for p in parts], ignore_index=True)
    main_ = pd.read_csv(os.path.join(RES, "lorenz_results.csv"))
    base = main_[main_.model.isin(["B", "C"]) & main_.missing_rate.isin([0.3, 0.7])]
    df = pd.concat([base, ea], ignore_index=True)
    # mask identity + recompute
    rec = []
    for _, r in ea.iterrows():
        p, seed = float(r.missing_rate), int(r.seed)
        data = prepare(X, p, seed)
        mk_a = np.load(os.path.join(EA, "imputations", f"lorenz_p{p}_s{seed}_masks.npz"))
        mk_m = np.load(os.path.join(RES, "imputations", f"lorenz_p{p}_s{seed}_masks.npz"))
        same = all(np.array_equal(mk_a[k], mk_m[k]) for k in ("M", "eval_mask", "mu", "sd"))
        ck = torch.load(os.path.join(EA, "checkpoints", f"{r.exp_id}.pt"), map_location="cpu", weights_only=False)
        c = ck["cfg"]
        mdl = build_model(c["model"], 3, c["m"], c["latent"], c["hidden"], c["window"], c["direction"])
        mdl.load_state_dict(ck["state_dict"])
        xh = impute(mdl, torch.tensor(data["X"]), torch.tensor(data["M"]), c["m"], c["tau"]).numpy()
        t = metrics(xh, data, "test")
        lg = pd.read_csv(os.path.join(EA, "logs", f"{r.exp_id}.csv"))
        rec.append({"exp_id": r.exp_id, "masks_identical_to_main": same, "tau_ckpt": c["tau"],
                    "rmse_rel_diff": abs(t["rmse"] - r.test_rmse) / r.test_rmse,
                    "mae_rel_diff": abs(t["mae"] - r.test_mae) / r.test_mae,
                    "log_len_matches": len(lg) == r.epochs_run,
                    "best_val_matches": bool(np.isclose(lg.val_rmse.min(), r.best_val_rmse, rtol=1e-6))})
    rc = pd.DataFrame(rec)
    rc.to_csv(os.path.join(EA, "expA_recompute.csv"), index=False)
    print("expA integrity: rows", len(rc), "| masks identical", int(rc.masks_identical_to_main.sum()),
          "| max rel diff", float(rc[["rmse_rel_diff", "mae_rel_diff"]].max().max()),
          "| log/best-val mismatches", int((~rc.log_len_matches).sum() + (~rc.best_val_matches).sum()))
    s = df.groupby(["tau", "model", "missing_rate"]).agg(
        rmse=("test_rmse", "mean"), rmse_sd=("test_rmse", "std"), mae=("test_mae", "mean"), mae_sd=("test_mae", "std"),
        epochs=("epochs_run", "mean"), n_epoch_cap=("epochs_run", lambda x: int((x >= 150).sum())),
        nonfinite=("nonfinite_steps", "sum"), train_time_s=("train_time_s", "mean"), n_seeds=("seed", "nunique")).reset_index()
    s["window_span_time"] = (M_DIM - 1) * s["tau"] * DT
    s.to_csv(os.path.join(EA, "expA_summary.csv"), index=False)
    print(s.round(4).to_string(index=False))
    # information that changes with tau
    a, b = prepare(X, 0.3, 0)["split"]["train"]
    J = step_jacobians(X, DT)
    info = []
    for tau in sorted(df.tau.unique()):
        tau = int(tau)
        ac = {f"autocorr_lag_tau_{v}": float(pd.Series(X[a:b, j]).autocorr(tau)) for j, v in enumerate("xyz")}
        ac.update({f"autocorr_lag_span_{v}": float(pd.Series(X[a:b, j]).autocorr((M_DIM - 1) * tau))
                   for j, v in enumerate("xyz")})
        Bk = backward_flow_jacobians(J, (M_DIM - 1) * tau)
        for p in (0.3, 0.7):
            data = prepare(X, p, 0)
            a0, b0 = data["split"]["test"]
            amps, ranks = [], []
            for t in range(max(a0, (M_DIM - 1) * tau), b0, 3):
                for i in range(3):
                    if data["eval_mask"][t, i] == 0:
                        continue
                    R = np.array([Bk[k * tau, t, j] / data["sd"][j] for k in range(M_DIM) for j in range(3)
                                  if data["M"][t - k * tau, j] > 0]).reshape(-1, 3)
                    sv = np.linalg.svd(R, compute_uv=False) if len(R) else np.zeros(1)
                    ranks.append(int((sv > 1e-10 * max(sv[0], 1e-300)).sum()))
                    if ranks[-1] == 3:
                        amps.append(np.linalg.norm(np.linalg.pinv(R).T @ (np.eye(3)[i] / data["sd"][i])))
            info.append({"tau": tau, "p": p, "window_span_time": (M_DIM - 1) * tau * DT, **ac,
                         "C_window_frac_rank<3": float(np.mean(np.array(ranks) < 3)),
                         "C_window_amp_median": float(np.median(amps)) if amps else np.nan,
                         "C_window_amp_p90": float(np.quantile(amps, 0.9)) if amps else np.nan})
    pd.DataFrame(info).to_csv(os.path.join(EA, "expA_information.csv"), index=False)
    print(pd.DataFrame(info).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
