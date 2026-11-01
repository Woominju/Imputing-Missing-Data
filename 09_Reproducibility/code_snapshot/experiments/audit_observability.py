"""Audit of the pattern-vs-error part of Experiment 4 (CPU only).

Problems found in run_observability.py's pattern analysis:
  1. one observation set (all variables, past lags t-(m-1)tau..t) was used for every
     model, but each model uses a different information set (B: own variable only;
     D: anchors up to t+window-1; linear: nearest observations on both sides)
  2. errors of x, y, z (different scales/dynamics) were pooled into one Spearman rho
  3. sigma_min of the observed-window Jacobian is target-agnostic and, at p=0.3 with
     m=8, tau=1, almost never small (no rank-deficient windows), so it barely varies

This script recomputes, per model with its own (past-window) information set and per
target variable, two quantities for each evaluated test entry (t, i):
  sigma_min  : smallest singular value of R (rows: observed entries in the window,
               d obs / d s(t), obs in train-normalised units, state in original units)
  amp        : ||c^T R^+||, c = d x_i(t)/d s(t) in normalised units, i.e. the
               linearised error amplification from observations to the target
               (finite iff c lies in the row space of R)
and correlates them with the absolute imputation error (Spearman, block-bootstrap 95% CI).
Imputations for p=0.7 are regenerated on CPU from the saved best checkpoints.
Only models whose information set is a past window are analysed (B, C); D and linear
are listed as not comparable with a past-window Jacobian.
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from experiments.run_lorenz import get_lorenz  # noqa: E402
from src.evaluate import impute  # noqa: E402
from src.missingness import prepare  # noqa: E402
from src.models import build_model  # noqa: E402
from src.observability import step_jacobians, backward_flow_jacobians  # noqa: E402
from src.train import RES  # noqa: E402

M_DIM, TAU, DT = 8, 1, 0.01
rng = np.random.default_rng(0)


BLOCK = 100  # samples per bootstrap block (1 time unit); errors are autocorrelated in time


def spearman_ci(a, b, t, nboot=500):
    """Spearman rho with a block bootstrap CI (blocks of BLOCK consecutive time steps),
    since neighbouring samples are strongly dependent."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rho = pd.Series(a).corr(pd.Series(b), method="spearman")
    blk = np.asarray(t) // BLOCK
    groups = [np.where(blk == g)[0] for g in np.unique(blk)]
    bs = []
    for _ in range(nboot):
        s = np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
        bs.append(pd.Series(a[s]).corr(pd.Series(b[s]), method="spearman"))
    return rho, np.quantile(bs, 0.025), np.quantile(bs, 0.975)


def main():
    torch.set_num_threads(8)
    X, _ = get_lorenz()
    J = step_jacobians(X, DT)
    Bk = backward_flow_jacobians(J, (M_DIM - 1) * TAU)
    rows = []
    for p in [0.3, 0.7]:
        for seed in [0, 1, 2]:
            data = prepare(X, p, seed)
            a0, b0 = data["split"]["test"]
            sd = data["sd"]
            for mdl_name in ["B", "C"]:
                ck = torch.load(os.path.join(RES, "checkpoints", f"lorenz_{mdl_name}_p{p}_s{seed}.pt"),
                                map_location="cpu", weights_only=False)
                c = ck["cfg"]
                assert c["m"] == M_DIM and c["tau"] == TAU
                mdl = build_model(mdl_name, 3, c["m"], c["latent"], c["hidden"], c["window"], c["direction"])
                mdl.load_state_dict(ck["state_dict"])
                xhat = impute(mdl, torch.tensor(data["X"]), torch.tensor(data["M"]), M_DIM, TAU).numpy()
                recs = []
                for t in range(max(a0, (M_DIM - 1) * TAU), b0):
                    for i in range(3):
                        if data["eval_mask"][t, i] == 0:
                            continue
                        vars_ = [i] if mdl_name == "B" else [0, 1, 2]
                        R = [Bk[k * TAU, t, j] / sd[j] for k in range(M_DIM) for j in vars_
                             if data["M"][t - k * TAU, j] > 0]
                        R = np.array(R).reshape(-1, 3)
                        cvec = np.eye(3)[i] / sd[i]
                        if len(R) == 0:
                            smin, amp, rank = 0.0, np.inf, 0
                        else:
                            sv = np.linalg.svd(R, compute_uv=False)
                            rank = int((sv > 1e-10 * sv[0]).sum())
                            smin = sv[2] if len(sv) == 3 else 0.0
                            Rp = np.linalg.pinv(R, rcond=1e-10)
                            resid = np.linalg.norm(cvec - R.T @ (Rp.T @ cvec)) / np.linalg.norm(cvec)
                            amp = np.linalg.norm(Rp.T @ cvec) if resid < 1e-6 else np.inf
                        recs.append((t, i, len(R), rank, smin, amp, abs(xhat[t, i] - data["X"][t, i])))
                r = pd.DataFrame(recs, columns=["t", "var", "n_obs_rows", "rank", "sigma_min", "amp", "abs_err"])
                for i, vn in [(None, "pooled"), (0, "x"), (1, "y"), (2, "z")]:
                    s = r if i is None else r[r["var"] == i]
                    fin = s[np.isfinite(s.amp)]
                    rho_s, lo_s, hi_s = spearman_ci(s.sigma_min, s.abs_err, s.t)
                    rho_a, lo_a, hi_a = spearman_ci(fin.amp, fin.abs_err, fin.t)
                    q = fin.amp.quantile([0.2, 0.8]).values
                    rows.append({
                        "model": mdl_name, "p": p, "seed": seed, "target": vn, "n": len(s),
                        "frac_rank<3": float((s["rank"] < 3).mean()),
                        "mean_err_rank<3": float(s[s["rank"] < 3].abs_err.mean()) if (s["rank"] < 3).any() else np.nan,
                        "mean_err_rank3": float(s[s["rank"] == 3].abs_err.mean()),
                        "sigma_min_p05": float(s.sigma_min.quantile(0.05)), "sigma_min_p95": float(s.sigma_min.quantile(0.95)),
                        "spearman_sigma_min": rho_s, "ci_lo_sigma_min": lo_s, "ci_hi_sigma_min": hi_s,
                        "spearman_amp": rho_a, "ci_lo_amp": lo_a, "ci_hi_amp": hi_a,
                        "mean_err_amp_low20": float(fin[fin.amp <= q[0]].abs_err.mean()),
                        "mean_err_amp_high20": float(fin[fin.amp >= q[1]].abs_err.mean()),
                        "test_rmse_recomputed": float(np.sqrt((s.abs_err ** 2).mean())),
                    })
                print(f"p={p} s={seed} {mdl_name} done", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "observability_audit.csv"), index=False)
    agg = out.groupby(["model", "p", "target"]).agg(
        n=("n", "mean"), frac_rank_def=("frac_rank<3", "mean"), err_rankdef=("mean_err_rank<3", "mean"),
        err_rank3=("mean_err_rank3", "mean"), rho_smin=("spearman_sigma_min", "mean"),
        rho_amp=("spearman_amp", "mean"), rho_amp_min=("spearman_amp", "min"), rho_amp_max=("spearman_amp", "max"),
        err_amp_low=("mean_err_amp_low20", "mean"), err_amp_high=("mean_err_amp_high20", "mean")).reset_index()
    agg.to_csv(os.path.join(RES, "observability_audit_summary.csv"), index=False)
    print(agg.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
