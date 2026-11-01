"""Experiment D (additional validation): sensitivity of the observability-vs-error association.

For B (own-variable past window) and C (all-variable past window), p in {0.3, 0.7}, seeds 0-2,
per evaluated test entry (t, i) of the main runs:
  rank, n_obs_rows, sigma_min of R and amp = ||c^T R^+|| (see audit_observability.py),
  abs error of the saved best checkpoint (re-imputed on CPU).
Spearman(amp, |err|) is then reported
  - per model / p / target / seed (pooling across targets is avoided)
  - restricted to full-rank windows (rank == 3) to separate the rank effect
  - by observed-row count bins (missing pattern)
  - with block-bootstrap CIs for block lengths 25, 100, 400 steps (temporal dependence)
An association is NOT evidence of diffeomorphism or identifiability.
Output: results/extra/expD_observability/{expD_records.csv.gz, expD_spearman.csv}
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from experiments.run_lorenz import get_lorenz  # noqa: E402
from experiments.extra.exp_b_contiguous import load_model  # noqa: E402
from src.evaluate import impute  # noqa: E402
from src.missingness import prepare  # noqa: E402
from src.observability import step_jacobians, backward_flow_jacobians  # noqa: E402

RES = os.path.join(ROOT, "results")
# IMD_EXTRA_RUN selects a sub-folder so separate runs (smoke vs overnight) never mix
OUT = os.path.join(RES, "extra", "expD_observability", os.environ.get("IMD_EXTRA_RUN", ""))


def refuse_overwrite():
    if os.path.exists(os.path.join(OUT, "expD_records.csv.gz")):
        raise SystemExit(f"output already exists in {OUT}; refusing to overwrite")
NAMES = ["x", "y", "z"]
rng = np.random.default_rng(0)


def spearman_block_ci(a, b, t, block, nboot=300):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 30:
        return np.nan, np.nan, np.nan
    rho = pd.Series(a).corr(pd.Series(b), method="spearman")
    blk = np.asarray(t) // block
    groups = [np.where(blk == g)[0] for g in np.unique(blk)]
    bs = []
    for _ in range(nboot):
        s = np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
        bs.append(pd.Series(a[s]).corr(pd.Series(b[s]), method="spearman"))
    return rho, np.nanquantile(bs, 0.025), np.nanquantile(bs, 0.975)


def records(seeds=(0, 1, 2), rates=(0.3, 0.7)):
    X, _ = get_lorenz()
    J = step_jacobians(X, 0.01)
    recs = []
    for p in rates:
        for seed in seeds:
            data = prepare(X, p, seed)
            a0, b0 = data["split"]["test"]
            sd = data["sd"]
            for name in "BC":
                mdl, c = load_model(name, p, seed)
                m, tau = c["m"], c["tau"]
                Bk = backward_flow_jacobians(J, (m - 1) * tau)
                xh = impute(mdl, torch.tensor(data["X"]), torch.tensor(data["M"]), m, tau).numpy()
                for t in range(max(a0, (m - 1) * tau), b0):
                    for i in range(3):
                        if data["eval_mask"][t, i] == 0:
                            continue
                        vars_ = [i] if name == "B" else [0, 1, 2]
                        R = np.array([Bk[k * tau, t, j] / sd[j] for k in range(m) for j in vars_
                                      if data["M"][t - k * tau, j] > 0]).reshape(-1, 3)
                        cvec = np.eye(3)[i] / sd[i]
                        if len(R) == 0:
                            rank, smin, amp = 0, 0.0, np.inf
                        else:
                            sv = np.linalg.svd(R, compute_uv=False)
                            rank = int((sv > 1e-10 * sv[0]).sum())
                            smin = sv[2] if len(sv) == 3 else 0.0
                            Rp = np.linalg.pinv(R, rcond=1e-10)
                            res = np.linalg.norm(cvec - R.T @ (Rp.T @ cvec)) / np.linalg.norm(cvec)
                            amp = np.linalg.norm(Rp.T @ cvec) if res < 1e-6 else np.inf
                        recs.append((name, p, seed, t, NAMES[i], len(R), rank, smin, amp,
                                     abs(xh[t, i] - data["X"][t, i])))
            print(f"expD records p={p} seed={seed}", flush=True)
    return pd.DataFrame(recs, columns=["model", "p", "seed", "t", "target", "n_obs_rows", "rank", "sigma_min",
                                       "amp", "abs_err"])


def main(**kw):
    torch.set_num_threads(8)
    refuse_overwrite()
    os.makedirs(OUT, exist_ok=True)
    r = records(**kw)
    r.to_csv(os.path.join(OUT, "expD_records.csv.gz"), index=False)
    rows = []
    for (mdl, p, seed, tgt), g in r.groupby(["model", "p", "seed", "target"]):
        fin = g[np.isfinite(g.amp)]
        full = fin[fin["rank"] == 3]
        base = {"model": mdl, "p": p, "seed": seed, "target": tgt, "n": len(g),
                "frac_rank_def": float((g["rank"] < 3).mean()), "frac_amp_inf": float((~np.isfinite(g.amp)).mean())}
        for block in (25, 100, 400):
            rho, lo, hi = spearman_block_ci(fin.amp, fin.abs_err, fin.t, block)
            rows.append({**base, "subset": "finite amp", "block": block, "n_used": len(fin), "rho": rho, "ci_lo": lo, "ci_hi": hi})
        rho, lo, hi = spearman_block_ci(full.amp, full.abs_err, full.t, 100)
        rows.append({**base, "subset": "rank==3 only", "block": 100, "n_used": len(full), "rho": rho, "ci_lo": lo, "ci_hi": hi})
        q = fin.n_obs_rows.quantile([1 / 3, 2 / 3]).values
        for lab, sel in [("n_obs_rows low third", fin.n_obs_rows <= q[0]),
                         ("n_obs_rows high third", fin.n_obs_rows > q[1])]:
            s = fin[sel]
            rho, lo, hi = spearman_block_ci(s.amp, s.abs_err, s.t, 100)
            rows.append({**base, "subset": lab, "block": 100, "n_used": len(s), "rho": rho, "ci_lo": lo, "ci_hi": hi})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "expD_spearman.csv"), index=False)
    piv = out[out.block.eq(100) | out.subset.ne("finite amp")]
    print(piv.groupby(["model", "p", "subset"]).agg(rho_mean=("rho", "mean"), rho_min=("rho", "min"),
                                                     rho_max=("rho", "max"),
                                                     ci_excl_0=("ci_lo", lambda x: int((x > 0).sum()))).round(3).to_string())


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--smoke":
        main(seeds=(0,), rates=(0.7,))
    else:
        main()
