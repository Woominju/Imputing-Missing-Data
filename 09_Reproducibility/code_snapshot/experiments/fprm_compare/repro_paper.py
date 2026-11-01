"""Reproduction check of the paper's 3D Lorenz FPRM experiment with the Python port (CPU only).

Paper setting (main text + Fig. 3g-k + SI): x complete, y or z incomplete, missing ratio r = 0.1..0.9,
200 repetitions, metrics rho and nRMSE (code: sqrt(MSE)/std(Ytest)). Targets at r=0.9: y rho=0.90 nRMSE=0.48,
z rho=0.81 nRMSE=0.68. Not specified for the 3D Lorenz figure: series length l, initial condition, whether E=3
(SI Table S1) or E=5 (official code example). We therefore run a grid over those unknowns.

Usage: python -m experiments.fprm_compare.repro_paper --out results/extra/fprm_comparison/v1 [--reps 50]
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from baselines.fprm.fprm import fprm_official_run
from baselines.fprm.ode45 import paper_lorenz3d

ICS = {"code_0.3step0.3": (0.3, 0.6, 0.9), "text_0.3step0.03": (0.3, 0.33, 0.36)}
RATES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def _job(args):
    ic, l, E, tau, tgt, r, rep = args
    _, X = paper_lorenz3d(x0=ICS[ic])
    X = X if l == "full" else X[:l]
    seed = 1_000_000 + rep                       # fixed, logged seeds (the official code uses rng('shuffle'))
    try:
        res = fprm_official_run(X[:, 0], X[:, {"y": 1, "z": 2}[tgt]], E, tau, r, np.random.default_rng(seed))
        res["error"] = ""
    except Exception as e:  # noqa: BLE001  -- recorded, never silently dropped
        res = {"error": f"{type(e).__name__}: {e}", "rho": np.nan, "nrmse_LL": np.nan, "mae": np.nan,
               "opt_success": False, "n_train": np.nan, "n_test": np.nan}
    res.update({"ic": ic, "l": len(X), "l_tag": str(l), "E": E, "tau": tau, "ref": "x", "target": tgt,
                "r": r, "rep": rep, "seed": seed})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    path = os.path.join(a.out, "repro_paper_lorenz3d_runs.csv")
    if os.path.exists(path):
        raise SystemExit(f"refusing to overwrite {path}")
    os.makedirs(a.out, exist_ok=True)
    jobs = []
    for E, tau in [(3, 3), (5, 3)]:
        for l in ["full", 500, 300, 100]:
            for tgt in ["y", "z"]:
                for r in RATES:
                    for rep in range(a.reps):
                        jobs.append(("code_0.3step0.3", l, E, tau, tgt, r, rep))
    for tgt in ["y", "z"]:                       # initial-condition sensitivity (paper text vs code), E=3
        for r in RATES:
            for rep in range(a.reps):
                jobs.append(("text_0.3step0.03", "full", 3, 3, tgt, r, rep))
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(_job, jobs, chunksize=8))
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    g = df.groupby(["ic", "l_tag", "l", "E", "tau", "target", "r"])
    s = g.agg(rho_mean=("rho", "mean"), rho_sd=("rho", "std"), nrmse_mean=("nrmse_LL", "mean"),
              nrmse_sd=("nrmse_LL", "std"), mae_mean=("mae", "mean"), n_train=("n_train", "mean"),
              n_test=("n_test", "mean"), reps=("rep", "count"), opt_fail=("opt_success", lambda v: int((~v.astype(bool)).sum())),
              errors=("error", lambda v: int((v != "").sum())))
    s.reset_index().to_csv(os.path.join(a.out, "repro_paper_lorenz3d_summary.csv"), index=False)
    meta = {"n_runs": len(df), "elapsed_s": time.time() - t0, "workers": a.workers,
            "threads_per_worker": os.environ.get("OPENBLAS_NUM_THREADS"),
            "nonfinite_metrics": int((~np.isfinite(df[["rho", "nrmse_LL", "mae"]].to_numpy())).sum()),
            "n_errors": int((df["error"] != "").sum()), "n_restarted": int((df.get("opt_restarts", 0) > 0).sum())}
    json.dump(meta, open(os.path.join(a.out, "repro_paper_meta.json"), "w"), indent=1)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
