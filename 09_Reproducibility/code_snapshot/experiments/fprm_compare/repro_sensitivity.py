"""Hypothesis check for the gap between the Python port and the paper's 3D Lorenz numbers (r=0.9: y rho=0.90,
z rho=0.81). The official recipe (ode45 adaptive output, mean dt~0.019) gives rho>0.95 for every length tried.
Here: UNIFORM sampling interval dt in {0.01..0.3} (DOP853, rtol 1e-10), same initial state (0.3,0.6,0.9), 100 time
units of transient discarded, l=500, x complete, r=0.9, official FPRM pipeline, 100 reps. Optional additive white
noise (std = delta * std(series)). This only tests whether coarser sampling / noise COULD explain the gap; the
paper's actual sampling is not published.
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from baselines.fprm.fprm import fprm_official_run
from baselines.fprm.ode45 import lorenz_rhs


def series(dt, l, burn=100.0):
    t = burn + np.arange(l) * dt
    sol = solve_ivp(lorenz_rhs(), (0, t[-1]), [0.3, 0.6, 0.9], t_eval=t, method="DOP853", rtol=1e-10, atol=1e-12)
    return sol.y.T


def _job(a):
    dt, delta, E, tau, tgt, rep = a
    X = series(dt, 500)
    rng = np.random.default_rng(2_000_000 + rep)
    if delta > 0:
        X = X + delta * X.std(0) * rng.normal(size=X.shape)
    try:
        r = fprm_official_run(X[:, 0], X[:, tgt], E, tau, 0.9, rng)
        r["error"] = ""
    except Exception as e:  # noqa: BLE001
        r = {"error": str(e), "rho": np.nan, "nrmse_LL": np.nan}
    r.update({"dt": dt, "delta": delta, "E": E, "tau": tau, "target": "xyz"[tgt], "rep": rep})
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=100)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    jobs = [(dt, 0.0, 3, 3, tg, k) for dt in [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3] for tg in (1, 2)
            for k in range(a.reps)]
    jobs += [(0.02, d, 3, 3, tg, k) for d in [0.05, 0.1, 0.2] for tg in (1, 2) for k in range(a.reps)]
    with ProcessPoolExecutor(12) as ex:
        df = pd.DataFrame(list(ex.map(_job, jobs, chunksize=8)))
    df.to_csv(os.path.join(a.out, "sensitivity_runs.csv"), index=False)
    s = df.groupby(["dt", "delta", "E", "tau", "target"]).agg(rho=("rho", "mean"), nrmse=("nrmse_LL", "mean"),
                                                              reps=("rep", "count"),
                                                              errors=("error", lambda v: int((v != "").sum())))
    s.reset_index().to_csv(os.path.join(a.out, "sensitivity_summary.csv"), index=False)
    print(s.round(3).to_string())


if __name__ == "__main__":
    main()
