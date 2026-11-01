"""Experiment 4: Lorenz observability (symbolic + numerical) and its relation to
imputation error of a trained model (needs results/imputations/lorenz_C_p0.3_s0.npy)."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.integrate import solve_ivp  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.lorenz import lorenz_rhs, SIGMA  # noqa: E402
from src.observability import (symbolic_checks, step_jacobians, backward_flow_jacobians,  # noqa: E402
                               svd_stats)
from src.missingness import prepare  # noqa: E402
from src.train import RES  # noqa: E402
from experiments.run_lorenz import get_lorenz  # noqa: E402

DT = 0.01


def q(a):
    a = a[np.isfinite(a)]
    return {"median": float(np.median(a)), "p05": float(np.quantile(a, 0.05)), "p95": float(np.quantile(a, 0.95)),
            "min": float(a.min()), "max": float(a.max())}


def main():
    rows = []
    sym = symbolic_checks()
    print(json.dumps(sym, indent=1))
    for k, v in sym.items():
        rows.append({"analysis": "symbolic", "quantity": k, "setting": "", "value": str(v)})

    X, _ = get_lorenz()
    T = X.shape[0]
    J = step_jacobians(X, DT)
    # check variational Jacobian against finite differences of solve_ivp at a few states
    errs = []
    for t in [100, 5000, 12345]:
        Jfd = np.zeros((3, 3))
        for j in range(3):
            e = np.zeros(3); e[j] = 1e-6
            a = solve_ivp(lorenz_rhs, (0, DT), X[t] + e, method="DOP853", rtol=1e-12, atol=1e-12).y[:, -1]
            b = solve_ivp(lorenz_rhs, (0, DT), X[t] - e, method="DOP853", rtol=1e-12, atol=1e-12).y[:, -1]
            Jfd[:, j] = (a - b) / 2e-6
        errs.append(np.abs(Jfd - J[t]).max() / np.abs(Jfd).max())
    rows.append({"analysis": "check", "quantity": "variational_vs_finite_diff_rel_err_max", "setting": "3 states",
                 "value": max(errs)})
    taus = [1, 2, 5, 10, 20]
    kmax = 2 * max(taus) + 7
    Bk = backward_flow_jacobians(J, kmax)
    valid = slice(kmax, T)
    x = X[valid, 0]

    # analytic O_x = [x, xdot, xddot]
    xs, ys, zs = X[valid].T
    DO = np.zeros((len(xs), 3, 3))
    DO[:, 0] = [1, 0, 0]
    DO[:, 1] = np.stack([-SIGMA * np.ones_like(xs), SIGMA * np.ones_like(xs), 0 * xs], -1)
    DO[:, 2] = np.stack([SIGMA * (28 - zs) + SIGMA ** 2, (-SIGMA - SIGMA ** 2) * np.ones_like(xs), -SIGMA * xs], -1)
    det_err = np.abs(np.linalg.det(DO) + SIGMA ** 2 * xs).max()
    rows.append({"analysis": "O_x=[x,xdot,xddot]", "quantity": "max|det+sigma^2 x| (numeric)", "setting": "", "value": det_err})
    _, smin, cond = svd_stats(DO)
    rows += [{"analysis": "O_x=[x,xdot,xddot]", "quantity": f"sigma_min_{k}", "setting": "", "value": v} for k, v in q(smin).items()]
    rows += [{"analysis": "O_x=[x,xdot,xddot]", "quantity": f"cond_{k}", "setting": "", "value": v} for k, v in q(cond).items()]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for tau in taus:
        F = np.stack([Bk[0, valid, 0], Bk[tau, valid, 0], Bk[2 * tau, valid, 0]], 1)  # rows e_x^T DPhi_{-k tau}
        det = np.linalg.det(F)
        pred = SIGMA ** 2 * (tau * DT) ** 3 * x
        sv, smin, cond = svd_stats(F)
        rel = np.median(np.abs(det - pred) / np.abs(pred))
        ranks = (sv > 1e-10 * sv[:, :1]).sum(1)
        set_ = f"tau={tau}dt"
        rows += [{"analysis": "F_x,tau=[x(t),x(t-tau),x(t-2tau)]", "quantity": "median_rel_err_det_vs_sigma2_tau3_x",
                  "setting": set_, "value": rel},
                 {"analysis": "F_x,tau=[x(t),x(t-tau),x(t-2tau)]", "quantity": "corr(det, x)", "setting": set_,
                  "value": float(np.corrcoef(det, x)[0, 1])},
                 {"analysis": "F_x,tau=[x(t),x(t-tau),x(t-2tau)]", "quantity": "frac_numerical_rank<3(1e-10)",
                  "setting": set_, "value": float(np.mean(ranks < 3))}]
        rows += [{"analysis": "F_x,tau=[x(t),x(t-tau),x(t-2tau)]", "quantity": f"sigma_min_{k}", "setting": set_, "value": v}
                 for k, v in q(smin).items()]
        rows += [{"analysis": "F_x,tau=[x(t),x(t-tau),x(t-2tau)]", "quantity": f"cond_{k}", "setting": set_, "value": v}
                 for k, v in q(cond).items()]
        # multi-view O_tau = [x(t), y(t - tau), z(t - 2 tau)]
        G = np.stack([Bk[0, valid, 0], Bk[tau, valid, 1], Bk[2 * tau, valid, 2]], 1)
        _, sming, condg = svd_stats(G)
        rows += [{"analysis": "O_tau=[x(t),y(t-tau),z(t-2tau)]", "quantity": f"sigma_min_{k}", "setting": set_, "value": v}
                 for k, v in q(sming).items()]
        rows += [{"analysis": "O_tau=[x(t),y(t-tau),z(t-2tau)]", "quantity": f"cond_{k}", "setting": set_, "value": v}
                 for k, v in q(condg).items()]
        # z-only delay map: local rank vs global non-injectivity (Z2 symmetry)
        Fz = np.stack([Bk[0, valid, 2], Bk[tau, valid, 2], Bk[2 * tau, valid, 2]], 1)
        _, sminz, condz = svd_stats(Fz)
        rows += [{"analysis": "F_z,tau=[z(t),z(t-tau),z(t-2tau)]", "quantity": f"cond_{k}", "setting": set_, "value": v}
                 for k, v in q(condz).items()]
        if tau in (1, 10):
            ax[0].scatter(x[::20], np.log10(cond[::20]), s=2, label=f"F_x tau={tau}dt")
    # global non-injectivity of z-delays: s and (-x,-y,z) give identical z(t) for all t
    s0 = X[0]
    a = solve_ivp(lorenz_rhs, (0, 5), s0, t_eval=np.arange(0, 5, DT), rtol=1e-11, atol=1e-11).y
    b = solve_ivp(lorenz_rhs, (0, 5), s0 * [-1, -1, 1], t_eval=np.arange(0, 5, DT), rtol=1e-11, atol=1e-11).y
    rows.append({"analysis": "F_z,tau=[z(t),z(t-tau),z(t-2tau)]", "quantity": "max|z(t;s)-z(t;Gamma s)| over 5 time units",
                 "setting": "Gamma(x,y,z)=(-x,-y,z)", "value": float(np.abs(a[2] - b[2]).max())})
    rows.append({"analysis": "F_z,tau=[z(t),z(t-tau),z(t-2tau)]", "quantity": "|s - Gamma s|",
                 "setting": "Gamma(x,y,z)=(-x,-y,z)", "value": float(np.linalg.norm(s0 - s0 * [-1, -1, 1]))})
    ax[0].set_xlabel("x(t)"); ax[0].set_ylabel("log10 cond(DF_x,tau)"); ax[0].legend(markerscale=4)
    ax[0].set_title("Delay map conditioning (near-singular at x=0)")

    # ---- relation to imputation error of trained model (m=8, tau=1 used in main runs)
    m, tau = 8, 1
    data = prepare(X, 0.3, 0)
    a0, b0 = data["split"]["test"]
    # The window below (all variables, past lags t-(m-1)tau..t) is the information set of
    # Model C only (B: own variable; D: up to t+window-1; linear: both sides), so only C
    # is analysed here. Per-variable / target-specific analysis: experiments/audit_observability.py
    for mdl in ["C"]:
        f = os.path.join(RES, "imputations", f"lorenz_{mdl}_p0.3_s0.npy")
        if not os.path.exists(f):
            print("missing", f)
            continue
        xhat = np.load(f)
        recs = []
        for t in range(max(a0, kmax), b0):
            for i in range(3):
                if data["eval_mask"][t, i] == 0:
                    continue
                Rrows = []
                for k in range(m):
                    for j in range(3):
                        if data["M"][t - k * tau, j] > 0:
                            Rrows.append(Bk[k * tau, t, j] / data["sd"][j])
                R = np.array(Rrows) if Rrows else np.zeros((1, 3))
                sv = np.linalg.svd(R, compute_uv=False)
                smin = sv[2] if len(sv) >= 3 else 0.0
                lag0_obs = int(sum(data["M"][t, j] for j in range(3)))
                recs.append((smin, len(Rrows), lag0_obs, abs(xhat[t, i] - data["X"][t, i])))
        r = np.array(recs)
        bins = np.quantile(r[:, 0], [0, .2, .4, .6, .8, 1.0])
        for lo, hi in zip(bins[:-1], bins[1:]):
            sel = (r[:, 0] >= lo) & (r[:, 0] <= hi)
            rows.append({"analysis": f"pattern_vs_error_model{mdl}_p0.3_s0", "quantity": "mean_abs_err",
                         "setting": f"sigma_min_obs_window in [{lo:.2e},{hi:.2e}] (n={sel.sum()})",
                         "value": float(r[sel, 3].mean())})
        for c in range(0, 3):
            sel = r[:, 2] == c
            if sel.any():
                rows.append({"analysis": f"pattern_vs_error_model{mdl}_p0.3_s0", "quantity": "mean_abs_err",
                             "setting": f"other vars observed at time t = {c} (n={sel.sum()})",
                             "value": float(r[sel, 3].mean())})
        rows.append({"analysis": f"pattern_vs_error_model{mdl}_p0.3_s0", "quantity": "spearman(sigma_min, abs_err)",
                     "setting": "", "value": float(pd.Series(r[:, 0]).corr(pd.Series(r[:, 3]), method="spearman"))})
        if mdl == "C":
            ax[1].scatter(np.log10(r[:, 0] + 1e-12), np.log10(r[:, 3] + 1e-8), s=2, alpha=.3)
            ax[1].set_xlabel("log10 sigma_min of observed delay-window Jacobian")
            ax[1].set_ylabel("log10 |error| (Model C, p=0.3)")
            ax[1].set_title("Missing pattern information vs error")
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "figures", "observability.png"), dpi=130)
    pd.DataFrame(rows).to_csv(os.path.join(RES, "observability_results.csv"), index=False)
    print(pd.DataFrame(rows).to_string(max_colwidth=70))


if __name__ == "__main__":
    main()
