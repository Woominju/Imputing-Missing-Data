"""Coupled Lorenz (two 3-D subsystems, coupling kappa on x; 6 variables): FPRM (condition A) and Adapted FPRM
(condition B) vs Model C (original checkpoints) / linear / locf on identical test targets. E=3, tau=3, CPU.

Condition A: reference x1 fully observed (M with that column = M_orig), targets = the 5 other variables.
    Official FPRM-T from x1 only. Model C evaluated on the same condition-A input (OOD for the all-MCAR-trained
    checkpoint -> diagnostic). At kappa=0 subsystem 2 is independent of x1, so FPRM from x1 cannot recover it
    (no shared dynamics): a mathematical control for "one reference is not enough".
Condition B: original all-MCAR masks. AF-interp-T for every (target, reference) pair; per-target reference chosen by
    val RMSE (AF-interp-T-valsel). Never reported as FPRM.
Reduced grid (cost): p in {0.3, 0.7}, seed 0 by default.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch

from baselines.fprm.fprm import fprm_impute
from src.evaluate import baseline_impute, impute
from src.missingness import prepare
from src.models import build_model

VARS = ["x1", "y1", "z1", "x2", "y2", "z2"]
E, TAU = 3, 3


def c_pred(k, p, seed, X, Min):
    ck = torch.load(f"results/checkpoints/coupled_k{k}_C_p{p}_s{seed}.pt", map_location="cpu", weights_only=False)
    c = ck["cfg"]
    mdl = build_model("C", 6, c["m"], c["latent"], c["hidden"], c["window"], c["direction"])
    mdl.load_state_dict(ck["state_dict"])
    return impute(mdl, torch.tensor(X), torch.tensor(Min), c["m"], c["tau"]).numpy()


def interp_obs(x, m):
    t = np.arange(len(x))
    return np.interp(t, t[m > 0], x[m > 0])


def score(xh, x, ev):
    e = xh[ev] - x[ev]
    return {"n_eval": int(ev.sum()), "sse": float((e ** 2).sum()), "sae": float(np.abs(e).sum()),
            "rmse": float(np.sqrt(np.mean(e ** 2))), "nrmse": float(np.sqrt(np.mean(e ** 2)) / x[ev].std(ddof=1)),
            "finite": bool(np.isfinite(xh[ev]).all())}


def run(k, p, seed, arch):
    Xraw = np.load(f"results/data/coupled_lorenz_k{k}_20000.npz")["X"]
    d = prepare(Xraw, p, seed)
    X, M, Mo, EVm = d["X"], d["M"], d["M_orig"], d["eval_mask"] > 0
    T_, n = X.shape
    N = T_ - (E - 1) * TAU
    Xin = X * M
    rows = []
    evs = {}
    for part in ("val", "test"):
        a, b = d["split"][part]
        ev = np.zeros_like(EVm)
        ev[a:b] = EVm[a:b]
        ev[N:] = False
        evs[part] = ev
    pred_rows = (evs["val"] | evs["test"])[:N]
    # sanity: recomputed C reproduces the archived test RMSE
    cB = c_pred(k, p, seed, X, M)
    a, b = d["split"]["test"]
    rm = np.sqrt((((cB[a:b] - X[a:b]) * EVm[a:b]) ** 2).sum() / EVm[a:b].sum())
    assert abs(rm - arch.loc[f"coupled_k{k}_C_p{p}_s{seed}", "test_rmse"]) < 1e-4
    lin, locf = baseline_impute("linear", Xin, M), baseline_impute("locf", Xin, M)
    base = dict(k=k, p=p, seed=seed, E=E, tau=TAU)
    # ---- condition A: reference x1 complete
    MA = M.copy()
    MA[:, 0] = Mo[:, 0]
    cA = c_pred(k, p, seed, X, MA)
    for j in range(1, n):
        xh, info = fprm_impute(X[:, 0], Xin[:, j], M[:, j], E, TAU, predict_rows=pred_rows[:, j],
                               rng=np.random.default_rng(seed))
        for name, pr in [("FPRM-T(x1)", xh), ("C-orig-condA(OOD)", cA[:, j]), ("linear", lin[:, j]),
                         ("locf", locf[:, j])]:
            rows.append({**base, "cond": "A_ref_x1", "method": name, "target": VARS[j],
                         **score(pr, X[:, j], evs["test"][:, j]), "jitter": info["jitter"] if "FPRM" in name else 0})
    # ---- condition B: all MCAR, Adapted FPRM per (target, ref)
    for j in range(n):
        best, best_val, best_pred = None, np.inf, None
        for r in [i for i in range(n) if i != j]:
            xr = interp_obs(Xin[:, r], M[:, r])
            xh, info = fprm_impute(xr, Xin[:, j], M[:, j], E, TAU, predict_rows=pred_rows[:, j],
                                   rng=np.random.default_rng(seed))
            sv = score(xh, X[:, j], evs["val"][:, j])["rmse"]
            rows.append({**base, "cond": "B", "method": f"AF-interp-T({VARS[r]})", "target": VARS[j],
                         **score(xh, X[:, j], evs["test"][:, j]), "val_rmse": sv, "jitter": info["jitter"]})
            if sv < best_val:
                best, best_val, best_pred = VARS[r], sv, xh
        for name, pr in [(f"AF-interp-T-valsel[{best}]", best_pred), ("C", cB[:, j]), ("linear", lin[:, j]),
                         ("locf", locf[:, j])]:
            rows.append({**base, "cond": "B", "method": name, "target": VARS[j], **score(pr, X[:, j], evs["test"][:, j])})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ks", default="0.0,0.5,2.0")
    ap.add_argument("--ps", default="0.3,0.7")
    ap.add_argument("--seeds", default="0")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "coupled_ab_runs.csv")
    if os.path.exists(path):
        raise SystemExit(f"refusing to overwrite {path}")
    arch = pd.read_csv("results/coupled_lorenz_results.csv").set_index("exp_id")
    rows, t0 = [], time.time()
    for k in a.ks.split(","):
        for p in map(float, a.ps.split(",")):
            for seed in map(int, a.seeds.split(",")):
                try:
                    rows += run(k, p, seed, arch)
                except Exception as e:  # noqa: BLE001
                    rows.append({"k": k, "p": p, "seed": seed, "cond": "error", "method": "ERROR", "error": repr(e)})
                pd.DataFrame(rows).to_csv(path + ".partial", index=False)
                print(f"done k={k} p={p} seed={seed} t={time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    os.remove(path + ".partial")


if __name__ == "__main__":
    main()
