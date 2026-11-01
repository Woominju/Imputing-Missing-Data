"""Condition A on our Lorenz data: one fully observed reference variable r, MCAR(p) on the others.

Mask: M_A = prepare(p, seed).M with column r replaced by M_orig[:, r] (=1). For non-reference variables the
missing entries and evaluation targets are therefore exactly those of the main protocol (same p, seed).
Evaluation targets: eval_mask = 1 in the test split, non-reference columns only, restricted to times k that have
an FPRM delay row (k < T-(E-1)*tau; the last (E-1)*tau test points are excluded for every method and counted).

Methods (all evaluated on the identical target set):
  FPRM-T  official transductive FPRM: GP trained on every row of the series whose target is observed
          (train/val/test splits all used; only *observed* values), predicts the hidden test entries.
  FPRM-I  inductive variant: GP trained only on train-split rows whose delay vector lies inside the train split.
  linear / locf / mean  classical baselines from src.evaluate.baseline_impute (target column only).
Model C is deliberately NOT evaluated here (its checkpoints were trained under all-variable MCAR -> OOD).

Usage: python -m experiments.fprm_compare.cond_a --out DIR --refs x --ps 0.3 --seeds 0 --etau 3,3
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from baselines.fprm.fprm import fprm_impute
from src.evaluate import baseline_impute
from src.missingness import prepare

VARS = "xyz"


def run_one(Xraw, ref, p, seed, E, tau, out_dir):
    r = VARS.index(ref)
    d = prepare(Xraw, p, seed)
    X, M = d["X"], d["M"].copy()
    M[:, r] = d["M_orig"][:, r]
    assert (M[:, r] == 1).all()
    T = len(X)
    N = T - (E - 1) * tau
    tr_a, tr_b = d["split"]["train"]
    Xin = X * M                                         # model input: hidden entries are zero
    targets = [j for j in range(3) if j != r]
    rows = []
    preds = {}
    for j in targets:
        E_j = d["eval_mask"][:, j].astype(bool)
        assert not (E_j & (M[:, j] > 0)).any()          # eval targets are exactly the missing entries
        evs = {}
        for part in ("val", "test"):                    # val: for E/tau selection only; test: reported
            a, b = d["split"][part]
            ev = np.zeros(T, bool)
            ev[a:b] = E_j[a:b]
            ev_def = ev.copy()
            ev_def[N:] = False
            evs[part] = (ev_def, int(ev.sum() - ev_def.sum()))
        k = np.arange(N)
        pred_rows = (evs["val"][0] | evs["test"][0])[:N]
        variants = {
            "FPRM-T": None,
            "FPRM-I": (k >= tr_a) & (k + (E - 1) * tau < tr_b),
        }
        for name, train_rows in variants.items():
            t0 = time.time()
            xhat, info = fprm_impute(X[:, r], Xin[:, j], M[:, j], E, tau, train_rows=train_rows,
                                     predict_rows=pred_rows, rng=np.random.default_rng(seed))
            sec = time.time() - t0
            preds[(name, j)] = xhat
            for part, (ev_def, n_excl) in evs.items():
                rows.append(_metrics(name, xhat, X[:, j], ev_def, ref, VARS[j], p, seed, E, tau, n_excl,
                                     sec, info, part))
        for kind in ["linear", "locf", "mean"]:
            xb = baseline_impute(kind, Xin, M)[:, j]
            for part, (ev_def, n_excl) in evs.items():
                rows.append(_metrics(kind, xb, X[:, j], ev_def, ref, VARS[j], p, seed, E, tau, n_excl, 0.0, {},
                                     part))
    tag = f"ref{ref}_p{p}_s{seed}_E{E}_tau{tau}"
    np.savez_compressed(os.path.join(out_dir, "preds", tag + ".npz"),
                        M_condA=M, **{f"{n}_{VARS[j]}": v for (n, j), v in preds.items()})
    return rows


def _metrics(method, xhat, x, ev, ref, tgt, p, seed, E, tau, n_excl, sec, info, split):
    e = xhat[ev] - x[ev]
    out = {"split": split, "method": method, "ref": ref, "target": tgt, "p": p, "seed": seed, "E": E, "tau": tau,
           "n_eval": int(ev.sum()), "n_excluded_tail": n_excl,
           "rmse": float(np.sqrt(np.mean(e ** 2))), "mae": float(np.mean(np.abs(e))),
           "sse": float(np.sum(e ** 2)), "sae": float(np.sum(np.abs(e))),
           "nrmse_LL": float(np.sqrt(np.mean(e ** 2)) / np.std(x[ev], ddof=1)),
           "rho": float(np.corrcoef(xhat[ev], x[ev])[0, 1]) if np.std(xhat[ev]) > 0 else np.nan,
           "finite": bool(np.isfinite(xhat[ev]).all()), "seconds": round(sec, 2)}
    out.update({k: info[k] for k in ["fit_method", "predict_method", "sigmaL", "sigmaF", "sigma", "n_train",
                                     "opt_success", "opt_restarts", "jitter"] if k in info})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--refs", default="x")
    ap.add_argument("--ps", default="0.3")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--etau", default="3,3", help="semicolon-separated E,tau pairs")
    ap.add_argument("--tag", default="cond_a")
    ap.add_argument("--data", default="results/data/lorenz_20000.npz")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "preds"), exist_ok=True)
    path = os.path.join(a.out, f"{a.tag}_runs.csv")
    if os.path.exists(path):
        raise SystemExit(f"refusing to overwrite {path}")
    Xraw = np.load(a.data)["X"]
    rows = []
    t0 = time.time()
    for et in a.etau.split(";"):
        E, tau = map(int, et.split(","))
        for ref in a.refs.split(","):
            for p in map(float, a.ps.split(",")):
                for seed in map(int, a.seeds.split(",")):
                    try:
                        rows += run_one(Xraw, ref, p, seed, E, tau, a.out)
                    except Exception as e:  # noqa: BLE001  -- recorded as an error row, grid continues
                        rows.append({"split": "error", "method": "ERROR", "ref": ref, "p": p, "seed": seed, "E": E,
                                     "tau": tau, "error": f"{type(e).__name__}: {e}", "finite": False})
                        print(f"ERROR ref={ref} p={p} seed={seed}: {e}", flush=True)
                    pd.DataFrame(rows).to_csv(path + ".partial", index=False)
                    print(f"done ref={ref} p={p} seed={seed} E={E} tau={tau} t={time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    os.remove(path + ".partial")
    df = df[df.split != "error"]
    # pooled over non-reference targets (RMSE from summed squared errors, not a mean of RMSEs)
    g = df.groupby(["split", "method", "ref", "p", "seed", "E", "tau"])
    pooled = g.apply(lambda s: pd.Series({"n_eval": s.n_eval.sum(), "rmse": np.sqrt(s.sse.sum() / s.n_eval.sum()),
                                          "mae": s.sae.sum() / s.n_eval.sum()})).reset_index()
    pooled.to_csv(os.path.join(a.out, f"{a.tag}_pooled.csv"), index=False)
    json.dump({"elapsed_s": time.time() - t0, "args": vars(a), "nonfinite_rows": int((~df.finite.astype(bool)).sum()),
               "error_rows": int((pd.read_csv(path).split == "error").sum())},
              open(os.path.join(a.out, f"{a.tag}_meta.json"), "w"), indent=1)
    print(pooled.to_string())


if __name__ == "__main__":
    main()
