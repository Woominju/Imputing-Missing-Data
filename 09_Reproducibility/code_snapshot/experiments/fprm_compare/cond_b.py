"""Condition B (AIM setting): MCAR(p) on ALL variables, no fully observed reference. Original masks of the main
protocol (prepare(p, seed)), unchanged.

Original FPRM is NOT applicable (its predictor delay vector must be complete). Only ADAPTED variants are run here,
and they are never reported as FPRM:
  AF-interp-T / -I : reference r's missing entries filled by linear interpolation of r's OWN observed entries
                     (input mask M only; offline: uses past and future observations of r), then FPRM
                     (T = transductive, as official; I = GP trained on train-split rows only).
  AF-complete-T    : no filling; only delay rows whose reference entries are all observed are used for training and
                     imputation. Coverage of the evaluation targets is reported; metrics are on that subset only.
  AF-interp-T-valsel : per target, reference chosen by val RMSE (val hidden entries; never test).
  AF-interp-T-avg  : OUR extension (not in the paper): mean of the two references' AF-interp-T predictions.
Comparators recomputed on CPU from the ORIGINAL checkpoints with the original masks: Model C, Model B; plus linear,
locf. Each comparator is evaluated on exactly the same target subset as the method it is compared with.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from baselines.fprm.fprm import fprm_impute
from src.evaluate import baseline_impute, impute
from src.missingness import prepare
from src.models import build_model

VARS = "xyz"


CKPT = {"dir": "results/checkpoints", "prefix": "lorenz"}


def nn_pred(model_name, p, seed, d):
    ck = torch.load(f"{CKPT['dir']}/{CKPT['prefix']}_{model_name}_p{p}_s{seed}.pt", map_location="cpu",
                    weights_only=False)
    c = ck["cfg"]
    if c.get("model") == "F":
        from experiments.aim_autonomous.model_f import ModelF
        mdl = ModelF(3, c["m"], c["latent"], c["hidden"])
        mdl.load_state_dict(ck["state_dict"])
        return impute(mdl, torch.tensor(d["X"]), torch.tensor(d["M"]), c["m"], c["tau"]).numpy()
    mdl = build_model("C" if model_name == "E" else model_name, 3, c["m"], c["latent"], c["hidden"], c["window"], c["direction"])
    mdl.load_state_dict(ck["state_dict"])
    return impute(mdl, torch.tensor(d["X"]), torch.tensor(d["M"]), c["m"], c["tau"]).numpy()


def interp_obs(x, m):
    t = np.arange(len(x))
    obs = m > 0
    return np.interp(t, t[obs], x[obs])


def complete_rows(m, N, E, tau):
    ok = np.ones(N, bool)
    for i in range(E):
        ok &= m[i * tau: i * tau + N] > 0
    return ok


def score(xhat, x, ev):
    e = xhat[ev] - x[ev]
    return {"n_eval": int(ev.sum()), "sse": float((e ** 2).sum()), "sae": float(np.abs(e).sum()),
            "rmse": float(np.sqrt(np.mean(e ** 2))) if ev.any() else np.nan,
            "mae": float(np.mean(np.abs(e))) if ev.any() else np.nan,
            "finite": bool(np.isfinite(xhat[ev]).all())}


def run_one(Xraw, p, seed, E, tau, ref_csv, extra_models=()):
    d = prepare(Xraw, p, seed)
    X, M, EV = d["X"], d["M"], d["eval_mask"].astype(bool)
    T = len(X)
    N = T - (E - 1) * tau
    Xin = X * M
    comp = {"C": nn_pred("C", p, seed, d), "B": nn_pred("B", p, seed, d),
            "linear": baseline_impute("linear", Xin, M), "locf": baseline_impute("locf", Xin, M)}
    for em in extra_models:
        comp[em] = nn_pred(em, p, seed, d)
    # sanity: recomputed C/B reproduce the archived test RMSE
    a, b = d["split"]["test"]
    for mname in ("C", "B") + tuple(extra_models):
        e = (comp[mname][a:b] - X[a:b]) * EV[a:b]
        rm = float(np.sqrt((e ** 2).sum() / EV[a:b].sum()))
        arch = float(ref_csv.loc[f"{CKPT['prefix']}_{mname}_p{p}_s{seed}", "test_rmse"])
        assert abs(rm - arch) < 1e-4, (mname, p, seed, rm, arch)
    tr_a, tr_b = d["split"]["train"]
    k = np.arange(N)
    rows, val_rmse, preds = [], {}, {}
    for j in range(3):
        ev = {}
        for part in ("val", "test"):
            pa, pb = d["split"][part]
            e_ = np.zeros(T, bool)
            e_[pa:pb] = EV[pa:pb, j]
            e_[N:] = False                                   # FPRM tail undefined -> excluded for all methods
            ev[part] = e_
        pred_rows = (ev["val"] | ev["test"])[:N]
        comp_j = {m_: v[:, j] for m_, v in comp.items()}      # comparators' predictions for target j
        for r in [i for i in range(3) if i != j]:
            xr = interp_obs(Xin[:, r], M[:, r])               # only r's observed entries enter
            for name, trn in (("AF-interp-T", None), ("AF-interp-I", (k >= tr_a) & (k + (E - 1) * tau < tr_b))):
                xh, info = fprm_impute(xr, Xin[:, j], M[:, j], E, tau, train_rows=trn, predict_rows=pred_rows,
                                       rng=np.random.default_rng(seed))
                preds[(name, j, r)] = xh
                for part in ("val", "test"):
                    s = score(xh, X[:, j], ev[part])
                    rows.append({"split": part, "method": name, "target": VARS[j], "ref": VARS[r], "subset": "defined",
                                 "coverage": 1.0, **s, "n_train": info["n_train"], "opt_restarts": info["opt_restarts"]})
                    if name == "AF-interp-T":
                        val_rmse[(j, r)] = s["rmse"] if part == "val" else val_rmse.get((j, r))
            cr = complete_rows(M[:, r], N, E, tau)
            xh, info = fprm_impute(X[:, r] * M[:, r], Xin[:, j], M[:, j], E, tau, train_rows=cr,
                                   predict_rows=pred_rows & cr, rng=np.random.default_rng(seed))
            assert not np.isfinite(xh[:N][~cr]).any()        # never imputes from an incomplete reference vector
            for part in ("val", "test"):
                sub = ev[part].copy()
                sub[:N] &= cr
                cov = sub.sum() / max(ev[part].sum(), 1)
                for mname, xm in [("AF-complete-T", xh)] + list(comp_j.items()):
                    rows.append({"split": part, "method": mname, "target": VARS[j], "ref": VARS[r],
                                 "subset": f"complete_ref_{VARS[r]}", "coverage": cov, **score(xm, X[:, j], sub),
                                 "n_train": info["n_train"] if mname.startswith("AF") else np.nan})
        # per-target selection / averaging (val only for selection)
        refs = [i for i in range(3) if i != j]
        rsel = min(refs, key=lambda r: val_rmse[(j, r)])
        avg = np.mean([preds[("AF-interp-T", j, r)] for r in refs], axis=0)
        for part in ("val", "test"):
            for mname, xm, rr in [("AF-interp-T-valsel", preds[("AF-interp-T", j, rsel)], VARS[rsel]),
                                  ("AF-interp-T-avg", avg, "both")] + [(m_, x_, "-") for m_, x_ in comp_j.items()]:
                rows.append({"split": part, "method": mname, "target": VARS[j], "ref": rr, "subset": "defined",
                             "coverage": 1.0, **score(xm, X[:, j], ev[part])})
    for r_ in rows:
        r_.update({"p": p, "seed": seed, "E": E, "tau": tau})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ps", default="0.1,0.3,0.5,0.7")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--etau", default="3,3")
    ap.add_argument("--tag", default="cond_b")
    ap.add_argument("--data", default="results/data/lorenz_20000.npz")
    ap.add_argument("--ckpt_dir", default="results/checkpoints")
    ap.add_argument("--prefix", default="lorenz")
    ap.add_argument("--arch_csv", default="results/lorenz_results.csv")
    ap.add_argument("--extra_models", default="")
    a = ap.parse_args()
    CKPT.update({"dir": a.ckpt_dir, "prefix": a.prefix})
    extra = tuple(m for m in a.extra_models.split(",") if m)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"{a.tag}_runs.csv")
    if os.path.exists(path):
        raise SystemExit(f"refusing to overwrite {path}")
    Xraw = np.load(a.data)["X"]
    ref_csv = pd.read_csv(a.arch_csv).drop_duplicates("exp_id", keep="last").set_index("exp_id")
    E, tau = map(int, a.etau.split(","))
    rows, t0 = [], time.time()
    for p in map(float, a.ps.split(",")):
        for seed in map(int, a.seeds.split(",")):
            rows += run_one(Xraw, p, seed, E, tau, ref_csv, extra)
            pd.DataFrame(rows).to_csv(path + ".partial", index=False)
            print(f"done p={p} seed={seed} t={time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    os.remove(path + ".partial")
    json.dump({"elapsed_s": time.time() - t0, "args": vars(a), "nonfinite": int((~df.finite).sum())},
              open(os.path.join(a.out, f"{a.tag}_meta.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
