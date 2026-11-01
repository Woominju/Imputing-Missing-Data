"""Real data, condition B (all variables MCAR, as in the main real-data protocol): ADAPTED FPRM vs A/B/C (original
checkpoints, recomputed on CPU and checked against the archived test RMSE) and linear/locf, identical targets.

Adapted FPRM here = AF-interp-T-valsel: the reference's missing entries are linearly interpolated from its own observed
entries (input mask only), then official transductive FPRM. Reference per target: all other variables if n <= 6,
else the 3 variables with the largest |corr| with the target on TRAIN-split rows where both are observed; among the
candidates the one with the lowest val RMSE is used. E, tau from the paper's SI Table S1 (EEG 3/3, SST(ENSO) 3/3,
exchange 4/4). This reference-selection rule is ours; it is not part of the paper's method.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch

from baselines.fprm.fprm import fprm_impute
from experiments.run_real_data import drop_duplicate_cols
from src.data_loader import load_real
from src.evaluate import baseline_impute, impute
from src.missingness import prepare
from src.models import build_model

ETAU = {"eeg": (3, 3), "enso": (3, 3), "exchange": (4, 4)}


def nn_pred(ds, model, p, seed, d):
    ck = torch.load(f"results/checkpoints/{ds}_{model}_p{p}_s{seed}.pt", map_location="cpu", weights_only=False)
    c = ck["cfg"]
    n = d["X"].shape[1]
    mdl = build_model(model, n, c["m"], c["latent"], c["hidden"], c["window"], c["direction"])
    mdl.load_state_dict(ck["state_dict"])
    return impute(mdl, torch.tensor(d["X"]), torch.tensor(d["M"]), c["m"], c["tau"]).numpy()


def run(ds, p, seed, arch):
    X0, names, _ = load_real(ds)
    X0, names, _ = drop_duplicate_cols(X0, names)
    d = prepare(X0, p, seed)
    X, M, EVm = d["X"], d["M"], d["eval_mask"] > 0
    T_, n = X.shape
    E, tau = ETAU[ds]
    N = T_ - (E - 1) * tau
    Xin = X * M
    comp = {m: nn_pred(ds, m, p, seed, d) for m in ("A", "B", "C")}
    a, b = d["split"]["test"]
    for m, xh in comp.items():
        rm = np.sqrt((((xh[a:b] - X[a:b]) * EVm[a:b]) ** 2).sum() / EVm[a:b].sum())
        assert abs(rm - arch.loc[f"{ds}_{m}_p{p}_s{seed}", "test_rmse"]) < 1e-3, (ds, m, p, seed, rm)
    comp["linear"] = baseline_impute("linear", Xin, M)
    comp["locf"] = baseline_impute("locf", Xin, M)
    evs = {}
    for part in ("val", "test"):
        pa, pb = d["split"][part]
        ev = np.zeros_like(EVm)
        ev[pa:pb] = EVm[pa:pb]
        ev[N:] = False
        evs[part] = ev
    tra, trb = d["split"]["train"]
    rows = []
    t_ = np.arange(T_)
    for j in range(n):
        if evs["test"][:, j].sum() == 0:
            continue
        if n <= 6:
            cands = [r for r in range(n) if r != j]
        else:
            both = (M[tra:trb, j] > 0)
            cs = []
            for r in range(n):
                if r == j:
                    continue
                bb = both & (M[tra:trb, r] > 0)
                cs.append((abs(np.corrcoef(X[tra:trb][bb, j], X[tra:trb][bb, r])[0, 1]) if bb.sum() > 10 else 0.0, r))
            cands = [r for _, r in sorted(cs, key=lambda v: -np.nan_to_num(v[0]))[:3]]
        pred_rows = (evs["val"][:N, j] | evs["test"][:N, j])
        best = (np.inf, None, None)
        for r in cands:
            obs = M[:, r] > 0
            if obs.sum() < 2:
                continue
            xr = np.interp(t_, t_[obs], Xin[obs, r])
            xh, info = fprm_impute(xr, Xin[:, j], M[:, j], E, tau, predict_rows=pred_rows, rng=np.random.default_rng(seed))
            ev = evs["val"][:, j]
            vr = np.sqrt(np.mean((xh[ev] - X[ev, j]) ** 2)) if ev.any() else np.inf
            if vr < best[0]:
                best = (vr, r, xh)
        ev = evs["test"][:, j]
        preds = {"AF-interp-T-valsel": best[2], **{m: v[:, j] for m, v in comp.items()}}
        for m, xh in preds.items():
            e = xh[ev] - X[ev, j]
            rows.append({"dataset": ds, "p": p, "seed": seed, "target": names[j], "method": m, "E": E, "tau": tau,
                         "ref": names[best[1]] if m.startswith("AF") else "", "n_eval": int(ev.sum()),
                         "sse": float((e ** 2).sum()), "sae": float(np.abs(e).sum()),
                         "finite": bool(np.isfinite(xh[ev]).all())})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--datasets", default="enso")
    ap.add_argument("--ps", default="0.1,0.3,0.5,0.7")
    ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tag = a.datasets.replace(",", "_")
    path = os.path.join(a.out, f"real_af_{tag}_runs.csv")
    if os.path.exists(path):
        raise SystemExit(f"refusing to overwrite {path}")
    arch = pd.read_csv("results/real_data_results.csv").drop_duplicates("exp_id", keep="last").set_index("exp_id")
    rows, t0 = [], time.time()
    for ds in a.datasets.split(","):
        for p in map(float, a.ps.split(",")):
            for seed in map(int, a.seeds.split(",")):
                try:
                    rows += run(ds, p, seed, arch)
                except Exception as e:  # noqa: BLE001
                    rows.append({"dataset": ds, "p": p, "seed": seed, "method": "ERROR", "error": repr(e)})
                pd.DataFrame(rows).to_csv(path + ".partial", index=False)
                print(f"done {ds} p={p} seed={seed} t={time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    os.remove(path + ".partial")


if __name__ == "__main__":
    main()
