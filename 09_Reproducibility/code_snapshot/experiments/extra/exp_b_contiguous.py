"""Experiment B (additional validation, NOT a replacement for the proposal's MCAR evaluation).

Question: when the temporally nearby observations of the target variable are missing (a
contiguous gap), can the models still recover it from the OTHER variables' observations?

Protocol (evaluation-only generalisation test, no retraining; CPU):
  - models: A, B, C trained under MCAR p (main Lorenz runs, best checkpoints), linear, locf, mean
  - data/masks: identical to the main runs (prepare(X, p, seed)); the other variables keep
    their MCAR missingness, so observation stays incomplete for every variable
  - for target variable i and gap length L, non-overlapping gaps are placed in the test split
    (fixed rng) and M[gap, i] = 0; evaluation positions = all gap positions (truth known)
  - every model is evaluated on exactly the same positions
  - errors are also split by position inside the gap: k < m (some own lags may still be
    observed before the gap for past-window models) vs k >= m (no own-variable observation in
    the m-step past window of B/C)
Model D is excluded (it also uses future steps; see README).
Output: results/extra/expB_contiguous/expB_results.csv
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from experiments.run_lorenz import get_lorenz  # noqa: E402
from src.evaluate import impute, baseline_impute  # noqa: E402
from src.missingness import prepare  # noqa: E402
from src.models import build_model  # noqa: E402

RES = os.path.join(ROOT, "results")
# IMD_EXTRA_RUN selects a sub-folder so separate runs (smoke vs overnight) never mix
OUT = os.path.join(RES, "extra", "expB_contiguous", os.environ.get("IMD_EXTRA_RUN", ""))


def refuse_overwrite():
    if os.path.exists(os.path.join(OUT, "expB_results.csv")):
        raise SystemExit(f"output already exists in {OUT}; refusing to overwrite")
GAPS = [5, 10, 25, 50, 100]
NAMES = ["x", "y", "z"]


def load_model(name, p, seed):
    ck = torch.load(os.path.join(RES, "checkpoints", f"lorenz_{name}_p{p}_s{seed}.pt"), map_location="cpu",
                    weights_only=False)
    c = ck["cfg"]
    mdl = build_model(name, 3, c["m"], c["latent"], c["hidden"], c["window"], c["direction"])
    mdl.load_state_dict(ck["state_dict"])
    return mdl, c


def gap_starts(a, b, L, rng):
    starts, t = [], a + 20 + int(rng.integers(0, 50))
    while t + L < b:
        starts.append(t)
        t += L + 100 + int(rng.integers(0, 50))
    return starts


def main(rates=(0.3, 0.7), seeds=(0, 1, 2)):
    torch.set_num_threads(8)
    refuse_overwrite()
    os.makedirs(OUT, exist_ok=True)
    X, _ = get_lorenz()
    rows = []
    for p in rates:
        for seed in seeds:
            data = prepare(X, p, seed)
            a, b = data["split"]["test"]
            models = {k: load_model(k, p, seed) for k in "ABC"}
            m = models["C"][1]["m"]
            for i in range(3):
                for L in GAPS:
                    rng = np.random.default_rng(1000 * seed + 10 * i + L)
                    Mg = data["M"].copy()
                    E = np.zeros_like(Mg)
                    pos = np.full(Mg.shape[0], -1)
                    for s in gap_starts(a, b, L, rng):
                        Mg[s:s + L, i] = 0
                        E[s:s + L, i] = 1
                        pos[s:s + L] = np.arange(L)
                    preds = {k: baseline_impute(k, data["X"] * Mg, Mg) for k in ("mean", "locf", "linear")}
                    for k, (mdl, c) in models.items():
                        preds[k] = impute(mdl, torch.tensor(data["X"]), torch.tensor(Mg), c["m"], c["tau"]).numpy()
                    sel = E[:, i] > 0
                    for k, xh in preds.items():
                        err = (xh[:, i] - data["X"][:, i])[sel]
                        kk = pos[sel]
                        rec = {"p_train_mcar": p, "seed": seed, "target": NAMES[i], "gap_len": L, "model": k,
                               "n_gaps": int(sel.sum() // L), "n_eval": int(sel.sum()),
                               "rmse": float(np.sqrt(np.mean(err ** 2))), "mae": float(np.mean(np.abs(err))),
                               "rmse_pos_lt_m": float(np.sqrt(np.mean(err[kk < m] ** 2))),
                               "rmse_pos_ge_m": float(np.sqrt(np.mean(err[kk >= m] ** 2))) if (kk >= m).any() else np.nan,
                               "eval_setting": "eval-only generalisation (models trained on MCAR)"}
                        rows.append(rec)
            print(f"expB p={p} seed={seed} done", flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(OUT, "expB_results.csv"), index=False)
    df = pd.DataFrame(rows)
    s = df.groupby(["p_train_mcar", "target", "gap_len", "model"]).agg(
        rmse=("rmse", "mean"), rmse_sd=("rmse", "std"), mae=("mae", "mean"),
        rmse_pos_ge_m=("rmse_pos_ge_m", "mean"), n_seeds=("seed", "nunique")).reset_index()
    s.to_csv(os.path.join(OUT, "expB_summary.csv"), index=False)
    print(s.pivot_table(index=["p_train_mcar", "target", "gap_len"], columns="model", values="rmse").round(3).to_string())


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--smoke":
        main(rates=(0.3,), seeds=(0,))
    else:
        main()
