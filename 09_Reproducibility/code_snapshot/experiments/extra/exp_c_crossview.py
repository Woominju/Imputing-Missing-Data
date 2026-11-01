"""Experiment C (additional validation): does Model C use the other variables' observations?

Protocol (evaluation-only input ablation on trained checkpoints; CPU):
  evaluation positions = the main test eval_mask entries of target variable i (identical across
  conditions and models). Input mask M' <= M is changed per condition:
    full         : M (as in the main evaluation)
    own_only     : other variables removed (M'[:, j != i] = 0)
    others_only  : target variable removed everywhere (M'[:, i] = 0)
    others_half  : 50% of the other variables' observed entries removed (fixed rng)
  Models: C (multi-view), B (single-view reference; only own variable by construction), A.
Caveat: removing inputs changes the input distribution relative to training (the models were
trained with MCAR rate p + hide_rate*(1-p)); a performance drop is therefore NOT the causal or
unique contribution of a view.
Output: results/extra/expC_crossview/expC_results.csv
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

RES = os.path.join(ROOT, "results")
# IMD_EXTRA_RUN selects a sub-folder so separate runs (smoke vs overnight) never mix
OUT = os.path.join(RES, "extra", "expC_crossview", os.environ.get("IMD_EXTRA_RUN", ""))


def refuse_overwrite():
    if os.path.exists(os.path.join(OUT, "expC_results.csv")):
        raise SystemExit(f"output already exists in {OUT}; refusing to overwrite")
NAMES = ["x", "y", "z"]


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
            for i in range(3):
                others = [j for j in range(3) if j != i]
                rng = np.random.default_rng(seed * 10 + i)
                conds = {}
                conds["full"] = data["M"].copy()
                Mo = data["M"].copy(); Mo[:, others] = 0; conds["own_only"] = Mo
                Mx = data["M"].copy(); Mx[:, i] = 0; conds["others_only"] = Mx
                Mh = data["M"].copy(); drop = rng.random((Mh.shape[0], 2)) < 0.5
                Mh[:, others] = Mh[:, others] * (~drop); conds["others_half"] = Mh
                E = data["eval_mask"][a:b, i] > 0
                for cname, Mc in conds.items():
                    assert np.all(Mc <= data["M"]) and np.all(Mc[a:b, i][E] == 0)
                    for k, (mdl, c) in models.items():
                        xh = impute(mdl, torch.tensor(data["X"]), torch.tensor(Mc), c["m"], c["tau"]).numpy()
                        err = (xh[a:b, i] - data["X"][a:b, i])[E]
                        rows.append({"p": p, "seed": seed, "target": NAMES[i], "condition": cname, "model": k,
                                     "n_eval": int(E.sum()), "rmse": float(np.sqrt(np.mean(err ** 2))),
                                     "mae": float(np.mean(np.abs(err))),
                                     "input_obs_rate_target": float(Mc[a:b, i].mean()),
                                     "input_obs_rate_others": float(Mc[a:b][:, others].mean())})
            print(f"expC p={p} seed={seed} done", flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(OUT, "expC_results.csv"), index=False)
    df = pd.DataFrame(rows)
    s = df.groupby(["p", "target", "condition", "model"]).agg(
        rmse=("rmse", "mean"), rmse_sd=("rmse", "std"), mae=("mae", "mean"), n_seeds=("seed", "nunique")).reset_index()
    s.to_csv(os.path.join(OUT, "expC_summary.csv"), index=False)
    print(s.pivot_table(index=["p", "target", "model"], columns="condition", values="rmse").round(4).to_string())


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--smoke":
        main(rates=(0.3,), seeds=(0,))
    else:
        main()
