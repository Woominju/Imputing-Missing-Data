"""Model C under condition A (one fully observed reference variable, MCAR(p) on the others).

--mode retrain : train a NEW Model C with the identical architecture, loss (src.train.L_mask), optimiser and
                 hyper-parameters as the original Lorenz checkpoint (cfg read from results/checkpoints), changing
                 only the data masks: M_A = M with the reference column fully observed. Training-time hiding H is
                 unchanged (it also hides reference entries, exactly as for any observed entry).
                 Model selection uses val RMSE on the non-reference hidden entries only (eval_mask[:, r] = 0).
--mode diag    : evaluate the ORIGINAL checkpoint (trained under all-variable MCAR) with condition-A inputs.
                 Out-of-distribution for that model -> diagnostic only, never the comparison result.

Outputs (never inside results/checkpoints or results/logs): set IMD_RESULTS_DIR to the output folder BEFORE launch;
src.train writes checkpoints/ and logs/ under it. Predictions: <out>/preds/<exp_id>.npy; rows: <out>/<mode>_runs.csv
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

from src.evaluate import impute, metrics
from src.missingness import prepare
from src.models import build_model

VARS = "xyz"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def append_row(path, row):
    """Append keeping every key (the project-wide append_row keeps a fixed column set and would drop ours)."""
    df = pd.DataFrame([row])
    if os.path.exists(path):
        cols = list(pd.read_csv(path, nrows=0).columns)
        df = df.reindex(columns=cols + [c for c in df.columns if c not in cols])
        assert list(df.columns[:len(cols)]) == cols and len(df.columns) == len(cols), "column set changed"
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def cond_a_data(Xraw, ref, p, seed):
    r = VARS.index(ref)
    d = prepare(Xraw, p, seed)
    d["M"] = d["M"].copy()
    d["M"][:, r] = d["M_orig"][:, r]
    d["eval_mask"] = d["eval_mask"].copy()
    d["eval_mask"][:, r] = 0.0
    assert (d["eval_mask"] * d["M"]).sum() == 0
    return d


CK = {"pattern": os.path.join(ROOT, "results", "checkpoints", "lorenz_C_p{p}_s{seed}.pt"), "prefix": "lorenz"}


def orig_ckpt(p, seed):
    return torch.load(CK["pattern"].format(p=p, seed=seed), map_location="cpu", weights_only=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["retrain", "diag"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--refs", default="x,y,z")
    ap.add_argument("--ps", default="0.1,0.3,0.5,0.7")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--max_epochs", type=int, default=0, help="smoke tests only (0 = original cfg)")
    ap.add_argument("--data", default=os.path.join(ROOT, "results", "data", "lorenz_20000.npz"))
    ap.add_argument("--ckpt_pattern", default=CK["pattern"], help="source of cfg (retrain) / weights (diag)")
    ap.add_argument("--prefix", default="lorenz")
    a = ap.parse_args()
    CK.update({"pattern": a.ckpt_pattern, "prefix": a.prefix})
    out = os.path.abspath(a.out)
    if a.mode == "retrain":
        assert os.environ.get("IMD_RESULTS_DIR", "").rstrip("/") == out.rstrip("/"), "set IMD_RESULTS_DIR=--out"
        assert torch.cuda.is_available(), "retrain mode is GPU only"
    os.makedirs(os.path.join(out, "preds"), exist_ok=True)
    csv = os.path.join(out, f"{a.mode}_runs.csv")
    done = set(pd.read_csv(csv).exp_id) if os.path.exists(csv) else set()
    Xraw = np.load(a.data)["X"]
    from src.train import run                          # imported after IMD_RESULTS_DIR is set
    for ref in a.refs.split(","):
        for p in map(float, a.ps.split(",")):
            for seed in map(int, a.seeds.split(",")):
                exp_id = f"{CK['prefix']}_condA_ref{ref}_C_{a.mode}_p{p}_s{seed}"
                if exp_id in done:
                    print("skip (done)", exp_id, flush=True)
                    continue
                d = cond_a_data(Xraw, ref, p, seed)
                ck = orig_ckpt(p, seed)
                cfg = dict(ck["cfg"])
                t0 = time.time()
                if a.mode == "retrain":
                    cfg["device"] = "cuda"
                    if a.max_epochs:
                        cfg["epochs"] = a.max_epochs
                    row, xhat, _ = run(cfg, Xraw, d, exp_id, save_ckpt=True, verbose=False)
                else:
                    mdl = build_model("C", 3, cfg["m"], cfg["latent"], cfg["hidden"], cfg["window"], cfg["direction"])
                    mdl.load_state_dict(ck["state_dict"])
                    xhat = impute(mdl, torch.tensor(d["X"]), torch.tensor(d["M"]), cfg["m"], cfg["tau"]).numpy()
                    te = metrics(xhat, d, "test")
                    row = {"exp_id": exp_id, "test_n_eval": te["n_eval"], "test_rmse": te["rmse"],
                           "test_mae": te["mae"], "best_val_rmse": metrics(xhat, d, "val")["rmse"]}
                np.save(os.path.join(out, "preds", exp_id + ".npy"), xhat.astype(np.float32))
                row.update({"exp_id": exp_id, "ref": ref, "p": p, "seed": seed, "mode": a.mode,
                            "wall_s": round(time.time() - t0, 1), "finite": bool(np.isfinite(xhat).all()),
                            "epochs_cfg": cfg["epochs"]})
                append_row(csv, row)
                print(f"{exp_id} test_rmse={row['test_rmse']:.4f} n={row['test_n_eval']} {row['wall_s']}s",
                      flush=True)


if __name__ == "__main__":
    sys.exit(main())
