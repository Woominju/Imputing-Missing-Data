"""Lorenz with a COARSER sampling interval (default dt=0.05 instead of 0.01), condition B (all variables MCAR).

Why: at dt=0.01 linear interpolation is already near-perfect on Lorenz (test RMSE 0.008-0.08), so the benchmark
cannot separate dynamics-informed reconstruction from statistical interpolation (AIM point 3). The paper-gap
sensitivity check also showed that the sampling interval dominates the difficulty of the FPRM task.
Same generator as the main data (src.lorenz.generate_lorenz: DOP853, 5000-sample transient, s0=(1,1,1)) except dt.
Same masks/splits/normalisation (src.missingness.prepare), same hyper-parameters as the original Lorenz checkpoints
for the same (p, seed). Models: B, C, E (= C + view-dropout training masks, see model_e_viewdrop.py) + linear/locf/
mean. Outputs only under --out (IMD_RESULTS_DIR must equal --out).
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch

import src.train as T
from experiments.aim_autonomous.model_e_viewdrop import forward_batch_viewdrop
from src.lorenz import generate_lorenz
from src.missingness import prepare

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ORIG_FB = T.forward_batch
from experiments.aim_autonomous.model_f import build as build_with_f  # noqa: E402
T.build_model = build_with_f                     # adds "F"; every other name -> src.models.build_model


def append_row(path, row):
    """Append one row keeping columns aligned by name (rewrites the file if a new column appears)."""
    new = pd.DataFrame([row])
    if not os.path.exists(path):
        new.to_csv(path, index=False)
        return
    old_cols = list(pd.read_csv(path, nrows=0).columns)
    if set(new.columns) <= set(old_cols):
        new.reindex(columns=old_cols).to_csv(path, mode="a", header=False, index=False)
    else:
        pd.concat([pd.read_csv(path), new], ignore_index=True).to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--ps", default="0.1,0.3,0.5,0.7")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--models", default="B,C,E")
    ap.add_argument("--max_epochs", type=int, default=0, help="smoke only")
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    assert os.environ.get("IMD_RESULTS_DIR", "").rstrip("/") == out.rstrip("/")
    os.makedirs(os.path.join(out, "preds"), exist_ok=True)
    os.makedirs(os.path.join(out, "data"), exist_ok=True)
    dpath = os.path.join(out, "data", f"lorenz_dt{a.dt}_20000.npz")
    if not os.path.exists(dpath):
        X, _, params = generate_lorenz(20000, dt=a.dt)
        np.savez_compressed(dpath, X=X, params=str(params))
    Xraw = np.load(dpath)["X"]
    csv = os.path.join(out, "runs.csv")
    done = set(pd.read_csv(csv).exp_id) if os.path.exists(csv) else set()
    for p in map(float, a.ps.split(",")):
        for seed in map(int, a.seeds.split(",")):
            d = prepare(Xraw, p, seed)
            todo = [m for m in a.models.split(",")] + ["linear", "locf", "mean"]
            for mname in todo:
                exp_id = f"lorenz_dt{a.dt}_{mname}_p{p}_s{seed}"
                if exp_id in done:
                    continue
                t0 = time.time()
                if mname in ("linear", "locf", "mean"):
                    cfg = {"dataset": f"lorenz_dt{a.dt}", "m": 8, "tau": 1, "latent": 16, "hidden": 64, "batch": 64,
                           "lr": 1e-3, "hide_rate": 0.2, "window": 8, "direction": -1}
                    row, xhat = T.run_baseline(mname, cfg, d, exp_id)
                else:
                    assert torch.cuda.is_available()
                    src_model = "C" if mname in ("E", "F") else mname
                    ck = torch.load(os.path.join(ROOT, "results/checkpoints", f"lorenz_{src_model}_p{p}_s{seed}.pt"),
                                    map_location="cpu", weights_only=False)
                    cfg = dict(ck["cfg"])
                    cfg.update({"device": "cuda", "dataset": f"lorenz_dt{a.dt}", "model": "F" if mname == "F" else src_model})
                    if mname == "F":
                        cfg["fusion"] = "learned_mlp"
                    if a.max_epochs:
                        cfg["epochs"] = a.max_epochs
                    T.forward_batch = forward_batch_viewdrop if mname == "E" else ORIG_FB
                    row, xhat, _ = T.run(cfg, Xraw, d, exp_id, save_ckpt=True, verbose=False)
                    T.forward_batch = ORIG_FB
                np.save(os.path.join(out, "preds", exp_id + ".npy"), xhat.astype(np.float32))
                row.update({"model": mname, "dt": a.dt, "wall_s": round(time.time() - t0, 1),
                            "finite": bool(np.isfinite(xhat).all())})
                append_row(csv, row)
                print(f"{exp_id} test_rmse={row['test_rmse']:.4f} {row['wall_s']}s", flush=True)


if __name__ == "__main__":
    main()
