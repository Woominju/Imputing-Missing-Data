"""Experiment 5: coupled Lorenz (6-D) over coupling strengths and missing rates."""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from experiments.common import run_grid  # noqa: E402
from src.lorenz import generate_coupled_lorenz  # noqa: E402
from src.train import add_args, RES  # noqa: E402

if __name__ == "__main__":
    ap = add_args(argparse.ArgumentParser())
    ap.add_argument("--kappas", default="0.0,0.5,2.0")
    ap.add_argument("--n_steps", type=int, default=20000)
    ap.add_argument("--out", default="coupled_lorenz_results.csv")
    args = ap.parse_args()
    for kappa in [float(k) for k in args.kappas.split(",")]:
        path = os.path.join(RES, "data", f"coupled_lorenz_k{kappa}_{args.n_steps}.npz")
        if os.path.exists(path):
            X = np.load(path)["X"]
        else:
            X, _, params = generate_coupled_lorenz(kappa, args.n_steps)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.savez_compressed(path, X=X, params=json.dumps(params))
        sync = float(np.sqrt(np.mean((X[:, :3] - X[:, 3:]) ** 2)))
        print(f"kappa={kappa} X={X.shape} rms(sys1-sys2)={sync:.3f}", flush=True)
        run_grid(f"coupled_k{kappa}", X, args, os.path.join(RES, args.out),
                 extra={"kappa": kappa, "sync_rms_diff": sync})
