"""Experiments 1-3 on the Lorenz system.

Example:
  python experiments/run_lorenz.py --models A,B,C --rates 0.3 --seeds 0 --epochs 5   # smoke
  python experiments/run_lorenz.py --models A,B,C,D
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from experiments.common import run_grid  # noqa: E402
from src.lorenz import generate_lorenz  # noqa: E402
from src.train import add_args, RES  # noqa: E402


def get_lorenz(n_steps=20000):
    path = os.path.join(RES, "data", f"lorenz_{n_steps}.npz")
    if os.path.exists(path):
        d = np.load(path, allow_pickle=True)
        return d["X"], json.loads(str(d["params"]))
    X, _, params = generate_lorenz(n_steps)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, X=X, params=json.dumps(params))
    return X, params


if __name__ == "__main__":
    ap = add_args(argparse.ArgumentParser())
    ap.add_argument("--n_steps", type=int, default=20000)
    ap.add_argument("--out", default="lorenz_results.csv")
    args = ap.parse_args()
    X, params = get_lorenz(args.n_steps)
    print("lorenz", X.shape, params)
    run_grid("lorenz", X, args, os.path.join(RES, args.out), keep_xhat=(0.3, 0))
