"""Experiment 6: real datasets with artificial MCAR missingness on observed entries.
Exact duplicate columns are dropped (keep first) so that a copy of a hidden value
cannot be read from its twin column."""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from experiments.common import run_grid  # noqa: E402
from src.data_loader import load_real  # noqa: E402
from src.train import add_args, RES  # noqa: E402


def drop_duplicate_cols(X, names):
    keep = []
    for j in range(X.shape[1]):
        if not any(np.array_equal(X[:, j], X[:, k], equal_nan=True) for k in keep):
            keep.append(j)
    return X[:, keep], [names[j] for j in keep], X.shape[1] - len(keep)


if __name__ == "__main__":
    ap = add_args(argparse.ArgumentParser())
    ap.add_argument("--datasets", default="enso,eeg,wind,exchange,transportation")
    ap.add_argument("--out", default="real_data_results.csv")
    args = ap.parse_args()
    for ds in args.datasets.split(","):
        X, names, meta = load_real(ds)
        X, names, ndup = drop_duplicate_cols(X, names)
        print(f"{ds}: X={X.shape} dropped_duplicate_cols={ndup}", flush=True)
        run_grid(ds, X, args, os.path.join(RES, args.out), extra={"dropped_dup_cols": ndup})
