"""POST-HOC DIAGNOSTIC data generation (see k0_common.py)."""
import json, os, sys
import numpy as np
from multiprocessing import Pool
from src.lorenz import generate_coupled_lorenz, integrate, lorenz_rhs
from experiments.phase6.gen6 import s0_of
from experiments.phase6.diag.k0_common import all_ids, ROOT


def make(args):
    j, n = args; s0 = s0_of(0, j)
    pj, ps = f"{ROOT}/data_J/k0.0_id{j}.npz", f"{ROOT}/data_S/k0.0_id{j}.npz"
    if not os.path.exists(pj):
        X, _, _ = generate_coupled_lorenz(0.0, n_steps=n, dt=0.05, s0=s0); np.savez_compressed(pj, X=X)
    if not os.path.exists(ps):
        X1 = integrate(lorenz_rhs, list(s0[:3]), n, 0.05, 5000); X2 = integrate(lorenz_rhs, list(s0[3:]), n, 0.05, 5000)
        np.savez_compressed(ps, X=np.c_[X1, X2])
    return j


if __name__ == "__main__":
    with Pool(int(sys.argv[1])) as p:
        n = sum(1 for _ in p.imap_unordered(make, all_ids()))
    print("DIAG GEN DONE", n, flush=True)
