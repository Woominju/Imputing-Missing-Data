"""S1R data (joint 6-D integration, identical to gen6.make)."""
import os, sys
import numpy as np
from multiprocessing import Pool
from src.lorenz import generate_coupled_lorenz
from experiments.phase6.gen6 import s0_of
from experiments.phase6.s1r.common import all_ids, ROOT


def make(a):
    ki, j, n = a; p = f"{ROOT}/data/k{float(ki)}_id{j}.npz"
    if not os.path.exists(p):
        X, _, _ = generate_coupled_lorenz(float(ki), n_steps=n, dt=0.05, s0=s0_of(ki, j)); np.savez_compressed(p, X=X)
    return p


if __name__ == "__main__":
    os.makedirs(f"{ROOT}/data", exist_ok=True)
    with Pool(int(sys.argv[1])) as p:
        n = sum(1 for _ in p.imap_unordered(make, [(ki, j, n) for ki in (0, 1) for j, n in all_ids()]))
    print("S1R GEN DONE", n, flush=True)
