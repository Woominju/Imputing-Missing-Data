"""Phase 9 data (protocol section 2): NEW independent coupled-Lorenz trajectories, generator and seed rule of gen6/gen8
(s0_of(kappa, id), ids in [200000, 299999] -> rng seeds disjoint from every earlier phase). Skips existing files (resumable).
usage: python experiments/phase9/gen9.py NPROC"""
import os, sys
import numpy as np
from multiprocessing import Pool
from src.lorenz import generate_coupled_lorenz
from experiments.phase6.gen6 import s0_of
from experiments.phase9.core9 import DDIR, lengths


def make(a):
    ki, j, n = a; p = f"{DDIR}/k{float(ki)}_id{j}.npz"
    if not os.path.exists(p):
        X, _, _ = generate_coupled_lorenz(float(ki), n_steps=n, dt=0.05, s0=s0_of(ki, j)); np.savez_compressed(p + ".tmp.npz", X=X); os.replace(p + ".tmp.npz", p)
    return p


if __name__ == "__main__":
    os.makedirs(DDIR, exist_ok=True)
    with Pool(int(sys.argv[1])) as pool:
        n = sum(1 for _ in pool.imap_unordered(make, [(ki, j, n) for ki in (0, 1) for j, n in lengths()], chunksize=2))
    print("P9 GEN trajectories", n, flush=True)
