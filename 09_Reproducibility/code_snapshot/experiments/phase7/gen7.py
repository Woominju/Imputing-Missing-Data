"""P7-B data: NEW independent trajectories, joint 6-D integration exactly as experiments/phase6/gen6.py (same seed rule s0_of)."""
import os, sys
import numpy as np
from multiprocessing import Pool
from src.lorenz import generate_coupled_lorenz
from experiments.phase6.gen6 import s0_of
from experiments.phase7.core7 import all_ids, DDIR


def make(a):
    ki, j, n = a; p = f"{DDIR}/k{float(ki)}_id{j}.npz"
    if not os.path.exists(p):
        X, _, _ = generate_coupled_lorenz(float(ki), n_steps=n, dt=0.05, s0=s0_of(ki, j)); np.savez_compressed(p + ".tmp.npz", X=X); os.replace(p + ".tmp.npz", p)
    return p


if __name__ == "__main__":
    os.makedirs(DDIR, exist_ok=True)
    with Pool(int(sys.argv[1])) as pool:
        n = sum(1 for _ in pool.imap_unordered(make, [(ki, j, n) for ki in (0, 1) for j, n in all_ids()]))
    print("P7 GEN DONE", n, flush=True)
