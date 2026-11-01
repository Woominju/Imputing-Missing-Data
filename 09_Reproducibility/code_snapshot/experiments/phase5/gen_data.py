"""Phase 5: independent coupled-Lorenz trajectories (CPU). Generator = src.lorenz.generate_coupled_lorenz (unchanged):
sigma 10, rho 28, beta 8/3, diffusive x-coupling kappa, DOP853 rtol 1e-10 atol 1e-12, dt 0.05, transient 5000 samples.
Roles (ids never overlap, every trajectory has its own initial condition):
  train  id 0..7   : 20000 samples  (one per replicate r)
  val    id 100..107: 5000 samples  (one per replicate r; early stopping only)
  test   id 200..209: 5000 samples  (shared by all replicates and arms)
s0 ~ np.random.default_rng(SEED_BASE + 10000*ki + id): x, y ~ U(-15, 15), z ~ U(5, 40) per subsystem (as Phase 4, new streams).
Phase 4 data (results/extra/phase4_coupled_pilot/data) is not touched."""
import json, os, sys
import numpy as np
from multiprocessing import Pool
from src.lorenz import generate_coupled_lorenz

O = "results/extra/phase5/data"; SEED_BASE = 20261028
KAPPAS = (0.0, 1.0)
ROLES = {"train": (range(0, 8), 20000), "val": (range(100, 108), 5000), "test": (range(200, 210), 5000)}


def s0_of(ki, j):
    r = np.random.default_rng(SEED_BASE + 10000 * ki + j)
    return tuple(float(v) for v in np.r_[r.uniform(-15, 15, 2), r.uniform(5, 40), r.uniform(-15, 15, 2), r.uniform(5, 40)])


def path(k, j):
    return f"{O}/k{float(k)}_id{j}.npz"


def make(args):
    ki, j, n = args; k = KAPPAS[ki]; p = path(k, j)
    if os.path.exists(p):
        return p
    X, names, params = generate_coupled_lorenz(k, n_steps=n, dt=0.05, s0=s0_of(ki, j))
    np.savez_compressed(p, X=X, names=np.array(names), params=json.dumps(params)); return p


if __name__ == "__main__":
    os.makedirs(O, exist_ok=True)
    jobs = [(ki, j, n) for ki in range(len(KAPPAS)) for _, (ids, n) in ROLES.items() for j in ids]
    with Pool(32) as pool:
        for p in pool.imap_unordered(make, jobs):
            print(p, flush=True)
