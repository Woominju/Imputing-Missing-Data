"""Phase 4: coupled-Lorenz data at dt = 0.05 (CPU). Generator = src.lorenz.generate_coupled_lorenz (unchanged): two identical Lorenz
systems (sigma 10, rho 28, beta 8/3), bidirectional diffusive coupling kappa*(x_other - x_self) in the x-equations only, DOP853
rtol 1e-10 atol 1e-12, 5000-sample transient (250 time units at dt 0.05), 20000 samples kept.
traj 0: the project's default s0 = (1,1,1,-5,3,20); traj j >= 1: independent s0 drawn from np.random.default_rng(1000 + j)
(x, y ~ U(-15, 15), z ~ U(5, 40) for each subsystem). Existing results/data/coupled_lorenz_k*.npz (dt 0.01) are not touched."""
import json, os, sys
import numpy as np
from multiprocessing import Pool
from src.lorenz import generate_coupled_lorenz

O = "results/extra/phase4_coupled_pilot/data"


def s0_of(j):
    if j == 0:
        return (1.0, 1.0, 1.0, -5.0, 3.0, 20.0)
    r = np.random.default_rng(1000 + j)
    return tuple(float(v) for v in np.r_[r.uniform(-15, 15, 2), r.uniform(5, 40), r.uniform(-15, 15, 2), r.uniform(5, 40)])


def make(args):
    k, j = args; path = f"{O}/coupled_k{k}_dt0.05_traj{j}.npz"
    if os.path.exists(path):
        return path
    X, names, params = generate_coupled_lorenz(k, n_steps=20000, dt=0.05, s0=s0_of(j))
    np.savez_compressed(path, X=X, names=np.array(names), params=json.dumps(params)); return path


if __name__ == "__main__":
    os.makedirs(O, exist_ok=True)
    ks = [float(v) for v in sys.argv[1].split(",")]; js = [int(v) for v in sys.argv[2].split(",")]
    with Pool(min(16, len(ks) * len(js))) as pool:
        for p in pool.map(make, [(k, j) for k in ks for j in js]):
            print(p, flush=True)
