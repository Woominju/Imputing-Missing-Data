"""Phase 6 data: NEW independent coupled-Lorenz trajectories (CPU). Same generator and settings as Phase 5
(src.lorenz.generate_coupled_lorenz, DOP853 rtol 1e-10 atol 1e-12, dt 0.05, transient 5000 samples), new seed streams:
s0 ~ np.random.default_rng(SEED6 + 100000*ki + id), SEED6 = 30261029, x, y ~ U(-15, 15), z ~ U(5, 40) per subsystem, independently.
Phase 5 seeds are 20261028 + 10000*ki + id (id <= 209) -> no overlap (checked in tests). Every id has its own initial condition.
G2 (replicate r = 0..23, test i = 0..7 per replicate; every trajectory is used in exactly one role):
  train target 1000+r (20000), train sham donor 1100+r (20000), val target 1200+r (5000), val donor 1300+r (5000),
  test target 2000+100r+i (5000), test donor 10000+100r+i (5000), normalisation pool 900..903 (20000)
G1 (supervised information audit): train 7000+i (i<32, 20000), val 7100+i (i<8, 5000), test 7200+i (i<32, 5000),
  donors (independent V for the exact-null control): 8000+i, 8100+i, 8200+i with the same lengths."""
import json, os
import numpy as np
from multiprocessing import Pool
from src.lorenz import generate_coupled_lorenz

O = "results/extra/phase6/data"; SEED6 = 30261029; KAPPAS = (0.0, 1.0); R6, NT6 = 24, 8


def roles():
    L = [(900 + q, 20000) for q in range(4)]
    for r in range(R6):
        L += [(1000 + r, 20000), (1100 + r, 20000), (1200 + r, 5000), (1300 + r, 5000)]
        L += [(2000 + 100 * r + i, 5000) for i in range(NT6)] + [(10000 + 100 * r + i, 5000) for i in range(NT6)]
    for base, n, cnt in ((7000, 20000, 32), (7100, 5000, 8), (7200, 5000, 32), (8000, 20000, 32), (8100, 5000, 8), (8200, 5000, 32)):
        L += [(base + i, n) for i in range(cnt)]
    return L


def s0_of(ki, j):
    r = np.random.default_rng(SEED6 + 100000 * ki + j)
    return tuple(float(v) for v in np.r_[r.uniform(-15, 15, 2), r.uniform(5, 40), r.uniform(-15, 15, 2), r.uniform(5, 40)])


def path(k, j):
    return f"{O}/k{float(k)}_id{j}.npz"


def make(args):
    ki, j, n = args; k = KAPPAS[ki]; p = path(k, j)
    if os.path.exists(p):
        return p
    X, names, params = generate_coupled_lorenz(k, n_steps=n, dt=0.05, s0=s0_of(ki, j))
    np.savez_compressed(p + ".tmp.npz", X=X, names=np.array(names), params=json.dumps(params)); os.replace(p + ".tmp.npz", p); return p


if __name__ == "__main__":
    os.makedirs(O, exist_ok=True)
    jobs = [(ki, j, n) for ki in range(2) for j, n in roles()]
    jobs.sort(key=lambda q: -q[2])
    with Pool(56) as pool:
        for i, p in enumerate(pool.imap_unordered(make, jobs)):
            if i % 50 == 0:
                print(i, len(jobs), p, flush=True)
    print("GEN DONE", len(jobs), flush=True)
