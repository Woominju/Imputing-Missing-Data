"""Phase 8 data (protocol v2 section 4.1): NEW independent trajectories with the gen6/gen7 generator and seed rule s0_of(kappa, id).
(a) replicate / donor / oracle-pool trajectories (core8.lengths()), stored as results/extra/phase8/data/k{K}_id{ID}.npz (X: (n, 6));
(b) the k-NN reference (D5a): for q = 0..N_REF-1 ONE independent trajectory id 70000+q integrated up to its own time index
    n_q ~ Unif{SPAN..N_TEST-1}; only the window s(n_q-7..n_q) is kept -> exactly i.i.d. draws from the evaluation law P.
usage: python experiments/phase8/gen8.py NPROC"""
import os, sys
import numpy as np
from multiprocessing import Pool
from src.lorenz import generate_coupled_lorenz
from experiments.phase6.gen6 import s0_of
from experiments.phase5.core import M_ as _M
from experiments.phase8.core8 import DDIR, SZ, lengths, ref_ids, ref_time


def make(a):
    ki, j, n = a; p = f"{DDIR}/k{float(ki)}_id{j}.npz"
    if not os.path.exists(p):
        X, _, _ = generate_coupled_lorenz(float(ki), n_steps=n, dt=0.05, s0=s0_of(ki, j)); np.savez_compressed(p + ".tmp.npz", X=X); os.replace(p + ".tmp.npz", p)
    return p


def ref_point(a):
    ki, q, j = a; n = ref_time(ki, q)
    X, _, _ = generate_coupled_lorenz(float(ki), n_steps=n + 1, dt=0.05, s0=s0_of(ki, j))
    return X[n - _M + 1:n + 1].astype(np.float64), n


def main(nproc):
    os.makedirs(DDIR, exist_ok=True)
    with Pool(nproc) as pool:
        n = sum(1 for _ in pool.imap_unordered(make, [(ki, j, n) for ki in (0, 1) for j, n in lengths()], chunksize=4))
        print("P8 GEN trajectories", n, flush=True)
        for ki in (0, 1):
            f = f"{DDIR}/ref_k{float(ki)}.npz"
            if os.path.exists(f):
                continue
            R = pool.map(ref_point, [(ki, q, j) for q, j in enumerate(ref_ids())], chunksize=16)
            np.savez_compressed(f + ".tmp.npz", S=np.stack([w for w, _ in R]), n=np.array([t for _, t in R]))
            os.replace(f + ".tmp.npz", f); print("P8 GEN reference", ki, len(R), flush=True)
    print("P8 GEN DONE", SZ, flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]))
