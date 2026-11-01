"""Oracle-only local Jacobians for P7-B test trajectories (uses the TRUE state; diagnostic strata only). Same computation as
experiments/phase6/jac6.py (back_jac_batch, every 5th time index, normalised units)."""
import os, sys
import numpy as np
from experiments.phase6.jac6 import back_jac_batch, STRIDE
from experiments.phase7.core7 import load, ids, norm_stats, NT7, OBS, R7, ROOT

O = f"{ROOT}/jac"


def run(k, r):
    os.makedirs(O, exist_ok=True); f = f"{O}/k{float(k)}_r{r}.npz"
    if os.path.exists(f):
        return f
    _, sd_o, _, sd_s = norm_stats(k); I = ids(r); Js, TR, TT = [], [], []
    for i in range(NT7):
        S = load(k, I["test"][i]); t = np.arange(7, len(S), STRIDE); Pq = back_jac_batch(S[t], k)
        J = np.stack([Pq[q][:, OBS, :] for q in range(8)], 2) / sd_o[None, :, None, None] * sd_s[None, None, None, :]
        Js.append(J.astype(np.float32)); TR.append(np.full(len(t), i)); TT.append(t)
    np.savez_compressed(f, J=np.concatenate(Js), traj=np.concatenate(TR), t=np.concatenate(TT)); return f


if __name__ == "__main__":
    from multiprocessing import Pool
    with Pool(int(sys.argv[1])) as pool:
        for f in pool.starmap(run, [(k, r) for k in (0.0, 1.0) for r in range(R7)]):
            pass
    print("P7 JAC DONE", flush=True)
