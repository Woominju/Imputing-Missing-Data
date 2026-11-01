"""Oracle-only local Jacobians of the full delay observation D(s) = [h_v(Phi^{-q dt}(s))]_{v in x1,z1,x2,z2; q=0..7} w.r.t.
the 6-D state s(t), at G2 test states (every 5th time index t >= 7 of each test target trajectory). Uses the TRUE hidden
state: diagnostic only, never an input of any model and never used by mask-only strata. Backward RK4 variational equations
(50 sub-steps per sample, vectorised version of experiments/phase5/obs_audit.back_jac, cross-checked in tests).
Units: rows divided by the per-view sd and columns multiplied by the per-state-coordinate sd of the fixed normalisation pool.
Output results/extra/phase6/jac/k{K}_r{R}.npz : J (N, 4 views, 8 lags, 6), traj index, time index."""
import os, sys
import numpy as np
from experiments.phase5.obs_audit import f6, j6, DT
from experiments.phase6.core6 import load, ids, norm_stats, NT6, OBS

O = "results/extra/phase6/jac"; STRIDE = 5


def back_jac_batch(S, k, n=8, sub=50):
    """S (N, 6) -> list over q of d s(t - q dt) / d s(t), each (N, 6, 6)."""
    N = len(S); x = S.copy(); P = np.broadcast_to(np.eye(6), (N, 6, 6)).copy(); h = -DT / sub; out = [P.copy()]
    rhs = lambda x, P: (f6(x, k), j6(x, k) @ P)
    for q in range(1, n):
        for _ in range(sub):
            a = rhs(x, P); b = rhs(x + h / 2 * a[0], P + h / 2 * a[1]); c = rhs(x + h / 2 * b[0], P + h / 2 * b[1]); d = rhs(x + h * c[0], P + h * c[1])
            x = x + h / 6 * (a[0] + 2 * b[0] + 2 * c[0] + d[0]); P = P + h / 6 * (a[1] + 2 * b[1] + 2 * c[1] + d[1])
        out.append(P.copy())
    return out


def run(k, r):
    os.makedirs(O, exist_ok=True); f = f"{O}/k{float(k)}_r{r}.npz"
    if os.path.exists(f):
        return f
    mu_o, sd_o, mu_s, sd_s = norm_stats(k); I = ids(r); Js, TR, TT = [], [], []
    for i in range(NT6):
        S = load(k, I["test"][i]); t = np.arange(7, len(S), STRIDE)
        Pq = back_jac_batch(S[t], k)                                       # 8 x (N, 6, 6)
        J = np.stack([Pq[q][:, OBS, :] for q in range(8)], 2)              # (N, 4 views, 8 lags, 6)
        J = J / sd_o[None, :, None, None] * sd_s[None, None, None, :]
        Js.append(J.astype(np.float32)); TR.append(np.full(len(t), i)); TT.append(t)
    np.savez_compressed(f, J=np.concatenate(Js), traj=np.concatenate(TR), t=np.concatenate(TT)); return f


if __name__ == "__main__":
    from multiprocessing import Pool
    jobs = [(k, r) for k in (0.0, 1.0) for r in range(24)]
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 16) as pool:
        for f in pool.starmap(run, jobs):
            print(f, flush=True)
