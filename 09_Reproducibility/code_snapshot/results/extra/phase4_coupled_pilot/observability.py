"""Phase 4 coupled-Lorenz observability audit (CPU; run BEFORE any coupled training; kappa is chosen from this file only).
State s = (x1,y1,z1,x2,y2,z2); observation candidates x1, z1, x2, z2 (y1, y2 hidden) and the stress-test observables
h+ = x1 + x2, h- = x1 - x2. Past delay windows m = 8, tau = 1, dt = 0.05, chronological split as src.missingness.
 (1) synchronisation and intrinsic dimension (TwoNN) per kappa;
 (2) local: rows d h(s_{t-k}) / d s_t via RK4 variational equations of the 6-D field -> rank, singular values (normalised
     units: observations / train sd, state / train sd), y-amplification ||e_y^T R^+||; LOCAL ONLY, says nothing global;
 (3) global finite-sample fibre ambiguity: 10-NN of a test window among clean train windows in the observed coordinates;
     spread of the true hidden quantities among neighbours (normalised), versus random-pair spread.
MCAR patterns are drawn independently per query (entries of the m x |views| window). Writes obs_*.csv here."""
import json, sys
import numpy as np, pandas as pd
from multiprocessing import Pool
from src.lorenz import SIGMA, RHO, BETA
from src.missingness import chrono_split

O = "results/extra/phase4_coupled_pilot"; m = 8; DT = 0.05; NQ = 300
KAPPAS = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0]
VIEWSETS = {"x1": ["x1"], "z1": ["z1"], "x2": ["x2"], "x1,z1": ["x1", "z1"], "x1,x2": ["x1", "x2"], "x1,z2": ["x1", "z2"],
            "x1,z1,x2,z2": ["x1", "z1", "x2", "z2"], "h+": ["h+"], "h-": ["h-"], "h+,h-": ["h+", "h-"]}


def obs_fun(S, v):
    return {"x1": S[..., 0], "z1": S[..., 2], "x2": S[..., 3], "z2": S[..., 5], "h+": S[..., 0] + S[..., 3], "h-": S[..., 0] - S[..., 3]}[v]


GRAD = {"x1": np.eye(6)[0], "z1": np.eye(6)[2], "x2": np.eye(6)[3], "z2": np.eye(6)[5], "h+": np.eye(6)[0] + np.eye(6)[3], "h-": np.eye(6)[0] - np.eye(6)[3]}


def f6(S, k):
    x1, y1, z1, x2, y2, z2 = (S[..., i] for i in range(6))
    return np.stack([SIGMA * (y1 - x1) + k * (x2 - x1), x1 * (RHO - z1) - y1, x1 * y1 - BETA * z1,
                     SIGMA * (y2 - x2) + k * (x1 - x2), x2 * (RHO - z2) - y2, x2 * y2 - BETA * z2], -1)


def j6(S, k):
    J = np.zeros(S.shape[:-1] + (6, 6))
    for o in (0, 3):
        x, y, z = S[..., o], S[..., o + 1], S[..., o + 2]
        J[..., o, o], J[..., o, o + 1] = -SIGMA - k, SIGMA
        J[..., o + 1, o], J[..., o + 1, o + 1], J[..., o + 1, o + 2] = RHO - z, -1.0, -x
        J[..., o + 2, o], J[..., o + 2, o + 1], J[..., o + 2, o + 2] = y, x, -BETA
    J[..., 0, 3] = k; J[..., 3, 0] = k
    return J


def step_jac(S, k, sub=25):
    s = S.copy(); P = np.broadcast_to(np.eye(6), S.shape[:-1] + (6, 6)).copy(); h = DT / sub
    rhs = lambda s, P: (f6(s, k), j6(s, k) @ P)
    for _ in range(sub):
        a = rhs(s, P); b = rhs(s + h / 2 * a[0], P + h / 2 * a[1]); c = rhs(s + h / 2 * b[0], P + h / 2 * b[1]); d = rhs(s + h * c[0], P + h * c[1])
        s = s + h / 6 * (a[0] + 2 * b[0] + 2 * c[0] + d[0]); P = P + h / 6 * (a[1] + 2 * b[1] + 2 * c[1] + d[1])
    return P


def twonn(Z, rng, n=4000):
    Z = Z[rng.choice(len(Z), n, replace=False)]
    D = np.sqrt(((Z[:, None] - Z[None]) ** 2).sum(-1)); np.fill_diagonal(D, np.inf); D.sort(1)
    mu = D[:, 1] / D[:, 0]; mu = mu[np.isfinite(mu) & (mu > 1)]
    return len(mu) / np.log(mu).sum()


def audit(k):
    rng = np.random.default_rng(int(k * 100) + 7)
    X = np.load(f"{O}/data/coupled_k{k}_dt0.05_traj0.npz")["X"]; T = len(X); sp = chrono_split(T)
    a_tr, b_tr = sp["train"]; a_te, b_te = sp["test"]; sd = X[a_tr:b_tr].std(0)
    Sn = X / sd
    dist = np.linalg.norm((X[:, :3] - X[:, 3:]) / sd[:3], axis=1)
    SY = dict(kappa=k, corr_x1_x2=np.corrcoef(X[:, 0], X[:, 3])[0, 1], corr_z1_z2=np.corrcoef(X[:, 2], X[:, 5])[0, 1],
              sync_dist_q10=np.quantile(dist, .1), sync_dist_q50=np.median(dist), frac_sync_lt0p05=(dist < 0.05).mean(),
              frac_sync_lt0p2=(dist < 0.2).mean(), min_dist=dist.min(), twonn_6d=twonn(Sn[a_tr:b_tr], rng), twonn_sub1=twonn(Sn[a_tr:b_tr, :3], rng))
    # ---- local Jacobians on test anchors ----
    tq = rng.choice(np.arange(a_te + m, b_te - 1), NQ, replace=False)
    Jstep = {t: step_jac(X[t][None], k)[0] for t in np.unique(np.r_[[tq - j for j in range(1, m)]].ravel())}
    LOC, GLB = [], []
    Bk = {}
    for t in tq:
        B = [np.eye(6)]
        for j in range(1, m):
            B.append(np.linalg.solve(Jstep[t - j], B[-1]))          # d s_{t-j} / d s_t
        Bk[t] = B
    osd = {v: obs_fun(X[a_tr:b_tr], v).std() for v in GRAD}
    for vs_name, vs in VIEWSETS.items():
        for pat in ("full", "mcar0.3", "mcar0.7"):
            for t in tq:
                obs = np.ones((m, len(vs)), bool) if pat == "full" else rng.random((m, len(vs))) >= float(pat[4:])
                R = np.array([GRAD[v] @ Bk[t][j] / osd[v] * sd for j in range(m) for vi, v in enumerate(vs) if obs[j, vi]])
                if len(R) == 0:
                    continue
                sv = np.linalg.svd(R, compute_uv=False); sv6 = np.r_[sv, np.zeros(6)][:6]
                rk = int((sv6 > 1e-9 * sv6[0]).sum())
                pinv = np.linalg.pinv(R) if rk == 6 else None
                LOC.append(dict(kappa=k, views=vs_name, pattern=pat, n_obs=len(R), rank=rk, s1=sv6[0], s3=sv6[2], s6=sv6[5],
                                amp_y1=np.linalg.norm(pinv[1]) if pinv is not None else np.inf,
                                amp_y2=np.linalg.norm(pinv[4]) if pinv is not None else np.inf))
    # ---- global finite-sample ambiguity ----
    def win(lo, hi, vs):
        tt = np.arange(lo + m - 1, hi)
        return tt, np.stack([np.stack([obs_fun(X[tt - j], v) / osd[v] for v in vs], -1) for j in range(m)], 1)
    rp = rng.integers(a_tr, b_tr, (2, 5000)); base = np.abs(Sn[rp[0]] - Sn[rp[1]])
    REF = dict(kappa=k, views="random pairs", pattern="-", dy1_q50=np.median(base[:, 1]), dy2_q50=np.median(base[:, 4]),
               ds1_q50=np.median(np.linalg.norm(Sn[rp[0], :3] - Sn[rp[1], :3], axis=1)), ds2_q50=np.median(np.linalg.norm(Sn[rp[0], 3:] - Sn[rp[1], 3:], axis=1)))
    GLB.append(REF)
    qte = rng.choice(np.arange(a_te + m - 1, b_te), NQ, replace=False)
    for vs_name, vs in VIEWSETS.items():
        ttr, Wtr = win(a_tr, b_tr, vs)
        for pat in ("full", "mcar0.3", "mcar0.7"):
            for t in qte:
                obs = np.ones((m, len(vs)), bool) if pat == "full" else rng.random((m, len(vs))) >= float(pat[4:])
                if obs.sum() == 0:
                    continue
                q = np.stack([np.stack([obs_fun(X[t - j], v) / osd[v] for v in vs]) for j in range(m)])
                dd = (((Wtr - q) ** 2) * obs).sum((1, 2)); nn = ttr[np.argpartition(dd, 10)[:10]]
                d = np.abs(Sn[nn] - Sn[t])
                nobs = {f"nobs_{v}": int(obs[:, i].sum()) for i, v in enumerate(vs)}
                row = dict(kappa=k, views=vs_name, pattern=pat, n_obs=int(obs.sum()), dy1_q50=np.median(d[:, 1]), dy2_q50=np.median(d[:, 4]),
                           ds1_q50=np.median(np.linalg.norm(Sn[nn, :3] - Sn[t, :3], axis=1)), ds2_q50=np.median(np.linalg.norm(Sn[nn, 3:] - Sn[t, 3:], axis=1)),
                           # symmetry diagnostics: neighbour closer to the swapped / Gamma-images of the query than to the query itself
                           frac_swap=np.mean(np.linalg.norm(Sn[nn] - Sn[t][[3, 4, 5, 0, 1, 2]], axis=1) < np.linalg.norm(Sn[nn] - Sn[t], axis=1)),
                           frac_gamma_both=np.mean(np.linalg.norm(Sn[nn] - Sn[t] * [-1, -1, 1, -1, -1, 1], axis=1) < np.linalg.norm(Sn[nn] - Sn[t], axis=1)),
                           frac_swap_gamma=np.mean(np.linalg.norm(Sn[nn] - (Sn[t] * [-1, -1, 1, -1, -1, 1])[[3, 4, 5, 0, 1, 2]], axis=1) < np.linalg.norm(Sn[nn] - Sn[t], axis=1)),
                           **nobs)
                GLB.append(row)
    return SY, LOC, GLB


if __name__ == "__main__":
    with Pool(len(KAPPAS)) as pool:
        res = pool.map(audit, KAPPAS)
    pd.DataFrame([r[0] for r in res]).to_csv(f"{O}/obs_sync.csv", index=False)
    pd.DataFrame([x for r in res for x in r[1]]).to_csv(f"{O}/obs_local.csv", index=False)
    pd.DataFrame([x for r in res for x in r[2]]).to_csv(f"{O}/obs_global.csv", index=False)
    print("ALL DONE")
