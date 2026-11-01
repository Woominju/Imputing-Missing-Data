"""Phase 5 Gate C: observation-level identifiability / information audit (CPU, no learning, no AE).
Uses the KNOWN system only for diagnostics. Domain: states on test trajectories (ids 200..209) after a 250-tu transient.
Units: state and observations divided by the per-coordinate sd of train trajectory id 0 (same kappa).
 (1) data sanity: synchronisation, symmetry statistics (Gamma x Gamma vs Gamma x Id), cross-trajectory independence
 (2) exact symmetry check by integration: z-trajectories of s and g(s) for g in {GxG, GxId}
 (3) local Jacobians d O_m / d s (variational RK4, cross-checked against central finite differences at 3 step sizes)
     for structured masks; numerical rank with relative tolerance 1e-8 * sigma_max (and sigma_min reported raw)
 (4) global ambiguity search: nearest neighbours in observed coordinates among 160k train windows; sign-flip fraction
 (5) physics reference estimator y_j = x_j + (xdot_j - kappa (x_o - x_j)) / sigma, past-only backward differences
Writes results/extra/phase5/gateC/*.csv"""
import json, os, sys
import numpy as np, pandas as pd, torch
from multiprocessing import Pool
from scipy.integrate import solve_ivp
from src.lorenz import SIGMA, RHO, BETA, coupled_lorenz_rhs

D = "results/extra/phase5/data"; O = "results/extra/phase5/gateC"; m = 8; DT = 0.05
G = {"GxG": np.array([-1, -1, 1, -1, -1, 1.]), "GxId": np.array([-1, -1, 1, 1, 1, 1.])}
VCOL = {"x1": 0, "z1": 2, "x2": 3, "z2": 5}


def load(k, j): return np.load(f"{D}/k{float(k)}_id{j}.npz")["X"]


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


def flow_back(s, k, n):
    """states s(t - q dt), q = 0..n-1, by accurate backward integration (DOP853)."""
    sol = solve_ivp(lambda t, x: coupled_lorenz_rhs(t, x, k), (0, -(n - 1) * DT), s, method="DOP853",
                    t_eval=-np.arange(n) * DT, rtol=1e-11, atol=1e-12)
    return sol.y.T


def back_jac(s, k, n, sub=50):
    """d s(t - q dt) / d s(t) for q = 0..n-1 via backward RK4 variational equations."""
    out = [np.eye(6)]; x = s.copy(); P = np.eye(6); h = -DT / sub
    rhs = lambda x, P: (f6(x, k), j6(x, k) @ P)
    for q in range(1, n):
        for _ in range(sub):
            a = rhs(x, P); b = rhs(x + h / 2 * a[0], P + h / 2 * a[1]); c = rhs(x + h / 2 * b[0], P + h / 2 * b[1]); d = rhs(x + h * c[0], P + h * c[1])
            x = x + h / 6 * (a[0] + 2 * b[0] + 2 * c[0] + d[0]); P = P + h / 6 * (a[1] + 2 * b[1] + 2 * c[1] + d[1])
        out.append(P.copy())
    return out


def obs_rows(Pq, mask, sd):
    """mask: dict view -> boolean (m,) over lags; rows d obs / d s_t in normalised units (obs/sd_v, s/sd)."""
    R = []
    for v, mk in mask.items():
        c = VCOL[v]
        for q in range(m):
            if mk[q]:
                R.append(Pq[q][c] * sd / sd[c])
    return np.array(R) if R else np.zeros((0, 6))


def mask_set(rng, p):
    full = np.ones(m, bool); none = np.zeros(m, bool)
    rnd = lambda: rng.random(m) >= p
    def no_x(): return none.copy()
    S = {
        "own1_full": {"x1": full, "z1": full},
        "own1_zonly_full": {"z1": full},
        "own1_x_only_full": {"x1": full},
        "joint_full": {"x1": full, "z1": full, "x2": full, "z2": full},
        "joint_zonly_full": {"z1": full, "z2": full},
        f"own1_mcar{p}": {"x1": rnd(), "z1": rnd()},
        f"joint_mcar{p}": {"x1": rnd(), "z1": rnd(), "x2": rnd(), "z2": rnd()},
        f"own1_L0_mcar{p}": {"x1": no_x(), "z1": rnd()},
        f"joint_L0_mcar{p}": {"x1": no_x(), "z1": rnd(), "x2": rnd(), "z2": rnd()},
        "joint_L0_x2full": {"x1": no_x(), "z1": full, "x2": full, "z2": full},
    }
    return S


def local_audit(args):
    k, i, ts, seed = args
    X = load(k, 200 + i); sd = load(k, 0).std(0); rng = np.random.default_rng(seed); rows = []
    for t in ts:
        s = X[t]; Pq = back_jac(s, k, m)
        # finite-difference cross-check of d s(t-7) / d s(t) (central differences, three step sizes)
        fd_err = {}
        for eps in (1e-3, 1e-4, 1e-5):
            Jfd = np.zeros((6, 6))
            for c in range(6):
                e = np.zeros(6); e[c] = eps
                Jfd[:, c] = (flow_back(s + e, k, m)[-1] - flow_back(s - e, k, m)[-1]) / (2 * eps)
            fd_err[eps] = np.linalg.norm(Jfd - Pq[-1]) / np.linalg.norm(Pq[-1])
        for p in (0.3, 0.7):
            for name, mk in mask_set(rng, p).items():
                if p == 0.3 and "mcar" not in name and name != "joint_L0_x2full":
                    continue
                R = obs_rows(Pq, mk, sd); n = R.shape[0]
                for cols, tag in ((list(range(6)), "s6"), ([0, 1, 2], "s1")):
                    A = R[:, cols]
                    sv = np.linalg.svd(A, compute_uv=False) if n else np.zeros(1)
                    rank = int((sv > 1e-8 * sv.max()).sum()) if n and sv.max() > 0 else 0
                    amp = np.nan
                    if n and rank == len(cols):
                        Ap = np.linalg.pinv(A); amp = float(np.linalg.norm(Ap[1]))    # ||e_{y1}^T A^+||  (y1 is col index 1 in both)
                    rows.append(dict(kappa=k, traj=200 + i, t=int(t), p=p if "mcar" in name or "L0" in name else np.nan, mask=name,
                                     wrt=tag, n_obs=n, rank=rank, full_rank=rank == len(cols), sv_min=float(sv.min()) if n else 0.0,
                                     sv_max=float(sv.max()) if n else 0.0, y1_amp=amp,
                                     **{"fd_relerr_eps1e-3": fd_err[1e-3], "fd_relerr_eps1e-4": fd_err[1e-4], "fd_relerr_eps1e-5": fd_err[1e-5]}))
    return rows


def symmetry_check(k):
    X = load(k, 200); s = X[1000]; out = []
    for g, v in G.items():
        a = solve_ivp(lambda t, x: coupled_lorenz_rhs(t, x, k), (0, 20), s, method="DOP853", t_eval=np.arange(0, 20, DT), rtol=1e-11, atol=1e-12).y
        b = solve_ivp(lambda t, x: coupled_lorenz_rhs(t, x, k), (0, 20), s * v, method="DOP853", t_eval=np.arange(0, 20, DT), rtol=1e-11, atol=1e-12).y
        dz = np.abs(a[[2, 5]] - b[[2, 5]]); dxe = np.abs(a * v[:, None] - b)       # equivariance residual g(Phi s) - Phi(g s)
        out.append(dict(kappa=k, g=g, max_dz_t1=float(dz[:, :20].max()), max_dz_t5=float(dz[:, :100].max()), max_dz_t20=float(dz.max()),
                        max_equivariance_residual_t1=float(dxe[:, :20].max()), max_equivariance_residual_t20=float(dxe.max())))
    return out


def sanity(k):
    rows = []
    for j in list(range(8)) + list(range(100, 108)) + list(range(200, 210)):
        X = load(k, j); sd = X.std(0)
        dist = np.linalg.norm((X[:, :3] - X[:, 3:]) / sd[:3], axis=1)
        rows.append(dict(kappa=k, traj=j, n=len(X), mean_x1=X[:, 0].mean(), mean_x2=X[:, 3].mean(), frac_x1_pos=(X[:, 0] > 0).mean(),
                         corr_x1_x2=np.corrcoef(X[:, 0], X[:, 3])[0, 1], corr_y1_y2=np.corrcoef(X[:, 1], X[:, 4])[0, 1],
                         corr_z1_z2=np.corrcoef(X[:, 2], X[:, 5])[0, 1], E_x1x2_over_sd2=(X[:, 0] * X[:, 3]).mean() / (sd[0] * sd[3]),
                         frac_sync_lt0p05=(dist < 0.05).mean(), sd_x1=sd[0], sd_y1=sd[1], sd_z1=sd[2], sd_y2=sd[4]))
    return rows


def windows(X, cols, sd):
    Xn = X[:, cols] / sd[cols]
    return np.stack([Xn[m - 1 - q: len(X) - q] for q in range(m)], -1).reshape(len(X) - m + 1, -1)   # row = time m-1+..


def global_ambiguity(k):
    """Nearest neighbours (observed coords) of test windows among 8 x 20000 train windows; spread of hidden y1 and
    fraction of neighbours whose y1 has the opposite sign. Complete windows; z-only windows are the Gamma counterexample."""
    sd = load(k, 0).std(0); out = []
    sets = {"own1(x1,z1)": [0, 2], "own1 z1 only": [2], "joint z1,z2 only": [2, 5], "joint all4": [0, 2, 3, 5], "own1 x1 only": [0]}
    tr = [load(k, j) for j in range(8)]; te = [load(k, 200 + i) for i in range(10)]
    rng = np.random.default_rng(5)
    for name, cols in sets.items():
        F = torch.tensor(np.concatenate([windows(X, cols, sd) for X in tr]), dtype=torch.float32)
        Ytr = np.concatenate([X[m - 1:, [1, 4]] / sd[[1, 4]] for X in tr])
        Q, Yq = [], []
        for X in te:
            idx = rng.choice(len(X) - m + 1, 200, replace=False); Q.append(windows(X, cols, sd)[idx]); Yq.append(X[m - 1:][idx][:, [1, 4]] / sd[[1, 4]])
        Q = torch.tensor(np.concatenate(Q), dtype=torch.float32); Yq = np.concatenate(Yq)
        dd, ii = torch.cdist(Q, F).topk(10, largest=False); ii = ii.numpy()
        Yn = Ytr[ii]                                                       # (nq, 10, 2)
        for c, yn in ((0, "y1"), (1, "y2")):
            flip = (np.sign(Yn[:, :, c]) != np.sign(Yq[:, None, c])).mean()
            out.append(dict(kappa=k, view=name, target=yn, nn_obs_dist_median=float(dd[:, 0].median()),
                            nn_abs_dy_median=float(np.median(np.abs(Yn[:, :, c] - Yq[:, None, c]))),
                            nn10_spread_median=float(np.median(Yn[:, :, c].std(1))), frac_opposite_sign=float(flip),
                            random_pair_abs_dy=float(np.median(np.abs(Yq[:, c] - rng.permutation(Yq[:, c]))))))
    return out


def physics_ref(k):
    """Past-only physics estimator of y_j on test trajectories (complete data, then masked availability)."""
    sd = load(k, 0).std(0); out = []
    for i in range(10):
        X = load(k, 200 + i)
        for j, (cx, cy, co) in enumerate(((0, 1, 3), (3, 4, 0))):
            x, y, xo = X[:, cx], X[:, cy], X[:, co]
            d1 = np.r_[np.nan, np.diff(x)] / DT
            d2 = np.r_[np.nan, np.nan, (3 * x[2:] - 4 * x[1:-1] + x[:-2]) / (2 * DT)]
            dc = np.r_[np.nan, (x[2:] - x[:-2]) / (2 * DT), np.nan]          # central: uses x(t+1) -> SMOOTHING, labelled
            for tag, dx in (("bd1_past", d1), ("bd2_past", d2), ("central_SMOOTHING", dc)):
                for kt, kk in (("with_kappa_term", k), ("without_kappa_term", 0.0)):
                    yh = x + (dx - kk * (xo - x)) / SIGMA
                    e = ((yh - y) / sd[cy])[np.isfinite(yh)]
                    out.append(dict(kappa=k, traj=200 + i, sub=j + 1, deriv=tag, variant=kt, rmse_norm=float(np.sqrt(np.mean(e ** 2)))))
    for p in (0.3, 0.7):
        need = {"bd1 (x_t, x_t-1, x_o,t)": 3, "bd2 (x_t, x_t-1, x_t-2, x_o,t)": 4, "bd1 without x_o": 2}
        for nm, q in need.items():
            out.append(dict(kappa=k, traj=-1, sub=0, deriv=nm, variant=f"availability_p{p}", rmse_norm=(1 - p) ** q))
    return out


if __name__ == "__main__":
    os.makedirs(O, exist_ok=True); torch.set_num_threads(8)
    jobs = []
    for k in (0.0, 1.0):
        rng = np.random.default_rng(11 + int(k))
        for i in range(10):
            ts = rng.choice(np.arange(500, 4900), 30, replace=False)
            jobs.append((k, i, ts, 1000 * int(k) + i))
    with Pool(24) as pool:
        loc = pool.map(local_audit, jobs)
        sym = pool.map(symmetry_check, (0.0, 1.0))
        san = pool.map(sanity, (0.0, 1.0))
        phy = pool.map(physics_ref, (0.0, 1.0))
    pd.DataFrame([r for rr in loc for r in rr]).to_csv(f"{O}/local_jacobian.csv", index=False)
    pd.DataFrame([r for rr in sym for r in rr]).to_csv(f"{O}/symmetry_check.csv", index=False)
    pd.DataFrame([r for rr in san for r in rr]).to_csv(f"{O}/data_sanity.csv", index=False)
    pd.DataFrame([r for rr in phy for r in rr]).to_csv(f"{O}/physics_reference.csv", index=False)
    print("local/sym/sanity/physics done", flush=True)
    pd.DataFrame([r for k in (0.0, 1.0) for r in global_ambiguity(k)]).to_csv(f"{O}/global_ambiguity.csv", index=False)
    print("ALL DONE", flush=True)
