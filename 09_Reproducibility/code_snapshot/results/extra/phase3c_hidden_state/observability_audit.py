"""Phase 3C CPU audit: Lorenz observed through h1 = x, h2 = z only (y hidden), dt = 0.05, past windows m = 8, tau = 1.
(a) local observability: flow-Jacobian rows d x_j(t-k)/d s(t) for patterns; singular values in normalised units
    (observations / train sd of x,z ; state / train sd of x,y,z); rank, condition number, y-amplification
    amp_y = ||e_y^T J^+|| (local sensitivity of the least-squares y estimate to observation perturbations).
(b) global ambiguity (finite sample): for a test window, k nearest train windows in the observed coordinates
    (only coordinates observed in the query window); spread of the true y and of the full state among neighbours;
    Gamma check for z-only: fraction of neighbours on the opposite lobe (sign x differs).
(c) analytical past-only finite-difference baseline for y under the x MCAR mask (no y, no future values used).
Writes observability_*.csv / .log in this directory."""
import numpy as np, pandas as pd
from src.missingness import prepare, chrono_split
from src.observability import step_jacobians, backward_flow_jacobians
SIG = 10.0; DT = 0.05; m = 8
O = "results/extra/phase3c_hidden_state"
X = np.load("results/extra/aim_autonomous_20261027/coarse_dt0.05/data/lorenz_dt0.05_20000.npz")["X"]; T = len(X)
sp = chrono_split(T); a_tr, b_tr = sp["train"]; a, b = sp["test"]
sd_state = X[a_tr:b_tr].std(0); sd_obs = sd_state[[0, 2]]
rng = np.random.default_rng(0)
# ---------- (a) local observability on test windows ----------
idx = np.arange(a + m, b - 1, 3)                     # J defined for t <= T-2
J = step_jacobians(X[:-1], DT, sub=20)                       # J[t] = DPhi_dt(s_t)
B = backward_flow_jacobians(J, m - 1)                        # B[k, t] = DPhi_{-k dt}(s_t)
def rows(t, obs):                                            # obs: (m, 2) bool for (x,z) at lags 0..m-1
    R = []
    for k in range(m):
        for j, var in enumerate([0, 2]):
            if obs[k, j]:
                R.append(B[k, t, var, :] / sd_obs[j] * sd_state)       # normalised obs per normalised state
    return np.array(R)
pats = {"x_full": np.c_[np.ones(m, bool), np.zeros(m, bool)], "z_full": np.c_[np.zeros(m, bool), np.ones(m, bool)],
        "xz_full": np.ones((m, 2), bool), "x_lag0_only": np.r_[[[True, False]], np.zeros((m - 1, 2), bool)],
        "xz_lag0_only": np.r_[[[True, True]], np.zeros((m - 1, 2), bool)]}
loc = []
for name in list(pats) + ["nat_p0.3", "nat_p0.3_x_part", "nat_p0.7"]:
    for t in idx:
        if name.startswith("nat"):
            p = 0.3 if "0.3" in name else 0.7
            obs = rng.random((m, 2)) >= p
            if name.endswith("x_part"): obs[:, 1] = False
        else:
            obs = pats[name]
        R = rows(t, obs)
        if len(R) == 0:
            loc.append(dict(pattern=name, t=t, n_obs=0, rank=0, smin=0.0, cond=np.inf, amp_y=np.inf)); continue
        sv = np.linalg.svd(R, compute_uv=False); rk = int((sv > 1e-8 * sv[0]).sum())
        amp = np.linalg.norm(np.linalg.pinv(R)[1]) if rk == 3 else np.inf
        loc.append(dict(pattern=name, t=t, n_obs=len(R), rank=rk, smin=sv[min(2, len(sv) - 1)] if len(sv) >= 3 else 0.0,
                        cond=sv[0] / sv[2] if len(sv) >= 3 and sv[2] > 0 else np.inf, amp_y=amp, x_near0=abs(X[t, 0]) < 2.0))
L = pd.DataFrame(loc); L.to_csv(f"{O}/observability_local.csv", index=False)
summ = L.groupby("pattern").agg(n_obs=("n_obs", "mean"), frac_rank3=("rank", lambda r: (r == 3).mean()),
                               smin_q10=("smin", lambda s: s.quantile(.1)), smin_q50=("smin", "median"),
                               cond_q50=("cond", "median"), cond_q90=("cond", lambda s: s.quantile(.9)),
                               ampy_q50=("amp_y", "median"), ampy_q90=("amp_y", lambda s: s.quantile(.9)))
print("## (a) local observability (normalised units)\n", summ.round(3).to_string())
xs = L[L.pattern == "x_full"]
print("x_full amp_y median |x|<2 vs |x|>=2:", round(xs[xs.x_near0].amp_y.median(), 2), round(xs[~xs.x_near0].amp_y.median(), 2))
# ---------- (b) global ambiguity: kNN in observed coordinates ----------
def windows(lo, hi):
    tt = np.arange(lo + m - 1, hi)
    W = np.stack([X[tt - k][:, [0, 2]] / sd_obs for k in range(m)], 1)       # (N, m, 2)
    return tt, W
ttr, Wtr = windows(a_tr, b_tr); tte, Wte = windows(a, b)
Str = X[ttr] / sd_state; Ste = X[tte] / sd_state
glob = []
q_idx = rng.choice(len(tte), 600, replace=False)
for name in ["x_full", "z_full", "xz_full", "nat_p0.3", "nat_p0.3_x_part", "x_lag0_only", "xz_lag0_only"]:
    for qi in q_idx:
        if name.startswith("nat"):
            obs = rng.random((m, 2)) >= 0.3
            if name.endswith("x_part"): obs[:, 1] = False
        else:
            obs = pats[name]
        if obs.sum() == 0: continue
        dd = (((Wtr - Wte[qi]) ** 2) * obs).sum((1, 2))
        nn = np.argpartition(dd, 10)[:10]
        dy = np.abs(Str[nn, 1] - Ste[qi, 1]); dS = np.linalg.norm(Str[nn] - Ste[qi], axis=1)
        opp = np.mean(np.sign(Str[nn, 0]) != np.sign(Ste[qi, 0]))
        glob.append(dict(pattern=name, dy_median=np.median(dy), dy_max=dy.max(), dS_max=dS.max(), opposite_lobe=opp, n_obs=int(obs.sum())))
G = pd.DataFrame(glob); G.to_csv(f"{O}/observability_global.csv", index=False)
print("\n## (b) finite-sample ambiguity, 10-NN in observed coords (normalised; y sd = 1)\n",
      G.groupby("pattern")[["n_obs", "dy_median", "dy_max", "dS_max", "opposite_lobe"]].agg(["median", lambda s: s.quantile(.9)]).round(3).to_string())
# ---------- (c) analytical past-only finite-difference baseline ----------
ytrue = X[:, 1]; mu_y, sd_y = X[a_tr:b_tr, 1].mean(), X[a_tr:b_tr, 1].std()
rows_c = []
for seed in [0, 1, 2]:
    d = prepare(X[:, [0, 2]], 0.3, seed); Mx = d["M"][:, 0] > 0                  # x input mask (observed entries)
    tt = np.arange(a, b)
    for name in ["bd1_full", "bd1_mask", "bdnear_mask", "bd2_mask"]:
        yh = np.full(len(tt), np.nan)
        for n_, t in enumerate(tt):
            if name == "bd1_full":
                yh[n_] = X[t, 0] + (X[t, 0] - X[t - 1, 0]) / (SIG * DT)
            elif not Mx[t]:
                continue                                                          # x_t not observed -> not computable
            elif name == "bd1_mask" and Mx[t - 1]:
                yh[n_] = X[t, 0] + (X[t, 0] - X[t - 1, 0]) / (SIG * DT)
            elif name == "bdnear_mask":
                js = [j for j in range(1, m) if Mx[t - j]]
                if js: j = js[0]; yh[n_] = X[t, 0] + (X[t, 0] - X[t - j, 0]) / (SIG * j * DT)
            elif name == "bd2_mask" and Mx[t - 1] and Mx[t - 2]:
                yh[n_] = X[t, 0] + (3 * X[t, 0] - 4 * X[t - 1, 0] + X[t - 2, 0]) / (2 * DT) / SIG
        ok = np.isfinite(yh)
        e = (yh[ok] - ytrue[tt][ok]) / sd_y
        rows_c.append(dict(seed=seed, estimator=name, coverage=ok.mean(), rmse_norm=np.sqrt((e ** 2).mean()), mae_norm=np.abs(e).mean()))
C_ = pd.DataFrame(rows_c); C_.to_csv(f"{O}/fd_baseline.csv", index=False)
print("\n## (c) finite-difference y baseline on test (normalised by train sd of y; mean predictor RMSE ~ 1)\n",
      C_.groupby("estimator")[["coverage", "rmse_norm", "mae_norm"]].agg(["mean", "min", "max"]).round(4).to_string())
