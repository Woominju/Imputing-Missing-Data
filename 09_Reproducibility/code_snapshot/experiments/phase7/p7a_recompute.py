"""Phase 7 Gate P7-A: independent recomputation (new code, raw sums only) of the G1 certificates on L0 and L0c0
(kappa = 1, p = 0.7), with three interval methods. Reads results/extra/phase6/g1/k1.0_p0.7.csv (and kappa 0 null).
(1) cluster bootstrap over the 32 test trajectories, percentile, seed 0, 10,000 draws -- same procedure as analyze_g1
    (reproduces the reported LCB exactly only if the RNG call order matches; reported separately);
(2) cluster bootstrap with an independent seed (12345) -- Monte-Carlo variability of the LCB;
(3) one-sided t bound on trajectory-level ratios G_i = S_i / n_i (equal-weight trajectory mean; a different estimand weight).
All three are APPROXIMATE (asymptotic) bounds; none has a proven finite-sample 95% coverage.
Output: results/extra/phase7/p7a/g1_recompute.csv"""
import numpy as np, pandas as pd
from scipy import stats

rows = []
for k in (1.0, 0.0):
    D = pd.read_csv(f"results/extra/phase6/g1/k{k}_p0.7.csv")
    for st in ("L0", "L0c0", "L0c1"):
        for q in ("cert_UV", "cert_UVc", "cert_UVs", "cert_U"):
            X = D[(D.stratum == st) & (D.quantity == q)].groupby("traj")[["sum", "n"]].sum()
            S, N = X["sum"].values, X["n"].values; est = S.sum() / N.sum(); T = len(S)
            out = dict(kappa=k, stratum=st, quantity=q, n_traj=T, n_points=int(N.sum()), estimate=est)
            for nm, seed in (("boot_seed0", 0), ("boot_seed12345", 12345)):
                rng = np.random.default_rng(seed); idx = rng.integers(0, T, (10000, T))
                b = S[idx].sum(1) / N[idx].sum(1); out[f"lcb95_{nm}"] = np.percentile(b, 5)
            g = S / N; out["traj_mean"] = g.mean(); out["lcb95_t_trajmean"] = g.mean() - stats.t.ppf(.95, T - 1) * g.std(ddof=1) / np.sqrt(T)
            out["p8_times_lcb_seed0"] = 0.7 ** 8 * max(out["lcb95_boot_seed0"], 0)
            rows.append(out)
R = pd.DataFrame(rows); R.to_csv("results/extra/phase7/p7a/g1_recompute.csv", index=False)
pd.set_option("display.width", 250); print(R.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
A = pd.read_csv("results/extra/phase6/g1/analysis_g1.csv")
print(A[(A.kappa == 1) & (A.p == 0.7) & A.stratum.isin(["L0", "L0c0"]) & (A.quantity == "G(UV)")][["stratum", "estimate", "lo"]].to_string(index=False, float_format=lambda v: f"{v:.6f}"))
