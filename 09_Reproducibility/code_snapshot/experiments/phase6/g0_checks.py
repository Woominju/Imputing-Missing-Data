"""Phase 6 Gate G0: read-only integrity checks of Phase 5 code/results (CPU). Nothing in Phase 5 is modified.
(1) Gate F MV-shift: original-time offset of every other-view input relative to the target entry.
(2) Phase 5 Sham donor map (train / val / test) and overlaps with real targets.
(3) Dependence of per-replicate B - Sham contrasts induced by the cyclic donor (r+1)%8; sensitivity intervals.
Outputs results/extra/phase6/g0/*.csv"""
import glob, numpy as np, pandas as pd
from scipy import stats
import experiments.phase5.external as X

O = "results/extra/phase6/g0"


def mvshift_offsets():
    rows = []
    for ds in ("beijing", "jena"):
        S = X.setup(ds, 0.7, 0); P, T, V = S["P"], S["T"], S["V"]; s, e = S["sp"][2]
        t = np.arange(s + 7, e)
        for k in range(V):                      # target view k at position t is original time P[t,k]
            for k2 in range(V):
                if k2 == k:
                    continue
                for lag in (0, 7):              # other view k2, lag `lag`: original time P[t-lag, k2]
                    off = P[t - lag, k2] - P[t, k]          # >0 : the input is in the FUTURE of the target (original time)
                    rows.append(dict(ds=ds, target_view=X.VIEWS[ds][k], input_view=X.VIEWS[ds][k2], lag=lag,
                                     frac_future=float((off > 0).mean()), median_offset_h=float(np.median(off)),
                                     max_offset_h=int(off.max()), min_offset_h=int(off.min())))
    D = pd.DataFrame(rows); D.to_csv(f"{O}/mvshift_time_offsets.csv", index=False); return D


def sham_map():
    rows = []
    for r in range(8):
        rows.append(dict(role="train", real=r, sham_donor=(r + 1) % 8, donor_is_real_of_replicate=(r + 1) % 8))
        rows.append(dict(role="val", real=100 + r, sham_donor=100 + (r + 1) % 8, donor_is_real_of_replicate=(r + 1) % 8))
    for i in range(10):
        rows.append(dict(role="test", real=200 + i, sham_donor=200 + (i + 1) % 10, donor_is_real_of_replicate=-1))
    D = pd.DataFrame(rows); D["donor_is_also_a_real_target"] = True
    D.to_csv(f"{O}/phase5_sham_donor_map.csv", index=False); return D


def bsham_dependence():
    D = pd.concat([pd.read_csv(f) for f in glob.glob("results/extra/phase5/eval/k*_probe.csv")], ignore_index=True)
    D = D[(D.seed == D.rep) & D.probe.isin(["mlp0", "mlp1"]) & D.target.str.startswith("y") & (D.stratum == "all")]
    D["g"] = D.feature.str.replace(r"[12]$", "", regex=True)
    D = D[(D.feature.str[-1] == D.j.astype(str)) & D.g.isin(["A", "B", "Sham"])]
    mse = D.groupby(["kappa", "p", "g", "rep", "j", "probe"]).apply(lambda x: x.sse.sum() / x.n.sum()).groupby(["kappa", "p", "g", "rep", "j"]).mean()
    R = np.sqrt(mse.groupby(["kappa", "p", "g", "rep"]).mean()).unstack("g")
    out = []
    for (k, p), x in R.groupby(level=[0, 1]):
        d = (x["B"] - x["Sham"]).values; n = len(d); m = d.mean(); sd = d.std(ddof=1)
        # cyclic lag-1 autocorrelation (replicate r's sham donor is replicate r+1's real trajectory)
        dc = d - m; rho1 = float((dc * np.roll(dc, -1)).sum() / (dc ** 2).sum())
        h = stats.t.ppf(.975, n - 1) * sd / np.sqrt(n)
        # sensitivity: variance of the mean under cyclic 1-dependence with correlation rho (bounded to the admissible range)
        def half(rho):
            v = sd ** 2 / n * (1 + 2 * rho)
            return stats.t.ppf(.975, n - 1) * np.sqrt(max(v, 0))
        # B_r - Sham_r vs. Sham_{r-1}: correlation of Sham_r with A_{r+1}, B_{r+1} (shared trajectory r+1 subsystem 3-j)
        cB = float(np.corrcoef(x["Sham"].values, np.roll(x["B"].values, -1))[0, 1])
        out.append(dict(kappa=k, p=p, n=n, mean=m, sd=sd, t_ci_lo=m - h, t_ci_hi=m + h, cyclic_rho1_hat=rho1,
                        ci_hi_rho_plus0p5=m + half(0.5), ci_lo_rho_plus0p5=m - half(0.5),
                        sign_test_p_two_sided=float(stats.binomtest(int((d < 0).sum()), n, 0.5).pvalue),
                        corr_Sham_r_B_rplus1=cB))
    D = pd.DataFrame(out); D.to_csv(f"{O}/phase5_bsham_dependence.csv", index=False); return D


if __name__ == "__main__":
    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 400)
    A = mvshift_offsets(); print(A[(A.lag == 0)].groupby(["ds", "target_view"]).agg(frac_future=("frac_future", "mean"), max_h=("max_offset_h", "max")).to_string())
    print(A[(A.ds == "beijing") & (A.target_view == "NO2") & (A.lag == 0)].to_string(index=False))
    print(sham_map().to_string(index=False))
    print(bsham_dependence().to_string(index=False, float_format=lambda v: f"{v:.4f}"))
