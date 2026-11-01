"""S1R pre-registered analysis (protocol section 10). Replicate metric = sqrt(mean_j MSE_j), MSE_j pooled over 8 test
trajectories, averaged over probe seeds. Paired t 95%, TOST(delta = 0.02), sign test, Wilcoxon."""
import glob, numpy as np, pandas as pd
from scipy import stats
D = pd.concat([pd.read_csv(f) for f in glob.glob("results/extra/phase6/s1r/eval/k*_r*.csv")]); D["g"] = D.arm.str[:-1]
m = D.groupby(["kappa", "rep", "g", "j", "probe"])[["sse", "n"]].sum(); m = (m.sse / m.n).groupby(["kappa", "rep", "g", "j"]).mean()
Q = np.sqrt(m.groupby(["kappa", "rep", "g"]).mean()).unstack("g"); rows = []
for k, x in Q.groupby(level=0):
    d = (x.B - x.Sham).values; n = len(d); se = d.std(ddof=1) / np.sqrt(n); t95, t90 = stats.t.ppf(.975, n - 1), stats.t.ppf(.95, n - 1)
    rows.append(dict(kappa=k, n=n, B=x.B.mean(), Sham=x.Sham.mean(), mean=d.mean(), ci95_lo=d.mean() - t95 * se, ci95_hi=d.mean() + t95 * se,
                     tost_equiv=bool(d.mean() - t90 * se > -0.02 and d.mean() + t90 * se < 0.02), d_z=d.mean() / d.std(ddof=1),
                     n_negative=int((d < 0).sum()), sign_p=stats.binomtest(int((d < 0).sum()), n, .5).pvalue, wilcoxon_p=stats.wilcoxon(d).pvalue))
R = pd.DataFrame(rows); R.to_csv("results/extra/phase6/s1r/s1r_results.csv", index=False)
k1, k0 = R[R.kappa == 1].iloc[0], R[R.kappa == 0].iloc[0]
s1r, nc = k1.ci95_hi < 0, (k0.ci95_lo <= 0 <= k0.ci95_hi)
print(R.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print("S1R detected:", s1r, "| NC-R passed:", nc, "->", "independent confirmation" if s1r and nc else ("held" if s1r else "S1 not replicated"))
