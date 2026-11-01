"""POST-HOC / EXPLORATORY (not pre-registered; written after seeing primary results, 2026-10-28 ~12:45).
(1) B - Sham: same architecture/input count, real vs. sham other-view columns -> isolates information from input-count cost.
(2) Difference-in-differences (B-A)_{kappa=1} - (B-A)_{kappa=0}: replicate r at kappa=1 and kappa=0 are independent,
    so CI uses Welch t on per-replicate contrasts. Same replicate metric as analyze.py (sqrt(mean_j MSE_j), MLP probes 0/1)."""
import glob, numpy as np, pandas as pd
from scipy import stats

D = pd.concat([pd.read_csv(f) for f in glob.glob("results/extra/phase5/eval/k*_probe.csv")], ignore_index=True)
D = D[(D.seed == D.rep) & D.probe.isin(["mlp0", "mlp1"]) & D.target.str.startswith("y")]
D["g"] = D.feature.str.replace(r"[12]$", "", regex=True)
D = D[(D.feature.str[-1] == D.j.astype(str)) & D.g.isin(["A", "B", "Sham"])]
mse = D.groupby(["kappa", "p", "stratum", "g", "rep", "j", "probe"]).apply(lambda x: x.sse.sum() / x.n.sum()).groupby(
    ["kappa", "p", "stratum", "g", "rep", "j"]).mean()
R = np.sqrt(mse.groupby(["kappa", "p", "stratum", "g", "rep"]).mean()).unstack("g")
out = []
for (k, p, s), x in R.groupby(level=[0, 1, 2]):
    for a, b in [("B", "Sham"), ("B", "A"), ("Sham", "A")]:
        d = (x[a] - x[b]).values; m = d.mean(); h = stats.t.ppf(.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
        out.append(dict(kappa=k, p=p, stratum=s, contrast=f"{a}-{b}", mean=m, lo=m - h, hi=m + h, n_neg=int((d < 0).sum())))
for p in [0.3, 0.7]:
    for s in ["all", "L0", "L1", "L2+"]:
        for a, b in [("B", "A"), ("B", "Sham")]:
            d1 = (R.loc[(1.0, p, s)][a] - R.loc[(1.0, p, s)][b]).values; d0 = (R.loc[(0.0, p, s)][a] - R.loc[(0.0, p, s)][b]).values
            m = d1.mean() - d0.mean(); se = np.sqrt(d1.var(ddof=1) / 8 + d0.var(ddof=1) / 8)
            dfw = se ** 4 / ((d1.var(ddof=1) / 8) ** 2 / 7 + (d0.var(ddof=1) / 8) ** 2 / 7); h = stats.t.ppf(.975, dfw) * se
            out.append(dict(kappa="DiD(1-0)", p=p, stratum=s, contrast=f"{a}-{b}", mean=m, lo=m - h, hi=m + h, n_neg=np.nan))
O = pd.DataFrame(out); O.to_csv("results/extra/phase5/analysis/posthoc_exploratory.csv", index=False)
pd.set_option("display.width", 200); pd.set_option("display.max_rows", 200)
print(O[O.stratum.isin(["all", "L0", "L2+"])].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
