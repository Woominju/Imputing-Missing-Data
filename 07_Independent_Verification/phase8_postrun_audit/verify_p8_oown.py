#!/usr/bin/env python
"""Protocol 5.5 supplementary item 4 (descriptive; analyze8 does not output it): oracle-level cross information
y_j RMSE(O_own_j) - y_j RMSE(O_raw), and the same for the hidden coordinate on the own-L strata. Both are fixed pool oracles
(conditional on the pool); the difference is a difference of two learned functions, NOT a Bayes-risk difference (B3).
No project imports. usage: python -I verify_p8_oown.py --root <root> --out <dir>"""
import argparse, glob, math, os
import numpy as np, pandas as pd
from scipy import stats
ap = argparse.ArgumentParser(); ap.add_argument("--root"); ap.add_argument("--out"); A = ap.parse_args(); os.chdir(A.root)
ACC = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("results/extra/phase8/eval/k*_acc.csv.gz"))])
g = ACC[ACC.pipeline.isin(["O_raw", "O_own1", "O_own2"])].groupby(["kappa", "p", "rep", "pipeline", "stratum", "coord"])[["sse", "n"]].sum().reset_index()
g["rmse"] = np.sqrt(g.sse / g.n.replace(0, np.nan)); rows = []
for (k, p), y in g.groupby(["kappa", "p"]):
    for j in (1, 2):
        c = 3 * j - 2
        for st in ("all", f"L{j}:L0", f"L{j}:L1", f"L{j}:L2+"):
            a = y[(y.pipeline == f"O_own{j}") & (y.stratum == st) & (y.coord == c)].set_index("rep").rmse.sort_index()
            b = y[(y.pipeline == "O_raw") & (y.stratum == st) & (y.coord == c)].set_index("rep").rmse.sort_index()
            d = (a - b).dropna().values; n = len(d)
            if n < 3: continue
            m = d.mean(); h = stats.t.ppf(.975, n - 1) * d.std(ddof=1) / math.sqrt(n)
            rows.append(dict(kappa=k, p=p, j=j, stratum=st, rmse_y_Oown=a.mean(), rmse_y_Oraw=b.mean(), diff=m, lo=m - h, hi=m + h, n=n))
R = pd.DataFrame(rows); R.to_csv(f"{A.out}/supp_oracle_cross_information.csv", index=False); print(R.round(4).to_string(index=False))
