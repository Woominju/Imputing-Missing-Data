#!/usr/bin/env python
"""POST-HOC sensitivity (NOT preregistered; never replaces a confirmatory result): recompute the p = 0.7 confirmatory contrasts
(I-1, I-2, IIa-1, IIa-2, IIb-1, IIb-2 descriptive) (a) on all 24 replicates, (b) excluding replicates whose 9 AE arms straddle the
NVIDIA driver change (580.105.08 -> 580.196.07), (c) on replicates whose arms all ran on the old / all on the new driver.
No project imports. usage: python -I posthoc_driver_sensitivity.py --root <root> --out <dir>"""
import argparse, glob, math, os
import numpy as np, pandas as pd
from scipy import stats

ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--out", required=True); A = ap.parse_args(); os.chdir(A.root)
E = pd.read_csv(f"{A.out}/training_environment.csv"); dr = E.groupby(["kappa", "p", "rep"]).driver.agg(lambda s: s.iloc[0] if s.nunique() == 1 else "mixed")
ACC = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("results/extra/phase8/eval/k*_p0.7_*_acc.csv.gz"))])
g = ACC.groupby(["kappa", "rep", "pipeline", "stratum", "coord"])[["sse", "n"]].sum(); S = g.sse.unstack("coord"); N = g.n.unstack("coord")[0]
risk = lambda k, pi, st: (S.xs((k, pi, st), level=("kappa", "pipeline", "stratum")).sum(1) / (6 * N.xs((k, pi, st), level=("kappa", "pipeline", "stratum"))))
rm = lambda k, pi, st, c: np.sqrt(S.xs((k, pi, st), level=("kappa", "pipeline", "stratum"))[c] / N.xs((k, pi, st), level=("kappa", "pipeline", "stratum")))
cert = lambda k, pi, j: (S.xs((k, "sym0", f"L{j}:L0"), level=("kappa", "pipeline", "stratum"))[3 * j - 2] - S.xs((k, pi, f"L{j}:L0"), level=("kappa", "pipeline", "stratum"))[3 * j - 2]) / N.xs((k, pi, f"L{j}:L0"), level=("kappa", "pipeline", "stratum"))
k = 1.0
C = {"I-1 G_tot(H)": (risk(k, "R1:Bfull", "H") - risk(k, "O_raw", "H"), .025, "lcb>0.02"),
     "I-2 G_lat(H)": (risk(k, "O_lat:Bfull", "H") - risk(k, "O_raw", "H"), .025, "lcb>0.02"),
     "IIa-1 CG_Bfull": (sum(rm(k, "R1:Bfull", "all", 3 * j - 2) - rm(k, f"R1:BfullSham{j}", "all", 3 * j - 2) for j in (1, 2)) / 2, .0125, "ucb<0"),
     "IIa-2 C_Bfull": (sum(cert(k, "O_lat:Bfull", j) for j in (1, 2)) / 2, .0125, "lcb>0"),
     "IIb-1 Delta_JF": (sum((rm(k, "R1:Bfull", "all", 3 * j - 2) - rm(k, f"R1:BfullSham{j}", "all", 3 * j - 2)) - (rm(k, "R1:Fact8", "all", 3 * j - 2) - rm(k, f"R1:Fact8Sham{j}", "all", 3 * j - 2)) for j in (1, 2)) / 2, .0125, "ucb<0"),
     "IIb-2 D_s6 (descr.)": (np.sqrt(risk(k, "R1:Bfull", "all")) - np.sqrt(risk(k, "R1:Fact8", "all")), .0125, "ucb<0")}
rows = []
for nm, (d, a, rule) in C.items():
    for sub, keep in (("all 24", lambda r: True), ("exclude mixed-driver reps", lambda r: dr[(k, .7, r)] != "mixed"),
                      ("old driver only", lambda r: dr[(k, .7, r)] == "580.105.08"), ("new driver only", lambda r: dr[(k, .7, r)] == "580.196.07")):
        x = d[[keep(r) for r in d.index]].values; n = len(x)
        if n < 3:
            continue
        m = x.mean(); se = x.std(ddof=1) / math.sqrt(n); q = stats.t.ppf(1 - a, n - 1); lo, hi = m - q * se, m + q * se
        dec = {"lcb>0.02": lo > .02, "lcb>0": lo > 0, "ucb<0": hi < 0}[rule]
        rows.append(dict(contrast=nm, subset=sub, n=n, mean=m, lcb=lo, ucb=hi, rule=rule, decision=bool(dec)))
R = pd.DataFrame(rows); R.to_csv(f"{A.out}/posthoc_driver_sensitivity.csv", index=False); print(R.round(5).to_string(index=False))
print("driver of AE arms per replicate (kappa = 1, p = 0.7):", dr.loc[(1.0, .7)].value_counts().to_dict())
