#!/usr/bin/env python
"""STANDALONE recomputation of every Phase 8 confirmatory statistic from the replicate-level tables in this bundle
(numpy + pandas + scipy only; no server, no project code). Definitions: protocol v2 (phase8_mathematical_protocol_v2.md).
Inputs (bundle_inputs/):
  rep_acc.csv.gz : kappa, p, rep, pipeline, stratum, coord, sse, n   (SSE summed over the 8 test trajectories x 5 masks of
                   replicate rep; n = number of (time, trajectory, mask) points in the stratum; identical for every coord)
  rep_imp.csv.gz : kappa, p, rep, method, view, stratum, sse, n      (missing positions only)
  rep_lat.csv.gz : evaluate8 latent diagnostics (per test trajectory)
Unit of inference = replicate (n = 24 per (kappa, p)). Coordinates: 0..5 = x1 y1 z1 x2 y2 z2 (normalised by FIXED constants).
usage: python recompute_from_bundle.py [bundle_inputs_dir]   -> prints the confirmatory table and writes recomputed_*.csv"""
import math, sys
import numpy as np, pandas as pd
from scipy import stats

D = sys.argv[1] if len(sys.argv) > 1 else "bundle_inputs"
DELTA_I, DELTA_Y, EPS = 0.02, 0.02, 0.01; YI = {1: 1, 2: 4}
acc = pd.read_csv(f"{D}/rep_acc.csv.gz"); S = acc.pivot_table(index=["kappa", "p", "rep", "pipeline", "stratum"], columns="coord", values="sse")
N = acc[acc.coord == 0].set_index(["kappa", "p", "rep", "pipeline", "stratum"]).n


def x(k, p, pi, st): return S.xs((k, p, pi, st), level=("kappa", "p", "pipeline", "stratum")).sort_index()
def n(k, p, pi, st): return N.xs((k, p, pi, st), level=("kappa", "p", "pipeline", "stratum")).sort_index()
def R(k, p, pi, st): return (x(k, p, pi, st).sum(1) / (6 * n(k, p, pi, st))).values                       # coordinate-mean MSE
def rm(k, p, pi, st, c): return np.sqrt((x(k, p, pi, st)[c] / n(k, p, pi, st)).values)                     # pooled RMSE of coord c
def C(k, p, pi, j):                                                                                        # P8.4 certificate on L_j0
    st = f"L{j}:L0"; return ((x(k, p, "sym0", st)[YI[j]] - x(k, p, pi, st)[YI[j]]) / n(k, p, pi, st)).values


def tb(d, a):
    d = np.asarray(d, float); m = d.mean(); se = d.std(ddof=1) / math.sqrt(len(d)); q = stats.t.ppf(1 - a, len(d) - 1); return m, m - q * se, m + q * se


def sign2(d):
    d = np.asarray(d); d = d[d != 0]; k = int((d > 0).sum()); n_ = len(d)
    cdf = lambda t: sum(math.comb(n_, i) for i in range(t + 1)) / 2 ** n_
    return min(1.0, 2 * min(cdf(k), 1 - cdf(k - 1) if k > 0 else 1.0))


out = []
# Family I (fixed sequence; one-sided alpha 0.025; rule LCB > delta_I)
go = True
for step, k, a, b in (("I-1", 1.0, "R1:Bfull", "O_raw"), ("I-2", 1.0, "O_lat:Bfull", "O_raw"), ("I-3", 0.0, "R1:Bfull", "O_raw"), ("I-4", 0.0, "O_lat:Bfull", "O_raw")):
    m, lo, hi = tb(R(k, .7, a, "H") - R(k, .7, b, "H"), .025); conf = go and lo > DELTA_I
    out.append(dict(step=step, contrast=f"R_H({a})-R_H({b}) k={k}", mean=m, lcb=lo, ucb=hi, tested=go, decision=conf)); go = conf
# kappa = 0 gates (per-j exact sign, two-sided 0.025 each; TOST 90% CI within +-eps)
CG = lambda F, j, k: rm(k, .7, f"R1:{F}", "all", YI[j]) - rm(k, .7, f"R1:{F}Sham{j}", "all", YI[j])
gate = {}
for fam, perj in (("IIa", {j: CG("Bfull", j, 0.0) for j in (1, 2)}), ("IIb", {j: CG("Bfull", j, 0.0) - CG("Fact8", j, 0.0) for j in (1, 2)})):
    avg = (perj[1] + perj[2]) / 2; m, lo, hi = tb(avg, .05); ps = [sign2(perj[j]) for j in (1, 2)]
    gate[fam] = "FAIL" if min(ps) < .025 else ("PASS" if (lo > -EPS and hi < EPS) else "UNDETERMINED")
    out.append(dict(step=f"gate {fam}", contrast="kappa=0 p=0.7", mean=m, lcb=lo, ucb=hi, tested=True, decision=gate[fam], sign_p=ps))
# IIa (alpha 0.0125): IIa-1 UCB < 0; IIa-2 LCB > 0
cg = (CG("Bfull", 1, 1.0) + CG("Bfull", 2, 1.0)) / 2; m, lo, hi = tb(cg, .0125); d1 = hi < 0
out.append(dict(step="IIa-1", contrast="CG_Bfull k=1", mean=m, lcb=lo, ucb=hi, tested=True, decision=bool(d1 and gate["IIa"] == "PASS"), practical=m <= -DELTA_Y))
c2 = (C(1.0, .7, "O_lat:Bfull", 1) + C(1.0, .7, "O_lat:Bfull", 2)) / 2; m, lo, hi = tb(c2, .0125)
out.append(dict(step="IIa-2", contrast="C_Bfull k=1 (O_lat readout, L_j0)", mean=m, lcb=lo, ucb=hi, tested=bool(d1), decision=bool(d1 and lo > 0 and gate["IIa"] == "PASS")))
# IIb (alpha 0.0125): IIb-1 UCB < 0; IIb-2 only if IIb-1 detected
dj = ((CG("Bfull", 1, 1.0) - CG("Fact8", 1, 1.0)) + (CG("Bfull", 2, 1.0) - CG("Fact8", 2, 1.0))) / 2; m, lo, hi = tb(dj, .0125); e1 = hi < 0
out.append(dict(step="IIb-1", contrast="Delta_JF k=1", mean=m, lcb=lo, ucb=hi, tested=True, decision=bool(e1 and gate["IIb"] == "PASS")))
s6 = np.sqrt(R(1.0, .7, "R1:Bfull", "all")) - np.sqrt(R(1.0, .7, "R1:Fact8", "all")); m, lo, hi = tb(s6, .0125)
out.append(dict(step="IIb-2", contrast="RMSE_s6(Bfull)-RMSE_s6(Fact8) k=1", mean=m, lcb=lo, ucb=hi, tested=bool(e1), decision=bool(e1 and hi < 0)))
T = pd.DataFrame(out); pd.set_option("display.width", 200); print(T.to_string(index=False)); T.to_csv("recomputed_confirmatory.csv", index=False)
# imputation summary (descriptive): pooled over the 4 views, mean over replicates
imp = pd.read_csv(f"{D}/rep_imp.csv.gz"); a = imp[imp.stratum == "all"].groupby(["kappa", "p", "rep", "method"])[["sse", "n"]].sum()
a["rmse"] = np.sqrt(a.sse / a.n); print(a.groupby(["kappa", "p", "method"]).rmse.mean().unstack("method").round(4).to_string())
