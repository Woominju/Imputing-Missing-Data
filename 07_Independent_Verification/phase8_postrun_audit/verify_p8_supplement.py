#!/usr/bin/env python
"""Supplementary independent computations for the Phase 8 post-run audit (descriptive only; nothing here is confirmatory).
  S1 evaluation-law identity: every pipeline is scored on identical (traj, mask, stratum) point sets (n equal across pipelines)
  S2 latent-NN state distance: replicate-paired Bfull - Fact8 / Fact16 (95% two-sided t), per (kappa, p)
  S3 Family III: AE:Fact8 - AE:Bfull per view x own-L stratum at BOTH kappa (is the B-full L_j0 x-view advantage coupling-specific?)
  S4 Family I share of G_tot in R_H(B-full); G_lat / G_rd split (descriptive)
  S5 readout-level vs latent-level mask sensitivity (pair files vs rho_phi)
No project imports; reads results/extra/phase8/eval only. usage: python -I verify_p8_supplement.py --root <root> --out <dir>"""
import argparse, glob, math, os
import numpy as np, pandas as pd
from scipy import stats

ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--out", required=True); A = ap.parse_args()
os.chdir(A.root); OUT = A.out; EV = "results/extra/phase8/eval"; LOG = open(f"{OUT}/supplement_log.txt", "w")


def say(*s):
    t = " ".join(str(x) for x in s); print(t, flush=True); LOG.write(t + "\n")


def tb(d, a=.025):
    d = np.asarray(d, float); n = len(d); m = d.mean(); se = d.std(ddof=1) / math.sqrt(n); q = stats.t.ppf(1 - a, n - 1)
    return dict(n=n, mean=m, lo=m - q * se, hi=m + q * se, n_neg=int((d < 0).sum()))


ACC = pd.concat([pd.read_csv(f, usecols=["pipeline", "traj", "mask", "stratum", "coord", "n", "kappa", "p", "rep"]) for f in sorted(glob.glob(f"{EV}/k*_acc.csv.gz"))])
s1 = ACC[ACC.coord == 0].groupby(["kappa", "p", "rep", "traj", "mask", "stratum"]).n.nunique()
say(f"S1 identical point sets across all 18 pipelines for every (cell, rep, traj, mask, stratum): {bool((s1 == 1).all())} ({len(s1)} groups)")
LAT = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{EV}/k*_lat.csv.gz"))]).dropna(subset=["nn_state_dist"])
L = LAT.groupby(["kappa", "p", "rep", "feat"]).nn_state_dist.mean().unstack("feat")
rows = []
for (k, p), y in L.groupby(level=["kappa", "p"]):
    for o in ("Fact8", "Fact16"):
        rows.append(dict(kappa=k, p=p, contrast=f"nnStateDist Bfull - {o}", **tb((y.Bfull - y[o]).values)))
S2 = pd.DataFrame(rows); S2.to_csv(f"{OUT}/supp_latent_nn_paired.csv", index=False); say("S2 latent-NN state distance (lower = better), paired over replicates:\n" + S2.round(4).to_string(index=False))
IMP = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{EV}/k*_imp.csv.gz"))])
t = IMP.groupby(["kappa", "p", "rep", "method", "view", "stratum"])[["sse", "n"]].sum().reset_index(); t["rmse"] = np.sqrt(t.sse / t.n.replace(0, np.nan))
rows = []
for (k, p, v, st), y in t[t.stratum.str.startswith("L")].groupby(["kappa", "p", "view", "stratum"]):
    a = y[y.method == "AE:Fact8"].set_index("rep").rmse.sort_index(); b = y[y.method == "AE:Bfull"].set_index("rep").rmse.sort_index()
    if len(a) == 24 and a.notna().all() and b.notna().all():
        rows.append(dict(kappa=k, p=p, view=v, stratum=st, fact8=a.mean(), bfull=b.mean(), **tb((a - b).values)))
S3 = pd.DataFrame(rows); S3.to_csv(f"{OUT}/supp_familyIII_fact8_minus_bfull_by_L.csv", index=False)
say("S3 AE:Fact8 - AE:Bfull imputation RMSE by view x own-L stratum (positive = B-full better):\n" + S3[S3.p == 0.7].round(4).to_string(index=False))
FI = pd.read_csv(f"{OUT}/indep_familyI_all.csv")
h = FI[(FI.stratum == "H")].pivot_table(index="kappa", columns="contrast", values="mean"); h["R_Bfull_H"] = FI[FI.stratum == "H"].groupby("kappa").R_Bfull.first()
h["share_Gtot_of_R"] = h.G_tot / h.R_Bfull_H; h["share_Glat_of_Gtot"] = h.G_lat / h.G_tot; h["share_Grd_of_Gtot"] = h.G_rd / h.G_tot
h.to_csv(f"{OUT}/supp_familyI_shares.csv"); say("S4 Family I H-stratum shares (descriptive):\n" + h.round(4).to_string())
dec = pd.read_csv(f"{OUT}/indep_mask_decomposition.csv"); rho = pd.read_csv(f"{OUT}/indep_familyIV_rho.csv")
say("S5 readout-level mask variance (R1:Bfull vs R1:Fact8) vs latent rho_phi (Bfull vs Fact8), p = 0.7:")
for k in (0.0, 1.0):
    d = dec[(dec.kappa == k) & (dec.p == 0.7)].set_index("pipeline"); r = rho[(rho.kappa == k) & (rho.p == 0.7) & (rho.contrast == "rho_Bfull_minus_Fact8")].iloc[0]
    say(f"   kappa={k}: mask_var R1:Bfull {d.loc['R1:Bfull', 'mask_var']:.5f}  R1:Fact8 {d.loc['R1:Fact8', 'mask_var']:.5f} | rho_phi Bfull {r.rho_Bfull:.4f} Fact8 {r.rho_other:.4f}")
PAIR = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{EV}/k*_pair.csv.gz"))])
pv = PAIR.groupby(["kappa", "p", "rep", "pipeline"]).apply(lambda y: (y.sse.sum() / (y.n.iloc[0] * 6 * y.traj.nunique() * y.pair.nunique())) / 2).unstack("pipeline")
rows = [dict(kappa=k, p=p, contrast="mask_var R1:Bfull - R1:Fact8", **tb((y["R1:Bfull"] - y["R1:Fact8"]).values)) for (k, p), y in pv.groupby(level=["kappa", "p"])]
S5 = pd.DataFrame(rows); S5.to_csv(f"{OUT}/supp_readout_mask_var_paired.csv", index=False); say(S5.round(5).to_string(index=False))
