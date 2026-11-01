"""P7-B POST-HOC verification and secondary summaries (written 2026-10-29, AFTER analyze7.py ran; NOT part of the frozen protocol).
Reads only the raw additive eval CSVs (results/extra/phase7/eval), the training rows and the frozen analysis outputs; never rewrites them.
Part A (verification): completeness of every eval cell, stratum-count consistency, an independent re-implementation of the primary
fixed-sequence contrast compared to analyze7's p7_fixed_sequence.csv.
Part B (pre-registered descriptive items analyze7 did not tabulate): coordinate table, Bfull - raw_joint, stability, mask / oracle strata,
p difference (paired over replicates: the same trajectories are used for p = 0.3 and 0.7).
Part C (EXPLORATORY, labelled as such): Phase 6 G3 criteria applied to Bfull; variance decomposition (replicate vs test trajectory vs
mask); exact sign-flip test; imputation (best_val) vs decodability correlation.
Writes results/extra/phase7/verify/*.csv and verify_summary.txt.
usage: PYTHONPATH=.pydeps:. python experiments/phase7/verify7.py"""
import glob, itertools, os
import numpy as np, pandas as pd
from scipy import stats

E = "results/extra/phase7/eval"; A = "results/extra/phase7/analysis"; O = "results/extra/phase7/verify"
DIM = {"s6": 6, "s1": 3, "s2": 3, "y1": 1, "y2": 1}; FEATS = ["Bself1", "Bself2", "Bfull", "raw_joint"]
CELLS = [(k, p, r) for k in (0.0, 1.0) for p in (0.3, 0.7) for r in range(24)]; NT, NM, NPT = 8, 5, 5000 - 7
OUT = []


def say(s=""):
    OUT.append(str(s)); print(s, flush=True)


def tci(d):
    d = np.asarray(d, float); n = len(d); m, sd = d.mean(), d.std(ddof=1); h = stats.t.ppf(.975, n - 1) * sd / np.sqrt(n)
    return dict(n=n, mean=m, sd=sd, lo=m - h, hi=m + h, n_neg=int((d < 0).sum()))


def main():
    os.makedirs(O, exist_ok=True)
    acc = pd.concat([pd.read_csv(f"{E}/k{k}_p{p}_r{r}_acc.csv") for k, p, r in CELLS], ignore_index=True)
    pair = pd.concat([pd.read_csv(f"{E}/k{k}_p{p}_r{r}_pair.csv") for k, p, r in CELLS], ignore_index=True)
    # ---------- A1 completeness / consistency
    say("== A1 completeness")
    ok = True
    for (k, p, r), g in acc.groupby(["kappa", "p", "rep"]):
        al = g[(g.stratum == "all") & (g.coord == "s6")]
        if sorted(al.feature.unique()) != sorted(FEATS) or len(al) != 4 * NT * NM or not (al.n == NPT).all():
            ok = False; say(f"incomplete {k} {p} {r}")
    say(f"cells {acc.groupby(['kappa','p','rep']).ngroups}/96, rows acc {len(acc)} pair {len(pair)}, all-stratum n == {NPT} everywhere: {ok}")
    say(f"non-finite sse: acc {int((~np.isfinite(acc.sse)).sum())}, pair {int((~np.isfinite(pair.sse)).sum())}")
    s6 = acc[acc.coord == "s6"].pivot_table(index=["kappa", "p", "rep", "feature", "traj", "mask"], columns="stratum", values="n")
    c1 = (s6[["both_L2+", "one_L2+", "none_L2+"]].sum(1) == s6["all"]).all()
    c2 = all((s6[[f"L{j}:L0", f"L{j}:L1", f"L{j}:L2+"]].sum(1) == s6["all"]).all() for j in (1, 2))
    orc = s6[[c for c in s6.columns if c.startswith("oracle:")]].sum(1)
    say(f"joint strata partition all: {c1}; L_j strata partition all: {c2}; oracle points per (traj, mask): {sorted(orc.unique())}")
    # ---------- A2 independent recompute of the primary contrast
    say("\n== A2 primary contrast, independent recompute vs analyze7")
    pr = acc[(acc.stratum == "all") & (acc.coord == "s6")].groupby(["kappa", "p", "rep", "feature"])[["sse", "n"]].sum()
    Q = np.sqrt(pr.sse / (pr.n * 6)).unstack("feature"); Q["Bself"] = (Q.Bself1 + Q.Bself2) / 2
    fs = pd.read_csv(f"{A}/p7_fixed_sequence.csv"); rows = []
    for _, s in fs.iterrows():
        d = (Q.loc[(s.kappa, s.p)].Bfull - Q.loc[(s.kappa, s.p)].Bself).values; t = tci(d)
        sign_p = 2 * stats.binom.cdf(min(t["n_neg"], 24 - t["n_neg"]), 24, .5)
        rows.append(dict(step=s.step, kappa=s.kappa, p=s.p, **t, abs_diff_mean_vs_analyze7=abs(t["mean"] - s["mean"]),
                         abs_diff_hi_vs_analyze7=abs(t["hi"] - s.ci95_hi), exact_sign_test_p=min(1.0, sign_p),
                         Q_Bfull=Q.loc[(s.kappa, s.p)].Bfull.mean(), Q_Bself=Q.loc[(s.kappa, s.p)].Bself.mean(),
                         Q_raw=Q.loc[(s.kappa, s.p)].raw_joint.mean()))
    R = pd.DataFrame(rows); R.to_csv(f"{O}/v_primary_recompute.csv", index=False); say(R.round(6).to_string(index=False))
    # ---------- B1 coordinate table (stratum all)
    say("\n== B1 RMSE by coordinate, stratum all (replicate means)")
    g = acc[acc.stratum == "all"].groupby(["kappa", "p", "rep", "feature", "coord"])[["sse", "n"]].sum().reset_index()
    g["rmse"] = np.sqrt(g.sse / (g.n * g.coord.map(DIM)))
    T = g.groupby(["kappa", "p", "feature", "coord"]).rmse.mean().unstack("coord")[["s6", "s1", "s2", "y1", "y2"]]
    T.to_csv(f"{O}/v_coord_table.csv"); say(T.round(4).to_string())
    # ---------- B2 Bfull - raw_joint and Bself - raw_joint, s6 all (from analyze7 contrasts, re-read)
    C = pd.read_csv(f"{A}/p7_contrasts.csv")
    say("\n== B2 contrasts vs raw_joint (s6, all) [reference only; raw superiority is not a necessary condition, DPI]")
    x = C[(C.stratum == "all") & (C.coord == "s6") & (C.contrast != "Bfull - Bself")][["kappa", "p", "contrast", "mean_a", "mean_b", "mean", "ci95_lo", "ci95_hi", "n_negative"]]
    x.to_csv(f"{O}/v_raw_contrasts.csv", index=False); say(x.round(4).to_string(index=False))
    # ---------- B3 stability
    St = pd.read_csv(f"{A}/p7_stability.csv")
    say("\n== B3 mask stability SD (s6, y1, y2; stratum all and both_L2+)")
    x = St[St.stratum.isin(["all", "both_L2+"]) & St.coord.isin(["s6", "y1", "y2"])][["kappa", "p", "stratum", "coord", "Bfull", "Bself", "raw_joint", "mean", "ci95_lo", "ci95_hi"]]
    x.to_csv(f"{O}/v_stability.csv", index=False); say(x.round(4).to_string(index=False))
    # ---------- B4 strata (s6) Bfull - Bself, Bfull - raw
    say("\n== B4 strata (s6): Bfull - Bself and Bfull - raw_joint")
    x = C[(C.coord == "s6") & (C.contrast.isin(["Bfull - Bself", "Bfull - raw_joint"]))][["kappa", "p", "stratum", "contrast", "mean_a", "mean_b", "mean", "ci95_lo", "ci95_hi", "n"]]
    x.to_csv(f"{O}/v_strata.csv", index=False); say(x[x.p == 0.7].round(4).to_string(index=False))
    # ---------- B5 p difference (paired over replicates; same trajectories for both p)
    say("\n== B5 p difference of the primary contrast, paired over replicates (same trajectories, different masks)")
    pdf = []
    for k in (0.0, 1.0):
        d7 = (Q.loc[(k, 0.7)].Bfull - Q.loc[(k, 0.7)].Bself); d3 = (Q.loc[(k, 0.3)].Bfull - Q.loc[(k, 0.3)].Bself)
        pdf.append(dict(kappa=k, contrast="Delta(p=0.7) - Delta(p=0.3), paired", **tci((d7 - d3).values)))
    pdf = pd.DataFrame(pdf); pdf.to_csv(f"{O}/v_p_difference.csv", index=False); say(pdf.round(4).to_string(index=False))
    # ---------- C1 Phase 6 G3 criteria applied post hoc to Bfull
    say("\n== C1 EXPLORATORY: Phase 6 G3 criteria (RMSE_s6 <= 0.2, stab SD_s6 <= 0.1, Bfull - raw CI hi < 0) applied to Bfull")
    acc_r = pd.read_csv(f"{A}/p7_accuracy_per_rep.csv"); stb_r = pd.read_csv(f"{A}/p7_stability_per_rep.csv")
    cr = []
    for (k, p, st), y in acc_r[(acc_r.feature == "Bfull") & (acc_r.coord == "s6")].groupby(["kappa", "p", "stratum"]):
        rm = y.rmse.mean(); sd = stb_r[(stb_r.feature == "Bfull") & (stb_r.coord == "s6") & (stb_r.kappa == k) & (stb_r.p == p) & (stb_r.stratum == st)].stab_sd.mean()
        cc = C[(C.kappa == k) & (C.p == p) & (C.stratum == st) & (C.coord == "s6") & (C.contrast == "Bfull - raw_joint")]
        hi = cc.ci95_hi.iloc[0] if len(cc) else np.nan
        cr.append(dict(kappa=k, p=p, stratum=st, rmse_s6=rm, stab_sd_s6=sd, raw_ci_hi=hi, a=rm <= .2, b=(sd <= .1) if np.isfinite(sd) else None,
                       c=bool(hi < 0) if np.isfinite(hi) else None))
    cr = pd.DataFrame(cr); cr["all3"] = cr.a & (cr.b == True) & (cr.c == True); cr.to_csv(f"{O}/v_g3_criteria_bfull.csv", index=False)
    say(cr.round(4).to_string(index=False))
    # ---------- C2 variance decomposition of the primary contrast
    say("\n== C2 EXPLORATORY: variance components of Delta (s6, all): replicate, test trajectory within replicate, mask within trajectory")
    t = acc[(acc.stratum == "all") & (acc.coord == "s6")].copy(); t["mse"] = t.sse / (t.n * 6)
    W = t.pivot_table(index=["kappa", "p", "rep", "traj", "mask"], columns="feature", values="mse")
    W["D"] = np.sqrt(W.Bfull) - (np.sqrt(W.Bself1) + np.sqrt(W.Bself2)) / 2
    vc = []
    for (k, p), y in W.groupby(level=[0, 1]):
        tr = y.D.groupby(level=[2, 3]).mean(); rp = tr.groupby(level=0).mean()
        vc.append(dict(kappa=k, p=p, sd_between_rep=rp.std(ddof=1), sd_traj_within_rep=(tr - tr.groupby(level=0).transform("mean")).std(ddof=1),
                       sd_mask_within_traj=(y.D - y.D.groupby(level=[2, 3]).transform("mean")).std(ddof=1)))
    vc = pd.DataFrame(vc); vc.to_csv(f"{O}/v_variance_components.csv", index=False); say(vc.round(5).to_string(index=False))
    # ---------- C3 imputation (val held-out observed RMSE) vs decodability
    say("\n== C3 EXPLORATORY: Spearman(best_val, Q_r s6) across replicates; best_val is on loss-active columns only (not comparable across arms)")
    tr_ = pd.read_csv(f"{A}/p7_training_rows.csv"); cc = []
    for (k, p, a), y in tr_.groupby(["kappa", "p", "arm"]):
        q = Q.loc[(k, p)][a].reindex(y.rep.values).values; rho, pv = stats.spearmanr(y.best_val.values, q)
        cc.append(dict(kappa=k, p=p, arm=a, spearman=rho, p_value=pv, mean_best_val=y.best_val.mean(), mean_Q_s6=q.mean()))
    cc = pd.DataFrame(cc); cc.to_csv(f"{O}/v_imputation_vs_decodability.csv", index=False); say(cc.round(4).to_string(index=False))
    # ---------- C4 training length
    say("\n== C4 epochs_run Bfull - mean(Bself), paired over replicates")
    e = tr_.pivot_table(index=["kappa", "p", "rep"], columns="arm", values="epochs_run"); e["d"] = e.Bfull - (e.Bself1 + e.Bself2) / 2
    ep = pd.DataFrame([dict(kappa=k, p=p, **tci(y.d.values)) for (k, p), y in e.groupby(level=[0, 1])]); ep.to_csv(f"{O}/v_epochs.csv", index=False)
    say(ep.round(2).to_string(index=False))
    open(f"{O}/verify_summary.txt", "w").write("\n".join(OUT) + "\n")


if __name__ == "__main__":
    main()
