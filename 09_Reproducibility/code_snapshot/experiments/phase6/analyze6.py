"""Phase 6 G2/G3 analysis (CPU), exactly as fixed in results/notes/phase6_g2_g3_protocol.md sections 5-6.
G2 unit = independent replicate r (24 per kappa, p). Q_r = sqrt(mean_j MSE_j), MSE_j pooled over the replicate's 8 test
trajectories, averaged over MLP probe seeds {0, 1}. Paired t (two-sided 95%), Wilcoxon, sign test; delta = 0.02 reported
separately (direction / magnitude / equivalence never merged into one label). Fixed sequence S1 -> S2 -> S3.
G3: shared 6-D readout; per replicate RMSE pooled over test trajectories and masks; stability = RMS pairwise difference / sqrt(2).
Writes results/extra/phase6/analysis/*.csv"""
import glob, os
import numpy as np, pandas as pd
from scipy import stats

E = "results/extra/phase6/eval"; O = "results/extra/phase6/analysis"; DELTA = 0.02
GEN = {"A": "A{j}", "Aslot": "Aslot{j}", "B": "B{j}", "Bcur": "Bcur{j}", "Sham": "Sham{j}", "raw_own": "raw_own{j}",
       "raw_joint": "raw_joint", "raw_owncur": "raw_owncur{j}"}
CONTRASTS = [("B", "Sham"), ("B", "A"), ("Aslot", "A"), ("B", "Aslot"), ("Sham", "Aslot"), ("Sham", "A"), ("B", "Bcur"),
             ("Bcur", "Aslot"), ("Bcur", "Sham"), ("raw_joint", "raw_own"), ("raw_owncur", "raw_own"), ("B", "raw_joint"), ("A", "raw_own")]


def load(kind):
    fs = sorted(glob.glob(f"{E}/k*_{kind}.csv"))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


def generic(df):
    rows = []
    for g, pat in GEN.items():
        for j in (1, 2):
            s = df[(df.feature == pat.format(j=j)) & (df.j == j)].copy(); s["g"] = g; rows.append(s)
    return pd.concat(rows, ignore_index=True)


def paired(d):
    d = np.asarray(d, float); n = len(d); m, sd = d.mean(), d.std(ddof=1); se = sd / np.sqrt(n)
    t95, t90 = stats.t.ppf(.975, n - 1), stats.t.ppf(.95, n - 1)
    try:
        wp = stats.wilcoxon(d).pvalue
    except ValueError:
        wp = np.nan
    return dict(n=n, mean=m, sd=sd, ci95_lo=m - t95 * se, ci95_hi=m + t95 * se, ci90_lo=m - t90 * se, ci90_hi=m + t90 * se,
                p_t=2 * stats.t.sf(abs(m / se), n - 1) if se > 0 else np.nan, d_z=m / sd if sd > 0 else np.nan,
                n_negative=int((d < 0).sum()), wilcoxon_p=wp, sign_p=stats.binomtest(int((d < 0).sum()), n, .5).pvalue,
                detected_neg=bool(m + t95 * se < 0), detected_pos=bool(m - t95 * se > 0), mean_le_minus_delta=bool(m <= -DELTA),
                ci_hi_lt_minus_delta=bool(m + t95 * se < -DELTA), tost_equiv=bool(m - t90 * se > -DELTA and m + t90 * se < DELTA))


def p5_label(r):
    """Phase 5 decide() label, reported only for continuity with the original H2 (B - A)."""
    if r["ci95_hi"] < 0 and r["mean"] <= -DELTA: return "SUPPORTED (improvement >= delta)"
    if r["ci95_hi"] < 0: return "detectable but < delta"
    if r["ci90_lo"] > -DELTA and r["ci90_hi"] < DELTA: return "EQUIVALENT to zero (TOST, delta)"
    if r["ci95_lo"] > 0: return "OPPOSITE (worse)"
    return "inconclusive"


def g2():
    P = load("g2"); P = generic(P); P = P[P.probe.isin(["mlp0", "mlp1"])]
    mse = P.groupby(["kappa", "p", "stratum", "g", "rep", "j", "probe"]).apply(lambda x: x.sse.sum() / max(x.n.sum(), 1)).groupby(
        ["kappa", "p", "stratum", "g", "rep", "j"]).mean()
    Q = np.sqrt(mse.groupby(["kappa", "p", "stratum", "g", "rep"]).mean()).unstack("g")
    Q.reset_index().to_csv(f"{O}/g2_replicate_metric.csv", index=False)
    lev = Q.groupby(level=[0, 1, 2]).agg(["mean", "std", "count"]); lev.columns = [f"{a}_{b}" for a, b in lev.columns]; lev.reset_index().to_csv(f"{O}/g2_levels.csv", index=False)
    rows = []
    for (k, p, s), x in Q.groupby(level=[0, 1, 2]):
        for a, b in CONTRASTS:
            if a in x and b in x:
                d = (x[a] - x[b]).dropna()
                if len(d) >= 3:
                    r = dict(kappa=k, p=p, stratum=s, contrast=f"{a} - {b}", mean_a=x[a].mean(), mean_b=x[b].mean(), **paired(d.values))
                    if (a, b) == ("B", "A"):
                        r["phase5_label"] = p5_label(r)
                    rows.append(r)
    C = pd.DataFrame(rows); C.to_csv(f"{O}/g2_contrasts.csv", index=False)
    # DiD (independent replicates across kappa): Welch
    dd = []
    for p in (0.3, 0.7):
        for s in ("all", "L0", "L1", "L2+", "c0", "c1"):
            for a, b in (("B", "Sham"), ("B", "A"), ("B", "Aslot")):
                try:
                    d1 = (Q.loc[(1.0, p, s)][a] - Q.loc[(1.0, p, s)][b]).dropna().values; d0 = (Q.loc[(0.0, p, s)][a] - Q.loc[(0.0, p, s)][b]).dropna().values
                except KeyError:
                    continue
                v1, v0 = d1.var(ddof=1) / len(d1), d0.var(ddof=1) / len(d0); se = np.sqrt(v1 + v0)
                dfw = se ** 4 / (v1 ** 2 / (len(d1) - 1) + v0 ** 2 / (len(d0) - 1)); m = d1.mean() - d0.mean(); h = stats.t.ppf(.975, dfw) * se
                dd.append(dict(p=p, stratum=s, contrast=f"DiD k1-k0 ({a} - {b})", mean=m, ci95_lo=m - h, ci95_hi=m + h, df=dfw, n1=len(d1), n0=len(d0)))
    DD = pd.DataFrame(dd); DD.to_csv(f"{O}/g2_did.csv", index=False)
    # fixed-sequence family
    def get(k, p, c, s="all"):
        r = C[(C.kappa == k) & (C.p == p) & (C.contrast == c) & (C.stratum == s)]; return r.iloc[0] if len(r) else None
    S1 = get(1.0, 0.7, "B - Sham"); S2 = DD[(DD.p == 0.7) & (DD.stratum == "all") & (DD.contrast == "DiD k1-k0 (B - Sham)")]
    S3 = get(1.0, 0.3, "B - Sham"); seq = []
    ok = S1 is not None and S1.detected_neg
    seq.append(dict(step="S1 B-Sham k1 p0.7", mean=S1["mean"] if S1 is not None else np.nan, ci95_lo=S1.ci95_lo if S1 is not None else np.nan,
                    ci95_hi=S1.ci95_hi if S1 is not None else np.nan, detected=ok, tested=True))
    s2d = bool(len(S2) and S2.iloc[0].ci95_hi < 0)
    seq.append(dict(step="S2 DiD(B-Sham) p0.7", mean=S2.iloc[0]["mean"] if len(S2) else np.nan, ci95_lo=S2.iloc[0].ci95_lo if len(S2) else np.nan,
                    ci95_hi=S2.iloc[0].ci95_hi if len(S2) else np.nan, detected=ok and s2d, tested=ok))
    ok2 = ok and s2d; s3d = bool(S3 is not None and S3.detected_neg)
    seq.append(dict(step="S3 B-Sham k1 p0.3", mean=S3["mean"] if S3 is not None else np.nan, ci95_lo=S3.ci95_lo if S3 is not None else np.nan,
                    ci95_hi=S3.ci95_hi if S3 is not None else np.nan, detected=ok2 and s3d, tested=ok2))
    pd.DataFrame(seq).to_csv(f"{O}/g2_fixed_sequence.csv", index=False)
    return C, DD, pd.DataFrame(seq), Q


def g3():
    H = load("g3"); HP = load("g3pair")
    if H.empty:
        return None
    dim = {"s6": 6, "s1": 3, "s2": 3, "y1": 1, "y2": 1}
    H = H[H["mask"] >= 0].copy() if "mask" in H else H
    acc = H[H.coord.isin(list(dim))].groupby(["kappa", "p", "rep", "feature", "stratum", "coord"])[["sse", "n"]].sum().reset_index()
    acc["rmse"] = np.sqrt(acc.sse / (np.maximum(acc.n, 1) * acc.coord.map(dim)))
    sg = H[H.coord.str.startswith("signerr")].groupby(["kappa", "p", "rep", "feature", "stratum", "coord"])[["sse", "n"]].sum().reset_index()
    sg["rate"] = sg.sse / np.maximum(sg.n, 1)
    st = HP.groupby(["kappa", "p", "rep", "feature", "stratum", "coord"])[["sse", "n"]].sum().reset_index()
    st["stab_sd"] = np.sqrt(st.sse / (np.maximum(st.n, 1) * st.coord.map(dim)) / 2)
    acc.to_csv(f"{O}/g3_accuracy_per_rep.csv", index=False); sg.to_csv(f"{O}/g3_signerr_per_rep.csv", index=False); st.to_csv(f"{O}/g3_stability_per_rep.csv", index=False)
    # criteria per (kappa, p, feature, stratum) on s6
    rows = []
    base = {"A1": "raw_own1", "A2": "raw_own2", "Aslot1": "raw_own1", "Aslot2": "raw_own2", "B1": "raw_joint", "B2": "raw_joint"}
    for (k, p, fn, s), x in acc[acc.coord == "s6"].groupby(["kappa", "p", "feature", "stratum"]):
        r = dict(kappa=k, p=p, feature=fn, stratum=s, rmse_s6=x.rmse.mean(), rmse_s6_sd=x.rmse.std(ddof=1), n_rep=len(x),
                 rmse_s1=acc[(acc.kappa == k) & (acc.p == p) & (acc.feature == fn) & (acc.stratum == s) & (acc.coord == "s1")].rmse.mean(),
                 rmse_s2=acc[(acc.kappa == k) & (acc.p == p) & (acc.feature == fn) & (acc.stratum == s) & (acc.coord == "s2")].rmse.mean())
        y = st[(st.kappa == k) & (st.p == p) & (st.feature == fn) & (st.stratum == s) & (st.coord == "s6")]
        r["stab_sd_s6"] = y.stab_sd.mean() if len(y) else np.nan
        if fn in base:
            b = acc[(acc.kappa == k) & (acc.p == p) & (acc.feature == base[fn]) & (acc.stratum == s) & (acc.coord == "s6")]
            m = x.merge(b, on="rep", suffixes=("", "_b"))
            if len(m) >= 3:
                q = paired((m.rmse - m.rmse_b).values); r.update(vs_raw=base[fn], diff_vs_raw=q["mean"], diff_ci95_hi=q["ci95_hi"], diff_ci95_lo=q["ci95_lo"])
        r["crit_a_rmse_le_0.2"] = r["rmse_s6"] <= 0.2; r["crit_b_stab_le_0.1"] = bool(r["stab_sd_s6"] <= 0.1) if not np.isnan(r["stab_sd_s6"]) else None
        r["crit_c_better_than_raw"] = bool(r.get("diff_ci95_hi", np.inf) < 0)
        r["approx_state_recovery"] = bool(r["crit_a_rmse_le_0.2"] and r["crit_b_stab_le_0.1"] and r["crit_c_better_than_raw"])
        rows.append(r)
    G = pd.DataFrame(rows); G.to_csv(f"{O}/g3_criteria.csv", index=False); return G


if __name__ == "__main__":
    os.makedirs(O, exist_ok=True)
    C, DD, SEQ, Q = g2(); G = g3()
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500); f = lambda v: f"{v:.4f}"
    print(Q.groupby(level=[0, 1, 2]).size().rename("n_rep_rows").reset_index().query("stratum == 'all'").to_string(index=False))
    print(SEQ.to_string(index=False, float_format=f))
    cols = ["kappa", "p", "contrast", "mean", "ci95_lo", "ci95_hi", "d_z", "n_negative", "n", "tost_equiv", "phase5_label"]
    print(C[C.stratum == "all"][cols].to_string(index=False, float_format=f))
