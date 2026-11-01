"""P7-B analysis (CPU) exactly as fixed in results/notes/phase7_p7b_protocol.md section 5.
Unit = independent replicate (24 per kappa, p). Q_r(F) = RMSE_s6 pooled over 8 test trajectories x 5 masks (stratum all);
Q_r(Bself) = mean_j Q_r(Bself_j). Delta_r = Q_r(Bfull) - Q_r(Bself). Paired t 95% two-sided; fixed sequence S1..S4.
Writes results/extra/phase7/analysis/*.csv"""
import glob, os
import numpy as np, pandas as pd
from scipy import stats

E = "results/extra/phase7/eval"; O = "results/extra/phase7/analysis"; D6 = 0.05; DIM = {"s6": 6, "s1": 3, "s2": 3, "y1": 1, "y2": 1}
SEQ = [("S1", 1.0, 0.7), ("S2", 1.0, 0.3), ("S3", 0.0, 0.7), ("S4", 0.0, 0.3)]


def paired(d):
    d = np.asarray(d, float); n = len(d); m, sd = d.mean(), d.std(ddof=1); se = sd / np.sqrt(n); t = stats.t.ppf(.975, n - 1)
    try:
        wp = stats.wilcoxon(d).pvalue
    except ValueError:
        wp = np.nan
    return dict(n=n, mean=m, sd=sd, ci95_lo=m - t * se, ci95_hi=m + t * se, d_z=m / sd if sd > 0 else np.nan, n_negative=int((d < 0).sum()),
                wilcoxon_p=wp, detected=bool(m + t * se < 0), mean_le_minus_delta=bool(m <= -D6), ci_hi_lt_minus_delta=bool(m + t * se < -D6))


def per_rep(kind, valcol):
    fs = sorted(glob.glob(f"{E}/k*_{kind}.csv")); A = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    g = A.groupby(["kappa", "p", "rep", "feature", "stratum", "coord"])[["sse", "n"]].sum().reset_index()
    g[valcol] = np.sqrt(g.sse / (np.maximum(g.n, 1) * g.coord.map(DIM))) / (np.sqrt(2) if kind == "pair" else 1)
    g.loc[g.n == 0, valcol] = np.nan
    return g


def main():
    os.makedirs(O, exist_ok=True)
    acc = per_rep("acc", "rmse"); pair = per_rep("pair", "stab_sd")
    acc.to_csv(f"{O}/p7_accuracy_per_rep.csv", index=False); pair.to_csv(f"{O}/p7_stability_per_rep.csv", index=False)
    W = acc.pivot_table(index=["kappa", "p", "stratum", "coord", "rep"], columns="feature", values="rmse")
    W["Bself"] = (W["Bself1"] + W["Bself2"]) / 2
    # primary + strata/coords (s6)
    rows = []
    for (k, p, st, c), x in W.groupby(level=[0, 1, 2, 3]):
        for a, b in (("Bfull", "Bself"), ("Bfull", "raw_joint"), ("Bself", "raw_joint")):
            d = (x[a] - x[b]).dropna()
            if len(d) >= 3:
                rows.append(dict(kappa=k, p=p, stratum=st, coord=c, contrast=f"{a} - {b}", mean_a=x[a].mean(), mean_b=x[b].mean(), **paired(d.values)))
    C = pd.DataFrame(rows); C.to_csv(f"{O}/p7_contrasts.csv", index=False)
    # own / other subsystem, matched by j: Bself_j vs Bfull on s_j (own) and s_{3-j} (other)
    sub = []
    for j in (1, 2):
        for role, c in (("own", f"s{j}"), ("other", f"s{3 - j}")):
            x = W.xs(("all", c), level=("stratum", "coord"))
            for (k, p), y in x.groupby(level=[0, 1]):
                sub.append(dict(kappa=k, p=p, j=j, role=role, Bself_j=y[f"Bself{j}"].mean(), Bfull=y["Bfull"].mean(), raw_joint=y["raw_joint"].mean(),
                                **{f"diff_{q}": v for q, v in paired((y["Bfull"] - y[f"Bself{j}"]).values).items()}))
    S = pd.DataFrame(sub); S.to_csv(f"{O}/p7_subsystem.csv", index=False)
    # stability (s6), all strata
    P = pair.pivot_table(index=["kappa", "p", "stratum", "coord", "rep"], columns="feature", values="stab_sd"); P["Bself"] = (P["Bself1"] + P["Bself2"]) / 2
    st = []
    for (k, p, s, c), x in P.groupby(level=[0, 1, 2, 3]):
        d = (x["Bfull"] - x["Bself"]).dropna()
        if len(d) >= 3:
            st.append(dict(kappa=k, p=p, stratum=s, coord=c, Bfull=x["Bfull"].mean(), Bself=x["Bself"].mean(), raw_joint=x["raw_joint"].mean(), **paired(d.values)))
    pd.DataFrame(st).to_csv(f"{O}/p7_stability.csv", index=False)
    # kappa DiD (Welch) on primary
    dd = []
    for p in (0.3, 0.7):
        d1 = (W.loc[(1.0, p, "all", "s6")]["Bfull"] - W.loc[(1.0, p, "all", "s6")]["Bself"]).values
        d0 = (W.loc[(0.0, p, "all", "s6")]["Bfull"] - W.loc[(0.0, p, "all", "s6")]["Bself"]).values
        r = stats.ttest_ind(d1, d0, equal_var=False); v1, v0 = d1.var(ddof=1) / len(d1), d0.var(ddof=1) / len(d0); se = np.sqrt(v1 + v0)
        dfw = se ** 4 / (v1 ** 2 / (len(d1) - 1) + v0 ** 2 / (len(d0) - 1)); m = d1.mean() - d0.mean(); h = stats.t.ppf(.975, dfw) * se
        dd.append(dict(p=p, contrast="DiD k1-k0 (Bfull - Bself), s6 all", mean=m, ci95_lo=m - h, ci95_hi=m + h, df=dfw))
    pd.DataFrame(dd).to_csv(f"{O}/p7_did.csv", index=False)
    # fixed sequence
    seq, go = [], True
    for name, k, p in SEQ:
        r = C[(C.kappa == k) & (C.p == p) & (C.stratum == "all") & (C.coord == "s6") & (C.contrast == "Bfull - Bself")]
        r = r.iloc[0] if len(r) else None
        det = bool(r is not None and r.detected)
        seq.append(dict(step=name, kappa=k, p=p, tested_confirmatory=go, detected=det, confirmed=go and det,
                        **({q: r[q] for q in ("n", "mean", "ci95_lo", "ci95_hi", "d_z", "n_negative", "ci_hi_lt_minus_delta", "mean_a", "mean_b")} if r is not None else {})))
        go = go and det
    Sq = pd.DataFrame(seq); Sq.to_csv(f"{O}/p7_fixed_sequence.csv", index=False)
    rows_ = pd.concat([pd.read_csv(f) for f in glob.glob("results/extra/phase7/runs/rows/*.csv")]); rows_.to_csv(f"{O}/p7_training_rows.csv", index=False)
    pd.set_option("display.width", 250); f = lambda v: f"{v:.4f}"
    print(Sq.to_string(index=False, float_format=f))
    print(S.to_string(index=False, float_format=f))


if __name__ == "__main__":
    main()
