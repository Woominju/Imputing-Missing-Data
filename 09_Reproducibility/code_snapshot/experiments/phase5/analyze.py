"""Phase 5 analysis (CPU), exactly as pre-registered in results/notes/phase5_hypothesis_protocol.md section 6.
Unit = replicate r (seed = r). Replicate metric = sqrt(mean_j MSE_j), MSE_j pooled over the 10 test trajectories (equal length)
and averaged over MLP probe seeds {0,1}. Contrasts are paired within replicate. Outputs results/extra/phase5/analysis/*.csv
and a markdown summary ANALYSIS.md."""
import glob, os
import numpy as np, pandas as pd
from scipy import stats

E = "results/extra/phase5/eval"; O = "results/extra/phase5/analysis"; DELTA = 0.02; NB = 2000
GEN = {"A": "A{j}", "B": "B{j}", "Bcur": "Bcur{j}", "C": "C", "Sham": "Sham{j}", "raw_own": "raw_own{j}", "raw_joint": "raw_joint",
       "raw_owncur": "raw_owncur{j}", "raw_cur": "raw_cur"}
CONTRASTS = [("H2 primary: B - A", "B", "A"), ("H1: raw_joint - raw_own", "raw_joint", "raw_own"),
             ("H1: raw_owncur - raw_own", "raw_owncur", "raw_own"), ("H1: raw_joint - raw_owncur", "raw_joint", "raw_owncur"),
             ("B - Bcur", "B", "Bcur"), ("Bcur - A", "Bcur", "A"), ("Sham - A", "Sham", "A"), ("C - B", "C", "B"),
             ("latent B - raw_joint", "B", "raw_joint"), ("latent A - raw_own", "A", "raw_own"), ("raw_cur - raw_own", "raw_cur", "raw_own")]


def load(kind):
    fs = sorted(glob.glob(f"{E}/k*_{kind}.csv"))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


def generic(df, fcol="feature"):
    rows = []
    for g, pat in GEN.items():
        for j in (1, 2):
            sub = df[(df[fcol] == pat.format(j=j)) & (df.j == j)].copy(); sub["g"] = g; rows.append(sub)
    return pd.concat(rows, ignore_index=True)


def decide(m, lo95, hi95, lo90, hi90):
    if hi95 < 0 and m <= -DELTA: return "SUPPORTED (improvement >= delta)"
    if hi95 < 0: return "detectable but < delta"
    if lo90 > -DELTA and hi90 < DELTA: return "EQUIVALENT to zero (TOST, delta)"
    if lo95 > 0: return "OPPOSITE (worse)"
    return "inconclusive"


def contrast_table(per_rep, per_traj, label):
    """per_rep: DataFrame index rep, columns generic -> RMSE; per_traj: index (rep, traj) -> MSE per generic."""
    out = []
    for name, a, b in CONTRASTS:
        if a not in per_rep or b not in per_rep:
            continue
        d = (per_rep[a] - per_rep[b]).dropna().values; n = len(d)
        if n < 2:
            continue
        m, se = d.mean(), d.std(ddof=1) / np.sqrt(n)
        t95, t90 = stats.t.ppf(.975, n - 1), stats.t.ppf(.95, n - 1)
        # crossed bootstrap over replicates x test trajectories (RMSE recomputed from MSE per traj)
        rng = np.random.default_rng(0); reps = per_traj.index.get_level_values(0).unique().values; trs = per_traj.index.get_level_values(1).unique().values
        A = per_traj[a].unstack(); B = per_traj[b].unstack()            # rep x traj (MSE)
        bs = []
        for _ in range(NB):
            rr = rng.choice(reps, len(reps)); tt = rng.choice(trs, len(trs))
            bs.append(np.mean(np.sqrt(A.loc[rr, tt].values.mean(1)) - np.sqrt(B.loc[rr, tt].values.mean(1))))
        # trajectory-level variability: per test trajectory, replicate-averaged difference of RMSE
        dtraj = (np.sqrt(A) - np.sqrt(B)).mean(0)
        try:
            wp = stats.wilcoxon(d).pvalue
        except ValueError:
            wp = np.nan
        out.append(dict(setting=label, contrast=name, n_rep=n, mean=m, ci95_lo=m - t95 * se, ci95_hi=m + t95 * se, ci90_lo=m - t90 * se,
                        ci90_hi=m + t90 * se, n_negative=int((d < 0).sum()), wilcoxon_p=wp, boot95_lo=np.percentile(bs, 2.5),
                        boot95_hi=np.percentile(bs, 97.5), traj_sd=float(dtraj.std(ddof=1)), traj_n_negative=int((dtraj < 0).sum()),
                        mean_a=per_rep[a].mean(), mean_b=per_rep[b].mean(),
                        decision=decide(m, m - t95 * se, m + t95 * se, m - t90 * se, m + t90 * se)))
    return out


def main():
    os.makedirs(O, exist_ok=True)
    P = load("probe"); P = P[P.seed == P.rep]                                   # main replicates only
    P = generic(P); P = P[P.target.str.startswith("y")]
    res, levels = [], []
    for (k, p), D in P.groupby(["kappa", "p"]):
        for stratum in ("all", "L0", "L0c1", "L1", "L2+", "c0", "c1"):
            S = D[(D.stratum == stratum) & D.probe.isin(["mlp0", "mlp1"])]
            # MSE per (rep, g, j, traj) averaged over probe seeds
            mt = S.groupby(["rep", "g", "j", "traj"]).apply(lambda x: x.sse.sum() / max(x.n.sum(), 1)).rename("mse").reset_index()
            per_traj = mt.groupby(["rep", "traj", "g"]).mse.mean().unstack("g")            # mean over j
            pj = S.groupby(["rep", "g", "j"]).apply(lambda x: x.sse.sum() / max(x.n.sum(), 1)).rename("mse").reset_index()
            per_rep = np.sqrt(pj.groupby(["rep", "g"]).mse.mean().unstack("g"))
            for g in per_rep:
                levels.append(dict(kappa=k, p=p, stratum=stratum, feature=g, rmse_mean=per_rep[g].mean(), rmse_sd_rep=per_rep[g].std(ddof=1), n_rep=per_rep[g].notna().sum()))
            res += contrast_table(per_rep, per_traj, dict(kappa=k, p=p, stratum=stratum).__repr__())
    R = pd.DataFrame(res); L = pd.DataFrame(levels)
    R.to_csv(f"{O}/contrasts_probe_mlp.csv", index=False); L.to_csv(f"{O}/levels_probe_mlp.csv", index=False)
    # ridge probe levels (secondary) and s_j ridge
    Pr = load("probe"); Pr = generic(Pr[Pr.seed == Pr.rep])
    rl = Pr[(Pr.probe == "ridge") & (Pr.stratum == "all")].groupby(["kappa", "p", "target", "g", "rep"]).apply(lambda x: np.sqrt(x.sse.sum() / x.n.sum())).rename("rmse").reset_index()
    rl["target"] = rl.target.str[0]; rl.groupby(["kappa", "p", "target", "g"]).rmse.agg(["mean", "std"]).reset_index().to_csv(f"{O}/levels_ridge.csv", index=False)
    # imputation
    I = load("imp"); I = I[I.seed == I.rep].copy(); I["g"] = I.arm.str.replace(r"[12]$", "", regex=True)
    I = I[(I.arm == "C") | (I.arm.str[-1].astype(str) == I.j.astype(str))]
    imp = I[I.stratum == "all"].groupby(["kappa", "p", "rep", "g"]).apply(lambda x: np.sqrt(x.sse.sum() / x.n.sum())).rename("rmse").reset_index()
    imp.to_csv(f"{O}/imputation_per_rep.csv", index=False)
    ic = []
    for (k, p), D in imp.groupby(["kappa", "p"]):
        W = D.pivot(index="rep", columns="g", values="rmse")
        for a, b in (("B", "A"), ("C", "B"), ("Bcur", "A"), ("Sham", "A"), ("B", "Bcur")):
            d = (W[a] - W[b]).values; se = d.std(ddof=1) / np.sqrt(len(d)); t = stats.t.ppf(.975, len(d) - 1)
            ic.append(dict(kappa=k, p=p, contrast=f"{a} - {b}", mean=d.mean(), ci95_lo=d.mean() - t * se, ci95_hi=d.mean() + t * se, n_negative=int((d < 0).sum()), mean_a=W[a].mean(), mean_b=W[b].mean()))
    pd.DataFrame(ic).to_csv(f"{O}/contrasts_imputation.csv", index=False)
    # H3
    H = load("h3"); H = generic(H[H.seed == H.rep])
    h3 = H.groupby(["kappa", "p", "kind", "stratum", "g", "rep"]).apply(lambda x: np.sqrt(x.sse.sum() / max(x.n.sum(), 1))).rename("v").reset_index()
    h3.groupby(["kappa", "p", "kind", "stratum", "g"]).v.agg(["mean", "std", "count"]).reset_index().to_csv(f"{O}/h3_levels.csv", index=False)
    hc = []
    for (k, p, kind, st), D in h3.groupby(["kappa", "p", "kind", "stratum"]):
        W = D.pivot(index="rep", columns="g", values="v")
        for a, b in (("B", "A"), ("C", "A"), ("B", "raw_joint"), ("A", "raw_own")):
            if a in W and b in W:
                d = (W[a] - W[b]).dropna().values; se = d.std(ddof=1) / np.sqrt(len(d)); t = stats.t.ppf(.975, len(d) - 1)
                hc.append(dict(kappa=k, p=p, kind=kind, stratum=st, contrast=f"{a} - {b}", mean=d.mean(), ci95_lo=d.mean() - t * se, ci95_hi=d.mean() + t * se, n_negative=int((d < 0).sum())))
    pd.DataFrame(hc).to_csv(f"{O}/h3_contrasts.csv", index=False)
    # dynamics
    Y = load("dyn"); Y = generic(Y[Y.seed == Y.rep])
    Y.groupby(["kappa", "p", "g", "rep"]).apply(lambda x: np.sqrt(x.sse.sum() / x.n.sum())).rename("rmse").reset_index() \
        .groupby(["kappa", "p", "g"]).rmse.agg(["mean", "std"]).reset_index().to_csv(f"{O}/dyn_levels.csv", index=False)
    # training-seed variability (replicate 0, p = 0.7, seeds 0,1,2)
    Ps = load("probe"); Ps = generic(Ps[(Ps.rep == 0) & (Ps.p == 0.7)]); Ps = Ps[Ps.target.str.startswith("y") & (Ps.stratum == "all") & Ps.probe.isin(["mlp0", "mlp1"]) & Ps.g.isin(["A", "B"])]
    sv = Ps.groupby(["kappa", "seed", "g", "j"]).apply(lambda x: x.sse.sum() / x.n.sum()).rename("mse").reset_index()
    sv = np.sqrt(sv.groupby(["kappa", "seed", "g"]).mse.mean()).unstack("g"); sv["B-A"] = sv["B"] - sv["A"]; sv.reset_index().to_csv(f"{O}/seed_variability.csv", index=False)
    # training diagnostics
    rows = pd.concat([pd.read_csv(f) for f in glob.glob("results/extra/phase5/runs/rows/*.csv")])
    rows.to_csv(f"{O}/training_rows.csv", index=False)
    print("analysis written", len(R), "contrasts")


if __name__ == "__main__":
    main()
