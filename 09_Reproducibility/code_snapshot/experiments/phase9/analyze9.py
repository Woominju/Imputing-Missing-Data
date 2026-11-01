"""Phase 9 analysis exactly as preregistered (protocol sections 4-6). Unit = replicate; trajectories x masks pooled inside a replicate.
  pilot : Gate D criteria G-P1..G-P4 (G-P5 = test log) -> PASS / FAIL. No effect conclusions.
  main  : fixed sequence P-1 -> P-2 -> P-3 -> P-4 at kappa = 1, p = 0.7 (one-sided alpha 0.025 each), practical thresholds reported
          separately; S2 / S3 flags; descriptive tables for every cell (imputation per view / stratum, retrieval, R1 decodability,
          mask variance, training diagnostics). Refuses smoke artefacts unless P9_SMOKE=1. Writes ANA/<stage>_<time>/ (never overwrites).
usage: analyze9.py STAGE"""
import glob, json, math, os, sys, time
import numpy as np, pandas as pd
from scipy import stats
from experiments.phase9.core9 import RUNS, EVAL, ANA, SZ, CONST, ARMS9, SMOKE

C = CONST


def tb(d, a=C["alpha"]):
    d = np.asarray(d, float); n = len(d); m = d.mean(); sd = d.std(ddof=1); se = sd / math.sqrt(n); q = stats.t.ppf(1 - a, n - 1)
    return dict(n=n, mean=m, sd=sd, lcb=m - q * se, ucb=m + q * se, n_neg=int((d < 0).sum()), n_pos=int((d > 0).sum()))


def sign1_lt(d, thr=0.0):
    d = np.asarray(d, float) - thr; d = d[d != 0]; k = int((d < 0).sum()); n = len(d)
    return float(stats.binomtest(k, n, 0.5, alternative="greater").pvalue) if n else 1.0


def load(stage, kind):
    fs = sorted(glob.glob(f"{EVAL}/{stage}_k*_{kind}.csv.gz"))
    if kind != "acc":
        fs = [f for f in fs if os.path.exists(f.replace(f"_{kind}.csv.gz", "_acc.csv.gz"))]
    D = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()
    if len(D) and D.smoke.astype(str).str.lower().eq("true").any() and not SMOKE:
        sys.exit(f"REFUSED: smoke artefacts in {kind}")
    return D


def main():
    stage = sys.argv[1]; out = f"{ANA}/{stage}_{time.strftime('%Y%m%d-%H%M%S')}"; os.makedirs(out); say = []
    P = lambda s: (say.append(str(s)), print(s, flush=True))
    rows = pd.concat([pd.read_csv(f) for f in glob.glob(f"{RUNS}/rows/{stage}_*.csv")], ignore_index=True)
    fails = glob.glob(f"{RUNS}/failed/{stage}_*.csv"); rows.to_csv(f"{out}/p9_training_rows.csv", index=False)
    imp, ret, acc, mv = load(stage, "imp"), load(stage, "ret"), load(stage, "acc"), load(stage, "mv")
    cells = rows.groupby(["kappa", "p"]).agg(n=("exp_id", "count"), valid=("valid", "all"), cap=("hit_cap", "sum"), nonfinite=("nonfinite_steps", "sum")).reset_index()
    ev = acc.groupby(["kappa", "p"]).rep.nunique().rename("evaluated_reps").reset_index(); cells = cells.merge(ev, how="left"); cells["invalid_runs"] = len(fails)
    cells.to_csv(f"{out}/p9_cells.csv", index=False); P("== cells\n" + cells.to_string(index=False))
    # replicate-level primary quantities
    a = imp[imp.stratum == "all"].groupby(["kappa", "p", "rep", "method"])[["sse", "n"]].sum(); a["rmse"] = np.sqrt(a.sse / a.n)
    IMP = a.rmse.unstack("method"); IMP.to_csv(f"{out}/p9_imputation_rep.csv")
    r = ret.groupby(["kappa", "p", "rep", "feat"])[["sum_nn", "sum_rand"]].sum(); r["rho_ret"] = r.sum_nn / r.sum_rand
    RET = r.rho_ret.unstack("feat"); RET.to_csv(f"{out}/p9_retrieval_rep.csv")
    summ = pd.concat([IMP.groupby(level=["kappa", "p"]).mean().add_prefix("imp_"), RET.groupby(level=["kappa", "p"]).mean().add_prefix("ret_")], axis=1)
    summ.to_csv(f"{out}/p9_cell_means.csv"); P("== cell means (imputation pooled RMSE; retrieval rho_ret)\n" + summ.round(4).to_string())
    # descriptive paired contrasts in every cell
    D = []
    for (k, p), y in IMP.groupby(level=["kappa", "p"]):
        z = RET.loc[(k, p)]
        for nm, d in (("imp P9-Bfull", y["AE:P9"] - y["AE:Bfull"]), ("imp P9-Fact8", y["AE:P9"] - y["AE:Fact8"]), ("imp P9-Fact16", y["AE:P9"] - y["AE:Fact16"]),
                      ("imp Bfull-Fact8", y["AE:Bfull"] - y["AE:Fact8"]), ("ret P9-Bfull", z.P9 - z.Bfull), ("ret P9-Fact8", z.P9 - z.Fact8),
                      ("ret P9-Fact16", z.P9 - z.Fact16), ("ret Bfull-Fact8", z.Bfull - z.Fact8)):
            D.append(dict(kappa=k, p=p, contrast=nm, **tb(d.values, .025)))
    D = pd.DataFrame(D); D.to_csv(f"{out}/p9_paired_all_cells.csv", index=False); P("== paired contrasts, all cells (descriptive except the primary cell)\n" + D.round(5).to_string(index=False))
    if stage == "pilot":
        p9 = rows[rows.arm == "P9"]
        g1 = bool(rows.valid.all() and not rows.hit_cap.any() and not fails and len(rows) == len(ARMS9) * 4 * SZ["R"]["pilot"])
        g2 = bool(np.isfinite(p9.tower_loss_last).all() and (p9.tower_loss_last > 0).all() and (p9.tower_targets_per_step > 0).all())
        tr = mv.dropna(subset=["rho_white"]) if "rho_white" in mv else pd.DataFrame()
        g3 = bool((summ.ret_P9 < C["pilot_rho_max"]).all())
        g4 = bool((summ["imp_AE:P9"] <= summ["imp_AE:Bfull"] + C["pilot_imp_slack"]).all())
        G = dict(G_P1_all_valid=g1, G_P2_tower_active=g2, G_P3_no_collapse_rho_lt_0_8=g3, G_P4_no_catastrophe=g4,
                 G_P5_integrity_tests="see results/notes/phase9_test_log.txt", PILOT_PASS=bool(g1 and g2 and g3 and g4))
        json.dump(G, open(f"{out}/p9_pilot_gate.json", "w"), indent=1); P("== PILOT GATE\n" + json.dumps(G, indent=1))
    else:
        k, p = C["primary"]; y = IMP.loc[(k, p)]; z = RET.loc[(k, p)]; seq, go = [], True
        for step, d, rule, thr, prac in (("P-1", y["AE:P9"] - y["AE:Bfull"], "ucb<0", 0.0, C["delta_imp"]), ("P-2", z.P9 - z.Bfull, "ucb<0", 0.0, C["delta_ret"]),
                                         ("P-3", y["AE:P9"] - y["AE:Fact8"], "ucb<eta", C["eta_ni"], None), ("P-4", z.P9 - z.Fact8, "ucb<0", 0.0, C["delta_ret"])):
            b = tb(d.values); det = bool(b["ucb"] < thr); conf = bool(go and det and len(d) == SZ["R"]["main"])
            seq.append(dict(step=step, rule=rule, tested=go, decision=conf, **b, sign_p=sign1_lt(d.values, thr),
                            practical=(bool(b["mean"] <= -prac) if prac else None))); go = conf
        S = pd.DataFrame(seq); S.to_csv(f"{out}/p9_confirmatory_sequence.csv", index=False); P("== CONFIRMATORY (kappa=1, p=0.7)\n" + S.to_string(index=False))
        flags = dict(S3_imputation_utility=bool(S.decision[0] and S.decision[2]), S2_shared_state_evidence=bool(S.decision[1] and S.decision[3]),
                     note="S1 (mathematical validity) is judged by document audit; AIM progress requires S1 and S2 and S3")
        json.dump(flags, open(f"{out}/p9_success_flags.json", "w"), indent=1); P(json.dumps(flags, indent=1))
    # descriptive extras
    v = imp.groupby(["kappa", "p", "rep", "method", "view", "stratum"])[["sse", "n"]].sum(); v["rmse"] = np.sqrt(v.sse / v.n.replace(0, np.nan))
    v.groupby(["kappa", "p", "method", "view", "stratum"]).rmse.mean().to_csv(f"{out}/p9_imputation_view_stratum.csv")
    g = acc.groupby(["kappa", "p", "rep", "pipeline", "stratum", "coord"])[["sse", "n"]].sum().reset_index()
    s6 = g.groupby(["kappa", "p", "rep", "pipeline", "stratum"]).apply(lambda q: math.sqrt(q.sse.sum() / (6 * q.n.iloc[0])) if q.n.iloc[0] else np.nan).rename("rmse_s6").reset_index()
    s6.groupby(["kappa", "p", "pipeline", "stratum"]).rmse_s6.mean().to_csv(f"{out}/p9_decodability_s6.csv")
    rs = ret.groupby(["kappa", "p", "rep", "feat"]).sum(numeric_only=True)
    ex = pd.DataFrame({"exact_hit_rate": rs.exact_hits / (rs.n_q * 10), **{f"rho_ret_{s}": rs[f"nn_{s}"] / rs[f"rand_{s}"].replace(0, np.nan) for s in ("H", "X0", "both_L2+", "one_L2+")}})
    ws = ret.groupby(["kappa", "p", "rep", "feat"])[["d_same", "d_rand"]].mean(); ex["within_over_between"] = ws.d_same / ws.d_rand
    ex.groupby(level=["kappa", "p", "feat"]).mean().to_csv(f"{out}/p9_retrieval_extras.csv")
    mv.groupby(["kappa", "p", "feat"]).mean(numeric_only=True).drop(columns=["rep"]).to_csv(f"{out}/p9_mask_variance.csv")
    rows.groupby(["kappa", "p", "arm"]).agg(best_val=("best_val", "mean"), epochs=("epochs_run", "mean"), tow_first=("tower_loss_first", "mean"),
                                           tow_last=("tower_loss_last", "mean"), wall=("wall_s", "mean")).to_csv(f"{out}/p9_training_by_arm.csv")
    open(f"{out}/p9_summary.txt", "w").write("\n".join(say) + "\n"); open(f"{ANA}/LATEST_{stage}", "w").write(out + "\n"); print("ANALYZE9 WROTE", out)


if __name__ == "__main__":
    main()
