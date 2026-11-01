"""Phase 8 analysis (CPU) exactly as fixed in results/notes/phase8_mathematical_protocol_v2.md sections 4.4-4.5, 5.3-5.5, 6, 7, 9.
Unit = independent replicate (n = SZ['R'] per kappa, p); test trajectories x masks pooled inside a replicate; oracle results are
conditional on the oracle pool / reference DB. Confirmatory families: I (alpha 0.025 one-sided, fixed sequence I-1..I-4),
IIa and IIb (0.0125 each, fixed sequence, kappa = 0 equivalence gate = per-j exact sign test + TOST eps). Everything else descriptive.
Refuses smoke artefacts unless P8_SMOKE=1.
Output versioning: every analysis whose INPUTS differ from all earlier ones (fingerprint over the scheduled output markers,
failure markers and every eval/knn/d5b/oracle file read) is written to a NEW directory ANA/<status>_<time>_<fp10>/, status =
'complete' (every scheduled task output exists and no task failed) or 'partial'; earlier directories are never modified;
ANA/LATEST names the newest one. Identical inputs -> 'UP-TO-DATE', nothing written."""
import glob, hashlib, json, os, sys, time
import numpy as np, pandas as pd
from scipy import stats
from experiments.phase5.core import SPAN
from experiments.phase8.core8 import ROOT, EVAL, ANA, RUNS, ORC, SZ, CONST, SMOKE, SYM_COORDS, load, norm_stats, pool_ids

C = CONST; YIDX = {1: 1, 2: 4}; OUT = []; OUTDIR = None


def say(s=""):
    OUT.append(str(s)); print(s, flush=True)


# ---------------------------------------------------------------- statistics helpers (tested)
def bounds(d, alpha):
    d = np.asarray(d, float); n = len(d); m = d.mean(); sd = d.std(ddof=1) if n > 1 else np.nan; se = sd / np.sqrt(n)
    q = stats.t.ppf(1 - alpha, n - 1); h95 = stats.t.ppf(.975, n - 1) * se
    return dict(n=n, mean=m, sd=sd, lcb=m - q * se, ucb=m + q * se, ci95_lo=m - h95, ci95_hi=m + h95,
                d_z=m / sd if sd and sd > 0 else np.nan, n_neg=int((d < 0).sum()), n_pos=int((d > 0).sum()))


def sign_test(d):
    d = np.asarray(d, float); d = d[d != 0]
    return 1.0 if len(d) == 0 else float(stats.binomtest(int((d > 0).sum()), len(d), 0.5).pvalue)


def tost(d, eps, alpha):
    b = bounds(d, alpha); return dict(tost_lo=b["lcb"], tost_hi=b["ucb"], equivalent=bool(b["lcb"] > -eps and b["ucb"] < eps))


def gate(per_j, avg):
    """per_j: dict j -> replicate array; avg: replicate array. -> PASS / FAIL / UNDETERMINED (protocol 5.3)."""
    ps = {j: sign_test(v) for j, v in per_j.items()}; t = tost(avg, C["eps_equiv"], C["alpha_tost"])
    st = "FAIL" if any(p < C["alpha_gate_sign"] for p in ps.values()) else ("PASS" if t["equivalent"] else "UNDETERMINED")
    return dict(status=st, **{f"sign_p_j{j}": p for j, p in ps.items()}, **t, **{f"avg_{a}": b for a, b in bounds(avg, C["alpha_tost"]).items()})


LABEL = {"PASS": "confirmatory", "UNDETERMINED": "확증 보류(대조 동등성 미확립)", "FAIL": "보류(게이트 FAIL)"}


# ---------------------------------------------------------------- loading
def guard(df, what):
    if "smoke" in df and df.smoke.astype(str).str.lower().eq("true").any() and not SMOKE:
        sys.exit(f"REFUSED: smoke artefacts in {what}")


def load_sums(kind):
    """eval sums of one kind. The acc file is the ONLY completeness marker (written last, atomically): side files (pair, imp, lat)
    of a replicate whose acc file is missing (e.g. killed at the hard deadline) are ignored."""
    fs = sorted(f for f in glob.glob(f"{EVAL}/k*_{kind}.csv.gz"))
    if kind != "acc":
        fs = [f for f in fs if os.path.exists(f.replace(f"_{kind}.csv.gz", "_acc.csv.gz"))]
    A = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True); guard(A, kind)
    return A


def risks(acc):
    """per (kappa, p, rep, pipeline, stratum): MSE (coordinate mean), per-coordinate sse and n."""
    g = acc.groupby(["kappa", "p", "rep", "pipeline", "stratum", "coord"])[["sse", "n"]].sum().reset_index()
    W = g.pivot_table(index=["kappa", "p", "rep", "pipeline", "stratum"], columns="coord", values="sse")
    n = g.groupby(["kappa", "p", "rep", "pipeline", "stratum"]).n.first()
    return W, n


def R(W, n, pipe, st, coords=range(6)):
    x = W.xs((pipe, st), level=("pipeline", "stratum")); nn = n.xs((pipe, st), level=("pipeline", "stratum"))
    return (x[list(coords)].sum(1) / (nn * len(list(coords)))).where(nn > 0)


def lb_pool(k, coords):
    """section 2.4: per pool-train trajectory sum_c E[s_c^2]/sigma_c^2 / 6 over n in [SPAN, N_TEST); trajectory-level t CI."""
    _, _, _, sd = norm_stats(k); v = []
    for q in pool_ids()["train"]:
        S = load(k, q)[SPAN:SZ["N_TEST"]]; v.append(sum((S[:, c] ** 2).mean() / sd[c] ** 2 for c in coords) / 6)
    v = np.array(v); m = v.mean(); h = stats.t.ppf(.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan
    return dict(LB=m, LB_lo=m - h, LB_hi=m + h, n_traj=len(v))


def status_and_fingerprint():
    """-> (status, fingerprint, missing task names, failed task names, failed AE runs)."""
    from experiments.phase8.sched8 import tasks, fmarker
    T = tasks(); missing = [nm for nm, _, _, o in T if not os.path.exists(o)]; failed = [nm for nm, _, _, _ in T if os.path.exists(fmarker(nm))]
    files = sorted(set(glob.glob(f"{EVAL}/*") + glob.glob(f"{ORC}/*") + glob.glob(f"{RUNS}/rows/*") + glob.glob(f"{RUNS}/failed/*")
                       + glob.glob(f"{ROOT}/failed/*")))
    h = hashlib.sha256()
    for f in files:
        st = os.stat(f); h.update(f"{f}|{st.st_size}|{st.st_mtime_ns}\n".encode())
    status = "complete" if not missing and not failed else "partial"
    return status, h.hexdigest(), missing, failed, sorted(glob.glob(f"{RUNS}/failed/*.csv"))


def main():
    global OUTDIR
    rf = glob.glob(f"{RUNS}/rows/*.csv")
    if not rf or not glob.glob(f"{EVAL}/k*_acc.csv.gz"):
        print("ANALYZE8 NOTHING TO ANALYZE (no completed training rows / evaluations yet)"); return
    rows = pd.concat([pd.read_csv(f) for f in rf], ignore_index=True); guard(rows, "training rows")    # refuse BEFORE writing anything
    os.makedirs(ANA, exist_ok=True); status, fp, missing, failed, bad_runs = status_and_fingerprint()
    if any(d.endswith(fp[:10]) for d in os.listdir(ANA)):
        print("ANALYZE8 UP-TO-DATE", status, fp[:10]); return
    OUTDIR = f"{ANA}/{status}_{time.strftime('%Y%m%d-%H%M%S')}_{fp[:10]}"; tmpdir = OUTDIR + ".tmp"
    for d in glob.glob(f"{ANA}/*_{fp[:10]}.tmp"):                      # leftover of an interrupted analysis of the SAME inputs
        __import__("shutil").rmtree(d)
    os.makedirs(tmpdir)
    final, OUTDIR = OUTDIR, tmpdir
    json.dump(dict(status=status, fingerprint=fp, n_missing=len(missing), missing=missing, failed_tasks=failed,
                   failed_ae_runs=[os.path.basename(x) for x in bad_runs], time=time.strftime("%F %T")), open(f"{OUTDIR}/p8_status.json", "w"), indent=1)
    say(f"== ANALYSIS STATUS: {status.upper()}  missing tasks={len(missing)}  failed tasks={len(failed)}  invalid AE runs={len(bad_runs)}")
    if status == "partial":
        say("   PARTIAL analysis: cells with missing replicates are 미실행 / 결론 불가 (protocol 9); a later complete analysis supersedes it.")
    rows.to_csv(f"{OUTDIR}/p8_training_rows.csv", index=False)
    cell = rows.groupby(["kappa", "p"]).agg(n=("exp_id", "count"), cap_rate=("hit_cap", "mean"), nonfinite=("nonfinite_steps", "sum"),
                                            finite=("finite_params", "all")).reset_index()
    fr = sorted(glob.glob(f"{RUNS}/failed/*.csv")); FR = pd.concat([pd.read_csv(x) for x in fr], ignore_index=True) if fr else pd.DataFrame(columns=["kappa", "p", "arm"])
    FR.to_csv(f"{OUTDIR}/p8_invalid_runs.csv", index=False)                  # reported, never excluded silently (protocol 9)
    nf = FR.groupby(["kappa", "p"]).size().rename("n_invalid_runs") if len(FR) else None
    cell = cell.merge(nf.reset_index(), how="left", on=["kappa", "p"]) if nf is not None else cell.assign(n_invalid_runs=0)
    cell["n_invalid_runs"] = cell.n_invalid_runs.fillna(0).astype(int)
    cell["inconclusive"] = cell.cap_rate > 0.10; cell.to_csv(f"{OUTDIR}/p8_cells.csv", index=False)
    say("== training cells"); say(cell.to_string(index=False))
    bad = {(r.kappa, r.p) for r in cell.itertuples() if r.inconclusive}
    acc = load_sums("acc"); W, n = risks(acc)
    done = acc.groupby(["kappa", "p"]).rep.nunique()
    for k in C["kappas"]:
        for p in C["ps"]:
            nr = int(done.get((k, p), 0))
            if nr < SZ["R"]:
                bad.add((k, p)); say(f"CELL k={k} p={p}: evaluated replicates {nr}/{SZ['R']} -> 미실행 포함, 결론 불가 (protocol 9)")
    pd.DataFrame([dict(kappa=k, p=p, evaluated_reps=int(done.get((k, p), 0)), inconclusive=(k, p) in bad) for k in C["kappas"] for p in C["ps"]]).to_csv(
        f"{OUTDIR}/p8_cell_completeness.csv", index=False)
    # ================= Family I (p = 0.7)
    say("\n== Family I (MSE, coordinate mean; oracle-pool conditional)"); p7 = C["p_family_I"]; FI, desc = [], []
    for k in C["kappas"]:
        for st in ["H", "X0", "H_minus_X0", "one_L2+", "both_L2+", "all"]:
            g_tot = (R(W, n, "R1:Bfull", st) - R(W, n, "O_raw", st)).loc[(k, p7)]; g_lat = (R(W, n, "O_lat:Bfull", st) - R(W, n, "O_raw", st)).loc[(k, p7)]
            for nm, d in (("G_tot", g_tot), ("G_lat", g_lat), ("G_rd", g_tot - g_lat)):
                dv = d.dropna().values
                desc.append(dict(kappa=k, p=p7, stratum=st, contrast=nm, **bounds(dv, C["alpha_I"]), sign_p_vs_delta=sign_test(dv - C["delta_I"])))
    D = pd.DataFrame(desc); D.to_csv(f"{OUTDIR}/p8_familyI_all.csv", index=False)
    go = True
    for step, k, nm in (("I-1", 1.0, "G_tot"), ("I-2", 1.0, "G_lat"), ("I-3", 0.0, "G_tot"), ("I-4", 0.0, "G_lat")):
        r_ = D[(D.kappa == k) & (D.stratum == "H") & (D.contrast == nm)].iloc[0]
        conf = bool(go and (k, p7) not in bad and r_.lcb > C["delta_I"])
        FI.append(dict(step=step, kappa=k, contrast=nm, tested_confirmatory=go, confirmed_gt_delta=conf, descriptive_no_gap_ucb_lt_delta=bool(r_.ucb < C["delta_I"]),   # reverse direction: descriptive, outside alpha
                       mean=r_["mean"], lcb=r_.lcb, ucb=r_.ucb, d_z=r_.d_z, n=r_.n, cell_inconclusive=(k, p7) in bad)); go = conf
    FI = pd.DataFrame(FI); FI.to_csv(f"{OUTDIR}/p8_familyI_sequence.csv", index=False); say(FI.to_string(index=False))
    # T_A, LB, info ratio
    TA = []
    for k in C["kappas"]:
        for st, cs in SYM_COORDS.items():
            if st != "X0" and k != 0.0:
                continue                                     # N1-L_j0 holds only at kappa = 0
            lb = lb_pool(k, cs)
            for p in C["ps"]:
                for f in ["O_raw", "R1:Bfull", "O_lat:Bfull"]:
                    try:
                        x = W.xs((f, st), level=("pipeline", "stratum")).loc[(k, p)]; nn = n.xs((f, st), level=("pipeline", "stratum")).loc[(k, p)]
                        s0 = W.xs(("sym0", st), level=("pipeline", "stratum")).loc[(k, p)]
                    except KeyError:
                        continue
                    ok = nn > 0
                    if ok.sum() < 2:
                        continue
                    tA = ((x[list(range(6))].sum(1) - s0[cs].sum(1)) / (6 * nn))[ok]      # (1/6) normalisation (protocol 4.4)
                    rA = (x[list(range(6))].sum(1) / (6 * nn))[ok]
                    TA.append(dict(kappa=k, p=p, stratum=st, pipeline=f, T_A=tA.mean(), T_A_ci_lo=bounds(tA, .025)["ci95_lo"], T_A_ci_hi=bounds(tA, .025)["ci95_hi"],
                                   R_A=rA.mean(), info_ratio_LB_over_R=lb["LB"] / rA.mean(), **lb))
    TA = pd.DataFrame(TA); TA.to_csv(f"{OUTDIR}/p8_TA_LB.csv", index=False); say("\n== T_A / LB (symmetric strata only)"); say(TA.round(4).to_string(index=False))
    # section 2.3 decomposition (stratum all)
    pair = load_sums("pair"); dec = []
    for (k, p, r, pipe), y in pair.groupby(["kappa", "p", "rep", "pipeline"]):
        vm = (y.groupby("pair").sse.sum() / (y.groupby("pair").n.first() * 6 * y.traj.nunique())).mean() / 2
        mse = float(R(W, n, pipe, "all").loc[(k, p, r)]); dec.append(dict(kappa=k, p=p, rep=r, pipeline=pipe, mse=mse, mask_var=vm, mask_avg_bias2=mse - vm))
    pd.DataFrame(dec).groupby(["kappa", "p", "pipeline"]).mean(numeric_only=True).drop(columns="rep").to_csv(f"{OUTDIR}/p8_mask_decomposition.csv")
    # kNN / D5b / saturation (descriptive)
    kn = [pd.read_csv(f) for f in glob.glob(f"{EVAL}/knn_k*.csv") if "ksel" not in f]
    if kn:
        K = pd.concat(kn); guard(K, "knn"); K = K.groupby(["kappa", "p", "rep", "pipeline", "stratum"]).agg(sse=("sse", "sum"), n=("n", "sum"))
        K["mse"] = K.sse / K.n.replace(0, np.nan); K = K.drop(columns=["sse"]).reset_index()     # n summed over 6 coords = 6 x points
        K.pivot_table(index=["kappa", "p", "stratum"], columns="pipeline", values="mse").to_csv(f"{OUTDIR}/p8_knn_vs_oraw.csv")
    d5 = [pd.read_csv(f) for f in glob.glob(f"{EVAL}/d5b_k*.csv")]
    if d5:
        pd.concat(d5).groupby(["kappa", "p", "stratum"]).d5b.agg(["mean", "std", "count"]).to_csv(f"{OUTDIR}/p8_d5b_EXPLORATORY.csv")
    sat = {f: json.load(open(f)) for f in glob.glob(f"{ORC}/sat_*.json")}; json.dump(sat, open(f"{OUTDIR}/p8_saturation.json", "w"), indent=1)
    # ================= Family II (CSIA)
    say("\n== Family II (y_j RMSE; R1 readout)")
    rm = lambda pipe, st, c, k, p: np.sqrt(W.xs((pipe, st), level=("pipeline", "stratum"))[c] / n.xs((pipe, st), level=("pipeline", "stratum"))).loc[(k, p)]
    def CG(F, j, k, p):
        return (rm(f"R1:{F}", "all", YIDX[j], k, p) - rm(f"R1:{F}Sham{j}", "all", YIDX[j], k, p)).values
    def CERT(pipe, j, k, p):
        st = f"L{j}:L0"; s0 = W.xs(("sym0", st), level=("pipeline", "stratum"))[YIDX[j]]; f = W.xs((pipe, st), level=("pipeline", "stratum"))[YIDX[j]]
        nn = n.xs((pipe, st), level=("pipeline", "stratum")); return ((s0 - f) / nn).loc[(k, p)].values
    II, GT = [], []
    for p in C["ps"]:
        for k in C["kappas"]:
          try:
            cgB = {j: CG("Bfull", j, k, p) for j in (1, 2)}; cgF = {j: CG("Fact8", j, k, p) for j in (1, 2)}; djf = {j: cgB[j] - cgF[j] for j in (1, 2)}
            s6 = lambda pipe: np.sqrt(W.xs((pipe, "all"), level=("pipeline", "stratum"))[list(range(6))].sum(1) / (6 * n.xs((pipe, "all"), level=("pipeline", "stratum")))).loc[(k, p)].values
            cert = {nm: (CERT(pipe, 1, k, p) + CERT(pipe, 2, k, p)) / 2 for nm, pipe in (("C_Bfull", "O_lat:Bfull"), ("C_Fact8", "O_lat:Fact8"), ("C_Oraw", "O_raw"),
                                                                                      ("C_R1Bfull", "R1:Bfull"))}
            cert["C_R1BfullSham"] = (CERT("R1:BfullSham1", 1, k, p) + CERT("R1:BfullSham2", 2, k, p)) / 2
            vals = {"CG_Bfull": (cgB[1] + cgB[2]) / 2, "CG_Fact8": (cgF[1] + cgF[2]) / 2, "Delta_JF": (djf[1] + djf[2]) / 2,
                    "D_s6_Bfull_minus_Fact8": s6("R1:Bfull") - s6("R1:Fact8"), "D_s6_Bfull_minus_Fact16": s6("R1:Bfull") - s6("R1:Fact16"), **cert}
            for nm, v in vals.items():
                II.append(dict(kappa=k, p=p, contrast=nm, **bounds(v, C["alpha_IIa"]), sign_p=sign_test(v)))
            if k == 0.0:
                GT.append(dict(p=p, family="IIa", **gate(cgB, vals["CG_Bfull"]))); GT.append(dict(p=p, family="IIb", **gate(djf, vals["Delta_JF"])))
          except KeyError:
            say(f"Family II k={k} p={p}: cell missing -> 미실행")
    II = pd.DataFrame(II); II.to_csv(f"{OUTDIR}/p8_familyII_all.csv", index=False); GT = pd.DataFrame(GT); GT.to_csv(f"{OUTDIR}/p8_gates.csv", index=False)
    say(GT[["p", "family", "status", "sign_p_j1", "sign_p_j2", "tost_lo", "tost_hi"]].to_string(index=False))
    pick = lambda k, nm: II[(II.kappa == k) & (II.p == p7) & (II.contrast == nm)].iloc[0]
    seq = []
    for fam, steps in (("IIa", [("IIa-1", "CG_Bfull", "ucb<0"), ("IIa-2", "C_Bfull", "lcb>0")]),
                       ("IIb", [("IIb-1", "Delta_JF", "ucb<0"), ("IIb-2", "D_s6_Bfull_minus_Fact8", "ucb<0")])):
        g = GT[(GT.p == p7) & (GT.family == fam)].iloc[0].status; go = True
        for step, nm, rule in steps:
            r_ = pick(1.0, nm); det = bool(r_.ucb < 0) if rule == "ucb<0" else bool(r_.lcb > 0); det = det and (1.0, p7) not in bad
            seq.append(dict(step=step, contrast=nm, gate=g, label=LABEL[g], tested=go, detected=det, confirmed=bool(go and det and g == "PASS"),
                            mean=r_["mean"], lcb=r_.lcb, ucb=r_.ucb, d_z=r_.d_z, n_neg=r_.n_neg,
                            practical=(r_["mean"] <= -(C["delta_6"] if nm.startswith("D_s6") else C["delta_y"])) if nm != "C_Bfull" else None))   # one-sided direction
            go = go and det
    SQ = pd.DataFrame(seq); SQ.to_csv(f"{OUTDIR}/p8_familyII_sequence.csv", index=False); say(SQ.to_string(index=False))
    # ================= GSR descriptive criteria (5.5; NOT confirmatory)
    lat = load_sums("lat"); rho = lat.groupby(["kappa", "p", "rep", "feat"]).agg(num=("rho_num", "mean"), den=("rho_den", "first")).reset_index()
    rho["rho"] = rho.num / rho.den; rho = rho.drop(columns=["num", "den"])
    rho.to_csv(f"{OUTDIR}/p8_rho_phi.csv", index=False)
    nnr = lat.dropna(subset=["nn_state_dist"]).groupby(["kappa", "p", "feat"])[["nn_state_dist", "rand_state_dist"]].mean(); nnr.to_csv(f"{OUTDIR}/p8_latent_nn.csv")
    gsr = []
    for k in C["kappas"]:
        rp = rho[(rho.kappa == k) & (rho.p == p7)].pivot(index="rep", columns="feat", values="rho"); dr = bounds((rp.Bfull - rp.Fact8).values, .025)
        c_ok = all(D[(D.kappa == k) & (D.stratum == s) & (D.contrast == "G_tot")].iloc[0].ucb < C["delta_I"] for s in ("both_L2+", "one_L2+", "H"))
        gsr.append(dict(kappa=k, p=p7, a_IIa2_certified=bool(k == 1.0 and SQ[SQ.step == "IIa-2"].iloc[0].confirmed),
                        b_IIb1=bool(k == 1.0 and SQ[SQ.step == "IIb-1"].iloc[0].confirmed), c_all_joint_strata_Gtot_ucb_lt_deltaI=c_ok,
                        d_rho_Bfull_lt_Fact8=bool(dr["ci95_hi"] < 0), rho_diff_mean=dr["mean"],
                        note="descriptive; even all True = empirical evidence compatible with GSR, not injectivity/diffeomorphism"))
    pd.DataFrame(gsr).to_csv(f"{OUTDIR}/p8_GSR_descriptive.csv", index=False); say("\n== GSR descriptive (not confirmatory)"); say(pd.DataFrame(gsr).to_string(index=False))
    # ================= Family III (imputation; descriptive)
    imp = load_sums("imp"); ti = imp.groupby(["kappa", "p", "rep", "method", "view", "stratum"])[["sse", "n"]].sum().reset_index()
    ti["rmse"] = np.sqrt(ti.sse / ti.n.replace(0, np.nan)); ti.groupby(["kappa", "p", "method", "view", "stratum"]).rmse.mean().to_csv(f"{OUTDIR}/p8_imputation.csv")
    rel = []
    for (k, p), y in ti[ti.stratum == "all"].groupby(["kappa", "p"]):
        for meth, pipe in (("AE:Bfull", "R1:Bfull"), ("AE:Fact8", "R1:Fact8"), ("AE:Fact16", "R1:Fact16")):
            g_ = y[y.method == meth].groupby("rep")[["sse", "n"]].sum(); a_ = np.sqrt(g_.sse / g_.n)
            b_ = np.sqrt(R(W, n, pipe, "all").loc[(k, p)]).reindex(a_.index)
            rho_s, pv = stats.spearmanr(a_.values, b_.values) if len(a_) > 2 else (np.nan, np.nan)
            rel.append(dict(kappa=k, p=p, method=meth, spearman_imp_vs_rmse_s6=rho_s, p_value=pv))
    pd.DataFrame(rel).to_csv(f"{OUTDIR}/p8_imputation_vs_state.csv", index=False)
    p7f = sorted(glob.glob(f"{EVAL}/p7imp_*.csv.gz"))
    if p7f:
        P7 = pd.concat([pd.read_csv(f) for f in p7f]); guard(P7, "p7imp"); g7 = P7.groupby(["kappa", "p", "rep", "method", "view", "stratum"])[["sse", "n"]].sum().reset_index()
        g7["rmse"] = np.sqrt(g7.sse / g7.n.replace(0, np.nan)); g7.groupby(["kappa", "p", "method", "view", "stratum"]).rmse.mean().to_csv(f"{OUTDIR}/p8_p7_imputation.csv")
    say(f"\nANALYZE8 DONE {status}")
    open(f"{OUTDIR}/p8_summary.txt", "w").write("\n".join(OUT) + "\n")
    os.replace(OUTDIR, final); OUTDIR = final
    open(f"{ANA}/LATEST.tmp", "w").write(os.path.basename(final) + "\n"); os.replace(f"{ANA}/LATEST.tmp", f"{ANA}/LATEST")
    print("ANALYZE8 WROTE", os.path.basename(final))


if __name__ == "__main__":
    main()
