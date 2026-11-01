#!/usr/bin/env python
"""Phase 8 post-run INDEPENDENT verification (audit 2026-10-31).

Independence contract
  * Imports NO project module (experiments.*, src.*). Only numpy / pandas / scipy (t quantiles, Spearman p reference) / torch
    (ONLY to deserialise the stored O_raw state dicts; the forward pass is re-implemented in numpy).
  * Everything is re-derived from protocol v2 definitions (results/notes/phase8_mathematical_protocol_v2.md):
      - seeds: seed_of(*parts) = int(sha256(repr(parts))[:8], 16); MCAR mask = rng(seed).random(shape) >= p   (protocol 1.1 / 4.1)
      - normalisation constants: mean / SD (ddof 0) of the Phase 6 pool ids 900..903 (protocol 1.1)
      - strata from the mask only (protocol 4.3), past-only 8-lag window, n >= SPAN = 7
      - raw window features (64-d) and the 4-layer ReLU MLP forward of O_raw
      - every contrast, bound, exact sign test, TOST, fixed sequence and gate (protocol 4.4-4.5, 5.1-5.5, 6, 7, 9)
  * Reads the replicate-level evaluation sums written by evaluate8 (results/extra/phase8/eval/*), the raw trajectories
    (results/extra/phase8/data, results/extra/phase6/data) and the stored oracles; writes ONLY to --out.

Levels
  L1  statistics re-derived from replicate-level SSE sums (all families) and compared with the analysis directory.
  L2  raw-data re-derivation, no model: masks, stratum counts n, sym0 SSE (T_A / P8.4 / LB inputs), zero and LOCF imputation SSE,
      LB from the raw oracle-pool trajectories -> compared with the eval files (checks masks, strata, normalisation, targets).
  L3  O_raw re-inference (own features + own numpy MLP forward) for p = 0.7, both kappa, all 24 replicates -> compared with the
      eval files (checks one side of G_tot / G_lat end-to-end).  R1 / O_lat sides cannot be re-derived without refitting or
      re-encoding (see reinfer_olat_spotcheck.py for a NON-independent consistency check of O_lat).
usage: python -I verify_p8_independent.py --root /home/minju/faicpaston/Imputing_Missing_Data --out <dir> [--skip-l2] [--skip-l3]
"""
import argparse, glob, hashlib, json, math, os, sys, time
import numpy as np, pandas as pd
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--analysis", default="results/extra/phase8/analysis/complete_20261031-181946_0f246de749")
ap.add_argument("--skip-l2", action="store_true"); ap.add_argument("--skip-l3", action="store_true")
A = ap.parse_args(); os.chdir(A.root); OUT = A.out; os.makedirs(OUT, exist_ok=True)
P8 = "results/extra/phase8"; EV = f"{P8}/eval"; AN = A.analysis
LOG = open(f"{OUT}/verify_log.txt", "w")


def say(*s):
    t = " ".join(str(x) for x in s); print(t, flush=True); LOG.write(t + "\n"); LOG.flush()


# ------------------------------------------------------------------ protocol constants (re-typed from protocol v2, not imported)
KAPPAS, PS, R, NT, NM, NTEST, SPAN, LAGS = (0.0, 1.0), (0.3, 0.7), 24, 8, 5, 5000, 7, 8
DELTA_I, DELTA_Y, DELTA_6, EPS = 0.02, 0.02, 0.05, 0.01
A_I, A_IIA, A_IIB, A_SIGN, A_TOST = 0.025, 0.0125, 0.0125, 0.025, 0.05
OBS = [0, 2, 3, 5]                     # x1 z1 x2 z2 inside s = (x1 y1 z1 x2 y2 z2)
YIDX = {1: 1, 2: 4}                    # y_j coordinate
SUBV = {1: [0, 1], 2: [2, 3]}          # canonical view columns of subsystem j
SYM = {"X0": [0, 1, 3, 4], "L1:L0": [0, 1], "L2:L0": [3, 4]}
CHECKS = []                            # (name, max_abs_diff, tol, ok)


def check(name, a, b, tol):
    a = np.asarray(a, float); b = np.asarray(b, float)
    both_nan = np.isnan(a) & np.isnan(b); d = np.where(both_nan, 0.0, np.abs(a - b))
    m = float(np.nanmax(d)) if d.size else 0.0; ok = bool(np.isfinite(d).all() and m <= tol)
    CHECKS.append(dict(check=name, n=int(d.size), max_abs_diff=m, tol=tol, ok=ok)); say(f"  [{'OK' if ok else 'MISMATCH'}] {name}: n={d.size} max|diff|={m:.3g} (tol {tol:g})")
    return ok


def sha256f(f):
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ statistics (own implementations)
def tbounds(d, alpha):
    d = np.asarray(d, float); n = len(d); m = d.mean(); sd = d.std(ddof=1); se = sd / math.sqrt(n); q = stats.t.ppf(1 - alpha, n - 1)
    return dict(n=n, mean=m, sd=sd, lcb=m - q * se, ucb=m + q * se, d_z=m / sd, n_neg=int((d < 0).sum()), n_pos=int((d > 0).sum()))


def binom_cdf(k, n):
    return sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n


def sign_two_sided(d):
    """exact two-sided sign test vs 0 (zeros dropped), p = min(1, 2 min tail); equals the 'minlike' rule for Binomial(n, 1/2)."""
    d = np.asarray(d, float); d = d[d != 0]; n = len(d); k = int((d > 0).sum())
    if n == 0:
        return 1.0
    return min(1.0, 2 * min(binom_cdf(k, n), 1 - binom_cdf(k - 1, n) if k > 0 else 1.0))


def sign_one_sided_gt(d):
    """exact one-sided sign test H1: median > 0."""
    d = np.asarray(d, float); d = d[d != 0]; n = len(d); k = int((d > 0).sum())
    return 1.0 if n == 0 else (1 - binom_cdf(k - 1, n) if k > 0 else 1.0)


def spearman(a, b):
    ra, rb = stats.rankdata(a), stats.rankdata(b); r = np.corrcoef(ra, rb)[0, 1]; n = len(a)
    t = r * math.sqrt((n - 2) / max(1e-300, 1 - r * r)); return r, 2 * stats.t.sf(abs(t), n - 2)


# ------------------------------------------------------------------ inputs + fingerprints
say(f"== Phase 8 independent verification  {time.strftime('%F %T')}  root={A.root}")
say(f"   python {sys.version.split()[0]}  numpy {np.__version__}  pandas {pd.__version__}")
inputs = sorted(glob.glob(f"{EV}/*") + glob.glob(f"{AN}/*") + glob.glob(f"{P8}/runs/rows/*.csv") + glob.glob(f"{P8}/oracle/*.json"))
with open(f"{OUT}/input_sha256.txt", "w") as fh:
    for f in inputs:
        fh.write(f"{sha256f(f)}  {f}\n")
say(f"   hashed {len(inputs)} input files -> input_sha256.txt")

acc_f = sorted(glob.glob(f"{EV}/k*_acc.csv.gz")); say(f"   acc files: {len(acc_f)} (expected 96)")
ACC = pd.concat([pd.read_csv(f) for f in acc_f], ignore_index=True)
assert not ACC.smoke.astype(str).str.lower().eq("true").any(), "smoke rows present"
g = ACC.groupby(["kappa", "p", "rep", "pipeline", "stratum", "coord"])[["sse", "n"]].sum()
SSE = g.sse.unstack("coord"); NN = g.n.unstack("coord")
assert (NN.nunique(axis=1) == 1).all(), "n differs across coordinates"
NN = NN[0]
cells = ACC.groupby(["kappa", "p"]).agg(reps=("rep", "nunique"), trajs=("traj", "nunique"), masks=("mask", "nunique"))
say("   replicate coverage per cell:\n" + cells.to_string())
ok_cov = bool((cells.reps == 24).all() and (cells.trajs == 192).all() and (cells.masks == 5).all())
CHECKS.append(dict(check="coverage 24 reps x 8 traj x 5 masks per cell", n=4, max_abs_diff=0.0, tol=0, ok=ok_cov))


def sse(k, p, pipe, st):
    return SSE.xs((k, p, pipe, st), level=("kappa", "p", "pipeline", "stratum")).sort_index()


def nn(k, p, pipe, st):
    return NN.xs((k, p, pipe, st), level=("kappa", "p", "pipeline", "stratum")).sort_index()


def risk(k, p, pipe, st, coords=range(6)):
    """coordinate-mean MSE per replicate (pooled over 8 test trajectories x 5 masks) -- protocol 1.1 / 4.4 / 9."""
    c = list(coords); return (sse(k, p, pipe, st)[c].sum(1) / (len(c) * nn(k, p, pipe, st))).values


def rmse_c(k, p, pipe, st, c):
    return np.sqrt((sse(k, p, pipe, st)[c] / nn(k, p, pipe, st)).values)


def rmse6(k, p, pipe, st="all"):
    return np.sqrt(risk(k, p, pipe, st))


ANA = {f: pd.read_csv(f"{AN}/{f}") for f in ["p8_familyI_all.csv", "p8_familyI_sequence.csv", "p8_familyII_all.csv", "p8_familyII_sequence.csv",
                                            "p8_gates.csv", "p8_TA_LB.csv", "p8_GSR_descriptive.csv", "p8_mask_decomposition.csv",
                                            "p8_imputation.csv", "p8_p7_imputation.csv", "p8_imputation_vs_state.csv", "p8_rho_phi.csv",
                                            "p8_latent_nn.csv", "p8_d5b_EXPLORATORY.csv", "p8_knn_vs_oraw.csv", "p8_cells.csv",
                                            "p8_cell_completeness.csv", "p8_training_rows.csv"]}

# ================================================================== L1: Family I
say("\n== L1 Family I (p = 0.7; coordinate-mean MSE; unit = replicate, n = 24; one-sided alpha 0.025)")
FI = []
for k in KAPPAS:
    for st in ["H", "X0", "H_minus_X0", "one_L2+", "both_L2+", "all"]:
        gt = risk(k, .7, "R1:Bfull", st) - risk(k, .7, "O_raw", st); gl = risk(k, .7, "O_lat:Bfull", st) - risk(k, .7, "O_raw", st)
        for nm, d in (("G_tot", gt), ("G_lat", gl), ("G_rd", gt - gl)):
            b = tbounds(d, A_I)
            FI.append(dict(kappa=k, p=.7, stratum=st, contrast=nm, **b, sign2_p_vs_delta=sign_two_sided(d - DELTA_I),
                           sign1_p_gt_delta=sign_one_sided_gt(d - DELTA_I), R_Bfull=risk(k, .7, "R1:Bfull", st).mean(),
                           R_Olat=risk(k, .7, "O_lat:Bfull", st).mean(), R_Oraw=risk(k, .7, "O_raw", st).mean()))
FI = pd.DataFrame(FI); FI.to_csv(f"{OUT}/indep_familyI_all.csv", index=False)
m = ANA["p8_familyI_all.csv"].merge(FI, on=["kappa", "stratum", "contrast"], suffixes=("_a", "_v"))
for c in ["mean", "sd", "lcb", "ucb", "d_z"]:
    check(f"FamilyI_all {c}", m[c + "_a"], m[c + "_v"], 1e-9)
check("FamilyI_all n_neg/n_pos", m.n_neg_a + 1000 * m.n_pos_a, m.n_neg_v + 1000 * m.n_pos_v, 0)
check("FamilyI_all sign p (two-sided, vs delta)", m.sign_p_vs_delta, m.sign2_p_vs_delta, 1e-12)
seq, go = [], True
for step, k, nm in (("I-1", 1.0, "G_tot"), ("I-2", 1.0, "G_lat"), ("I-3", 0.0, "G_tot"), ("I-4", 0.0, "G_lat")):
    r_ = FI[(FI.kappa == k) & (FI.stratum == "H") & (FI.contrast == nm)].iloc[0]; conf = bool(go and r_.lcb > DELTA_I)
    seq.append(dict(step=step, kappa=k, contrast=nm, tested_confirmatory=go, confirmed_gt_delta=conf, reverse_ucb_lt_delta_descr=bool(r_.ucb < DELTA_I),
                    mean=r_["mean"], lcb=r_.lcb, ucb=r_.ucb, sign1_p_gt_delta=r_.sign1_p_gt_delta)); go = conf
SEQ_I = pd.DataFrame(seq); SEQ_I.to_csv(f"{OUT}/indep_familyI_sequence.csv", index=False); say(SEQ_I.to_string(index=False))
a = ANA["p8_familyI_sequence.csv"]
check("FamilyI sequence decisions (tested, confirmed, reverse)", (a.tested_confirmatory.astype(int) * 4 + a.confirmed_gt_delta.astype(int) * 2 + a.descriptive_no_gap_ucb_lt_delta.astype(int)).values,
      (SEQ_I.tested_confirmatory.astype(int) * 4 + SEQ_I.confirmed_gt_delta.astype(int) * 2 + SEQ_I.reverse_ucb_lt_delta_descr.astype(int)).values, 0)

# ---- T_A, LB, info ratio
say("\n== L1/L2 T_A, LB (symmetric strata) -- LB re-derived from the RAW oracle-pool train trajectories")


def norm_consts(k):
    S = np.concatenate([np.load(f"results/extra/phase6/data/k{float(k)}_id{900 + q}.npz")["X"] for q in range(4)])
    return S[:, OBS].mean(0), S[:, OBS].std(0), S.mean(0), S.std(0)


NORM = {k: norm_consts(k) for k in KAPPAS}
with open(f"{OUT}/norm_constants.json", "w") as fh:
    json.dump({str(k): dict(mu_o=v[0].tolist(), sd_o=v[1].tolist(), mu_s=v[2].tolist(), sd_s=v[3].tolist()) for k, v in NORM.items()}, fh, indent=1)


def lb_raw(k, coords):
    sd = NORM[k][3]; v = []
    for q in range(40):
        S = np.load(f"{P8}/data/k{float(k)}_id{60000 + q}.npz")["X"][SPAN:NTEST]
        v.append(sum((S[:, c] ** 2).mean() / sd[c] ** 2 for c in coords) / 6)
    v = np.array(v); m_ = v.mean(); h = stats.t.ppf(.975, 39) * v.std(ddof=1) / math.sqrt(40)
    return dict(LB=m_, LB_lo=m_ - h, LB_hi=m_ + h, n_traj=40, var_match=[float(np.mean([(np.load(f"{P8}/data/k{float(k)}_id{60000 + q}.npz")["X"][SPAN:NTEST, c] ** 2).mean() for q in range(40)]) / sd[c] ** 2) for c in coords])


TA = []
for k in KAPPAS:
    for st, cs in SYM.items():
        if st != "X0" and k != 0.0:
            continue
        lb = lb_raw(k, cs)
        for p in PS:
            for f in ["O_raw", "R1:Bfull", "O_lat:Bfull"]:
                x, n_, s0 = sse(k, p, f, st), nn(k, p, f, st), sse(k, p, "sym0", st); ok = (n_ > 0).values
                if ok.sum() < 2:
                    continue
                tA = ((x[list(range(6))].sum(1) - s0[cs].sum(1)) / (6 * n_)).values[ok]; rA = (x[list(range(6))].sum(1) / (6 * n_)).values[ok]
                b = tbounds(tA, .025)
                TA.append(dict(kappa=k, p=p, stratum=st, pipeline=f, reps_with_points=int(ok.sum()), T_A=tA.mean(), T_A_ci_lo=b["lcb"], T_A_ci_hi=b["ucb"],
                               R_A=rA.mean(), info_ratio_LB_over_R=lb["LB"] / rA.mean(), LB=lb["LB"], LB_lo=lb["LB_lo"], LB_hi=lb["LB_hi"],
                               E_s2_over_sigma2_per_coord=json.dumps([round(z, 5) for z in lb["var_match"]]), points_in_stratum=int(n_.values.sum())))
TA = pd.DataFrame(TA); TA.to_csv(f"{OUT}/indep_TA_LB.csv", index=False); say(TA.round(4).drop(columns=["E_s2_over_sigma2_per_coord"]).to_string(index=False))
m = ANA["p8_TA_LB.csv"].merge(TA, on=["kappa", "p", "stratum", "pipeline"], suffixes=("_a", "_v"))
for c in ["T_A", "T_A_ci_lo", "T_A_ci_hi", "R_A", "info_ratio_LB_over_R", "LB", "LB_lo", "LB_hi"]:
    check(f"TA_LB {c}", m[c + "_a"], m[c + "_v"], 1e-7)
check("TA_LB row count", [len(ANA["p8_TA_LB.csv"])], [len(TA)], 0)

# ================================================================== L1: Family II
say("\n== L1 Family II (y_j RMSE; R1 readout; unit = replicate)")


def CG(F, j, k, p):
    return rmse_c(k, p, f"R1:{F}", "all", YIDX[j]) - rmse_c(k, p, f"R1:{F}Sham{j}", "all", YIDX[j])


def CERT(pipe, j, k, p):
    st = f"L{j}:L0"; c = YIDX[j]; return ((sse(k, p, "sym0", st)[c] - sse(k, p, pipe, st)[c]) / nn(k, p, pipe, st)).values


II, GT, PERJ = [], [], []
for p in PS:
    for k in KAPPAS:
        cgB = {j: CG("Bfull", j, k, p) for j in (1, 2)}; cgF = {j: CG("Fact8", j, k, p) for j in (1, 2)}; djf = {j: cgB[j] - cgF[j] for j in (1, 2)}
        vals = {"CG_Bfull": (cgB[1] + cgB[2]) / 2, "CG_Fact8": (cgF[1] + cgF[2]) / 2, "Delta_JF": (djf[1] + djf[2]) / 2,
                "D_s6_Bfull_minus_Fact8": rmse6(k, p, "R1:Bfull") - rmse6(k, p, "R1:Fact8"), "D_s6_Bfull_minus_Fact16": rmse6(k, p, "R1:Bfull") - rmse6(k, p, "R1:Fact16")}
        for nm, pipe in (("C_Bfull", "O_lat:Bfull"), ("C_Fact8", "O_lat:Fact8"), ("C_Oraw", "O_raw"), ("C_R1Bfull", "R1:Bfull")):
            vals[nm] = (CERT(pipe, 1, k, p) + CERT(pipe, 2, k, p)) / 2
        vals["C_R1BfullSham"] = (CERT("R1:BfullSham1", 1, k, p) + CERT("R1:BfullSham2", 2, k, p)) / 2
        for nm, v in vals.items():
            II.append(dict(kappa=k, p=p, contrast=nm, **tbounds(v, A_IIA), sign_p=sign_two_sided(v)))
        for j in (1, 2):
            for nm, v in (("CG_Bfull", cgB[j]), ("CG_Fact8", cgF[j]), ("Delta_JF", djf[j]), ("C_Bfull", CERT("O_lat:Bfull", j, k, p))):
                PERJ.append(dict(kappa=k, p=p, j=j, contrast=nm, **tbounds(v, A_IIA), sign_p=sign_two_sided(v)))
        if k == 0.0:
            for fam, perj, avg in (("IIa", cgB, vals["CG_Bfull"]), ("IIb", djf, vals["Delta_JF"])):
                sp = {j: sign_two_sided(perj[j]) for j in (1, 2)}; tb = tbounds(avg, A_TOST); eq = bool(tb["lcb"] > -EPS and tb["ucb"] < EPS)
                st_ = "FAIL" if min(sp.values()) < A_SIGN else ("PASS" if eq else "UNDETERMINED")
                GT.append(dict(p=p, family=fam, status=st_, sign_p_j1=sp[1], sign_p_j2=sp[2], tost_lo=tb["lcb"], tost_hi=tb["ucb"], equivalent=eq,
                               avg_mean=tb["mean"], avg_sd=tb["sd"], margin_over_sd=EPS / tb["sd"]))
II = pd.DataFrame(II); II.to_csv(f"{OUT}/indep_familyII_all.csv", index=False); GT = pd.DataFrame(GT); GT.to_csv(f"{OUT}/indep_gates.csv", index=False)
pd.DataFrame(PERJ).to_csv(f"{OUT}/indep_familyII_per_j.csv", index=False)
say(GT.to_string(index=False))
m = ANA["p8_familyII_all.csv"].merge(II, on=["kappa", "p", "contrast"], suffixes=("_a", "_v"))
for c in ["mean", "sd", "lcb", "ucb", "d_z", "sign_p"]:
    check(f"FamilyII_all {c}", m[c + "_a"], m[c + "_v"], 1e-9)
m = ANA["p8_gates.csv"].merge(GT, on=["p", "family"], suffixes=("_a", "_v"))
for c in ["sign_p_j1", "sign_p_j2", "tost_lo", "tost_hi"]:
    check(f"gates {c}", m[c + "_a"], m[c + "_v"], 1e-9)
check("gates status", (m.status_a == m.status_v).astype(int), np.ones(len(m)), 0)
seq = []
for fam, steps in (("IIa", [("IIa-1", "CG_Bfull", "ucb<0"), ("IIa-2", "C_Bfull", "lcb>0")]), ("IIb", [("IIb-1", "Delta_JF", "ucb<0"), ("IIb-2", "D_s6_Bfull_minus_Fact8", "ucb<0")])):
    gs = GT[(GT.p == .7) & (GT.family == fam)].iloc[0].status; go = True
    for step, nm, rule in steps:
        r_ = II[(II.kappa == 1.0) & (II.p == .7) & (II.contrast == nm)].iloc[0]; det = bool(r_.ucb < 0) if rule == "ucb<0" else bool(r_.lcb > 0)
        thr = DELTA_6 if nm.startswith("D_s6") else DELTA_Y
        seq.append(dict(step=step, contrast=nm, gate=gs, tested=go, detected=det, confirmed=bool(go and det and gs == "PASS"), mean=r_["mean"], lcb=r_.lcb, ucb=r_.ucb,
                        practical_mean_beyond_delta=(bool(r_["mean"] <= -thr) if nm != "C_Bfull" else None),
                        practical_bound_beyond_delta=(bool(r_.ucb <= -thr) if nm != "C_Bfull" else None)))
        go = go and det
SEQ_II = pd.DataFrame(seq); SEQ_II.to_csv(f"{OUT}/indep_familyII_sequence.csv", index=False); say(SEQ_II.to_string(index=False))
a = ANA["p8_familyII_sequence.csv"]
check("FamilyII sequence decisions (tested, detected, confirmed)", (a.tested.astype(int) * 4 + a.detected.astype(int) * 2 + a.confirmed.astype(int)).values,
      (SEQ_II.tested.astype(int) * 4 + SEQ_II.detected.astype(int) * 2 + SEQ_II.confirmed.astype(int)).values, 0)

# ================================================================== L1: section 2.3 decomposition, Family IV, GSR
say("\n== L1 mask-variance decomposition (stratum all), Family IV, GSR descriptive criteria")
PAIR = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{EV}/k*_pair.csv.gz"))], ignore_index=True)
dec = []
for (k, p, r, pipe), y in PAIR.groupby(["kappa", "p", "rep", "pipeline"]):
    per_pair = y.groupby("pair").sse.sum() / (y.groupby("pair").n.first() * 6 * y.traj.nunique()); vm = per_pair.mean() / 2
    mse = float(risk(k, p, pipe, "all")[r]); dec.append(dict(kappa=k, p=p, rep=r, pipeline=pipe, mse=mse, mask_var=vm, mask_avg_bias2=mse - vm))
DEC = pd.DataFrame(dec).groupby(["kappa", "p", "pipeline"]).mean(numeric_only=True).drop(columns="rep").reset_index()
DEC["stability_sd2_le_mse"] = DEC.mask_var <= DEC.mse; DEC.to_csv(f"{OUT}/indep_mask_decomposition.csv", index=False)
m = ANA["p8_mask_decomposition.csv"].merge(DEC, on=["kappa", "p", "pipeline"], suffixes=("_a", "_v"))
for c in ["mse", "mask_var", "mask_avg_bias2"]:
    check(f"mask_decomposition {c}", m[c + "_a"], m[c + "_v"], 1e-9)
LAT = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{EV}/k*_lat.csv.gz"))], ignore_index=True)
rho = LAT.groupby(["kappa", "p", "rep", "feat"]).agg(num=("rho_num", "mean"), den=("rho_den", "first")).reset_index(); rho["rho"] = rho.num / rho.den
m = ANA["p8_rho_phi.csv"].merge(rho, on=["kappa", "p", "rep", "feat"], suffixes=("_a", "_v")); check("rho_phi", m.rho_a, m.rho_v, 1e-12)
lnn = LAT.dropna(subset=["nn_state_dist"]).groupby(["kappa", "p", "feat"])[["nn_state_dist", "rand_state_dist"]].mean().reset_index()
lnn["ratio_nn_over_random"] = lnn.nn_state_dist / lnn.rand_state_dist; lnn.to_csv(f"{OUT}/indep_latent_nn.csv", index=False)
m = ANA["p8_latent_nn.csv"].merge(lnn, on=["kappa", "p", "feat"], suffixes=("_a", "_v")); check("latent_nn", m.nn_state_dist_a, m.nn_state_dist_v, 1e-12)
FIV = []
for k in KAPPAS:
    for p in PS:
        rp = rho[(rho.kappa == k) & (rho.p == p)].pivot(index="rep", columns="feat", values="rho")
        for other in ("Fact8", "Fact16"):
            b = tbounds((rp.Bfull - rp[other]).values, .025); FIV.append(dict(kappa=k, p=p, contrast=f"rho_Bfull_minus_{other}", **b,
                                                                             rho_Bfull=rp.Bfull.mean(), rho_other=rp[other].mean()))
FIV = pd.DataFrame(FIV); FIV.to_csv(f"{OUT}/indep_familyIV_rho.csv", index=False); say(FIV.round(5).to_string(index=False))
gsr = []
for k in KAPPAS:
    dr = FIV[(FIV.kappa == k) & (FIV.p == .7) & (FIV.contrast == "rho_Bfull_minus_Fact8")].iloc[0]
    c_ok = {s: bool(FI[(FI.kappa == k) & (FI.stratum == s) & (FI.contrast == "G_tot")].iloc[0].ucb < DELTA_I) for s in ("both_L2+", "one_L2+", "H")}
    gsr.append(dict(kappa=k, p=.7, a_IIa2_certified=bool(k == 1.0 and SEQ_II[SEQ_II.step == "IIa-2"].iloc[0].confirmed),
                    b_IIb1=bool(k == 1.0 and SEQ_II[SEQ_II.step == "IIb-1"].iloc[0].confirmed), c_all=all(c_ok.values()), c_per_stratum=json.dumps(c_ok),
                    d_rho=bool(dr.ucb < 0), rho_diff_mean=dr["mean"]))
GSR = pd.DataFrame(gsr); GSR.to_csv(f"{OUT}/indep_GSR_descriptive.csv", index=False); say(GSR.to_string(index=False))
a = ANA["p8_GSR_descriptive.csv"]
check("GSR flags a,b,c,d", (a.a_IIa2_certified.astype(int) * 8 + a.b_IIb1.astype(int) * 4 + a.c_all_joint_strata_Gtot_ucb_lt_deltaI.astype(int) * 2 + a.d_rho_Bfull_lt_Fact8.astype(int)).values,
      (GSR.a_IIa2_certified.astype(int) * 8 + GSR.b_IIb1.astype(int) * 4 + GSR.c_all.astype(int) * 2 + GSR.d_rho.astype(int)).values, 0)
check("GSR rho_diff_mean", a.rho_diff_mean, GSR.rho_diff_mean, 1e-12)

# ================================================================== L1: Family III
say("\n== L1 Family III (missing-position RMSE, normalised units; descriptive)")
IMP = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{EV}/k*_imp.csv.gz"))], ignore_index=True)
ti = IMP.groupby(["kappa", "p", "rep", "method", "view", "stratum"])[["sse", "n"]].sum().reset_index(); ti["rmse"] = np.sqrt(ti.sse / ti.n.replace(0, np.nan))
pv = ti.groupby(["kappa", "p", "method", "view", "stratum"]).rmse.mean().reset_index()
m = ANA["p8_imputation.csv"].merge(pv, on=["kappa", "p", "method", "view", "stratum"], suffixes=("_a", "_v")); check("imputation per-view rmse", m.rmse_a, m.rmse_v, 1e-12)
check("imputation row count", [len(ANA["p8_imputation.csv"])], [len(pv)], 0)
allv = ti[ti.stratum == "all"]
pooled = allv.groupby(["kappa", "p", "rep", "method"])[["sse", "n"]].sum(); pooled["rmse_pooled"] = np.sqrt(pooled.sse / pooled.n)
simple = allv.groupby(["kappa", "p", "rep", "method"]).agg(rmse_simple_mean=("rmse", "mean"), n_views=("view", "nunique"))
T3 = pooled.join(simple).reset_index(); T3.to_csv(f"{OUT}/indep_familyIII_rep_level.csv", index=False)
S3 = T3.groupby(["kappa", "p", "method"]).agg(n_views=("n_views", "first"), rmse_pooled=("rmse_pooled", "mean"), rmse_pooled_sd=("rmse_pooled", "std"),
                                             rmse_simple_mean=("rmse_simple_mean", "mean")).reset_index()
S3.to_csv(f"{OUT}/indep_familyIII_summary.csv", index=False); say(S3.round(4).to_string(index=False))
# paired per-view / per-stratum contrasts AE:Fact8 - AE:Bfull etc. (descriptive, 95% two-sided)
D3 = []
for (k, p), y in ti.groupby(["kappa", "p"]):
    for a_, b_ in (("AE:Fact8", "AE:Bfull"), ("AE:Fact16", "AE:Bfull"), ("LOCF", "AE:Bfull"), ("O_imp", "AE:Bfull"), ("AE:BfullSham1", "AE:Bfull"), ("AE:BfullSham2", "AE:Bfull")):
        for (v, st), z in y.groupby(["view", "stratum"]):
            pa = z[z.method == a_].set_index("rep").rmse.sort_index(); pb = z[z.method == b_].set_index("rep").rmse.sort_index()
            if len(pa) == 24 and len(pb) == 24 and pa.notna().all() and pb.notna().all():
                b = tbounds((pa - pb).values, .025); D3.append(dict(kappa=k, p=p, contrast=f"{a_} - {b_}", view=v, stratum=st, **b))
        pa = T3[(T3.kappa == k) & (T3.p == p) & (T3.method == a_)].set_index("rep").rmse_pooled.sort_index()
        pb = T3[(T3.kappa == k) & (T3.p == p) & (T3.method == b_)].set_index("rep").rmse_pooled.sort_index()
        if len(pa) == 24 and T3[(T3.kappa == k) & (T3.p == p) & (T3.method == a_)].n_views.iloc[0] == 4:
            b = tbounds((pa - pb).values, .025); D3.append(dict(kappa=k, p=p, contrast=f"{a_} - {b_}", view="pooled4", stratum="all", **b))
D3 = pd.DataFrame(D3); D3.to_csv(f"{OUT}/indep_familyIII_paired.csv", index=False)
say(D3[(D3.stratum == "all")].round(4).to_string(index=False))
rel = []
for (k, p), y in allv.groupby(["kappa", "p"]):
    for meth, pipe in (("AE:Bfull", "R1:Bfull"), ("AE:Fact8", "R1:Fact8"), ("AE:Fact16", "R1:Fact16")):
        g_ = y[y.method == meth].groupby("rep")[["sse", "n"]].sum().sort_index(); a_ = np.sqrt(g_.sse / g_.n).values; b_ = rmse6(k, p, pipe)
        rs, pv_ = spearman(a_, b_); rel.append(dict(kappa=k, p=p, method=meth, spearman=rs, p_value=pv_))
REL = pd.DataFrame(rel); REL.to_csv(f"{OUT}/indep_imputation_vs_state.csv", index=False)
m = ANA["p8_imputation_vs_state.csv"].merge(REL, on=["kappa", "p", "method"]); check("imputation_vs_state spearman", m.spearman_imp_vs_rmse_s6, m.spearman, 1e-12)
P7 = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{EV}/p7imp_*.csv.gz"))], ignore_index=True)
g7 = P7.groupby(["kappa", "p", "rep", "method", "view", "stratum"])[["sse", "n"]].sum().reset_index(); g7["rmse"] = np.sqrt(g7.sse / g7.n.replace(0, np.nan))
p7v = g7.groupby(["kappa", "p", "method", "view", "stratum"]).rmse.mean().reset_index()
m = ANA["p8_p7_imputation.csv"].merge(p7v, on=["kappa", "p", "method", "view", "stratum"], suffixes=("_a", "_v")); check("P7 imputation per-view rmse", m.rmse_a, m.rmse_v, 1e-12)
a7 = g7[g7.stratum == "all"]; p7p = a7.groupby(["kappa", "p", "rep", "method"])[["sse", "n"]].sum(); p7p["rmse_pooled"] = np.sqrt(p7p.sse / p7p.n)
p7s = a7.groupby(["kappa", "p", "rep", "method"]).agg(rmse_simple_mean=("rmse", "mean"), n_views=("view", "nunique"))
P7S = p7p.join(p7s).reset_index().groupby(["kappa", "p", "method"]).agg(n_views=("n_views", "first"), rmse_pooled=("rmse_pooled", "mean"),
                                                                        rmse_simple_mean=("rmse_simple_mean", "mean")).reset_index()
P7S.to_csv(f"{OUT}/indep_P7_imputation_summary.csv", index=False); say(P7S.round(4).to_string(index=False))

# ================================================================== L1: exploratory D5b, kNN, saturation; training rows
say("\n== L1 exploratory (D5b, kNN) and training-cell checks")
d5 = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{EV}/d5b_k*.csv"))]).groupby(["kappa", "p", "stratum"]).d5b.agg(["mean", "std", "count"]).reset_index()
m = ANA["p8_d5b_EXPLORATORY.csv"].merge(d5, on=["kappa", "p", "stratum"], suffixes=("_a", "_v")); check("D5b mean", m.mean_a, m.mean_v, 1e-12); check("D5b count", m.count_a, m.count_v, 0)
d5.to_csv(f"{OUT}/indep_d5b_EXPLORATORY.csv", index=False)
K = pd.concat([pd.read_csv(f) for f in glob.glob(f"{EV}/knn_k*.csv") if "ksel" not in f]).groupby(["kappa", "p", "rep", "pipeline", "stratum"]).agg(sse=("sse", "sum"), n=("n", "sum"))
K["mse"] = K.sse / K.n.replace(0, np.nan); KP = K.reset_index().pivot_table(index=["kappa", "p", "stratum"], columns="pipeline", values="mse").reset_index()
m = ANA["p8_knn_vs_oraw.csv"].merge(KP, on=["kappa", "p", "stratum"], suffixes=("_a", "_v")); check("kNN vs O_raw (kNN)", m.kNN_a, m.kNN_v, 1e-12); check("kNN vs O_raw (O_raw)", m.O_raw_a, m.O_raw_v, 1e-12)
KP["kNN_minus_Oraw"] = KP.kNN - KP.O_raw; KP.to_csv(f"{OUT}/indep_knn_vs_oraw.csv", index=False)
ksel = pd.concat([pd.read_csv(f) for f in glob.glob(f"{EV}/knn_ksel_k*.csv")]); ksel.to_csv(f"{OUT}/knn_k_selection.csv", index=False)
rows = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{P8}/runs/rows/*.csv"))], ignore_index=True)
cell = rows.groupby(["kappa", "p"]).agg(n=("exp_id", "count"), cap_rate=("hit_cap", "mean"), nonfinite=("nonfinite_steps", "sum"), finite=("finite_params", "all"),
                                        valid=("valid", "all"), uniq_reps=("rep", "nunique"), uniq_arms=("arm", "nunique"),
                                        code_md5=("code_md5", lambda s: ",".join(sorted(set(s)))), epochs_mean=("epochs_run", "mean"), epochs_max=("epochs_run", "max")).reset_index()
cell.to_csv(f"{OUT}/indep_training_cells.csv", index=False); say(cell.to_string(index=False))
seed_ok = bool((rows.seed == rows.arm.map(lambda a: 8000 if a.startswith("Bfull") else (8100 if a.endswith("_8") or a.endswith("_8d") else 8200)) + rows.rep).all())
params_ok = bool((rows.n_params == rows.arm.map(lambda a: 57840 if a.startswith("Bfull") else (27944 if "_8" in a else 31552))).all())
lat_ok = bool((rows.latent == rows.arm.map(lambda a: 16 if (a.startswith("Bfull") or "_16" in a) else 8)).all())
dup = int(rows.duplicated(["kappa", "p", "rep", "arm"]).sum())
CHECKS.append(dict(check="training rows: 864, unique (k,p,r,arm), seeds = seed0 + r, params/latent per protocol 5.2, all valid, no cap",
                   n=len(rows), max_abs_diff=float(dup), tol=0, ok=bool(len(rows) == 864 and dup == 0 and seed_ok and params_ok and lat_ok and rows.valid.all() and not rows.hit_cap.any())))
say(f"  training rows {len(rows)} dup={dup} seed_ok={seed_ok} params_ok={params_ok} latent_ok={lat_ok} gpu(cuda_visible) counts={rows.cuda_visible.value_counts().to_dict()}")
rows.groupby(["kappa", "p", "arm"]).agg(best_val=("best_val", "mean"), epochs=("epochs_run", "mean"), best_epoch=("best_epoch", "mean")).reset_index().to_csv(f"{OUT}/indep_training_by_arm.csv", index=False)


# ================================================================== L2: raw-data re-derivation (no model)
def seed_of(*parts):
    return int(hashlib.sha256(repr(parts).encode()).hexdigest()[:8], 16)


def mcar(shape, p, seed):
    return (np.random.default_rng(seed).random(shape) >= p).astype(np.float32)


def strata_own(M):
    """protocol 4.3, written from the definition: L_j = # observed x_j in window n-7..n; n_obs = all observed entries; n >= 7."""
    T = len(M); c = np.vstack([np.zeros((1, 4)), np.cumsum(M, 0)]); W = np.full((T, 4), -1.0); W[SPAN:] = c[SPAN + 1:] - c[:T - SPAN]   # sum of M[n-7..n] = c[n+1] - c[n-7]
    ok = np.zeros(T, bool); ok[SPAN:] = True; L1, L2, nob = W[:, 0], W[:, 2], W.sum(1)
    G = {"all": ok, "both_L2+": ok & (L1 >= 2) & (L2 >= 2), "one_L2+": ok & ((L1 >= 2) != (L2 >= 2)), "H": ok & (L1 <= 1) & (L2 <= 1),
         "X0": ok & (L1 == 0) & (L2 == 0)}
    G["H_minus_X0"] = G["H"] & ~G["X0"]
    for j, L in ((1, L1), (2, L2)):
        G[f"L{j}:L0"], G[f"L{j}:L1"], G[f"L{j}:L2+"] = ok & (L == 0), ok & (L == 1), ok & (L >= 2)
    G["nobs<6"], G["nobs6-9"], G["nobs10-13"], G["nobs>=14"] = ok & (nob <= 5), ok & (nob >= 6) & (nob <= 9), ok & (nob >= 10) & (nob <= 13), ok & (nob >= 14)
    return G


def locf_own(O, M):
    out = np.zeros_like(O)
    for c in range(4):
        last = -1
        idx = np.where(M[:, c] > 0)[0]
        # prev observed strictly before n
        pos = np.searchsorted(idx, np.arange(len(O)), side="left") - 1
        has = pos >= 0; out[has, c] = O[idx[pos[has]], c]
    return out


if not A.skip_l2:
    say("\n== L2 raw-data re-derivation: masks (own seed_of / MCAR), strata n, sym0 SSE, zero / LOCF imputation SSE")
    acc_sym = ACC[ACC.pipeline == "sym0"].set_index(["kappa", "p", "rep", "traj", "mask", "stratum", "coord"]).sort_index()
    impB = IMP[IMP.method.isin(["zero", "LOCF"])].set_index(["kappa", "p", "rep", "method", "view", "traj", "mask", "stratum"]).sort_index()
    R_acc, R_imp, mask_frac, test_hash = [], [], [], {}
    for k in KAPPAS:
        mu_o, sd_o, mu_s, sd_s = NORM[k]
        for r in range(R):
            for i in range(NT):
                tid = 50200 + 10 * r + i; X = np.load(f"{P8}/data/k{float(k)}_id{tid}.npz")["X"]; test_hash[(k, tid)] = hashlib.sha256(X.tobytes()).hexdigest()
                O = ((X[:, OBS] - mu_o) / sd_o).astype(np.float32); S = ((X - mu_s) / sd_s).astype(np.float32); E0 = (S - (-mu_s / sd_s).astype(np.float32)) ** 2
                for p in PS:
                    for e in range(NM):
                        M = mcar((NTEST, 4), p, seed_of("p8test", k, p, r, i, e)); G = strata_own(M); mask_frac.append(dict(kappa=k, p=p, frac_obs=M.mean()))
                        for st, sel in G.items():
                            s_ = E0[sel].sum(0)
                            for c in range(6):
                                R_acc.append((k, p, r, tid, e, st, c, float(s_[c]), int(sel.sum())))
                        Lc = locf_own(O, M)
                        for meth, Y in (("zero", np.zeros_like(O)), ("LOCF", Lc)):
                            for c in range(4):
                                j = 1 if c in SUBV[1] else 2; miss = (M[:, c] == 0); miss[:SPAN] = False
                                for st in ("all", f"L{j}:L0", f"L{j}:L1", f"L{j}:L2+", "nobs<6", "nobs6-9", "nobs10-13", "nobs>=14"):
                                    sel = miss & G[st]; R_imp.append((k, p, r, meth, c, tid, e, st, float(((Y[sel, c] - O[sel, c]) ** 2).sum()), int(sel.sum())))
    RA = pd.DataFrame(R_acc, columns=["kappa", "p", "rep", "traj", "mask", "stratum", "coord", "sse", "n"]).set_index(["kappa", "p", "rep", "traj", "mask", "stratum", "coord"]).sort_index()
    RI = pd.DataFrame(R_imp, columns=["kappa", "p", "rep", "method", "view", "traj", "mask", "stratum", "sse", "n"]).set_index(
        ["kappa", "p", "rep", "method", "view", "traj", "mask", "stratum"]).sort_index()
    j_ = acc_sym.join(RA, rsuffix="_v", how="outer"); check("L2 strata point counts n (all strata, 96 cells x 40 masks)", j_.n, j_.n_v, 0)
    check("L2 sym0 SSE (relative)", (j_.sse / j_.sse_v.where(j_.sse_v > 0, 1.0)).where(j_.sse_v > 0, 1.0), np.ones(len(j_)), 1e-4)
    j_ = impB.join(RI, rsuffix="_v", how="outer"); check("L2 imputation n (zero, LOCF)", j_.n, j_.n_v, 0)
    check("L2 imputation SSE zero/LOCF (relative)", (j_.sse / j_.sse_v.where(j_.sse_v > 0, 1.0)).where(j_.sse_v > 0, 1.0), np.ones(len(j_)), 1e-4)
    MF = pd.DataFrame(mask_frac).groupby(["kappa", "p"]).frac_obs.agg(["mean", "min", "max"]); say("  observed fraction of regenerated test masks:\n" + MF.to_string())
    # design probabilities vs empirical stratum frequencies (p = 0.7)
    emp = RA.xs(0, level="coord").groupby(["kappa", "p", "stratum"]).n.sum() / RA.xs(0, level="coord").xs("all", level="stratum").groupby(["kappa", "p"]).n.sum()
    q = 0.7; Lp = lambda L: math.comb(8, L) * (1 - q) ** L * q ** (8 - L)
    design = {"X0": q ** 16, "H": (Lp(0) + Lp(1)) ** 2, "L1:L0": q ** 8}
    say("  stratum frequency p=0.7 (empirical vs design): " + json.dumps({s: dict(design=round(v, 5), k0=round(float(emp.loc[(0.0, .7, s)]), 5), k1=round(float(emp.loc[(1.0, .7, s)]), 5)) for s, v in design.items()}))
    pd.Series(test_hash).to_csv(f"{OUT}/test_traj_sha256.csv")

# ================================================================== L3: O_raw re-inference (own features + numpy forward)
if not A.skip_l3:
    import torch
    say("\n== L3 O_raw re-inference, p = 0.7 (own raw-window features, numpy MLP forward from the stored state dict)")

    def raw_window(O, M):
        """64-d past-only window: for view c (x1 z1 x2 z2) and lag k = 0..7: M[n-k] * O[n-k] (0 if n-k < 0), then the mask bits;
        order = [values (view-major, lag-minor), masks (view-major, lag-minor)]."""
        T = len(O); V = np.zeros((T, 4, LAGS), np.float32); MV = np.zeros((T, 4, LAGS), np.float32)
        for kk in range(LAGS):
            V[kk:, :, kk] = (O * M)[:T - kk]; MV[kk:, :, kk] = M[:T - kk]
        return np.concatenate([(V * MV).reshape(T, -1), MV.reshape(T, -1)], 1)

    rows3 = []
    for k in KAPPAS:
        mu_o, sd_o, mu_s, sd_s = NORM[k]; o = torch.load(f"{P8}/oracle/raw_k{float(k)}_p0.7.pt", map_location="cpu", weights_only=False)
        Wt = [(o["state"][f"{l}.weight"].numpy().T.astype(np.float32), o["state"][f"{l}.bias"].numpy().astype(np.float32)) for l in (0, 2, 4, 6)]
        fmu, fsd = o["mu"].astype(np.float32), o["sd"].astype(np.float32); say(f"  kappa={k}: O_raw width {o['width']} din {o['din']} dout {o['dout']} pool-val risk {o['val_risk']:.5f}")
        for r in range(R):
            for i in range(NT):
                tid = 50200 + 10 * r + i; X = np.load(f"{P8}/data/k{float(k)}_id{tid}.npz")["X"]
                O = ((X[:, OBS] - mu_o) / sd_o).astype(np.float32); S = ((X - mu_s) / sd_s).astype(np.float32)
                for e in range(NM):
                    M = mcar((NTEST, 4), 0.7, seed_of("p8test", k, 0.7, r, i, e)); h = (raw_window(O, M) - fmu) / fsd
                    for li, (w, b) in enumerate(Wt):
                        h = h @ w + b; h = np.maximum(h, 0) if li < 3 else h
                    E2 = (h - S) ** 2
                    for st, sel in strata_own(M).items():
                        s_ = E2[sel].sum(0)
                        for c in range(6):
                            rows3.append((k, 0.7, r, tid, e, st, c, float(s_[c])))
    R3 = pd.DataFrame(rows3, columns=["kappa", "p", "rep", "traj", "mask", "stratum", "coord", "sse"]).set_index(["kappa", "p", "rep", "traj", "mask", "stratum", "coord"]).sort_index()
    ref = ACC[(ACC.pipeline == "O_raw") & (ACC.p == 0.7)].set_index(["kappa", "p", "rep", "traj", "mask", "stratum", "coord"]).sort_index()
    j_ = ref.join(R3, rsuffix="_v", how="inner"); check("L3 O_raw SSE re-inference vs eval (relative, p=0.7, all strata)",
                                                       (j_.sse / j_.sse_v.where(j_.sse_v > 0, 1.0)).where(j_.sse_v > 1e-6, 1.0), np.ones(len(j_)), 2e-3)
    check("L3 O_raw rows matched", [len(j_)], [len(ref)], 0)
    g3 = R3.groupby(["kappa", "rep", "stratum", "coord"]).sse.sum().unstack("coord")
    for k in KAPPAS:
        nH = nn(k, .7, "O_raw", "H").values; vH = (g3.xs((k, "H"), level=("kappa", "stratum")).sort_index().sum(1).values) / (6 * nH)
        check(f"L3 R_H(O_raw) per replicate kappa={k}", risk(k, .7, "O_raw", "H"), vH, 1e-5)
        gt_v = risk(k, .7, "R1:Bfull", "H") - vH; b = tbounds(gt_v, A_I); say(f"  kappa={k} G_tot(H) with re-inferred O_raw: mean {b['mean']:.6f} LCB {b['lcb']:.6f}")

# ================================================================== summary
CK = pd.DataFrame(CHECKS); CK.to_csv(f"{OUT}/verification_checks.csv", index=False)
say(f"\n== CHECKS: {int(CK.ok.sum())}/{len(CK)} OK"); say(CK[~CK.ok].to_string(index=False) if (~CK.ok).any() else "   no mismatches")
say(f"== DONE {time.strftime('%F %T')}")
