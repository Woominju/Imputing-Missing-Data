#!/usr/bin/env python
"""Phase 9 INDEPENDENT verification (imports NO project module; numpy / pandas / scipy only).
L1  re-derives from the replicate-level eval sums: pooled missing-position imputation RMSE, rho_ret, every paired contrast, the
    preregistered fixed sequence P-1..P-4 (or the pilot gate), and compares with the analysis directory.
L2  raw-data re-derivation without any model: regenerates every test mask (own seed_of = sha256(repr(parts))[:8], MCAR rng >= p),
    the Phase 8 strata definition, the fixed normalisation (Phase 6 pool 900..903), and recomputes zero / LOCF imputation SSE and the
    missing-position counts n for every (cell, rep, traj, mask, view, stratum) -> compared with the eval files.
E   execution / integrity: expected outputs, training rows (valid, cap, seeds, lam, tower targets, smoke flag), failure markers, tmp files,
    trajectory identity vs Phase 6-8 data (exact first-state hashes), hidden-y non-use is a code property (tested in test_phase9).
usage: python -I verify9_independent.py --root <project root> --stage {pilot,main} --analysis <analysis dir> --out <dir>"""
import argparse, glob, hashlib, json, math, os, sys
import numpy as np, pandas as pd
from scipy import stats

ap = argparse.ArgumentParser(); ap.add_argument("--root"); ap.add_argument("--stage"); ap.add_argument("--analysis"); ap.add_argument("--out"); A = ap.parse_args()
os.chdir(A.root); os.makedirs(A.out, exist_ok=True); R9 = "results/extra/phase9"; EV = f"{R9}/eval"; ST = A.stage
LOG = open(f"{A.out}/verify9_{ST}_log.txt", "w"); CK = []
say = lambda *s: (print(*s, flush=True), LOG.write(" ".join(map(str, s)) + "\n"))
NREP = {"main": 24, "pilot": 6}[ST]; BASE = {"main": 200000, "pilot": 210000}[ST]; OBS = [0, 2, 3, 5]; SPAN = 7


def check(nm, a, b, tol):
    a, b = np.asarray(a, float), np.asarray(b, float); d = np.where(np.isnan(a) & np.isnan(b), 0, np.abs(a - b)); m = float(np.nanmax(d)) if d.size else 0.0
    ok = bool(np.isfinite(d).all() and m <= tol); CK.append(dict(check=nm, n=int(d.size), max_abs_diff=m, tol=tol, ok=ok)); say(f"  [{'OK' if ok else 'MISMATCH'}] {nm} n={d.size} max={m:.3g}")


def tb(d, a=0.025):
    d = np.asarray(d, float); n = len(d); m = d.mean(); se = d.std(ddof=1) / math.sqrt(n); q = stats.t.ppf(1 - a, n - 1); return dict(n=n, mean=m, lcb=m - q * se, ucb=m + q * se)


def seed_of(*p):
    return int(hashlib.sha256(repr(p).encode()).hexdigest()[:8], 16)


def mcar(shape, p, s):
    return (np.random.default_rng(s).random(shape) >= p).astype(np.float32)


def strata_x(M):
    T = len(M); c = np.vstack([np.zeros((1, 4)), np.cumsum(M, 0)]); W = np.full((T, 4), -1.0); W[SPAN:] = c[SPAN + 1:] - c[:T - SPAN]
    ok = np.zeros(T, bool); ok[SPAN:] = True; nob = W.sum(1)
    G = {"all": ok}
    for j, L in ((1, W[:, 0]), (2, W[:, 2])):
        G[f"L{j}:L0"], G[f"L{j}:L1"], G[f"L{j}:L2+"] = ok & (L == 0), ok & (L == 1), ok & (L >= 2)
    G["nobs<6"], G["nobs6-9"], G["nobs10-13"], G["nobs>=14"] = ok & (nob <= 5), ok & (nob >= 6) & (nob <= 9), ok & (nob >= 10) & (nob <= 13), ok & (nob >= 14)
    return G


def locf(O, M):
    out = np.zeros_like(O)
    for c in range(O.shape[1]):
        idx = np.where(M[:, c] > 0)[0]; pos = np.searchsorted(idx, np.arange(len(O)), "left") - 1; h = pos >= 0; out[h, c] = O[idx[pos[h]], c]
    return out


say(f"== Phase 9 independent verification, stage {ST}")
IMP = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{EV}/{ST}_k*_imp.csv.gz"))]); RET = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{EV}/{ST}_k*_ret.csv.gz"))])
assert not IMP.smoke.astype(str).str.lower().eq("true").any()
a = IMP[IMP.stratum == "all"].groupby(["kappa", "p", "rep", "method"])[["sse", "n"]].sum(); a["rmse"] = np.sqrt(a.sse / a.n); I = a.rmse.unstack("method")
r = RET.groupby(["kappa", "p", "rep", "feat"])[["sum_nn", "sum_rand"]].sum(); r["rho"] = r.sum_nn / r.sum_rand; Rt = r.rho.unstack("feat")
cov = I.groupby(level=["kappa", "p"]).size(); say("  replicates per cell:", cov.to_dict()); CK.append(dict(check="coverage", n=4, max_abs_diff=0, tol=0, ok=bool((cov == NREP).all())))
aI = pd.read_csv(f"{A.analysis}/p9_imputation_rep.csv").set_index(["kappa", "p", "rep"]); aR = pd.read_csv(f"{A.analysis}/p9_retrieval_rep.csv").set_index(["kappa", "p", "rep"])
check("imputation pooled RMSE per rep x method", aI.sort_index().values, I.sort_index()[aI.columns].values, 1e-12)
check("rho_ret per rep x feat", aR.sort_index().values, Rt.sort_index()[aR.columns].values, 1e-12)
P = pd.read_csv(f"{A.analysis}/p9_paired_all_cells.csv"); rows = []
for (k, p), y in I.groupby(level=["kappa", "p"]):
    z = Rt.loc[(k, p)]
    for nm, d in (("imp P9-Bfull", y["AE:P9"] - y["AE:Bfull"]), ("imp P9-Fact8", y["AE:P9"] - y["AE:Fact8"]), ("ret P9-Bfull", z.P9 - z.Bfull), ("ret P9-Fact8", z.P9 - z.Fact8)):
        rows.append(dict(kappa=k, p=p, contrast=nm, **tb(d.values)))
V = pd.DataFrame(rows); V.to_csv(f"{A.out}/verify9_{ST}_paired.csv", index=False); m = P.merge(V, on=["kappa", "p", "contrast"], suffixes=("_a", "_v"))
for c in ("mean", "lcb", "ucb"):
    check(f"paired contrasts {c}", m[c + "_a"], m[c + "_v"], 1e-12)
if ST == "main":
    y = I.loc[(1.0, 0.7)]; z = Rt.loc[(1.0, 0.7)]; go, seq = True, []
    for step, d, thr in (("P-1", y["AE:P9"] - y["AE:Bfull"], 0.0), ("P-2", z.P9 - z.Bfull, 0.0), ("P-3", y["AE:P9"] - y["AE:Fact8"], 0.01), ("P-4", z.P9 - z.Fact8, 0.0)):
        b = tb(d.values); dec = bool(go and b["ucb"] < thr and len(d) == 24); seq.append(dict(step=step, tested=go, decision=dec, **b)); go = dec
    S = pd.DataFrame(seq); S.to_csv(f"{A.out}/verify9_main_sequence.csv", index=False); say(S.to_string(index=False))
    a_ = pd.read_csv(f"{A.analysis}/p9_confirmatory_sequence.csv"); check("confirmatory decisions", a_.decision.astype(int) * 2 + a_.tested.astype(int), S.decision.astype(int) * 2 + S.tested.astype(int), 0)
    for c in ("mean", "ucb"):
        check(f"confirmatory {c}", a_[c], S[c], 1e-12)
# ---- L2 raw-data re-derivation (zero / LOCF / counts)
S6 = {k: np.concatenate([np.load(f"results/extra/phase6/data/k{k}_id{900 + q}.npz")["X"] for q in range(4)]) for k in (0.0, 1.0)}
NORM = {k: (S6[k][:, OBS].mean(0), S6[k][:, OBS].std(0)) for k in S6}
ref = IMP[IMP.method.isin(["zero", "LOCF"])].set_index(["kappa", "p", "rep", "method", "view", "traj", "mask", "stratum"]).sort_index(); out = []
for k in (0.0, 1.0):
    mu, sd = NORM[k]
    for rr in range(NREP):
        for i in range(8):
            tid = BASE + 200 + 10 * rr + i; O = ((np.load(f"{R9}/data/k{k}_id{tid}.npz")["X"][:, OBS] - mu) / sd).astype(np.float32)
            for p in (0.3, 0.7):
                for e in range(5):
                    M = mcar((5000, 4), p, seed_of("p9test", ST, k, p, rr, i, e)); G = strata_x(M); Lc = locf(O, M)
                    for mt, Y in (("zero", np.zeros_like(O)), ("LOCF", Lc)):
                        for c in range(4):
                            j = 1 if c < 2 else 2; miss = M[:, c] == 0; miss[:SPAN] = False
                            for g in ("all", f"L{j}:L0", f"L{j}:L1", f"L{j}:L2+", "nobs<6", "nobs6-9", "nobs10-13", "nobs>=14"):
                                sel = miss & G[g]; out.append((k, p, rr, mt, c, tid, e, g, float(((Y[sel, c] - O[sel, c]) ** 2).sum()), int(sel.sum())))
L2 = pd.DataFrame(out, columns=["kappa", "p", "rep", "method", "view", "traj", "mask", "stratum", "sse", "n"]).set_index(["kappa", "p", "rep", "method", "view", "traj", "mask", "stratum"]).sort_index()
j_ = ref.join(L2, rsuffix="_v", how="outer"); check("L2 missing-position counts n (all strata)", j_.n, j_.n_v, 0)
check("L2 zero/LOCF SSE (relative)", (j_.sse / j_.sse_v.where(j_.sse_v > 0, 1)).where(j_.sse_v > 0, 1), np.ones(len(j_)), 1e-4)
# ---- E integrity
TR = pd.concat([pd.read_csv(f) for f in glob.glob(f"{R9}/runs/rows/{ST}_*.csv")]); so = {"P9": 9000, "Bfull": 9000, "8": 9100, "16": 9200}
off = 0 if ST == "main" else 500; exp_seed = TR.arm.map(lambda a: so.get(a, so.get(a.split("_")[-1]))) + off + TR.rep
CK.append(dict(check="training rows: count, valid, no cap, seeds, lam, tower active, not smoke", n=len(TR), max_abs_diff=0, tol=0,
               ok=bool(len(TR) == 6 * 4 * NREP and TR.valid.all() and not TR.hit_cap.any() and (TR.seed == exp_seed).all() and (TR[TR.arm == "P9"].lam == 1).all()
                       and (TR[TR.arm != "P9"].lam == 0).all() and (TR[TR.arm == "P9"].tower_targets_per_step > 0).all() and not TR.smoke.astype(str).str.lower().eq("true").any())))
tmp = [f for f in glob.glob(f"{R9}/**/*", recursive=True) if ".tmp" in os.path.basename(f)]; fails = glob.glob(f"{R9}/failed/*") + glob.glob(f"{R9}/runs/failed/*")
CK.append(dict(check="no tmp files, no failure markers", n=len(tmp) + len(fails), max_abs_diff=0, tol=0, ok=not tmp and not fails))
prev = set()
for d in ("results/extra/phase6/data", "results/extra/phase7/data", "results/extra/phase8/data"):
    for f in glob.glob(f"{d}/k*_id*.npz"):
        prev.add(np.load(f)["X"][0].tobytes())
mine = [np.load(f)["X"][0].tobytes() for f in glob.glob(f"{R9}/data/k*_id*.npz")]
CK.append(dict(check="Phase 9 trajectories distinct from each other and from all P6-P8 trajectories (first state)", n=len(mine), max_abs_diff=0, tol=0,
               ok=len(set(mine)) == len(mine) and not (set(mine) & prev)))
pd.DataFrame(CK).to_csv(f"{A.out}/verify9_{ST}_checks.csv", index=False); say(pd.DataFrame(CK).to_string(index=False)); say(f"== {sum(c['ok'] for c in CK)}/{len(CK)} OK")
