"""Phase 3D-1 error decomposition (CPU forward passes of the 8 Phase 3C checkpoints; no training).
 2.1 observed vs missing lag-0 error on the DATA mask (same xhat as Phase 3C; synthetic truth known everywhere).
 2.2 same-state two-mask decomposition (exactly the Phase 3C consistency masks: Generator(4242+seed), p=0.3, test split).
 3   FDR lag-wise error on the data mask + target-centric combination of lags (diagnostic, not a protocol change).
Reproduction checks against Phase 3C files are written to repro_checks.csv. Outputs go to phase3d1_audit/ only."""
import numpy as np, pandas as pd, torch
from common import RUNS, X3, P3C, OUT, P, span, m, tau, load_model, full_outputs, block_boot, rmse_ratio
from experiments.fdr import xz_data
from src.delay_embedding import delay_embed

E21, CTX, E22, LAG, COMB, CHK, BOOT = [], [], [], [], [], [], []
per_time = {}                      # (seed, tag, key) -> (num, den) per test time for paired bootstrap
cons = pd.read_csv(f"{P3C}/eval_consistency.csv")

for _, r in RUNS.sort_values(["seed", "tag"]).iterrows():
    seed = int(r.seed); mdl, c = load_model(r); d, _ = xz_data.build(X3, P, seed, r.vars)
    X, M = torch.tensor(d["X"]), torch.tensor(d["M"]); Xn, Mn, Ev = d["X"].astype(np.float64), d["M"], d["eval_mask"]
    Y, MV = full_outputs(mdl, X, M); xh = Y[..., 0]; n = Xn.shape[1]; names = ["x", "z"][:n]
    saved = np.load(f"{P3C}/{r.exp_id}_xhat_past.npy")
    a, b = d["split"]["test"]; e_te = ((xh[a:b] - Xn[a:b]) ** 2 * Ev[a:b]).sum() / Ev[a:b].sum()
    CHK.append(dict(check="xhat CPU recompute vs saved GPU xhat_past (max abs diff)", run=r.exp_id, value=float(np.abs(xh - saved).max())))
    CHK.append(dict(check="test imputation RMSE recompute - runs.csv", run=r.exp_id, value=float(np.sqrt(e_te) - r.test_rmse)))
    # ---- 2.1 observed / missing / all, per split and variable ----
    for split in ("val", "test"):
        a, b = d["split"][split]; sq = (xh[a:b] - Xn[a:b]) ** 2; Mo, Mi = Mn[a:b], Ev[a:b]
        for vi, vn in list(enumerate(names)) + [(None, "pooled")]:
            sl = slice(None) if vi is None else slice(vi, vi + 1)
            for cat, w in (("observed", Mo), ("missing", Mi), ("all", np.ones_like(Mo))):
                ww = w[:, sl]; E21.append(dict(model=r.tag, seed=seed, vars=r.vars, split=split, var=vn, cat=cat,
                                               n=int(ww.sum()), rmse=float(np.sqrt((sq[:, sl] * ww).sum() / ww.sum()))))
                if split == "test" and vi is None:
                    per_time[(seed, r.tag, cat)] = ((sq * w).sum(1), w.sum(1))
        if split == "test":            # context of missing entries (test)
            nobs_same = MV[a:b, :, 1:].sum(-1)                             # observed same-variable lags 1..7
            for vi, vn in enumerate(names):
                miss = Mi[:, vi] > 0
                strata = [("same-var lags1..7 obs <=3", nobs_same[:, vi] <= 3), ("same-var lags1..7 obs 4-5", (nobs_same[:, vi] >= 4) & (nobs_same[:, vi] <= 5)),
                          ("same-var lags1..7 obs 6-7", nobs_same[:, vi] >= 6)]
                if n == 2:
                    o = 1 - vi; strata += [(f"other var at t observed", Mo[:, o] > 0), (f"other var at t missing", Mo[:, o] == 0)]
                for sn, s in strata:
                    q = miss & s; CTX.append(dict(model=r.tag, seed=seed, vars=r.vars, var=vn, stratum=sn, n=int(q.sum()),
                                                  rmse=float(np.sqrt(sq[q, vi].mean())) if q.any() else np.nan))
    # ---- 3 lag-wise (FDR) on the data mask; C gives k=0 only ----
    for split in ("val", "test"):
        a, b = d["split"][split]; K = Y.shape[-1]
        for k in range(K):
            t = np.arange(a + k, b)                                         # anchor t and target t-k both in split
            pred, tru, mk = Y[t, :, k], Xn[t - k], Mn[t - k]
            nob = MV[t].sum((1, 2))
            for vi, vn in list(enumerate(names)) + [(None, "pooled")]:
                sl = slice(None) if vi is None else slice(vi, vi + 1)
                for cat, w in (("observed", mk), ("missing", 1 - mk), ("all", np.ones_like(mk))):
                    for bn, bs in (("all windows", np.ones(len(t), bool)), ("window obs <=9", nob <= 9), ("window obs 10-12", (nob >= 10) & (nob <= 12)), ("window obs >=13", nob >= 13)):
                        ww = w[bs][:, sl]
                        LAG.append(dict(model=r.tag, seed=seed, vars=r.vars, split=split, var=vn, lag=k, cat=cat, window_bin=bn, n=int(ww.sum()),
                                        rmse=float(np.sqrt((((pred[bs][:, sl] - tru[bs][:, sl]) ** 2) * ww).sum() / max(ww.sum(), 1)))))
        if K > 1:                                                           # target-centric: estimate of x(u) from anchor u+k
            u = np.arange(a, b - (K - 1)); est = np.stack([Y[u + k, :, k] for k in range(K)], -1)   # (U, n, K)
            mi = 1 - Mn[u]
            for lab, e in [(f"lag {k} only (anchor u+{k})", est[..., k]) for k in range(K)] + [("mean over lags 0..7", est.mean(-1)), ("mean over lags 0..3", est[..., :4].mean(-1))]:
                COMB.append(dict(model=r.tag, seed=seed, vars=r.vars, split=split, estimator=lab, n=int(mi.sum()),
                                 rmse_missing=float(np.sqrt((((e - Xn[u]) ** 2) * mi).sum() / mi.sum()))))
    # ---- 2.2 same-state two masks (Phase 3C consistency protocol) ----
    if r.vars == "xz":
        a, b = d["split"]["test"]; Xs = X[a:b]; g = torch.Generator().manual_seed(4242 + seed)
        M1 = (torch.rand(Xs.shape, generator=g) >= P).float(); M2 = (torch.rand(Xs.shape, generator=g) >= P).float()
        outs = []
        for Mq in (M1, M2):
            V, MVq = delay_embed((Xs * Mq)[None], Mq[None], m, tau)
            with torch.no_grad():
                outs.append(mdl.decode(mdl.encoder(V, MVq)[0]).numpy().astype(np.float64))
        o1, o2 = outs; S = Xs[span:].numpy().astype(np.float64); m1, m2 = M1[span:].numpy(), M2[span:].numpy()
        CHK.append(dict(check="same-state dR_q50 recompute - eval_consistency.csv (all)", run=r.exp_id,
                        value=float(np.median(np.linalg.norm(o1 - o2, axis=1)) - cons[(cons.model == r.tag) & (cons.seed == seed) & (cons["bin"] == "all")].dR_q50.iloc[0])))
        e1, e2, dd, eb = o1 - S, o2 - S, o1 - o2, (o1 + o2) / 2 - S
        cats = {"both observed": (m1 > 0) & (m2 > 0), "both missing": (m1 == 0) & (m2 == 0), "first only observed": (m1 > 0) & (m2 == 0), "second only observed": (m1 == 0) & (m2 > 0)}
        for vi, vn in [(0, "x"), (1, "z"), (None, "pooled")]:
            sl = slice(None) if vi is None else slice(vi, vi + 1)
            for cn, cm in cats.items():
                q = cm[:, sl]; f = lambda v: float(np.sqrt((v[:, sl][q] ** 2).mean()))
                E22.append(dict(model=r.tag, seed=seed, var=vn, cat=cn, n=int(q.sum()), rms_diff=f(dd), rms_err1=f(e1), rms_err2=f(e2), rms_err_mean=f(eb),
                                med_abs_diff=float(np.median(np.abs(dd[:, sl][q]))), med_abs_err1=float(np.median(np.abs(e1[:, sl][q]))),
                                identity_resid=float(((e1 ** 2 + e2 ** 2) / 2 - eb ** 2 - dd ** 2 / 4)[:, sl][q].mean())))
            mixed = (cats["first only observed"] | cats["second only observed"])[:, sl]
            eo = np.where(m1[:, sl] > 0, e1[:, sl], e2[:, sl]); em = np.where(m1[:, sl] > 0, e2[:, sl], e1[:, sl])
            E22.append(dict(model=r.tag, seed=seed, var=vn, cat="one observed: observed side", n=int(mixed.sum()), rms_err1=float(np.sqrt((eo[mixed] ** 2).mean())), rms_diff=float(np.sqrt((dd[:, sl][mixed] ** 2).mean()))))
            E22.append(dict(model=r.tag, seed=seed, var=vn, cat="one observed: missing side", n=int(mixed.sum()), rms_err1=float(np.sqrt((em[mixed] ** 2).mean())), rms_diff=float(np.sqrt((dd[:, sl][mixed] ** 2).mean()))))
        # per-time arrays for paired bootstrap (pooled variables)
        mm = cats["both missing"]; oo = cats["both observed"]; mx = ~(mm | oo)
        miss_side = np.where(m1 > 0, e2, e1); obs_side = np.where(m1 > 0, e1, e2)
        per_time[(seed, r.tag, "ss_both_missing_err")] = (((e1 ** 2 + e2 ** 2) / 2 * mm).sum(1), mm.sum(1))
        per_time[(seed, r.tag, "ss_both_missing_diff")] = ((dd ** 2 * mm).sum(1), mm.sum(1))
        per_time[(seed, r.tag, "ss_both_observed_err")] = (((e1 ** 2 + e2 ** 2) / 2 * oo).sum(1), oo.sum(1))
        per_time[(seed, r.tag, "ss_one_obs_missing_side_err")] = ((miss_side ** 2 * mx).sum(1), mx.sum(1))
        per_time[(seed, r.tag, "ss_one_obs_observed_side_err")] = ((obs_side ** 2 * mx).sum(1), mx.sum(1))
        per_time[(seed, r.tag, "ss_all_coords_err")] = (((e1 ** 2 + e2 ** 2) / 2).sum(1), np.full(len(e1), 2.0))
    print("done", r.exp_id, flush=True)

# paired block bootstrap C-xz minus FDR-xz (same test times; per seed; seeds are NOT independent trajectories)
for seed in (0, 1, 2):
    for key in ("observed", "missing", "all", "ss_both_missing_err", "ss_both_missing_diff", "ss_both_observed_err",
                "ss_one_obs_missing_side_err", "ss_one_obs_observed_side_err", "ss_all_coords_err"):
        (nc, dc), (nf, df_) = per_time[(seed, "C-xz", key)], per_time[(seed, "FDR-xz", key)]
        fc, ff = rmse_ratio(nc, dc), rmse_ratio(nf, df_); T = len(nc); allidx = np.arange(T)
        lo, hi = block_boot(lambda i: fc(i) - ff(i), T, seed=seed)
        BOOT.append(dict(seed=seed, metric=key, C=fc(allidx), FDR=ff(allidx), C_minus_FDR=fc(allidx) - ff(allidx), ci_lo=lo, ci_hi=hi))

for nm, R in [("obs_miss_errors", E21), ("obs_miss_context", CTX), ("samestate_decomp", E22), ("fdr_lag_errors", LAG),
              ("fdr_target_combination", COMB), ("repro_checks_errordecomp", CHK), ("paired_bootstrap", BOOT)]:
    pd.DataFrame(R).to_csv(f"{OUT}/{nm}.csv", index=False)
print("ALL DONE")
