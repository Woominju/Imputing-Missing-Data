"""AIM latent analysis: structure of Model C's view-wise and shared latents (Exp 0) and their
stability when the missingness mask of the same state is re-sampled (Exp 4). Eval-only, CPU,
saved Lorenz checkpoints; no model, loss or training is added.

Model C as implemented (src/models.py; parameter shapes checked in the checkpoints):
  view encoder E_i : [v_i * mv_i, mv_i] in R^{2m} -> z_i in R^d, separate weights per variable
                     (group_mlp 2m -> 64 -> 64 -> d, GELU); d = latent = 16, m = 8, tau = 1
                     v_i(t) = [x_i(t), x_i(t - tau), ..., x_i(t - (m-1) tau)] (past only)
  fusion           : z = sum_i w_i z_i / max(sum_i w_i, 1), w_i = observed entries of v_i(t)
  decoder D_i      : z -> x_i(t) (current value only), separate weights per variable
Latents are extracted exactly as in ModelC.forward; the decoded values are checked against
src.evaluate.impute on the main evaluation mask.

Exp 0 (the condition is recorded in every row):
  full : every entry of every window observed (no hidden target). Maximal-information reference;
         outside the training mask distribution (training inputs miss p + hide_rate (1 - p)).
  eval : main evaluation mask M (prepare(); test targets hidden); views with w_i = 0 excluded.
  per view: PCA spectrum, participation ratio, k90 / k99, scale. Per view pair (test split):
  same-time distance vs distances between times |t - t'| > THEILER, identity / centred-identity
  R^2 (z_j predicted by z_i itself: the coordinate compatibility that averaging relies on),
  affine R^2 and CCA (fitted on the validation split), principal angles between top-3 PCA
  subspaces. Shared z: PCA spectrum and participation ratio.
Exp 4: for test times spaced SPACING steps and every target variable i, K_MASKS masks per time
  drawn from the evaluation distribution (entry observed w.p. 1 - p, times M_orig; the target
  (t, i) is always hidden). Only the m-step window enters Model C, so masks are drawn on the
  window. Reported: mask share of latent / decoded variance (eta), within-state vs between-state
  distances, leave-one-out nearest-centroid state retrieval, by observed-entry count, own_obs and
  others_t. Uncertainty: block bootstrap over consecutive selected times (BLOCK times per block).
Output: results/extra/aim_latent/<run>/
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from experiments.extra.aim_multiview_rescue import load_model  # noqa: E402
from experiments.extra.aim_window_audit import load_data, refuse_overwrite  # noqa: E402
from src.evaluate import impute  # noqa: E402

RES = os.path.join(ROOT, "results")
NAMES = ["x", "y", "z"]
THEILER, SPACING, N_TIMES, K_MASKS, BLOCK, NBOOT = 50, 15, 200, 32, 10, 1000
DYN_STEPS = (1, 2, 5, 10)
NOBS_BINS = [0, 4, 8, 12, 16, 20, 24]
# figure colours (validated reference palette): categorical views, ordinal blue ramp for p
VIEW_C = {"x": "#2a78d6", "y": "#eb6834", "z": "#1baf7a", "fused": "#52514e"}
P_C = {0.1: "#86b6ef", 0.3: "#3987e5", 0.5: "#1c5cab", 0.7: "#0d366b"}


def latents(mdl, V, MV):
    """V, MV: (..., n, m) arrays -> z_i (..., n, d), w (..., n), z (..., d), xhat (..., n)."""
    V, MV = torch.tensor(V, dtype=torch.float32), torch.tensor(MV, dtype=torch.float32)
    with torch.no_grad():
        zi = mdl.encoder.enc(torch.cat([V * MV, MV], -1))
        w = MV.sum(-1)
        z = (zi * w[..., None]).sum(-2) / w.sum(-1, keepdim=True).clamp_min(1.0)
        xh = mdl.decode(z)
    return zi.numpy(), w.numpy(), z.numpy(), xh.numpy()


def windows(A, ts, m, tau):
    """A: (T, n) -> (len(ts), n, m), lag k at index k (as delay_embed)."""
    idx = ts[:, None] - tau * np.arange(m)[None]
    assert idx.min() >= 0
    return A[idx].transpose(0, 2, 1)


def pca(Z):
    Zc = Z - Z.mean(0)
    ev = np.linalg.svd(Zc, compute_uv=False) ** 2 / max(len(Z) - 1, 1)
    frac = ev / ev.sum()
    cum = np.cumsum(frac)
    return {"total_var": float(ev.sum()), "pr": float(ev.sum() ** 2 / (ev ** 2).sum()),
            "k90": int(np.searchsorted(cum, 0.90) + 1), "k99": int(np.searchsorted(cum, 0.99) + 1),
            "pc1_frac": float(frac[0])}, frac


def far_pairs(t, n_pairs, rng):
    a, b = rng.integers(0, len(t), 4 * n_pairs), rng.integers(0, len(t), 4 * n_pairs)
    keep = np.abs(t[a] - t[b]) > THEILER
    return a[keep][:n_pairs], b[keep][:n_pairs]


def inv_sqrt(C):
    e, U = np.linalg.eigh(C)
    return U @ np.diag(e ** -0.5) @ U.T


def cca(Xv, Yv, Xt, Yt, reg=1e-3):
    """Canonical correlations fitted on (Xv, Yv); returns (fit-set, test-set) correlations."""
    mx, my = Xv.mean(0), Yv.mean(0)
    A, B = Xv - mx, Yv - my
    d = A.shape[1]
    Cxx, Cyy, Cxy = A.T @ A / len(A), B.T @ B / len(B), A.T @ B / len(A)
    Cxx = Cxx + reg * np.trace(Cxx) / d * np.eye(d)
    Cyy = Cyy + reg * np.trace(Cyy) / d * np.eye(d)
    Wx, Wy = inv_sqrt(Cxx), inv_sqrt(Cyy)
    U, s, Vt = np.linalg.svd(Wx @ Cxy @ Wy)
    P, Q = (Xt - mx) @ (Wx @ U), (Yt - my) @ (Wy @ Vt.T)
    return s, np.array([np.corrcoef(P[:, k], Q[:, k])[0, 1] for k in range(d)])


def r2(Y, Yhat):
    return float(1 - ((Y - Yhat) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())


def pair_stats(Ziv, Zjv, Zit, Zjt, t, rng):
    same = np.linalg.norm(Zit - Zjt, axis=1)
    a, b = far_pairs(t, 20000, rng)
    d_wi = np.linalg.norm(Zit[a] - Zit[b], axis=1)
    d_wj = np.linalg.norm(Zjt[a] - Zjt[b], axis=1)
    d_x = np.linalg.norm(Zit[a] - Zjt[b], axis=1)
    tvi, tvj = Zit.var(0).sum(), Zjt.var(0).sum()
    Xa = np.c_[Ziv, np.ones(len(Ziv))]
    coef = np.linalg.solve(Xa.T @ Xa + 1e-6 * np.trace(Xa.T @ Xa) * np.eye(Xa.shape[1]), Xa.T @ Zjv)
    cv, ct = cca(Ziv, Zjv, Zit, Zjt)
    ui = np.linalg.svd(Zit - Zit.mean(0), full_matrices=False)[2][:3].T
    uj = np.linalg.svd(Zjt - Zjt.mean(0), full_matrices=False)[2][:3].T
    cos = np.linalg.svd(ui.T @ uj, compute_uv=False)
    return {"n_val": len(Ziv), "n_test": len(Zit),
            "d_same_median": float(np.median(same)), "d_far_within_i_median": float(np.median(d_wi)),
            "d_far_within_j_median": float(np.median(d_wj)), "d_far_cross_median": float(np.median(d_x)),
            "ratio_same_over_far_cross": float(np.median(same) / np.median(d_x)),
            "ratio_same_over_far_within": float(np.median(same) / np.median(np.r_[d_wi, d_wj])),
            "centroid_dist_rel": float(np.linalg.norm(Zit.mean(0) - Zjt.mean(0)) / np.sqrt((tvi + tvj) / 2)),
            "scale_ratio_i_over_j": float(np.sqrt(tvi / tvj)),
            "r2_identity_i_to_j": r2(Zjt, Zit), "r2_centred_identity_i_to_j": r2(Zjt - Zjt.mean(0), Zit - Zit.mean(0)),
            "r2_affine_i_to_j": r2(Zjt, np.c_[Zit, np.ones(len(Zit))] @ coef),
            "cca1_val": cv[0], "cca2_val": cv[1], "cca3_val": cv[2],
            "cca1_test": ct[0], "cca2_test": ct[1], "cca3_test": ct[2], "n_cca_test_gt_0.9": int((ct > 0.9).sum()),
            "top3_subspace_cos_mean": float(cos.mean()), "top3_subspace_cos_min": float(cos.min())}


def exp0(rates, seeds, out):
    vrows, prows, srows, spec, check = [], [], [], [], []
    keep_fig = {}
    for p in rates:
        for seed in seeds:
            rng = np.random.default_rng(100 * seed + int(round(10 * p)))
            data = load_data("lorenz", p, seed)
            X, M = data["X"], data["M"]
            n = X.shape[1]
            mdl, c = load_model("lorenz", "C", p, seed, n)
            m, tau, span = c["m"], c["tau"], (c["m"] - 1) * c["tau"]
            ts = {k: np.arange(max(a, span), b) for k, (a, b) in data["split"].items() if k in ("val", "test")}
            ref = impute(mdl, torch.tensor(X), torch.tensor(M), m, tau).numpy()
            for cond, Mc in [("full", data["M_orig"]), ("eval", M)]:
                lat = {}
                for part, t in ts.items():
                    lat[part] = latents(mdl, windows(X, t, m, tau), windows(Mc, t, m, tau))
                zi, w, z, xh = lat["test"]
                if cond == "eval":
                    check.append({"p": p, "seed": seed, "max_abs_diff_vs_impute": float(np.abs(xh - ref[ts["test"]]).max())})
                base = {"p": p, "seed": seed, "condition": cond}
                for i in range(n):
                    sel = w[:, i] > 0
                    st, frac = pca(zi[sel, i])
                    vrows.append({**base, "view": NAMES[i], "n": int(sel.sum()), "mean_obs_in_view": float(w[sel, i].mean()),
                                  "mean_norm": float(np.linalg.norm(zi[sel, i], axis=1).mean()), **st})
                    spec += [{**base, "latent": NAMES[i], "component": k + 1, "frac": f} for k, f in enumerate(frac)]
                for i in range(n):
                    for j in range(n):
                        if i == j:
                            continue
                        wv, wt = lat["val"][1], w
                        sv = (wv[:, i] > 0) & (wv[:, j] > 0)
                        stt = (wt[:, i] > 0) & (wt[:, j] > 0)
                        prows.append({**base, "view_i": NAMES[i], "view_j": NAMES[j],
                                      **pair_stats(lat["val"][0][sv, i], lat["val"][0][sv, j], zi[stt, i], zi[stt, j],
                                                   ts["test"][stt], rng)})
                st, frac = pca(z)
                srows.append({**base, "n": len(z), "n_all_views_empty": int((w.sum(1) == 0).sum()),
                              "mean_obs_total": float(w.sum(1).mean()), **st})
                spec += [{**base, "latent": "fused", "component": k + 1, "frac": f} for k, f in enumerate(frac)]
                if seed == seeds[0]:
                    keep_fig[(p, cond)] = (z, X[ts["test"]])
            print(f"exp0 p={p} seed={seed} done", flush=True)
    pd.DataFrame(vrows).to_csv(os.path.join(out, "exp0_view_stats.csv"), index=False)
    pd.DataFrame(prows).to_csv(os.path.join(out, "exp0_pair_stats.csv"), index=False)
    pd.DataFrame(srows).to_csv(os.path.join(out, "exp0_shared_stats.csv"), index=False)
    pd.DataFrame(spec).to_csv(os.path.join(out, "exp0_spectra.csv"), index=False)
    chk = pd.DataFrame(check)
    chk.to_csv(os.path.join(out, "exp0_extraction_check.csv"), index=False)
    print("latent extraction vs impute(): max abs diff", chk.max_abs_diff_vs_impute.max())
    assert chk.max_abs_diff_vs_impute.max() < 1e-4, "latent extraction does not reproduce Model C"
    return keep_fig


def eta_stats(Z, sel, K):
    """Mask share of variance for per-time sufficient statistics restricted to times sel."""
    W, S1, S2 = Z
    vw = W[sel].mean()
    vt = S2[sel].sum() / (len(sel) * K) - (np.linalg.norm(S1[sel].sum(0) / (len(sel) * K)) ** 2
                                          if S1.ndim == 2 else (S1[sel].sum() / (len(sel) * K)) ** 2)
    return vw / vt


def suff(Zk):
    """Zk: (N, K, d) or (N, K) -> per-time within variance, sum, sum of squares."""
    if Zk.ndim == 2:
        Zk = Zk[..., None]
    c = Zk.mean(1, keepdims=True)
    return ((Zk - c) ** 2).sum(-1).mean(1), Zk.sum(1), (Zk ** 2).sum(-1).sum(1)


def block_boot(stat, N, rng):
    nb = int(np.ceil(N / BLOCK))
    blocks = [np.arange(b * BLOCK, min(N, (b + 1) * BLOCK)) for b in range(nb)]
    vals = [stat(np.concatenate([blocks[k] for k in rng.integers(0, nb, nb)])) for _ in range(NBOOT)]
    return np.nanquantile(vals, [0.025, 0.975])


def exp4(rates, seeds, out, n_times, k_masks):
    recs, rows, dyn = [], [], []
    for p in rates:
        for seed in seeds:
            rng = np.random.default_rng(10000 + 100 * seed + int(round(10 * p)))
            data = load_data("lorenz", p, seed)
            X, Mo = data["X"], data["M_orig"]
            n = X.shape[1]
            mdl, c = load_model("lorenz", "C", p, seed, n)
            m, tau, span = c["m"], c["tau"], (c["m"] - 1) * c["tau"]
            a, b = data["split"]["test"]
            ts = np.arange(a + span, min(b, len(X)) - max(DYN_STEPS), SPACING)[:n_times]
            N = len(ts)
            Xw, Ow = windows(X, ts, m, tau), windows(Mo, ts, m, tau)
            S = X[ts]
            nn_state = np.sort(np.linalg.norm(S[:, None] - S[None], axis=-1), 1)[:, 1]
            per_t = {}
            for i in range(n):
                Mk = (rng.random((N, k_masks, n, m)) >= p).astype(np.float32) * Ow[:, None]
                Mk[:, :, i, 0] = 0
                zi, w, z, xh = latents(mdl, np.broadcast_to(Xw[:, None], Mk.shape), Mk)
                Oref = Ow.copy()
                Oref[:, i, 0] = 0
                zref = latents(mdl, Xw, Oref)[2]
                for dstep in DYN_STEPS:
                    t2 = ts + dstep
                    O2 = windows(Mo, t2, m, tau)
                    O2[:, i, 0] = 0
                    z2 = latents(mdl, windows(X, t2, m, tau), O2)[2]
                    dyn.append({"p": p, "seed": seed, "target": NAMES[i], "steps": dstep,
                                "dz_ref_median": float(np.median(np.linalg.norm(z2 - zref, axis=1))),
                                "ds_state_median": float(np.median(np.linalg.norm(X[t2] - S, axis=1)))})
                cen = z.mean(1)
                dz_c = np.linalg.norm(z - cen[:, None], axis=-1)
                zt = z - z.reshape(-1, z.shape[-1]).mean(0)
                tot_sd = np.sqrt((zt ** 2).sum(-1).mean())
                q = z.reshape(-1, z.shape[-1])
                D = np.linalg.norm(q[:, None] - cen[None], axis=-1)  # (N K, N)
                own = np.repeat(np.arange(N), k_masks)
                loo = (k_masks * cen[own] - q) / (k_masks - 1)
                D[np.arange(len(q)), own] = np.linalg.norm(q - loo, axis=1)
                that = D.argmin(1).reshape(N, k_masks)
                correct = that == np.arange(N)[:, None]
                s_err = np.linalg.norm(S[that] - S[:, None], axis=-1)
                err = xh[..., i] - X[ts, i][:, None]
                nobs = Mk.sum((-1, -2))
                own_obs = Mk[:, :, i, 1:].sum(-1)
                oth_t = Mk[:, :, :, 0].sum(-1)
                ia, ib = far_pairs(ts, 20000, rng)
                ka, kb = rng.integers(0, k_masks, len(ia)), rng.integers(0, k_masks, len(ia))
                d_between = np.linalg.norm(z[ia, ka] - z[ib, kb], axis=1)
                kk = rng.integers(0, k_masks, (N, 2))
                kk[:, 1] = (kk[:, 0] + 1 + rng.integers(0, k_masks - 1, N)) % k_masks
                d_within = np.linalg.norm(z[np.arange(N), kk[:, 0]] - z[np.arange(N), kk[:, 1]], axis=1)
                recs.append(pd.DataFrame({
                    "p": p, "seed": seed, "target": NAMES[i], "t": np.repeat(ts, k_masks), "mask": np.tile(np.arange(k_masks), N),
                    "n_obs": nobs.ravel().astype(int), "own_obs": own_obs.ravel().astype(int),
                    "others_t": oth_t.ravel().astype(int), "dz_centroid_rel": (dz_c / tot_sd).ravel(),
                    "dz_ref_rel": (np.linalg.norm(z - zref[:, None], axis=-1) / tot_sd).ravel(),
                    "err_target": err.ravel(), "retrieved": correct.ravel(), "state_err_retrieved": s_err.ravel()}))
                per_t[i] = {"z": suff(z), "xt": suff(xh[..., i]), "xall": suff(xh), "acc": correct.mean(1),
                            "sq": (err ** 2).mean(1)}
                eta = lambda s, key=i: eta_stats(per_t[key]["z"], s, k_masks)  # noqa: E731
                allt = np.arange(N)
                row = {"p": p, "seed": seed, "target": NAMES[i], "n_times": N, "k_masks": k_masks,
                       "eta_z": eta(allt), "eta_xhat_target": eta_stats(per_t[i]["xt"], allt, k_masks),
                       "eta_xhat_all": eta_stats(per_t[i]["xall"], allt, k_masks),
                       "latent_total_sd": float(tot_sd),
                       "xhat_target_var_over_true_var": float(xh[..., i].mean(1).var() / X[ts, i].var()),
                       "d_within_median": float(np.median(d_within)), "d_between_median": float(np.median(d_between)),
                       "ratio_within_over_between": float(np.median(d_within) / np.median(d_between)),
                       "retrieval_acc": float(correct.mean()), "chance_acc": 1 / N,
                       "state_err_retrieved_median": float(np.median(s_err)),
                       "state_nn_dist_median": float(np.median(nn_state)),
                       "rmse_target": float(np.sqrt((err ** 2).mean())),
                       "mean_n_obs": float(nobs.mean())}
                row["eta_z_lo"], row["eta_z_hi"] = block_boot(eta, N, rng)
                row["acc_lo"], row["acc_hi"] = block_boot(lambda s: per_t[i]["acc"][s].mean(), N, rng)
                row["rmse_lo"], row["rmse_hi"] = block_boot(lambda s: np.sqrt(per_t[i]["sq"][s].mean()), N, rng)
                rows.append(row)
            allt = np.arange(N)
            pooled = lambda s, key: np.mean([eta_stats(per_t[i][key], s, k_masks) for i in range(n)])  # noqa: E731
            row = {"p": p, "seed": seed, "target": "all", "n_times": N, "k_masks": k_masks,
                   "eta_z": pooled(allt, "z"), "eta_xhat_target": pooled(allt, "xt"), "eta_xhat_all": pooled(allt, "xall"),
                   "retrieval_acc": float(np.mean([per_t[i]["acc"].mean() for i in range(n)])), "chance_acc": 1 / N,
                   "rmse_target": float(np.sqrt(np.mean([per_t[i]["sq"].mean() for i in range(n)])))}
            for k in ("d_within_median", "d_between_median", "ratio_within_over_between", "state_err_retrieved_median",
                      "state_nn_dist_median", "mean_n_obs", "latent_total_sd", "xhat_target_var_over_true_var"):
                row[k] = float(np.mean([r[k] for r in rows[-n:]]))
            row["eta_z_lo"], row["eta_z_hi"] = block_boot(lambda s: pooled(s, "z"), N, rng)
            row["acc_lo"], row["acc_hi"] = block_boot(lambda s: np.mean([per_t[i]["acc"][s].mean() for i in range(n)]), N, rng)
            row["rmse_lo"], row["rmse_hi"] = block_boot(
                lambda s: np.sqrt(np.mean([per_t[i]["sq"][s].mean() for i in range(n)])), N, rng)
            rows.append(row)
            print(f"exp4 p={p} seed={seed}: N={N} K={k_masks} eta_z={row['eta_z']:.3f} acc={row['retrieval_acc']:.3f}",
                  flush=True)
    rec = pd.concat(recs, ignore_index=True)
    rec.to_csv(os.path.join(out, "exp4_records.csv.gz"), index=False)
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(out, "exp4_summary.csv"), index=False)
    pd.DataFrame(dyn).to_csv(os.path.join(out, "exp4_dynamics_reference.csv"), index=False)
    rec["n_obs_bin"] = pd.cut(rec.n_obs, NOBS_BINS, right=False, include_lowest=True).astype(str)
    agg = dict(n=("t", "size"), dz_centroid_rel_median=("dz_centroid_rel", "median"),
               dz_ref_rel_median=("dz_ref_rel", "median"), retrieval_acc=("retrieved", "mean"),
               state_err_retrieved_median=("state_err_retrieved", "median"),
               rmse_target=("err_target", lambda e: float(np.sqrt(np.mean(e ** 2)))))
    rec.groupby(["p", "seed", "n_obs_bin"]).agg(**agg).reset_index().to_csv(
        os.path.join(out, "exp4_by_nobs.csv"), index=False)
    rec["info_flag"] = np.select([(rec.own_obs == 0) & (rec.others_t == 0), rec.own_obs == 0, rec.others_t == 0],
                                 ["own_obs=0 & others_t=0", "own_obs=0", "others_t=0"], "other")
    rec.groupby(["p", "seed", "info_flag"]).agg(**agg).reset_index().to_csv(
        os.path.join(out, "exp4_by_flag.csv"), index=False)
    return rec, summ


def figures(out, keep_fig, rec, summ):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8984",
                         "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e",
                         "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "font.size": 9,
                         "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "lines.linewidth": 2})
    fdir = os.path.join(out, "figures")
    os.makedirs(fdir, exist_ok=True)
    spec = pd.read_csv(os.path.join(out, "exp0_spectra.csv"))
    rates = sorted(spec.p.unique())
    fig, axs = plt.subplots(1, len(rates), figsize=(3.2 * len(rates), 3), sharey=True, squeeze=False)
    for ax, p in zip(axs[0], rates):
        s = spec[(spec.p == p) & (spec.condition == "eval")]
        for lat, g in s.groupby("latent"):
            for k, (seed, gs) in enumerate(g.groupby("seed")):
                ax.plot(gs.component, gs.frac.cumsum(), color=VIEW_C[lat], alpha=1 if k == 0 else 0.45,
                        lw=2 if k == 0 else 1, label=lat if k == 0 else None, marker="o" if k == 0 else None, ms=3)
        ax.axhline(0.99, color="#8a8984", lw=0.8, ls="--")
        ax.set_title(f"p = {p}")
        ax.set_xlabel("PCA component")
    axs[0][0].set_ylabel("cumulative explained variance")
    axs[0][0].legend(frameon=False, title="latent")
    fig.suptitle("Exp 0: PCA spectra of view latents z_i and fused z (eval mask, test; seeds 0-2)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(fdir, "exp0_spectra.png"), dpi=150)
    plt.close(fig)

    pr = pd.read_csv(os.path.join(out, "exp0_pair_stats.csv"))
    pr["pair"] = [f"{min(a, b)}-{max(a, b)}" for a, b in zip(pr.view_i, pr.view_j)]
    mets = [("r2_identity_i_to_j", "identity R² (z_i as z_j)"), ("r2_centred_identity_i_to_j", "centred identity R²"),
            ("r2_affine_i_to_j", "affine R² (fit on val)"), ("cca1_test", "1st canonical corr. (test)")]
    for cond in ("eval", "full"):
        fig, axs = plt.subplots(1, 4, figsize=(13, 3), squeeze=False)
        for ax, (col, lab) in zip(axs[0], mets):
            d = pr[pr.condition == cond]
            for k, (pair, g) in enumerate(d.groupby("pair")):
                mean = g.groupby("p")[col].mean()
                ax.plot(mean.index, mean.values, color=list(VIEW_C.values())[k], marker="o", ms=4, label=pair)
                ax.scatter(g.p, g[col], color=list(VIEW_C.values())[k], s=10, alpha=0.5)
            ax.set_title(lab)
            ax.set_xlabel("missing rate p")
            ax.axhline(0, color="#8a8984", lw=0.8)
        axs[0][0].legend(frameon=False, title="view pair")
        fig.suptitle(f"Exp 0: cross-view coordinate compatibility ({cond} mask; lines = seed mean, dots = seed x direction)",
                     fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(fdir, f"exp0_alignment_{cond}.png"), dpi=150)
        plt.close(fig)

    div = LinearSegmentedColormap.from_list("div", ["#eb6834", "#e8e7e3", "#2a78d6"])
    seq = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#3987e5", "#0d366b"])
    fig, axs = plt.subplots(len(rates), 2, figsize=(7.5, 3.1 * len(rates)), squeeze=False)
    for r, p in enumerate(rates):
        z, s = keep_fig[(p, "eval")]
        zc = z - z.mean(0)
        u = np.linalg.svd(zc, full_matrices=False)[2]
        pc = zc @ u[:3].T
        lim = np.abs(s[:, 0]).max()
        h0 = axs[r][0].scatter(pc[:, 0], pc[:, 1], c=s[:, 0], cmap=div, vmin=-lim, vmax=lim, s=3, linewidths=0)
        h1 = axs[r][1].scatter(pc[:, 0], pc[:, 2], c=s[:, 2], cmap=seq, s=3, linewidths=0)
        axs[r][0].set(xlabel="PC1", ylabel="PC2", title=f"p = {p}: colour = Lorenz x (normalised)")
        axs[r][1].set(xlabel="PC1", ylabel="PC3", title=f"p = {p}: colour = Lorenz z (normalised)")
        fig.colorbar(h0, ax=axs[r][0], shrink=0.8)
        fig.colorbar(h1, ax=axs[r][1], shrink=0.8)
    fig.suptitle("Exp 0: fused latent z, top PCs (eval mask, test split, seed 0)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(fdir, "exp0_shared_pca.png"), dpi=150)
    plt.close(fig)

    s = summ[summ.target == "all"]
    fig, axs = plt.subplots(1, 3, figsize=(11, 3), squeeze=False)
    for ax, (col, lab) in zip(axs[0], [("eta_z", "mask share of latent variance"),
                                       ("eta_xhat_target", "mask share of decoded target variance"),
                                       ("retrieval_acc", "state retrieval accuracy (LOO)")]):
        mean = s.groupby("p")[col].mean()
        ax.plot(mean.index, mean.values, color="#2a78d6", marker="o", ms=5)
        ax.scatter(s.p, s[col], color="#2a78d6", s=12, alpha=0.5)
        if col in ("eta_z", "retrieval_acc"):
            lo, hi = col.replace("retrieval_", "") + "_lo", col.replace("retrieval_", "") + "_hi"
            ax.vlines(s.p, s[lo], s[hi], color="#2a78d6", alpha=0.35, lw=1)
        ax.set(title=lab, xlabel="missing rate p", ylim=(0, 1.02))
    fig.suptitle("Exp 4: stability vs discrimination under re-sampled masks (dots = seeds, bars = block-bootstrap 95% CI)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(fdir, "exp4_eta_retrieval.png"), dpi=150)
    plt.close(fig)

    rec = rec.assign(n_obs_bin=pd.cut(rec.n_obs, NOBS_BINS, right=False, include_lowest=True))
    g = rec.groupby(["p", "n_obs_bin"], observed=True).agg(dz=("dz_centroid_rel", "median"), acc=("retrieved", "mean"),
                                                           n=("t", "size")).reset_index()
    g = g[g.n >= 50]
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.2), squeeze=False)
    for p, gp in g.groupby("p"):
        x = [iv.left + 2 for iv in gp.n_obs_bin]
        axs[0][0].plot(x, gp.dz, color=P_C.get(p, "#2a78d6"), marker="o", ms=4, label=f"p = {p}")
        axs[0][1].plot(x, gp.acc, color=P_C.get(p, "#2a78d6"), marker="o", ms=4, label=f"p = {p}")
    axs[0][0].set(title="distance to own-state centroid / latent total sd", xlabel="observed entries in window (bin centre)")
    axs[0][1].set(title="state retrieval accuracy", xlabel="observed entries in window (bin centre)", ylim=(0, 1.02))
    axs[0][0].legend(frameon=False)
    fig.suptitle("Exp 4: dependence on available information (all seeds and targets pooled; bins with n >= 50)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(fdir, "exp4_by_nobs.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rates", default="0.1,0.3,0.5,0.7")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--run", default="v1")
    ap.add_argument("--smoke", action="store_true", help="p=0.3, seed 0, 30 times x 8 masks")
    args = ap.parse_args()
    torch.set_num_threads(4)
    rates = [0.3] if args.smoke else [float(x) for x in args.rates.split(",")]
    seeds = [0] if args.smoke else [int(x) for x in args.seeds.split(",")]
    n_times, k_masks = (30, 8) if args.smoke else (N_TIMES, K_MASKS)
    out = os.path.join(RES, "extra", "aim_latent", args.run)
    for f in ("exp0_pair_stats.csv", "exp4_summary.csv", "exp4_records.csv.gz"):
        refuse_overwrite(os.path.join(out, f))
    os.makedirs(out, exist_ok=True)
    keep_fig = exp0(rates, seeds, out)
    rec, summ = exp4(rates, seeds, out, n_times, k_masks)
    figures(out, keep_fig, rec, summ)
    pr = pd.read_csv(os.path.join(out, "exp0_pair_stats.csv"))
    print(pr.groupby(["condition", "p"])[["ratio_same_over_far_cross", "r2_identity_i_to_j", "r2_centred_identity_i_to_j",
                                          "r2_affine_i_to_j", "cca1_test", "cca3_test", "top3_subspace_cos_mean"]]
          .mean().round(3).to_string())
    print(pd.read_csv(os.path.join(out, "exp0_view_stats.csv")).groupby(["condition", "p", "view"])[
        ["pr", "k90", "k99", "total_var", "mean_obs_in_view"]].mean().round(3).to_string())
    print(pd.read_csv(os.path.join(out, "exp0_shared_stats.csv")).groupby(["condition", "p"])[
        ["pr", "k90", "k99", "total_var"]].mean().round(3).to_string())
    print(summ[summ.target == "all"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
