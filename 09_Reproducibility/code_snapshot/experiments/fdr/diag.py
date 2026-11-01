"""Phase 3B diagnostics shared by the existing-checkpoint analysis (dt=0.01) and the 2x2 pilot (dt=0.05). CPU only.

For one model and one (p, seed) data preparation:
  * same-state pairs: two independent natural MCAR masks at the evaluation rate p on the SAME test windows
    -> continuous distributions of d_z (latent, scaled within model), d_R (lag-0 state output), e1, e2 (state errors),
       a stabilised ratio d_z / max(d_R, delta), and the subset where both reconstructions are accurate;
  * near-collision analysis with TRUE states: output-space nearest neighbours between the two mask sets at different
    times, binned by the true-state distance, compared with the same-state (distance 0) pairs;
  * latent -> state kNN probe (fit on train-split windows, evaluated on test windows) and latent/state neighbourhood
    overlap;
  * for FDR models: full delay-vector reconstruction error on all lags (truth known in simulation).
All latent distances are divided by the model's own scale (median latent distance between random test times, NATe1)."""
import numpy as np
import torch

from src.delay_embedding import delay_embed

THRESH = [dict(name="strict", e=0.10, R=0.10, z=0.30), dict(name="loose", e=0.25, R=0.25, z=0.50)]
DELTA = 0.02          # floor for d_R in the ratio (about the median same-state output gap of the best models / 2)


def _encode(model, X, M, m, tau):
    V, MV = delay_embed((X * M)[None], M[None], m, tau)
    with torch.no_grad():
        z = model.encoder(V, MV)[0]
        full = model.forward_full(V, MV)[0] if hasattr(model, "forward_full") else None
        r = model.decode(z[None])[0]
    return z, r, full, V[0]


def _q(x, qs=(0.1, 0.25, 0.5, 0.75, 0.9)):
    x = x.flatten().float()
    return {f"q{int(q * 100)}": float(x.quantile(q)) for q in qs}


def knn_predict(Ztr, Ytr, Zte, k=10):
    out = []
    for s in range(0, len(Zte), 2048):
        d = torch.cdist(Zte[s:s + 2048], Ztr)
        idx = d.topk(k, largest=False).indices
        out.append(Ytr[idx].mean(1))
    return torch.cat(out)


def analyse(model, d, p, m, tau, gen_seed=4242, label=""):
    a_tr, b_tr = d["split"]["train"]; a, b = d["split"]["test"]
    X = torch.tensor(d["X"][a:b]); span = (m - 1) * tau; S = X[span:]
    g = torch.Generator().manual_seed(gen_seed)
    M1 = (torch.rand(X.shape, generator=g) < 1 - p).float(); M2 = (torch.rand(X.shape, generator=g) < 1 - p).float()
    z1, r1, f1, V = _encode(model, X, M1, m, tau); z2, r2, f2, _ = _encode(model, X, M2, m, tau)
    perm = torch.randperm(len(S), generator=g)
    sc = float((z1 - z1[perm]).norm(dim=1).median())
    dz = (z1 - z2).norm(dim=1) / sc; dR = (r1 - r2).norm(dim=1); e1 = (r1 - S).norm(dim=1); e2 = (r2 - S).norm(dim=1)
    res = {"label": label, "latent_scale": sc, "n_windows": len(S)}
    for nm, v in [("dz", dz), ("dR", dR), ("e1", e1), ("e2", e2), ("ratio_dz_over_max_dR_delta", dz * sc / dR.clamp_min(DELTA))]:
        res.update({f"{nm}_{k}": v_ for k, v_ in _q(v).items()})
    for th in THRESH:
        acc = (e1 <= th["e"]) & (e2 <= th["e"]); difz = dz >= th["z"]
        res[f"frac_accurate_{th['name']}"] = float(acc.float().mean())
        res[f"A_{th['name']}"] = float((acc & (dR <= th["R"]) & difz).float().mean())
        res[f"B_{th['name']}"] = float((((e1 > th["e"]) | (e2 > th["e"])) & difz).float().mean())
        res[f"C_{th['name']}"] = float((acc & ~difz).float().mean())
        if acc.any():
            res[f"dz_median_given_accurate_{th['name']}"] = float(dz[acc].median())
            res[f"dR_median_given_accurate_{th['name']}"] = float(dR[acc].median())
    # near-collisions with true states: for NATe1 sample t, nearest NATe2 output at a different time (|t-t'| > m*tau)
    DR = torch.cdist(r1, r2); tt = torch.arange(len(S))
    DR[(tt[:, None] - tt[None, :]).abs() <= m * tau] = float("inf")
    nn = DR.argmin(1); dR_nn = DR.min(1).values; ds_nn = (S - S[nn]).norm(dim=1); dz_nn = (z1 - z2[nn]).norm(dim=1) / sc
    close = dR_nn < 0.05
    res["nn_frac_output_close"] = float(close.float().mean())
    for lo, hi in [(0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 1e9)]:
        sel = close & (ds_nn >= lo) & (ds_nn < hi)
        res[f"nn_close_state_{lo}-{hi if hi < 1e8 else 'inf'}_frac"] = float(sel.float().mean() / max(close.float().mean(), 1e-9))
        res[f"nn_close_state_{lo}-{hi if hi < 1e8 else 'inf'}_dz_median"] = float(dz_nn[sel].median()) if sel.any() else float("nan")
    res["same_state_dz_median_for_output_close"] = float(dz[dR < 0.05].median()) if (dR < 0.05).any() else float("nan")
    # latent -> state probe (train windows with their own natural mask draw)
    Xtr = torch.tensor(d["X"][a_tr:b_tr]); Mtr = (torch.rand(Xtr.shape, generator=g) < 1 - p).float()
    ztr, _, _, _ = _encode(model, Xtr, Mtr, m, tau); Str = Xtr[span:]
    sub = torch.randperm(len(ztr), generator=g)[:6000]
    pred = knn_predict(ztr[sub], Str[sub], z1)
    res["probe_knn_state_rmse"] = float(((pred - S) ** 2).mean().sqrt())
    res["decoder_state_rmse"] = float(((r1 - S) ** 2).mean().sqrt())
    k = 10; sub2 = torch.randperm(len(S), generator=g)[:2000]
    Dz = torch.cdist(z1[sub2], z1[sub2]); Ds = torch.cdist(S[sub2], S[sub2]); Dz.fill_diagonal_(float("inf")); Ds.fill_diagonal_(float("inf"))
    iz = Dz.topk(k, largest=False).indices; is_ = Ds.topk(k, largest=False).indices
    res["knn10_overlap_latent_vs_state"] = float(np.mean([len(set(iz[i].tolist()) & set(is_[i].tolist())) / k for i in range(len(sub2))]))
    if f1 is not None:                                                        # full delay reconstruction
        # truth = delay vectors of the UNMASKED series. (First version compared against the masked input windows V,
        # whose unobserved entries are 0 -> fdr_* metrics were wrong; found because lag-0 RMSE != decoder_state_rmse.)
        V = delay_embed(X[None], torch.ones_like(X)[None], m, tau)[0][0]
        assert torch.allclose(V[..., 0], S)
        assert abs(float(((f1[..., 0] - V[..., 0]) ** 2).mean().sqrt()) - float(((r1 - S) ** 2).mean().sqrt())) < 1e-5
        res["fdr_full_rmse"] = float(((f1 - V) ** 2).mean().sqrt())
        for kk in range(m):
            res[f"fdr_lag{kk}_rmse"] = float(((f1[..., kk] - V[..., kk]) ** 2).mean().sqrt())
        per_win = ((f1 - V) ** 2).flatten(1).sum(1).sqrt()                     # ||R(z) - F_all(s)|| per window
        res.update({f"fdr_eps_{k_}": v_ for k_, v_ in _q(per_win, (0.5, 0.9, 0.99)).items()})
        res["fdr_full_out_gap_median"] = float((f1 - f2).flatten(1).norm(dim=1).median())
    return res
