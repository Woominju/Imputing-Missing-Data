"""Phase 3D-2 Track A evaluation (CPU). Models: C, Ccap, FDR{1/8, 0.35, 0.65, 1.0} x seeds 0-4. Seeds 0-2 of C and FDR1/8 are the
Phase 3C checkpoints (p3c_C-xz / p3c_FDR-xz, uniform FDR objective), every other model is from this directory.
Protocol = Phase 3D-1 (corrected): missing-position vs observed-position lag-0 error (data mask, test); same-state two fresh
masks (Generator(4242+seed), p=0.3) -> both-missing error and output difference; y probes ridge/kNN (deterministic) and MLP with
probe-init seeds 0,1,2 (fit train, select val, predict test); FDR lag-wise missing-target error and the pre-specified bidir
lag mean (FUTURE-USING, reported separately); per-lag training-loss shares and batch-mean gradient cosine at the checkpoint.
Writes eval_*.csv + test predictions for paired bootstrap (preds_eval/)."""
import os, sys, json
import numpy as np, pandas as pd, torch
from multiprocessing import Pool
sys.path.insert(0, "results/extra/phase3d1_audit")
from experiments.fdr import probes, xz_data
from experiments.fdr.features import latents
from experiments.fdr.model import FDRModel
from experiments.fdr.lagweight import CapMatchedC, fdr_lagweighted_loss, lag_weights
from src.delay_embedding import delay_embed, pad_front
from src.models import build_model
from src.train import masked_loss

O = "results/extra/phase3d2_lag_weight"; P3C = "results/extra/phase3c_hidden_state"; m, tau, span, P = 8, 1, 7, 0.3
VARIANTS = ["C", "Ccap", "FDR1/8", "FDR0.35", "FDR0.65", "FDR1.0"]


def ckpt(v, s):
    if v == "C" and s <= 2: return f"{P3C}/checkpoints/p3c_C-xz_dt0.05_p0.3_s{s}.pt"
    if v == "FDR1/8" and s <= 2: return f"{P3C}/checkpoints/p3c_FDR-xz_dt0.05_p0.3_s{s}.pt"
    return f"{O}/checkpoints/p3d2_{v.replace('/', 'o')}_s{s}.pt"


def load(v, s):
    ck = torch.load(ckpt(v, s), map_location="cpu", weights_only=False)
    mdl = {"C": lambda: build_model("C", 2, m, 16, 64), "Ccap": lambda: CapMatchedC(2, m, 16, 64)}.get(v, lambda: FDRModel(2, m, 16, 64))()
    mdl.load_state_dict(ck["state_dict"]); mdl.eval(); return mdl


@torch.no_grad()
def outputs(mdl, X, M):
    Xp, Mp = pad_front(X * M, M, m, tau); V, MV = delay_embed(Xp[None], Mp[None], m, tau)
    z = mdl.encoder(V, MV); o = mdl.decode_full(z) if hasattr(mdl, "decode_full") else mdl.decode(z)[..., None]
    return o[0].numpy().astype(np.float64), z[0].numpy()


def rows_of(d, s): a, b = d["split"][s]; return np.arange(max(a, span), b)


def evaluate(job):
    v, s = job; torch.set_num_threads(3); R = dict(variant=v, seed=s); out = {}
    if not os.path.exists(ckpt(v, s)):
        return dict(**R, missing_checkpoint=True), {}
    mdl = load(v, s); d, yi = xz_data.build(xz_data.load_raw(), P, s, "xz"); yn = yi["y_norm"].astype(np.float64)
    X, M = torch.tensor(d["X"]), torch.tensor(d["M"]); Xn = d["X"].astype(np.float64); Mn, Ev = d["M"], d["eval_mask"]
    Y, Z = outputs(mdl, X, M); xh = Y[..., 0]; a, b = d["split"]["test"]
    sq = (xh[a:b] - Xn[a:b]) ** 2
    R.update(miss_rmse=np.sqrt((sq * Ev[a:b]).sum() / Ev[a:b].sum()), obs_rmse=np.sqrt((sq * Mn[a:b]).sum() / Mn[a:b].sum()),
             miss_rmse_x=np.sqrt((sq[:, 0] * Ev[a:b, 0]).sum() / Ev[a:b, 0].sum()), miss_rmse_z=np.sqrt((sq[:, 1] * Ev[a:b, 1]).sum() / Ev[a:b, 1].sum()),
             n_params=sum(p.numel() for p in mdl.parameters()))
    va, vb = d["split"]["val"]; R["val_miss_rmse"] = np.sqrt((((xh[va:vb] - Xn[va:vb]) ** 2) * Ev[va:vb]).sum() / Ev[va:vb].sum())
    out["miss_num"] = (sq * Ev[a:b]).sum(1); out["miss_den"] = Ev[a:b].sum(1)
    # FDR lag-wise (missing targets) + bidir (future-using)
    if Y.shape[-1] > 1:
        for k in range(m):
            t = np.arange(a + k, b); mk = 1 - Mn[t - k]
            R[f"lag{k}_miss_rmse"] = np.sqrt((((Y[t, :, k] - Xn[t - k]) ** 2) * mk).sum() / mk.sum())
        u = np.arange(a, b - (m - 1)); est = np.stack([Y[u + k, :, k] for k in range(m)], -1).mean(-1); mi = 1 - Mn[u]
        R["bidir_mean_lags_miss_rmse_FUTURE"] = np.sqrt((((est - Xn[u]) ** 2) * mi).sum() / mi.sum())
        t = np.arange(a, b); full = np.stack([Xn[t - k] for k in range(m)], -1)
        R["full_delay_rmse_all"] = np.sqrt(((Y[t] - full) ** 2).mean())
    # same-state two masks
    Xs = X[a:b]; g = torch.Generator().manual_seed(4242 + s)
    M1 = (torch.rand(Xs.shape, generator=g) >= P).float(); M2 = (torch.rand(Xs.shape, generator=g) >= P).float(); oo = []
    for Mq in (M1, M2):
        V, MV = delay_embed((Xs * Mq)[None], Mq[None], m, tau)
        with torch.no_grad():
            oo.append(mdl.decode(mdl.encoder(V, MV)[0]).numpy().astype(np.float64))
    S = Xs[span:].numpy(); m1, m2 = M1[span:].numpy(), M2[span:].numpy(); mm = (m1 == 0) & (m2 == 0); ob = (m1 > 0) & (m2 > 0)
    e = ((oo[0] - S) ** 2 + (oo[1] - S) ** 2) / 2; dd = (oo[0] - oo[1]) ** 2
    R.update(ss_bothmiss_err=np.sqrt(e[mm].mean()), ss_bothmiss_diff=np.sqrt(dd[mm].mean()), ss_bothobs_err=np.sqrt(e[ob].mean()), ss_bothobs_diff=np.sqrt(dd[ob].mean()))
    out["ssmm_num"] = (e * mm).sum(1); out["ssmm_den"] = mm.sum(1); out["ssmmd_num"] = (dd * mm).sum(1)
    # y probes
    tr, va_, te = rows_of(d, "train"), rows_of(d, "val"), rows_of(d, "test")
    for name in ("ridge", "knn"):
        pr, hp = getattr(probes, name)(Z[tr], yn[tr], Z[va_], yn[va_], Z[te]); R[f"y_{name}"] = np.sqrt(((pr - yn[te]) ** 2).mean()); out[f"y_{name}_err2"] = (pr - yn[te]) ** 2
    ml = []
    for ps in (0, 1, 2):
        pr, hp = probes.mlp(Z[tr], yn[tr], Z[va_], yn[va_], Z[te], seed=ps); ml.append(pr); R[f"y_mlp_ps{ps}"] = np.sqrt(((pr - yn[te]) ** 2).mean())
    R["y_mlp_mean_of_3"] = np.mean([R[f"y_mlp_ps{q}"] for q in (0, 1, 2)]); R["y_mlp_sd_of_3"] = np.std([R[f"y_mlp_ps{q}"] for q in (0, 1, 2)], ddof=1)
    out["y_mlp_err2"] = np.mean([(p_ - yn[te]) ** 2 for p_ in ml], 0)
    # per-lag loss shares + batch-mean gradient cosine (train segments, fixed generator 777, alpha-weighted objective of the model)
    if Y.shape[-1] > 1:
        alpha = {"FDR1/8": 1 / 8, "FDR0.35": 0.35, "FDR0.65": 0.65, "FDR1.0": 1.0}[v]; w = lag_weights(alpha, m)
        Xt, Mt = torch.tensor(d["X"]), torch.tensor(d["M"]); ta, tb = d["split"]["train"]; gg = torch.Generator().manual_seed(777); W = span + 64
        shared = [p for n_, p in mdl.named_parameters() if not n_.startswith("dec.4")]; terms_acc = np.zeros(m); g0 = gr = 0
        for _ in range(20):
            st = torch.randint(ta, tb - W + 1, (64,), generator=gg); idx = st[:, None] + torch.arange(W)[None]
            Xs_, Ms_ = Xt[idx], Mt[idx]; H = Ms_ * (torch.rand(Ms_.shape, generator=gg) < 0.2).float(); Bm = Ms_ - H
            V, MV = delay_embed(Xs_ * Bm, Bm, m, tau); loss, _, terms = fdr_lagweighted_loss(mdl.forward_full(V, MV), Xs_, H, m, tau, w, per_lag=True)
            terms_acc += np.array([float(t_) for t_ in terms])
            gl0 = torch.autograd.grad(masked_loss(mdl.forward_full(V, MV)[..., 0], Xs_[:, span:], H[:, span:]), shared, retain_graph=True)
            rest = sum(terms[1:]) if alpha < 1 else None
            g0 = g0 + torch.cat([x.flatten() for x in gl0])
            if rest is not None:
                gr = gr + torch.cat([x.flatten() for x in torch.autograd.grad(rest, shared)])
        sh = terms_acc / terms_acc.sum()
        R.update({f"loss_share_lag{k}": sh[k] for k in range(m)})
        R["cos_grad_lag0C_vs_otherlags"] = float(torch.dot(g0, gr) / (g0.norm() * gr.norm())) if alpha < 1 else np.nan
    return R, out


if __name__ == "__main__":
    jobs = [(v, s) for v in VARIANTS for s in range(5)]
    with Pool(15) as pool:
        res = pool.map(evaluate, jobs)
    os.makedirs(f"{O}/preds_eval", exist_ok=True)
    for (v, s), (_, out) in zip(jobs, res):
        if out:
            np.savez(f"{O}/preds_eval/{v.replace('/', 'o')}_s{s}.npz", **out)
    pd.DataFrame([r[0] for r in res]).to_csv(f"{O}/eval_trackA.csv", index=False)
    print("ALL DONE")
