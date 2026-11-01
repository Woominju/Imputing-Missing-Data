"""Phase 3D-1 §3: per-lag share of the FDR training loss (with the 1/c(u) weights of experiments.fdr.loss) and a limited
gradient diagnostic at the saved checkpoints. CPU, no optimizer step, no parameter is changed.
Batches imitate training: segments W = span + 64 from the TRAIN split, H ~ Bernoulli(0.2) * M, input B = M - H
(fixed generator 777, independent of the training generator). Also evaluates, on the same batches, the C-style lag-0 loss
(src.train.masked_loss on anchors) for C and for FDR's lag-0 head, on train and on val segments."""
import numpy as np, pandas as pd, torch
from common import RUNS, X3, OUT, P, span, m, tau, load_model
from experiments.fdr import xz_data
from experiments.fdr.loss import multiplicity
from src.delay_embedding import delay_embed
from src.train import masked_loss

NB, BS, KT = 30, 64, 64; W = span + KT
c = multiplicity(W, m, tau); w = torch.where(c > 0, 1 / c.clamp_min(1), torch.zeros_like(c))
ROWS, AN, GR, GM = [], [], [], []
# analytic lag share of the loss weight for uniformly placed hidden entries
tot = float((c > 0).sum())
for k in range(m):
    s = span - k * tau; AN.append(dict(lag=k, weight_share=float(w[s:s + KT].sum()) / tot))


def batches(d, split, seed_):
    X, M = torch.tensor(d["X"]), torch.tensor(d["M"]); a, b = d["split"][split]; g = torch.Generator().manual_seed(seed_)
    for _ in range(NB):
        s = torch.randint(a, b - W + 1, (BS,), generator=g); idx = s[:, None] + torch.arange(W)[None]
        Xs, Ms = X[idx], M[idx]; H = Ms * (torch.rand(Ms.shape, generator=g) < 0.2).float(); yield Xs, Ms, H


def per_lag_terms(Yf, Xs, H):
    """Returns list of per-lag loss terms L_k (sum_k L_k = experiments.fdr.loss.fdr_masked_loss) and raw per-lag MSE."""
    den = (H * (c > 0).float()[None, :, None]).flatten(1).sum(1); keep = den > 0; terms, raw = [], []
    for k in range(m):
        s = span - k * tau; tgt, h = Xs[:, s:s + KT], H[:, s:s + KT]
        num = (h * w[s:s + KT, None] * (tgt - Yf[..., k]) ** 2).flatten(1).sum(1)
        terms.append((num[keep] / den[keep]).mean())
        raw.append(((h * (tgt - Yf[..., k]) ** 2).sum() / h.sum()).detach())
    return terms, raw


for _, r in RUNS.sort_values(["seed", "tag"]).iterrows():
    mdl, _ = load_model(r); d, _ = xz_data.build(X3, P, int(r.seed), r.vars)
    for split in ("train", "val"):
        acc = {};
        for Xs, Ms, H in batches(d, split, 777):
            Bm = Ms - H; V, MV = delay_embed(Xs * Bm, Bm, m, tau)
            if r.decoder == "full":
                Yf = mdl.forward_full(V, MV); terms, raw = per_lag_terms(Yf, Xs, H)
                for k in range(m):
                    acc.setdefault(f"L_{k}", []).append(float(terms[k])); acc.setdefault(f"rawmse_{k}", []).append(float(raw[k]))
                lag0 = Yf[..., 0]
                if split == "train":           # gradient diagnostic on shared parameters
                    shared = [p for n_, p in mdl.named_parameters() if not n_.startswith("dec.4")]
                    g0 = torch.autograd.grad(terms[0], shared, retain_graph=True)
                    gr = torch.autograd.grad(sum(terms[1:]), shared, retain_graph=True)
                    gc = torch.autograd.grad(masked_loss(lag0, Xs[:, span:], H[:, span:]), shared)
                    f = lambda gs: torch.cat([x.flatten() for x in gs])
                    v0, vr, vc = f(g0), f(gr), f(gc)
                    acc_g = acc.setdefault("_g", [0, 0, 0]); acc_g[0] = acc_g[0] + v0; acc_g[1] = acc_g[1] + vr; acc_g[2] = acc_g[2] + vc
                    GR.append(dict(model=r.tag, seed=int(r.seed), norm_g_lag0_term=float(v0.norm()), norm_g_other_lags=float(vr.norm()),
                                   cos_lag0_vs_other=float(torch.dot(v0, vr) / (v0.norm() * vr.norm())),
                                   norm_g_lag0_unweighted_Closs=float(vc.norm()), cos_Closs_vs_other_lags=float(torch.dot(vc, vr) / (vc.norm() * vr.norm()))))
            else:
                with torch.no_grad():
                    lag0 = mdl(V, MV)
            with torch.no_grad():
                acc.setdefault("C_style_lag0_hidden_mse", []).append(float(masked_loss(lag0, Xs[:, span:], H[:, span:])))
        g_ = acc.pop("_g", None)
        if g_ is not None:                      # cosine of batch-averaged gradients (less dominated by mini-batch noise)
            cs = lambda a_, b_: float(torch.dot(a_, b_) / (a_.norm() * b_.norm()))
            GM.append(dict(model=r.tag, seed=int(r.seed), n_batches=NB, cos_mean_g_lag0_vs_other=cs(g_[0], g_[1]),
                           cos_mean_g_Closs_vs_other=cs(g_[2], g_[1]), norm_mean_g_lag0=float(g_[0].norm() / NB),
                           norm_mean_g_other=float(g_[1].norm() / NB), norm_mean_g_total=float((g_[0] + g_[1]).norm() / NB)))
        ROWS.append(dict(model=r.tag, seed=int(r.seed), split=split, **{k: float(np.mean(v)) for k, v in acc.items()}))
    print("done", r.exp_id, flush=True)

pd.DataFrame(AN).to_csv(f"{OUT}/fdr_lag_weight_share_analytic.csv", index=False)
pd.DataFrame(ROWS).to_csv(f"{OUT}/lag_loss_terms.csv", index=False)
pd.DataFrame(GR).to_csv(f"{OUT}/lag_gradient_diag.csv", index=False)
pd.DataFrame(GM).to_csv(f"{OUT}/lag_gradient_mean.csv", index=False)
print("ALL DONE")
