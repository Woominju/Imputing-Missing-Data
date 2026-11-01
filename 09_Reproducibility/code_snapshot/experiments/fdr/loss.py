"""Phase 3B: multiplicity-corrected masked FDR loss and its training forward pass.

Segment of length W = span + K, span = (m-1)*tau. Anchors t = span..W-1 (index j = t - span in the model output).
Output Yhat[b, j, i, k] predicts x_i(u) with u = t - k*tau. Multiplicity of raw position u:
    c(u) = #{k in 0..m-1 : span <= u + k*tau <= W-1}.
Loss per segment b (raw-entry average; every hidden raw entry has total weight 1):
    L_b = sum_{u,i} H[b,u,i] 1{c(u)>0} (1/c(u)) sum_{k: anchor exists} (x[b,u,i] - Yhat[b,u+k tau-span,i,k])^2
          / sum_{u,i} H[b,u,i] 1{c(u)>0}
L = mean of L_b over segments with a non-zero denominator (same convention as src.train.masked_loss)."""
import torch

from src.delay_embedding import delay_embed


def multiplicity(W, m, tau):
    span = (m - 1) * tau
    c = torch.zeros(W)
    for k in range(m):
        lo, hi = max(0, span - k * tau), W - 1 - k * tau        # u range with an anchor at u + k tau in [span, W-1]
        if hi >= lo:
            c[lo:hi + 1] += 1
    return c


def fdr_masked_loss(Yfull, Xs, H, m, tau):
    """Yfull (B, K, n, m), Xs/H (B, W, n). Returns (loss or None, number of hidden raw entries with c>0)."""
    B, W, n = Xs.shape
    span = (m - 1) * tau
    K = W - span
    assert Yfull.shape[1] == K
    c = multiplicity(W, m, tau).to(Xs.device)
    w = torch.where(c > 0, 1.0 / c.clamp_min(1), torch.zeros_like(c))           # (W,)
    num = torch.zeros(B, device=Xs.device)
    for k in range(m):
        s = span - k * tau                                                       # raw positions u = s .. s+K-1
        tgt, h = Xs[:, s:s + K], H[:, s:s + K]
        num = num + (h * w[s:s + K, None] * (tgt - Yfull[..., k]) ** 2).flatten(1).sum(1)
    den = (H * (c > 0).float()[None, :, None]).flatten(1).sum(1)
    keep = den > 0
    if keep.sum() == 0:
        return None, den.sum()
    return (num[keep] / den[keep]).mean(), den.sum()


def sample_hide(cfg, Ms, gen, Q):
    """Training hiding H (and input B = M - H). Identical RNG call sequence to src.train.forward_batch when Q == 0 and to
    experiments.aim_autonomous.model_e_viewdrop.forward_batch_viewdrop when Q > 0, so C and FDR runs with the same seed
    see the same batches and the same H. View-dropout rule (as Model E): with prob Q per segment keep 1 (prob 1/2) or
    2 (prob 1/2) random views; every OBSERVED entry (M=1) of the dropped views is added to H (it becomes a target).
    Kept views keep their natural MCAR mask and item hiding; nothing is completed or invented."""
    Bsz, W, n = Ms.shape
    H = Ms * (torch.rand(Ms.shape, generator=gen, device=Ms.device) < cfg["hide_rate"]).float()
    if Q > 0:
        apply = torch.rand(Bsz, generator=gen, device=Ms.device) < Q
        k_keep = 1 + (torch.rand(Bsz, generator=gen, device=Ms.device) < 0.5).long()
        order = torch.argsort(torch.rand(Bsz, n, generator=gen, device=Ms.device), dim=1)
        rank = torch.argsort(order, dim=1)
        drop = (rank >= k_keep[:, None]) & apply[:, None]
        H = torch.where(drop[:, None, :], Ms, H)
    return H


def make_forward_batch(Q, target):
    """target: 'current' (Model C loss on lag 0 at anchors, = src.train.masked_loss) or 'full' (FDR loss)."""
    import src.train as T

    def fb(model, cfg, Xs, Ms, gen):
        H = sample_hide(cfg, Ms, gen, Q)
        Bm = Ms - H
        span = (cfg["m"] - 1) * cfg["tau"]
        V, MV = delay_embed(Xs * Bm, Bm, cfg["m"], cfg["tau"])                   # masking BEFORE windowing
        if target == "current":
            return T.masked_loss(model(V, MV), Xs[:, span:], H[:, span:]), H[:, span:].sum()
        return fdr_masked_loss(model.forward_full(V, MV), Xs, H, cfg["m"], cfg["tau"])
    return fb
