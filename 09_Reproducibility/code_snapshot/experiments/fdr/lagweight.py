"""Phase 3D-2: lag-weighted Full Delay Reconstruction objective + model variants. experiments/fdr/loss.py and model.py are
NOT modified; this module only adds.

Objective. Segment of length W = span + K, anchors t = span..W-1, output Yhat[b, t-span, i, k] ~ x_i(t - k tau).
Lag weights w_k >= 0 (here w_0 = alpha, w_k = (1-alpha)/(m-1), k >= 1). For a raw position u let
    A(u)   = {k : span <= u + k tau <= W-1}                     (lags at which u is a target inside the segment)
    c_w(u) = sum_{k in A(u)} w_k                                (weighted multiplicity)
Per segment b:
    L_b = sum_{u,i} H[b,u,i] 1{c_w(u)>0} sum_{k in A(u)} (w_k / c_w(u)) (x[b,u,i] - Yhat[b,u+k tau-span,i,k])^2
          / sum_{u,i} H[b,u,i] 1{c_w(u)>0}
L = mean of L_b over segments with a non-zero denominator.
Properties (tested in tests/test_lagweight.py):
  * every hidden raw entry with c_w(u) > 0 carries total weight exactly 1 (sum_k w_k/c_w(u) = 1);
  * w_k = 1/m reproduces experiments.fdr.loss.fdr_masked_loss (1/c(u) multiplicity correction) exactly;
  * alpha = 1: c_w(u) = 1{u >= span}, so L equals src.train.masked_loss on the lag-0 output (Model C objective); positions
    u < span lose their (lag >= 1) targets, i.e. the target coverage is that of C, not of FDR.
Interior positions (A(u) = all lags) have c_w = sum_k w_k = 1, so there the lag-k share is exactly w_k."""
import torch
import torch.nn as nn

from experiments.aim_autonomous.model_f import LearnedFusionEncoder
from experiments.fdr.loss import sample_hide
from experiments.fdr.model import FDRModel
from src.delay_embedding import delay_embed
from src.models import MultiViewEncoder, group_mlp


def lag_weights(alpha, m):
    return torch.tensor([alpha] + [(1.0 - alpha) / (m - 1)] * (m - 1), dtype=torch.float64)


def weighted_multiplicity(W, m, tau, w):
    span = (m - 1) * tau
    cw = torch.zeros(W, dtype=torch.float64)
    for k in range(m):
        lo, hi = max(0, span - k * tau), W - 1 - k * tau
        if hi >= lo:
            cw[lo:hi + 1] += w[k]
    return cw


def pair_weights(W, m, tau, w):
    """(W, m) weight of target (u, k); zero where lag k has no anchor for u or c_w(u) = 0."""
    span = (m - 1) * tau; K = W - span
    cw = weighted_multiplicity(W, m, tau, w); P = torch.zeros(W, m, dtype=torch.float64)
    for k in range(m):
        s = span - k * tau
        P[s:s + K, k] = torch.where(cw[s:s + K] > 0, w[k] / cw[s:s + K].clamp_min(1e-300), torch.zeros(K, dtype=torch.float64))
    return P, cw


def fdr_lagweighted_loss(Yfull, Xs, H, m, tau, w, per_lag=False):
    """Yfull (B, K, n, m), Xs/H (B, W, n). Returns (loss or None, hidden count[, per-lag terms summing to loss])."""
    B, W, n = Xs.shape; span = (m - 1) * tau; K = W - span
    assert Yfull.shape[1] == K
    P, cw = pair_weights(W, m, tau, w)
    P = P.to(Xs.device, Xs.dtype); valid = (cw > 0).to(Xs.device, Xs.dtype)
    den = (H * valid[None, :, None]).flatten(1).sum(1); keep = den > 0
    nums = []
    for k in range(m):
        s = span - k * tau
        nums.append((H[:, s:s + K] * P[s:s + K, k][None, :, None] * (Xs[:, s:s + K] - Yfull[..., k]) ** 2).flatten(1).sum(1))
    if keep.sum() == 0:
        return (None, den.sum()) + (([],) if per_lag else ())
    terms = [(nk[keep] / den[keep]).mean() for nk in nums]
    loss = sum(terms)
    return (loss, den.sum()) + ((terms,) if per_lag else ())


def make_forward_batch_lagweight(alpha, m):
    w = lag_weights(alpha, m)

    def fb(model, cfg, Xs, Ms, gen):
        H = sample_hide(cfg, Ms, gen, 0.0)                      # same RNG sequence as C / FDR (Q = 0)
        Bm = Ms - H
        V, MV = delay_embed(Xs * Bm, Bm, cfg["m"], cfg["tau"])  # masking BEFORE windowing
        return fdr_lagweighted_loss(model.forward_full(V, MV), Xs, H, cfg["m"], cfg["tau"], w)
    return fb


class CapMatchedC(nn.Module):
    """Model C with the decoder hidden width raised (default 67) so the parameter count matches FDR (n=2: 24,102 vs 24,112).
    Encoder, fusion, loss: identical to C."""
    uses_delay = True

    def __init__(self, n, m, latent=16, hidden=64, dec_hidden=67, **_):
        super().__init__()
        self.encoder = MultiViewEncoder(n, m, latent, hidden)
        self.dec = group_mlp(n, latent, dec_hidden, 1)
        self.n = n

    def decode(self, z):
        zz = z.unsqueeze(-2).expand(*z.shape[:-1], self.n, z.shape[-1])
        return self.dec(zz).squeeze(-1)

    def forward(self, V, MV):
        return self.decode(self.encoder(V, MV))


class FDRLearned(FDRModel):
    """FDR decoder + the learned mask-aware fusion of Model F (experiments.aim_autonomous.model_f.LearnedFusionEncoder)."""

    def __init__(self, n, m, latent=16, hidden=64, **_):
        super().__init__(n, m, latent, hidden)
        self.encoder = LearnedFusionEncoder(self.encoder, n, m, latent, hidden)
