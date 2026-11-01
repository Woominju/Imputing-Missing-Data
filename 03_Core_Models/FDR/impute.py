"""Phase 3B FDR imputation protocols.
  past  : x_hat_i(u) = lag-0 output at anchor t = u           (uses observations at times <= u only)
  bidir : mean over anchors t = u + k*tau, k = 0..kmax with t <= T-1, of the lag-k output  (uses times <= u + kmax*tau)
Both use only X*M as input."""
import torch

from src.delay_embedding import delay_embed, pad_front


@torch.no_grad()
def anchor_outputs(model, X, M, m, tau, chunk=1024):
    """Full outputs for every anchor t = 0..T-1 (front padding as in src.evaluate.impute): (T, n, m)."""
    model.eval()
    T = X.shape[0]
    Xp, Mp = pad_front(X * M, M, m, tau)
    span = (m - 1) * tau
    out = []
    for s in range(0, T, chunk):
        e = min(T, s + chunk)
        V, MV = delay_embed(Xp[None, s:e + span], Mp[None, s:e + span], m, tau)
        out.append(model.forward_full(V, MV)[0])
    return torch.cat(out, 0)


@torch.no_grad()
def impute_fdr(model, X, M, m, tau, mode="past", kmax=None):
    Y = anchor_outputs(model, X, M, m, tau)
    if mode == "past":
        return Y[..., 0]
    kmax = m - 1 if kmax is None else kmax
    T, n = X.shape
    acc = torch.zeros(T, n, device=X.device)
    cnt = torch.zeros(T, 1, device=X.device)
    for k in range(kmax + 1):
        sh = k * tau
        if sh >= T:
            break
        acc[:T - sh] += Y[sh:, :, k]                     # anchor t = u + sh predicts u at lag k
        cnt[:T - sh] += 1
    return acc / cnt
