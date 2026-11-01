"""Delay embedding of *already masked* series.

For a masked segment Xin (B, W, n) and its mask (B, W, n), returns for every
target time t >= (m-1)*tau the per-variable delay vectors
    v_i(t) = [x_i(t), x_i(t-tau), ..., x_i(t-(m-1)tau)]   (zeros where masked)
with matching masks. Masking is always applied before this function is called,
so a hidden value never re-enters through another lag.
"""
import torch


def delay_embed(Xin, Min, m, tau):
    """Xin, Min: (B, W, n) -> V, MV: (B, W - (m-1)tau, n, m); lag k at index k."""
    span = (m - 1) * tau
    W = Xin.shape[1]
    K = W - span
    assert K > 0, "segment shorter than delay span"
    lags = [Xin[:, span - k * tau: span - k * tau + K] for k in range(m)]
    mlags = [Min[:, span - k * tau: span - k * tau + K] for k in range(m)]
    return torch.stack(lags, -1), torch.stack(mlags, -1)


def pad_front(X, M, m, tau):
    """Prepend (m-1)tau unobserved steps so every time index has a delay vector."""
    span = (m - 1) * tau
    zx = X.new_zeros((span,) + X.shape[1:])
    return torch.cat([zx, X], 0), torch.cat([zx, M], 0)
