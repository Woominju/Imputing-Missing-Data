"""Full-series imputation, metrics and classical baselines."""
import numpy as np
import torch

from .delay_embedding import delay_embed, pad_front


@torch.no_grad()
def impute(model, X, M, m, tau, chunk=1024):
    """X, M: (T, n) tensors (normalised). Returns xhat (T, n) using only X*M as input."""
    model.eval()
    T, n = X.shape
    Xin = X * M
    if not model.uses_delay:
        return torch.cat([model(Xin[s:s + chunk][None], M[s:s + chunk][None])[0]
                          for s in range(0, T, chunk)], 0)
    Xp, Mp = pad_front(Xin, M, m, tau)
    span = (m - 1) * tau
    zs = []
    for s in range(0, T, chunk):
        e = min(T, s + chunk)
        V, MV = delay_embed(Xp[None, s:e + span], Mp[None, s:e + span], m, tau)
        if hasattr(model, "rollout"):
            zs.append(model.encoder(V, MV)[0])
        else:
            zs.append(model(V, MV)[0])
    if not hasattr(model, "rollout"):
        return torch.cat(zs, 0)
    # Model D: every time t is predicted from anchors t -/+ k (k < window); average them.
    z = torch.cat(zs, 0)
    acc = torch.zeros(T, n, device=X.device)
    cnt = torch.zeros(T, 1, device=X.device)
    zk = z
    for k in range(model.window):
        if k > 0:
            zk = model.rk4(zk)
        pred = model.decode(zk)  # prediction for time anchor + dir*k
        if model.direction > 0:
            acc[k:] += pred[:T - k]
            cnt[k:] += 1
        else:
            acc[:T - k] += pred[k:]
            cnt[:T - k] += 1
    return acc / cnt


def metrics(xhat, data, part):
    """RMSE/MAE on artificially hidden entries of the given split, in normalised
    (train z-score) units and original units."""
    a, b = data["split"][part]
    E = data["eval_mask"][a:b]
    x = data["X"][a:b]
    d = (xhat[a:b] - x) * E
    cnt = E.sum()
    out = {"n_eval": int(cnt)}
    if cnt == 0:
        return out
    out["rmse"] = float(np.sqrt((d ** 2).sum() / cnt))
    out["mae"] = float(np.abs(d).sum() / cnt)
    do = d * data["sd"]
    out["rmse_orig"] = float(np.sqrt((do ** 2).sum() / cnt))
    out["mae_orig"] = float(np.abs(do).sum() / cnt)
    cv = E.sum(0)
    out["rmse_per_var"] = np.sqrt((d ** 2).sum(0) / np.maximum(cv, 1)).tolist()
    return out


def baseline_impute(kind, X, M):
    """Classical baselines on normalised data (numpy). 'mean' uses train mean (=0
    after normalisation); 'locf' uses past only; 'linear' interpolates between
    observed neighbours (uses past and future observations)."""
    T, n = X.shape
    if kind == "mean":
        return np.zeros_like(X)
    out = np.zeros_like(X)
    t = np.arange(T)
    for i in range(n):
        obs = M[:, i] > 0
        if obs.sum() == 0:
            continue
        if kind == "linear":
            out[:, i] = np.interp(t, t[obs], X[obs, i])
        elif kind == "locf":
            idx = np.where(obs, t, -1)
            idx = np.maximum.accumulate(idx)
            out[:, i] = np.where(idx >= 0, X[np.maximum(idx, 0), i], 0.0)
        else:
            raise ValueError(kind)
    return out
