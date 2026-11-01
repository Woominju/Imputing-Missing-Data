"""Phase 3C feature extraction for every anchor time t (front padding as src.evaluate.impute): latent zeta_t of a trained
encoder, or the raw masked delay window (values*mask, mask). Input is X*M only (masking before windowing)."""
import torch

from src.delay_embedding import delay_embed, pad_front


@torch.no_grad()
def latents(model, X, M, m, tau, chunk=2048):
    model.eval()
    Xp, Mp = pad_front(X * M, M, m, tau); span = (m - 1) * tau; T = X.shape[0]; out = []
    for s in range(0, T, chunk):
        e = min(T, s + chunk)
        V, MV = delay_embed(Xp[None, s:e + span], Mp[None, s:e + span], m, tau)
        out.append(model.encoder(V, MV)[0])
    return torch.cat(out)


@torch.no_grad()
def raw_windows(X, M, m, tau, cols=None):
    Xp, Mp = pad_front(X * M, M, m, tau)
    V, MV = delay_embed(Xp[None], Mp[None], m, tau)
    V, MV = V[0], MV[0]
    if cols is not None:
        V, MV = V[:, cols], MV[:, cols]
    return torch.cat([V.flatten(1), MV.flatten(1)], 1)
