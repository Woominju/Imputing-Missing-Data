"""Phase 3B minimal Full Delay Reconstruction (FDR) model. Experimental module; src/ is not modified.

Same per-view mask-aware encoder and scalar observation-count fusion as Model C (src.models.MultiViewEncoder),
latent k. The only change is the decoder: for every view i, R_i: R^k -> R^m outputs the whole past delay vector
    Yhat[..., i, k] ~ x_i(t - k*tau),  k = 0..m-1   (same lag order as src.delay_embedding.delay_embed).
forward() returns the lag-0 output (B, K, n) so that src.evaluate.impute / src.train.run (val selection) work unchanged;
forward_full() returns (B, K, n, m)."""
import torch
import torch.nn as nn

from src.models import MultiViewEncoder, group_mlp


class FDRModel(nn.Module):
    uses_delay = True

    def __init__(self, n, m, latent=16, hidden=64, **_):
        super().__init__()
        self.encoder = MultiViewEncoder(n, m, latent, hidden)
        self.dec = group_mlp(n, latent, hidden, m)
        self.n, self.m = n, m

    def decode_full(self, z):                      # (..., d) -> (..., n, m)
        zz = z.unsqueeze(-2).expand(*z.shape[:-1], self.n, z.shape[-1])
        return self.dec(zz)

    def forward_full(self, V, MV):
        return self.decode_full(self.encoder(V, MV))

    def decode(self, z):                           # lag-0 state, same signature as ModelC.decode
        return self.decode_full(z)[..., 0]

    def forward(self, V, MV):
        return self.forward_full(V, MV)[..., 0]
