"""Models A-D.

Shapes: n variables, delay dimension m, K target times per segment.
All models output normalised reconstructions xhat of shape (B, K, n).
"""
import math
import torch
import torch.nn as nn


class GroupLinear(nn.Module):
    """n independent linear maps applied in parallel: (..., n, i) -> (..., n, o)."""

    def __init__(self, n, i, o):
        super().__init__()
        self.w = nn.Parameter(torch.empty(n, i, o))
        self.b = nn.Parameter(torch.zeros(n, o))
        nn.init.uniform_(self.w, -1 / math.sqrt(i), 1 / math.sqrt(i))

    def forward(self, x):
        return torch.einsum("...ni,nio->...no", x, self.w) + self.b


def group_mlp(n, i, h, o, depth=2):
    layers, d = [], i
    for _ in range(depth):
        layers += [GroupLinear(n, d, h), nn.GELU()]
        d = h
    layers.append(GroupLinear(n, d, o))
    return nn.Sequential(*layers)


def mlp(i, h, o, depth=2, act=nn.GELU):
    layers, d = [], i
    for _ in range(depth):
        layers += [nn.Linear(d, h), act()]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


class ModelA(nn.Module):
    """Mask-aware MLP on the current time step only (no delay embedding)."""
    uses_delay = False

    def __init__(self, n, hidden=128, **_):
        super().__init__()
        self.net = mlp(2 * n, hidden, n, depth=3)

    def forward(self, Xc, Mc):  # (B, K, n)
        return self.net(torch.cat([Xc * Mc, Mc], -1))


class ModelB(nn.Module):
    """Single-view delay AE: x_i(t) reconstructed only from variable i's delay vector."""
    uses_delay = True

    def __init__(self, n, m, latent=16, hidden=64, **_):
        super().__init__()
        self.enc = group_mlp(n, 2 * m, hidden, latent)
        self.dec = group_mlp(n, latent, hidden, 1)

    def forward(self, V, MV):  # (B, K, n, m)
        z = self.enc(torch.cat([V * MV, MV], -1))
        return self.dec(z).squeeze(-1)


class MultiViewEncoder(nn.Module):
    """Per-variable encoders E_i and mask-aware mean fusion A.

    z(t) = sum_i w_i z_i / sum_i w_i with w_i = number of observed entries of
    v_i(t); a view with no observed entries contributes nothing."""

    def __init__(self, n, m, latent, hidden):
        super().__init__()
        self.enc = group_mlp(n, 2 * m, hidden, latent)

    def forward(self, V, MV):
        zi = self.enc(torch.cat([V * MV, MV], -1))  # (B, K, n, d)
        w = MV.sum(-1, keepdim=True)  # (B, K, n, 1)
        return (zi * w).sum(-2) / w.sum(-2).clamp_min(1.0)


class ModelC(nn.Module):
    """Multi-view delay AE: shared latent z(t) decoded to all variables."""
    uses_delay = True

    def __init__(self, n, m, latent=16, hidden=64, **_):
        super().__init__()
        self.encoder = MultiViewEncoder(n, m, latent, hidden)
        self.dec = group_mlp(n, latent, hidden, 1)
        self.n = n

    def decode(self, z):  # (..., d) -> (..., n)
        zz = z.unsqueeze(-2).expand(*z.shape[:-1], self.n, z.shape[-1])
        return self.dec(zz).squeeze(-1)

    def forward(self, V, MV):
        return self.decode(self.encoder(V, MV))


class ModelD(ModelC):
    """Shared latent dynamics: z(t0) from the multi-view encoder, propagated by a
    learned vector field dz/dt = f_theta(z) with RK4, decoded by h_{theta,i}.

    direction = +1 propagates forward from the first time of the window,
    direction = -1 propagates backward from the last time of the window.
    No dynamics loss is used; training uses only the masked reconstruction loss."""
    uses_delay = True

    def __init__(self, n, m, latent=16, hidden=64, window=8, direction=-1, h=1.0, **_):
        super().__init__(n, m, latent, hidden)
        self.f = mlp(latent, hidden, latent, depth=2, act=nn.Tanh)
        nn.init.normal_(self.f[-1].weight, std=1e-3)
        nn.init.zeros_(self.f[-1].bias)
        self.window, self.direction, self.h = window, direction, h

    def rk4(self, z):
        h, f = self.h * self.direction, self.f
        k1 = f(z)
        k2 = f(z + 0.5 * h * k1)
        k3 = f(z + 0.5 * h * k2)
        k4 = f(z + h * k3)
        return z + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

    def rollout(self, z0, steps):
        zs = [z0]
        for _ in range(steps - 1):
            zs.append(self.rk4(zs[-1]))
        return torch.stack(zs, 1)  # (B, steps, d), index k = k steps from anchor

    def forward_window(self, V, MV):
        """V, MV: (B, K=window, n, m). Returns xhat for the K window times (B, K, n)."""
        K = V.shape[1]
        a = 0 if self.direction > 0 else K - 1
        z0 = self.encoder(V[:, a], MV[:, a])
        x = self.decode(self.rollout(z0, K))
        return x if self.direction > 0 else x.flip(1)


def build_model(name, n, m, latent, hidden, window=8, direction=-1):
    return {"A": ModelA, "B": ModelB, "C": ModelC, "D": ModelD}[name](
        n=n, m=m, latent=latent, hidden=hidden, window=window, direction=direction)
