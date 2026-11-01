"""Regression tests added during the code audit (CPU only).
Run: python tests/test_integrity.py"""
import os
import sys

import numpy as np
import torch
from scipy.integrate import solve_ivp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.evaluate import impute  # noqa: E402
from src.lorenz import lorenz_rhs, generate_lorenz  # noqa: E402
from src.models import build_model  # noqa: E402
from src.observability import step_jacobians, backward_flow_jacobians  # noqa: E402
from src.train import DEFAULTS, forward_batch  # noqa: E402

m, tau, n, K = 4, 2, 3, 5


class Spy(torch.nn.Module):
    """Wraps a model and records the tensors it receives."""

    def __init__(self, inner):
        super().__init__()
        self.inner, self.seen = inner, []

    def forward(self, *a):
        self.seen.append([t.detach().clone() for t in a])
        return self.inner(*a)

    def forward_window(self, *a):
        self.seen.append([t.detach().clone() for t in a])
        return self.inner.forward_window(*a)


def test_forward_batch_hides_H():
    """The real training path: inputs at H positions are zero, masks are B = M - H,
    and changing X at H positions changes nothing the model sees."""
    for name in "ABCD":
        cfg = {**DEFAULTS, "model": name, "m": m, "tau": tau, "hide_rate": 0.3, "window": K, "device": "cpu"}
        W = K if name == "A" else (m - 1) * tau + K
        Xs = torch.randn(3, W, n)
        Ms = (torch.rand(3, W, n) > 0.2).float()
        H = Ms * (torch.rand(Ms.shape, generator=torch.Generator().manual_seed(7)) < 0.3).float()
        seen = []
        for perturb in (0.0, 1e3):
            torch.manual_seed(0)
            spy = Spy(build_model(name, n, m, 8, 16, window=K))
            forward_batch(spy, cfg, Xs + perturb * H, Ms, torch.Generator().manual_seed(7))
            seen.append(spy.seen[0])
        for a, b in zip(*seen):
            assert torch.equal(a, b), name  # model inputs identical despite perturbing hidden values
        if name == "A":
            Xin, B = seen[0]
            assert torch.equal(B, Ms - H) and torch.all(Xin[H > 0] == 0)
        else:
            V, MV = seen[0]
            assert torch.all(V[MV == 0] == 0)


def info_range(name, t=30, T=60, window=K):
    """Returns the set of time offsets d such that perturbing an observed value at t+d
    changes the imputation at time t."""
    torch.manual_seed(0)
    mdl = build_model(name, n, m, 8, 16, window=window)
    X = torch.randn(T, n)
    M = torch.ones(T, n)
    M[t, 0] = 0  # target missing
    base = impute(mdl, X, M, m, tau)[t, 0]
    used = []
    for d in range(-15, 15):
        X2 = X.clone()
        X2[t + d, 1] += 5.0
        if not torch.isclose(impute(mdl, X2, M, m, tau)[t, 0], base, atol=0, rtol=1e-7):
            used.append(d)
    return used


def test_information_ranges():
    span = (m - 1) * tau
    lags = sorted(-k * tau for k in range(m))
    assert info_range("A") == [0]
    assert info_range("B") == []  # only own variable; variable 1 is never used for variable 0
    assert info_range("C") == lags  # past delay lags only, no future
    d = info_range("D")
    # D (backward rollout from anchors t..t+K-1): lags of anchors -> [-span, K-1]
    assert min(d) == -span and max(d) == K - 1 and all(-span <= x <= K - 1 for x in d), d
    print("info ranges: A", info_range("A"), "C", lags, "D", (min(d), max(d)))


def test_backward_flow_jacobian_fd():
    X, _, _ = generate_lorenz(n_steps=400, transient=1000)
    J = step_jacobians(X, 0.01)
    Bk = backward_flow_jacobians(J, 20)
    for t, k in [(100, 1), (200, 5), (300, 20)]:
        fd = np.zeros((3, 3))
        for j in range(3):
            e = np.zeros(3); e[j] = 1e-6
            f = lambda s0: solve_ivp(lorenz_rhs, (0, -k * 0.01), s0, method="DOP853", rtol=1e-12, atol=1e-12).y[:, -1]
            fd[:, j] = (f(X[t] + e) - f(X[t] - e)) / 2e-6
        err = np.abs(fd - Bk[k, t]).max() / np.abs(fd).max()
        assert err < 1e-5, (t, k, err)
        # and the backward-integrated state really is the stored past sample
        assert np.allclose(solve_ivp(lorenz_rhs, (0, -k * 0.01), X[t], method="DOP853", rtol=1e-12,
                                     atol=1e-12).y[:, -1], X[t - k], atol=1e-6)


if __name__ == "__main__":
    for f in [test_forward_batch_hides_H, test_information_ranges, test_backward_flow_jacobian_fd]:
        f()
        print("PASS", f.__name__, flush=True)
