"""Phase 3B CPU tests for the FDR module (must all pass before any GPU run).
Run: PYTHONPATH=.pydeps python -m experiments.fdr.tests.test_fdr"""
import time

import numpy as np
import torch

from experiments.fdr.impute import anchor_outputs, impute_fdr
from experiments.fdr.loss import fdr_masked_loss, make_forward_batch, multiplicity, sample_hide
from experiments.fdr.model import FDRModel
from src.delay_embedding import delay_embed
import src.train as T

CASES = [(8, 1, 64), (4, 3, 20), (5, 2, 9), (3, 5, 4), (8, 5, 36)]


def _ramp(B, W, n):
    """x_i(u) = 1000*i + u + 0.01*b : every raw position has a unique value."""
    b = torch.arange(B)[:, None, None] * 0.01
    u = torch.arange(W)[None, :, None].float()
    i = torch.arange(n)[None, None, :] * 1000.0
    return b + u + i


def test_1_shape_and_alignment():
    for m, tau, K in CASES:
        span = (m - 1) * tau; W = span + K; B, n = 2, 3
        X = _ramp(B, W, n)
        V, MV = delay_embed(X, torch.ones_like(X), m, tau)
        assert V.shape == (B, K, n, m)
        for j in range(K):
            for k in range(m):
                assert torch.equal(V[:, j, :, k], X[:, span + j - k * tau, :]), (m, tau, j, k)
        # an oracle output equal to the true lag values gives zero loss; a one-step shift does not
        H = torch.ones_like(X)
        assert fdr_masked_loss(V.clone(), X, H, m, tau)[0].item() == 0.0
        assert fdr_masked_loss(V + 1.0, X, H, m, tau)[0].item() > 0.5
        model = FDRModel(n, m, latent=4, hidden=8)
        assert model.forward_full(V, MV).shape == (B, K, n, m) and model(V, MV).shape == (B, K, n)


def _ref_loss(Y, X, H, m, tau):
    """Plain-loop reference of the multiplicity-corrected loss."""
    B, W, n = X.shape; span = (m - 1) * tau; K = W - span
    out = []
    for b in range(B):
        num = den = 0.0
        for u in range(W):
            anchors = [(u + k * tau, k) for k in range(m) if span <= u + k * tau <= W - 1]
            c = len(anchors)
            for i in range(n):
                if H[b, u, i] > 0 and c > 0:
                    den += 1
                    num += sum((X[b, u, i] - Y[b, t - span, i, k]) ** 2 for t, k in anchors) / c
        if den > 0:
            out.append(num / den)
    return float(np.mean(out))


def test_3_multiplicity():
    g = torch.Generator().manual_seed(0)
    for m, tau, K in CASES:
        span = (m - 1) * tau; W = span + K
        c = multiplicity(W, m, tau)
        brute = torch.tensor([sum(1 for k in range(m) if span <= u + k * tau <= W - 1) for u in range(W)]).float()
        assert torch.equal(c, brute)
        # full multiplicity exists iff K >= (m-1)tau + 1 ; every position covered iff K >= tau
        assert bool((c == m).any()) == (K >= span + 1), (m, tau, K)
        assert bool((c > 0).all()) == (K >= tau), (m, tau, K)
        B, n = 3, 2
        X = torch.randn(B, W, n, generator=g); H = (torch.rand(B, W, n, generator=g) < 0.4).float()
        Y = torch.randn(B, K, n, m, generator=g)
        got = fdr_masked_loss(Y, X, H, m, tau)[0].item()
        assert abs(got - _ref_loss(Y, X, H, m, tau)) < 1e-5, (m, tau, K)
    # boundary: K = span exactly -> no position reaches c = m (the earlier "K >= (m-1)tau" rule was off by one)
    m, tau = 4, 3; span = 9
    assert not bool((multiplicity(span + span, m, tau) == m).any())
    assert bool((multiplicity(span + span + 1, m, tau) == m).any())
    # each hidden raw entry carries total weight 1: a constant error e on every lag output gives loss e^2
    X = torch.zeros(1, 30, 1); H = torch.ones_like(X); m, tau = 5, 2; K = 30 - 8
    assert abs(fdr_masked_loss(torch.full((1, K, 1, m), 0.5), X, H, m, tau)[0].item() - 0.25) < 1e-7
    # targets restricted to hidden entries: zero H -> no loss
    assert fdr_masked_loss(torch.randn(1, K, 1, m), X, torch.zeros_like(X), m, tau)[0] is None


def test_2_mask_leakage():
    torch.manual_seed(0); g = torch.Generator().manual_seed(1)
    m, tau, K, n = 8, 1, 40, 3; W = (m - 1) * tau + K
    model = FDRModel(n, m, latent=16, hidden=32)
    cfg = {"m": m, "tau": tau, "hide_rate": 0.3}
    X = torch.randn(4, W, n, generator=g); M = (torch.rand(4, W, n, generator=g) > 0.3).float()
    for Q in (0.0, 0.15):
        H = sample_hide(cfg, M, torch.Generator().manual_seed(5), Q); Bm = M - H
        assert bool(((H <= M)).all()), "targets must be originally observed entries"
        X2 = X + H * 1e3 * torch.randn(X.shape, generator=g) + (1 - M) * 1e3 * torch.randn(X.shape, generator=g)
        z1 = model.encoder(*delay_embed(X * Bm, Bm, m, tau)); z2 = model.encoder(*delay_embed(X2 * Bm, Bm, m, tau))
        assert torch.equal(z1, z2), "hidden or missing values leaked into the encoding"
    # evaluation: changing values at unobserved positions does not change any imputation (past and bidir)
    Xs = torch.randn(300, n, generator=g); Ms = (torch.rand(300, n, generator=g) > 0.5).float()
    Xs2 = Xs + (1 - Ms) * 1e3 * torch.randn(Xs.shape, generator=g)
    for mode in ("past", "bidir"):
        assert torch.equal(impute_fdr(model, Xs, Ms, m, tau, mode), impute_fdr(model, Xs2, Ms, m, tau, mode))


def test_4_past_only_causality():
    torch.manual_seed(0); g = torch.Generator().manual_seed(2)
    n = 3
    for m, tau in [(8, 1), (4, 3)]:
        model = FDRModel(n, m, latent=8, hidden=16)
        X = torch.randn(400, n, generator=g); M = (torch.rand(400, n, generator=g) > 0.4).float()
        base_p = impute_fdr(model, X, M, m, tau, "past"); base_b = impute_fdr(model, X, M, m, tau, "bidir")
        h = (m - 1) * tau
        for t0 in (50, 200, 350):
            X2, M2 = X.clone(), M.clone()
            X2[t0 + 1:] = 1e3 * torch.randn(X2[t0 + 1:].shape, generator=g)
            M2[t0 + 1:] = (torch.rand(M2[t0 + 1:].shape, generator=g) > 0.5).float()
            assert torch.equal(impute_fdr(model, X2, M2, m, tau, "past")[:t0 + 1], base_p[:t0 + 1]), (m, tau, t0)
            # bidir at u uses times <= u + h : outputs for u <= t0 - h are unaffected
            assert torch.equal(impute_fdr(model, X2, M2, m, tau, "bidir")[:t0 - h + 1], base_b[:t0 - h + 1]), (m, tau, t0)
            # and bidir does use the future (sanity: some output in (t0-h, t0] changes)
            assert not torch.equal(impute_fdr(model, X2, M2, m, tau, "bidir")[t0 - h + 1:t0 + 1], base_b[t0 - h + 1:t0 + 1])


def test_5_gradients_and_stability():
    torch.manual_seed(0); g = torch.Generator().manual_seed(3)
    m, tau, K, n = 8, 1, 64, 3; W = (m - 1) * tau + K
    model = FDRModel(n, m, latent=16, hidden=64)
    cfg = {"m": m, "tau": tau, "hide_rate": 0.2}
    X = torch.randn(16, W, n, generator=g)
    for Q in (0.0, 0.15):
        for pobs in (0.3, 0.05):
            M = (torch.rand(16, W, n, generator=g) < pobs).float()
            loss, nh = make_forward_batch(Q, "full")(model, cfg, X, M, torch.Generator().manual_seed(9))
            model.zero_grad(); loss.backward()
            for name, p in model.named_parameters():
                assert p.grad is not None and torch.isfinite(p.grad).all(), name
            assert any(p.grad.abs().sum() > 0 for p in model.encoder.parameters())
            assert any(p.grad.abs().sum() > 0 for p in model.dec.parameters())
    # fully missing inputs: encoder (clamped denominator) stays finite; no hidden targets -> loss None, no NaN
    M0 = torch.zeros(4, W, n)
    V, MV = delay_embed(X[:4] * M0, M0, m, tau)
    assert torch.isfinite(model.forward_full(V, MV)).all()
    assert make_forward_batch(0.0, "full")(model, cfg, X[:4], M0, torch.Generator().manual_seed(0))[0] is None


def test_5b_same_batches_and_hiding_as_reference():
    """Q=0/current reproduces src.train.forward_batch; Q>0/current reproduces the archived Model E forward."""
    import experiments.aim_autonomous.model_e_viewdrop as EV
    from src.models import ModelC
    torch.manual_seed(0); g = torch.Generator().manual_seed(4)
    m, tau, K, n = 8, 1, 64, 3; W = (m - 1) * tau + K
    cfg = {"m": m, "tau": tau, "hide_rate": 0.2, "model": "C"}
    model = ModelC(n, m, 16, 64)
    X = torch.randn(8, W, n, generator=g); M = (torch.rand(8, W, n, generator=g) > 0.7).float()
    a = T.forward_batch(model, cfg, X, M, torch.Generator().manual_seed(11))[0]
    b = make_forward_batch(0.0, "current")(model, cfg, X, M, torch.Generator().manual_seed(11))[0]
    assert torch.equal(a, b)
    EV.Q = 0.15
    a = EV.forward_batch_viewdrop(model, cfg, X, M, torch.Generator().manual_seed(12))[0]
    b = make_forward_batch(0.15, "current")(model, cfg, X, M, torch.Generator().manual_seed(12))[0]
    assert torch.equal(a, b)


def test_6_identity_sanity():
    """Toy copy task: fully observed sinusoids, every entry a target. A small FDR model trained briefly must reconstruct
    lag k with lower error than the same output compared against lag k+1 (temporal alignment of the full-delay target).
    Frequencies are high enough that a one-lag shift (tau=2 steps) changes values strongly (first version used
    0.07/0.031 rad per step, which made shifted and correct targets nearly identical; test design fix, not a code fix).
    This checks implementation alignment only, not any property of a learned embedding."""
    torch.manual_seed(0)
    m, tau, K, n = 6, 2, 30, 2; span = (m - 1) * tau; W = span + K
    t = torch.arange(4000).float()
    series = torch.stack([torch.sin(0.9 * t), torch.cos(0.55 * t + 1.0)], 1)
    model = FDRModel(n, m, latent=8, hidden=32); opt = torch.optim.Adam(model.parameters(), 3e-3)
    g = torch.Generator().manual_seed(0)
    for step in range(400):
        s = torch.randint(0, 4000 - W, (32,), generator=g); idx = s[:, None] + torch.arange(W)[None]
        Xs = series[idx]; H = torch.ones_like(Xs)
        V, MV = delay_embed(Xs, torch.ones_like(Xs), m, tau)                  # inputs visible: pure alignment check
        loss, _ = fdr_masked_loss(model.forward_full(V, MV), Xs, H, m, tau)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        Y = model.forward_full(V, MV)
        right = np.mean([((Y[..., k] - Xs[:, span - k * tau: span - k * tau + K]) ** 2).mean().item() for k in range(m)])
        wrong = np.mean([((Y[..., k] - Xs[:, span - (k + 1) * tau: span - (k + 1) * tau + K]) ** 2).mean().item() for k in range(m - 1)])
    assert right < 0.01 and wrong > 10 * right, (right, wrong)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            t0 = time.time(); fn(); print(f"PASS {name} ({time.time() - t0:.2f}s)", flush=True)
    print("ALL PASSED")
