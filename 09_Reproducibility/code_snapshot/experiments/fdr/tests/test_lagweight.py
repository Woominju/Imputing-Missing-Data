"""Phase 3D-2 CPU tests for the lag-weighted FDR objective (must pass before any GPU run).
Run: PYTHONPATH=.pydeps python -m experiments.fdr.tests.test_lagweight"""
import torch

import src.train as T
from experiments.fdr.lagweight import (CapMatchedC, FDRLearned, fdr_lagweighted_loss, lag_weights, make_forward_batch_lagweight,
                                       pair_weights, weighted_multiplicity)
from experiments.fdr.loss import fdr_masked_loss, make_forward_batch
from experiments.fdr.model import FDRModel
from experiments.fdr.tests.test_fdr import CASES, _ramp
from src.delay_embedding import delay_embed
from src.models import build_model


def _rand(B, W, n, seed):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(B, W, n, generator=g, dtype=torch.float64)
    H = (torch.rand(B, W, n, generator=g) < 0.3).double()
    return X, H, g


def test_1_total_weight_one():
    for m, tau, K in CASES:
        W = (m - 1) * tau + K
        for a in (1 / m, 0.35, 0.65, 1.0, 0.0):
            P, cw = pair_weights(W, m, tau, lag_weights(a, m))
            tot = P.sum(1)
            assert torch.allclose(tot[cw > 0], torch.ones_like(tot[cw > 0])), (m, tau, a)
            assert (tot[cw == 0] == 0).all()


def test_2_uniform_equals_original_fdr():
    for m, tau, K in CASES:
        span = (m - 1) * tau; W = span + K; B, n = 4, 3
        X, H, g = _rand(B, W, n, m * 100 + tau)
        Y = torch.randn(B, K, n, m, generator=g, dtype=torch.float64)
        a = fdr_masked_loss(Y, X, H, m, tau)[0]; b = fdr_lagweighted_loss(Y, X, H, m, tau, lag_weights(1 / m, m))[0]
        # the original computes c(u) in float32 (torch.zeros(W)), so 1/c(u) carries ~1e-8 relative rounding
        assert torch.allclose(a, b, rtol=1e-6, atol=0), (m, tau, a.item(), b.item())


def test_3_alpha_one_equals_C_objective():
    for m, tau, K in CASES:
        span = (m - 1) * tau; W = span + K; B, n = 4, 3
        X, H, g = _rand(B, W, n, m * 7 + tau)
        Y = torch.randn(B, K, n, m, generator=g, dtype=torch.float64)
        a = T.masked_loss(Y[..., 0], X[:, span:], H[:, span:]); b = fdr_lagweighted_loss(Y, X, H, m, tau, lag_weights(1.0, m))[0]
        assert torch.allclose(a, b, rtol=1e-12, atol=1e-14), (m, tau)
        cw = weighted_multiplicity(W, m, tau, lag_weights(1.0, m))
        assert (cw[:span] == 0).all() and (cw[span:] == 1).all()          # C coverage: positions u < span carry no target
        Y2 = Y.clone(); Y2[..., 1:] += 5.0                                  # lags >= 1 are irrelevant at alpha = 1
        assert torch.equal(fdr_lagweighted_loss(Y2, X, H, m, tau, lag_weights(1.0, m))[0], b)


def test_4_alignment_oracle_and_shift():
    for m, tau, K in CASES:
        span = (m - 1) * tau; W = span + K; X = _ramp(2, W, 3).double(); H = torch.ones_like(X)
        V, _ = delay_embed(X, torch.ones_like(X), m, tau)
        for a in (1 / m, 0.35, 1.0):
            w = lag_weights(a, m)
            assert fdr_lagweighted_loss(V, X, H, m, tau, w)[0].item() == 0.0
            assert fdr_lagweighted_loss(V.roll(1, dims=-1), X, H, m, tau, w)[0].item() > 0.0 or a == 1.0
            assert fdr_lagweighted_loss(V + 1.0, X, H, m, tau, w)[0].item() > 0.5


def test_5_per_lag_terms_sum_and_interior_share():
    m, tau, K = 8, 1, 64; span = 7; W = span + K
    X, H, g = _rand(3, W, 2, 5); Y = torch.randn(3, K, 2, m, generator=g, dtype=torch.float64)
    for a in (1 / m, 0.35, 0.65, 1.0):
        loss, _, terms = fdr_lagweighted_loss(Y, X, H, m, tau, lag_weights(a, m), per_lag=True)
        assert torch.allclose(sum(terms), loss)
        P, cw = pair_weights(W, m, tau, lag_weights(a, m))
        assert torch.allclose(P[span:K], lag_weights(a, m).expand(K - span, m))   # interior share = w_k exactly


def test_6_no_target_leak_in_training_pass():
    """Hidden values are targets only: changing them changes the loss but never the model input."""
    cfg = dict(m=8, tau=1, hide_rate=0.2); B, W, n = 4, 71, 2
    torch.manual_seed(0); model = FDRModel(n, 8, 16, 64).double()
    Xs = torch.randn(B, W, n, dtype=torch.float64); Ms = (torch.rand(B, W, n) > 0.3).double()
    seen = {}

    def spy(V, MV):
        seen["V"] = (V * MV).clone(); return FDRModel.forward_full(model, V, MV)
    model.forward_full = spy
    fb = make_forward_batch_lagweight(0.35, 8)
    l1, _ = fb(model, cfg, Xs, Ms, torch.Generator().manual_seed(1)); V1 = seen["V"]
    H = Ms * (torch.rand(Ms.shape, generator=torch.Generator().manual_seed(1)) < 0.2).double()
    X2 = Xs + 10.0 * H                                                      # perturb hidden entries only
    l2, _ = fb(model, cfg, X2, Ms, torch.Generator().manual_seed(1)); V2 = seen["V"]
    assert torch.equal(V1, V2) and not torch.isclose(l1, l2)


def test_7_same_rng_as_original_fdr_and_C():
    """Uniform-weight training pass consumes the generator exactly like the original FDR pass (same H, same batches)."""
    cfg = dict(m=8, tau=1, hide_rate=0.2); torch.manual_seed(0); model = FDRModel(2, 8, 16, 64)
    Xs = torch.randn(4, 71, 2); Ms = (torch.rand(4, 71, 2) > 0.3).float()
    g1, g2 = torch.Generator().manual_seed(3), torch.Generator().manual_seed(3)
    a, _ = make_forward_batch(0.0, "full")(model, cfg, Xs, Ms, g1); b, _ = make_forward_batch_lagweight(1 / 8, 8)(model, cfg, Xs, Ms, g2)
    assert torch.allclose(a, b.float(), rtol=1e-6) and torch.equal(g1.get_state(), g2.get_state())


def test_8_param_counts():
    n = lambda mdl: sum(p.numel() for p in mdl.parameters())
    c, f, cc = build_model("C", 2, 8, 16, 64, 8, -1), FDRModel(2, 8, 16, 64), CapMatchedC(2, 8, 16, 64)
    assert (n(c), n(f), n(cc)) == (23202, 24112, 24102), (n(c), n(f), n(cc))
    fl = FDRLearned(4, 8, 16, 64); V = torch.randn(2, 5, 4, 8); MV = torch.ones_like(V); MV[:, :, 2] = 0
    assert fl.forward_full(V, MV).shape == (2, 5, 4, 8)


if __name__ == "__main__":
    for name, f in sorted(globals().items()):
        if name.startswith("test_"):
            f(); print("PASS", name, flush=True)
    print("ALL PASS")
