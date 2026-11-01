"""Smoke tests: shapes, masking, leakage, time alignment, finite gradients, learning.
Run: python tests/test_smoke.py"""
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.delay_embedding import delay_embed, pad_front  # noqa: E402
from src.evaluate import impute  # noqa: E402
from src.lorenz import generate_lorenz  # noqa: E402
from src.missingness import prepare  # noqa: E402
from src.models import build_model  # noqa: E402
from src.train import DEFAULTS, run, forward_batch  # noqa: E402

torch.manual_seed(0)
m, tau, n = 4, 3, 3


def test_delay_alignment():
    X = torch.arange(20.0).reshape(20, 1).repeat(1, n)
    V, MV = delay_embed(X[None], torch.ones_like(X)[None], m, tau)
    span = (m - 1) * tau
    assert V.shape == (1, 20 - span, n, m)
    for t in range(V.shape[1]):
        for k in range(m):
            assert V[0, t, 0, k] == X[span + t - k * tau, 0]  # lag 0 == target time
    Xp, Mp = pad_front(X, torch.ones_like(X), m, tau)
    V, MV = delay_embed(Xp[None], Mp[None], m, tau)
    assert V.shape[1] == 20 and V[0, 0, 0, 0] == 0 and MV[0, 0, 0, 1:].sum() == 0


def test_no_leakage_inference():
    X = torch.randn(200, n)
    M = (torch.rand(200, n) > 0.4).float()
    X2 = X.clone()
    X2[M == 0] = 1e3 * torch.randn(int((M == 0).sum()))  # change hidden values only
    for name in "ABCD":
        mdl = build_model(name, n, m, 8, 16, window=5)
        assert torch.allclose(impute(mdl, X, M, m, tau), impute(mdl, X2, M, m, tau)), name


def test_no_leakage_training():
    """Values at H positions must not affect xhat during training forward."""
    cfg = {**DEFAULTS, "m": m, "tau": tau, "hide_rate": 0.3, "device": "cpu"}
    for name in "ABCD":
        cfg["model"] = name
        mdl = build_model(name, n, m, 8, 16, window=5)
        W = 5 + (m - 1) * tau if name == "D" else 12 + (0 if name == "A" else (m - 1) * tau)
        Xs, Ms = torch.randn(4, W, n), torch.ones(4, W, n)
        outs = []
        for perturb in (False, True):
            g = torch.Generator().manual_seed(1)
            H = Ms * (torch.rand(Ms.shape, generator=torch.Generator().manual_seed(1)) < 0.3).float()
            Xp = Xs + 1e3 * H if perturb else Xs
            B = Ms - H
            if name == "A":
                outs.append(mdl(Xp * B, B))
            else:
                V, MV = delay_embed(Xp * B, B, m, tau)
                outs.append(mdl.forward_window(V, MV) if name == "D" else mdl(V, MV))
        assert torch.allclose(*outs), name
        loss, nh = forward_batch(mdl, cfg, Xs, Ms, torch.Generator().manual_seed(1))
        assert torch.isfinite(loss) and nh > 0
        loss.backward()
        for p in mdl.parameters():
            assert p.grad is None or torch.isfinite(p.grad).all(), name


def test_learns_lorenz():
    X, _, _ = generate_lorenz(n_steps=3000, transient=1000)
    data = prepare(X, 0.3, 0)
    assert data["eval_mask"].sum() > 0 and np.all(data["M"] <= data["M_orig"])
    assert np.allclose(data["eval_mask"] * data["M"], 0)
    for name in "ABCD":
        cfg = {**DEFAULTS, "model": name, "dataset": "smoke", "seed": 0, "m": 6, "tau": 3, "epochs": 4,
               "steps_per_epoch": 50, "device": "cuda" if torch.cuda.is_available() else "cpu"}
        row, xhat, log = run(cfg, X, data, f"smoke_{name}", save_ckpt=False, verbose=False)
        assert xhat.shape == X.shape and np.isfinite(xhat).all()
        assert row["nonfinite_steps"] == 0
        assert log[-1]["train_loss"] < log[0]["train_loss"] * 1.05, (name, log)
        assert row["test_rmse"] < 1.0, (name, row["test_rmse"])  # better than mean imputation (~1)
        print(name, "train", [round(l["train_loss"], 4) for l in log], "test_rmse", round(row["test_rmse"], 4))


if __name__ == "__main__":
    for f in [test_delay_alignment, test_no_leakage_inference, test_no_leakage_training, test_learns_lorenz]:
        f()
        print("PASS", f.__name__, flush=True)
