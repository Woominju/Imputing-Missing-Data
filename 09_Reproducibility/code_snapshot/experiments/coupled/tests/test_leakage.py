"""Phase 4 CPU leakage tests (must pass before any coupled GPU run).
Run: PYTHONPATH=.pydeps python -m experiments.coupled.tests.test_leakage"""
import numpy as np, torch
from experiments.coupled import data as CD
from experiments.aim_autonomous.model_f import ModelF
from experiments.fdr.lagweight import FDRLearned
from experiments.fdr.model import FDRModel
from src.models import build_model


def test_1_hidden_not_in_ae_data():
    X6 = CD.load_raw(1.0, 0); d, h = CD.build(X6, 0.3, 0)
    Y = X6.copy(); Y[:, [1, 4]] = np.random.default_rng(0).normal(size=(len(Y), 2))     # change hidden y1, y2 only
    d2, _ = CD.build(Y, 0.3, 0)
    for k in ("X", "M", "eval_mask", "mu", "sd"):
        assert np.array_equal(d[k], d2[k]), k
    assert d["X"].shape[1] == 4 and d["views"] == ["x1", "z1", "x2", "z2"]


def test_2_normalisation_train_only_and_splits():
    X6 = CD.load_raw(0.0, 0); d, h = CD.build(X6, 0.7, 1); a, b = d["split"]["train"]
    Z = X6.copy(); Z[b:] += 100.0                                                        # change val/test only
    d2, h2 = CD.build(Z, 0.7, 1)
    assert np.allclose(d["mu"], d2["mu"]) and np.allclose(d["sd"], d2["sd"]) and np.allclose(h["mu"], h2["mu"])
    s = d["split"]; assert s["train"][1] == s["val"][0] and s["val"][1] == s["test"][0] and s["test"][1] == len(X6)


def test_3_models_ignore_unobserved_values():
    """Outputs depend on X only through X*M: perturbing masked entries changes nothing (all four model types)."""
    torch.manual_seed(0); V = torch.randn(2, 5, 4, 8); MV = (torch.rand(2, 5, 4, 8) > 0.5).float(); MV[:, :, 2] = 0
    for mdl in (build_model("C", 4, 8, 16, 64), ModelF(4, 8, 16, 64), FDRModel(4, 8, 16, 64), FDRLearned(4, 8, 16, 64)):
        mdl.eval(); V2 = V + 50 * (1 - MV)
        with torch.no_grad():
            assert torch.allclose(mdl(V * MV, MV), mdl(V2 * MV, MV))
            assert torch.allclose(mdl(V, MV), mdl(V2, MV))                              # encoders multiply V by MV internally


if __name__ == "__main__":
    for name, f in sorted(globals().items()):
        if name.startswith("test_"):
            f(); print("PASS", name, flush=True)
    print("ALL PASS")
