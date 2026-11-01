"""Phase 3C leakage / boundary tests (CPU). Run: PYTHONPATH=.pydeps python -m experiments.fdr.tests.test_phase3c_leakage"""
import time
import numpy as np
import torch

from experiments.fdr import probes, xz_data
from experiments.fdr.features import latents
from experiments.fdr.loss import make_forward_batch
from experiments.fdr.model import FDRModel
import src.train as T

X3 = xz_data.load_raw()


def _perturbed_y():
    Xp = X3.copy(); Xp[:, 1] = np.random.default_rng(0).normal(0, 50, len(Xp)); return Xp


def test_y_does_not_enter_ae_data():
    for var in ("xz", "x"):
        d0, _ = xz_data.build(X3, 0.3, 1, var); d1, _ = xz_data.build(_perturbed_y(), 0.3, 1, var)
        for k in ("X", "M", "M_orig", "eval_mask", "mu", "sd"):
            assert np.array_equal(d0[k], d1[k]), (var, k)
        assert d0["X"].shape[1] == (2 if var == "xz" else 1)
    dxz, _ = xz_data.build(X3, 0.3, 1, "xz"); dx, _ = xz_data.build(X3, 0.3, 1, "x")
    assert np.array_equal(dxz["M"][:, :1], dx["M"]) and np.array_equal(dxz["eval_mask"][:, :1], dx["eval_mask"])


def test_normalisation_train_only():
    Xp = X3.copy(); d0, _ = xz_data.build(X3, 0.3, 0); a, b = d0["split"]["val"][0], len(X3)
    Xp[a:, [0, 2]] += 100.0
    d1, _ = xz_data.build(Xp, 0.3, 0)
    assert np.array_equal(d0["mu"], d1["mu"]) and np.array_equal(d0["sd"], d1["sd"])


def test_training_independent_of_y():
    """Two short CPU trainings through src.train.run with original vs perturbed y give identical weights/val score."""
    torch.manual_seed(0)
    cfg = dict(model="FDR", dataset="t", m=8, tau=1, latent=16, hidden=32, batch=16, targets=64, lr=1e-3, epochs=2,
               steps_per_epoch=5, patience=10, hide_rate=0.2, clip=5.0, window=8, direction=-1, device="cpu", seed=0)
    T.build_model = lambda name, n, m, latent, hidden, window, direction: FDRModel(n, m, latent, hidden)
    T.forward_batch = make_forward_batch(0.0, "full")
    T.RES = "results/extra/phase3c_hidden_state/test_artifacts"   # keep run() logs out of results/logs
    outs = []
    for Xr in (X3, _perturbed_y()):
        d, _ = xz_data.build(Xr, 0.3, 0)
        row, xhat, log = T.run(cfg, Xr[:, [0, 2]], d, "leaktest", save_ckpt=False, verbose=False)
        outs.append((row["best_val_rmse"], xhat))
    assert outs[0][0] == outs[1][0] and np.array_equal(outs[0][1], outs[1][1])


def test_probe_uses_train_val_only():
    g = np.random.default_rng(1)
    Ftr, Fva, Fte = g.normal(size=(500, 5)), g.normal(size=(200, 5)), g.normal(size=(200, 5))
    w = g.normal(size=5); ytr, yva = Ftr @ w, Fva @ w
    for name, f in probes.PROBES.items():
        p1, h1 = f(Ftr, ytr, Fva, yva, Fte)
        p2, h2 = f(Ftr, ytr, Fva, yva, Fte)          # test labels are never an argument of a probe
        assert np.allclose(p1, p2) and h1 == h2 or name == "mlp"
    import inspect
    for f in probes.PROBES.values():
        assert "yte" not in inspect.signature(f).parameters


def test_latents_do_not_see_hidden_or_y():
    d, _ = xz_data.build(X3, 0.3, 0); model = FDRModel(2, 8, 16, 32)
    X, M = torch.tensor(d["X"]), torch.tensor(d["M"])
    X2 = X + (1 - M) * 1e3
    assert torch.equal(latents(model, X, M, 8, 1), latents(model, X2, M, 8, 1))


def test_split_rows_disjoint_and_ordered():
    d, _ = xz_data.build(X3, 0.3, 0); sp = d["split"]
    assert sp["train"][1] <= sp["val"][0] and sp["val"][1] <= sp["test"][0]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            t0 = time.time(); fn(); print(f"PASS {name} ({time.time() - t0:.2f}s)", flush=True)
    print("ALL PASSED")
