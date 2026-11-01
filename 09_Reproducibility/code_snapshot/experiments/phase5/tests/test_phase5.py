"""Gate B integrity tests for the Phase 5 arms (CPU). Run: PYTHONPATH=.pydeps:. python experiments/phase5/tests/test_phase5.py"""
import numpy as np, torch
import experiments.phase5.core as C
from experiments.fdr.loss import fdr_masked_loss
from src.delay_embedding import delay_embed

K, P, R = 1.0, 0.7, 0


def t_masks_identical_across_arms():
    full = C.mcar((20000, 4), P, C.seed_of("train", K, P, R))
    for arm in C.ARMS:
        D = C.build(K, P, R, arm)
        assert np.array_equal(D["Mtr"], full[:, D["ins"]]), arm
        for i in (0, 9):
            assert np.array_equal(D["test"](i)[1], C.test_masks(i, P)[:, D["ins"]]), arm


def t_outputs_are_own_views():
    """regression (pilot bug 10:37): decoded columns must be the arm's own-subsystem views for every arm"""
    for arm in C.ARMS:
        D = C.build(K, P, R, arm); dec = [D["ins"][o] for o in D["outs"]]
        want = [0, 1, 2, 3] if arm == "C" else C.SUB[int(arm[-1])]
        assert dec == want, (arm, dec)
        assert max(D["outs"]) < len(D["ins"])


def t_hidden_never_enters():
    orig = C.load
    def fake(k, j):
        X = orig(k, j).copy(); X[:, [1, 4]] = np.random.default_rng(j).normal(size=(len(X), 2)) * 1e3; return X
    a = C.build(K, P, R, "C"); C.load = fake
    try:
        b = C.build(K, P, R, "C")
    finally:
        C.load = orig
    assert np.array_equal(a["Xtr"], b["Xtr"]) and np.array_equal(a["Xva"], b["Xva"]) and np.array_equal(a["test"](3)[0], b["test"](3)[0])


def t_train_only_normalisation():
    orig = C.load
    def fake(k, j):
        X = orig(k, j); return X * 7 + 3 if j >= 100 else X
    a = C.build(K, P, R, "B1"); C.load = fake
    try:
        b = C.build(K, P, R, "B1")
    finally:
        C.load = orig
    assert np.allclose(a["mu"], b["mu"]) and np.allclose(a["sd"], b["sd"])


def model_for(arm, D):
    torch.manual_seed(0); return C.ArmModel(len(D["ins"]), D["outs"], D["lagmask"])


def t_past_only():
    for arm in ("A1", "B1", "C", "Bcur2"):
        D = C.build(K, P, R, arm); mdl = model_for(arm, D)
        X, M = torch.tensor(D["Xtr"][:600]), torch.tensor(D["Mtr"][:600])
        z0, _ = C.encode_series(mdl, X, M); X2 = X.clone(); X2[300:] += 5.0; M2 = M.clone(); M2[300:] = 1
        z1, _ = C.encode_series(mdl, X2, M2)
        assert torch.allclose(z0[:300], z1[:300]) and not torch.allclose(z0[300:], z1[300:]), arm


def t_bcur_lag_restriction():
    D = C.build(K, P, R, "Bcur1"); mdl = model_for("Bcur1", D)
    X, M = torch.tensor(D["Xtr"][:400]), torch.ones(400, 4)
    z0, _ = C.encode_series(mdl, X, M)
    X1 = X.clone(); X1[200, 2:] += 3.0                          # other views at time 200 -> lag 0 at t=200, lag>=1 at t>200
    z1, _ = C.encode_series(mdl, X1, M)
    assert not torch.allclose(z0[200], z1[200]) and torch.allclose(z0[201:208], z1[201:208])
    X2 = X.clone(); X2[200, :2] += 3.0                          # own views keep all lags
    z2, _ = C.encode_series(mdl, X2, M)
    assert not torch.allclose(z0[203], z2[203])


def t_sham_columns():
    a, s = C.build(K, P, R, "B1"), C.build(K, P, R, "Sham1")
    assert a["ins"] == s["ins"]
    raw_a, raw_s = a["Xtr"] * a["sd"] + a["mu"], s["Xtr"] * s["sd"] + s["mu"]
    assert np.allclose(raw_a[:, :2], raw_s[:, :2], atol=1e-3) and not np.allclose(raw_a[:, 2:], raw_s[:, 2:], atol=1e-1)
    other = C.load(K, 1)[:, C.OBS][:, C.SUB[2]]
    assert np.allclose(raw_s[:, 2:], other, atol=1e-3)


def t_loss_targets_only():
    D = C.build(K, P, R, "B1"); mdl = model_for("B1", D)
    Xs = torch.tensor(D["Xtr"][None, :71]); Ms = torch.tensor(D["Mtr"][None, :71])
    H = Ms * (torch.rand(Ms.shape, generator=torch.Generator().manual_seed(0)) < 0.2).float(); Bm = Ms - H
    def L(Xs):
        V, MV = delay_embed(Xs * Bm, Bm, 8, 1)
        return fdr_masked_loss(mdl.forward_full(V, MV), Xs[..., D["outs"]], H[..., D["outs"]], 8, 1)[0]
    X2 = Xs.clone(); X2[..., 2:] = torch.where(Bm[..., 2:] == 0, X2[..., 2:] + 9.0, X2[..., 2:])   # other cols where not input
    assert torch.allclose(L(Xs), L(X2))
    X3 = Xs.clone(); X3[..., :2] = torch.where(H[..., :2] > 0, X3[..., :2] + 1.0, X3[..., :2])     # own hidden targets
    assert not torch.allclose(L(Xs), L(X3))


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("t_"):
            f(); print("PASS", name)
