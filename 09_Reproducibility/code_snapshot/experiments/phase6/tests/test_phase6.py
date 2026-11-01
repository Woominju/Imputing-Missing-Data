"""Phase 6 integrity tests (CPU). Run: PYTHONPATH=.pydeps:. python experiments/phase6/tests/test_phase6.py"""
import glob, numpy as np, torch
import experiments.phase6.gen6 as G
import experiments.phase6.core6 as C
from experiments.phase5.core import ArmModel, encode_series, SUB, OBS
from experiments.phase6.evaluate6 import own_counts


def t_seeds_disjoint_from_phase5_and_unique():
    p5 = {20261028 + 10000 * ki + j for ki in (0, 1) for j in list(range(8)) + list(range(100, 108)) + list(range(200, 210))}
    ids = [j for j, _ in G.roles()]; assert len(ids) == len(set(ids))
    p6 = [G.SEED6 + 100000 * ki + j for ki in (0, 1) for j in ids]; assert len(set(p6)) == len(p6) and not (set(p6) & p5)
    s0 = {G.s0_of(ki, j) for ki in (0, 1) for j in ids}; assert len(s0) == len(p6)
    assert len(glob.glob("results/extra/phase6/data/k*_id*.npz")) == 2 * len(ids)


def t_roles_never_reused_across_replicates():
    used = []
    for r in range(C.R6):
        I = C.ids(r); used += [I["train"], I["train_donor"], I["val"], I["val_donor"]] + I["test"] + I["test_donor"]
    assert len(used) == len(set(used))


def t_masks_identical_across_arms_and_sham_columns():
    k, p, r = 1.0, 0.7, 3; I = C.ids(r); mu, sd, _, _ = C.norm_stats(k)
    B = {a: C.build(k, p, r, a) for a in C.ARMS6}
    for j in (1, 2):
        own, oth = SUB[j], SUB[3 - j]
        for a in (f"A{j}", f"Aslot{j}", f"B{j}", f"Bcur{j}", f"Sham{j}"):
            D = B[a]; full = np.zeros((20000, 4), np.float32); full[:, D["ins"]] = D["Mtr"]
            ref = C.mcar((20000, 4), p, C.mask_seeds(k, p, r)["train"]); assert np.array_equal(full[:, D["ins"]], ref[:, D["ins"]])
            assert np.array_equal(D["Xtr"][:, :2], B[f"A{j}"]["Xtr"])                     # own columns identical (same normalisation)
        real = ((C.load(k, I["train"])[:, OBS] - mu) / sd).astype(np.float32); donor = ((C.load(k, I["train_donor"])[:, OBS] - mu) / sd).astype(np.float32)
        assert np.array_equal(B[f"B{j}"]["Xtr"][:, 2:], real[:, oth]) and np.array_equal(B[f"Sham{j}"]["Xtr"][:, 2:], donor[:, oth])
        Xt, _ = B[f"Sham{j}"]["test"](0); dt = ((C.load(k, I["test_donor"][0])[:, OBS] - mu) / sd).astype(np.float32)
        assert np.array_equal(Xt[:, 2:], dt[:, oth])
        assert B[f"Sham{j}"]["Xtr"].shape[1] == 4 and B[f"A{j}"]["Xtr"].shape[1] == 2


def t_no_hidden_state_in_inputs():
    D = C.build(1.0, 0.3, 0, "B1"); X = C.load(1.0, C.ids(0)["train"]); mu, sd, _, _ = C.norm_stats(1.0)
    assert np.allclose(D["Xtr"] * sd + mu, X[:, OBS], atol=1e-3)


def _model(arm):
    D = C.build(1.0, 0.7, 0, arm); torch.manual_seed(0); return D, ArmModel(len(D["ins"]), D["outs"], D["lagmask"])


def t_aslot_ignores_other_views_and_matches_B_params():
    D, m = _model("Aslot1"); _, mb = _model("B1")
    assert sum(q.numel() for q in m.parameters()) == sum(q.numel() for q in mb.parameters())
    X, M = torch.tensor(D["Xtr"][:600]), torch.ones(600, 4)
    z0, _ = encode_series(m, X, M); X2 = X.clone(); X2[:, 2:] = torch.randn(600, 2) * 5; z1, _ = encode_series(m, X2, M)
    assert torch.allclose(z0, z1)


def t_bcur_only_lag0_of_other_views():
    D, m = _model("Bcur2"); X, M = torch.tensor(D["Xtr"][:600]), torch.ones(600, 4)
    z0, _ = encode_series(m, X, M); X2 = X.clone(); X2[:-1, 2:] += 3.0; z1, _ = encode_series(m, X2, M)   # change other views except at last t
    assert torch.allclose(z0[-1], z1[-1]) and not torch.allclose(z0[-2], z1[-2])


def t_past_only():
    D, m = _model("B1"); X, M = torch.tensor(D["Xtr"][:600]), torch.ones(600, 4)
    z0, _ = encode_series(m, X, M); X2 = X.clone(); X2[300:] += 7.0; z1, _ = encode_series(m, X2, M)
    assert torch.allclose(z0[:300], z1[:300])


def t_own_counts():
    M = C.test_mask(1.0, 0.7, 0, 0, 0); L, c = own_counts(M)
    for t in (7, 100, 4999):
        assert L[1][t] == M[t - 7:t + 1, 0].sum() and L[2][t] == M[t - 7:t + 1, 2].sum() and c[1][t] == M[t, 2]


def t_jacobian_batch_matches_phase5():
    from experiments.phase5.obs_audit import back_jac
    from experiments.phase6.jac6 import back_jac_batch
    S = C.load(1.0, 900)[[100, 2000, 7000]]
    Pb = back_jac_batch(S, 1.0)
    for n in range(3):
        Ps = back_jac(S[n], 1.0, 8)
        assert all(np.allclose(Pb[q][n], Ps[q], rtol=1e-9, atol=1e-9) for q in range(8))


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("t_"):
            f(); print("PASS", n, flush=True)
