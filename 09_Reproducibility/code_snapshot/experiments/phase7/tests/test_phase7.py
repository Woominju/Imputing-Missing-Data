"""Phase 7 P7-B integrity tests (CPU). Run: PYTHONPATH=.pydeps:. python experiments/phase7/tests/test_phase7.py"""
import glob, numpy as np, torch
import experiments.phase7.core7 as C
import experiments.phase6.gen6 as G6, experiments.phase6.core6 as C6
from experiments.phase6.diag.k0_common import all_ids as diag_ids
from experiments.phase6.s1r.common import all_ids as s1r_ids
from experiments.phase5.core import ArmModel, encode_series, OBS
from experiments.fdr.loss import fdr_masked_loss
from src.delay_embedding import delay_embed


def t_ids_new_and_disjoint():
    p7 = [j for j, _ in C.all_ids()]; assert len(p7) == len(set(p7)) == 24 * 10
    old = {j for j, _ in G6.roles()} | {j for j, _ in diag_ids()} | {j for j, _ in s1r_ids()}
    assert not (set(p7) & old)
    p5 = {20261028 + 10000 * ki + j for ki in (0, 1) for j in list(range(8)) + list(range(100, 108)) + list(range(200, 210))}
    s = {G6.SEED6 + 100000 * ki + j for ki in (0, 1) for j in p7}; assert not (s & p5) and len(s) == 2 * len(p7)
    assert len(glob.glob(f"{C.DDIR}/k*_id*.npz")) == 2 * len(p7)


def t_arms_identical_except_loss_columns():
    B = {a: C.build(1.0, 0.7, 5, a) for a in C.ARMS7}
    for a in C.ARMS7:
        for key in ("Xtr", "Mtr", "Xva", "Mva"):
            assert np.array_equal(B[a][key], B["Bfull"][key])
        assert B[a]["ins"] == B[a]["outs"] == [0, 1, 2, 3] and np.array_equal(B[a]["lagmask"], B["Bfull"]["lagmask"])
        assert np.array_equal(B[a]["Hva"][:, B[a]["loss_cols"]], B["Bfull"]["Hva"][:, B[a]["loss_cols"]])
        off = [c for c in range(4) if c not in B[a]["loss_cols"]]; assert (B[a]["Hva"][:, off] == 0).all()
        for i in (0, 7):
            for e in (0, 4):
                Xa, Ma = B[a]["test"](i, e); Xb, Mb = B["Bfull"]["test"](i, e); assert np.array_equal(Xa, Xb) and np.array_equal(Ma, Mb)
    torch.manual_seed(7005); m1 = ArmModel(4, [0, 1, 2, 3], B["Bself1"]["lagmask"]); torch.manual_seed(7005); m2 = ArmModel(4, [0, 1, 2, 3], B["Bfull"]["lagmask"])
    assert all(torch.equal(x, y) for x, y in zip(m1.parameters(), m2.parameters()))   # same init, same parameter count


def t_inactive_heads_get_zero_gradient():
    D = C.build(1.0, 0.7, 0, "Bfull"); X = torch.tensor(D["Xtr"][:71 * 8]).reshape(8, 71, 4); M = torch.tensor(D["Mtr"][:71 * 8]).reshape(8, 71, 4)
    H = M * (torch.rand(M.shape, generator=torch.Generator().manual_seed(0)) < 0.2).float(); Bm = M - H
    V, MV = delay_embed(X * Bm, Bm, 8, 1)
    for arm, lc in C.LOSS_COLS.items():
        torch.manual_seed(1); m = ArmModel(4, [0, 1, 2, 3], D["lagmask"])
        loss, _ = fdr_masked_loss(m.forward_full(V, MV)[..., lc, :], X[..., lc], H[..., lc], 8, 1); loss.backward()
        last = m.dec[-1]; off = [c for c in range(4) if c not in lc]
        assert all(last.w.grad[c].abs().sum() == 0 and last.b.grad[c].abs().sum() == 0 for c in off)
        assert all(last.w.grad[c].abs().sum() > 0 for c in lc)


def t_no_hidden_state_in_inputs():
    D = C.build(0.0, 0.3, 2, "Bself2"); mu, sd, _, _ = C.norm_stats(0.0)
    assert np.allclose(D["Xtr"] * sd + mu, C.load(0.0, C.ids(2)["train"])[:, OBS], atol=1e-3) and D["Xtr"].shape[1] == 4


def t_normalisation_is_phase6_pool():
    assert all(np.array_equal(a, b) for a, b in zip(C.norm_stats(1.0), C6.norm_stats(1.0)))


def t_past_only():
    D = C.build(1.0, 0.7, 0, "Bfull"); torch.manual_seed(0); m = ArmModel(4, [0, 1, 2, 3], D["lagmask"])
    X, M = torch.tensor(D["Xtr"][:600]), torch.ones(600, 4); z0, _ = encode_series(m, X, M); X2 = X.clone(); X2[300:] += 5; z1, _ = encode_series(m, X2, M)
    assert torch.allclose(z0[:300], z1[:300])


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("t_"):
            f(); print("PASS", n, flush=True)
