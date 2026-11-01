"""Phase 8 integrity tests (CPU; Gate B). Run: PYTHONPATH=.pydeps:. python experiments/phase8/tests/test_phase8.py
Runs under P8_SMOKE=1 with P8_ROOT = a fresh temporary directory (set below BEFORE importing core8); full-size checks use core8.FULL.
Includes an end-to-end smoke run of the whole pipeline (gen -> 9 arms -> oracles -> kNN -> evaluation -> analysis) on CPU."""
import glob, hashlib, itertools, os, re, subprocess, sys, tempfile
TMP = tempfile.mkdtemp(prefix="p8smoke_"); os.environ["P8_SMOKE"] = "1"; os.environ["P8_ROOT"] = TMP; os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np, torch, sympy as sp
from scipy import stats
import experiments.phase8.core8 as C8
from experiments.phase5.core import ArmModel, encode_series, raw_features, OBS, SPAN, SUB, mcar
from src.lorenz import integrate, coupled_lorenz_rhs

PY = sys.executable; ENV = dict(os.environ, PYTHONPATH=".pydeps:."); PROTO = "results/notes/phase8_mathematical_protocol_v2.md"


def run(args, ok=True, env=ENV):
    r = subprocess.run([PY] + args, env=env, capture_output=True, text=True)
    if ok:
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return r


# ------------------------------------------------------------------ design / ids / constants
def t_seeds_unique_and_disjoint_full_size():
    from experiments.phase6.gen6 import SEED6, roles
    from experiments.phase6.diag.k0_common import all_ids as diag_ids
    from experiments.phase6.s1r.common import all_ids as s1r_ids
    from experiments.phase7.core7 import all_ids as p7_ids
    old = C8.SZ; C8.SZ = C8.FULL
    try:
        p8 = [j for j, _ in C8.lengths()] + C8.ref_ids()
        assert len(p8) == len(set(p8)) == 24 * 20 + 48 + 30000
        seeds8 = {SEED6 + 100000 * ki + j for ki in (0, 1) for j in p8}; assert len(seeds8) == 2 * len(p8)
        prev = {j for j, _ in roles()} | {j for j, _ in diag_ids()} | {j for j, _ in s1r_ids()} | {j for j, _ in p7_ids()}
        seeds_prev = {SEED6 + 100000 * ki + j for ki in (0, 1) for j in prev}
        p5 = {20261028 + 10000 * ki + j for ki in (0, 1) for j in list(range(8)) + list(range(100, 108)) + list(range(200, 210))}
        assert not (seeds8 & seeds_prev) and not (seeds8 & p5) and not (set(p8) & prev)
        pool = set(C8.pool_ids()["train"] + C8.pool_ids()["val"]); rep = {j for r in range(24) for v in C8.ids(r).values() for j in (v if isinstance(v, list) else [v])}
        assert not (pool & rep) and not (set(C8.ref_ids()) & (pool | rep))
    finally:
        C8.SZ = old


def t_constants_match_protocol_text():
    T = open(PROTO, encoding="utf-8").read(); c = C8.CONST
    must = ["δ_I = 0.02", "δ_y = 0.02", "δ₆ = 0.05", "**ε = 0.01**", "α_I = 0.025", "0.0125씩", "{256, 512, 1024}", "{5, 10, 20, 50, 100}", "{10, 20, 40}",
            "**3850.667**", "3080.533", "K = 4", "60000 + q, q = 0..39", "60100 + q, q = 0..7", "70000 + q, q = 0..29,999", "train 50000 + r", "val 50100 + r",
            "test 50200 + 10r + i", "train 51000 + r", "val 51100 + r", "test 51200 + 10r + i", "8000 + r", "8100 + r", "8200 + r", "57,840", "27,944",
            "31,552", "864", "55.50", "epoch = 1,000 step", "epoch ≤ 60, patience 8", "Linear 4층"]
    miss = [m for m in must if m not in T]; assert not miss, miss
    assert (c["delta_I"], c["delta_y"], c["delta_6"], c["eps_equiv"], c["alpha_I"], c["alpha_IIa"], c["alpha_IIb"]) == (0.02, 0.02, 0.05, 0.01, 0.025, 0.0125, 0.0125)
    assert c["orc_widths"] == (256, 512, 1024) and c["knn_k_grid"] == (5, 10, 20, 50, 100) and c["sat_ntraj"] == (10, 20, 40)
    F = C8.FULL; assert (F["R"], F["NT"], F["NMASK"], F["NPOOL_TR"], F["NPOOL_VA"], F["KPOOL"], F["N_REF"], F["ORC_EPOCH_STEPS"], F["ORC_MAX_EPOCHS"], F["ORC_PATIENCE"]) == \
        (24, 8, 5, 40, 8, 4, 30000, 1000, 60, 8)
    assert (F["AE_EPOCHS"], F["AE_STEPS"], F["AE_PATIENCE"]) == (300, 100, 15)
    assert len(C8.ARMS8) * 24 * 2 * 2 == 864


def t_arm_specs_and_param_counts():
    lm = lambda n: np.ones((n, 8), np.float32); P = lambda m: sum(q.numel() for q in m.parameters())
    assert P(ArmModel(4, [0, 1, 2, 3], lm(4), latent=16)) == 57840 and P(ArmModel(2, [0, 1], lm(2), latent=8)) == 27944 and P(ArmModel(2, [0, 1], lm(2), latent=16)) == 31552
    S = {a: C8.arm_spec(a) for a in C8.ARMS8}
    assert S["Bfull"]["ins"] == [0, 1, 2, 3] and set(S["Bfull"]["src"].values()) == {"real"}
    assert S["BfullSham1"]["src"] == {0: "real", 1: "real", 2: "donor", 3: "donor"} and S["BfullSham2"]["src"] == {0: "donor", 1: "donor", 2: "real", 3: "real"}
    for j in (1, 2):
        assert S[f"A{j}_8"]["ins"] == SUB[j] and S[f"A{j}_8"]["latent"] == 8 and S[f"A{j}_16"]["latent"] == 16
        assert set(S[f"A{j}_8d"]["src"].values()) == {"donor"} and S[f"A{j}_8d"]["ins"] == SUB[j]
        assert C8.init_seed(f"A{j}_8", 5) == C8.init_seed(f"A{j}_8d", 5) == 8105
    assert C8.init_seed("Bfull", 3) == C8.init_seed("BfullSham1", 3) == C8.init_seed("BfullSham2", 3) == 8003 and C8.init_seed("A1_16", 3) == 8203
    assert C8.SYM_COORDS == {"X0": [0, 1, 3, 4], "L1:L0": [0, 1], "L2:L0": [3, 4]}


# ------------------------------------------------------------------ mathematics
def t_trapping_identity_and_bound():
    x1, y1, z1, x2, y2, z2, k = sp.symbols("x1 y1 z1 x2 y2 z2 k", real=True); s, r, b = 10, 28, sp.Rational(8, 3); c = s + r
    f = [s * (y1 - x1) + k * (x2 - x1), x1 * (r - z1) - y1, x1 * y1 - b * z1, s * (y2 - x2) + k * (x1 - x2), x2 * (r - z2) - y2, x2 * y2 - b * z2]
    X = [x1, y1, z1, x2, y2, z2]; V = x1 ** 2 + y1 ** 2 + (z1 - c) ** 2 + x2 ** 2 + y2 ** 2 + (z2 - c) ** 2
    dV = sum(sp.diff(V, v) * fv for v, fv in zip(X, f))
    rep = 2 * (s - 1) * (x1 ** 2 + x2 ** 2) + (b - 2) * ((z1 - c) ** 2 + (z2 - c) ** 2) + b * (z1 ** 2 + z2 ** 2) + 2 * k * (x1 - x2) ** 2
    assert sp.expand((-2 * V + 2 * b * c ** 2) - dV - rep) == 0
    assert abs(float(b * c ** 2) - C8.CONST["V_bound"]) < 1e-9 and 2 * (15 ** 2 + 15 ** 2 + 33 ** 2) == 3078
    zz = sp.symbols("z"); g = -2 * b * zz ** 2 + 2 * b * c * zz + 2 * (zz - c) ** 2; zs = sp.solve(sp.diff(g, zz), zz)[0]
    assert abs(float(g.subs(zz, zs)) - C8.CONST["V_bound_sharp"]) < 1e-9 and abs(float(sp.sqrt(C8.CONST["V_bound_sharp"])) - 55.50) < 0.01
    for kap in (0.0, 1.0):                         # worst box corner, transient included
        S = integrate(coupled_lorenz_rhs, [15, 15, 5, -15, -15, 5], 4000, 0.05, 0, args=(kap,))
        Vt = (S[:, [0, 1, 3, 4]] ** 2).sum(1) + ((S[:, [2, 5]] - 38) ** 2).sum(1); assert Vt.max() <= C8.CONST["V_bound_sharp"] + 1e-6


def t_symmetry_group_equivariance():
    s0 = np.array([3.0, -2.0, 20.0, -7.0, 4.0, 30.0]); g12 = np.array([-1, -1, 1, -1, -1, 1.0]); g1 = np.array([-1, -1, 1, 1, 1, 1.0]); pi = [3, 4, 5, 0, 1, 2]
    tr = lambda s, k: integrate(coupled_lorenz_rhs, list(s), 200, 0.05, 0, args=(k,))
    for k in (0.0, 1.0):
        A = tr(s0, k)
        assert np.abs(tr(s0 * g12, k) - A * g12).max() < 1e-5 * np.abs(A).max()            # gamma_12: all kappa
        assert np.abs(tr(s0[pi], k) - A[:, pi]).max() < 1e-5 * np.abs(A).max()             # pi: all kappa
    A0 = tr(s0, 0.0); assert np.abs(tr(s0 * g1, 0.0) - A0 * g1).max() < 1e-5 * np.abs(A0).max()   # gamma_1: kappa = 0
    A1 = tr(s0, 1.0); assert np.abs(tr(s0 * g1, 1.0) - A1 * g1).max() > 1.0                       # gamma_1 broken at kappa = 1


def t_window_invariance_on_symmetric_strata():
    rng = np.random.default_rng(0); S = rng.normal(size=(60, 6)) * 5 + [0, 0, 25, 0, 0, 25]; g12 = np.array([-1, -1, 1, -1, -1, 1.0]); g1 = np.array([-1, -1, 1, 1, 1, 1.0])
    k = 0.0; O, Og12, Og1 = C8.nz_obs(k, S), C8.nz_obs(k, S * g12), C8.nz_obs(k, S * g1)
    M = (rng.random((60, 4)) > 0.5).astype(np.float32); n = 40; M[n - 7:n + 1, 0] = 0; M[n - 7:n + 1, 2] = 0       # X0 at n
    F, Fg = raw_features(O, M, [0, 1, 2, 3]), raw_features(Og12, M, [0, 1, 2, 3]); assert np.array_equal(F[n], Fg[n]) and C8.strata(M)["X0"][n]
    M2 = (rng.random((60, 4)) > 0.5).astype(np.float32); M2[n - 7:n + 1, 0] = 0; M2[n, 2] = 1                       # L1:L0 at n, x2 observed
    assert np.array_equal(raw_features(O, M2, [0, 1, 2, 3])[n], raw_features(Og1, M2, [0, 1, 2, 3])[n]) and C8.strata(M2)["L1:L0"][n]
    assert not np.array_equal(raw_features(O, M2, [0, 1, 2, 3])[n], raw_features(Og12, M2, [0, 1, 2, 3])[n])     # gamma_12 changes x2 -> not invariant


def t_strata_probabilities_and_X0_share_of_H():
    M = (np.random.default_rng(1).random((400000, 4)) >= 0.7).astype(np.float32); G = C8.strata(M); N = G["all"].sum()
    q = 0.3; pL = stats.binom(8, q).cdf(1); exact = {"H": pL ** 2, "X0": 0.7 ** 16, "L1:L0": 0.7 ** 8, "nobs<6": stats.binom(32, q).cdf(5)}
    for g, v in exact.items():
        f = G[g].sum() / N; assert abs(f - v) < 5 * np.sqrt(v * (1 - v) / N) + 1e-4, (g, f, v)
    assert abs(exact["X0"] / exact["H"] - 0.0509) < 5e-4
    assert (G["X0"] <= G["H"]).all() and np.array_equal(G["H_minus_X0"], G["H"] & ~G["X0"])
    assert np.array_equal(G["both_L2+"] | G["one_L2+"] | G["H"], G["all"]) and not (G["both_L2+"] & G["one_L2+"]).any()


def t_TA_estimator_one_sixth():
    k = 1.0; _, _, mu, sd = C8.norm_stats(k); rng = np.random.default_rng(2); s = rng.normal(size=(5000, 6)) * sd + mu
    St = (s - mu) / sd; f = St + rng.normal(scale=0.3, size=St.shape); z = C8.sym0_pred(k, len(s))
    assert np.allclose(z[0], -mu / sd) and np.allclose((St - z) ** 2, (s / sd) ** 2, atol=1e-4)
    cs = C8.SYM_COORDS["X0"]; tau = (((St - f) ** 2).sum(1) - ((St + mu / sd)[:, cs] ** 2).sum(1)) / 6
    sse_f, sse_0 = ((St - f) ** 2).sum(0), ((St - z) ** 2).sum(0); TA = (sse_f.sum() - sse_0[cs].sum()) / (6 * len(s))   # analyze8 formula
    assert abs(tau.mean() - TA) < 1e-5 and abs(TA - (((St - f) ** 2).mean() - ((s[:, cs] / sd[cs]) ** 2).sum(1).mean() / 6)) < 1e-5


def t_stats_helpers_and_gate_logic():
    from experiments.phase8.analyze8 import bounds, sign_test, tost, gate
    d = np.random.default_rng(3).normal(0.1, 1, 24); b = bounds(d, 0.025); se = d.std(ddof=1) / np.sqrt(24)
    assert abs(b["lcb"] - (d.mean() - stats.t.ppf(0.975, 23) * se)) < 1e-12 and abs(b["ci95_hi"] - (d.mean() + stats.t.ppf(0.975, 23) * se)) < 1e-12
    assert abs(sign_test(d) - stats.binomtest(int((d > 0).sum()), 24, 0.5).pvalue) < 1e-12
    rng = np.random.default_rng(4)
    assert gate({1: np.full(24, 0.02) + rng.normal(0, 1e-3, 24), 2: rng.normal(0, 1e-3, 24)}, rng.normal(0, 1e-3, 24))["status"] == "FAIL"
    small = {1: rng.normal(0, 1e-3, 24), 2: rng.normal(0, 1e-3, 24)}; assert gate(small, (small[1] + small[2]) / 2)["status"] == "PASS"
    noisy = {1: rng.normal(0, 0.05, 24), 2: rng.normal(0, 0.05, 24)}; g = gate(noisy, (noisy[1] + noisy[2]) / 2)
    assert g["status"] in ("UNDETERMINED", "FAIL") and not tost(noisy[1], 0.01, 0.05)["equivalent"]


def t_r1_readout_identical_to_phase6_rule():
    import experiments.phase6.evaluate6 as E6
    E6.DEV = torch.device("cpu"); rng = np.random.default_rng(5); F, Y, Fv, Yv = rng.normal(size=(600, 7)), rng.normal(size=(600, 6)), rng.normal(size=(200, 7)), rng.normal(size=(200, 6))
    a = E6.mlp_fit(F, Y, Fv, Yv, 0, 128)(Fv); b = C8.fit_mlp(F, Y, Fv, Yv, 0, 128, torch.device("cpu"))(Fv); assert np.array_equal(a, b)


def t_encoders_past_only():
    torch.manual_seed(0); m = ArmModel(2, [0, 1], np.ones((2, 8), np.float32), latent=8); X = torch.randn(400, 2); M = torch.ones(400, 2)
    z0, _ = encode_series(m, X, M); X2 = X.clone(); X2[200:] += 5; z1, _ = encode_series(m, X2, M); assert torch.allclose(z0[:200], z1[:200])


def t_knn_window_layout_and_ties():
    from experiments.phase8.knn8 import windows, knn_predict
    rng = np.random.default_rng(6); O = rng.normal(size=(300, 4)).astype(np.float32); S = rng.normal(size=(300, 6)).astype(np.float32); ts = np.arange(7, 300)
    Rw, _ = windows(O, np.ones_like(O), ts); R = torch.tensor(Rw); RS = torch.tensor(S[ts]); g = torch.Generator().manual_seed(0)
    q = np.array([50, 120]); Q, QM = windows(O, np.ones_like(O), q)
    P = knn_predict(torch.tensor(Q), torch.tensor(QM), R, RS, [1], g)[1]; assert np.allclose(P, S[q], atol=1e-6)        # exact match -> own state
    Pz = knn_predict(torch.tensor(Q), torch.zeros_like(torch.tensor(QM)), R, RS, [5], g)[5]; assert np.isfinite(Pz).all()   # all-missing: random tie-break


def t_d5b_conditional_mask_sampler_exact():
    from experiments.phase8.knn8 import sample_window_mask, in_stratum
    rng = np.random.default_rng(7); X = [sample_window_mask(0.7, "H", rng) for _ in range(20000)]
    Y = []
    while len(Y) < 20000:
        m = (rng.random((8, 4)) >= 0.7).astype(np.float32)
        if in_stratum(m[:, 0].sum(), m[:, 2].sum(), "H"):
            Y.append(m)
    X, Y = np.stack(X), np.stack(Y)
    assert all(in_stratum(m[:, 0].sum(), m[:, 2].sum(), "H") for m in X[:500])
    for f in (lambda A: A[:, :, 0].sum(1), lambda A: A[:, :, 1].sum(1), lambda A: A.sum((1, 2))):   # L1, z1 count, n_obs distributions
        a, b = f(X), f(Y); assert abs(a.mean() - b.mean()) < 0.05 and abs(a.std() - b.std()) < 0.05
    assert abs(X[:, :, 0].mean(0) - Y[:, :, 0].mean(0)).max() < 0.02                                      # positions uniform given count
    assert sample_window_mask(0.3, "X0", rng)[:, [0, 2]].sum() == 0                                      # P(X0) = 4e-9 still exact


def t_full_task_plan_and_dry_run_no_side_effects():
    root = os.path.join(TMP, "dry_full_root"); env = {k_: v for k_, v in ENV.items() if k_ != "P8_SMOKE"}; env["P8_ROOT"] = root
    out = run(["experiments/phase8/sched8.py", "24", "--dry"], env=env).stdout.splitlines()
    assert out[0] == "1276 tasks" and not os.path.exists(root)
    kinds = [l.split(" |")[0].split(":")[0] + (":" + l.split(":")[1] if l.startswith(("orc", "knn")) else "") for l in out[1:]]
    cnt = {k_: kinds.count(k_) for k_ in set(kinds)}
    assert cnt["train"] == 864 and cnt["orc:lat"] == 192 and cnt["eval"] == 96 and cnt["p7imp"] == 96 and "analyze" not in cnt
    assert all(cnt[f"orc:{x}"] == 4 for x in ("raw", "own1", "own2", "imp", "sat")) and cnt["knn:eval"] == cnt["knn:d5b"] == 4


def t_p7imp_pipeline_with_mock_weights():
    """exercises eval_p7imp8 end-to-end WITHOUT the trained P7 weights (random init) -> no Phase 7 outcome is computed pre-freeze."""
    code = r'''
import sys, torch, numpy as np, pandas as pd
import experiments.phase8.eval_p7imp8 as P, experiments.phase7.core7 as C7
from experiments.phase5.core import ArmModel
def fake_load(path, map_location=None, weights_only=False):
    arm = path.split("_")[-2]; D = C7.build(1.0, 0.7, 0, arm); torch.manual_seed(0)
    return {"state_dict": ArmModel(4, D["outs"], D["lagmask"]).state_dict(), "loss_cols": D["loss_cols"]}
P.torch.load = fake_load; sys.argv = ["x", "--kappa", "1.0", "--p", "0.7", "--rep", "0"]; P.main()
from experiments.phase8.core8 import EVAL
d = pd.read_csv(f"{EVAL}/p7imp_k1.0_p0.7_r0.csv.gz")
assert set(d.method) == {"zero", "LOCF", "P7:Bfull", "P7:Bself1", "P7:Bself2"}
assert set(d[d.method == "P7:Bself1"].view) == {0, 1} and set(d[d.method == "P7:Bself2"].view) == {2, 3}   # own views only
assert set(d[d.method == "P7:Bfull"].view) == {0, 1, 2, 3} and (d.n >= 0).all()
print("MOCK_OK")
'''
    r = run(["-c", code]); assert "MOCK_OK" in r.stdout
    os.remove(f"{TMP}/eval/p7imp_k1.0_p0.7_r0.csv.gz")          # mock output removed (random weights; meaningless)


def t_scheduler_reservation_guards_and_priority():
    import time as _t
    code = r'''
import os, sys, time
import experiments.phase8.sched8 as S
S.tasks = lambda: [("sleeper", ["-c", "import time; time.sleep(120)"], [], "/nonexistent/p8_never")]
os.environ["HARD_DEADLINE_EPOCH"] = str(time.time() + 4); os.environ["DEADLINE_EPOCH"] = str(time.time() + 2)
sys.argv = ["x", "1"]; t0 = time.time(); S.main(); print("ELAPSED", time.time() - t0)
'''
    t0 = _t.time(); r = run(["-c", code]); assert _t.time() - t0 < 60
    log = open(f"{TMP}/logs/sched8.log").read(); assert "killed sleeper (hard deadline; not a failure, resumable)" in log and "EXIT hard deadline" in log
    assert not os.path.exists(f"{TMP}/failed/sleeper.txt")                         # deadline kill is NOT a permanent failure -> resumable
    env = {k_: v for k_, v in ENV.items() if k_ != "P8_SMOKE"}; env["P8_ROOT"] = os.path.join(TMP, "noroot")
    rr = run(["experiments/phase8/sched8.py", "2"], ok=False, env=env); assert rr.returncode != 0 and "REFUSED" in rr.stdout + rr.stderr
    assert not os.path.exists(os.path.join(TMP, "noroot"))
    env["P8_ROOT"] = os.path.join(TMP, "dry2"); names = [l.split(" |")[0] for l in run(["experiments/phase8/sched8.py", "24", "--dry"], env=env).stdout.splitlines()[1:]]
    last7 = max(i for i, nm in enumerate(names) if ":0.7" in nm and not nm.startswith("p7imp")); first3 = min(i for i, nm in enumerate(names) if ":0.3" in nm)
    assert last7 < first3                                                         # whole p = 0.7 pipeline before p = 0.3
    for p in ("0.7", "0.3"):
        idx = [i for i, nm in enumerate(names) if f":{p}" in nm and not nm.startswith("p7imp")]
        tr = [i for i in idx if names[i].startswith("train")]; dn = [i for i in idx if names[i].startswith(("eval", "orc:lat"))]
        assert max(dn) < min(tr)                                                  # downstream preferred over further training
    REQ = ("orc:raw", "orc:own1", "orc:own2", "orc:imp", "orc:lat", "eval", "train"); EXT = ("orc:sat", "knn:eval", "knn:d5b")
    req7 = [i for i, nm in enumerate(names) if ":0.7" in nm and nm.startswith(REQ)]; ext7 = [i for i, nm in enumerate(names) if ":0.7" in nm and nm.startswith(EXT)]
    assert len(req7) == 4 * 2 + 2 * 2 * 24 + 2 * 24 + 9 * 2 * 24 and len(ext7) == 6 and max(req7) < min(ext7)   # T1 before T2 (exploratory)
    assert max(ext7) < first3 and names[-1].startswith("p7imp")                  # T2 before p = 0.3; Phase 7 imputation last
    from experiments.phase8.sched8 import tasks_tiered
    old = C8.SZ; C8.SZ = C8.FULL
    try:
        tiers = [t for t, _ in tasks_tiered()]; assert tiers == sorted(tiers) and set(tiers) == {1, 2, 3, 4}   # tier 5 (P7 imputation) absent in smoke; checked via dry list
    finally:
        C8.SZ = old


def t_atomic_marker_last_and_launcher_refuses():
    src = {f: open(f"experiments/phase8/{f}").read() for f in ("evaluate8.py", "knn8.py", "oracle8.py", "train8.py", "eval_p7imp8.py", "gen8.py")}
    assert '("pair", PAIR), ("imp", IMP), ("lat", LAT), ("acc", ACC)' in src["evaluate8.py"] and "os.replace" in src["evaluate8.py"]
    for f, s in src.items():
        assert "os.replace" in s, f
    assert src["knn8.py"].index("knn_ksel_k") < src["knn8.py"].index('f = f"{EVAL}/knn_k')
    root = os.path.join(TMP, "nofreeze")                                          # isolated root: valid before AND after Gate C
    r = subprocess.run(["bash", "experiments/phase8/launch_p8.sh", "2026-10-29 17:53"], env=dict(os.environ, P8_ROOT=root), capture_output=True, text=True)
    assert r.returncode == 2 and "REFUSED: no freeze file" in r.stdout and not os.path.exists(root)


def t_duplicate_runs_refused():
    import time as _t
    root = os.path.join(TMP, "dup"); env = dict(ENV, P8_ROOT=root)
    code = r'''
import sys
import experiments.phase8.sched8 as S
S.tasks = lambda: [("sleeper", ["-c", "import time; time.sleep(25)"], [], "/nonexistent/p8_never2")]
sys.argv = ["x", "1"]; S.main()
'''
    p1 = subprocess.Popen([PY, "-c", code], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for _ in range(60):
        if os.path.exists(f"{root}/logs/sched8.log") and "START" in open(f"{root}/logs/sched8.log").read():
            break
        _t.sleep(0.5)
    r2 = run(["experiments/phase8/sched8.py", "1"], ok=False, env=env)          # second scheduler on the same ROOT
    p1.kill(); p1.wait(); subprocess.run(["pkill", "-u", str(os.getuid()), "-f", "p8_never2|time.sleep\\(25\\)"])
    assert r2.returncode != 0 and "REFUSED: another sched8" in r2.stdout + r2.stderr
    root2 = os.path.join(TMP, "dup2"); os.makedirs(root2); lk = f"{root2}/run.lock"; open(lk, "w").close()
    holder = subprocess.Popen(["flock", lk, "sleep", "30"], start_new_session=True); _t.sleep(1)
    r = subprocess.run(["bash", "experiments/phase8/launch_p8.sh", "2026-10-29 17:53"], env=dict(os.environ, P8_ROOT=root2), capture_output=True, text=True)
    os.killpg(holder.pid, 9); holder.wait()                                        # kill flock AND its child (which holds the lock)
    assert r.returncode == 3 and "REFUSED: another Phase 8 run holds" in r.stdout, r.stdout + r.stderr
    dummy = subprocess.Popen([PY, "-c", "import time; time.sleep(30)", "experiments/phase8/train8.py"]); _t.sleep(1)
    r = subprocess.run(["bash", "experiments/phase8/launch_p8.sh", "2026-10-29 17:53"], env=dict(os.environ, P8_ROOT=root2), capture_output=True, text=True)
    dummy.kill(); dummy.wait()
    assert r.returncode == 3 and "REFUSED: a Phase 8 worker process is still running" in r.stdout, r.stdout + r.stderr


def t_smoke_never_uses_gpu():
    assert C8.SMOKE and C8.device().type == "cpu"
    src = open("experiments/phase8/sched8.py").read(); assert 'smoke = os.environ.get("P8_SMOKE") == "1"' in src and 'gpus = [""] if smoke else' in src


# ------------------------------------------------------------------ smoke end-to-end (CPU)
def t_smoke_end_to_end():
    run(["experiments/phase8/gen8.py", "32"])
    # data-dependent unit checks
    D = C8.build(1.0, 0.7, 0, "Bfull"); I = C8.ids(0)
    assert np.allclose(D["Xtr"], C8.nz_obs(1.0, C8.load(1.0, I["train"]))) and D["Xtr"].shape[1] == 4                    # hidden y never in inputs
    Ref, nref = C8.load_ref(1.0); assert Ref.shape == (C8.SZ["N_REF"], 8, 6) and all(C8.ref_time(1.0, q) == nref[q] for q in range(5))
    real = C8.ids
    try:                                                    # sham swap: donor := real  =>  Sham arms reduce exactly to Real arms
        C8.ids = lambda r: {**real(r), "dtrain": real(r)["train"], "dval": real(r)["val"], "dtest": real(r)["test"]}
        for j in (1, 2):
            A, B = C8.build(0.0, 0.7, 1, f"BfullSham{j}"), C8.build(0.0, 0.7, 1, "Bfull")
            assert all(np.array_equal(A[x], B[x]) for x in ("Xtr", "Mtr", "Xva", "Mva", "Hva")) and np.array_equal(A["test"](1, 1)[0], B["test"](1, 1)[0])
            A, B = C8.build(0.0, 0.7, 1, f"A{j}_8d"), C8.build(0.0, 0.7, 1, f"A{j}_8"); assert all(np.array_equal(A[x], B[x]) for x in ("Xtr", "Mtr", "Xva", "Hva"))
            assert np.array_equal(C8.state_target(0.0, real(1)["train"], real(1)["train"], j), C8.state_target(0.0, real(1)["train"], real(1)["train"]))
    finally:
        C8.ids = real
    S1, S0 = C8.build(1.0, 0.7, 0, "BfullSham1"), C8.build(1.0, 0.7, 0, "Bfull")
    assert np.array_equal(S1["Mtr"], S0["Mtr"]) and np.array_equal(S1["Xtr"][:, :2], S0["Xtr"][:, :2]) and not np.array_equal(S1["Xtr"][:, 2:], S0["Xtr"][:, 2:])
    import experiments.phase8.oracle8 as O8
    seen = []; orig = O8.load
    O8.load = lambda k, j: (seen.append(j), orig(k, j))[1]
    try:
        O8.pool_arrays(1.0, 0.7, "raw", "train"); O8.pool_arrays(1.0, 0.7, "imp", "val")
    finally:
        O8.load = orig
    assert set(seen) <= set(C8.pool_ids()["train"] + C8.pool_ids()["val"])                                              # oracles never see test/replicate data
    _, _, w = O8.pool_arrays(1.0, 0.7, "imp", "val"); Mv = C8.pool_mask(1.0, 0.7, "val", C8.pool_ids()["val"][0], 0, C8.SZ["N_VAL"])
    assert np.array_equal(w[:C8.SZ["N_VAL"] - SPAN], 1 - Mv[SPAN:])                                                     # O_imp loss only on missing entries
    import pandas as pd, json
    bad = ("1.0", "0.7", 0, "Bfull"); eid = f"k1.0_p0.7_r0_Bfull_s{C8.init_seed('Bfull', 0)}"
    # (1) FAILURE INJECTION, direct: invalid run -> rc 3, NO completion marker, NO checkpoint, failure record written
    ri = run(["experiments/phase8/train8.py", "--kappa", "1.0", "--p", "0.7", "--rep", "0", "--arm", "Bfull"], ok=False, env=dict(ENV, P8_INJECT="nonfinite:1.0:0.7:Bfull:0"))
    assert ri.returncode == 3 and "INVALID" in ri.stdout
    assert not os.path.exists(f"{TMP}/runs/rows/{eid}.csv") and not os.path.exists(f"{TMP}/runs/ckpt/{eid}.pt") and os.path.exists(f"{TMP}/runs/failed/{eid}.csv")
    # (2) full scheduler run WITH the injected fault: task fails permanently, its dependants never start, analysis = PARTIAL
    run(["experiments/phase8/sched8.py", "6"], env=dict(ENV, P8_INJECT="nonfinite:1.0:0.7:Bfull:0")); log = open(f"{TMP}/logs/sched8.log").read()
    assert "end train:1.0:0.7:0:Bfull rc=3" in log and os.path.exists(f"{TMP}/failed/train_1.0_0.7_0_Bfull.txt")
    assert "start eval:1.0:0.7:0" not in log and "start orc:lat:1.0:0.7:0:Bfull" not in log and not os.path.exists(f"{TMP}/eval/k1.0_p0.7_r0_acc.csv.gz")
    assert "start eval:1.0:0.7:1" in log                                           # other replicates unaffected
    A = lambda: f"{TMP}/analysis/" + open(f"{TMP}/analysis/LATEST").read().strip()
    a1 = A(); st = json.load(open(f"{a1}/p8_status.json")); assert os.path.basename(a1).startswith("partial_") and st["status"] == "partial"
    assert "train:1.0:0.7:0:Bfull" in st["failed_tasks"] and "eval:1.0:0.7:0" in st["missing"] and len(st["failed_ae_runs"]) == 1
    cc = pd.read_csv(f"{a1}/p8_cell_completeness.csv"); x = cc[(cc.kappa == 1.0) & (cc.p == 0.7)].iloc[0]; assert x.evaluated_reps == C8.SZ["R"] - 1 and bool(x.inconclusive)
    assert len(pd.read_csv(f"{a1}/p8_invalid_runs.csv")) == 1 and pd.read_csv(f"{a1}/p8_cells.csv").n_invalid_runs.sum() == 1
    need = ["p8_familyI_sequence.csv", "p8_familyII_sequence.csv", "p8_gates.csv", "p8_TA_LB.csv", "p8_imputation.csv", "p8_GSR_descriptive.csv",
            "p8_mask_decomposition.csv", "p8_knn_vs_oraw.csv", "p8_rho_phi.csv", "p8_summary.txt", "p8_status.json"]
    assert all(os.path.exists(f"{a1}/{f}") for f in need), os.listdir(a1)
    rows = pd.read_csv(f"{a1}/p8_training_rows.csv"); assert len(rows) == 9 * 2 * 2 * C8.SZ["R"] - 1 and rows.smoke.all() and rows.valid.all()
    assert set(rows.n_params[rows.arm == "Bfull"]) == {57840} and set(rows.n_params[rows.arm == "A1_8"]) == {27944}
    # (3) restart WITHOUT the fault: the permanently failed task is NOT retried; nothing recomputed; analysis UP-TO-DATE
    n0 = log.count("start "); run(["experiments/phase8/sched8.py", "6"]); log = open(f"{TMP}/logs/sched8.log").read()
    assert log.count("start ") == n0 and "analysis rc=0 ANALYZE8 UP-TO-DATE" in log
    # (4) RESUME after a deadline kill: one evaluation lost its marker -> only that task reruns; a NEW partial analysis; old one kept
    os.remove(f"{TMP}/eval/k0.0_p0.3_r1_acc.csv.gz"); run(["experiments/phase8/sched8.py", "6"]); log2 = open(f"{TMP}/logs/sched8.log").read()[len(log):]
    assert [l.split("start ")[1].split(" gpu")[0] for l in log2.splitlines() if " start " in l] == ["eval:0.0:0.3:1"]
    a2 = A(); assert a2 != a1 and os.path.isdir(a1) and os.path.basename(a2).startswith("partial_")
    # (5) after the failure is cleared (documented manual step), resume completes the cell -> COMPLETE analysis; partial ones preserved
    os.remove(f"{TMP}/failed/train_1.0_0.7_0_Bfull.txt"); os.rename(f"{TMP}/runs/failed/{eid}.csv", f"{TMP}/runs/failed_cleared_{eid}.csv")
    run(["experiments/phase8/sched8.py", "6"]); log3 = open(f"{TMP}/logs/sched8.log").read()
    a3 = A(); assert os.path.basename(a3).startswith("complete_") and json.load(open(f"{a3}/p8_status.json"))["status"] == "complete"
    assert os.path.isdir(a1) and os.path.isdir(a2) and len(glob.glob(f"{TMP}/analysis/partial_*")) == 2
    assert "start train:1.0:0.7:0:Bfull" in log3[len(log2) + len(log):] and os.path.exists(f"{TMP}/eval/k1.0_p0.7_r0_acc.csv.gz")
    # (6) analysis must REFUSE smoke artefacts when not in smoke mode, writing nothing
    nd = len(os.listdir(f"{TMP}/analysis")); env = {k_: v for k_, v in ENV.items() if k_ != "P8_SMOKE"}; rr = run(["experiments/phase8/analyze8.py"], ok=False, env=env)
    assert rr.returncode != 0 and "REFUSED" in (rr.stdout + rr.stderr) and len(os.listdir(f"{TMP}/analysis")) == nd


def t_previous_phases_untouched():
    for line in open("results/extra/phase7/PROTOCOL_FREEZE_p7b.txt"):
        if re.match(r"^[0-9a-f]{32}  ", line):
            h, f = line.split(); assert hashlib.md5(open(f, "rb").read()).hexdigest() == h, f
    assert hashlib.sha256(open("results/notes/phase8_mathematical_protocol.md", "rb").read()).hexdigest().startswith("c5e37088")


if __name__ == "__main__":
    only = sys.argv[1:]
    for n, f in list(globals().items()):
        if n.startswith("t_") and (not only or n in only):
            f(); print("PASS", n, flush=True)
    print("ALL PASS; smoke root", TMP)
