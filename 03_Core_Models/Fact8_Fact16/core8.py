"""Phase 8 core (protocol results/notes/phase8_mathematical_protocol_v2.md). Phase 5/6/7 files are imported, never modified.
Single source of truth for constants (CONST, checked against the protocol text by tests), trajectory ids / seeds, mask seeds,
arm specifications (section 5.2), data assembly (real / donor / sham), mask-only strata (section 4.3) and the R1 readout rule
(identical to experiments.phase6.evaluate6.mlp_fit, device-agnostic copy; equality is tested).
Views (canonical column order): x1, z1, x2, z2. Hidden y1, y2 never enter inputs, targets of the AE, normalisation or loss.
Environment: P8_ROOT (default results/extra/phase8); P8_SMOKE=1 selects tiny sizes for CPU smoke tests only (rows record it and
analyze8 refuses smoke artefacts unless itself in smoke mode)."""
import hashlib, os
import numpy as np, torch
import experiments.phase6.core6 as C6
from experiments.phase5.core import seed_of, mcar, M_, TAU, SPAN, OBS, SUB

ROOT = os.environ.get("P8_ROOT", "results/extra/phase8"); SMOKE = os.environ.get("P8_SMOKE") == "1"
DDIR = f"{ROOT}/data"; RUNS = f"{ROOT}/runs"; ORC = f"{ROOT}/oracle"; EVAL = f"{ROOT}/eval"; ANA = f"{ROOT}/analysis"

FULL = dict(R=24, NT=8, NMASK=5, N_TRAIN=20000, N_VAL=5000, N_TEST=5000, NPOOL_TR=40, NPOOL_VA=8, KPOOL=4, N_REF=30000,
            ORC_EPOCH_STEPS=1000, ORC_MAX_EPOCHS=60, ORC_PATIENCE=8, AE_EPOCHS=300, AE_STEPS=100, AE_PATIENCE=15,
            KNN_STRIDE=10, D5B_MASKS=100, LATNN_Q=1000)
SMALL = dict(R=2, NT=2, NMASK=2, N_TRAIN=400, N_VAL=200, N_TEST=200, NPOOL_TR=2, NPOOL_VA=1, KPOOL=2, N_REF=150,
             ORC_EPOCH_STEPS=3, ORC_MAX_EPOCHS=2, ORC_PATIENCE=1, AE_EPOCHS=2, AE_STEPS=3, AE_PATIENCE=1,
             KNN_STRIDE=10, D5B_MASKS=3, LATNN_Q=20)
SZ = SMALL if SMOKE else FULL

# Pre-registered constants (protocol v2 sections in comments). tests/test_phase8.py checks these against the protocol text.
CONST = dict(
    delta_I=0.02,            # 4.5 Family I practical threshold (MSE, coordinate mean)
    delta_y=0.02,            # 5.3 practical threshold for CG (y_j RMSE)
    delta_6=0.05,            # 5.3 IIb-2 practical threshold (RMSE_s6)
    eps_equiv=0.01,          # 5.3 TOST equivalence margin for kappa = 0 gates (y_j RMSE)
    alpha_I=0.025,           # 4.5 one-sided, fixed sequence
    alpha_IIa=0.0125, alpha_IIb=0.0125,   # 5.3 one-sided, Bonferroni split of 0.025
    alpha_gate_sign=0.025,   # 5.3 G1: per-j exact sign test, two-sided, each
    alpha_tost=0.05,         # 5.3 G2: TOST (90% CI)
    orc_widths=(256, 512, 1024), orc_lr=1e-3, orc_batch=1024, sat_ntraj=(10, 20, 40),
    knn_k_grid=(5, 10, 20, 50, 100),
    ridge_grid=(1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0),
    V_bound=3850.6666666666665, V_bound_sharp=3080.5333333333333, coord_box=55.50,
    p_family_I=0.7, kappas=(0.0, 1.0), ps=(0.3, 0.7),
)
AE_CFG = dict(m=M_, tau=TAU, hidden=64, batch=64, targets=64, lr=1e-3, epochs=SZ["AE_EPOCHS"], steps_per_epoch=SZ["AE_STEPS"],
              patience=SZ["AE_PATIENCE"], hide_rate=0.2, clip=5.0)

# ---------------------------------------------------------------- ids / seeds (4.1)
def ids(r):
    NT = SZ["NT"]
    return dict(train=50000 + r, val=50100 + r, test=[50200 + 10 * r + i for i in range(NT)],
                dtrain=51000 + r, dval=51100 + r, dtest=[51200 + 10 * r + i for i in range(NT)])


def pool_ids():
    return dict(train=[60000 + q for q in range(SZ["NPOOL_TR"])], val=[60100 + q for q in range(SZ["NPOOL_VA"])])


def ref_ids():
    return [70000 + q for q in range(SZ["N_REF"])]


def lengths():
    """(id, n_steps) of every full trajectory Phase 8 generates (both kappas use the same ids, seeds differ by kappa)."""
    out = []
    for r in range(SZ["R"]):
        I = ids(r)
        out += [(I["train"], SZ["N_TRAIN"]), (I["val"], SZ["N_VAL"]), (I["dtrain"], SZ["N_TRAIN"]), (I["dval"], SZ["N_VAL"])]
        out += [(q, SZ["N_TEST"]) for q in I["test"] + I["dtest"]]
    P = pool_ids(); out += [(q, SZ["N_TRAIN"]) for q in P["train"]] + [(q, SZ["N_VAL"]) for q in P["val"]]
    return out


def ref_time(k, q):
    """time index n_q ~ Unif{SPAN..N_TEST-1} of reference point q (one point per independent trajectory, D5a)."""
    return int(np.random.default_rng(seed_of("p8ref", float(k), q)).integers(SPAN, SZ["N_TEST"]))


def mask_seeds(k, p, r):
    return dict(train=seed_of("p8train", k, p, r), val=seed_of("p8val", k, p, r), valhide=seed_of("p8valhide", k, p, r))


def test_mask(k, p, r, i, e):
    return mcar((SZ["N_TEST"], 4), p, seed_of("p8test", k, p, r, i, e))


def pool_mask(k, p, split, q, d, n):
    return mcar((n, 4), p, seed_of("p8pool", k, p, split, q, d))


# ---------------------------------------------------------------- data access
def load(k, j):
    return np.load(f"{DDIR}/k{float(k)}_id{j}.npz")["X"]


def load_ref(k):
    Z = np.load(f"{DDIR}/ref_k{float(k)}.npz"); return Z["S"], Z["n"]       # S: (N_REF, 8, 6) chronological windows


def norm_stats(k):
    return C6.norm_stats(k)          # FIXED constants: Phase 6 pool 900..903 (mu_o, sd_o, mu_s, sd_s); NOT population moments


def nz_obs(k, X6):
    mu, sd, _, _ = norm_stats(k); return ((X6[:, OBS] - mu) / sd).astype(np.float32)


def nz_state(k, X6):
    _, _, mu, sd = norm_stats(k); return ((X6 - mu) / sd).astype(np.float32)


# ---------------------------------------------------------------- arms (5.2)
ARMS8 = ["Bfull", "BfullSham1", "BfullSham2", "A1_8", "A2_8", "A1_8d", "A2_8d", "A1_16", "A2_16"]


def arm_spec(arm):
    """ins: canonical view indices; src[c]: 'real' or 'donor' for each canonical view c used; latent; init seed offset."""
    if arm == "Bfull":
        return dict(ins=[0, 1, 2, 3], src={c: "real" for c in range(4)}, latent=16, seed0=8000)
    if arm.startswith("BfullSham"):
        j = int(arm[-1]); o = 3 - j
        return dict(ins=[0, 1, 2, 3], src={**{c: "real" for c in SUB[j]}, **{c: "donor" for c in SUB[o]}}, latent=16, seed0=8000, j=j)
    j = int(arm[1]); lat = int(arm.split("_")[1].rstrip("d")); donor = arm.endswith("d")
    return dict(ins=list(SUB[j]), src={c: ("donor" if donor else "real") for c in SUB[j]}, latent=lat,
                seed0={8: 8100, 16: 8200}[lat], j=j)


def init_seed(arm, r):
    return arm_spec(arm)["seed0"] + r


def _assemble(k, real_id, donor_id, spec):
    Xr = load(k, real_id); Xd = load(k, donor_id) if "donor" in spec["src"].values() else None
    n = len(Xr); O = nz_obs(k, Xr)
    if Xd is not None:
        Od = nz_obs(k, Xd[:n])
        for c, s in spec["src"].items():
            if s == "donor":
                O[:, c] = Od[:, c]
    return O[:, spec["ins"]]


def state_target(k, real_id, donor_id, sham_j=None):
    """normalised 6-D readout target; for a sham composite (own j real, other subsystem from donor) -> S^sharp (P8.3 iv)."""
    S = nz_state(k, load(k, real_id))
    if sham_j is not None:
        o = 3 - sham_j; Sd = nz_state(k, load(k, donor_id)[:len(S)]); cols = [0, 1, 2] if o == 1 else [3, 4, 5]
        S = S.copy(); S[:, cols] = Sd[:, cols]
    return S


def build(k, p, r, arm):
    spec = arm_spec(arm); I = ids(r); ms = mask_seeds(k, p, r); ins = spec["ins"]
    Xtr = _assemble(k, I["train"], I["dtrain"], spec); Xva = _assemble(k, I["val"], I["dval"], spec)
    Mtr = mcar((len(Xtr), 4), p, ms["train"])[:, ins]; Mva = mcar((len(Xva), 4), p, ms["val"])[:, ins]
    Hva = Mva * (np.random.default_rng(ms["valhide"]).random(Mva.shape) < 0.2).astype(np.float32)
    outs = list(range(len(ins)))                                  # every decoded view is loss-active in every P8 arm
    test = lambda i, e=0: (_assemble(k, I["test"][i], I["dtest"][i], spec), test_mask(k, p, r, i, e)[:, ins])
    return dict(ins=ins, outs=outs, loss_cols=outs, lagmask=np.ones((len(ins), M_), np.float32), latent=spec["latent"],
                Xtr=Xtr, Mtr=Mtr, Xva=Xva, Mva=Mva, Hva=Hva, test=test, mask_seeds=ms)


# ---------------------------------------------------------------- mask-only strata (4.3)
def x_counts(M4):
    """L_j(n) = # observed x_j entries in the 8-lag window ending at n (x1 = col 0, x2 = col 2); n_obs(n) = all observed."""
    c = np.cumsum(np.r_[np.zeros((1, 4)), M4], 0); W = c[M_:] - c[:-M_]; W = np.r_[np.full((M_ - 1, 4), np.nan), W]
    return W[:, 0], W[:, 2], W.sum(1)


NOBS_BINS = [(-1, 5, "nobs<6"), (5, 9, "nobs6-9"), (9, 13, "nobs10-13"), (13, 99, "nobs>=14")]


def strata(M4):
    """dict name -> boolean array over time (rows < SPAN are False)."""
    L1, L2, n = x_counts(M4); ok = ~np.isnan(n); L1, L2, n = np.nan_to_num(L1, nan=-1), np.nan_to_num(L2, nan=-1), np.nan_to_num(n, nan=-1)
    a, b = L1 >= 2, L2 >= 2
    G = {"all": ok, "both_L2+": ok & a & b, "one_L2+": ok & (a ^ b), "H": ok & ~a & ~b,
         "X0": ok & (L1 == 0) & (L2 == 0)}
    G["H_minus_X0"] = G["H"] & ~G["X0"]
    for j, L in ((1, L1), (2, L2)):
        G[f"L{j}:L0"], G[f"L{j}:L1"], G[f"L{j}:L2+"] = ok & (L == 0), ok & (L == 1), ok & (L >= 2)
    for lo, hi, nm in NOBS_BINS:
        G[nm] = ok & (n > lo) & (n <= hi)
    return G


SYM_COORDS = {"X0": [0, 1, 3, 4], "L1:L0": [0, 1], "L2:L0": [3, 4]}   # N1: coordinates whose Bayes predictor is raw 0


def sym0_pred(k, n):
    """predictor 'raw zero' in normalised units: -mu_c / sigma_c for every coordinate (used for LB / T_A / P8.4)."""
    _, _, mu, sd = norm_stats(k); return np.tile((-mu / sd).astype(np.float32), (n, 1))


# ---------------------------------------------------------------- R1 readout (identical rule to evaluate6.mlp_fit)
def fit_mlp(Ftr, Ytr, Fva, Yva, seed, width, dev):
    mu, sd = Ftr.mean(0), Ftr.std(0); sd = np.where(sd > 1e-8, sd, 1.0)
    T = lambda F: torch.tensor((F - mu) / sd, dtype=torch.float32, device=dev)
    X, Xv = T(Ftr), T(Fva); Y = torch.tensor(Ytr, dtype=torch.float32, device=dev).reshape(len(Ytr), -1)
    Yv = torch.tensor(Yva, dtype=torch.float32, device=dev).reshape(len(Yva), -1)
    torch.manual_seed(seed); g = torch.Generator(device=dev).manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], width), torch.nn.ReLU(), torch.nn.Linear(width, width), torch.nn.ReLU(),
                              torch.nn.Linear(width, Y.shape[1])).to(dev)
    opt = torch.optim.Adam(net.parameters(), 1e-3); best, state, bad = np.inf, None, 0
    for ep in range(200):
        perm = torch.randperm(len(X), generator=g, device=dev)
        for s in range(0, len(X), 512):
            i = perm[s:s + 512]; loss = ((net(X[i]) - Y[i]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            v = float(((net(Xv) - Yv) ** 2).mean())
        if v < best - 1e-7:
            best, state, bad = v, {k: q.clone() for k, q in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 20:
                break
    net.load_state_dict(state); net.eval()
    def pred(F):
        with torch.no_grad():
            return net(T(F)).cpu().numpy()
    return pred


def fit_ridge(Ftr, Ytr, Fva, Yva):
    """R0: ridge on standardised features, lambda chosen on val (descriptive)."""
    mu, sd = Ftr.mean(0), Ftr.std(0); sd = np.where(sd > 1e-8, sd, 1.0); A = np.c_[(Ftr - mu) / sd, np.ones(len(Ftr))]
    Av = np.c_[(Fva - mu) / sd, np.ones(len(Fva))]; best = (np.inf, None)
    for lam in CONST["ridge_grid"]:
        Rg = lam * np.eye(A.shape[1]); Rg[-1, -1] = 0; Wt = np.linalg.solve(A.T @ A + Rg, A.T @ Ytr)
        v = float(((Av @ Wt - Yva) ** 2).mean()); best = min(best, (v, Wt), key=lambda z: z[0])
    Wt = best[1]
    return lambda F: np.c_[(F - mu) / sd, np.ones(len(F))] @ Wt


def device():
    if SMOKE:
        return torch.device("cpu")                    # smoke tests NEVER touch a GPU (no reservation needed)
    if torch.cuda.is_available():
        return torch.device("cuda")
    raise RuntimeError("Phase 8 runs require CUDA (CPU only in P8_SMOKE=1)")


def code_md5():
    fs = ["experiments/phase8/core8.py", "experiments/phase8/train8.py", "experiments/phase5/core.py", "src/delay_embedding.py",
          "src/models.py", "experiments/fdr/loss.py"]
    return hashlib.md5(b"".join(open(f, "rb").read() for f in fs)).hexdigest()[:8]
