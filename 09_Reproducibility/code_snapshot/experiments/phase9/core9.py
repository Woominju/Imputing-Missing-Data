"""Phase 9 core (protocol results/notes/phase9_preregistered_protocol.md). Phase 5-8 modules are IMPORTED, never modified.
Single source of truth for Phase 9 constants, ids, seeds, masks, arm table and data assembly.
Stages: 'pilot' (development seeds / ids, no conclusions) and 'main' (confirmatory). Env: P9_ROOT (default results/extra/phase9),
P9_SMOKE=1 -> tiny sizes, CPU only (rows record it; analysis refuses smoke artefacts)."""
import hashlib, os
import numpy as np, torch
from experiments.phase5.core import seed_of, mcar, M_, TAU, SPAN, SUB
from experiments.phase8.core8 import nz_obs, nz_state, norm_stats, strata, fit_mlp     # frozen helpers (P8)

ROOT = os.environ.get("P9_ROOT", "results/extra/phase9"); SMOKE = os.environ.get("P9_SMOKE") == "1"
DDIR, RUNS, EVAL, ANA = f"{ROOT}/data", f"{ROOT}/runs", f"{ROOT}/eval", f"{ROOT}/analysis"
FULL = dict(R={"main": 24, "pilot": 6}, NT=8, NMASK=5, N_TRAIN=20000, N_VAL=5000, N_TEST=5000, AE_EPOCHS=300, AE_STEPS=100, AE_PATIENCE=15,
            RET_Q=1000)
SMALL = dict(R={"main": 2, "pilot": 2}, NT=2, NMASK=2, N_TRAIN=400, N_VAL=200, N_TEST=200, AE_EPOCHS=2, AE_STEPS=3, AE_PATIENCE=1, RET_Q=20)
SZ = SMALL if SMOKE else FULL
CONST = dict(lam=1.0, delta_imp=0.01, eta_ni=0.01, delta_ret=0.02, alpha=0.025, kappas=(0.0, 1.0), ps=(0.3, 0.7), primary=(1.0, 0.7),
             pilot_rho_max=0.8, pilot_imp_slack=0.02)
AE_CFG = dict(m=M_, tau=TAU, hidden=64, batch=64, targets=64, lr=1e-3, epochs=SZ["AE_EPOCHS"], steps_per_epoch=SZ["AE_STEPS"],
              patience=SZ["AE_PATIENCE"], hide_rate=0.2, clip=5.0)                      # identical to Phase 8 AE_CFG
BASE = {"main": 200000, "pilot": 210000}
SEED0 = {"main": {"P9": 9000, "Bfull": 9000, 8: 9100, 16: 9200}, "pilot": {"P9": 9500, "Bfull": 9500, 8: 9600, 16: 9700}}
ARMS9 = ["P9", "Bfull", "A1_8", "A2_8", "A1_16", "A2_16"]


def ids(stage, r):
    b = BASE[stage]; return dict(train=b + r, val=b + 100 + r, test=[b + 200 + 10 * r + i for i in range(SZ["NT"])])


def lengths():
    out = []
    for st in ("pilot", "main"):
        for r in range(SZ["R"][st]):
            I = ids(st, r); out += [(I["train"], SZ["N_TRAIN"]), (I["val"], SZ["N_VAL"])] + [(q, SZ["N_TEST"]) for q in I["test"]]
    return out


def mask_seeds(stage, k, p, r):
    return dict(train=seed_of("p9train", stage, k, p, r), val=seed_of("p9val", stage, k, p, r), valhide=seed_of("p9valhide", stage, k, p, r))


def test_mask(stage, k, p, r, i, e):
    return mcar((SZ["N_TEST"], 4), p, seed_of("p9test", stage, k, p, r, i, e))


def load(k, j):
    return np.load(f"{DDIR}/k{float(k)}_id{j}.npz")["X"]


def arm_spec(arm, stage):
    if arm in ("P9", "Bfull"):
        return dict(ins=[0, 1, 2, 3], latent=16, seed0=SEED0[stage][arm], tower=(arm == "P9"))
    j = int(arm[1]); lat = int(arm.split("_")[1])
    return dict(ins=list(SUB[j]), latent=lat, seed0=SEED0[stage][lat], tower=False, j=j)


def init_seed(arm, stage, r):
    return arm_spec(arm, stage)["seed0"] + r


def build(stage, k, p, r, arm):
    """same structure as core8.build (real views only; Phase 9 has no donor/sham arms)."""
    spec = arm_spec(arm, stage); I = ids(stage, r); ms = mask_seeds(stage, k, p, r); ins = spec["ins"]
    Xtr = nz_obs(k, load(k, I["train"]))[:, ins]; Xva = nz_obs(k, load(k, I["val"]))[:, ins]
    Mtr = mcar((len(Xtr), 4), p, ms["train"])[:, ins]; Mva = mcar((len(Xva), 4), p, ms["val"])[:, ins]
    Hva = Mva * (np.random.default_rng(ms["valhide"]).random(Mva.shape) < 0.2).astype(np.float32)
    outs = list(range(len(ins)))
    return dict(ins=ins, outs=outs, loss_cols=outs, lagmask=np.ones((len(ins), M_), np.float32), latent=spec["latent"], tower=spec["tower"],
                Xtr=Xtr, Mtr=Mtr, Xva=Xva, Mva=Mva, Hva=Hva, mask_seeds=ms)


def tower_loss(Ys, Yt, D, m, tau):
    """L_tow (design B2-B3): same anchor/lag alignment and multiplicity weight 1/c(u) as experiments.fdr.loss.fdr_masked_loss, but the
    target of raw entry u (originally MISSING, D = 1 - O) is the teacher output Yt (computed from the superset input O, no gradient).
    Ys, Yt: (B, K, n, m); D: (B, W, n). Returns (loss or None, number of target entries)."""
    from experiments.fdr.loss import multiplicity
    B, W, n = D.shape; span = (m - 1) * tau; K = W - span
    c = multiplicity(W, m, tau).to(D.device); w = torch.where(c > 0, 1.0 / c.clamp_min(1), torch.zeros_like(c))
    num = torch.zeros(B, device=D.device)
    for k in range(m):
        s = span - k * tau
        num = num + (D[:, s:s + K] * w[s:s + K, None] * (Yt[..., k] - Ys[..., k]) ** 2).flatten(1).sum(1)
    den = (D * (c > 0).float()[None, :, None]).flatten(1).sum(1); keep = den > 0
    if keep.sum() == 0:
        return None, den.sum()
    return (num[keep] / den[keep]).mean(), den.sum()


def device():
    if SMOKE:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    raise RuntimeError("Phase 9 runs require CUDA (CPU only with P9_SMOKE=1)")


def code_md5():
    fs = ["experiments/phase9/core9.py", "experiments/phase9/train9.py", "experiments/phase5/core.py", "src/delay_embedding.py", "experiments/fdr/loss.py"]
    return hashlib.md5(b"".join(open(f, "rb").read() for f in fs)).hexdigest()[:8]
