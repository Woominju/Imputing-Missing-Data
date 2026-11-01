"""Phase 6 G2/G3 core: replicate assembly for the independent confirmatory experiment. Reuses the Phase 5 ArmModel
(FDR-Learned, unchanged) and Phase 5 helpers; Phase 5 files are not modified.
Differences from Phase 5 (all for control validity, results/notes/phase6_g2_g3_protocol.md):
  * every replicate r has its own train target, train sham donor, val target, val donor, 8 test targets and 8 test donors
    (no trajectory is reused across replicates or roles -> replicate contrasts are independent);
  * ONE fixed normalisation per kappa (mean/sd of the observed views x1, z1, x2, z2 on the normalisation pool 900..903,
    train role) applied identically to every arm and every column, Real or Sham;
  * Arm Aslot_j: the 4-slot B architecture (same parameter count as B/Sham/Bcur) with the other-subsystem slots ALWAYS
    masked (lag mask 0) -> capacity-matched own-only control. Equal parameter count is NOT equal effective capacity.
Views (column order of the 4-view matrix): x1, z1, x2, z2."""
import hashlib, os
import numpy as np
from experiments.phase5.core import ArmModel, encode_series, raw_features, seed_of, mcar, M_, TAU, SPAN, OBS, SUB

DDIR = "results/extra/phase6/data"; R6, NT6, NMASK = 24, 8, 5
ARMS6 = ["A1", "A2", "Aslot1", "Aslot2", "B1", "B2", "Bcur1", "Bcur2", "Sham1", "Sham2"]
P_INDEX = {0.3: 0, 0.7: 1}


def load(k, j):
    return np.load(f"{DDIR}/k{float(k)}_id{j}.npz")["X"]


def ids(r):
    return dict(train=1000 + r, train_donor=1100 + r, val=1200 + r, val_donor=1300 + r,
                test=[2000 + 100 * r + i for i in range(NT6)], test_donor=[10000 + 100 * r + i for i in range(NT6)])


_NORM = {}


def norm_stats(k):
    """fixed per-kappa normalisation of the 4 observed views and of the 6-D state (state stats only for probes)."""
    if k not in _NORM:
        S = np.concatenate([load(k, 900 + q) for q in range(4)])
        _NORM[k] = (S[:, OBS].mean(0), S[:, OBS].std(0), S.mean(0), S.std(0))
    return _NORM[k]


def arm_spec(arm):
    """-> (input view indices into the 4 views, output indices into `ins`, lag mask (n_in, m), sham?)"""
    j = int(arm[-1]); own, oth = SUB[j], SUB[3 - j]
    if arm.startswith("Aslot"):
        lm = np.ones((4, M_), np.float32); lm[2:] = 0
        return own + oth, [0, 1], lm, False
    if arm.startswith("A"):
        return own, [0, 1], np.ones((2, M_), np.float32), False
    lm = np.ones((4, M_), np.float32)
    if arm.startswith("Bcur"):
        lm[2:, 1:] = 0
    return own + oth, [0, 1], lm, arm.startswith("Sham")


def mask_seeds(k, p, r):
    return dict(train=seed_of("p6train", k, p, r), val=seed_of("p6val", k, p, r), valhide=seed_of("p6valhide", k, p, r))


def test_mask(k, p, r, i, e):
    return mcar((5000, 4), p, seed_of("p6test", k, p, r, i, e))


def init_seed(r):
    return 6000 + r


def build(k, p, r, arm):
    ins, outs, lm, sham = arm_spec(arm); j = int(arm[-1]); I = ids(r); mu, sd, _, _ = norm_stats(k)
    def assemble(real_id, donor_id):
        X4 = load(k, real_id)[:, OBS]
        if sham:
            X4 = X4.copy(); X4[:, SUB[3 - j]] = load(k, donor_id)[:len(X4), OBS][:, SUB[3 - j]]
        return ((X4 - mu) / sd).astype(np.float32)[:, ins]
    ms = mask_seeds(k, p, r)
    Xtr = assemble(I["train"], I["train_donor"]); Xva = assemble(I["val"], I["val_donor"])
    Mtr = mcar((len(Xtr), 4), p, ms["train"])[:, ins]; Mva = mcar((len(Xva), 4), p, ms["val"])[:, ins]
    Hva = Mva * (np.random.default_rng(ms["valhide"]).random(Mva.shape) < 0.2).astype(np.float32)
    Hva[:, [c for c in range(len(ins)) if c not in outs]] = 0
    def test(i, e=0):
        return assemble(I["test"][i], I["test_donor"][i]), test_mask(k, p, r, i, e)[:, ins]
    return dict(ins=ins, outs=outs, lagmask=lm, Xtr=Xtr, Mtr=Mtr, Xva=Xva, Mva=Mva, Hva=Hva, test=test, mask_seeds=ms)


def code_md5():
    fs = ["experiments/phase6/core6.py", "experiments/phase6/train6.py", "experiments/phase5/core.py", "src/delay_embedding.py",
          "src/models.py", "experiments/fdr/loss.py"]
    return hashlib.md5(b"".join(open(f, "rb").read() for f in fs if os.path.exists(f))).hexdigest()[:8]
