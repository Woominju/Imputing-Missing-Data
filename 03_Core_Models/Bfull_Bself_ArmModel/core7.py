"""Phase 7 Gate P7-B core (protocol results/notes/phase7_p7b_protocol.md). Phase 5/6 files are imported, never modified.
Arms (ALL: the same ArmModel = Phase 5 FDR-Learned; 4 input views in canonical order x1 z1 x2 z2; decoder outputs ALL 4 views
(full past delay vector each); identical parameter count, initialisation seed, mask arrays, hiding RNG stream, data, budget):
  Bself1 : FDR reconstruction loss on view columns {x1, z1} only      (decoder heads of x2, z2 receive no loss)
  Bself2 : FDR reconstruction loss on view columns {x2, z2} only
  Bfull  : FDR reconstruction loss on all 4 view columns
Only the set of loss-active target columns differs; hence the effective gradient paths and the amount of supervision differ.
Early stopping (no ground truth of missing values): lag-0 RMSE on a fixed 20% of OBSERVED val entries of the LOSS-ACTIVE
columns (same rule as Phase 5/6). Hidden y1, y2 never enter inputs, targets, normalisation or loss.
Data: NEW independent trajectories (gen6 generator/seed rule), ids 40000+; normalisation = Phase 6 G2 pool constants."""
import hashlib, os
import numpy as np
import experiments.phase6.core6 as C6
from experiments.phase5.core import seed_of, mcar, M_, OBS, SUB

R7, NT7, NMASK = 24, 8, 5
ROOT = "results/extra/phase7"; DDIR = f"{ROOT}/data"
ARMS7 = ["Bself1", "Bself2", "Bfull"]
LOSS_COLS = {"Bself1": [0, 1], "Bself2": [2, 3], "Bfull": [0, 1, 2, 3]}


def ids(r):
    return dict(train=40000 + r, val=40200 + r, test=[41000 + 100 * r + i for i in range(NT7)])


def all_ids():
    out = []
    for r in range(R7):
        I = ids(r); out += [(I["train"], 20000), (I["val"], 5000)] + [(q, 5000) for q in I["test"]]
    return out


def load(k, j):
    return np.load(f"{DDIR}/k{float(k)}_id{j}.npz")["X"]


def norm_stats(k):
    return C6.norm_stats(k)                      # Phase 6 pool 900..903 (results/extra/phase6/data), unchanged


def mask_seeds(k, p, r):
    return dict(train=seed_of("p7train", k, p, r), val=seed_of("p7val", k, p, r), valhide=seed_of("p7valhide", k, p, r))


def test_mask(k, p, r, i, e):
    return mcar((5000, 4), p, seed_of("p7test", k, p, r, i, e))


def init_seed(r):
    return 7000 + r


def build(k, p, r, arm):
    ins, outs, lc = [0, 1, 2, 3], [0, 1, 2, 3], LOSS_COLS[arm]; I = ids(r); mu, sd, _, _ = norm_stats(k)
    nz = lambda q: ((load(k, q)[:, OBS] - mu) / sd).astype(np.float32)
    ms = mask_seeds(k, p, r); Xtr, Xva = nz(I["train"]), nz(I["val"])
    Mtr = mcar((len(Xtr), 4), p, ms["train"]); Mva = mcar((len(Xva), 4), p, ms["val"])
    Hva = Mva * (np.random.default_rng(ms["valhide"]).random(Mva.shape) < 0.2).astype(np.float32)
    Hva[:, [c for c in range(4) if c not in lc]] = 0
    test = lambda i, e=0: (nz(I["test"][i]), test_mask(k, p, r, i, e))
    return dict(ins=ins, outs=outs, loss_cols=lc, lagmask=np.ones((4, M_), np.float32), Xtr=Xtr, Mtr=Mtr, Xva=Xva, Mva=Mva, Hva=Hva,
                test=test, mask_seeds=ms)


def code_md5():
    fs = ["experiments/phase7/core7.py", "experiments/phase7/train7.py", "experiments/phase5/core.py", "src/delay_embedding.py",
          "src/models.py", "experiments/fdr/loss.py"]
    return hashlib.md5(b"".join(open(f, "rb").read() for f in fs)).hexdigest()[:8]
