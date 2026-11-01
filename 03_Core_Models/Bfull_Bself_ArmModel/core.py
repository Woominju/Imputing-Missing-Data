"""Phase 5 core: replicate data assembly, arm model, series encoding. src/ and earlier experiment modules are NOT modified.

Views (observed, in this column order): x1, z1, x2, z2. Hidden: y1, y2 (and the full 6-D state) -- used only by probes.
Arms (protocol section 4): A_j (own in, own out), B_j (all in, own out), Bcur_j (all in, other views lag>=1 always masked,
own out), C (all in, all out), Sham_j (own real + other-subsystem views from an independent trajectory, own out).
Masks: entry-level MCAR on the full (time x view) grid, then windows are cut (src.delay_embed), so overlapping windows
share mask values. Normalisation: mean/sd of OBSERVED entries of the training input matrix of that run."""
import hashlib, os
import numpy as np, torch, torch.nn as nn
from src.models import group_mlp, mlp
from src.delay_embedding import delay_embed, pad_front

DDIR = "results/extra/phase5/data"; M_, TAU = 8, 1; SPAN = (M_ - 1) * TAU
OBS = [0, 2, 3, 5]; VIEWS = ["x1", "z1", "x2", "z2"]; SUB = {1: [0, 1], 2: [2, 3]}
ARMS = ["A1", "A2", "B1", "B2", "Bcur1", "Bcur2", "C", "Sham1", "Sham2"]
P_INDEX = {0.3: 0, 0.7: 1}


def load(k, j):
    return np.load(f"{DDIR}/k{float(k)}_id{j}.npz")["X"]


def seed_of(*parts):
    return int(hashlib.sha256(repr(parts).encode()).hexdigest()[:8], 16)


def mcar(shape, p, seed):
    return (np.random.default_rng(seed).random(shape) >= p).astype(np.float32)


def arm_spec(arm):
    """-> (input view indices (into the 4 views), output view indices, lag-mask (n_in, m) of 0/1, sham?)"""
    j = int(arm[-1]) if arm[-1] in "12" else None
    own, oth = (SUB[j], SUB[3 - j]) if j else (None, None)
    if arm.startswith("A"):
        return own, [0, 1], np.ones((2, M_), np.float32), False   # outputs index into `ins` (bug fix 10:37: was `own`)
    if arm == "C":
        return [0, 1, 2, 3], [0, 1, 2, 3], np.ones((4, M_), np.float32), False
    ins = own + oth                                           # own views first, then other views
    lm = np.ones((4, M_), np.float32)
    if arm.startswith("Bcur"):
        lm[2:, 1:] = 0                                        # other views: lag 0 only
    return ins, list(range(2)), lm, arm.startswith("Sham")   # outputs index into `ins` (own = first two)


def test_masks(i, p, extra=0):
    base = 9000 + i if extra == 0 else 9100 + 10 * i + extra
    return mcar((5000, 4), p, base + 100000 * P_INDEX[p])


def build(k, p, r, arm):
    """Returns dict with normalised inputs/masks for train, val (+fixed val hide), and a function for test trajectories.
    Column order of the returned matrices = arm input order."""
    ins, outs, lm, sham = arm_spec(arm)
    j = int(arm[-1]) if arm[-1] in "12" else None
    def assemble(real_id, sham_id):
        X4 = load(k, real_id)[:, OBS]
        if sham:
            X4o = load(k, sham_id)[:, OBS]
            X4 = X4.copy(); X4[:, SUB[3 - j]] = X4o[:len(X4), SUB[3 - j]]
        return X4[:, ins]
    Xtr = assemble(r, (r + 1) % 8); Xva = assemble(100 + r, 100 + (r + 1) % 8)
    Mtr = mcar((len(Xtr), 4), p, seed_of("train", k, p, r))[:, ins]          # identical 4-col mask for every arm
    Mva = mcar((len(Xva), 4), p, seed_of("val", k, p, r))[:, ins]
    mu = np.array([Xtr[Mtr[:, c] > 0, c].mean() for c in range(len(ins))]); sd = np.array([Xtr[Mtr[:, c] > 0, c].std() for c in range(len(ins))])
    nz = lambda X: ((X - mu) / sd).astype(np.float32)
    Hva = Mva * (np.random.default_rng(seed_of("valhide", k, p, r)).random(Mva.shape) < 0.2).astype(np.float32)
    Hva[:, [c for c in range(len(ins)) if c not in outs]] = 0                # extra hiding only on target columns
    def test(i, extra=0):
        X = assemble(200 + i, 200 + (i + 1) % 10)
        return nz(X), test_masks(i, p, extra)[:, ins]
    return dict(ins=ins, outs=outs, lagmask=lm, Xtr=nz(Xtr), Mtr=Mtr, Xva=nz(Xva), Mva=Mva, Hva=Hva, mu=mu, sd=sd, test=test)


class ArmModel(nn.Module):
    """FDR-Learned with n_in input views and n_out decoded views (full past delay vector per decoded view).
    Encoder E_i: group MLP (2m -> hidden -> latent); learned mask-aware fusion (Model F rule: views with no observed lag are
    zeroed, observed fraction w_i appended); decoder: group MLP (latent -> hidden -> m) for each output view."""
    uses_delay = True

    def __init__(self, n_in, outs, lagmask, m=M_, latent=16, hidden=64):
        super().__init__()
        self.enc = group_mlp(n_in, 2 * m, hidden, latent)
        self.fuse = mlp(n_in * latent + n_in, hidden, latent)
        self.dec = group_mlp(len(outs), latent, hidden, m)
        self.register_buffer("lagmask", torch.tensor(lagmask))
        self.n_in, self.n_out, self.m = n_in, len(outs), m
        self.outs = list(outs)

    def encode(self, V, MV):
        MV = MV * self.lagmask; V = V * MV
        zi = self.enc(torch.cat([V, MV], -1))
        w = MV.sum(-1) / self.m; a = (w > 0).float()
        return self.fuse(torch.cat([(zi * a[..., None]).flatten(-2), w], -1))

    def decode_full(self, z):
        return self.dec(z.unsqueeze(-2).expand(*z.shape[:-1], self.n_out, z.shape[-1]))

    def forward_full(self, V, MV):
        return self.decode_full(self.encode(V, MV))

    def forward(self, V, MV):
        return self.forward_full(V, MV)[..., 0]


@torch.no_grad()
def encode_series(model, X, M, chunk=4096):
    """X, M: (T, n_in) tensors; past-only windows (front-padded as unobserved). Returns z (T, d), lag-0 output (T, n_out)."""
    model.eval(); Xp, Mp = pad_front(X * M, M, M_, TAU); T = X.shape[0]; zs, ys = [], []
    for s in range(0, T, chunk):
        e = min(T, s + chunk)
        V, MV = delay_embed(Xp[None, s:e + SPAN], Mp[None, s:e + SPAN], M_, TAU)
        z = model.encode(V, MV)[0]; zs.append(z); ys.append(model.decode_full(z)[..., 0])
    return torch.cat(zs), torch.cat(ys)


def raw_features(X, M, cols, lags=M_):
    """Past-only raw masked window features [values*mask, mask] for the given columns and first `lags` lags; (T, 2*len*lags)."""
    Xt, Mt = torch.as_tensor(X), torch.as_tensor(M)
    Xp, Mp = pad_front(Xt * Mt, Mt, M_, TAU); V, MV = delay_embed(Xp[None], Mp[None], M_, TAU)
    V, MV = V[0][:, cols, :lags], MV[0][:, cols, :lags]
    return torch.cat([(V * MV).flatten(1), MV.flatten(1)], 1).numpy()


def code_md5():
    fs = ["experiments/phase5/core.py", "experiments/phase5/train.py", "src/delay_embedding.py", "src/models.py", "experiments/fdr/loss.py"]
    return hashlib.md5(b"".join(open(f, "rb").read() for f in fs if os.path.exists(f))).hexdigest()[:8]
