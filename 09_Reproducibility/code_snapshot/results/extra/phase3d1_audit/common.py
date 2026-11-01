"""Phase 3D-1 shared helpers (CPU only). Reuses the 8 Phase 3C checkpoints read-only; never writes outside phase3d1_audit/."""
import numpy as np, pandas as pd, torch
from experiments.fdr import xz_data
from experiments.fdr.model import FDRModel
from src.delay_embedding import delay_embed, pad_front
from src.models import build_model

torch.set_num_threads(8)
P3C = "results/extra/phase3c_hidden_state"; OUT = "results/extra/phase3d1_audit"
m, tau, span, P, BLOCK = 8, 1, 7, 0.3, 100
X3 = xz_data.load_raw()
RUNS = pd.read_csv(f"{P3C}/runs.csv")


def load_model(r):
    ck = torch.load(f"{P3C}/checkpoints/{r.exp_id}.pt", map_location="cpu", weights_only=False); c = ck["cfg"]
    n = 2 if r.vars == "xz" else 1
    mdl = FDRModel(n, m, c["latent"], c["hidden"]) if r.decoder == "full" else build_model("C", n, m, c["latent"], c["hidden"], c["window"], c["direction"])
    mdl.load_state_dict(ck["state_dict"]); mdl.eval(); return mdl, c


@torch.no_grad()
def full_outputs(mdl, X, M, chunk=2048):
    """Front-padded as src.evaluate.impute. Returns (T, n, m_out) decoder outputs (m_out = 1 for C) and (T, n, m) input masks."""
    Xp, Mp = pad_front(X * M, M, m, tau); T = X.shape[0]; out, mv = [], []
    for s in range(0, T, chunk):
        e = min(T, s + chunk)
        V, MV = delay_embed(Xp[None, s:e + span], Mp[None, s:e + span], m, tau)
        z = mdl.encoder(V, MV)
        o = mdl.decode_full(z) if hasattr(mdl, "decode_full") else mdl.decode(z)[..., None]
        out.append(o[0]); mv.append(MV[0])
    return torch.cat(out).numpy().astype(np.float64), torch.cat(mv).numpy()


def block_boot(stat, T, B=1000, L=BLOCK, seed=0):
    """Circular moving-block bootstrap over test time. stat(idx) -> float. Returns (2.5%, 97.5%)."""
    rng = np.random.default_rng(seed); nb = int(np.ceil(T / L)); v = []
    for _ in range(B):
        idx = ((rng.integers(0, T, nb)[:, None] + np.arange(L)[None]) % T).ravel()[:T]
        v.append(stat(idx))
    return float(np.quantile(v, .025)), float(np.quantile(v, .975))


def rmse_ratio(num, den):
    """num, den: per-time arrays; returns idx -> RMSE over the resampled times."""
    return lambda idx: float(np.sqrt(num[idx].sum() / max(den[idx].sum(), 1)))
