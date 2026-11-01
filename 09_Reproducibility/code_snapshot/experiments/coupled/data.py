"""Phase 4 coupled-Lorenz data: the autoencoder sees ONLY the observed views (default x1, z1, x2, z2); the hidden state
(y1, y2 and the full 6-D state) is kept in a separate dict for post-hoc evaluation. Masks, splits and train-only
normalisation come from src.missingness.prepare applied to the observed columns only."""
import numpy as np

from src.missingness import prepare

COLS = {"x1": 0, "y1": 1, "z1": 2, "x2": 3, "y2": 4, "z2": 5}
VIEWS = ("x1", "z1", "x2", "z2")


def path(kappa, traj):
    return f"results/extra/phase4_coupled_pilot/data/coupled_k{float(kappa)}_dt0.05_traj{traj}.npz"


def load_raw(kappa, traj):
    return np.load(path(kappa, traj))["X"]


def build(X6, p, seed, views=VIEWS):
    obs = [COLS[v] for v in views]
    d = prepare(np.ascontiguousarray(X6[:, obs]), p, seed)
    d["views"] = list(views)
    a, b = d["split"]["train"]
    mu, sd = X6[a:b].mean(0), X6[a:b].std(0)
    hidden = {"state_raw": X6.astype(np.float64), "state_norm": ((X6 - mu) / sd).astype(np.float64), "mu": mu, "sd": sd, "split": d["split"]}
    return d, hidden
