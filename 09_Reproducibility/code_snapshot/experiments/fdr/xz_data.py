"""Phase 3C data: Lorenz observed through x and z only; y kept in a SEPARATE array for post-hoc evaluation.
The autoencoder data dict is built by src.missingness.prepare from the (x, z) columns only, so masks, splits and
normalisation never see y. The x-only variant uses column 0 of the SAME (x, z) preparation, so its x mask, evaluation
positions and normalisation are identical to those of the x,z models."""
import numpy as np

from src.missingness import prepare

DATA = "results/extra/aim_autonomous_20261027/coarse_dt0.05/data/lorenz_dt0.05_20000.npz"


def load_raw(path=DATA):
    return np.load(path)["X"]


def build(Xraw3, p, seed, variables="xz"):
    """Returns (d, yinfo). d: AE data dict over the observed variables; yinfo: hidden y, its train-split stats."""
    d = prepare(np.ascontiguousarray(Xraw3[:, [0, 2]]), p, seed)
    if variables == "x":
        d = dict(d)
        for k in ("X", "M", "M_orig", "eval_mask"):
            d[k] = np.ascontiguousarray(d[k][:, :1])
        d["mu"], d["sd"] = d["mu"][:1], d["sd"][:1]
    elif variables != "xz":
        raise ValueError(variables)
    a, b = d["split"]["train"]
    y = Xraw3[:, 1].astype(np.float64)
    yinfo = {"y_raw": y, "mu": float(y[a:b].mean()), "sd": float(y[a:b].std()), "split": d["split"]}
    yinfo["y_norm"] = ((y - yinfo["mu"]) / yinfo["sd"]).astype(np.float32)
    return d, yinfo
