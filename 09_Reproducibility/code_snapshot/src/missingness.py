"""Masks, chronological splits and train-only normalisation.

Conventions (1 = observed, 0 = missing):
  M_orig : entries originally observed in the raw data (finite values)
  M      : M_orig with additional artificial MCAR missingness at rate p
           (these artificially removed entries are the evaluation targets)
  H      : training-time extra hiding, H <= M (sampled per batch in train.py)
  B      : model input mask, B = M - H
"""
import numpy as np


def mcar_mask(shape, p, rng):
    """M_{t,i} ~ Bernoulli(1 - p), independently for every entry and variable."""
    return (rng.random(shape) >= p).astype(np.float32)


def chrono_split(T, frac=(0.7, 0.15, 0.15)):
    a = int(round(T * frac[0]))
    b = int(round(T * (frac[0] + frac[1])))
    return {"train": (0, a), "val": (a, b), "test": (b, T)}


def prepare(X_raw, p, seed, frac=(0.7, 0.15, 0.15)):
    """Build normalised data and masks shared by every model for one (p, seed)."""
    rng = np.random.default_rng(seed)
    T, n = X_raw.shape
    M_orig = np.isfinite(X_raw).astype(np.float32)
    M = M_orig * mcar_mask((T, n), p, rng)
    split = chrono_split(T, frac)
    a, b = split["train"]
    # normalisation stats from *observed* train entries only
    Xtr = np.where(M[a:b] > 0, X_raw[a:b], np.nan)
    mu = np.nanmean(Xtr, 0)
    sd = np.nanstd(Xtr, 0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    Xn = (np.where(M_orig > 0, X_raw, 0.0) - mu) / sd
    Xn = np.where(M_orig > 0, Xn, 0.0).astype(np.float32)
    eval_mask = (M_orig * (1 - M)).astype(np.float32)  # artificially hidden, ground truth known
    return {"X": Xn, "M": M, "M_orig": M_orig, "eval_mask": eval_mask,
            "mu": mu.astype(np.float32), "sd": sd.astype(np.float32), "split": split, "p": p, "seed": seed}
