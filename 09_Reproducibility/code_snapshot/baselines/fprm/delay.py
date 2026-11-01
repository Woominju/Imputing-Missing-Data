"""Delay embedding, ported from the official `PhaSpaRecon.m`.

MATLAB (1-based):
    function xn = PhaSpaRecon(s, tau, m)          % NOTE argument order: (s, tau, m)
    xn = zeros(m, len - (m-1)*tau);
    for i = 1:m
        xn(i,:) = s(1+(i-1)*tau : len-(m-i)*tau);
    end
and the caller transposes it (`YX = ss'`). Row k (1-based) is therefore
    [s(k), s(k+tau), ..., s(k+(m-1)*tau)]          -- a *forward* (future-direction) delay vector.

Python (0-based): row k = [s[k], s[k+tau], ..., s[k+(m-1)*tau]], k = 0 .. len-(m-1)*tau-1.
Row k is anchored at time k, i.e. its first column is s[k].
"""
import numpy as np


def pha_spa_recon(s, tau, m):
    """Return the (len-(m-1)*tau, m) matrix `PhaSpaRecon(s, tau, m)'` (already transposed)."""
    s = np.asarray(s, dtype=float).ravel()
    n_rows = len(s) - (m - 1) * tau
    if n_rows < 1:
        raise ValueError("delay time or embedding dimension is too large")
    return np.stack([s[i * tau: i * tau + n_rows] for i in range(m)], axis=1)


def n_rows(T, E, tau):
    """Number of delay vectors (= MATLAB `m = l-(E-1)*tau` in Eq. 13 of the paper)."""
    return T - (E - 1) * tau
