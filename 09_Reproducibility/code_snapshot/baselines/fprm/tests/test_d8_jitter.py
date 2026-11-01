"""Phase-2 (2026-10-27) independent check of the D8 jitter path in `gp.py`, which test_fprm.py does not exercise.
The optimiser is replaced by a fixed result and `cho_factor` is made to fail a given number of times, so the loop
runs deterministically. Checks: (1) the diagonal actually added equals the recorded `jitter` = sigmaF^2*10^k,
(2) k is the smallest value that succeeds, (3) without failures jitter stays 0, (4) exhausting k raises.
Run: python -m baselines.fprm.tests.test_d8_jitter
"""
import time
from types import SimpleNamespace

import numpy as np
from scipy.linalg import cho_factor as real_cho_factor

import baselines.fprm.gp as G


def _fit_with_failures(n_fail):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3))
    y = np.sin(X[:, 0]) + 0.1 * rng.normal(size=60)
    eta = np.log([1.3, 0.8, 0.05])
    seen = []

    def fake_minimize(fun, x0, **kw):
        return SimpleNamespace(x=eta.copy(), fun=0.0, success=True, message="fixed", nit=0)

    def flaky_cho_factor(K, **kw):
        seen.append(np.diag(K).copy())
        if len(seen) <= n_fail:
            raise np.linalg.LinAlgError("forced")
        return real_cho_factor(K, **kw)

    orig = (G.minimize, G.cho_factor)
    G.minimize, G.cho_factor = fake_minimize, flaky_cho_factor
    try:
        gp = G.FitrgpLike().fit(X, y)
    finally:
        G.minimize, G.cho_factor = orig
    return gp, seen


def test_no_failure_no_jitter():
    gp, seen = _fit_with_failures(0)
    assert gp.jitter == 0.0 and len(seen) == 1


def test_added_diagonal_equals_recorded_jitter():
    for n_fail in [1, 2, 3, 5]:
        gp, seen = _fit_with_failures(n_fail)
        k = -12 + (n_fail - 1)
        expected = gp.sigmaF ** 2 * 10.0 ** k
        assert np.isclose(gp.jitter, expected, rtol=1e-12), (n_fail, gp.jitter, expected)
        added = seen[-1] - seen[0]
        tol = 8 * np.finfo(float).eps * seen[0].max() * (n_fail + 1)   # diagonal ~sigmaF^2; subtraction loses bits
        assert np.allclose(added, gp.jitter, rtol=1e-9, atol=tol), (n_fail, added[:3], gp.jitter)
        assert len(seen) == n_fail + 1


def test_exhausted_jitter_raises():
    try:
        _fit_with_failures(100)
    except np.linalg.LinAlgError:
        return
    raise AssertionError("expected LinAlgError after k=-4")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            t0 = time.time(); fn(); print(f"PASS {name} ({time.time() - t0:.2f}s)")
    print("ALL PASSED")
