"""Correctness tests for the FPRM port. Run: python -m baselines.fprm.tests.test_fprm (pytest-compatible names)."""
import sys
import time

import numpy as np
from scipy.integrate import solve_ivp

from baselines.fprm.delay import pha_spa_recon
from baselines.fprm.fprm import fprm_impute, fprm_official_run, holdout_rows, official_metrics, zscore_train
from baselines.fprm.gp import FitrgpLike, _sqdist, neg_log_marglik
from baselines.fprm.ode45 import _B5, _BI, lorenz_rhs, ode45_matlab, paper_lorenz3d


def _phaspa_matlab_literal(s, tau, m):
    """Literal 1-based transliteration of PhaSpaRecon.m, then transposed like `ss'`."""
    s = list(s)
    ln = len(s)
    xn = np.zeros((m, ln - (m - 1) * tau))
    for i in range(1, m + 1):
        lo, hi = 1 + (i - 1) * tau, ln - (m - i) * tau          # MATLAB s(lo:hi), inclusive, 1-based
        xn[i - 1, :] = s[lo - 1:hi]
    return xn.T


def test_delay_matches_matlab_indexing():
    s = np.arange(1, 31, dtype=float)                           # value == 1-based MATLAB index
    for tau, m in [(3, 5), (3, 3), (1, 4), (2, 1)]:
        P = pha_spa_recon(s, tau, m)
        assert P.shape == (30 - (m - 1) * tau, m)
        assert np.array_equal(P, _phaspa_matlab_literal(s, tau, m))
        # row k (0-based) = [s[k], s[k+tau], ...]: forward (future-direction) delays anchored at k
        k = 4
        assert np.array_equal(P[k], s[k + tau * np.arange(m)])
    # the official example call PhaSpaRecon(y,3,5) -> tau=3, E=5: first row is y(1),y(4),y(7),y(10),y(13)
    assert np.array_equal(pha_spa_recon(s, 3, 5)[0], [1, 4, 7, 10, 13])


def test_zscore_matlab_n_minus_1_and_zero_guard():
    X = np.array([[1.0, 5.0], [3.0, 5.0], [5.0, 5.0]])
    mu, sd = zscore_train(X)
    assert np.allclose(mu, [3, 5]) and np.allclose(sd, [2.0, 1.0])  # std([1 3 5]) = 2 with N-1; zero sd -> 1


def test_gp_gradient_finite_difference():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 3))
    y = np.sin(X[:, 0]) + 0.1 * rng.normal(size=60)
    D2 = _sqdist(X, X)
    eta = np.log([0.8, 0.7, 0.2])
    f, g = neg_log_marglik(eta, D2, y)
    num = np.zeros(3)
    for i in range(3):
        e = np.zeros(3)
        e[i] = 1e-6
        num[i] = (neg_log_marglik(eta + e, D2, y, False) - neg_log_marglik(eta - e, D2, y, False)) / 2e-6
    assert np.allclose(g, num, rtol=1e-4, atol=1e-5), (g, num)


def test_gp_posterior_mean_closed_form_and_defaults():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(80, 2))
    y = 3.0 + np.cos(X[:, 0]) * X[:, 1] + 0.05 * rng.normal(size=80)
    gp = FitrgpLike(rng=rng).fit(X, y)
    assert gp.fit_method == "exact" and gp.opt_result.success
    assert np.isclose(gp.info()["sigmaL0"], np.mean(np.std(X, 0, ddof=1)))   # fitrgp initial length scale
    assert gp.sigma >= gp.sigma_lb * (1 - 1e-9)
    Xs = rng.normal(size=(10, 2))
    K = gp.sigmaF ** 2 * np.exp(-0.5 * _sqdist(X, X) / gp.sigmaL ** 2) + gp.sigma ** 2 * np.eye(80)
    Ks = gp.sigmaF ** 2 * np.exp(-0.5 * _sqdist(Xs, X) / gp.sigmaL ** 2)
    Ki = np.linalg.inv(K)
    one = np.ones(80)
    beta = (one @ Ki @ y) / (one @ Ki @ one)                                   # GLS constant basis
    mean = beta + Ks @ Ki @ (y - beta)
    assert np.allclose(gp.predict(Xs), mean, atol=1e-6)
    assert np.sqrt(np.mean((gp.predict(X) - y) ** 2)) < 0.1


def test_gp_switches_to_subset_of_data_above_2000():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(2300, 3))
    y = np.sin(X[:, 0]) + X[:, 1] ** 2
    gp = FitrgpLike(rng=rng).fit(X, y)
    assert gp.fit_method == "sd"
    err = np.sqrt(np.mean((gp.predict(X[:300]) - y[:300]) ** 2))
    assert err < 0.05, err


def test_ode45_tableau_and_accuracy():
    s = np.array([1.0, 1, 1, 1])
    assert np.allclose(_BI @ s, _B5)                       # continuous extension at s=1 hits the step end
    t, Y = ode45_matlab(lorenz_rhs(), (0, 1), [1.0, 1.0, 1.0])
    assert np.all(np.diff(t) > 0) and t[-1] == 1.0
    ref = solve_ivp(lorenz_rhs(), (0, 1), [1.0, 1.0, 1.0], t_eval=t, rtol=1e-11, atol=1e-11).y.T
    assert np.max(np.abs(Y - ref)) < 0.05 * np.max(np.abs(ref)), np.max(np.abs(Y - ref))
    # default Refine=4: within every accepted step there are 3 interpolated points -> len-1 divisible by 4
    assert (len(t) - 1) % 4 == 0


def test_paper_lorenz_generation_shape():
    t, X = paper_lorenz3d()
    assert X.shape[1] == 3 and len(X) > 300 and np.isfinite(X).all()
    assert np.std(np.diff(t)) > 0                          # non-uniform sampling, as in MATLAB tspan=[t0 tf]


def test_holdout_size():
    rng = np.random.default_rng(0)
    tr, te = holdout_rows(101, 0.9, rng)
    assert te.sum() == 91 and tr.sum() == 10 and not (tr & te).any()


def _toy(T=900, seed=0):
    t, X = paper_lorenz3d()
    return X[:T, 0], X[:T, 1]


def test_impute_no_leakage_and_alignment():
    x, y = _toy()
    T = len(x)
    E, tau = 3, 3
    rng = np.random.default_rng(5)
    m = (rng.random(T) > 0.5).astype(float)
    y_nan = np.where(m > 0, y, np.nan)                     # hidden truth replaced by NaN
    y_garbage = np.where(m > 0, y, 1e6)                    # hidden truth replaced by garbage
    a, info_a = fprm_impute(x, np.where(m > 0, y, 0.0), m, E, tau, rng=np.random.default_rng(7))
    b, _ = fprm_impute(x, y_nan, m, E, tau, rng=np.random.default_rng(7))
    c, _ = fprm_impute(x, y_garbage, m, E, tau, rng=np.random.default_rng(7))
    assert np.array_equal(a, b, equal_nan=True) and np.array_equal(a, c, equal_nan=True)
    N = T - (E - 1) * tau
    assert a.shape == (T,) and np.isnan(a[N:]).all()       # tail rows undefined
    miss = (m[:N] == 0)
    assert np.isfinite(a[:N][miss]).all() and np.isnan(a[:N][~miss]).all()
    assert info_a["n_train"] == int((m[:N] > 0).sum())
    # imputations at time k really target y[k] (not a shifted index): error must be small vs shifted truth
    k = np.where(miss)[0]
    e0 = np.mean((a[k] - y[k]) ** 2)
    e_shift = np.mean((a[k[k > 0]] - y[k[k > 0] - 1]) ** 2)
    assert e0 < e_shift and e0 < 0.05 * np.var(y), (e0, e_shift)


def test_seed_reproducibility():
    x, y = _toy()
    r1 = fprm_official_run(x, y, 3, 3, 0.5, np.random.default_rng(11))
    r2 = fprm_official_run(x, y, 3, 3, 0.5, np.random.default_rng(11))
    r3 = fprm_official_run(x, y, 3, 3, 0.5, np.random.default_rng(12))
    assert r1 == r2 and r1["mse"] != r3["mse"]


def test_restart_never_worsens_likelihood():
    """Near noise-free smooth target (the case that made the first L-BFGS-B run terminate ABNORMALly)."""
    x, y = _toy(1500)
    P = pha_spa_recon(x, 3, 3)
    tgt = y[:len(P)]
    for seed in range(3):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(P), 600, replace=False)
        mu, sd = zscore_train(P[idx])
        gp = FitrgpLike(rng=rng).fit((P[idx] - mu) / sd, tgt[idx])
        assert gp.nll <= gp.first_opt["nll"] + 1e-9
        assert np.isfinite(gp.predict((P[:50] - mu) / sd)).all()


def test_tiny_training_sets_do_not_crash():
    """Regression for the overflow crash (D7): l=100, r=0.9 -> about 9 training rows."""
    _, X = paper_lorenz3d()
    X = X[:100]
    for E, tau in [(3, 3), (5, 3)]:
        for tgt in (1, 2):
            for rep in range(30):
                res = fprm_official_run(X[:, 0], X[:, tgt], E, tau, 0.9, np.random.default_rng(1_000_000 + rep))
                assert np.isfinite(res["nrmse_LL"]) and np.isfinite(res["sigmaL"])


def test_official_metrics():
    yt = np.array([1.0, 2, 3, 4])
    yf = np.array([1.5, 2, 2.5, 4])
    m = official_metrics(yt, yf)
    assert np.isclose(m["mse"], 0.125) and np.isclose(m["nrmse_LL"], np.sqrt(0.125) / np.std(yt, ddof=1))
    assert np.isclose(m["mae"], 0.25)


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            t0 = time.time()
            try:
                fn()
                print(f"PASS {name} ({time.time() - t0:.2f}s)")
            except Exception as e:  # noqa: BLE001
                fails += 1
                print(f"FAIL {name}: {type(e).__name__}: {e}")
    print("ALL PASSED" if fails == 0 else f"{fails} FAILED")
    sys.exit(1 if fails else 0)
