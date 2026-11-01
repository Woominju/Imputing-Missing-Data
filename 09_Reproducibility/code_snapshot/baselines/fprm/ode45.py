"""Emulation of MATLAB `ode45(f, [t0 tf], y0)` with default options, used to regenerate the paper's Lorenz data.

The official generator calls `ode45` with a two-element tspan, so the output grid is the solver's own adaptive
steps with the default Refine=4 (3 interpolated points per step + the step end). The time spacing is therefore
NON-uniform. This module re-implements MATLAB's ode45 algorithm (Dormand-Prince 5(4), FSAL, infinity-norm error
control, the ntrp45 continuous extension, and MATLAB's initial-step and step-adaptation rules) with the defaults
RelTol=1e-3, AbsTol=1e-6, MaxStep=0.1*|tf-t0|, Refine=4.

Caveat: it was written from the published ode45 algorithm, not from MathWorks' source, and has NOT been checked
against MATLAB output (no MATLAB on this server). The step sequence, and hence the series length and sampling,
may differ from MATLAB's.
"""
import numpy as np

_C = np.array([0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1])
_A = [[],
      [1 / 5],
      [3 / 40, 9 / 40],
      [44 / 45, -56 / 15, 32 / 9],
      [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
      [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
      [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84]]
_B5 = np.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0])
_E = np.array([71 / 57600, 0, -71 / 16695, 71 / 1920, -17253 / 339200, 22 / 525, -1 / 40])
_BI = np.array([[1, -183 / 64, 37 / 12, -145 / 128],
                [0, 0, 0, 0],
                [0, 1500 / 371, -1000 / 159, 1000 / 371],
                [0, -125 / 32, 125 / 12, -375 / 64],
                [0, 9477 / 3392, -729 / 106, 25515 / 6784],
                [0, -11 / 7, 11 / 3, -55 / 28],
                [0, 3 / 2, -4, 5 / 2]])


def ode45_matlab(f, tspan, y0, rtol=1e-3, atol=1e-6, refine=4, max_step=None):
    t0, tf = float(tspan[0]), float(tspan[1])
    y = np.asarray(y0, float).ravel().copy()
    pow_ = 1 / 5
    threshold = atol / rtol
    hmax = 0.1 * abs(tf - t0) if max_step is None else max_step
    t = t0
    f0 = np.asarray(f(t, y), float)
    absh = min(hmax, abs(tf - t0))
    rh = np.max(np.abs(f0) / np.maximum(np.abs(y), threshold)) / (0.8 * rtol ** pow_)
    if absh * rh > 1:
        absh = 1 / rh
    ts, ys = [t], [y.copy()]
    K = np.zeros((7, len(y)))
    K[0] = f0
    done = False
    while not done:
        hmin = 16 * np.spacing(t)
        absh = min(hmax, max(hmin, absh))
        h = absh
        if 1.1 * absh >= abs(tf - t):
            h = tf - t
            absh = abs(h)
            done = True
        nofailed = True
        while True:
            for s in range(1, 6):
                K[s] = f(t + _C[s] * h, y + h * (np.asarray(_A[s]) @ K[:s]))
            tnew = t + h
            if done:
                tnew = tf
            ynew = y + h * (_B5[:6] @ K[:6])
            K[6] = f(tnew, ynew)
            err = absh * np.max(np.abs(_E @ K) / np.maximum(np.maximum(np.abs(y), np.abs(ynew)), threshold))
            if err > rtol:
                if absh <= hmin:
                    raise RuntimeError("step size underflow")
                if nofailed:
                    nofailed = False
                    absh = max(hmin, absh * max(0.1, 0.8 * (rtol / err) ** pow_))
                else:
                    absh = max(hmin, 0.5 * absh)
                h = absh
                done = False
            else:
                break
        if refine > 1:
            for j in range(1, refine):
                sj = j / refine
                ts.append(t + h * sj)
                ys.append(y + h * (K.T @ (_BI @ np.array([sj, sj ** 2, sj ** 3, sj ** 4]))))
        ts.append(tnew)
        ys.append(ynew.copy())
        if nofailed:
            temp = 1.25 * (err / rtol) ** pow_
            absh = absh / temp if temp > 0.2 else 5.0 * absh
        t, y = tnew, ynew
        K[0] = K[6]
    return np.array(ts), np.array(ys)


def lorenz_rhs(sigma=10.0, rho=28.0, beta=8 / 3):
    def f(t, X):
        x, y, z = X
        return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])
    return f


def paper_lorenz3d(x0=(0.3, 0.6, 0.9), tspan=(10, 40), discard=100):
    """3D Lorenz following the official generator recipe (`multiCoupledLorenz2` with N=1, i.e. no coupling):
    ode45 on tspan=[10 40], then `X = X(101:end,:)`. Default x0 follows the code (`0.3:0.3:...`); the paper text
    says increments of 0.03, i.e. (0.3, 0.33, 0.36). The exact 3D Lorenz script was not published."""
    t, X = ode45_matlab(lorenz_rhs(), tspan, x0)
    return t[discard:], X[discard:]
