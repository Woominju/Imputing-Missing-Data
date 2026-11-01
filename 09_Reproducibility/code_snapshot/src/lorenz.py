"""Lorenz and coupled-Lorenz data generation."""
import numpy as np
from scipy.integrate import solve_ivp

SIGMA, RHO, BETA = 10.0, 28.0, 8.0 / 3.0


def lorenz_rhs(t, s, sigma=SIGMA, rho=RHO, beta=BETA):
    x, y, z = s
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]


def coupled_lorenz_rhs(t, s, kappa, sigma=SIGMA, rho=RHO, beta=BETA):
    """Two identical Lorenz systems diffusively coupled through x (bidirectional).
    s = (x1, y1, z1, x2, y2, z2). This is an experimental choice, not the
    proposal's unique model."""
    x1, y1, z1, x2, y2, z2 = s
    return [sigma * (y1 - x1) + kappa * (x2 - x1), x1 * (rho - z1) - y1, x1 * y1 - beta * z1,
            sigma * (y2 - x2) + kappa * (x1 - x2), x2 * (rho - z2) - y2, x2 * y2 - beta * z2]


def integrate(rhs, s0, n_steps, dt=0.01, transient=5000, rtol=1e-10, atol=1e-12, args=()):
    """Integrate with DOP853 and return n_steps samples after dropping `transient` samples."""
    total = transient + n_steps
    t_eval = np.arange(total) * dt
    sol = solve_ivp(rhs, (0.0, t_eval[-1]), s0, method="DOP853", t_eval=t_eval,
                    rtol=rtol, atol=atol, args=args)
    assert sol.success, sol.message
    return sol.y.T[transient:]


def generate_lorenz(n_steps=20000, dt=0.01, transient=5000, s0=(1.0, 1.0, 1.0)):
    X = integrate(lorenz_rhs, list(s0), n_steps, dt, transient)
    params = dict(system="lorenz", sigma=SIGMA, rho=RHO, beta=BETA, dt=dt, transient=transient,
                  n_steps=n_steps, s0=list(s0), solver="DOP853", rtol=1e-10, atol=1e-12)
    return X, ["x", "y", "z"], params


def generate_coupled_lorenz(kappa, n_steps=20000, dt=0.01, transient=5000,
                            s0=(1.0, 1.0, 1.0, -5.0, 3.0, 20.0)):
    X = integrate(coupled_lorenz_rhs, list(s0), n_steps, dt, transient, args=(kappa,))
    params = dict(system="coupled_lorenz", kappa=kappa, sigma=SIGMA, rho=RHO, beta=BETA, dt=dt,
                  transient=transient, n_steps=n_steps, s0=list(s0), solver="DOP853",
                  rtol=1e-10, atol=1e-12)
    return X, ["x1", "y1", "z1", "x2", "y2", "z2"], params
