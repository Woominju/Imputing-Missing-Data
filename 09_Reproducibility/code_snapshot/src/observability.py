"""Observability of the Lorenz system: symbolic checks and Jacobians of
(delay) observation maps along the trajectory.

All Jacobians are taken w.r.t. the current state s(t) in original units.
Delay rows use the flow Jacobian: d x_j(t - k dt)/d s(t) = e_j^T DPhi_{-k dt}(s(t)).
"""
import numpy as np
import sympy as sp

from .lorenz import SIGMA, RHO, BETA


def symbolic_checks():
    x, y, z, sg, rh, bt, tau = sp.symbols("x y z sigma rho beta tau", real=True)
    s = sp.Matrix([x, y, z])
    f = sp.Matrix([sg * (y - x), x * (rh - z) - y, x * y - bt * z])
    Jf = f.jacobian(s)
    xdot = f[0]
    xddot = (sp.Matrix([xdot]).jacobian(s) * f)[0]
    O = sp.Matrix([x, xdot, xddot])
    detO = sp.factor(O.jacobian(s).det())
    # small-tau delay map: x(t - k tau) ~ x - k tau xdot + (k tau)^2/2 xddot  (2nd-order Taylor)
    F = sp.Matrix([x, x - tau * xdot + tau ** 2 / 2 * xddot, x - 2 * tau * xdot + 2 * tau ** 2 * xddot])
    detF = sp.factor(F.jacobian(s).det())
    # multi-view [x(t), y(t - tau), z(t - 2tau)] to first order
    fy, fz = f[1], f[2]
    G = sp.Matrix([x, y - tau * fy, z - 2 * tau * fz])
    # G is truncated at O(tau), so only the det up to O(tau) is meaningful
    detG = sp.series(sp.expand(G.jacobian(s).det()), tau, 0, 2).removeO()
    # z-only observation: symmetry (x,y,z)->(-x,-y,z) leaves z(t) and the flow invariant
    zdot = f[2]
    zddot = (sp.Matrix([zdot]).jacobian(s) * f)[0]
    detOz = sp.factor(sp.Matrix([z, zdot, zddot]).jacobian(s).det())
    sym_inv = sp.simplify(f.subs({x: -x, y: -y}, simultaneous=True) - sp.Matrix([-f[0], -f[1], f[2]]))
    return {
        "det_D_O_x": str(detO),
        "det_D_O_x_equals_-sigma^2*x": bool(sp.simplify(detO + sg ** 2 * x) == 0),
        "det_D_F_x_tau_2nd_order": str(detF),
        "det_D_F_x_tau_equals_sigma^2*tau^3*x": bool(sp.simplify(detF - sg ** 2 * tau ** 3 * x) == 0),
        "det_D_O_multiview_1st_order": str(detG),
        "det_D_O_z": str(detOz),
        "lorenz_Z2_symmetry_holds": bool(sym_inv == sp.zeros(3, 1)),
        "jacobian_f": str(Jf),
    }


def lorenz_f(S):
    x, y, z = S[..., 0], S[..., 1], S[..., 2]
    return np.stack([SIGMA * (y - x), x * (RHO - z) - y, x * y - BETA * z], -1)


def lorenz_jac(S):
    x, y, z = S[..., 0], S[..., 1], S[..., 2]
    J = np.zeros(S.shape[:-1] + (3, 3))
    J[..., 0, 0], J[..., 0, 1] = -SIGMA, SIGMA
    J[..., 1, 0], J[..., 1, 1], J[..., 1, 2] = RHO - z, -1.0, -x
    J[..., 2, 0], J[..., 2, 1], J[..., 2, 2] = y, x, -BETA
    return J


def step_jacobians(S, dt, sub=20):
    """DPhi_dt(s) for every row of S (N, 3) via RK4 on the variational equations."""
    N = S.shape[0]
    s = S.copy()
    P = np.broadcast_to(np.eye(3), (N, 3, 3)).copy()
    h = dt / sub

    def rhs(s, P):
        return lorenz_f(s), lorenz_jac(s) @ P

    for _ in range(sub):
        k1 = rhs(s, P)
        k2 = rhs(s + h / 2 * k1[0], P + h / 2 * k1[1])
        k3 = rhs(s + h / 2 * k2[0], P + h / 2 * k2[1])
        k4 = rhs(s + h * k3[0], P + h * k3[1])
        s = s + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        P = P + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    return P


def backward_flow_jacobians(J, kmax):
    """J[t] = DPhi_dt(s(t)). Returns B[k][t] = DPhi_{-k dt}(s(t)) for k <= kmax
    (rows t < k are NaN)."""
    T = J.shape[0]
    Jinv = np.linalg.inv(J)  # DPhi_{-dt}(s(t+1)) = J[t]^{-1}
    B = np.full((kmax + 1, T, 3, 3), np.nan)
    B[0] = np.eye(3)
    for k in range(1, kmax + 1):
        # DPhi_{-k dt}(s(t)) = DPhi_{-dt}(s(t-k+1)) ... composed: Jinv[t-k] @ B[k-1][t]
        B[k, k:] = Jinv[:T - k] @ B[k - 1, k:]
    return B


def svd_stats(A):
    sv = np.linalg.svd(A, compute_uv=False)
    smin = sv[..., -1]
    return sv, smin, sv[..., 0] / np.maximum(smin, 1e-300)
