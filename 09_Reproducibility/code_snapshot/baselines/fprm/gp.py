"""Exact GP regression reproducing the defaults of MATLAB `fitrgp(X, y)` as called by the official code
(`Mdl = fitrgp(Ztrain, Ytrain)`, no name-value options) and `predict(Mdl, Z)`.

MATLAB defaults reproduced (fitrgp documentation, R2024b):
  BasisFunction   'constant'          -> mean function beta (one scalar), estimated by GLS given the kernel
  KernelFunction  'squaredexponential' -> k(x,x') = sigmaF^2 exp(-|x-x'|^2 / (2 sigmaL^2)), one shared length scale
  KernelParameters initial [mean(std(X)); std(y)/sqrt(2)]  (std with N-1 normalisation)
  Sigma (noise sd) initial std(y)/sqrt(2), SigmaLowerBound 1e-2*std(y), ConstantSigma false
  Standardize     false
  FitMethod       'exact' if n <= 2000, else 'sd' (subset of data) with ActiveSetSize min(1000, n),
                  ActiveSetMethod 'random'
  PredictMethod   'exact' if n <= 10000, else 'bcd'
  Optimizer       'quasinewton' (fminunc, BFGS) on the log marginal likelihood

Known differences from MATLAB (documented in results/notes/fprm_python_port_report.md):
  D1. Optimiser: scipy L-BFGS-B with analytic gradient instead of fminunc quasi-Newton. Same objective and
      initial point; final hyper-parameters may differ slightly (different line search / stopping rule).
  D2. Noise lower bound enforced as a box bound on log(sigma) instead of MATLAB's internal reparametrisation.
  D3. For n > 10000 MATLAB predicts with 'bcd' (iterative approximation of the exact solve); we always use the
      exact Cholesky solve, i.e. the quantity 'bcd' approximates.
  D4. After an 'sd' fit, beta and alpha for prediction are recomputed by GLS on all n training points with the
      fitted kernel/noise parameters (MATLAB's exact internal handling of beta here is not documented).
  D5. Random active-set selection uses numpy's RNG, so the subset differs from MATLAB's for the same seed.
  D6. If L-BFGS-B from the MATLAB initial point does not converge (e.g. ABNORMAL line-search termination on the
      flat large-length-scale plateau seen for x->y on Lorenz), it is restarted from two fixed alternative points
      (noise at 2x its lower bound; length scale x3) and the lowest-NLL solution is kept. The objective is
      unchanged; `opt_restarts` in `info()` records when this happened.
  D7. Numerical guard: a non-finite kernel matrix or a failed Cholesky returns NLL=+inf (the line search then backs
      off). Without this, tiny training sets (e.g. n=10 at l=100, r=0.9) drove exp() to overflow and crashed.
      (Box bounds on the log-parameters were tried and rejected: with them L-BFGS-B stopped at the initial point
      after one iteration on a 2300-point test.) MATLAB's behaviour in that regime is unknown.
  D8. Final posterior solve on all n points: if the Cholesky of K + sigma^2 I fails numerically (seen for n~16000
      densely sampled Lorenz delay vectors with sigma at its lower bound), the smallest diagonal jitter
      sigmaF^2 * 10^k, k = -12, -11, ..., -4, that makes it succeed is added and recorded as `jitter`.
      MATLAB would use the iterative 'bcd' predictor for n > 10000 instead of a Cholesky.
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

EXACT_FIT_MAX_N = 2000
EXACT_PREDICT_MAX_N = 10000
ACTIVE_SET_SIZE = 1000


def _sqdist(A, B):
    d = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * A @ B.T
    return np.maximum(d, 0.0)


def _gls(L, y):
    """beta (constant basis) and residual solve given Cholesky factor of K."""
    n = len(y)
    one = np.ones(n)
    Ki_y = cho_solve(L, y)
    Ki_1 = cho_solve(L, one)
    beta = (one @ Ki_y) / (one @ Ki_1)
    alpha = Ki_y - beta * Ki_1          # = K^{-1}(y - beta)
    return beta, alpha


def neg_log_marglik(eta, D2, y, need_grad=True):
    """eta = [log sigmaL, log sigmaF, log sigma]; beta profiled out by GLS.
    Returns NLL (and gradient wrt eta). By the envelope theorem the beta-dependence drops out of the gradient."""
    sL, sF, sn = np.exp(eta)
    n = len(y)
    with np.errstate(all="ignore"):
        Kf = sF ** 2 * np.exp(-0.5 * D2 / sL ** 2)
    K = Kf + (sn ** 2) * np.eye(n)
    fail = (np.inf, np.zeros(3)) if need_grad else np.inf
    if not np.isfinite(K).all():
        return fail
    try:
        L = cho_factor(K, lower=True, check_finite=False)
    except np.linalg.LinAlgError:
        return fail
    if not np.isfinite(L[0]).all():
        return fail
    beta, alpha = _gls(L, y)
    r = y - beta
    nll = 0.5 * r @ alpha + np.log(np.diag(L[0])).sum() + 0.5 * n * np.log(2 * np.pi)
    if not need_grad:
        return nll
    Kinv = cho_solve(L, np.eye(n), check_finite=False)
    W = Kinv - np.outer(alpha, alpha)
    dK_dL = Kf * D2 / sL ** 2           # d K / d log sigmaL
    dK_dF = 2.0 * Kf                    # d K / d log sigmaF
    g = np.array([0.5 * (W * dK_dL).sum(), 0.5 * (W * dK_dF).sum(), 0.5 * np.trace(W) * 2.0 * sn ** 2])
    return nll, g


class FitrgpLike:
    """`fitrgp(X, y)` with MATLAB defaults (see module docstring). Use `.fit(X, y)` then `.predict(Xnew)`."""

    def __init__(self, rng=None):
        self.rng = np.random.default_rng(0) if rng is None else rng

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float).ravel()
        n = len(y)
        assert np.isfinite(X).all() and np.isfinite(y).all(), "non-finite training data"
        sy = np.std(y, ddof=1)
        self.sigma_lb = 1e-2 * sy
        init = np.log([np.mean(np.std(X, axis=0, ddof=1)), sy / np.sqrt(2), sy / np.sqrt(2)])
        if n <= EXACT_FIT_MAX_N:
            self.fit_method = "exact"
            idx = np.arange(n)
        else:
            self.fit_method = "sd"
            idx = np.sort(self.rng.choice(n, size=min(ACTIVE_SET_SIZE, n), replace=False))
        D2 = _sqdist(X[idx], X[idx])
        bounds = [(None, None), (None, None), (np.log(self.sigma_lb), None)]
        init[2] = max(init[2], np.log(self.sigma_lb))
        res = minimize(neg_log_marglik, init, args=(D2, y[idx]), jac=True, method="L-BFGS-B", bounds=bounds)
        self.opt_restarts = 0
        self.first_opt = {"success": bool(res.success), "nll": float(res.fun), "message": str(res.message)}
        if not res.success:                                      # D6: fixed fallback starts
            alts = [init.copy(), init.copy()]
            alts[0][2] = np.log(2 * self.sigma_lb)
            alts[1][0] = init[0] + np.log(3.0)
            for x0 in alts:
                r2 = minimize(neg_log_marglik, x0, args=(D2, y[idx]), jac=True, method="L-BFGS-B", bounds=bounds)
                self.opt_restarts += 1
                if r2.fun < res.fun:
                    res = r2
        self.opt_result = res
        self.eta0 = init
        self.eta = res.x
        self.sigmaL, self.sigmaF, self.sigma = np.exp(res.x)
        self.nll = float(res.fun)
        # posterior weights on all n training points (exact solve; see D3, D4)
        self.Xtr = X
        K = self.sigmaF ** 2 * np.exp(-0.5 * _sqdist(X, X) / self.sigmaL ** 2) + self.sigma ** 2 * np.eye(n)
        self.jitter = 0.0
        for k in [None] + list(range(-12, -3)):                                   # D8
            try:
                if k is not None:
                    self.jitter = self.sigmaF ** 2 * 10.0 ** k
                    K[np.diag_indices(n)] += self.jitter - (self.sigmaF ** 2 * 10.0 ** (k - 1) if k > -12 else 0.0)
                L = cho_factor(K, lower=True, check_finite=False)
                break
            except np.linalg.LinAlgError:
                continue
        else:
            raise np.linalg.LinAlgError("posterior Cholesky failed even with jitter 1e-4*sigmaF^2")
        self.beta, self.alpha = _gls(L, y)
        self.predict_method = "exact" if n <= EXACT_PREDICT_MAX_N else "exact (MATLAB default would be bcd)"
        return self

    def predict(self, Xnew, chunk=4096):
        Xnew = np.asarray(Xnew, float)
        out = np.empty(len(Xnew))
        for s in range(0, len(Xnew), chunk):
            Ks = self.sigmaF ** 2 * np.exp(-0.5 * _sqdist(Xnew[s:s + chunk], self.Xtr) / self.sigmaL ** 2)
            out[s:s + chunk] = self.beta + Ks @ self.alpha
        return out

    def info(self):
        return {"fit_method": self.fit_method, "predict_method": self.predict_method,
                "sigmaL": float(self.sigmaL), "sigmaF": float(self.sigmaF), "sigma": float(self.sigma),
                "beta": float(self.beta), "nll": self.nll, "opt_success": bool(self.opt_result.success),
                "opt_nit": int(self.opt_result.nit), "sigmaL0": float(np.exp(self.eta0[0])),
                "opt_restarts": self.opt_restarts, "jitter": float(self.jitter), "first_opt_success": self.first_opt["success"],
                "first_opt_nll": self.first_opt["nll"]}
