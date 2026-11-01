"""FPRM (Full-Partial Reconstruction Mapping) ported from the official `FPRM mian code.mlx`.

Official data flow (MATLAB, 1-based), reproduced line by line in `fprm_official_run`:
    ss  = PhaSpaRecon(y(:,2),3,5); YX = ss';    % full reconstruction  M(x_i)   (complete variable)
    ss2 = PhaSpaRecon(y(:,1),3,5); YY = ss2';   % partial reconstruction M(x_j) (incomplete variable)
    X = YX; Y = YY;
    N = length(Y);                               % = l-(E-1)*tau rows
    cvp = cvpartition(N,'Holdout',r);            % random rows -> "missing" (test); rest -> training
    Xtrain = X(idxTrn,:);  Ytrain = Y(idxTrn,1); % target = FIRST column of M(x_j) = x_j(k)
    [Ztrain,tr_mu,tr_sigma] = zscore(Xtrain);    % standardise predictors with training-row stats
    tr_sigma(tr_sigma==0) = 1;
    Mdl = fitrgp(Ztrain,Ytrain);                 % target NOT standardised
    Ztest = (X(idxTest,:)-tr_mu)./tr_sigma;  YFit = predict(Mdl,Ztest);
    L = loss(Mdl,Ztest,Ytest);  LL = sqrt(L)/std(Ytest);  CO = corr([YFit Ytest]);  MAE = mean(|Ytest-YFit|)

Note: `PhaSpaRecon(s, tau, m)` -> the official call uses tau=3, E=5. SI Table S1 lists E=3, tau=3 for the 3D Lorenz
system and E=4, tau=3 for the 120D coupled Lorenz system; the code's E=5 matches neither. Callers choose E, tau.

Missingness in the official code is at the level of *rows of the delay matrix*: row k is "missing" when the
target value x_j(k) is missing. Rows k > l-(E-1)*tau (the last (E-1)*tau time points) never exist, so those time
points can be neither trained on nor imputed.
"""
import numpy as np

from .delay import pha_spa_recon
from .gp import FitrgpLike


def zscore_train(Xtr):
    """MATLAB `zscore` (N-1 normalisation) with the official zero-sd guard."""
    mu = Xtr.mean(0)
    sd = Xtr.std(0, ddof=1)
    sd = np.where(sd == 0, 1.0, sd)
    return mu, sd


def official_metrics(y_true, y_fit):
    """Metrics exactly as computed in the official script (MATLAB std/corr use N-1)."""
    y_true = np.asarray(y_true, float)
    y_fit = np.asarray(y_fit, float)
    mse = np.mean((y_true - y_fit) ** 2)                 # loss(Mdl,...) default for regression GP = MSE
    return {"mse": float(mse),
            "nrmse_LL": float(np.sqrt(mse) / np.std(y_true, ddof=1)),   # LL = (L^0.5)./std(Ytest)
            "rho": float(np.corrcoef(y_fit, y_true)[0, 1]),             # CO = corr([YFit Ytest])
            "mae": float(np.mean(np.abs(y_true - y_fit))),
            "mean_rel_err": float(np.mean(np.abs((y_true - y_fit) / y_fit))),  # Re (relative to YFit, as coded)
            "n_test": int(len(y_true))}


def holdout_rows(N, r, rng):
    """`cvpartition(N,'Holdout',r)`: a random subset of round(r*N) rows is the test (missing) set.
    (MATLAB's exact rounding of r*N and its RNG stream are not reproduced; see report.)"""
    n_test = int(round(r * N))
    test = np.zeros(N, bool)
    test[rng.choice(N, size=n_test, replace=False)] = True
    return ~test, test


def fprm_official_run(x_full, x_incomplete, E, tau, r, rng):
    """One repetition of the official script. `x_incomplete` is the *complete* ground-truth series of the
    incomplete variable; the script itself removes rows (row-level holdout) and only evaluates on them."""
    X = pha_spa_recon(x_full, tau, E)
    Y = pha_spa_recon(x_incomplete, tau, E)
    N = len(Y)
    idx_trn, idx_test = holdout_rows(N, r, rng)
    Ytrain = Y[idx_trn, 0]
    mu, sd = zscore_train(X[idx_trn])
    gp = FitrgpLike(rng=rng).fit((X[idx_trn] - mu) / sd, Ytrain)
    yfit = gp.predict((X[idx_test] - mu) / sd)
    out = official_metrics(Y[idx_test, 0], yfit)
    out.update({"N_rows": N, "n_train": int(idx_trn.sum())})
    out.update(gp.info())
    return out


def fprm_impute(x_ref, x_tgt_masked, m_tgt, E, tau, train_rows=None, predict_rows=None, rng=None):
    """FPRM imputation of one incomplete variable on a single time series (transductive by default).

    x_ref        : (T,) fully observed reference variable (predictor).
    x_tgt_masked : (T,) target variable with missing entries already zeroed (X*M); hidden values never enter.
    m_tgt        : (T,) 1 = observed, 0 = missing for the target.
    train_rows   : optional bool (n_rows,) restricting which delay rows may be used for training
                   (official: all rows whose target is observed -> transductive).
    predict_rows : optional bool (n_rows,) rows to impute (default: all rows with missing target).
    Returns xhat (T,) with NaN where FPRM is undefined or not requested, plus diagnostics.
    """
    T = len(x_ref)
    X = pha_spa_recon(x_ref, tau, E)
    N = len(X)
    obs = np.asarray(m_tgt[:N]) > 0
    trn = obs if train_rows is None else (obs & train_rows)
    prd = (~obs) if predict_rows is None else predict_rows
    assert not (trn & ~obs).any()
    y_trn = np.asarray(x_tgt_masked[:N])[trn]          # observed target values only
    mu, sd = zscore_train(X[trn])
    gp = FitrgpLike(rng=rng).fit((X[trn] - mu) / sd, y_trn)
    xhat = np.full(T, np.nan)
    xhat[:N][prd] = gp.predict((X[prd] - mu) / sd)
    info = gp.info()
    info.update({"N_rows": N, "n_train": int(trn.sum()), "n_pred": int(prd.sum()),
                 "undefined_tail": T - N})
    return xhat, info
