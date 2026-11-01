"""Phase 3D-1 §4: what the y-probe results mean. CPU; no AE training, no MLP probe refit.
 (a) determinism check: ridge and kNN probes on natural latents recomputed and compared with eval_probe_valid_rows.csv
 (b) latent geometry: PCA spectrum / participation ratio of train-standardised latents (what the probes actually see)
 (c) y from the FDR decoded delay vector WITHOUT fitting: 2nd-order backward difference y ~ x + xdot/sigma on decoded lags 0..2
 (d) same-format linear readouts (ridge, train fit / val alpha / test only): FDR decoded 8-lag (x,z) vector; completed-series
     windows (observed values kept, missing = each model's past imputation) for C and FDR; true full window (oracle ceiling)
All targets are y on test rows t >= span; y never enters the AE."""
import json, numpy as np, pandas as pd, torch
from common import RUNS, X3, P3C, OUT, P, span, m, tau, load_model, full_outputs
from experiments.fdr import probes, xz_data
from experiments.fdr.features import latents

SIG, DT = 10.0, 0.05
V = pd.read_csv(f"{P3C}/recovery/eval_probe_valid_rows.csv")
CHK, GEO, YR = [], [], []


def rows_of(d, s): a, b = d["split"][s]; return np.arange(max(a, span), b)


def window(A, t):                                  # (T, n) -> features [A(t), A(t-1), ..., A(t-7)] for rows t
    return np.concatenate([A[t - k] for k in range(m)], 1)


def ridge_fit(F, yn, d, label, extra):
    tr, va, te = rows_of(d, "train"), rows_of(d, "val"), rows_of(d, "test")
    pr, hp = probes.ridge(F[tr], yn[tr], F[va], yn[va], F[te])
    YR.append(dict(label=label, method="ridge (train fit, val alpha)", test_rmse=float(np.sqrt(((pr - yn[te]) ** 2).mean())), alpha=hp["alpha"], **extra))


for _, r in RUNS.sort_values(["seed", "tag"]).iterrows():
    seed = int(r.seed); mdl, _ = load_model(r); d, yi = xz_data.build(X3, P, seed, r.vars); yn = yi["y_norm"].astype(np.float64)
    X, M = torch.tensor(d["X"]), torch.tensor(d["M"]); Z = latents(mdl, X, M, m, tau).numpy()
    tr, va, te = rows_of(d, "train"), rows_of(d, "val"), rows_of(d, "test"); ex = dict(model=r.tag, seed=seed)
    # (a) determinism check
    for name in ("ridge", "knn"):
        pr, hp = getattr(probes, name)(Z[tr], yn[tr], Z[va], yn[va], Z[te])
        rec = V[(V.label == f"latent_{r.tag}") & (V.seed == seed) & (V.probe == name)].iloc[0]
        CHK.append(dict(**ex, probe=name, recomputed=float(np.sqrt(((pr - yn[te]) ** 2).mean())), recorded=rec.test_rmse,
                        hp_recomputed=json.dumps({k: v for k, v in hp.items() if k != "val_rmse"}), hp_recorded=rec.hp))
    # (b) geometry of the standardised train latents
    Zs = (Z[tr] - Z[tr].mean(0)) / np.where(Z[tr].std(0) > 1e-8, Z[tr].std(0), 1); ev = np.linalg.eigvalsh(np.cov(Zs.T))[::-1]
    GEO.append(dict(**ex, participation_ratio=float(ev.sum() ** 2 / (ev ** 2).sum()), top1=float(ev[0] / ev.sum()),
                    top3=float(ev[:3].sum() / ev.sum()), n_dims_90pct=int(np.searchsorted(np.cumsum(ev) / ev.sum(), 0.9) + 1),
                    raw_std_min=float(Z[tr].std(0).min()), raw_std_max=float(Z[tr].std(0).max())))
    Y, _ = full_outputs(mdl, X, M); sd, mu = d["sd"].astype(np.float64), d["mu"].astype(np.float64)
    xr = lambda A: A * sd[0] + mu[0]                                   # normalised x -> raw x
    ynorm = lambda yraw: (yraw - yi["mu"]) / yi["sd"]
    # (c) analytic y from FDR decoded lags (no fitting)
    if r.decoder == "full":
        x0, x1, x2 = (xr(Y[te, 0, k]) for k in range(3))
        yhat = ynorm(x0 + (3 * x0 - 4 * x1 + x2) / (2 * DT) / SIG)
        YR.append(dict(label="FD2 on FDR decoded lags 0..2", method="analytic, no fit", test_rmse=float(np.sqrt(((yhat - yn[te]) ** 2).mean())), **ex))
        ridge_fit(Y[:, :, :].reshape(len(Y), -1), yn, d, "FDR decoded delay vector (n x 8)", ex)
    # (d) completed series windows (observed kept, missing = past imputation), same format for C and FDR
    xh = Y[..., 0]; Xc = np.where(d["M"] > 0, d["X"], xh)
    Fc = np.zeros((len(Xc), Xc.shape[1] * m)); idx = np.arange(span, len(Xc)); Fc[idx] = window(Xc, idx)
    ridge_fit(Fc, yn, d, "completed-series window (obs kept, missing=past imputation)", ex)
    xc = xr(Xc[:, 0]); yhat = ynorm(xc[te] + (3 * xc[te] - 4 * xc[te - 1] + xc[te - 2]) / (2 * DT) / SIG)
    YR.append(dict(label="FD2 on completed series", method="analytic, no fit", test_rmse=float(np.sqrt(((yhat - yn[te]) ** 2).mean())), **ex))
    print("done", r.exp_id, flush=True)

# references that do not depend on a model (seed only changes the split stats -> identical here; computed for seed 0)
d, yi = xz_data.build(X3, P, 0, "xz"); yn = yi["y_norm"].astype(np.float64); te = rows_of(d, "test")
Ft = np.zeros((len(d["X"]), 2 * m)); idx = np.arange(span, len(Ft)); Ft[idx] = window(d["X"].astype(np.float64), idx)
ridge_fit(Ft, yn, d, "TRUE full (x,z) window, oracle (no missingness)", dict(model="reference", seed=0))
ridge_fit(Ft[:, :2], yn, d, "TRUE current (x_t,z_t), oracle", dict(model="reference", seed=0))
xr0 = X3[:, 0]; yhat = (xr0[te] + (3 * xr0[te] - 4 * xr0[te - 1] + xr0[te - 2]) / (2 * DT) / SIG - yi["mu"]) / yi["sd"]
YR.append(dict(label="FD2 on TRUE x (approximation floor)", method="analytic, no fit", test_rmse=float(np.sqrt(((yhat - yn[te]) ** 2).mean())), model="reference", seed=0))
for nm, R in (("probe_determinism_check", CHK), ("latent_geometry", GEO), ("y_readouts", YR)):
    pd.DataFrame(R).to_csv(f"{OUT}/{nm}.csv", index=False)
print("ALL DONE")
