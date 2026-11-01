"""Phase 3C evaluation (CPU). The AE was trained without y; y is used here only for supervised probes / analysis.
Rows: anchor times t >= span inside each chronological split (train rows fit probes, val rows choose probe
hyper-parameters, test rows are only predicted). Inputs to every encoder/probe: X*M (the data's input mask) unless
stated otherwise. Outputs: eval_*.csv in this directory."""
import json, os, sys
import numpy as np, pandas as pd, torch
from experiments.fdr import probes, xz_data
from experiments.fdr.features import latents, raw_windows
from experiments.fdr.model import FDRModel
from src.delay_embedding import delay_embed
from src.evaluate import impute, metrics
from src.models import build_model
torch.set_num_threads(8)
O = "results/extra/phase3c_hidden_state"; m, tau, span, P = 8, 1, 7, 0.3; SIG, DT = 10.0, 0.05
BLOCK = 100                      # test block length in steps (5 time units); justified by eval_acf.csv
X3 = xz_data.load_raw(); runs = pd.read_csv(f"{O}/runs.csv")
R_probe, R_cons, R_dyn, R_block, R_imp = [], [], [], [], []


def rows_of(split, d):
    a, b = d["split"][split]; return np.arange(max(a, span), b)


def probe_all(F, yn, d, label, extra):
    """F: (T, f) features for all t. Fit on train rows, select on val rows, predict test rows. Returns test preds per probe."""
    tr, va, te = rows_of("train", d), rows_of("val", d), rows_of("test", d)
    out = {}
    for name, (pred, hp) in probes.run_all(F[tr], yn[tr], F[va], yn[va], F[te]).items():
        err = pred - yn[te]
        R_probe.append(dict(label=label, probe=name, test_rmse=float(np.sqrt((err ** 2).mean())), val_rmse=hp["val_rmse"],
                            hp=json.dumps({k: v for k, v in hp.items() if k != "val_rmse"}), **extra))
        blocks = [np.sqrt((err[i:i + BLOCK] ** 2).mean()) for i in range(0, len(err) - BLOCK + 1, BLOCK)]
        R_block.append(dict(label=label, probe=name, block_rmse=json.dumps([round(float(v), 5) for v in blocks]), **extra))
        out[name] = pred
    return out


def load_model(r):
    ck = torch.load(f"{O}/checkpoints/{r.exp_id}.pt", map_location="cpu", weights_only=False); c = ck["cfg"]
    n = 2 if r.vars == "xz" else 1
    mdl = FDRModel(n, m, c["latent"], c["hidden"]) if r.decoder == "full" else build_model("C", n, m, c["latent"], c["hidden"], c["window"], c["direction"])
    mdl.load_state_dict(ck["state_dict"]); mdl.eval(); return mdl, c


for seed in [int(s) for s in sorted(runs.seed.unique())]:
    dxz, yi = xz_data.build(X3, P, seed, "xz"); yn = yi["y_norm"].astype(np.float64)
    Xt, Mt = torch.tensor(dxz["X"]), torch.tensor(dxz["M"])
    te = rows_of("test", dxz)
    # ---- controls (per seed) ----
    base = dict(seed=seed, model="control", decoder="-", vars="-", input="-")
    R_probe.append(dict(label="mean_predictor", probe="-", test_rmse=float(np.sqrt(((yn[te] - yn[rows_of('train', dxz)].mean()) ** 2).mean())), **base))
    probe_all(raw_windows(Xt, Mt, m, tau).numpy(), yn, dxz, "raw_xz_window", {**base, "input": "natural xz"})
    probe_all(raw_windows(Xt, Mt, m, tau, cols=[0]).numpy(), yn, dxz, "raw_x_window", {**base, "input": "natural x (same x mask)"})
    Mz = Mt.clone(); Mz[:, 0] = 0
    probe_all(raw_windows(Xt, Mz, m, tau, cols=[1]).numpy(), yn, dxz, "raw_z_window", {**base, "input": "natural z only (x removed)"})
    # analytical past-only finite difference (2nd-order backward where x_t, x_{t-1}, x_{t-2} observed; else not computable)
    Mx = dxz["M"][:, 0] > 0; xr = X3[:, 0]
    ok = Mx[te] & Mx[te - 1] & Mx[te - 2]
    yfd = xr[te] + (3 * xr[te] - 4 * xr[te - 1] + xr[te - 2]) / (2 * DT) / SIG
    e = ((yfd - yi["y_raw"][te]) / yi["sd"])[ok]
    R_probe.append(dict(label="fd_backward2_x", probe="analytic", test_rmse=float(np.sqrt((e ** 2).mean())), coverage=float(ok.mean()), **base))
    for dec in ("current", "full"):                                      # random (untrained) encoders, same architecture
        torch.manual_seed(seed)
        rnd = FDRModel(2, m, 16, 64) if dec == "full" else build_model("C", 2, m, 16, 64, 8, -1)
        probe_all(latents(rnd, Xt, Mt, m, tau).numpy(), yn, dxz, f"random_encoder_{dec}", {**base, "model": "random", "decoder": dec, "vars": "xz", "input": "natural xz"})
    # ---- trained models ----
    for _, r in runs[runs.seed == seed].sort_values("tag").iterrows():
        mdl, c = load_model(r)
        d = dxz if r.vars == "xz" else xz_data.build(X3, P, seed, "x")[0]
        X_, M_ = torch.tensor(d["X"]), torch.tensor(d["M"])
        past = impute(mdl, X_, M_, m, tau).numpy(); mt = metrics(past, d, "test")
        R_imp.append(dict(tag=r.tag, seed=seed, test_rmse=mt["rmse"], rmse_x=mt["rmse_per_var"][0],
                          rmse_z=mt["rmse_per_var"][1] if r.vars == "xz" else np.nan, recorded=r.test_rmse,
                          best_epoch=r.best_epoch, final_epoch=r.final_epoch, hit_cap=r.hit_cap, n_params=r.n_params))
        info = dict(seed=seed, model=r.tag, decoder=r.decoder, vars=r.vars)
        Z = latents(mdl, X_, M_, m, tau).numpy()
        preds = probe_all(Z, yn, d, f"latent_{r.tag}", {**info, "input": "natural (own data mask)"})
        if r.vars == "xz":
            Mzr = M_.clone(); Mzr[:, 1] = 0                            # BUG FIX (2026-10-27): z removed = column 1
            Zx = latents(mdl, X_, Mzr, m, tau).numpy()                 # z removed post hoc, SAME x mask (OOD for this AE)
            probe_all(Zx, yn, d, f"latent_{r.tag}_zremoved_refit", {**info, "input": "x only, z removed post hoc (OOD)"})
            Mx_only = M_.clone(); Mx_only[:, 1] = M_[:, 1]; Mxz0 = M_.clone(); Mxz0[:, 0] = 0
            Zz = latents(mdl, X_, Mxz0, m, tau).numpy()
            probe_all(Zz, yn, d, f"latent_{r.tag}_xremoved_refit", {**info, "input": "z only, x removed (stress, Gamma)"})
        # ---- dynamics: latent_t -> y_{t+D} (kNN probe, same protocol), neighbourhood overlap with true state ----
        for D in (1, 5):
            yD = np.r_[yn[D:], np.full(D, np.nan)]
            tr_, va_, te_ = [r_[np.isfinite(yD[r_])] for r_ in (rows_of("train", d), rows_of("val", d), rows_of("test", d))]
            pr, hp = probes.knn(Z[tr_], yD[tr_], Z[va_], yD[va_], Z[te_])
            R_dyn.append(dict(**info, target=f"y(t+{D})", probe="knn", test_rmse=float(np.sqrt(((pr - yD[te_]) ** 2).mean()))))
        S = (X3 - X3[d["split"]["train"][0]:d["split"]["train"][1]].mean(0)) / X3[d["split"]["train"][0]:d["split"]["train"][1]].std(0)
        sub = te[np.random.default_rng(seed).choice(len(te), 2000, replace=False)]
        Dz = torch.cdist(torch.tensor(Z[sub]), torch.tensor(Z[sub])); Ds = torch.cdist(torch.tensor(S[sub]), torch.tensor(S[sub]))
        Dz.fill_diagonal_(np.inf); Ds.fill_diagonal_(np.inf)
        iz, is_ = Dz.topk(10, largest=False).indices, Ds.topk(10, largest=False).indices
        R_dyn.append(dict(**info, target="knn10_overlap_latent_vs_state", probe="-",
                          test_rmse=float(np.mean([len(set(iz[i].tolist()) & set(is_[i].tolist())) / 10 for i in range(len(sub))]))))
        # ---- same-state mask consistency: two fresh natural masks at rate p on test windows ----
        if r.vars == "xz":
            a, b = d["split"]["test"]; Xs = X_[a:b]; g = torch.Generator().manual_seed(4242 + seed)
            M1 = (torch.rand(Xs.shape, generator=g) >= P).float(); M2 = (torch.rand(Xs.shape, generator=g) >= P).float()
            outs = []
            for Mq in (M1, M2):
                V, MV = delay_embed((Xs * Mq)[None], Mq[None], m, tau)
                with torch.no_grad():
                    z = mdl.encoder(V, MV)[0]; o = mdl.decode(z[None])[0]
                outs.append((z.numpy(), o.numpy(), MV[0, :, 0].sum(-1).numpy()))
            Strue = Xs[span:].numpy(); ytrue = yn[a + span:b]
            # fixed y-probe (MLP refit on natural-mask train latents is costly; use ridge+knn fitted on natural latents)
            tr, va = rows_of("train", d), rows_of("val", d)
            ypred = []
            for z_, _, _ in outs:
                pk, _ = probes.knn(Z[tr], yn[tr], Z[va], yn[va], z_); ypred.append(pk)
            z1, o1, nx1 = outs[0]; z2, o2, nx2 = outs[1]
            perm = np.random.default_rng(0).permutation(len(z1)); sc = np.median(np.linalg.norm(z1 - z1[perm], axis=1))
            dz = np.linalg.norm(z1 - z2, axis=1) / sc; dR = np.linalg.norm(o1 - o2, axis=1)
            e1 = np.linalg.norm(o1 - Strue, axis=1); e2 = np.linalg.norm(o2 - Strue, axis=1)
            ey1, ey2 = np.abs(ypred[0] - ytrue), np.abs(ypred[1] - ytrue); dy = np.abs(ypred[0] - ypred[1])
            for bin_name, sel in [("all", np.ones(len(dz), bool)), ("x_obs<=4", np.minimum(nx1, nx2) <= 4), ("x_obs>=7", np.minimum(nx1, nx2) >= 7)]:
                acc = sel & (e1 <= 0.1) & (e2 <= 0.1)
                R_cons.append(dict(**info, bin=bin_name, n=int(sel.sum()), dz_q50=np.median(dz[sel]), dz_q90=np.quantile(dz[sel], .9),
                                   dR_q50=np.median(dR[sel]), exz_q50=np.median(e1[sel]), y_err_q50=np.median(ey1[sel]),
                                   dy_q50=np.median(dy[sel]), frac_accurate=acc.sum() / max(sel.sum(), 1),
                                   dz_q50_accurate=np.median(dz[acc]) if acc.any() else np.nan,
                                   dy_q50_accurate=np.median(dy[acc]) if acc.any() else np.nan))
        print("done", r.tag, seed, flush=True)
    # ACF of a probe error series (block length justification), on the raw xz control ridge errors
    err = probes.ridge(raw_windows(Xt, Mt, m, tau).numpy()[rows_of('train', dxz)], yn[rows_of('train', dxz)],
                       raw_windows(Xt, Mt, m, tau).numpy()[rows_of('val', dxz)], yn[rows_of('val', dxz)],
                       raw_windows(Xt, Mt, m, tau).numpy()[te])[0] - yn[te]
    e2_ = (err ** 2) - (err ** 2).mean(); acf = [float((e2_[:-L] * e2_[L:]).mean() / (e2_ ** 2).mean()) for L in (1, 5, 10, 20, 40, 80, 100)]
    pd.DataFrame([dict(seed=seed, lags="1,5,10,20,40,80,100", acf_sq_err=json.dumps([round(v, 3) for v in acf]))]).to_csv(
        f"{O}/eval_acf.csv", mode="a", header=not os.path.exists(f"{O}/eval_acf.csv"), index=False)
for nm, R in [("probe", R_probe), ("consistency", R_cons), ("dynamics", R_dyn), ("blocks", R_block), ("imputation", R_imp)]:
    pd.DataFrame(R).to_csv(f"{O}/eval_{nm}.csv", index=False)
print("ALL DONE")
