"""Phase 3D-2 Track A analysis: per-variant seed tables and paired block-bootstrap contrasts (per seed; same data, masks,
evaluation positions) of every variant against FDR1/8 (original FDR) and against C."""
import numpy as np, pandas as pd

O = "results/extra/phase3d2_lag_weight"; V = ["C", "Ccap", "FDR1/8", "FDR0.35", "FDR0.65", "FDR1.0"]
E = pd.read_csv(f"{O}/eval_trackA.csv")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 300)
MET = ["miss_rmse", "miss_rmse_x", "miss_rmse_z", "val_miss_rmse", "obs_rmse", "ss_bothmiss_err", "ss_bothmiss_diff", "ss_bothobs_err",
       "y_ridge", "y_knn", "y_mlp_mean_of_3", "y_mlp_sd_of_3", "bidir_mean_lags_miss_rmse_FUTURE", "full_delay_rmse_all",
       "loss_share_lag0", "loss_share_lag7", "cos_grad_lag0C_vs_otherlags", "n_params"]
print("## mean over seeds 0-4"); print(E.groupby("variant")[MET].mean().reindex(V).T.round(4).to_string())
print("\n## per seed");
for mt in ["miss_rmse", "ss_bothmiss_err", "ss_bothmiss_diff", "y_mlp_mean_of_3", "y_knn", "y_ridge", "obs_rmse"]:
    print(mt); print(E.pivot_table(index="variant", columns="seed", values=mt).reindex(V).round(4).to_string())
print("\n## FDR lag-wise missing-target RMSE (mean over seeds)")
print(E.groupby("variant")[[f"lag{k}_miss_rmse" for k in range(8)]].mean().reindex(V[2:]).round(4).to_string())
print(E.groupby("variant")[[f"loss_share_lag{k}" for k in range(8)]].mean().reindex(V[2:]).round(3).to_string())


def boot(fn, T, B=1000, L=100, seed=0):
    rng = np.random.default_rng(seed); nb = int(np.ceil(T / L)); v = []
    for _ in range(B):
        idx = ((rng.integers(0, T, nb)[:, None] + np.arange(L)[None]) % T).ravel()[:T]; v.append(fn(idx))
    return np.quantile(v, [.025, .975])


R = []
import os
for s in range(5):
    if not all(os.path.exists(f"{O}/preds_eval/{v.replace('/', 'o')}_s{s}.npz") for v in V):
        print(f"seed {s}: some variant missing (failed/absent run) -> excluded from paired contrasts"); continue
    A = {v: np.load(f"{O}/preds_eval/{v.replace('/', 'o')}_s{s}.npz") for v in V}
    for ref in ("FDR1/8", "C"):
        for v in V:
            if v == ref:
                continue
            for met, num, den in (("miss_rmse", "miss_num", "miss_den"), ("bothmiss_err", "ssmm_num", "ssmm_den"), ("bothmiss_diff", "ssmmd_num", "ssmm_den"),
                                  ("y_mlp(mean of 3 inits)", "y_mlp_err2", None), ("y_knn", "y_knn_err2", None), ("y_ridge", "y_ridge_err2", None)):
                f = lambda w, i: np.sqrt(A[w][num][i].sum() / A[w][den][i].sum()) if den else np.sqrt(A[w][num][i].mean())
                T = len(A[v][num]); fn = lambda i: f(v, i) - f(ref, i); lo, hi = boot(fn, T, seed=s)
                R.append(dict(seed=s, ref=ref, variant=v, metric=met, diff=fn(np.arange(T)), ci_lo=lo, ci_hi=hi,
                              sign="lower" if hi < 0 else ("higher" if lo > 0 else "n.s.")))
F = pd.DataFrame(R); F.to_csv(f"{O}/paired_contrasts.csv", index=False)
print("\n## paired contrasts (variant - ref); counts over 5 seeds of CI-excluding-0 direction")
print(F.groupby(["ref", "metric", "variant"]).agg(mean_diff=("diff", "mean"), signs=("sign", lambda s: dict(s.value_counts()))).to_string())
