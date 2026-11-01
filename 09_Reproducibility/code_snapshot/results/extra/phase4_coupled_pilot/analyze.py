"""Phase 4 analysis of eval_models_<tag>.csv: per-condition tables (seed values, never pooled windows), 2x2 factorial
contrasts per (kappa, traj, p, seed) with paired circular block bootstrap (block 100 test steps, 1000 resamples) on the
per-time arrays in preds_eval/. Contrasts: decoder main effect (FDR - C, mean over fusion), fusion main effect
(Learned - Scalar, mean over decoder), interaction ((FDR-L - C-L) - (FDR-S - C-S)). Negative = lower error for the first term."""
import sys
import numpy as np, pandas as pd

O = "results/extra/phase4_coupled_pilot"; tag = sys.argv[1] if len(sys.argv) > 1 else "pilot"
E = pd.read_csv(f"{O}/eval_models_{tag}.csv"); C = pd.read_csv(f"{O}/eval_controls_{tag}.csv")
MODELS = ["C-Scalar", "C-Learned", "FDR-Scalar", "FDR-Learned"]
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500); pd.set_option("display.max_columns", 50)
MET = ["imp_rmse_all", "imp_rmse_sub1", "imp_rmse_sub2", "ss_bothmiss_err_all", "ss_bothmiss_diff_all", "lat_mlp_y1", "lat_mlp_y2",
       "lat_knn_y1", "lat_knn_y2", "lat_ridge_y1", "lat_ridge_y2", "lat_knn_s6", "knn10_overlap_latent_vs_state6", "lat_knn_y2_t+1",
       "lat_sub1only_knn_y2", "lat_sub2only_knn_y1", "raw_decoder_rmse_at_observed", "imp_rmse_bidir_FUTURE"]
print("## per-model means over seeds (traj 0) and seed range")
for (k, tj, p), g in E.groupby(["kappa", "traj", "p"]):
    print(f"\n### kappa={k} traj={tj} p={p}  n_seeds={g.seed.nunique()}")
    print(g.groupby("model")[MET].mean().reindex(MODELS).T.round(4).to_string())
print("\n## identifiability-level same-state consistency (both-missing coordinates; mean over seeds)")
lev = [c for c in E.columns if c.startswith("ss_bothmiss_") and ("_L" in c)]
print(E.groupby(["kappa", "p", "model"])[lev].mean().round(3).T.to_string())
print("\n## controls (mean over seeds)")
print(C.groupby(["kappa", "traj", "p"]).mean(numeric_only=True).drop(columns="seed").T.round(4).to_string())


def boot(fn, T, B=1000, L=100, seed=0):
    rng = np.random.default_rng(seed); nb = int(np.ceil(T / L)); v = []
    for _ in range(B):
        idx = ((rng.integers(0, T, nb)[:, None] + np.arange(L)[None]) % T).ravel()[:T]; v.append(fn(idx))
    return np.quantile(v, [.025, .975])


R = []
for (k, tj, p, s), g in E.groupby(["kappa", "traj", "p", "seed"]):
    if set(g.model) != set(MODELS):
        continue
    A = {mo: np.load(f"{O}/preds_eval/{g[g.model == mo].exp_id.iloc[0]}.npz") for mo in MODELS}
    for met, num, den in (("imputation", "imp_num", "imp_den"), ("bothmiss_err", "ss_num", "ss_den"), ("bothmiss_diff", "ssd_num", "ss_den"),
                          ("knn_y1", "knn_y1_err2", None), ("knn_y2", "knn_y2_err2", None)):
        def val(mo, idx):
            a = A[mo][num][idx]; return np.sqrt(a.sum() / A[mo][den][idx].sum()) if den else np.sqrt(a.mean())
        T = len(A["C-Scalar"][num])
        cons = {"decoder (FDR-C)": lambda i: (val("FDR-Scalar", i) + val("FDR-Learned", i) - val("C-Scalar", i) - val("C-Learned", i)) / 2,
                "fusion (Learned-Scalar)": lambda i: (val("C-Learned", i) + val("FDR-Learned", i) - val("C-Scalar", i) - val("FDR-Scalar", i)) / 2,
                "interaction": lambda i: (val("FDR-Learned", i) - val("C-Learned", i)) - (val("FDR-Scalar", i) - val("C-Scalar", i))}
        full = np.arange(T)
        for cn, fn in cons.items():
            lo, hi = boot(fn, T, seed=s)
            R.append(dict(kappa=k, traj=tj, p=p, seed=s, metric=met, contrast=cn, estimate=fn(full), ci_lo=lo, ci_hi=hi,
                          sign="lower for 1st" if hi < 0 else ("higher for 1st" if lo > 0 else "CI includes 0")))
F = pd.DataFrame(R); F.to_csv(f"{O}/factorial_contrasts_{tag}.csv", index=False)
print("\n## factorial contrasts per seed (paired block bootstrap)")
print(F.pivot_table(index=["kappa", "traj", "p", "metric", "contrast"], columns="seed", values="estimate").round(4).to_string())
print(F.groupby(["kappa", "traj", "p", "metric", "contrast"]).sign.agg(lambda s: dict(s.value_counts())).to_string())
