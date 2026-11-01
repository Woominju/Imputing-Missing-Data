"""Phase 4 post-pilot CPU context (no training): (1) classical imputation baselines on the SAME test eval positions
(LOCF past-only; linear interpolation uses the future); (2) within-model view-removal contrast for the OWN subsystem:
kNN y1 from latents with all four views vs with only subsystem-1 views (and y2 vs subsystem-2 views), from eval_models_pilot.csv."""
import numpy as np, pandas as pd
from experiments.coupled import data as CD
from src.evaluate import baseline_impute
O = "results/extra/phase4_coupled_pilot"; R = []
for k in (0.0, 1.0):
    for p in (0.3, 0.7):
        for s in (0, 1, 2):
            d, _ = CD.build(CD.load_raw(k, 0), p, s); a, b = d["split"]["test"]; E = d["eval_mask"][a:b]
            for kind in ("locf", "linear"):
                xh = baseline_impute(kind, d["X"], d["M"])
                R.append(dict(kappa=k, p=p, seed=s, baseline=kind, uses_future=kind == "linear", imp_rmse=float(np.sqrt((((xh[a:b] - d["X"][a:b]) ** 2) * E).sum() / E.sum()))))
B = pd.DataFrame(R); B.to_csv(f"{O}/classical_baselines_pilot.csv", index=False)
print(B.groupby(["kappa", "p", "baseline"]).imp_rmse.mean().round(4).to_string())
E = pd.read_csv(f"{O}/eval_models_pilot.csv")
V = E[["kappa", "p", "model", "seed", "lat_knn_y1", "lat_sub1only_knn_y1", "lat_knn_y2", "lat_sub2only_knn_y2"]].copy()
V["y1: all4 - own-only"] = V.lat_knn_y1 - V.lat_sub1only_knn_y1; V["y2: all4 - own-only"] = V.lat_knn_y2 - V.lat_sub2only_knn_y2
V.to_csv(f"{O}/view_removal_own_subsystem.csv", index=False)
print(V.groupby(["kappa", "p", "model"])[["lat_knn_y1", "lat_sub1only_knn_y1", "y1: all4 - own-only", "lat_knn_y2", "lat_sub2only_knn_y2", "y2: all4 - own-only"]].mean().round(4).to_string())
