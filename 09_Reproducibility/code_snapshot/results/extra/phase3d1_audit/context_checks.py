"""Phase 3D-1 context checks (CPU, no models trained):
 (1) classical imputation baselines on the SAME test eval positions: LOCF (past only) and linear interpolation (uses the
     future) - the reference class for FDR's future-using lag combination in fdr_target_combination.csv;
 (2) coupled-Lorenz design facts from the EXISTING generated series (results/data/coupled_lorenz_k*.npz, dt=0.01):
     synchronisation level between subsystems for each coupling kappa;
 (3) MCAR probability that each subsystem keeps >= q observed entries of one view in an m=8 window."""
import numpy as np, pandas as pd
from math import comb
from common import X3, OUT, P
from experiments.fdr import xz_data
from src.evaluate import baseline_impute

R = []
for seed in (0, 1, 2):
    d, _ = xz_data.build(X3, P, seed, "xz"); a, b = d["split"]["test"]
    for kind in ("locf", "linear"):
        xh = baseline_impute(kind, d["X"], d["M"]); E = d["eval_mask"][a:b]
        R.append(dict(seed=seed, baseline=kind, uses_future=kind == "linear",
                      test_rmse_missing=float(np.sqrt((((xh[a:b] - d["X"][a:b]) ** 2) * E).sum() / E.sum()))))
pd.DataFrame(R).to_csv(f"{OUT}/classical_baselines_test.csv", index=False)

S = []
for k in ("0.0", "0.5", "2.0"):
    Xc = np.load(f"results/data/coupled_lorenz_k{k}_20000.npz")["X"]; s1, s2 = Xc[:, :3], Xc[:, 3:]
    dist = np.linalg.norm(s1 - s2, axis=1) / np.linalg.norm(s1.std(0))
    S.append(dict(kappa=float(k), corr_x1_x2=float(np.corrcoef(Xc[:, 0], Xc[:, 3])[0, 1]), corr_z1_z2=float(np.corrcoef(Xc[:, 2], Xc[:, 5])[0, 1]),
                  median_rel_state_distance=float(np.median(dist)), frac_time_rel_distance_below_0p05=float((dist < 0.05).mean())))
pd.DataFrame(S).to_csv(f"{OUT}/coupled_sync_check.csv", index=False)

Q = []
for p in (0.3, 0.5, 0.7):
    for q in (1, 2, 3):
        one = sum(comb(8, j) * (1 - p) ** j * p ** (8 - j) for j in range(q, 9))
        Q.append(dict(p_missing=p, min_obs_per_view_window=q, P_one_view=one, P_both_subsystem_views_independent=one ** 2))
pd.DataFrame(Q).to_csv(f"{OUT}/mcar_window_obs_prob.csv", index=False)
pd.set_option("display.width", 200)
for f in ("classical_baselines_test", "coupled_sync_check", "mcar_window_obs_prob"):
    print(pd.read_csv(f"{OUT}/{f}.csv").round(4).to_string())
