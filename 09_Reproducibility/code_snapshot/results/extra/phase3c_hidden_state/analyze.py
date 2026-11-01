"""Phase 3C aggregation: per-seed tables, paired FDR - C differences, moving-block bootstrap of the paired probe RMSE
difference over test blocks (block = 100 steps; blocks resampled with replacement, 2000 reps, rng seed 0)."""
import json, numpy as np, pandas as pd
O = "results/extra/phase3c_hidden_state"; pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
P = pd.read_csv(f"{O}/eval_probe.csv"); I = pd.read_csv(f"{O}/eval_imputation.csv"); C = pd.read_csv(f"{O}/eval_consistency.csv")
D = pd.read_csv(f"{O}/eval_dynamics.csv"); B = pd.read_csv(f"{O}/eval_blocks.csv"); A = pd.read_csv(f"{O}/eval_acf.csv")
print("## imputation (test RMSE on artificially missing x,z entries; past-only)"); print(I.round(4).to_string(index=False))
print("\n## hidden-y probes: test RMSE (y in train-sd units), per seed"); pv = P.pivot_table(index=["label", "probe"], columns="seed", values="test_rmse")
pv["mean"] = pv.mean(1); print(pv.round(4).to_string())
cov = P[P.label == "fd_backward2_x"][["seed", "test_rmse", "coverage"]]; print("\nFD baseline (2nd-order backward, coverage-limited):\n", cov.round(4).to_string(index=False))
print("\n## paired FDR - C differences of probe RMSE (xz models, natural input), per seed + block bootstrap 95% CI")
rows = []
for probe in ["ridge", "knn", "mlp"]:
    for seed in sorted(P.seed.unique()):
        bc = B[(B.label == "latent_C-xz") & (B.probe == probe) & (B.seed == seed)]; bf = B[(B.label == "latent_FDR-xz") & (B.probe == probe) & (B.seed == seed)]
        if bc.empty or bf.empty: continue
        c = np.array(json.loads(bc.block_rmse.iloc[0])) ** 2; f = np.array(json.loads(bf.block_rmse.iloc[0])) ** 2
        rc = P[(P.label == "latent_C-xz") & (P.probe == probe) & (P.seed == seed)].test_rmse.iloc[0]
        rf = P[(P.label == "latent_FDR-xz") & (P.probe == probe) & (P.seed == seed)].test_rmse.iloc[0]
        rng = np.random.default_rng(0); nb = len(c); bs = []
        for _ in range(2000):
            i = rng.integers(0, nb, nb); bs.append(np.sqrt(f[i].mean()) - np.sqrt(c[i].mean()))
        rows.append(dict(probe=probe, seed=seed, C=rc, FDR=rf, diff=rf - rc, ci_lo=np.quantile(bs, .025), ci_hi=np.quantile(bs, .975), n_blocks=nb))
print(pd.DataFrame(rows).round(4).to_string(index=False))
print("\n## multi-view: same x mask; latent probes with z removed post hoc (OOD), x-only models, raw windows (seed 0 and all)")
sel = P[P.label.str.contains("zremoved|xremoved|latent_C-x$|latent_FDR-x$|raw_x_window|raw_xz_window|raw_z_window|latent_C-xz$|latent_FDR-xz$", regex=True)]
print(sel.pivot_table(index="label", columns=["probe", "seed"], values="test_rmse").round(4).to_string())
print("\n## consistency (two fresh natural masks, test)"); print(C.round(4).to_string(index=False))
print("\n## dynamics / neighbourhood"); print(D.pivot_table(index=["model", "target"], columns="seed", values="test_rmse").round(4).to_string())
print("\n## ACF of squared probe error (raw xz ridge), lags 1,5,10,20,40,80,100"); print(A.to_string(index=False))
