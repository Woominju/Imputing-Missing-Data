"""Phase 3D-1 §5: split-boundary audit of the Phase 3C future-y probe (evaluate.py lines 86-90).
Original: yD = y shifted by D over the WHOLE series; rows t of each split kept if yD[t] is finite -> for train and val the
last D anchors have targets y(t+D) in the NEXT split. Fixed: keep t in [a, b) only if t + D < b.
Both variants are recomputed here with the identical kNN probe; the original is checked against eval_dynamics.csv.
eval_dynamics.csv and evaluate.py are not modified."""
import numpy as np, pandas as pd, torch
from common import RUNS, X3, P3C, OUT, P, span, m, tau, load_model
from experiments.fdr import probes, xz_data
from experiments.fdr.features import latents

dyn = pd.read_csv(f"{P3C}/eval_dynamics.csv"); R = []
for _, r in RUNS.sort_values(["seed", "tag"]).iterrows():
    seed = int(r.seed); mdl, _ = load_model(r); d, yi = xz_data.build(X3, P, seed, r.vars); yn = yi["y_norm"].astype(np.float64)
    Z = latents(mdl, torch.tensor(d["X"]), torch.tensor(d["M"]), m, tau).numpy()
    rows = {s: np.arange(max(a, span), b) for s, (a, b) in d["split"].items()}
    for D in (1, 5):
        yD = np.r_[yn[D:], np.full(D, np.nan)]
        orig = {s: v[np.isfinite(yD[v])] for s, v in rows.items()}
        fixed = {s: v[v + D < d["split"][s][1]] for s, v in rows.items()}
        leak = {s: int((orig[s] + D >= d["split"][s][1]).sum()) for s in rows}
        out = {}
        for name, rr in (("original", orig), ("fixed", fixed)):
            pr, hp = probes.knn(Z[rr["train"]], yD[rr["train"]], Z[rr["val"]], yD[rr["val"]], Z[rr["test"]])
            out[name] = (float(np.sqrt(((pr - yD[rr["test"]]) ** 2).mean())), hp["k"], len(rr["test"]))
        rec = dyn[(dyn.model == r.tag) & (dyn.seed == seed) & (dyn.target == f"y(t+{D})")].test_rmse.iloc[0]
        R.append(dict(model=r.tag, seed=seed, target=f"y(t+{D})", leaked_train_rows=leak["train"], leaked_val_rows=leak["val"],
                      leaked_test_rows=leak["test"], n_train_rows=len(orig["train"]), recorded_eval_dynamics=rec,
                      original_recomputed=out["original"][0], original_k=out["original"][1], fixed=out["fixed"][0], fixed_k=out["fixed"][1],
                      n_test_rows=out["fixed"][2], fixed_minus_original=out["fixed"][0] - out["original"][0]))
    print("done", r.exp_id, flush=True)
pd.DataFrame(R).to_csv(f"{OUT}/eval_dynamics_boundary_fixed.csv", index=False)
print(pd.DataFrame(R).round(5).to_string())
