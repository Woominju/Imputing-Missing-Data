"""Gate F analysis (protocol §10): per-seed paired weekly-block bootstrap of mean per-variable z-RMSE contrasts.
SV = composite of single-view models (view k scored from SV_k). Outputs results/extra/phase5/external/analysis/."""
import os, numpy as np, pandas as pd

D = "results/extra/phase5/external"; OUT = f"{D}/analysis"; B = 2000
GN = ["g1", "g2-3", "g4-7", "g8+"]
os.makedirs(OUT, exist_ok=True)


def load(ds, p, s, m):
    if m == "SV":
        V = np.load(f"{D}/{ds}_p{p}_s{s}_MV.npz")["SSE"].shape[1]; SSE = N = None
        for k in range(V):
            z = np.load(f"{D}/{ds}_p{p}_s{s}_SV{k}.npz")
            if SSE is None:
                SSE, N = np.zeros_like(z["SSE"]), np.zeros_like(z["N"])
            SSE[:, k], N[:, k] = z["SSE"][:, k], z["N"][:, k]
        return SSE, N
    z = np.load(f"{D}/{ds}_p{p}_s{s}_{m}.npz"); return z["SSE"], z["N"]


def metric(SSE, N, w, g=None, k=None):
    """SSE, N: (blocks, V, G); w: block weights (bootstrap counts). Mean over views of RMSE, or one view."""
    S = np.einsum("b,bvg->vg", w, SSE); n = np.einsum("b,bvg->vg", w, N)
    if g is not None:
        S, n = S[:, g], n[:, g]
    else:
        S, n = S.sum(1), n.sum(1)
    r = np.sqrt(S / np.maximum(n, 1))
    return r[k] if k is not None else r.mean()


rows, lv = [], []
rng = np.random.default_rng(20261028)
for ds in ["beijing", "jena"]:
    for p in [0.3, 0.7]:
        for s in [0, 1, 2]:
            Ms = {m: load(ds, p, s, m) for m in ["MV", "MVshift", "SV", "mean", "LOCF", "linear_FUTURE"]}
            nb, V = Ms["MV"][0].shape[:2]
            W = np.stack([np.bincount(rng.integers(0, nb, nb), minlength=nb) for _ in range(B)]).astype(float)
            one = np.ones(nb)
            scopes = [("all", None, None)] + [(GN[g], g, None) for g in range(4)] + [(f"view{k}", None, k) for k in range(V)]
            for sc, g, k in scopes:
                for m, (S_, N_) in Ms.items():
                    lv.append(dict(ds=ds, p=p, seed=s, scope=sc, model=m, zrmse=metric(S_, N_, one, g, k),
                                   n=float(N_[:, :, g].sum() if g is not None else (N_[:, k].sum() if k is not None else N_.sum()))))
                for a, b in [("MV", "MVshift"), ("MV", "SV"), ("MVshift", "SV"), ("MV", "LOCF")]:
                    d0 = metric(*Ms[a], one, g, k) - metric(*Ms[b], one, g, k)
                    bd = np.array([metric(*Ms[a], w, g, k) - metric(*Ms[b], w, g, k) for w in W])
                    lo, hi = np.percentile(bd, [2.5, 97.5])
                    rows.append(dict(ds=ds, p=p, seed=s, scope=sc, contrast=f"{a}-{b}", diff=d0, lo=lo, hi=hi,
                                     excl0=bool(lo > 0 or hi < 0)))
C = pd.DataFrame(rows); L = pd.DataFrame(lv)
C.to_csv(f"{OUT}/contrasts.csv", index=False); L.to_csv(f"{OUT}/levels.csv", index=False)
# summary: across-seed mean and whether all 3 seeds' CIs exclude 0 on the same side
summ = C.groupby(["ds", "p", "scope", "contrast"]).apply(lambda x: pd.Series(dict(
    mean_diff=x["diff"].mean(), min_diff=x["diff"].min(), max_diff=x["diff"].max(),
    all3_neg=bool((x["hi"] < 0).all()), all3_pos=bool((x["lo"] > 0).all())))).reset_index()
summ.to_csv(f"{OUT}/summary.csv", index=False)
pd.set_option("display.width", 200); pd.set_option("display.max_rows", 500)
print(summ[summ.scope.isin(["all"] + GN)].to_string(float_format=lambda v: f"{v:.4f}"))
print(L[L.scope == "all"].groupby(["ds", "p", "model"]).zrmse.agg(["mean", "min", "max"]).to_string(float_format=lambda v: f"{v:.4f}"))
