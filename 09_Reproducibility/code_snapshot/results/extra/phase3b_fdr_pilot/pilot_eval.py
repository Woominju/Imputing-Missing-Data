"""Phase 3B: evaluate the 2x2 pilot checkpoints (CPU). Same data, masks, evaluation positions for all four models.
  official test RMSE (archived metric, lag-0 past-only impute) and, for FDR, bidirectional RMSE (separate protocol);
  same-state consistency / near-collision / probe / full-delay error via experiments.fdr.diag.analyse;
  restricted-input patterns (artificial complete views and natural-mask view removal) as in Phase 3A."""
import glob, os, numpy as np, pandas as pd, torch
from experiments.fdr.diag import analyse
from experiments.fdr.impute import impute_fdr
from experiments.fdr.model import FDRModel
from src.delay_embedding import delay_embed
from src.evaluate import impute, metrics
from src.models import build_model
from src.missingness import prepare
torch.set_num_threads(8)
O = "results/extra/phase3b_fdr_pilot"; P, SEED = 0.7, 0
Xraw = np.load("results/extra/aim_autonomous_20261027/coarse_dt0.05/data/lorenz_dt0.05_20000.npz")["X"]
d = prepare(Xraw, P, SEED); a, b = d["split"]["test"]
runs = pd.read_csv(f"{O}/pilot_runs.csv"); rows, pats = [], []
for _, r in runs.sort_values("tag").iterrows():
    ck = torch.load(f"{O}/checkpoints/{r.exp_id}.pt", map_location="cpu", weights_only=False); c = ck["cfg"]
    mdl = FDRModel(3, c["m"], c["latent"], c["hidden"]) if r.decoder == "full" else build_model("C", 3, c["m"], c["latent"], c["hidden"], c["window"], c["direction"])
    mdl.load_state_dict(ck["state_dict"]); mdl.eval()
    X, M = torch.tensor(d["X"]), torch.tensor(d["M"])
    past = impute(mdl, X, M, c["m"], c["tau"]).numpy()
    res = dict(tag=r.tag, exp_id=r.exp_id, n_params=r.n_params, epochs_run=r.epochs_run, recorded_test_rmse=r.test_rmse,
               test_rmse_past=metrics(past, d, "test")["rmse"], val_rmse_past=metrics(past, d, "val")["rmse"])
    if r.decoder == "full":
        bi = impute_fdr(mdl, X, M, c["m"], c["tau"], "bidir").numpy()
        res["test_rmse_bidir"] = metrics(bi, d, "test")["rmse"]
        assert np.allclose(impute_fdr(mdl, X, M, c["m"], c["tau"], "past").numpy(), past, atol=1e-6)
    res.update(analyse(mdl, d, P, c["m"], c["tau"], label=r.tag)); rows.append(res)
    # restricted-input patterns on the same test windows
    Xt = torch.tensor(d["X"][a:b]); S = Xt[c["m"] - 1:]; g = torch.Generator().manual_seed(4242)
    Mn = (torch.rand(Xt.shape, generator=g) < 1 - P).float()
    for nm, views in {"x": [0], "y": [1], "z": [2], "xy": [0, 1], "xz": [0, 2], "xyz": [0, 1, 2]}.items():
        for fam in ["ART", "NATV"]:
            Mv = torch.zeros_like(Xt); Mv[:, views] = 1 if fam == "ART" else Mn[:, views]
            V, MV = delay_embed((Xt * Mv)[None], Mv[None], c["m"], c["tau"])
            with torch.no_grad(): out = mdl(V, MV)[0]
            miss = (1 - Mv[c["m"] - 1:]).bool()
            pats.append(dict(tag=r.tag, pattern=f"{fam}:{nm}", state_err_median=float((out - S).norm(dim=1).median()),
                             rmse_unobserved_lag0=float(((out - S) ** 2)[miss].mean().sqrt()) if miss.any() else np.nan))
    print(r.tag, "done", flush=True)
df = pd.DataFrame(rows); df.to_csv(f"{O}/pilot_eval.csv", index=False); pf = pd.DataFrame(pats); pf.to_csv(f"{O}/pilot_patterns.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
print(df.set_index("tag").T.to_string())
print(pf.pivot(index="pattern", columns="tag", values=["state_err_median", "rmse_unobserved_lag0"]).round(3).to_string())
