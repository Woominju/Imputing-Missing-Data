"""Phase 3B section 3: continuous consistency distributions and true-state near-collision analysis on EXISTING
Model C / E (Q=0.5) checkpoints, Lorenz dt=0.01, p in {0.3, 0.7}, seeds 0-2. CPU, no training."""
import numpy as np, pandas as pd, torch
from experiments.fdr.diag import analyse
from src.models import build_model
from src.missingness import prepare
torch.set_num_threads(8)
Xraw = np.load("results/data/lorenz_20000.npz")["X"]; rows = []
for kind, tmpl in [("C", "results/checkpoints/lorenz_C_p{p}_s{s}.pt"), ("E", "results/extra/aim_autonomous_20261027/model_E_viewdrop/checkpoints/lorenz_E_viewdrop_p{p}_s{s}.pt")]:
    for p in [0.3, 0.7]:
        for s in [0, 1, 2]:
            ck = torch.load(tmpl.format(p=p, s=s), map_location="cpu", weights_only=False); c = ck["cfg"]
            mdl = build_model("C", 3, c["m"], c["latent"], c["hidden"], c["window"], c["direction"]); mdl.load_state_dict(ck["state_dict"]); mdl.eval()
            r = analyse(mdl, prepare(Xraw, p, s), p, c["m"], c["tau"], label=f"{kind}_p{p}_s{s}"); r.update(model=kind, p=p, seed=s); rows.append(r)
            print(r["label"], flush=True)
df = pd.DataFrame(rows); df.to_csv("results/extra/phase3b_fdr_pilot/existing_ckpt_diag.csv", index=False)
cols = ["dz_q50", "dz_q90", "dR_q50", "dR_q90", "e1_q50", "ratio_dz_over_max_dR_delta_q50", "frac_accurate_strict", "A_strict", "A_loose",
        "C_strict", "C_loose", "dz_median_given_accurate_strict", "same_state_dz_median_for_output_close", "nn_frac_output_close",
        "nn_close_state_0-0.05_frac", "nn_close_state_0-0.05_dz_median", "nn_close_state_0.05-0.1_dz_median", "nn_close_state_0.2-inf_frac",
        "probe_knn_state_rmse", "decoder_state_rmse", "knn10_overlap_latent_vs_state"]
pd.set_option("display.width", 250); print(df.groupby(["model", "p"])[cols].agg(["mean", "min", "max"]).round(3).T.to_string())
