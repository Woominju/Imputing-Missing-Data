"""Phase 3C correction: the first evaluate.py built the "z removed" input with column 0 (x) zeroed, i.e. it duplicated
the x-removed stress test. This script recomputes ONLY the z-removed results correctly (column 1 = z zeroed, x mask
unchanged) for the xz models, with (a) probes refit on z-removed latents (train/val/test) and (b) the probe fitted on
natural latents applied unchanged to z-removed test latents (OOD transfer). Output: eval_probe_zremoved.csv."""
import numpy as np, pandas as pd, torch
from experiments.fdr import probes, xz_data
from experiments.fdr.features import latents
from experiments.fdr.model import FDRModel
from src.models import build_model
torch.set_num_threads(8); O = "results/extra/phase3c_hidden_state"; m, tau, span = 8, 1, 7
X3 = xz_data.load_raw(); runs = pd.read_csv(f"{O}/runs.csv"); rows = []
def rws(d, s): a, b = d["split"][s]; return np.arange(max(a, span), b)
for _, r in runs[runs.vars == "xz"].iterrows():
    seed = int(r.seed); d, yi = xz_data.build(X3, 0.3, seed, "xz"); yn = yi["y_norm"].astype(np.float64)
    ck = torch.load(f"{O}/checkpoints/{r.exp_id}.pt", map_location="cpu", weights_only=False); c = ck["cfg"]
    mdl = FDRModel(2, m, c["latent"], c["hidden"]) if r.decoder == "full" else build_model("C", 2, m, c["latent"], c["hidden"], c["window"], c["direction"])
    mdl.load_state_dict(ck["state_dict"]); mdl.eval()
    X_, M_ = torch.tensor(d["X"]), torch.tensor(d["M"]); Mzr = M_.clone(); Mzr[:, 1] = 0
    assert torch.equal(Mzr[:, 0], M_[:, 0]) and Mzr[:, 1].sum() == 0
    Zn = latents(mdl, X_, M_, m, tau).numpy(); Zz = latents(mdl, X_, Mzr, m, tau).numpy()
    tr, va, te = rws(d, "train"), rws(d, "val"), rws(d, "test")
    for name, f in probes.PROBES.items():
        pr, _ = f(Zz[tr], yn[tr], Zz[va], yn[va], Zz[te])
        rows.append(dict(seed=seed, model=r.tag, probe=name, mode="refit_on_z_removed", test_rmse=float(np.sqrt(((pr - yn[te]) ** 2).mean()))))
        pr, _ = f(Zn[tr], yn[tr], Zn[va], yn[va], Zz[te])
        rows.append(dict(seed=seed, model=r.tag, probe=name, mode="natural_probe_applied_to_z_removed", test_rmse=float(np.sqrt(((pr - yn[te]) ** 2).mean()))))
    print("done", r.tag, seed, flush=True)
df = pd.DataFrame(rows); df.to_csv(f"{O}/eval_probe_zremoved.csv", index=False)
pd.set_option("display.width", 250); print(df.pivot_table(index=["model", "mode"], columns=["probe", "seed"], values="test_rmse").round(4).to_string())
