"""Phase 4 post-pilot CPU check: is the own-subsystem kNN gain from removing the other subsystem's views an information effect or
a kNN-geometry effect? MLP probe (probe-init seeds 0, 1; fit train / select val / predict test) on latents computed with
(a) all four views and (b) own-subsystem views only, for y1 (own = x1, z1) and y2 (own = x2, z2). No training of autoencoders."""
import glob, os, sys
import numpy as np, pandas as pd, torch
from multiprocessing import Pool
sys.path.insert(0, "results/extra/phase4_coupled_pilot")
from evaluate import CTOR, outputs, rows_of, O
from experiments.coupled import data as CD
from experiments.fdr import probes


def run(path):
    torch.set_num_threads(3); eid = os.path.basename(path)[:-3]; r = pd.read_csv(f"{O}/rows/{eid}.csv").iloc[0]
    mdl = CTOR[r.model_name](); mdl.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["state_dict"]); mdl.eval()
    d, hid = CD.build(CD.load_raw(r.kappa, int(r.traj)), r.p, int(r.seed)); Sn = hid["state_norm"]; X = torch.tensor(d["X"])
    tr, va, te = rows_of(d, "train"), rows_of(d, "val"), rows_of(d, "test"); out = dict(exp_id=eid, kappa=r.kappa, p=r.p, model=r.model_name, seed=int(r.seed))
    for tn, col, keep in (("y1", 1, [0, 1]), ("y2", 4, [2, 3])):
        for vn, M in (("all4", d["M"]), ("own_only", np.where(np.isin(np.arange(4), keep)[None], d["M"], 0).astype(np.float32))):
            Z = outputs(mdl, X, torch.tensor(M))[1]; y = Sn[:, col]
            out[f"mlp_{tn}_{vn}"] = float(np.mean([np.sqrt(((probes.mlp(Z[tr], y[tr], Z[va], y[va], Z[te], seed=ps)[0] - y[te]) ** 2).mean()) for ps in (0, 1)]))
    return out


if __name__ == "__main__":
    with Pool(16) as pool:
        R = pd.DataFrame(pool.map(run, sorted(glob.glob(f"{O}/checkpoints/p4_*_t0_*.pt"))))
    R["y1: all4 - own"] = R.mlp_y1_all4 - R.mlp_y1_own_only; R["y2: all4 - own"] = R.mlp_y2_all4 - R.mlp_y2_own_only
    R.to_csv(f"{O}/view_removal_mlp.csv", index=False)
    pd.set_option("display.width", 250)
    print(R.groupby(["kappa", "p", "model"])[["mlp_y1_all4", "mlp_y1_own_only", "y1: all4 - own", "mlp_y2_all4", "mlp_y2_own_only", "y2: all4 - own"]].mean().round(4).to_string())
    print((R[["y1: all4 - own", "y2: all4 - own"]] < 0).groupby([R.kappa, R.p]).sum())
