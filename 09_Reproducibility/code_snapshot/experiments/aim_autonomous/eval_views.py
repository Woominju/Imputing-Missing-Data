"""CPU evaluation of Model C (original checkpoints) and Model E (view-dropout) under identical inputs.

Conditions (targets are always test-split hidden entries, eval_mask=1; hidden values never enter any input):
  multi          : original MCAR masks M (condition B, the AIM setting)
  from_<i>_full  : ONLY view i given, completely observed (M_orig); all other variables absent.
                   Same prediction-time information as FPRM with reference i.
  condA_ref<i>   : condition A input (M with column i fully observed) -> comparable to FPRM / C-retrainA.
For from_*/condA_* the target set excludes the last 6 test points (FPRM E=3,tau=3 tail) to match FPRM rows.
Latent consistency (the fused code z that the decoder reads):
  cons_<i> = median_t ||z_i(t) - z_all(t)|| / median_{t,t'} ||z_all(t) - z_all(t')||
  with z_all = code from all three complete views, z_i = code from complete view i alone (0 = identical code).
Symmetry probe for the z view: corr(xhat, x) and mean(xhat)/std(x) for x, y targets from the z view alone.
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

from src.delay_embedding import delay_embed, pad_front
from src.evaluate import impute
from src.missingness import prepare
from src.models import build_model

VARS = "xyz"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TAIL = 6


@torch.no_grad()
def latent(model, X, M, m, tau, chunk=2048):
    Xp, Mp = pad_front(X * M, M, m, tau)
    span = (m - 1) * tau
    zs = []
    for s in range(0, len(X), chunk):
        e = min(len(X), s + chunk)
        V, MV = delay_embed(Xp[None, s:e + span], Mp[None, s:e + span], m, tau)
        zs.append(model.encoder(V, MV)[0])
    return torch.cat(zs, 0).numpy()


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    c = ck["cfg"]
    if c.get("model") == "F":
        from experiments.aim_autonomous.model_f import ModelF
        mdl = ModelF(3, c["m"], c["latent"], c["hidden"])
    else:
        mdl = build_model("C", 3, c["m"], c["latent"], c["hidden"], c["window"], c["direction"])
    mdl.load_state_dict(ck["state_dict"])
    mdl.eval()
    return mdl, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ps", default="0.1,0.3,0.5,0.7")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--data", default="results/data/lorenz_20000.npz")
    ap.add_argument("--c_ckpt", default="results/checkpoints/lorenz_C_p{p}_s{seed}.pt")
    ap.add_argument("--e_ckpt", default="checkpoints/lorenz_E_viewdrop_p{p}_s{seed}.pt", help="relative to --e_dir")
    ap.add_argument("--arch_csv", default="results/lorenz_results.csv")
    ap.add_argument("--arch_id", default="lorenz_C_p{p}_s{seed}")
    ap.add_argument("--e_label", default="E")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    Xraw = np.load(os.path.join(ROOT, a.data))["X"]
    arch = pd.read_csv(os.path.join(ROOT, a.arch_csv)).drop_duplicates("exp_id", keep="last").set_index("exp_id")
    rows, lat = [], []
    rng = np.random.default_rng(0)
    for p in map(float, a.ps.split(",")):
        for seed in map(int, a.seeds.split(",")):
            d = prepare(Xraw, p, seed)
            X, M, Mo, EV = d["X"], d["M"], d["M_orig"], d["eval_mask"] > 0
            T_ = len(X)
            ta, tb = d["split"]["test"]
            models = {"C": os.path.join(ROOT, a.c_ckpt.format(p=p, seed=seed)),
                      a.e_label: os.path.join(a.e_dir, a.e_ckpt.format(p=p, seed=seed))}
            for name, path in models.items():
                if not os.path.exists(path):
                    continue
                mdl, c = load(path)
                Xt = torch.tensor(X)
                inputs = {"multi": M}
                for i in range(3):
                    Mi = np.zeros_like(M)
                    Mi[:, i] = Mo[:, i]
                    inputs[f"from_{VARS[i]}_full"] = Mi
                    MA = M.copy()
                    MA[:, i] = Mo[:, i]
                    inputs[f"condA_ref{VARS[i]}"] = MA
                for cond, Min in inputs.items():
                    xh = impute(mdl, Xt, torch.tensor(Min), c["m"], c["tau"]).numpy()
                    tg = range(3) if cond == "multi" else [j for j in range(3) if VARS[j] != cond[-1] and
                                                            not cond.startswith(f"from_{VARS[j]}")]
                    if cond.startswith("from_"):
                        tg = [j for j in range(3) if VARS[j] != cond[5]]
                    for j in tg:
                        ev = np.zeros(T_, bool)
                        ev[ta:tb] = EV[ta:tb, j]
                        if cond != "multi":
                            ev[T_ - TAIL:] = False
                        assert not (Min[ev, j] > 0).any()        # targets are never inputs
                        e = xh[ev, j] - X[ev, j]
                        r = {"model": name, "p": p, "seed": seed, "cond": cond, "target": VARS[j], "n": int(ev.sum()),
                             "sse": float((e ** 2).sum()), "sae": float(np.abs(e).sum()),
                             "rmse": float(np.sqrt(np.mean(e ** 2))), "finite": bool(np.isfinite(xh[ev, j]).all())}
                        if cond == "from_z_full":
                            r["corr"] = float(np.corrcoef(xh[ev, j], X[ev, j])[0, 1])
                            r["mean_pred_over_std"] = float(xh[ev, j].mean() / X[ev, j].std())
                        rows.append(r)
                    if cond == "multi" and name == "C":                 # sanity vs archived result
                        ee = (xh[ta:tb] - X[ta:tb]) * EV[ta:tb]
                        rm = np.sqrt((ee ** 2).sum() / EV[ta:tb].sum())
                        assert abs(rm - arch.loc[a.arch_id.format(p=p, seed=seed), "test_rmse"]) < 1e-4
                # latent consistency on test times
                z_all = latent(mdl, Xt, torch.tensor(Mo), c["m"], c["tau"])[ta:tb]
                scale = np.median(np.linalg.norm(z_all[rng.integers(0, len(z_all), 4000)] -
                                                 z_all[rng.integers(0, len(z_all), 4000)], axis=1))
                lr = {"model": name, "p": p, "seed": seed, "scale_between_times": float(scale)}
                for i in range(3):
                    Mi = np.zeros_like(M)
                    Mi[:, i] = Mo[:, i]
                    zi = latent(mdl, Xt, torch.tensor(Mi), c["m"], c["tau"])[ta:tb]
                    lr[f"cons_{VARS[i]}"] = float(np.median(np.linalg.norm(zi - z_all, axis=1)) / scale)
                zn = latent(mdl, Xt, torch.tensor(M), c["m"], c["tau"])[ta:tb]
                lr["cons_natural_mask"] = float(np.median(np.linalg.norm(zn - z_all, axis=1)) / scale)
                lat.append(lr)
            print(f"done p={p} seed={seed}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "view_eval_runs.csv"), index=False)
    pd.DataFrame(lat).to_csv(os.path.join(a.out, "latent_consistency.csv"), index=False)


if __name__ == "__main__":
    main()
