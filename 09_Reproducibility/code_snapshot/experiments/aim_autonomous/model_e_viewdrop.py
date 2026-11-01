"""Model E = Model C architecture + loss, trained with VIEW-SUBSET DROPOUT in the training masks (experimental).

Motivation (results/notes/aim_autonomous_20261027.md, section C):
  * C fuses per-view codes by an observation-weighted mean z = sum_i w_i z_i / sum_i w_i, which is only meaningful
    if the view codes z_i live in one coordinate system. Existing diagnostics (aim_latent v1): identity R^2 < 0,
    affine R^2 0.07-0.85 between view codes -> not directly averageable.
  * C's single-complete-view cross reconstruction is poor (x-view -> y RMSE 0.10-0.39, aim_crossview v1) although
    the x delay coordinates do determine y (FPRM, the GP on the x delay vector, reaches far lower error), while
    single-view inputs essentially never occur in C's training masks (P(one non-empty view) <= 6e-6 for p<=0.3).
Change: ONLY the training-time hiding H. With probability Q per training segment, a random non-empty proper subset
of views is kept (1 view w.p. 1/2, 2 views w.p. 1/2) and every observed entry of the dropped variables in that
segment is added to H. The loss is unchanged (src.train.masked_loss on H), so dropped variables become
reconstruction targets from the kept views.
Why no latent alignment loss: minimising squared error given a view subset S estimates E[x(t) | v_S(t)]. When v_S
identifies the state (generic Takens setting), all such estimates coincide with the true state value; when it
does not (Lorenz z-view: Gamma(x,y,z)=(-x,-y,z) gives identical z-delay vectors), forcing z_z == z_x would
be impossible to satisfy and would distort the code, whereas the conditional mean is the correct target.
Q = 0.5 fixed a priori (not tuned). Architecture, optimiser, all other hyper-parameters: copied from the original C
checkpoint for the same (p, seed). Model selection: val RMSE on the standard eval mask, as for C.
Original src/ files are not modified: src.train.forward_batch is replaced in this process only.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch

import src.train as T
from src.delay_embedding import delay_embed
from src.missingness import prepare

Q = 0.5
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def forward_batch_viewdrop(model, cfg, Xs, Ms, gen):
    assert cfg["model"] == "C"
    Bsz, W, n = Ms.shape
    H = Ms * (torch.rand(Ms.shape, generator=gen, device=Ms.device) < cfg["hide_rate"]).float()
    apply = torch.rand(Bsz, generator=gen, device=Ms.device) < Q
    k_keep = 1 + (torch.rand(Bsz, generator=gen, device=Ms.device) < 0.5).long()          # 1 or 2 views kept
    order = torch.argsort(torch.rand(Bsz, n, generator=gen, device=Ms.device), dim=1)
    rank = torch.argsort(order, dim=1)                                                       # random rank per view
    drop = (rank >= k_keep[:, None]) & apply[:, None]                                        # (B, n)
    H = torch.where(drop[:, None, :], Ms, H)
    B = Ms - H
    span = (cfg["m"] - 1) * cfg["tau"]
    V, MV = delay_embed(Xs * B, B, cfg["m"], cfg["tau"])
    xhat = model(V, MV)
    return T.masked_loss(xhat, Xs[:, span:], H[:, span:]), H[:, span:].sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ps", default="0.1,0.3,0.5,0.7")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--max_epochs", type=int, default=0, help="smoke only")
    ap.add_argument("--q", type=float, default=Q, help="view-dropout probability (default 0.5 = main run)")
    a = ap.parse_args()
    globals()["Q"] = a.q                                          # read by forward_batch_viewdrop
    out = os.path.abspath(a.out)
    assert os.environ.get("IMD_RESULTS_DIR", "").rstrip("/") == out.rstrip("/")
    assert torch.cuda.is_available()
    T.forward_batch = forward_batch_viewdrop                       # in-process replacement only
    os.makedirs(os.path.join(out, "preds"), exist_ok=True)
    csv = os.path.join(out, "train_runs.csv")
    done = set(pd.read_csv(csv).exp_id) if os.path.exists(csv) else set()
    Xraw = np.load(os.path.join(ROOT, "results", "data", "lorenz_20000.npz"))["X"]
    for p in map(float, a.ps.split(",")):
        for seed in map(int, a.seeds.split(",")):
            exp_id = f"lorenz_E_viewdrop_p{p}_s{seed}"
            if exp_id in done:
                continue
            ck = torch.load(os.path.join(ROOT, "results", "checkpoints", f"lorenz_C_p{p}_s{seed}.pt"),
                            map_location="cpu", weights_only=False)
            cfg = dict(ck["cfg"])
            cfg["device"] = "cuda"
            if a.max_epochs:
                cfg["epochs"] = a.max_epochs
            d = prepare(Xraw, p, seed)                              # condition B: original masks
            t0 = time.time()
            row, xhat, _ = T.run(cfg, Xraw, d, exp_id, save_ckpt=True, verbose=False)
            np.save(os.path.join(out, "preds", exp_id + ".npy"), xhat.astype(np.float32))
            row.update({"variant": f"viewdrop_Q{a.q}", "wall_s": round(time.time() - t0, 1),
                        "finite": bool(np.isfinite(xhat).all()), "epochs_cfg": cfg["epochs"]})
            pd.DataFrame([row]).to_csv(csv, mode="a", header=not os.path.exists(csv), index=False)
            print(f"{exp_id} test_rmse={row['test_rmse']:.4f} epochs={row['epochs_run']} {row['wall_s']}s", flush=True)


if __name__ == "__main__":
    main()
