"""Model F (experimental) = Model C with the observation-weighted MEAN fusion replaced by a learned, mask-aware fusion.

Why (results/notes/aim_autonomous_20261027.md, C): C fuses per-view codes by z = sum_i w_i z_i / sum_i w_i, which is a
consistent state representation only if every view code lives in the same coordinates (Proposition 3). Measured:
view codes are not directly averageable (aim_latent v1), and training C with view dropout (Model E) trades
multi-view accuracy for single-view accuracy at every dropout rate tried (Q = 0.15, 0.3, 0.5) - consistent with the
averaging step forcing one decoder to read both averaged and single-view codes.
Change: fusion z = G([a_1 z_1, ..., a_n z_n, w_1, ..., w_n]) with a_i = 1{view i has an observation}, w_i = fraction
of observed lags; G is an MLP (same width as the encoders). Per-view encoders E_i, the shared decoder, the loss
(src.train L_mask), optimiser and all hyper-parameters are those of C. With --q > 0 the view-dropout training masks
of Model E are used as well, so fusion and view-dropout effects can be separated.
What this does NOT give: no guarantee that z(t) is the same for different view subsets (no consistency constraint),
no diffeomorphism. It only tests whether the multi-view vs single-view trade-off comes from mean fusion.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import experiments.aim_autonomous.model_e_viewdrop as EV
import src.train as T
from src.missingness import prepare
from src.models import ModelC, build_model, mlp

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class LearnedFusionEncoder(nn.Module):
    def __init__(self, base_enc, n, m, latent, hidden):
        super().__init__()
        self.enc = base_enc.enc                       # identical per-view encoder architecture as C
        self.m = m
        self.fuse = mlp(n * latent + n, hidden, latent)

    def forward(self, V, MV):
        zi = self.enc(torch.cat([V * MV, MV], -1))    # (B, K, n, d)
        w = MV.sum(-1) / self.m                        # (B, K, n) fraction of observed lags
        a = (w > 0).float()
        return self.fuse(torch.cat([(zi * a[..., None]).flatten(-2), w], -1))


class ModelF(ModelC):
    def __init__(self, n, m, latent=16, hidden=64, **_):
        super().__init__(n, m, latent, hidden)
        self.encoder = LearnedFusionEncoder(self.encoder, n, m, latent, hidden)


def build(name, n, m, latent, hidden, window=8, direction=-1):
    return ModelF(n, m, latent, hidden) if name == "F" else build_model(name, n, m, latent, hidden, window, direction)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ps", default="0.1,0.3,0.5,0.7")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--q", type=float, default=0.0)
    ap.add_argument("--max_epochs", type=int, default=0, help="smoke only")
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    assert os.environ.get("IMD_RESULTS_DIR", "").rstrip("/") == out.rstrip("/")
    assert torch.cuda.is_available()
    T.build_model = build                                         # in-process replacement only
    if a.q > 0:
        EV.Q = a.q
        orig = EV.forward_batch_viewdrop

        def fb(model, cfg, Xs, Ms, gen):
            return orig(model, {**cfg, "model": "C"}, Xs, Ms, gen)   # same masking/loss path as C/E
        T.forward_batch = fb
    os.makedirs(os.path.join(out, "preds"), exist_ok=True)
    csv = os.path.join(out, "train_runs.csv")
    done = set(pd.read_csv(csv).exp_id) if os.path.exists(csv) else set()
    Xraw = np.load(os.path.join(ROOT, "results/data/lorenz_20000.npz"))["X"]
    for p in map(float, a.ps.split(",")):
        for seed in map(int, a.seeds.split(",")):
            exp_id = f"lorenz_F_q{a.q}_p{p}_s{seed}"
            if exp_id in done:
                continue
            ck = torch.load(os.path.join(ROOT, "results/checkpoints", f"lorenz_C_p{p}_s{seed}.pt"),
                            map_location="cpu", weights_only=False)
            cfg = dict(ck["cfg"])
            cfg.update({"device": "cuda", "model": "F", "fusion": "learned_mlp", "viewdrop_q": a.q})
            if a.max_epochs:
                cfg["epochs"] = a.max_epochs
            d = prepare(Xraw, p, seed)
            t0 = time.time()
            row, xhat, _ = T.run(cfg, Xraw, d, exp_id, save_ckpt=True, verbose=False)
            np.save(os.path.join(out, "preds", exp_id + ".npy"), xhat.astype(np.float32))
            row.update({"variant": f"F_q{a.q}", "wall_s": round(time.time() - t0, 1),
                        "finite": bool(np.isfinite(xhat).all())})
            pd.DataFrame([row]).to_csv(csv, mode="a", header=not os.path.exists(csv), index=False)
            print(f"{exp_id} test_rmse={row['test_rmse']:.4f} epochs={row['epochs_run']} {row['wall_s']}s", flush=True)


if __name__ == "__main__":
    main()
