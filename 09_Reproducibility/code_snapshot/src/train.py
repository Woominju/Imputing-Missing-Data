"""Training with the single masked-reconstruction objective L_mask."""
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from .delay_embedding import delay_embed
from .evaluate import impute, metrics, baseline_impute
from .models import build_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# IMD_RESULTS_DIR lets additional experiments write to a separate results folder
RES = os.environ.get("IMD_RESULTS_DIR", os.path.join(ROOT, "results"))


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def segment_len(cfg):
    if cfg["model"] == "A":
        return cfg["targets"]
    K = cfg["window"] if cfg["model"] == "D" else cfg["targets"]
    return (cfg["m"] - 1) * cfg["tau"] + K


def masked_loss(xhat, x, H):
    """L_mask = mean over samples with sum(H)>0 of sum H (x - xhat)^2 / sum H."""
    num = (H * (x - xhat) ** 2).flatten(1).sum(1)
    den = H.flatten(1).sum(1)
    keep = den > 0
    if keep.sum() == 0:
        return None
    return (num[keep] / den[keep]).mean()


def forward_batch(model, cfg, Xs, Ms, gen):
    """Xs, Ms: (B, W, n) segments from the train split. Samples H <= M, feeds X*B, B
    (B = M - H) and returns (loss, n_hidden). Delay vectors are built after masking."""
    H = Ms * (torch.rand(Ms.shape, generator=gen, device=Ms.device) < cfg["hide_rate"]).float()
    B = Ms - H
    Xin = Xs * B
    if cfg["model"] == "A":
        xhat = model(Xin, B)
        tgt, Ht = Xs, H
    else:
        span = (cfg["m"] - 1) * cfg["tau"]
        V, MV = delay_embed(Xin, B, cfg["m"], cfg["tau"])
        xhat = model.forward_window(V, MV) if cfg["model"] == "D" else model(V, MV)
        tgt, Ht = Xs[:, span:], H[:, span:]
    return masked_loss(xhat, tgt, Ht), Ht.sum()


def run(cfg, X_raw, data, exp_id, save_ckpt=True, verbose=True):
    """Train one model on prepared data; returns a result row (dict)."""
    dev = torch.device(cfg["device"])
    set_seed(cfg["seed"])
    gen = torch.Generator(device=dev).manual_seed(cfg["seed"] + 1000)
    X = torch.tensor(data["X"], device=dev)
    M = torch.tensor(data["M"], device=dev)
    T, n = X.shape
    model = build_model(cfg["model"], n, cfg["m"], cfg["latent"], cfg["hidden"],
                        cfg["window"], cfg["direction"]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    W = segment_len(cfg)
    a, b = data["split"]["train"]
    max_start = b - W
    assert max_start > a, "train split shorter than one segment"
    log, best, best_state, bad = [], np.inf, None, 0
    t0 = time.time()
    nonfinite = 0
    for ep in range(cfg["epochs"]):
        model.train()
        losses = []
        for _ in range(cfg["steps_per_epoch"]):
            s = torch.randint(a, max_start + 1, (cfg["batch"],), generator=gen, device=dev)
            idx = s[:, None] + torch.arange(W, device=dev)[None]
            loss, nh = forward_batch(model, cfg, X[idx], M[idx], gen)
            if loss is None:
                continue
            opt.zero_grad()
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
            if not (torch.isfinite(loss) and torch.isfinite(gnorm)):
                nonfinite += 1
                continue
            opt.step()
            losses.append(loss.item())
        xhat = impute(model, X, M, cfg["m"], cfg["tau"]).cpu().numpy()
        val = metrics(xhat, data, "val")
        vr = val.get("rmse", np.nan)
        log.append({"epoch": ep, "train_loss": float(np.mean(losses)) if losses else np.nan,
                    "val_rmse": vr, "val_mse": vr ** 2, "time_s": time.time() - t0})
        if vr < best - 1e-6:
            best, bad = vr, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if verbose and (ep % 10 == 0 or ep == cfg["epochs"] - 1):
            print(f"[{exp_id}] ep {ep} train {log[-1]['train_loss']:.4f} val_rmse {vr:.4f}", flush=True)
        if bad >= cfg["patience"]:
            break
    train_time = time.time() - t0
    model.load_state_dict(best_state)
    xhat = impute(model, X, M, cfg["m"], cfg["tau"]).cpu().numpy()
    test = metrics(xhat, data, "test")
    os.makedirs(os.path.join(RES, "logs"), exist_ok=True)
    pd.DataFrame(log).to_csv(os.path.join(RES, "logs", f"{exp_id}.csv"), index=False)
    if save_ckpt:
        os.makedirs(os.path.join(RES, "checkpoints"), exist_ok=True)
        torch.save({"state_dict": best_state, "cfg": cfg}, os.path.join(RES, "checkpoints", f"{exp_id}.pt"))
    row = result_row(cfg, exp_id, test, data, n)
    row.update({"best_val_rmse": best, "final_train_loss": log[-1]["train_loss"], "epochs_run": len(log),
                "train_time_s": train_time, "n_params": sum(p.numel() for p in model.parameters()),
                "nonfinite_steps": nonfinite,
                "device": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu"})
    return row, xhat, log


def run_baseline(kind, cfg, data, exp_id):
    t0 = time.time()
    xhat = baseline_impute(kind, data["X"] * data["M"], data["M"])
    row = result_row({**cfg, "model": kind}, exp_id, metrics(xhat, data, "test"), data, data["X"].shape[1])
    row.update({"best_val_rmse": metrics(xhat, data, "val").get("rmse"), "train_time_s": time.time() - t0,
                "device": "cpu"})
    return row, xhat


def result_row(cfg, exp_id, test, data, n):
    row = {"exp_id": exp_id, "dataset": cfg["dataset"], "model": cfg["model"], "missing_rate": data["p"],
           "seed": data["seed"], "m": cfg["m"], "tau": cfg["tau"], "latent": cfg["latent"],
           "hidden": cfg["hidden"], "batch": cfg["batch"], "lr": cfg["lr"], "hide_rate": cfg["hide_rate"],
           "window": cfg["window"] if cfg["model"] == "D" else "", "direction": cfg["direction"] if cfg["model"] == "D" else "",
           "n_vars": n, "test_n_eval": test.get("n_eval"), "test_rmse": test.get("rmse"),
           "test_mae": test.get("mae"), "test_rmse_orig": test.get("rmse_orig"),
           "test_mae_orig": test.get("mae_orig")}
    pv = test.get("rmse_per_var")
    row["test_rmse_per_var"] = json.dumps([round(v, 5) for v in pv]) if pv is not None and n <= 12 else ""
    return row


DEFAULTS = dict(m=8, tau=5, latent=16, hidden=64, batch=64, targets=64, lr=1e-3, epochs=60,
                steps_per_epoch=100, patience=10, hide_rate=0.2, clip=5.0, window=8, direction=-1,
                device="cuda" if torch.cuda.is_available() else "cpu")


def add_args(ap):
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k}", type=type(v), default=v)
    ap.add_argument("--models", default="A,B,C")
    ap.add_argument("--rates", default="0.1,0.3,0.5,0.7")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--baselines", default="mean,locf,linear")
    ap.add_argument("--tag", default="")
    ap.add_argument("--skip_from", default="", help="comma-separated CSVs under results/ whose exp_ids are skipped")
    return ap
