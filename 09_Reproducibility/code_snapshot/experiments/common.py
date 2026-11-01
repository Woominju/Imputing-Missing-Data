"""Shared grid runner: same split and masks for every model at a given (p, seed)."""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.missingness import prepare  # noqa: E402
from src.train import run, run_baseline, RES  # noqa: E402


BASE_COLS = ["exp_id", "dataset", "model", "missing_rate", "seed", "m", "tau", "latent", "hidden", "batch", "lr",
             "hide_rate", "window", "direction", "n_vars", "test_n_eval", "test_rmse", "test_mae", "test_rmse_orig",
             "test_mae_orig", "test_rmse_per_var"]
BASELINE_TAIL = ["best_val_rmse", "train_time_s", "device"]
MODEL_TAIL = ["best_val_rmse", "final_train_loss", "epochs_run", "train_time_s", "n_params", "nonfinite_steps",
              "device"]
EXTRA_COLS = ["kappa", "sync_rms_diff", "dropped_dup_cols", "error"]
ALL_COLS = list(dict.fromkeys(BASE_COLS + MODEL_TAIL + EXTRA_COLS))


def append_row(path, row):
    """Append with a fixed column set so baseline and model rows stay aligned."""
    df = pd.DataFrame([{c: row.get(c, "") for c in ALL_COLS}])
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def read_part(path):
    """Reads a result part. Also repairs files written by the earlier append_row, whose
    header came from the first (baseline) row while model rows carried extra columns."""
    import csv
    rows = list(csv.reader(open(path)))
    header, body = rows[0], rows[1:]
    if header == ALL_COLS:
        return pd.read_csv(path)
    extras = [c for c in header if c in EXTRA_COLS]
    recs = []
    for r in body:
        tail = BASELINE_TAIL if r[2] in ("mean", "locf", "linear") else MODEL_TAIL
        cols = BASE_COLS + tail + extras
        if len(r) != len(cols):  # error rows: exp_id, dataset, model, missing_rate, seed, error (+extras)
            cols = ["exp_id", "dataset", "model", "missing_rate", "seed", "error"] + extras
        assert len(r) == len(cols), (path, len(r), r[:3])
        recs.append(dict(zip(cols, r)))
    df = pd.DataFrame(recs).replace("", np.nan)
    return df.apply(lambda c: pd.to_numeric(c, errors="ignore") if c.name not in ("device", "error") else c)


def done_ids(path):
    if not os.path.exists(path):
        return set()
    return set(read_part(path)["exp_id"])


def run_grid(dataset, X_raw, args, out_csv, extra=None, keep_xhat=None):
    """Runs baselines + models over rates x seeds. Skips exp_ids already in out_csv.
    keep_xhat: (p, seed) for which imputations are saved for plotting."""
    cfg = {k: getattr(args, k) for k in ["m", "tau", "latent", "hidden", "batch", "targets", "lr", "epochs",
                                          "steps_per_epoch", "patience", "hide_rate", "clip", "window",
                                          "direction", "device"]}
    cfg["dataset"] = dataset
    rates = [float(r) for r in args.rates.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    models = [m for m in args.models.split(",") if m]
    baselines = [b for b in args.baselines.split(",") if b]
    finished = done_ids(out_csv)
    # resume: also skip exp_ids already recorded in other CSVs (e.g. a part written by an
    # interrupted run), so the resumed run can write to a fresh file
    for f in [f for f in getattr(args, "skip_from", "").split(",") if f]:
        finished |= done_ids(os.path.join(RES, f))
    os.makedirs(os.path.join(RES, "imputations"), exist_ok=True)
    for p in rates:
        for seed in seeds:
            data = prepare(X_raw, p, seed)
            mask_path = os.path.join(RES, "imputations", f"{dataset}_p{p}_s{seed}_masks.npz")
            if not os.path.exists(mask_path):
                np.savez_compressed(mask_path, M=data["M"], M_orig=data["M_orig"], eval_mask=data["eval_mask"],
                                    mu=data["mu"], sd=data["sd"], split=json.dumps(data["split"]))
            save = keep_xhat is not None and (p, seed) == keep_xhat
            for kind in baselines:
                eid = f"{dataset}_{kind}_p{p}_s{seed}{args.tag}"
                if eid in finished:
                    continue
                row, xhat = run_baseline(kind, {**cfg, "seed": seed}, data, eid)
                row.update(extra or {})
                append_row(out_csv, row)
                if save:
                    np.save(os.path.join(RES, "imputations", f"{eid}.npy"), xhat)
            for mdl in models:
                eid = f"{dataset}_{mdl}_p{p}_s{seed}{args.tag}"
                if eid in finished:
                    continue
                # IMD_GPU_DEADLINE (unix time): do not start a training run that could pass it
                dl = float(os.environ.get("IMD_GPU_DEADLINE", "inf"))
                if time.time() + float(os.environ.get("IMD_RUN_BUDGET_S", "0")) >= dl:
                    print(f"SKIP {eid}: GPU deadline", flush=True)
                    continue
                c = {**cfg, "model": mdl, "seed": seed}
                try:
                    row, xhat, _ = run(c, X_raw, data, eid)
                except RuntimeError as e:  # e.g. OOM: record and continue
                    row, xhat = {"exp_id": eid, "dataset": dataset, "model": mdl, "missing_rate": p,
                                 "seed": seed, "error": str(e)[:200]}, None
                    print("ERROR", eid, e, flush=True)
                row.update(extra or {})
                append_row(out_csv, row)
                print(f"done {eid} test_rmse={row.get('test_rmse')}", flush=True)
                if save and xhat is not None:
                    np.save(os.path.join(RES, "imputations", f"{eid}.npy"), xhat)
