"""Independent integrity check of every completed result row (CPU only).

For each row in results/parts/*.csv:
  - regenerate the data + masks with prepare() and compare with the saved masks npz
  - baselines: recompute imputation and metrics
  - models: load the checkpoint, check its cfg matches the row, re-impute on CPU and
    recompute test RMSE/MAE (normalised and original units) and validation RMSE
Also reports duplicate exp_ids and missing experiments.
Writes results/verification/recompute_check.csv.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from experiments.common import read_part  # noqa: E402
from experiments.run_lorenz import get_lorenz  # noqa: E402
from experiments.run_real_data import drop_duplicate_cols  # noqa: E402
from src.data_loader import load_real  # noqa: E402
from src.evaluate import impute, metrics, baseline_impute  # noqa: E402
from src.missingness import prepare  # noqa: E402
from src.models import build_model  # noqa: E402
from src.train import RES  # noqa: E402

BASELINES = ("mean", "locf", "linear")
PLAN = {"lorenz_p": ("lorenz", "ABCD"), "coupled_k": ("coupled", "CD"), "real_": ("real", "ABC")}
_raw = {}


def raw_data(ds):
    if ds not in _raw:
        if ds == "lorenz":
            _raw[ds] = get_lorenz()[0]
        elif ds.startswith("coupled_k"):
            _raw[ds] = np.load(os.path.join(RES, "data", f"coupled_lorenz_k{ds[9:]}_20000.npz"))["X"]
        else:
            X, names, _ = load_real(ds)
            _raw[ds] = drop_duplicate_cols(X, names)[0]
    return _raw[ds]


def expected_ids(part, d):
    ds = d.dataset.iloc[0]
    models = next(v[1] for k, v in PLAN.items() if os.path.basename(part).startswith(k))
    rates = [float(part.split("_p")[-1][:-4])] if "lorenz_p" in part else [0.1, 0.3, 0.5, 0.7]
    return {f"{ds}_{m}_p{p}_s{s}" for p in rates for s in (0, 1, 2) for m in list(BASELINES) + list(models)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="")
    args = ap.parse_args()
    torch.set_num_threads(8)
    files = sorted(f for f in os.listdir(os.path.join(RES, "parts")) if f.endswith(".csv"))
    if args.parts:
        files = [f for f in files if any(f.startswith(p) for p in args.parts.split(","))]
    out, summary = [], []
    all_ids = []
    for f in files:
        path = os.path.join(RES, "parts", f)
        d = read_part(path)
        all_ids += list(d.exp_id)
        exp = expected_ids(f, d)
        dup = d.exp_id[d.exp_id.duplicated()].tolist()
        summary.append({"part": f, "rows": len(d), "expected": len(exp), "missing": sorted(exp - set(d.exp_id)),
                        "unexpected": sorted(set(d.exp_id) - exp), "duplicates": dup})
        cache = {}
        for _, r in d.iterrows():
            ds, p, seed = r.dataset, float(r.missing_rate), int(r.seed)
            key = (ds, p, seed)
            if key not in cache:
                data = prepare(raw_data(ds), p, seed)
                mk = np.load(os.path.join(RES, "imputations", f"{ds}_p{p}_s{seed}_masks.npz"))
                mask_ok = (np.array_equal(mk["M"], data["M"]) and np.array_equal(mk["eval_mask"], data["eval_mask"])
                           and np.allclose(mk["mu"], data["mu"]) and np.allclose(mk["sd"], data["sd"])
                           and json.loads(str(mk["split"])) == {k: list(v) for k, v in data["split"].items()})
                cache = {key: (data, mask_ok)}
            data, mask_ok = cache[key]
            rec = {"part": f, "exp_id": r.exp_id, "mask_file_matches": mask_ok}
            if r.model in BASELINES:
                xhat = baseline_impute(r.model, data["X"] * data["M"], data["M"])
            else:
                ck = torch.load(os.path.join(RES, "checkpoints", f"{r.exp_id}.pt"), map_location="cpu",
                                weights_only=False)
                c = ck["cfg"]
                rec["ckpt_cfg_matches"] = all(
                    [c["model"] == r.model, c["seed"] == seed, c["m"] == r.m, c["tau"] == r.tau,
                     c["latent"] == r.latent, c["hidden"] == r.hidden, c["dataset"] == ds])
                n = data["X"].shape[1]
                mdl = build_model(c["model"], n, c["m"], c["latent"], c["hidden"], c["window"], c["direction"])
                mdl.load_state_dict(ck["state_dict"])
                xhat = impute(mdl, torch.tensor(data["X"]), torch.tensor(data["M"]), c["m"], c["tau"]).numpy()
                v = metrics(xhat, data, "val")
                rec["val_rmse_recomputed"] = v["rmse"]
                rec["val_rel_diff"] = abs(v["rmse"] - r.best_val_rmse) / r.best_val_rmse
            t = metrics(xhat, data, "test")
            for k in ["rmse", "mae", "rmse_orig", "mae_orig"]:
                rec[f"test_{k}_csv"] = r[f"test_{k}"]
                rec[f"test_{k}_recomputed"] = t[k]
                rec[f"{k}_rel_diff"] = abs(t[k] - r[f"test_{k}"]) / max(abs(r[f"test_{k}"]), 1e-12)
            rec["n_eval_matches"] = t["n_eval"] == r.test_n_eval
            out.append(rec)
        print(f"{f}: {len(d)} rows checked", flush=True)
    dup_all = pd.Series(all_ids)[pd.Series(all_ids).duplicated()].tolist()
    o = pd.DataFrame(out)
    os.makedirs(os.path.join(RES, "verification"), exist_ok=True)
    o.to_csv(os.path.join(RES, "verification", "recompute_check.csv"), index=False)
    pd.DataFrame(summary).to_csv(os.path.join(RES, "verification", "completeness.csv"), index=False)
    rel = o[[c for c in o.columns if c.endswith("rel_diff")]]
    print("rows:", len(o), "| duplicate exp_ids across parts:", dup_all)
    print("max rel diff per metric:\n", rel.max().to_string())
    print("mask mismatches:", int((~o.mask_file_matches).sum()),
          "| ckpt cfg mismatches:", int((o.get("ckpt_cfg_matches") == False).sum()),  # noqa: E712
          "| n_eval mismatches:", int((~o.n_eval_matches).sum()))
    bad = o[(rel > 1e-4).any(axis=1)]
    print("rows with any rel diff > 1e-4:", len(bad))
    if len(bad):
        print(bad[["exp_id"] + list(rel.columns)].to_string())
    for s in summary:
        print(s["part"], s["rows"], "/", s["expected"], "missing:", len(s["missing"]),
              "unexpected:", s["unexpected"], "dups:", s["duplicates"])


if __name__ == "__main__":
    main()
