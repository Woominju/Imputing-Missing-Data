"""Checks that result parts (incl. rows repaired by read_part) are correctly aligned:
1) exp_id fields == parsed model / missing_rate / seed / dataset
2) type/range checks on numeric columns and device
3) model rows: epochs_run == len(epoch log) and best_val_rmse == min(val_rmse) of the log
4) where imputations were saved: test RMSE/MAE recomputed from the .npy and masks"""
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from experiments.common import read_part  # noqa: E402
from src.train import RES  # noqa: E402

BASELINES = ("mean", "locf", "linear")


def main():
    problems, checked = [], {"rows": 0, "log_checks": 0, "recomputed": 0}
    for f in sorted(glob.glob(os.path.join(RES, "parts", "*.csv"))):
        d = read_part(f)
        for _, r in d.iterrows():
            checked["rows"] += 1
            m = re.fullmatch(r"(.+)_(mean|locf|linear|A|B|C|D)_p([0-9.]+)_s(\d+)", r.exp_id)
            if not m or (m[1], m[2], float(m[3]), int(m[4])) != (r.dataset, r.model, float(r.missing_rate), int(r.seed)):
                problems.append(f"{r.exp_id}: id/field mismatch")
                continue
            if isinstance(r.get("error"), str):
                continue
            for c in ["test_rmse", "test_mae", "test_rmse_orig", "test_mae_orig", "best_val_rmse", "train_time_s"]:
                if not (np.isfinite(r[c]) and r[c] >= 0):
                    problems.append(f"{r.exp_id}: bad {c}={r[c]}")
            if not r.test_mae <= r.test_rmse + 1e-9:
                problems.append(f"{r.exp_id}: MAE > RMSE")
            if r.device not in ("cpu", "NVIDIA L40S"):
                problems.append(f"{r.exp_id}: device={r.device}")
            if r.model not in BASELINES:
                if not (float(r.epochs_run).is_integer() and 1 <= r.epochs_run <= 150 and r.nonfinite_steps == 0
                        and float(r.n_params).is_integer()):
                    problems.append(f"{r.exp_id}: bad epochs/n_params/nonfinite")
                lp = os.path.join(RES, "logs", f"{r.exp_id}.csv")
                if os.path.exists(lp):
                    lg = pd.read_csv(lp)
                    checked["log_checks"] += 1
                    if len(lg) != r.epochs_run or not np.isclose(lg.val_rmse.min(), r.best_val_rmse, rtol=1e-6):
                        problems.append(f"{r.exp_id}: log mismatch (len {len(lg)} vs {r.epochs_run}, "
                                        f"min val {lg.val_rmse.min()} vs {r.best_val_rmse})")
                    if not np.isclose(lg.train_loss.iloc[-1], r.final_train_loss, rtol=1e-6, equal_nan=True):
                        problems.append(f"{r.exp_id}: final_train_loss mismatch")
                else:
                    problems.append(f"{r.exp_id}: no epoch log")
            xp = os.path.join(RES, "imputations", f"{r.exp_id}.npy")
            mp = os.path.join(RES, "imputations", f"{r.dataset}_p{r.missing_rate}_s{int(r.seed)}_masks.npz")
            if os.path.exists(xp) and os.path.exists(mp) and r.dataset == "lorenz":
                from experiments.run_lorenz import get_lorenz
                X, _ = get_lorenz()
                mk = np.load(mp)
                Xn = (X - mk["mu"]) / mk["sd"]
                a, b = json.loads(str(mk["split"]))["test"]
                E = mk["eval_mask"][a:b]
                dd = (np.load(xp)[a:b] - Xn[a:b]) * E
                rm, ma = np.sqrt((dd ** 2).sum() / E.sum()), np.abs(dd).sum() / E.sum()
                checked["recomputed"] += 1
                if not (np.isclose(rm, r.test_rmse, rtol=1e-4) and np.isclose(ma, r.test_mae, rtol=1e-4)):
                    problems.append(f"{r.exp_id}: recomputed rmse {rm:.6f} mae {ma:.6f} vs csv "
                                    f"{r.test_rmse:.6f} {r.test_mae:.6f}")
    print("checked", checked)
    print("problems:", len(problems))
    for p in problems:
        print("  ", p)
    return problems


if __name__ == "__main__":
    main()
