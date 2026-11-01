"""Writes results/logs/overnight_status.txt: what actually completed / did not, from files on disk.
Safe to run any time (read-only except for the status file)."""
import glob
import os
import subprocess
import sys
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from experiments.common import read_part  # noqa: E402

RES = os.path.join(ROOT, "results")
EXPECTED = {"lorenz_p": 21, "coupled_k": 60, "real_": 72}


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"(failed: {e})"


def tail(path, n=12):
    if not os.path.exists(path):
        return "(missing)"
    with open(path) as f:
        return "".join(f.readlines()[-n:]).rstrip()


def main():
    L = [f"# Overnight status  (written {datetime.now():%Y-%m-%d %H:%M:%S} KST)", ""]
    L += ["## GPU reservation / usage", sh("labres-check 2>&1 | tail -n 6"),
          "gpu apps: " + sh("nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\\n' ' '"),
          "watchdog log tail:", tail(os.path.join(RES, "logs", "watchdog.log"), 6), ""]
    L += ["## Main grid completeness (rows written / planned)"]
    allp = []
    for f in sorted(glob.glob(os.path.join(RES, "parts", "*.csv"))):
        d = read_part(f)
        exp = next(v for k, v in EXPECTED.items() if os.path.basename(f).startswith(k))
        err = int(d["error"].notna().sum()) if "error" in d else 0
        L.append(f"  {os.path.basename(f):28s} {len(d):3d}/{exp:3d}  {'COMPLETE' if len(d) >= exp else 'INCOMPLETE'}"
                 f"  error_rows={err}")
        allp.append(d)
    df = pd.concat(allp, ignore_index=True)
    L += ["", "## Test RMSE (normalised units) mean +- sd over seeds [n seeds]; MAE mean"]
    g = df.dropna(subset=["test_rmse"]).groupby(["dataset", "missing_rate", "model"]).agg(
        r=("test_rmse", "mean"), s=("test_rmse", "std"), a=("test_mae", "mean"), n=("seed", "nunique")).reset_index()
    g["cell"] = g.apply(lambda x: f"{x.r:.4f}+-{0 if np.isnan(x.s) else x.s:.4f}[{x.n}] mae {x.a:.4f}", axis=1)
    order = [m for m in ["mean", "locf", "linear", "A", "B", "C", "D"] if m in g.model.unique()]
    piv = g.pivot_table(index=["dataset", "missing_rate"], columns="model", values="cell", aggfunc="first")[order]
    L.append(piv.to_string())
    L += ["", "## Verification outputs"]
    for name in ["verification/recompute_check.csv"]:
        p = os.path.join(RES, name)
        if os.path.exists(p):
            v = pd.read_csv(p)
            rel = v[[c for c in v.columns if c.endswith("rel_diff")]]
            L.append(f"  {name}: rows={len(v)} max_rel_diff={float(rel.max().max()):.2e} "
                     f"mask_mismatch={int((~v.mask_file_matches).sum())} "
                     f"(file time {datetime.fromtimestamp(os.path.getmtime(p)):%m-%d %H:%M})")
    L.append("  verify_results / make_figures: see results/logs/overnight_queue.log")
    L += ["", "## Additional experiments (results/extra/)"]
    for lab, pat in [("Exp A tau sensitivity (GPU0)", "expA_tau/lorenz_tau*.csv"),
                     ("Exp A summary", "expA_tau/expA_summary.csv"),
                     ("Exp B contiguous gaps (CPU, eval-only)", "expB_contiguous/overnight/expB_summary.csv"),
                     ("Exp C cross-view ablation (CPU, eval-only)", "expC_crossview/overnight/expC_summary.csv"),
                     ("Exp D observability sensitivity (CPU)", "expD_observability/overnight/expD_spearman.csv")]:
        fs = glob.glob(os.path.join(RES, "extra", pat))
        if not fs:
            L.append(f"  {lab}: NOT AVAILABLE")
            continue
        n = sum(len(read_part(f)) if "lorenz_tau" in f else len(pd.read_csv(f)) for f in fs)
        L.append(f"  {lab}: {len(fs)} file(s), {n} rows")
    ea = glob.glob(os.path.join(RES, "extra", "expA_tau", "lorenz_tau*.csv"))
    if ea:
        d = pd.concat([read_part(f) for f in ea])
        L.append(f"  Exp A runs done {len(d)}/36: " + ", ".join(
            f"tau{int(t)}={len(x)}" for t, x in d.groupby("tau")))
    L += ["", "## Errors in logs"]
    errs = sh(f"grep -lE 'Traceback|ERROR|Killed|out of memory' {RES}/logs/run_*.txt {RES}/extra/expA_tau/run_*.txt "
              f"{RES}/logs/overnight_queue.log 2>/dev/null")
    L.append(errs or "  none found")
    skips = sh(f"grep -h '^SKIP' {RES}/extra/expA_tau/run_*.txt 2>/dev/null")
    if skips:
        L += ["  runs skipped by GPU deadline guard:", skips]
    L += ["", "## Flags", f"  STOP_GPU exists: {os.path.exists(os.path.join(RES, 'logs', 'STOP_GPU'))}",
          f"  QUEUE_DONE exists: {os.path.exists(os.path.join(RES, 'logs', 'QUEUE_DONE'))}"]
    L += ["", "## Not automated (needs a person): interpretation, FPRM/Lag-FPRM/MVCM (no implementation available),",
          "   retraining with contiguous-gap masks (Exp B variant), resuming incomplete grids (README: --skip_from)"]
    with open(os.path.join(RES, "logs", "overnight_status.txt"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L[:40]))


if __name__ == "__main__":
    main()
