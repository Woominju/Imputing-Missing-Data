"""AIM analysis, step 2b: A vs B vs C on the rescue targets, stratified by current-time information.

Reads the per-target errors of aim_multiview_rescue (same checkpoints, masks and targets; no
model is evaluated again) and adds the stratum
  others_t = number of other variables observed at time t (= others_obs_at_t).
Within one stratum, A and C receive the same observed values at time t (target hidden), and C
additionally receives the lags t - k tau (k >= 1) of every variable. So within a stratum
  sigma(inputs of A) is contained in sigma(inputs of C)
and C vs A measures the practical contribution of the delay (lagged) inputs. Groups are still
the mask groups of aim_window_audit (GROUPS); others_t = "any" pools the strata.
Datasets without A, B and C checkpoints (coupled Lorenz) are skipped.
Output: results/extra/aim_rescue/<run>/{abc_pooled.csv, abc_by_target.csv, abc_by_seed.csv}
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from experiments.extra.aim_multiview_rescue import pooled  # noqa: E402
from experiments.extra.aim_window_audit import out_dir, refuse_overwrite  # noqa: E402

OUTPUTS = ("abc_pooled.csv", "abc_by_target.csv", "abc_by_seed.csv")


def with_any(rec):
    """Records once per stratum others_t and once more with others_t = 'any'."""
    a = rec.assign(others_t=rec.others_obs_at_t.astype(str))
    return pd.concat([a, rec.assign(others_t="any")], ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="v1", help="aim_rescue run holding rescue_records.csv.gz")
    ap.add_argument("--run", default="v1_abc")
    args = ap.parse_args()
    out = out_dir(args.run)
    for f in OUTPUTS:
        refuse_overwrite(os.path.join(out, f))
    rec = pd.read_csv(os.path.join(out_dir(args.src), "rescue_records.csv.gz"))
    keep = [ds for ds, g in rec.groupby("dataset") if all(g[f"err_{k}"].notna().all() for k in "ABC")]
    print("datasets with A, B and C:", keep, "| skipped:", sorted(set(rec.dataset) - set(keep)))
    rec = with_any(rec[rec.dataset.isin(keep)])
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(0)
    seed = pooled(rec, rng, extra=("others_t", "seed"))
    imp = lambda s, x, y: 100 * (1 - s[f"rmse_{x}"] / s[f"rmse_{y}"])  # noqa: E731
    seed["imp_C_vs_A_seed"] = imp(seed, "C", "A")
    seed.to_csv(os.path.join(out, "abc_by_seed.csv"), index=False)
    pool = pooled(rec, rng, extra=("others_t",))
    sv = seed[seed.n > 0].groupby(["dataset", "p", "others_t", "group"]).imp_C_vs_A_seed
    pool = pool.merge(sv.agg(imp_C_vs_A_seed_mean="mean", imp_C_vs_A_seed_sd="std",
                             n_seeds_C_better_than_A=lambda x: int((x > 0).sum())).reset_index(),
                      on=["dataset", "p", "others_t", "group"], how="left")
    pool.to_csv(os.path.join(out, "abc_pooled.csv"), index=False)
    pooled(rec, rng, extra=("others_t", "target")).to_csv(os.path.join(out, "abc_by_target.csv"), index=False)
    cols = ["p", "group", "others_t", "n", "n_blocks", "rmse_A", "rmse_B", "rmse_C", "imp_C_vs_A", "imp_C_vs_A_lo",
            "imp_C_vs_A_hi", "n_seeds_C_better_than_A", "imp_C_vs_B", "imp_C_vs_B_lo", "imp_C_vs_B_hi"]
    sel = pool[pool.group.isin(["all", "no_complete_view", "no_own_obs"]) & (pool.n > 0)]
    print(sel[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
