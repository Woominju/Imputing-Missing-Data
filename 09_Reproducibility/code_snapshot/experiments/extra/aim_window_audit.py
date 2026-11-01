"""AIM analysis, step 1: completeness of the delay views at every evaluated test entry (no models).

For the main test targets (t, i) (eval_mask = 1 in the test split; never fed to any model), the
input mask M of the main evaluation (1 = observed) and the checkpoints' m and tau:
  W_j(t) = sum_{k=0}^{m-1} M[t - k tau, j]     observed entries in view j's delay window
  c_j(t) = prod_{k=0}^{m-1} M[t - k tau, j]    view j complete (W_j(t) == m)
Indices before the series start count as missing (as pad_front does).
The target itself is hidden (M[t, i] = 0), so the target's own view is never complete and
own_obs = W_i(t) counts lags k = 1..m-1 only. A complete view at (t, i) is therefore always a
view of another variable, and "no complete view" (sum_j c_j(t) = 0) is a condition on the others.
Mask groups (overlapping flags, see GROUPS):
  complete_other_view    : at least one other variable's view is complete
  no_complete_view       : no view is complete
  no_own_obs             : W_i(t) = 0 (the whole own window is missing)
  no_own_obs_no_complete : both of the last two
MCAR reference fractions (independent entries, M_orig = 1): P(no_own_obs) = p^(m-1),
P(view j complete) = (1-p)^m, P(no_complete_view) = (1 - (1-p)^m)^(n-1).
Patterns arise naturally from the main evaluation masks; no input is removed.
Output: results/extra/aim_rescue/<run>/{window_audit.csv.gz, window_audit_summary.csv}
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from experiments.recompute_from_checkpoints import raw_data  # noqa: E402
from src.missingness import prepare  # noqa: E402

RES = os.path.join(ROOT, "results")
GROUPS = {
    "all": lambda d: np.ones(len(d), bool),
    "complete_other_view": lambda d: d.n_complete_views.values >= 1,
    "no_complete_view": lambda d: d.n_complete_views.values == 0,
    "no_own_obs": lambda d: d.own_obs.values == 0,
    "no_own_obs_no_complete": lambda d: (d.own_obs.values == 0) & (d.n_complete_views.values == 0),
}


def out_dir(run):
    return os.path.join(RES, "extra", "aim_rescue", run)


def refuse_overwrite(path):
    if os.path.exists(path):
        raise SystemExit(f"{path} already exists; refusing to overwrite (use another --run)")


def ckpt_path(ds, model, p, seed):
    return os.path.join(RES, "checkpoints", f"{ds}_{model}_p{p}_s{seed}.pt")


def delay_cfg(ds, p, seed, models="ABCD"):
    """m, tau shared by every delay checkpoint of (ds, p, seed)."""
    cfgs = {k: torch.load(ckpt_path(ds, k, p, seed), map_location="cpu", weights_only=False)["cfg"]
            for k in models if k != "A" and os.path.exists(ckpt_path(ds, k, p, seed))}
    assert cfgs, f"no delay checkpoint for {ds} p={p} seed={seed}"
    mt = {(c["m"], c["tau"]) for c in cfgs.values()}
    assert len(mt) == 1, f"m/tau differ across checkpoints: {mt}"
    return mt.pop()


def load_data(ds, p, seed):
    """prepare() masks, checked against the masks saved by the main run."""
    data = prepare(raw_data(ds), p, seed)
    mk = np.load(os.path.join(RES, "imputations", f"{ds}_p{p}_s{seed}_masks.npz"))
    ok = (np.array_equal(mk["M"], data["M"]) and np.array_equal(mk["eval_mask"], data["eval_mask"])
          and json.loads(str(mk["split"])) == {k: list(v) for k, v in data["split"].items()})
    assert ok, f"masks of {ds} p={p} seed={seed} differ from the saved main-run masks"
    return data


def window_records(data, m, tau, block):
    """One row per evaluated test entry (t, i), in (t, i) order."""
    M = data["M"]
    T, n = M.shape
    span = (m - 1) * tau
    Mp = np.concatenate([np.zeros((span, n), M.dtype), M])
    W = sum(Mp[span - k * tau: span - k * tau + T] for k in range(m)).astype(int)
    full = W == m
    a, b = data["split"]["test"]
    tt, ii = np.nonzero(data["eval_mask"][a:b] > 0)
    tt = tt + a
    assert np.all(M[tt, ii] == 0), "an evaluation target is visible in the input mask"
    comp = full[tt]
    assert not comp[np.arange(len(tt)), ii].any(), "target view complete although the target is hidden"
    Wt = W[tt]
    own = Wt[np.arange(len(tt)), ii]
    d = pd.DataFrame({"t": tt, "target": ii, "block": (tt - a) // block, "own_obs": own,
                      "n_obs_others": Wt.sum(1) - own, "n_obs_total": Wt.sum(1),
                      "n_complete_views": comp.sum(1), "others_obs_at_t": M[tt].sum(1).astype(int)})
    for j in range(n):
        d[f"obs_v{j}"] = Wt[:, j]
    return d


def summarise(rec, m, n):
    rows = []
    for (ds, pp, seed), g in rec.groupby(["dataset", "p", "seed"]):
        r = {"dataset": ds, "p": pp, "seed": seed, "m": m, "n_vars": n, "n_targets": len(g),
             "own_obs_mean": g.own_obs.mean(), "n_obs_others_mean": g.n_obs_others.mean()}
        for name, f in GROUPS.items():
            sel = f(g)
            r[f"n_{name}"] = int(sel.sum())
            r[f"frac_{name}"] = float(sel.mean())
        q = (1 - pp) ** m
        r["mcar_frac_no_own_obs"] = pp ** (m - 1)
        r["mcar_frac_no_complete_view"] = (1 - q) ** (n - 1)
        r["mcar_frac_complete_other_view"] = 1 - (1 - q) ** (n - 1)
        r["mcar_frac_no_own_obs_no_complete"] = pp ** (m - 1) * (1 - q) ** (n - 1)
        rows.append(r)
    return pd.DataFrame(rows)


def audit(datasets, rates, seeds, block):
    recs, sums = [], []
    for ds in datasets:
        for p in rates:
            for seed in seeds:
                m, tau = delay_cfg(ds, p, seed)
                data = load_data(ds, p, seed)
                d = window_records(data, m, tau, block)
                d.insert(0, "seed", seed)
                d.insert(0, "p", p)
                d.insert(0, "dataset", ds)
                d["m"], d["tau"] = m, tau
                recs.append(d)
                sums.append(summarise(d, m, data["M"].shape[1]))
                print(f"audit {ds} p={p} seed={seed}: {len(d)} targets", flush=True)
    return pd.concat(recs, ignore_index=True), pd.concat(sums, ignore_index=True)


def add_args(ap):
    ap.add_argument("--datasets", default="lorenz")
    ap.add_argument("--rates", default="0.1,0.3,0.5,0.7")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--block", type=int, default=100, help="time steps per bootstrap block")
    ap.add_argument("--run", default=os.environ.get("IMD_EXTRA_RUN", ""), help="output sub-folder")
    ap.add_argument("--smoke", action="store_true", help="p=0.3, seed 0 only")
    return ap


def parse(args):
    rates = [0.3] if args.smoke else [float(x) for x in args.rates.split(",")]
    seeds = [0] if args.smoke else [int(x) for x in args.seeds.split(",")]
    return args.datasets.split(","), rates, seeds


def main():
    args = add_args(argparse.ArgumentParser()).parse_args()
    torch.set_num_threads(4)
    out = out_dir(args.run)
    for f in ("window_audit.csv.gz", "window_audit_summary.csv"):
        refuse_overwrite(os.path.join(out, f))
    os.makedirs(out, exist_ok=True)
    rec, s = audit(*parse(args), args.block)
    rec.to_csv(os.path.join(out, "window_audit.csv.gz"), index=False)
    s.to_csv(os.path.join(out, "window_audit_summary.csv"), index=False)
    cols = ["dataset", "p", "seed", "n_targets"] + [f"frac_{g}" for g in GROUPS if g != "all"] + \
           ["mcar_frac_complete_other_view", "mcar_frac_no_complete_view", "mcar_frac_no_own_obs"]
    print(s[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
