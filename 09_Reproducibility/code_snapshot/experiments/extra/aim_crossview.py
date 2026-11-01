"""AIM cross-view analysis (Exp 1): can Model C's view-specific inputs support reconstruction of the
other variables, and where does it break down? Eval-only, CPU, saved Lorenz checkpoints.

What is tested: x_j(t) predicted by the shared decoder from inputs restricted to other views.
The decoder outputs the current value x_j(t), not the delay vector F_j, so this is NOT a test
of learning F_j o F_i^{-1}; it is a diagnostic of cross-view reconstruction of current values.

Targets: the main test targets (t, j) (eval_mask = 1; never fed as input). Input masks of Model C:
  multi          main mask M (all views, natural MCAR) = the original Model C
  others_nat     M with the target view j removed (other views natural; = Exp C "others_only")
  own_nat        only view j (natural; = Exp C "own_only")
  from_<i>_nat   only view i != j, natural MCAR entries of view i         (single-view input)
  from_<i>_full  only view i != j, every entry of view i observed (M_orig) (OOD diagnostic:
                 reveals entries of variable i hidden in the main mask, never x_j(t))
Single-view inputs are rare in training (inputs miss p + hide_rate (1 - p)); see
exp1_training_frequency.csv. References on the same targets: A, B (main checkpoints), linear
(uses future observations), locf, mean (0 after normalisation).
Symmetry control: Gamma(x, y, z) = (-x, -y, z) commutes with the Lorenz flow, so Gamma applied to
the whole series is another exact trajectory (checked by re-integration with the generator's
solver). It is used for evaluation only. Under the same mask, z-only inputs of the original and
the Gamma series are identical, so any z-only predictor gives the same output for both, and its
pooled MSE on x (or y) is bounded below by mean((a - b)^2) / 4 (a, b = the two truths).
Uncertainty: paired block bootstrap over 100-step blocks of the test split (within seed; pooled
intervals resample blocks within each seed).
Output: results/extra/aim_crossview/<run>/
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.integrate import solve_ivp

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from experiments.extra.aim_multiview_rescue import block_sums, load_model, main_rmse, paired_boot  # noqa: E402
from experiments.extra.aim_window_audit import GROUPS, load_data, refuse_overwrite, window_records  # noqa: E402
from experiments.run_lorenz import get_lorenz  # noqa: E402
from src.evaluate import baseline_impute, impute  # noqa: E402
from src.lorenz import lorenz_rhs  # noqa: E402

RES = os.path.join(ROOT, "results")
NAMES = ["x", "y", "z"]
GAMMA = np.array([-1.0, -1.0, 1.0])
BLOCK, NBOOT = 100, 1000
OBS_BINS = [0, 1, 5, 8, 9]  # view-i observed entries: 0 | 1-4 | 5-7 | 8 (complete)
OUTPUTS = ("exp1_records.csv.gz", "exp1_by_seed.csv", "exp1_pooled.csv", "exp1_by_view_obs.csv",
           "exp1_groups.csv", "exp1_symmetry.csv", "exp1_checks.csv", "exp1_training_frequency.csv")


def predict_all(models, X, M, Mo, m, tau):
    """Predictions (T, n) for every input condition / reference; target column chosen later."""
    C = models["C"][0]
    run = lambda Mi: impute(C, torch.tensor(X), torch.tensor(Mi.astype(np.float32)), m, tau).numpy()  # noqa: E731
    n = X.shape[1]
    P = {"multi": run(M)}
    for i in range(n):
        Mi = np.zeros_like(M)
        Mi[:, i] = M[:, i]
        P[f"view{i}_nat"] = run(Mi)
        Mf = np.zeros_like(M)
        Mf[:, i] = Mo[:, i]
        P[f"view{i}_full"] = run(Mf)
        Mo_ = M.copy()
        Mo_[:, i] = 0
        P[f"without{i}"] = run(Mo_)
    for k in ("A", "B"):
        if k in models:
            mdl, c = models[k]
            P[k] = impute(mdl, torch.tensor(X), torch.tensor(M), c["m"], c["tau"]).numpy()
    for k in ("linear", "locf"):
        P[k] = baseline_impute(k, X * M, M)
    P["mean"] = np.zeros_like(X)
    return P


def errors(rec, P, X, n):
    """Wide per-target error columns; single-view columns are NaN where the input view is the target."""
    tt, jj = rec.t.values, rec.target.values
    out = {}
    for k in ("multi", "A", "B", "linear", "locf", "mean"):
        if k in P:
            out[f"err_{k}"] = P[k][tt, jj] - X[tt, jj]
    out["err_others_nat"] = np.array([P[f"without{j}"][t, j] for t, j in zip(tt, jj)]) - X[tt, jj]
    out["err_own_nat"] = np.array([P[f"view{j}_nat"][t, j] for t, j in zip(tt, jj)]) - X[tt, jj]
    for i in range(n):
        for kind in ("nat", "full"):
            e = P[f"view{i}_{kind}"][tt, jj] - X[tt, jj]
            out[f"err_from_{NAMES[i]}_{kind}"] = np.where(jj == i, np.nan, e)
    return pd.DataFrame(out, index=rec.index)


def gamma_series(X, mu, sd):
    """Normalised Gamma-partner series; the z column is copied exactly."""
    raw = X.astype(np.float64) * sd + mu
    XG = ((raw * GAMMA - mu) / sd).astype(np.float32)
    XG[:, 2] = X[:, 2]
    return XG


def check_gamma_trajectory(t0, steps=300):
    """Re-integrate s(t0) and Gamma s(t0) with the generator's solver; both deviations from the
    stored (resp. Gamma-transformed stored) series should be equally small."""
    raw, prm = get_lorenz()
    te = np.arange(steps) * prm["dt"]
    kw = dict(method=prm["solver"], t_eval=te, rtol=prm["rtol"], atol=prm["atol"])
    s = solve_ivp(lorenz_rhs, (0, te[-1]), raw[t0], **kw).y.T
    g = solve_ivp(lorenz_rhs, (0, te[-1]), raw[t0] * GAMMA, **kw).y.T
    ref = raw[t0:t0 + steps]
    return {"t0": t0, "steps": steps, "max_dev_original": float(np.abs(s - ref).max()),
            "max_dev_gamma": float(np.abs(g - ref * GAMMA).max()),
            "max_dev_gamma_vs_gamma_of_reintegrated": float(np.abs(g - s * GAMMA).max())}


def training_frequency(rates, m, n=3, hide=0.2):
    rows = []
    for p in rates:
        q = p + hide * (1 - p)
        e = q ** m
        rows.append({"p": p, "train_input_missing_rate": q, "P_view_empty": e,
                     "P_exactly_one_nonempty_view": n * (1 - e) * e ** (n - 1),
                     "P_view_complete_train": (1 - q) ** m, "P_view_complete_eval": (1 - p) ** m})
    return pd.DataFrame(rows)


def summarise(rec, keys, conds, rng, ref="multi"):
    rows = []
    for kv, g in rec.groupby(keys):
        per_seed = [s for _, s in g.groupby("seed")]
        for cnd in conds:
            col = f"err_{cnd}"
            if col not in g or g[col].isna().all():
                continue
            assert g[col].notna().all(), f"partial NaN in {col}"
            e, er = g[col].values, g[f"err_{ref}"].values
            r = dict(zip(keys, kv if isinstance(kv, tuple) else (kv,)))
            r.update({"condition": cnd, "n": len(g), "n_seeds": len(per_seed),
                      "rmse": float(np.sqrt(np.mean(e ** 2))), "mae": float(np.mean(np.abs(e))),
                      "rmse_ref": float(np.sqrt(np.mean(er ** 2)))})
            r["ratio_to_ref"] = r["rmse"] / r["rmse_ref"]
            seed_rmse = [np.sqrt(np.mean(s[col].values ** 2)) for s in per_seed]
            r["rmse_seed_mean"], r["rmse_seed_sd"] = np.mean(seed_rmse), (np.std(seed_rmse, ddof=1) if len(seed_rmse) > 1 else np.nan)
            nblocks = sum(s.block.nunique() for s in per_seed)
            if len(g) >= 30 and nblocks >= 5:
                bt = paired_boot([block_sums(s, [cnd, ref]) for s in per_seed], [cnd, ref], rng)
                r["rmse_lo"], r["rmse_hi"] = np.nanquantile(bt[:, 0], [0.025, 0.975])
                r["ratio_lo"], r["ratio_hi"] = np.nanquantile(bt[:, 0] / bt[:, 1], [0.025, 0.975])
            rows.append(r)
    return pd.DataFrame(rows)


def run(rates, seeds, out):
    recs, checks, sym = [], [], []
    ref_rmse = main_rmse()
    for p in rates:
        for seed in seeds:
            data = load_data("lorenz", p, seed)
            X, M, Mo = data["X"], data["M"], data["M_orig"]
            n = X.shape[1]
            models = {k: load_model("lorenz", k, p, seed, n) for k in "ABC"}
            m, tau = models["C"][1]["m"], models["C"][1]["tau"]
            rec = window_records(data, m, tau, BLOCK)
            tt, jj = rec.t.values, rec.target.values
            P = predict_all(models, X, M, Mo, m, tau)
            # the target is never an input: zero in every input mask used for its own prediction
            assert np.all(M[tt, jj] == 0)
            E = errors(rec, P, X, n)
            for k in ("multi", "A", "B"):
                key = "C" if k == "multi" else k
                rm = float(np.sqrt(np.nanmean(E[f"err_{k}"] ** 2)))
                main = ref_rmse[f"lorenz_{key}_p{p}_s{seed}"]
                checks.append({"p": p, "seed": seed, "check": f"overall RMSE {key} vs main CSV", "value": rm,
                               "reference": main, "rel_diff": abs(rm - main) / main})
            # Gamma partner: same mask, evaluation only
            XG = gamma_series(X, data["mu"].astype(np.float64), data["sd"].astype(np.float64))
            PG = predict_all({"C": models["C"]}, XG, M, Mo, m, tau)
            for kind in ("nat", "full"):
                d = float(np.abs(PG[f"view2_{kind}"] - P[f"view2_{kind}"]).max())
                checks.append({"p": p, "seed": seed, "check": f"z-only ({kind}) prediction identical on Gamma series",
                               "value": d, "reference": 0.0, "rel_diff": d})
            for cond in ("multi", "view0_nat", "view1_nat", "view2_nat", "view0_full", "view1_full", "view2_full"):
                for j in range(n):
                    sel = jj == j
                    if cond.startswith(f"view{j}_"):
                        continue  # own view of the target; full would reveal it
                    a, b = X[tt[sel], j], XG[tt[sel], j]
                    pa, pb = P[cond][tt[sel], j], PG[cond][tt[sel], j]
                    # Gamma-equivariant counterpart of the original prediction, in normalised units
                    mu, sd = data["mu"][j], data["sd"][j]
                    pa_g = ((pa * sd + mu) * GAMMA[j] - mu) / sd
                    sym.append({"p": p, "seed": seed, "condition": cond, "target": NAMES[j], "n": int(sel.sum()),
                                "rmse_original": float(np.sqrt(np.mean((pa - a) ** 2))),
                                "rmse_gamma": float(np.sqrt(np.mean((pb - b) ** 2))),
                                "rmse_pooled": float(np.sqrt(np.mean(np.r_[(pa - a) ** 2, (pb - b) ** 2]))),
                                "pair_lower_bound_if_identical_input": float(np.sqrt(np.mean((a - b) ** 2) / 4)),
                                "max_abs_pred_diff": float(np.abs(pb - pa).max()),
                                "equivariance_rmse": float(np.sqrt(np.mean((pb - pa_g) ** 2)))})
            r = pd.concat([rec, E], axis=1)
            r.insert(0, "seed", seed)
            r.insert(0, "p", p)
            recs.append(r)
            print(f"exp1 p={p} seed={seed}: {len(r)} targets", flush=True)
    rec = pd.concat(recs, ignore_index=True)
    return rec, pd.DataFrame(checks), pd.DataFrame(sym)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rates", default="0.1,0.3,0.5,0.7")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--run", default="v1")
    ap.add_argument("--smoke", action="store_true", help="p=0.3, seed 0")
    args = ap.parse_args()
    torch.set_num_threads(4)
    rates = [0.3] if args.smoke else [float(x) for x in args.rates.split(",")]
    seeds = [0] if args.smoke else [int(x) for x in args.seeds.split(",")]
    out = os.path.join(RES, "extra", "aim_crossview", args.run)
    for f in OUTPUTS:
        refuse_overwrite(os.path.join(out, f))
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(0)
    rec, chk, sym = run(rates, seeds, out)
    rec["target_name"] = rec.target.map(dict(enumerate(NAMES)))
    traj = pd.DataFrame([check_gamma_trajectory(t0) for t0 in (17000, 18000, 19000)])
    for _, r in traj.iterrows():
        chk = pd.concat([chk, pd.DataFrame([{"p": np.nan, "seed": np.nan,
                                              "check": f"Gamma trajectory re-integration t0={int(r.t0)} (300 steps)",
                                              "value": r.max_dev_gamma, "reference": r.max_dev_original,
                                              "rel_diff": r.max_dev_gamma_vs_gamma_of_reintegrated}])])
    # consistency with Exp C (same masks, positions and checkpoints) where it overlaps
    ec = os.path.join(RES, "extra", "expC_crossview", "overnight", "expC_results.csv")
    if os.path.exists(ec):
        e = pd.read_csv(ec)
        e = e[e.model.eq("C") & e.condition.isin(["own_only", "others_only"])]
        for _, r in e.iterrows():
            if r.p not in rates or r.seed not in seeds:
                continue
            g = rec[(rec.p == r.p) & (rec.seed == r.seed) & (rec.target_name == r.target)]
            col = "err_own_nat" if r.condition == "own_only" else "err_others_nat"
            v = float(np.sqrt(np.mean(g[col] ** 2)))
            chk = pd.concat([chk, pd.DataFrame([{"p": r.p, "seed": r.seed, "check": f"Exp C {r.condition} {r.target}",
                                                  "value": v, "reference": r.rmse,
                                                  "rel_diff": abs(v - r.rmse) / r.rmse}])])
    chk.to_csv(os.path.join(out, "exp1_checks.csv"), index=False)
    bad = chk[~chk.check.str.startswith("Gamma trajectory") & (chk.rel_diff > 1e-4)]
    print(chk.groupby(chk.check.str.split(" ").str[:3].str.join(" ")).rel_diff.max().to_string())
    assert bad.empty, f"integrity checks failed:\n{bad}"
    rec.to_csv(os.path.join(out, "exp1_records.csv.gz"), index=False)
    sym.to_csv(os.path.join(out, "exp1_symmetry.csv"), index=False)
    training_frequency(rates, 8).to_csv(os.path.join(out, "exp1_training_frequency.csv"), index=False)
    conds = ["multi", "others_nat", "own_nat"] + [f"from_{v}_{k}" for v in NAMES for k in ("nat", "full")] + \
            ["A", "B", "linear", "locf", "mean"]
    summarise(rec, ["p", "seed", "target_name"], conds, rng).to_csv(os.path.join(out, "exp1_by_seed.csv"), index=False)
    pool = summarise(rec, ["p", "target_name"], conds, rng)
    pool.to_csv(os.path.join(out, "exp1_pooled.csv"), index=False)
    rows = []
    for i, v in enumerate(NAMES):
        sub = rec[rec.target != i].copy()
        sub["view_obs_bin"] = pd.cut(sub[f"obs_v{i}"], OBS_BINS, right=False).astype(str)
        s = summarise(sub, ["p", "target_name", "view_obs_bin"], [f"from_{v}_nat", f"from_{v}_full", "others_nat"], rng)
        s.insert(2, "input_view", v)
        rows.append(s)
    pd.concat(rows).to_csv(os.path.join(out, "exp1_by_view_obs.csv"), index=False)
    gl = []
    for name, f in GROUPS.items():
        s = summarise(rec[f(rec)], ["p", "target_name"], ["multi", "others_nat", "own_nat", "A", "B", "linear"], rng)
        s.insert(0, "group", name)
        gl.append(s)
    pd.concat(gl).to_csv(os.path.join(out, "exp1_groups.csv"), index=False)
    piv = pool.pivot_table(index=["p", "target_name"], columns="condition", values="rmse")
    print(piv[[c for c in conds if c in piv]].round(3).to_string())
    print(sym.groupby(["condition", "target", "p"])[["rmse_original", "rmse_gamma", "rmse_pooled",
                                                     "pair_lower_bound_if_identical_input", "max_abs_pred_diff",
                                                     "equivariance_rmse"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
