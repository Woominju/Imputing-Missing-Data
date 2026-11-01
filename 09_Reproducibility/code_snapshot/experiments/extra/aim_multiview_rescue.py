"""AIM analysis, step 2: multi-view rescue under incomplete delay embeddings (eval-only; CPU).

Question: when the delay views are incomplete (no complete view at all, or no own-variable
observation in the window), does Model C (all views) recover the hidden value better than
Model B (own view only) on the same targets?

Protocol:
  - best checkpoints of the main runs (no retraining), main masks (prepare(); checked against
    the saved masks), main test targets. Every model is scored on exactly the same entries.
  - mask groups from aim_window_audit.window_records (patterns that occur naturally under the
    main MCAR evaluation; no input is removed, so the input distribution is that of the main test)
  - models A, B, C where checkpoints exist; references linear (uses future observations) and locf
  - improvement of X over Y: 100 (1 - RMSE_X / RMSE_Y); negative = X worse
  - uncertainty: paired block bootstrap over time blocks of the test split (same resampled
    targets for both models); pooled-over-seeds intervals resample blocks within each seed
  - completeness vs amount of information: complete_other_view and no_complete_view differ in
    n_obs_others by construction, so they are also compared within equal n_obs_others strata
    (matched table)
  - negative control: B never sees the other variables, and under MCAR their mask is independent
    of B's input and of the state, so B's RMSE should not differ between groups defined only by
    the other variables (complete_other_view vs no_complete_view) beyond sampling noise.
Caveats: C uses strictly more observed inputs than B, so C < B is evidence that combining views
helps, not of a diffeomorphic correspondence between views. The coupled Lorenz grid has no A/B
checkpoints, so C vs B is unavailable there without training B.
Overall test RMSE of every model is recomputed and compared with the main results CSV.
Output: results/extra/aim_rescue/<run>/{rescue_records.csv.gz, rescue_by_seed.csv,
        rescue_pooled.csv, rescue_by_target.csv, rescue_matched.csv, rescue_check.csv}
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from experiments.extra.aim_window_audit import (GROUPS, add_args, ckpt_path, delay_cfg, load_data,  # noqa: E402
                                                out_dir, parse, refuse_overwrite, window_records)
from src.evaluate import impute, baseline_impute  # noqa: E402
from src.models import build_model  # noqa: E402

RES = os.path.join(ROOT, "results")
MODELS = ("A", "B", "C")
REFS = ("linear", "locf")
PAIRS = [("C", "B"), ("C", "A"), ("C", "linear"), ("B", "A")]
MAIN_CSVS = ("lorenz_results.csv", "coupled_lorenz_results.csv", "real_data_results.csv")
MIN_N, MIN_BLOCKS, NBOOT = 30, 5, 1000
OUTPUTS = ("rescue_records.csv.gz", "rescue_by_seed.csv", "rescue_pooled.csv", "rescue_by_target.csv",
           "rescue_matched.csv", "rescue_check.csv")


def load_model(ds, name, p, seed, n):
    ck = torch.load(ckpt_path(ds, name, p, seed), map_location="cpu", weights_only=False)
    c = ck["cfg"]
    assert c["model"] == name and c["dataset"] == ds and c["seed"] == seed
    mdl = build_model(name, n, c["m"], c["latent"], c["hidden"], c["window"], c["direction"])
    mdl.load_state_dict(ck["state_dict"])
    return mdl, c


def main_rmse():
    d = pd.concat([pd.read_csv(os.path.join(RES, f)) for f in MAIN_CSVS if os.path.exists(os.path.join(RES, f))])
    return dict(zip(d.exp_id, d.test_rmse))


def records(datasets, rates, seeds, block):
    ref = main_rmse()
    recs, checks = [], []
    for ds in datasets:
        for p in rates:
            for seed in seeds:
                m, tau = delay_cfg(ds, p, seed)
                data = load_data(ds, p, seed)
                X, M = data["X"], data["M"]
                d = window_records(data, m, tau, block)
                tt, ii = d.t.values, d.target.values
                preds = {}
                for k in MODELS:
                    if os.path.exists(ckpt_path(ds, k, p, seed)):
                        mdl, c = load_model(ds, k, p, seed, X.shape[1])
                        preds[k] = impute(mdl, torch.tensor(X), torch.tensor(M), c["m"], c["tau"]).numpy()
                for k in REFS:
                    preds[k] = baseline_impute(k, X * M, M)
                for k, xh in preds.items():
                    d[f"err_{k}"] = xh[tt, ii] - X[tt, ii]
                    rmse = float(np.sqrt(np.mean(d[f"err_{k}"] ** 2)))
                    main = ref.get(f"{ds}_{k}_p{p}_s{seed}", np.nan)
                    checks.append({"dataset": ds, "p": p, "seed": seed, "model": k, "n_eval": len(d),
                                   "rmse_recomputed": rmse, "rmse_main_csv": main,
                                   "rel_diff": abs(rmse - main) / main})
                d.insert(0, "seed", seed)
                d.insert(0, "p", p)
                d.insert(0, "dataset", ds)
                recs.append(d)
                print(f"rescue {ds} p={p} seed={seed}: {len(d)} targets, models {sorted(preds)}", flush=True)
    return pd.concat(recs, ignore_index=True), pd.DataFrame(checks)


def block_sums(g, models):
    """Per-block squared-error sums (models x blocks) and counts for one (dataset, p, seed) stratum."""
    blocks = np.unique(g.block.values)
    idx = np.searchsorted(blocks, g.block.values)
    sq = np.stack([np.bincount(idx, g[f"err_{k}"].values ** 2, len(blocks)) for k in models])
    return sq, np.bincount(idx, minlength=len(blocks)).astype(float)


def paired_boot(strata, models, rng):
    """strata: list of (sq, cnt) per seed. Resample blocks within each stratum, pool, return
    bootstrap RMSE (NBOOT x models)."""
    sq_tot = np.zeros((NBOOT, len(models)))
    cnt_tot = np.zeros(NBOOT)
    for sq, cnt in strata:
        nb = len(cnt)
        pick = rng.integers(0, nb, (NBOOT, nb))
        sq_tot += sq[:, pick].sum(-1).T
        cnt_tot += cnt[pick].sum(-1)
    return np.sqrt(sq_tot / np.maximum(cnt_tot, 1)[:, None])


def compare(groups, models, rng):
    """groups: list of per-seed DataFrames (one stratum each). Point estimates on the pooled
    targets, paired block-bootstrap 95% intervals for every pair in PAIRS."""
    allg = pd.concat(groups)
    n = len(allg)
    nblocks = sum(g.block.nunique() for g in groups)
    r = {"n": n, "n_blocks": nblocks}
    for k in models:
        e = allg[f"err_{k}"].values
        r[f"rmse_{k}"] = float(np.sqrt(np.mean(e ** 2))) if n else np.nan
        r[f"mae_{k}"] = float(np.mean(np.abs(e))) if n else np.nan
    ok = n >= MIN_N and nblocks >= MIN_BLOCKS
    boot = paired_boot([block_sums(g, models) for g in groups if len(g)], models, rng) if ok else None
    for x, y in PAIRS:
        if x not in models or y not in models:
            continue
        r[f"imp_{x}_vs_{y}"] = 100 * (1 - r[f"rmse_{x}"] / r[f"rmse_{y}"]) if n else np.nan
        r[f"imp_mae_{x}_vs_{y}"] = 100 * (1 - r[f"mae_{x}"] / r[f"mae_{y}"]) if n else np.nan
        if ok:
            b = 100 * (1 - boot[:, models.index(x)] / boot[:, models.index(y)])
            r[f"imp_{x}_vs_{y}_lo"], r[f"imp_{x}_vs_{y}_hi"] = np.nanquantile(b, [0.025, 0.975])
        else:
            r[f"imp_{x}_vs_{y}_lo"] = r[f"imp_{x}_vs_{y}_hi"] = np.nan
    return r


def model_list(rec):
    return [k for k in MODELS + REFS if f"err_{k}" in rec and rec[f"err_{k}"].notna().all()]


def by_seed(rec, rng):
    rows = []
    for (ds, p, seed), g in rec.groupby(["dataset", "p", "seed"]):
        models = model_list(g)
        for name, f in GROUPS.items():
            rows.append({"dataset": ds, "p": p, "seed": seed, "group": name, **compare([g[f(g)]], models, rng)})
    return pd.DataFrame(rows)


def pooled(rec, rng, extra=()):
    rows = []
    keys = ["dataset", "p"] + list(extra)
    for kv, g in rec.groupby(keys):
        models = model_list(g)
        for name, f in GROUPS.items():
            per_seed = [s[f(s)] for _, s in g.groupby("seed")]
            r = dict(zip(keys, kv))
            r.update({"group": name, "n_seeds": len(per_seed), **compare(per_seed, models, rng)})
            if "B" in models and "C" in models:
                imps = [100 * (1 - np.sqrt(np.mean(s.err_C ** 2)) / np.sqrt(np.mean(s.err_B ** 2)))
                        for s in per_seed if len(s)]
                r["imp_C_vs_B_seed_mean"] = np.mean(imps) if imps else np.nan
                r["imp_C_vs_B_seed_sd"] = np.std(imps, ddof=1) if len(imps) > 1 else np.nan
                r["n_seeds_C_better"] = int(sum(i > 0 for i in imps))
            rows.append(r)
    return pd.DataFrame(rows)


def matched(rec, rng, min_n=20):
    """complete_other_view vs no_complete_view at equal n_obs_others (pooled over seeds).
    MSE per stratum, averaged with the combined stratum counts as weights, so both groups get
    the same distribution of observation counts. 95% intervals of the RMSE ratio from a block
    bootstrap (blocks resampled within each seed; stratum weights fixed)."""
    rows = []
    for (ds, p), g in rec.groupby(["dataset", "p"]):
        models = model_list(g)
        comp = g.n_complete_views >= 1
        cnt = pd.crosstab(g.n_obs_others, comp)
        if cnt.shape[1] < 2:
            continue
        keep = cnt.index[(cnt[True] >= min_n) & (cnt[False] >= min_n)]
        gg = g[g.n_obs_others.isin(keep)]
        r = {"dataset": ds, "p": p, "strata_n_obs_others": ",".join(map(str, keep)),
             "n_complete": int(cnt.loc[keep, True].sum()), "n_no_complete": int(cnt.loc[keep, False].sum()),
             "n_excluded": int(len(g) - len(gg))}
        if not len(keep):
            rows.append(r)
            continue
        w = (cnt.loc[keep].sum(1) / cnt.loc[keep].values.sum()).values
        # sums per (seed-block unit, stratum, complete flag, model)
        unit = pd.factorize(list(zip(gg.seed, gg.block)))[0]
        s = np.searchsorted(keep.values, gg.n_obs_others.values)
        f = (gg.n_complete_views.values >= 1).astype(int)
        U, S = unit.max() + 1, len(keep)
        sq = np.zeros((U, S, 2, len(models)))
        c = np.zeros((U, S, 2))
        np.add.at(c, (unit, s, f), 1)
        for k, mname in enumerate(models):
            np.add.at(sq[..., k], (unit, s, f), gg[f"err_{mname}"].values ** 2)
        useed = pd.Series(gg.seed.values).groupby(unit).first().values

        def rmse_wt(sq_, c_):  # -> (2, models); index 0 = no_complete, 1 = complete
            mse = np.where(c_[..., None] > 0, sq_ / np.maximum(c_, 1)[..., None], np.nan)  # empty stratum -> NaN
            return np.sqrt(np.einsum("s,sfk->fk", w, mse))

        est = rmse_wt(sq.sum(0), c.sum(0))
        boot = []
        for _ in range(NBOOT):
            pick = np.concatenate([rng.choice(np.where(useed == sd)[0], (useed == sd).sum()) for sd in np.unique(useed)])
            boot.append(rmse_wt(sq[pick].sum(0), c[pick].sum(0)))
        boot = np.array(boot)
        for k, mname in enumerate(models):
            r[f"rmse_{mname}_complete"], r[f"rmse_{mname}_no_complete"] = est[1, k], est[0, k]
            r[f"ratio_{mname}_no_complete_over_complete"] = est[0, k] / est[1, k]
            r[f"ratio_{mname}_lo"], r[f"ratio_{mname}_hi"] = np.nanquantile(boot[:, 0, k] / boot[:, 1, k], [0.025, 0.975])
        rows.append(r)
    return pd.DataFrame(rows)


def print_table(pool):
    cols = ["dataset", "p", "group", "n", "rmse_B", "rmse_C", "imp_C_vs_B", "imp_C_vs_B_lo", "imp_C_vs_B_hi",
            "n_seeds_C_better", "rmse_A", "rmse_linear"]
    print(pool[[c for c in cols if c in pool]].round(4).to_string(index=False))


def main():
    ap = add_args(argparse.ArgumentParser())
    args = ap.parse_args()
    torch.set_num_threads(4)
    out = out_dir(args.run)
    for f in OUTPUTS:
        refuse_overwrite(os.path.join(out, f))
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(0)
    rec, chk = records(*parse(args), args.block)
    chk.to_csv(os.path.join(out, "rescue_check.csv"), index=False)
    print("overall test RMSE vs main CSV: max rel diff", chk.rel_diff.max(), "| rows", len(chk))
    assert chk.rel_diff.max() < 1e-4, "re-imputed RMSE differs from the main results"
    rec.to_csv(os.path.join(out, "rescue_records.csv.gz"), index=False)
    by_seed(rec, rng).to_csv(os.path.join(out, "rescue_by_seed.csv"), index=False)
    pool = pooled(rec, rng)
    pool.to_csv(os.path.join(out, "rescue_pooled.csv"), index=False)
    pooled(rec, rng, extra=("target",)).to_csv(os.path.join(out, "rescue_by_target.csv"), index=False)
    mt = matched(rec, rng)
    mt.to_csv(os.path.join(out, "rescue_matched.csv"), index=False)
    print_table(pool)
    print(mt[[c for c in mt if c.startswith("ratio_") or c in ("dataset", "p", "strata_n_obs_others", "n_complete",
                                                                 "n_no_complete")]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
