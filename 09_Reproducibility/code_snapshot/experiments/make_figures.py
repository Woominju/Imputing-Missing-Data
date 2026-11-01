"""Merge per-process result parts, write mean/std summaries and figures."""
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.train import RES  # noqa: E402
from experiments.run_lorenz import get_lorenz  # noqa: E402
from experiments.common import read_part  # noqa: E402

FIG = os.path.join(RES, "figures")
ORDER = ["mean", "locf", "linear", "A", "B", "C", "D"]


def merge(prefix, out):
    parts = sorted(glob.glob(os.path.join(RES, "parts", f"{prefix}*.csv")))
    if not parts:
        return None
    df = pd.concat([read_part(p) for p in parts], ignore_index=True).drop_duplicates("exp_id", keep="last")
    df.to_csv(os.path.join(RES, out), index=False)
    return df


def summarize(df, keys, out):
    g = df.dropna(subset=["test_rmse"]).groupby(keys + ["model", "missing_rate"])
    s = g.agg(rmse_mean=("test_rmse", "mean"), rmse_std=("test_rmse", "std"), mae_mean=("test_mae", "mean"),
              mae_std=("test_mae", "std"), n_seeds=("seed", "nunique")).reset_index()
    s.to_csv(os.path.join(RES, out), index=False)
    return s


def rmse_vs_rate(s, group_col, fname, title):
    groups = s[group_col].unique()
    fig, axes = plt.subplots(1, len(groups), figsize=(4.2 * len(groups), 3.6), squeeze=False)
    for ax, gname in zip(axes[0], groups):
        d = s[s[group_col] == gname]
        for mdl in [m for m in ORDER if m in d.model.unique()]:
            e = d[d.model == mdl].sort_values("missing_rate")
            ax.errorbar(e.missing_rate, e.rmse_mean, yerr=e.rmse_std.fillna(0), marker="o", capsize=3, label=mdl)
        ax.set_yscale("log"); ax.set_title(str(gname)); ax.set_xlabel("missing rate p")
        ax.set_ylabel("test RMSE (normalised)")
    axes[0][0].legend(fontsize=8)
    fig.suptitle(title); fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname), dpi=130); plt.close(fig)


def lorenz_series_figs():
    X, _ = get_lorenz()
    mk = os.path.join(RES, "imputations", "lorenz_p0.3_s0_masks.npz")
    if not os.path.exists(mk):
        return
    d = np.load(mk)
    M, sd, mu = d["M"], d["sd"], d["mu"]
    a = int(0.85 * len(X)); w = slice(a, a + 400)
    t = np.arange(len(X))[w]
    names = ["x", "y", "z"]
    fig, ax = plt.subplots(3, 1, figsize=(11, 6), sharex=True)
    for i in range(3):
        ax[i].plot(t, X[w, i], "k-", lw=1, label="true")
        miss = M[w, i] == 0
        ax[i].plot(t[miss], X[w, i][miss], "rx", ms=4, label="artificially missing")
        ax[i].set_ylabel(names[i])
    ax[0].legend(fontsize=8); ax[0].set_title("Lorenz test segment, p=0.3, seed 0: true series and missing positions")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "lorenz_missing_positions.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    for i in range(3):
        ax[i].plot(t, X[w, i], "k-", lw=1.5, label="true")
        for mdl in ["linear", "A", "B", "C", "D"]:
            f = os.path.join(RES, "imputations", f"lorenz_{mdl}_p0.3_s0.npy")
            if os.path.exists(f):
                xh = np.load(f)[w, i] * sd[i] + mu[i]
                miss = M[w, i] == 0
                ax[i].plot(t[miss], xh[miss], ".", ms=4, label=mdl)
        ax[i].set_ylabel(names[i])
    ax[0].legend(fontsize=8, ncol=6); ax[0].set_title("Imputed values at missing positions (p=0.3, seed 0)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "lorenz_reconstruction.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for mdl in ["A", "B", "C", "D"]:
        f = os.path.join(RES, "logs", f"lorenz_{mdl}_p0.3_s0.csv")
        if os.path.exists(f):
            lg = pd.read_csv(f)
            ax[0].plot(lg.epoch, lg.train_loss, label=mdl)
            ax[1].plot(lg.epoch, lg.val_mse, label=mdl)
    ax[0].set_title("Training L_mask"); ax[1].set_title("Validation MSE on held-out missing entries")
    for a_ in ax:
        a_.set_yscale("log"); a_.set_xlabel("epoch"); a_.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "lorenz_loss_curves.png"), dpi=130); plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    df = merge("lorenz_p", "lorenz_results.csv")
    if df is not None:
        s = summarize(df, ["dataset"], "lorenz_summary.csv")
        rmse_vs_rate(s, "dataset", "lorenz_rmse_vs_rate.png", "Lorenz")
        print(s.pivot(index="model", columns="missing_rate", values="rmse_mean").reindex(ORDER).round(4))
    df = merge("coupled_k", "coupled_lorenz_results.csv")
    if df is not None:
        s = summarize(df, ["dataset"], "coupled_lorenz_summary.csv")
        rmse_vs_rate(s, "dataset", "coupled_lorenz_rmse_vs_rate.png", "Coupled Lorenz")
    df = merge("real_", "real_data_results.csv")
    if df is not None:
        s = summarize(df, ["dataset"], "real_data_summary.csv")
        rmse_vs_rate(s, "dataset", "real_data_rmse_vs_rate.png", "Real data (artificial MCAR)")
    lorenz_series_figs()
