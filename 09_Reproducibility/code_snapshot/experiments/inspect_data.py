"""Inspect every file in Data/ and write results/dataset_summary.csv."""
import os
import sys
import glob
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.data_loader import DATA_DIR, LOADERS  # noqa: E402

FILE_OF = {"eeg": "EEG data.xlsx", "enso": "ENSO data.xlsx", "exchange": "Exchange rates.xlsx",
           "transportation": "Transportation.xlsx", "wind": "Wind speed.csv"}


def sheet_info(path):
    if path.endswith(".csv"):
        df = pd.read_csv(path, header=None)
        return f"csv raw {df.shape}"
    sheets = pd.read_excel(path, sheet_name=None, header=None)
    return "; ".join(f"{k} raw {v.shape}" for k, v in sheets.items())


def main():
    rows = []
    for key, fname in FILE_OF.items():
        path = os.path.join(DATA_DIR, fname)
        X, names, meta = LOADERS[key]()
        finite = np.isfinite(X)
        nan = np.isnan(X).sum()
        inf = np.isinf(X).sum()
        Xf = np.where(finite, X, np.nan)
        mu, sd = np.nanmean(Xf, 0), np.nanstd(Xf, 0)
        z = np.abs((Xf - mu) / np.where(sd > 0, sd, 1))
        const_cols = [names[j] for j in range(X.shape[1]) if not (sd[j] > 0)]
        # identical-column pairs (possible copy errors)
        dup = []
        for a in range(X.shape[1]):
            for b in range(a + 1, X.shape[1]):
                if np.allclose(np.nan_to_num(X[:, a]), np.nan_to_num(X[:, b])):
                    dup.append(f"{names[a]}={names[b]}")
        eq_rows = [names[a] + "~" + names[b] for a in range(X.shape[1]) for b in range(a + 1, X.shape[1])
                   if np.mean(np.isclose(X[:, a], X[:, b])) > 0.05]
        lag1 = np.nanmean([pd.Series(X[:, j]).autocorr(1) for j in range(X.shape[1]) if sd[j] > 0])
        ti = meta.get("time_index")
        interval = "1 month" if ti is not None and np.all(np.diff(ti) == 1) else (
            "irregular" if ti is not None else "unknown (assumed uniform)")
        rows.append({
            "dataset": key, "file": fname, "sheets": sheet_info(path),
            "T": X.shape[0], "n_vars": X.shape[1], "n_numeric_vars": int(np.sum(np.isfinite(X).any(0))),
            "dtype": str(X.dtype), "time_axis": meta["time_axis"], "time_interval": interval,
            "dropped_non_variables": meta["dropped"],
            "period": f"{meta.get('start','')}..{meta.get('end','')}" if "start" in meta else "",
            "n_missing": int(nan), "missing_ratio": nan / X.size, "n_inf": int(inf),
            "n_|z|>5": int(np.nansum(z > 5)), "max_|z|": float(np.nanmax(z)),
            "constant_cols": ";".join(const_cols), "identical_cols": ";".join(dup),
            "cols_equal_>5%_of_rows": ";".join(eq_rows[:10]) + ("..." if len(eq_rows) > 10 else ""),
            "mean_lag1_autocorr": lag1,
            "min": float(np.nanmin(Xf)), "max": float(np.nanmax(Xf)),
            "first_cols": ",".join(names[:6]),
            "usable_for_experiment": "yes" if (X.shape[0] >= 200 and not inf) else "check",
        })
        print(key, X.shape, "nan", nan, "inf", inf, "lag1", round(lag1, 3), "dup", dup[:5], "eq", eq_rows[:5])
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    out = os.path.join(ROOT, "results", "dataset_summary.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print("saved", out)


if __name__ == "__main__":
    main()
