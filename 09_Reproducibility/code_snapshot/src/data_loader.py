"""Loaders for the real datasets in Data/.

Each loader returns (X, names, meta) where X is a float array of shape (T, n)
with time along axis 0. Structure was determined by inspecting the files
(see experiments/inspect_data.py); identifiers/time columns are removed.
"""
import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Data")


def _p(name):
    return os.path.join(DATA_DIR, name)


def load_eeg():
    # Row 0 holds channel indices 1..64; rows 1.. are samples.
    raw = pd.read_excel(_p("EEG data.xlsx"), header=None)
    names = [f"ch{int(c)}" for c in raw.iloc[0].values]
    X = raw.iloc[1:].to_numpy(dtype=float)
    return X, names, {"time_axis": "rows (sample index)", "dropped": "row 0 = channel ids"}


def load_enso():
    # 10 rows of source notes, header at row 10: YEAR, Month, NINO4, NINO34, NINO3, NINO1+2
    raw = pd.read_excel(_p("ENSO data.xlsx"), sheet_name="Sheet1", header=None)
    body = raw.iloc[11:].reset_index(drop=True)
    body.columns = ["YEAR", "Month", "NINO4", "NINO34", "NINO3", "NINO1+2"]
    body = body.apply(pd.to_numeric, errors="coerce").dropna(subset=["YEAR", "Month"])
    names = ["NINO4", "NINO34", "NINO3", "NINO1+2"]
    X = body[names].to_numpy(dtype=float)
    return X, names, {"time_axis": "monthly (YEAR, Month)", "dropped": "YEAR, Month, note rows",
                      "start": f"{int(body.YEAR.iloc[0])}-{int(body.Month.iloc[0])}",
                      "end": f"{int(body.YEAR.iloc[-1])}-{int(body.Month.iloc[-1])}",
                      "time_index": (body.YEAR * 12 + body.Month).to_numpy()}


def load_exchange():
    # Sheet 'original data': row 0 = column index, row 1 = currency names, rows 2.. = values.
    raw = pd.read_excel(_p("Exchange rates.xlsx"), sheet_name="original data", header=None)
    names = [str(c) for c in raw.iloc[1].values]
    X = raw.iloc[2:].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return X, names, {"time_axis": "rows (daily, no date column)", "dropped": "index row, name row"}


def load_transportation():
    raw = pd.read_excel(_p("Transportation.xlsx"), header=None)
    X = raw.to_numpy(dtype=float)
    return X, [f"s{j}" for j in range(X.shape[1])], {"time_axis": "rows (no time column)", "dropped": "none"}


def load_wind():
    # Header row holds column ids 1..231. Columns are each z-scored over rows,
    # so columns are treated as variables and rows as time (see inspect_data).
    raw = pd.read_csv(_p("Wind speed.csv"))
    X = raw.to_numpy(dtype=float)
    return X, [f"w{c}" for c in raw.columns], {"time_axis": "rows (assumed; columns z-scored)", "dropped": "header row"}


LOADERS = {
    "eeg": load_eeg,
    "enso": load_enso,
    "exchange": load_exchange,
    "transportation": load_transportation,
    "wind": load_wind,
}


def load_real(name):
    return LOADERS[name]()
