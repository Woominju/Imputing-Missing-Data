"""Phase 5 supplement (read-only, CPU): gap-aware complementarity + targeted integrity checks.
Own-history features start at lag g (mimics the own variable being missing for g-1 preceding steps);
'others' use lags 0..L (cross-view observed now). Ridge, chronological 70/15/15, train-only scaling.
Usage: python -I experiments/external/audit_external_gap.py <repo_root> <out_dir>"""
import sys, os, io, zipfile, json
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV

root, out = sys.argv[1], sys.argv[2]
ext = os.path.join(root, "Data/external")

def r2(df, target, feats):
    X = pd.concat({f"{p}_l{l}": df[p].shift(l) for p, ls in feats for l in ls}, axis=1)
    y = df[target]; ok = X.notna().all(1) & y.notna()
    X, y = X[ok].values, y[ok].values
    n = len(y); a, b = int(.7 * n), int(.85 * n)
    mu, sd = X[:a].mean(0), X[:a].std(0) + 1e-12
    m = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit((X[:a] - mu) / sd, y[:a])
    yt = y[b:]; pr = m.predict((X[b:] - mu) / sd)
    return round(float(1 - ((yt - pr) ** 2).mean() / ((yt - y[:a].mean()) ** 2).mean()), 4)

def table(df, targets, pool, gaps, W):
    res = {}
    for tg in targets:
        oth = [c for c in pool if c != tg]
        res[tg] = {}
        for g in gaps:
            own = [(tg, list(range(g, g + W)))]
            res[tg][f"g{g}"] = {"own": r2(df, tg, own),
                                "own+oth_now": r2(df, tg, own + [(c, list(range(0, W))) for c in oth]),
                                "oth_only": r2(df, tg, [(c, list(range(0, W))) for c in oth])}
    return res

R = {}
# Jena hourly
J = pd.read_csv(os.path.join(ext, "jena/raw/jena_climate_2009_2016.csv"))
J["t"] = pd.to_datetime(J["Date Time"], format="%d.%m.%Y %H:%M:%S")
d = J["t"].diff()
R["jena_nonmono_at"] = [str(x) for x in J.loc[d <= pd.Timedelta(0), "t"]]
R["jena_gaps"] = [(str(J["t"].iloc[i - 1]), str(d.iloc[i])) for i in np.where(d > pd.Timedelta("10min"))[0]]
R["jena_sentinel_at"] = sorted(set(str(x.date()) for x in J.loc[J["wv (m/s)"] < 0, "t"]))
Jc = J.drop_duplicates("t").set_index("t").drop(columns="Date Time").sort_index()
Jc.loc[Jc["wv (m/s)"] < 0, "wv (m/s)"] = np.nan
Jc.loc[Jc["max. wv (m/s)"] < 0, "max. wv (m/s)"] = np.nan
wv = Jc["wv (m/s)"]; run = wv.groupby((wv != wv.shift()).cumsum()).transform("size")
R["jena_wv_long_const_runs(>36)"] = sorted(set(str(x.date()) for x in wv[run > 36].index))[:20]
Jc["u"] = Jc["wv (m/s)"] * np.cos(np.deg2rad(Jc["wd (deg)"])); Jc["v"] = Jc["wv (m/s)"] * np.sin(np.deg2rad(Jc["wd (deg)"]))  # vector components before averaging
Jh = Jc.resample("1h").mean()
pool = ["T (degC)", "p (mbar)", "VPact (mbar)", "u", "v"]
R["jena_hourly"] = table(Jh, ["T (degC)", "p (mbar)", "VPact (mbar)", "u"], pool, [1, 3, 6, 12], 6)
print("jena", flush=True)
# Appliances 10min
A = pd.read_csv(os.path.join(ext, "appliances/raw/energydata_complete.csv")); A.index = pd.to_datetime(A.pop("date"))
pool = ["T2", "RH_2", "T3", "T5", "T_out", "RH_out", "Appliances"]
R["appl_10min"] = table(A, ["T2", "RH_2", "T5", "Appliances"], pool, [1, 3, 6, 12], 6)
print("appl", flush=True)
# Beijing
fr = {}
with zipfile.ZipFile(os.path.join(ext, "beijing/raw/PRSA2017_Data_20130301-20170228.zip")) as z:
    for n in z.namelist():
        if n.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(z.read(n))); df.index = pd.to_datetime(df[["year", "month", "day", "hour"]]); fr[df.station.iloc[0]] = df
sts = sorted(fr)
ident = {}
for c in ["TEMP", "PRES", "DEWP", "WSPM", "RAIN"]:
    pairs = []
    for i, a in enumerate(sts):
        for b in sts[i + 1:]:
            m = fr[a][c].notna() & fr[b][c].notna()
            if (fr[a][c][m] == fr[b][c][m]).mean() > 0.9: pairs.append(f"{a}={b}")
    ident[c] = pairs
R["beijing_identical_station_pairs(>90%)"] = ident
S = fr["Aotizhongxin"]
pool = ["PM2.5", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "WSPM"]
R["beijing_aoti_hourly"] = table(S, ["PM2.5", "NO2", "O3", "TEMP"], pool, [1, 3, 6, 12], 6)
Pm = pd.concat({s: fr[s]["PM2.5"] for s in sts}, axis=1)
R["beijing_pm25_station_views"] = {f"g{g}": {"own": r2(Pm, "Aotizhongxin", [("Aotizhongxin", list(range(g, g + 6)))]),
    "own+Dongsi,Guanyuan,Wanliu_now": r2(Pm, "Aotizhongxin", [("Aotizhongxin", list(range(g, g + 6)))] + [(s, list(range(6))) for s in ["Dongsi", "Guanyuan", "Wanliu"]]),
    "Dingling,Huairou_now(remote)": r2(Pm, "Aotizhongxin", [(s, list(range(6))) for s in ["Dingling", "Huairou"]])} for g in [1, 3, 6, 12]}
# natural-missing co-occurrence within station: fraction of PM2.5-missing rows where all pollutants missing
pol = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
B = pd.concat(fr.values())
R["beijing_frac_rows_all_pollutants_missing"] = round(float(B[pol].isna().all(1).mean()), 4)
R["beijing_frac_rows_all_met_missing"] = round(float(B[["TEMP", "PRES", "DEWP", "WSPM"]].isna().all(1).mean()), 4)
R["beijing_PM_cap_999_count"] = {c: int((B[c] >= 999).sum()) for c in ["PM2.5", "PM10"]}
R["beijing_O3_gt500"] = int((B["O3"] > 500).sum())
R["beijing_CO_step"] = sorted(B["CO"].dropna().unique()[:10].tolist())
# split boundaries (70/15/15 of 35064 hours)
n = 35064; t = S.index
R["beijing_split_dates"] = [str(t[int(.7 * n)]), str(t[int(.85 * n)])]
n = len(Jh); R["jena_hourly_split_dates"] = [str(Jh.index[int(.7 * n)]), str(Jh.index[int(.85 * n)])]
n = len(A); R["appl_split_dates"] = [str(A.index[int(.7 * n)]), str(A.index[int(.85 * n)])]
json.dump(R, open(os.path.join(out, "audit_gap.json"), "w"), indent=1)
print(json.dumps(R, indent=1))
