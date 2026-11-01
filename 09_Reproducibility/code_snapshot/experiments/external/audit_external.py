"""Phase 5: read-only integrity audit of Jena / Appliances / Beijing PRSA (CPU only).
Raw files are opened read-only; Beijing zip is read in place (not extracted).
Usage: python -I experiments/external/audit_external.py <repo_root> <out_dir>"""
import sys, os, io, zipfile, json, hashlib
import numpy as np, pandas as pd

root, out = sys.argv[1], sys.argv[2]
ext = os.path.join(root, "Data/external")
R = {}

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def gaps(t, step):
    d = t.diff().dropna()
    return {"n": int(len(t)), "start": str(t.iloc[0]), "end": str(t.iloc[-1]),
            "dt_mode": str(d.mode().iloc[0]), "n_dup_ts": int(t.duplicated().sum()),
            "n_nonmono": int((d <= pd.Timedelta(0)).sum()),
            "n_gaps": int((d > step).sum()), "max_gap": str(d.max()),
            "n_missing_slots": int(((d[d > step] / step) - 1).sum()),
            "expected_n_regular": int((t.iloc[-1] - t.iloc[0]) / step) + 1}

def colstats(df):
    s = {}
    for c in df.columns:
        x = df[c]
        s[c] = {"nan": int(x.isna().sum()), "min": float(x.min()), "max": float(x.max()),
                "mean": float(x.mean()), "std": float(x.std()), "n_unique": int(x.nunique()),
                "top_val_frac": float(x.value_counts(normalize=True).iloc[0]),
                "max_run_const": int(x.groupby((x != x.shift()).cumsum()).size().max())}
    return s

def corr_pairs(df, thr=0.95):
    c = df.corr().abs()
    return [(a, b, round(float(c.loc[a, b]), 4)) for i, a in enumerate(c) for b in list(c)[i + 1:] if c.loc[a, b] >= thr]

def lagged_r2(df, target, preds, lags=(0,), past=True):
    """Train-split ridge R^2 on later-split rows (chronological), using cur/past values of preds."""
    from sklearn.linear_model import RidgeCV
    X = pd.concat({f"{p}_l{l}": df[p].shift(l) for p in preds for l in lags}, axis=1)
    y = df[target]
    ok = X.notna().all(1) & y.notna()
    X, y = X[ok].values, y[ok].values
    n = len(y); a, b = int(.7 * n), int(.85 * n)
    mu, sd = X[:a].mean(0), X[:a].std(0) + 1e-12
    m = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit((X[:a] - mu) / sd, y[:a])
    yt = y[b:]; pr = m.predict((X[b:] - mu) / sd)
    return float(1 - ((yt - pr) ** 2).mean() / ((yt - y[:a].mean()) ** 2).mean())

def knn_r2(df, target, preds, lags, k=10, nmax=20000):
    """Nonlinear: kNN regression R^2 (train->test, chronological)."""
    from sklearn.neighbors import KNeighborsRegressor
    X = pd.concat({f"{p}_l{l}": df[p].shift(l) for p in preds for l in lags}, axis=1)
    y = df[target]; ok = X.notna().all(1) & y.notna()
    X, y = X[ok].values, y[ok].values
    n = len(y); a, b = int(.7 * n), int(.85 * n)
    mu, sd = X[:a].mean(0), X[:a].std(0) + 1e-12
    Xa, ya = (X[:a] - mu) / sd, y[:a]
    if len(ya) > nmax:
        idx = np.linspace(0, len(ya) - 1, nmax).astype(int); Xa, ya = Xa[idx], ya[idx]
    m = KNeighborsRegressor(k).fit(Xa, ya)
    yt = y[b:]; pr = m.predict((X[b:] - mu) / sd)
    return float(1 - ((yt - pr) ** 2).mean() / ((yt - y[:a].mean()) ** 2).mean())

# ---------------- Jena ----------------
p = os.path.join(ext, "jena/raw/jena_climate_2009_2016.csv")
J = pd.read_csv(p)
J["t"] = pd.to_datetime(J["Date Time"], format="%d.%m.%Y %H:%M:%S")
num = J.drop(columns=["Date Time", "t"])
r = {"file": p, "sha256": sha(p), "shape": list(J.shape), "time": gaps(J["t"], pd.Timedelta("10min")),
     "cols": colstats(num), "corr95": corr_pairs(num)}
r["sentinel_-9999"] = {c: int((num[c] <= -9998).sum()) for c in num}
r["neg_wind"] = {c: int((num[c] < 0).sum()) for c in ["wv (m/s)", "max. wv (m/s)"]}
# exact-duplicate rows on timestamp
dupt = J[J["t"].duplicated(keep=False)]
r["dup_ts_identical_rows"] = int(dupt.duplicated(subset=list(J.columns.drop("Date Time"))).sum())
# derived-variable identity checks (Magnus/ideal gas style): residuals
T, P, Td, rh = num["T (degC)"], num["p (mbar)"], num["Tdew (degC)"], num["rh (%)"]
VPmax, VPact, VPdef, sh, H2OC, rho, Tpot = (num[c] for c in ["VPmax (mbar)", "VPact (mbar)", "VPdef (mbar)", "sh (g/kg)", "H2OC (mmol/mol)", "rho (g/m**3)", "Tpot (K)"])
ok = (num > -9000).all(1)
def res(a, b): d = (a - b)[ok]; return {"mae": float(d.abs().mean()), "max": float(d.abs().max())}
r["identities"] = {
    "VPdef=VPmax-VPact": res(VPdef, VPmax - VPact),
    "VPact=rh*VPmax/100": res(VPact, rh * VPmax / 100),
    "Tpot=(T+273.15)*(1000/p)^0.286": res(Tpot, (T + 273.15) * (1000 / P) ** 0.286),
    "H2OC=1000*VPact/p": res(H2OC, 1000 * VPact / P),
    "sh~622*VPact/(p-0.378VPact)": res(sh, 622 * VPact / (P - 0.378 * VPact)),
    "rho~p*100/(287.05*(T+273.15))*1000": res(rho, P * 100 / (287.05 * (T + 273.15)) * 1000),
    "VPmax~6.1078*exp(17.27T/(T+237.3))": res(VPmax, 6.1078 * np.exp(17.27 * T / (T + 237.3))),
    "VPact~VPmax(Td)": res(VPact, 6.1078 * np.exp(17.27 * Td / (Td + 237.3))),
}
# hourly-resampled regular grid for complementarity checks
Jc = J.drop_duplicates("t").set_index("t")[num.columns].copy()
for c in ["wv (m/s)", "max. wv (m/s)"]:
    Jc.loc[Jc[c] < 0, c] = np.nan
Jh = Jc.resample("1h").mean()
r["hourly_rows"] = int(len(Jh)); r["hourly_allnan_rows"] = int(Jh.isna().all(1).sum())
Jh["u"] = Jh["wv (m/s)"] * np.cos(np.deg2rad(Jh["wd (deg)"])); Jh["v"] = Jh["wv (m/s)"] * np.sin(np.deg2rad(Jh["wd (deg)"]))
core = ["p (mbar)", "T (degC)", "rh (%)", "VPact (mbar)", "u", "v"]
r["hourly_core_corr"] = Jh[core].corr().round(3).to_dict()
# complementarity: predict each core var from (a) own past 24h, (b) own past + other vars' past
comp = {}
L = list(range(1, 25, 3))
for tg in ["T (degC)", "p (mbar)", "rh (%)", "VPact (mbar)", "u"]:
    others = [c for c in ["T (degC)", "p (mbar)", "VPact (mbar)", "u", "v"] if c != tg]
    comp[tg] = {"own_past_ridge": lagged_r2(Jh, tg, [tg], L),
                "own+others_past_ridge": lagged_r2(Jh, tg, [tg] + others, L),
                "others_lag0_ridge": lagged_r2(Jh, tg, others, [0]),
                "others_lag0_knn": knn_r2(Jh, tg, others, [0])}
r["complementarity_hourly_h24"] = comp
R["jena"] = r
print("jena done", flush=True)

# ---------------- Appliances ----------------
p = os.path.join(ext, "appliances/raw/energydata_complete.csv")
A = pd.read_csv(p); A["t"] = pd.to_datetime(A["date"])
num = A.drop(columns=["date", "t"]).astype(float)
r = {"file": p, "sha256": sha(p), "shape": list(A.shape), "time": gaps(A["t"], pd.Timedelta("10min")),
     "cols": colstats(num), "corr95": corr_pairs(num)}
r["rv1_eq_rv2"] = bool(np.allclose(num.rv1, num.rv2))
r["rv_corr_with_others_max"] = float(num.drop(columns=["rv1", "rv2"]).corrwith(num.rv1).abs().max())
# hourly-constant check for weather (Chievres station interpolated hourly -> 10 min)
w = ["T_out", "Press_mm_hg", "RH_out", "Windspeed", "Visibility", "Tdewpoint"]
Ai = A.set_index("t")
on_hour = Ai.index.minute == 0
lin = {}
for c in w:
    x = Ai[c]; xh = x[on_hour]
    interp = xh.reindex(x.index).interpolate("time")
    lin[c] = float((x - interp).abs()[~on_hour].max())
r["weather_max_dev_from_hourly_linear_interp"] = lin
r["frac_rows_lights_zero"] = float((num.lights == 0).mean())
r["appliances_quantiles"] = num.Appliances.quantile([0, .5, .9, .99, 1]).to_dict()
r["appliances_quantization"] = sorted(num.Appliances.unique()[:15].tolist())
comp = {}
Ah = Ai.drop(columns=["date"])
for tg in ["T2", "RH_2", "T6", "Appliances"]:
    others = [c for c in ["T1", "T3", "T_out", "RH_1", "RH_out", "T8"] if c != tg]
    comp[tg] = {"own_past_ridge": lagged_r2(Ah, tg, [tg], list(range(1, 25, 3))),
                "own+others_past_ridge": lagged_r2(Ah, tg, [tg] + others, list(range(1, 25, 3))),
                "others_lag0_ridge": lagged_r2(Ah, tg, others, [0]),
                "others_lag0_knn": knn_r2(Ah, tg, others, [0])}
r["complementarity_10min_h4"] = comp
R["appliances"] = r
print("appl done", flush=True)

# ---------------- Beijing ----------------
zp = os.path.join(ext, "beijing/raw/PRSA2017_Data_20130301-20170228.zip")
other = {f: sha(os.path.join(ext, "beijing/raw", f)) for f in sorted(os.listdir(os.path.join(ext, "beijing/raw")))}
r = {"zip": zp, "all_files_sha256": other, "stations": {}}
frames = {}
with zipfile.ZipFile(zp) as z:
    r["zip_testzip"] = z.testzip()
    for n in z.namelist():
        if not n.endswith(".csv"): continue
        df = pd.read_csv(io.BytesIO(z.read(n)))
        st = df.station.iloc[0]; frames[st] = df
B = pd.concat(frames.values(), ignore_index=True)
B["t"] = pd.to_datetime(B[["year", "month", "day", "hour"]])
r["columns"] = list(B.columns.drop("t")); r["n_rows"] = int(len(B))
numc = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
for st, df in frames.items():
    t = pd.to_datetime(df[["year", "month", "day", "hour"]])
    g = gaps(t, pd.Timedelta("1h"))
    g["nan_frac"] = {c: round(float(df[c].isna().mean()), 4) for c in numc + ["wd"]}
    g["row_any_nan"] = float(df[numc].isna().any(axis=1).mean())
    g["No_contiguous"] = bool((df["No"].diff().dropna() == 1).all())
    r["stations"][st] = g
r["overall_nan_frac"] = {c: round(float(B[c].isna().mean()), 4) for c in numc + ["wd"]}
r["value_ranges"] = {c: [float(B[c].min()), float(B[c].quantile(.5)), float(B[c].max())] for c in numc}
r["wd_levels"] = sorted(B.wd.dropna().unique().tolist())
# quantization / suspicious values
r["CO_unique_frac"] = float(B.CO.nunique() / B.CO.notna().sum())
r["PM10_lt_PM25_frac"] = float((B["PM10"] < B["PM2.5"]).mean())
r["O3_max_runs"] = {st: int(df.O3.groupby((df.O3 != df.O3.shift()).cumsum()).size().max()) for st, df in frames.items()}
r["max_const_run"] = {c: int(max(df[c].groupby((df[c] != df[c].shift()).cumsum()).size().max() for df in frames.values())) for c in numc}
# missing block lengths per variable (natural), pooled
def runs(m):
    m = m.astype(int); d = np.diff(np.r_[0, m, 0]); s = np.where(d == 1)[0]; e = np.where(d == -1)[0]; return e - s
blk = {}
for c in numc:
    L = np.concatenate([runs(df[c].isna().values) for df in frames.values()])
    blk[c] = {"n_blocks": int(len(L)), "median": float(np.median(L)) if len(L) else 0, "p95": float(np.percentile(L, 95)) if len(L) else 0, "max": int(L.max()) if len(L) else 0}
r["natural_missing_blocks"] = blk
# co-missingness (is missingness shared across vars at same station/time?)
M = B[numc].isna()
r["comiss_jaccard_selected"] = {f"{a}|{b}": round(float((M[a] & M[b]).sum() / max((M[a] | M[b]).sum(), 1)), 3)
                                for a, b in [("PM2.5", "PM10"), ("TEMP", "PRES"), ("TEMP", "DEWP"), ("SO2", "NO2"), ("CO", "O3"), ("PM2.5", "TEMP")]}
# weather identical across stations? (shared meteorological source)
W = {st: df.set_index(pd.to_datetime(df[["year", "month", "day", "hour"]]))[["TEMP", "PRES", "DEWP", "WSPM", "PM2.5"]] for st, df in frames.items()}
sts = sorted(W)
xs = {}
for c in ["TEMP", "PRES", "DEWP", "WSPM", "PM2.5"]:
    mat = pd.concat({s: W[s][c] for s in sts}, axis=1)
    cc = mat.corr().values[np.triu_indices(len(sts), 1)]
    eq = [float((mat[a] == mat[b]).mean()) for i, a in enumerate(sts) for b in sts[i + 1:]]
    xs[c] = {"cross_station_corr_min": float(cc.min()), "median": float(np.median(cc)), "max_frac_identical_pair": float(max(eq))}
r["cross_station"] = xs
# complementarity at one station (Aotizhongxin), hourly
S = W["Aotizhongxin"].copy()
fA = frames["Aotizhongxin"].set_index(pd.to_datetime(frames["Aotizhongxin"][["year", "month", "day", "hour"]]))
for c in ["NO2", "CO", "O3", "PM10", "SO2"]: S[c] = fA[c]
comp = {}
L = list(range(1, 25, 3))
for tg in ["PM2.5", "TEMP", "O3", "NO2"]:
    others = [c for c in ["PM2.5", "TEMP", "PRES", "DEWP", "WSPM", "NO2", "CO", "O3"] if c != tg]
    comp[tg] = {"own_past_ridge": lagged_r2(S, tg, [tg], L),
                "own+others_past_ridge": lagged_r2(S, tg, [tg] + others, L),
                "others_lag0_ridge": lagged_r2(S, tg, others, [0]),
                "others_lag0_knn": knn_r2(S, tg, others, [0])}
r["complementarity_aoti_h24"] = comp
# cross-station: PM2.5 at Aoti from other stations' PM2.5 lag0
Pm = pd.concat({s: W[s]["PM2.5"] for s in sts}, axis=1)
r["pm25_aoti_from_other_stations_lag0_ridge"] = lagged_r2(Pm, "Aotizhongxin", [s for s in sts if s != "Aotizhongxin"], [0])
r["pm25_aoti_own_past_ridge"] = lagged_r2(Pm, "Aotizhongxin", ["Aotizhongxin"], L)
R["beijing"] = r
print("beij done", flush=True)

with open(os.path.join(out, "audit.json"), "w") as f:
    json.dump(R, f, indent=1, default=str)
print("ok")
