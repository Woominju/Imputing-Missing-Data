"""Phase 5 Gate F: reference-free imputation on real data (design: results/notes/phase5_external_dataset_audit.md).
prep   : python external.py prep            -> Data/external/<ds>/processed/<ds>.npz  (raw files read-only; Beijing zip read in place)
train  : python external.py train DS P SEED MODEL     MODEL in {MV, MVshift, SV<k>}
Models are the Phase 5 ArmModel (FDR-Learned): MV = all views in/out; SV<k> = view k only (single-view, same encoder type);
MVshift = MV on data where view k is circularly shifted by 14*k days INSIDE each chronological split (daily phase kept, synoptic
alignment destroyed; every entry keeps its own value, mask and own history). Split 70/15/15 chronological; normalisation from
observed train entries; MCAR(p) injected on every selected view; evaluation entries = test rows with M_orig = 1 and M_inj = 0,
excluding the first span rows of the split and (for every model) the entries whose shifted window crosses the wrap point.
Early stopping: lag-0 RMSE on a fixed 20% of OBSERVED val entries hidden from the input (no ground truth of missing values).
Outputs results/extra/phase5/external/<ds>_p<P>_s<S>_<MODEL>.npz with weekly-block SSE/N per view x own-gap stratum."""
import io, os, sys, time, zipfile, json
import numpy as np, pandas as pd, torch
from experiments.phase5.core import ArmModel, encode_series, seed_of
from experiments.fdr.loss import fdr_masked_loss
from src.delay_embedding import delay_embed

ROOT_E = "Data/external"; OUT = "results/extra/phase5/external"; M_, SPAN = 8, 7
VIEWS = {"beijing": ["PM2.5", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "WSPM"], "jena": ["p", "T", "rh", "u", "v"]}
GSTRATA = [(1, 1), (2, 3), (4, 7), (8, 10 ** 9)]


def prep():
    zp = f"{ROOT_E}/beijing/raw/PRSA2017_Data_20130301-20170228.zip"
    with zipfile.ZipFile(zp) as z:
        n = [q for q in z.namelist() if "Aotizhongxin" in q][0]; df = pd.read_csv(io.BytesIO(z.read(n)))
    t = pd.to_datetime(df[["year", "month", "day", "hour"]]); X = df[VIEWS["beijing"]].values.astype(np.float64)
    os.makedirs(f"{ROOT_E}/beijing/processed", exist_ok=True)
    np.savez_compressed(f"{ROOT_E}/beijing/processed/beijing.npz", X=X, t=t.astype(str).values, views=VIEWS["beijing"], source=zp + "::" + n)
    J = pd.read_csv(f"{ROOT_E}/jena/raw/jena_climate_2009_2016.csv")
    J["t"] = pd.to_datetime(J["Date Time"], format="%d.%m.%Y %H:%M:%S"); J = J.drop_duplicates("t").set_index("t").sort_index()
    bad = (J["wv (m/s)"] < 0) | (J["max. wv (m/s)"] < 0)
    J.loc[bad, ["wv (m/s)", "wd (deg)"]] = np.nan
    J["u"] = J["wv (m/s)"] * np.cos(np.deg2rad(J["wd (deg)"])); J["v"] = J["wv (m/s)"] * np.sin(np.deg2rad(J["wd (deg)"]))
    H = J[["p (mbar)", "T (degC)", "rh (%)", "u", "v"]].resample("1h").mean()
    H = H.loc["2009-01-01 01:00":"2016-12-31 23:00"]
    os.makedirs(f"{ROOT_E}/jena/processed", exist_ok=True)
    np.savez_compressed(f"{ROOT_E}/jena/processed/jena.npz", X=H.values, t=H.index.astype(str).values, views=VIEWS["jena"], source="jena_climate_2009_2016.csv hourly mean")
    print("prep done", flush=True)


def split(T):
    a, b = int(round(.7 * T)), int(round(.85 * T)); return [(0, a), (a, b), (b, T)]


def shift_index(T, V):
    """perm[t, k] = original time index shown at position t for view k (circular shift by 14*k days within each split)."""
    P = np.zeros((T, V), int)
    for (s, e) in split(T):
        L = e - s
        for k in range(V):
            o = (24 * 14 * k) % L; P[s:e, k] = s + (np.arange(L) - o) % L
    return P


def setup(ds, p, seed):
    Z = np.load(f"{ROOT_E}/{ds}/processed/{ds}.npz", allow_pickle=True); X = Z["X"]; T, V = X.shape
    Mo = np.isfinite(X).astype(np.float32); Mi = (np.random.default_rng(seed_of("ext", ds, p, seed)).random((T, V)) >= p).astype(np.float32)
    M = Mo * Mi; sp = split(T); a, b = sp[0]
    mu = np.array([X[a:b][M[a:b, k] > 0, k].mean() for k in range(V)]); sd = np.array([X[a:b][M[a:b, k] > 0, k].std() for k in range(V)])
    Xn = np.where(Mo > 0, (np.nan_to_num(X) - mu) / sd, 0.0).astype(np.float32)
    Hv = np.zeros_like(M); vs, ve = sp[1]
    Hv[vs + SPAN:ve] = M[vs + SPAN:ve] * (np.random.default_rng(seed_of("extval", ds, p, seed)).random((ve - vs - SPAN, V)) < 0.2)
    P = shift_index(T, V)
    # entries excluded for all models: first SPAN rows of each split; entries whose shifted window crosses the wrap
    excl = np.zeros((T, V), bool)
    for (s, e) in sp:
        excl[s:s + SPAN] = True
        for k in range(V):
            wrap = s + ((24 * 14 * k) % (e - s))
            excl[P[wrap:min(wrap + SPAN, e), k], k] = True
    return dict(X=Xn, Mo=Mo, M=M, Hv=Hv, sp=sp, mu=mu, sd=sd, P=P, excl=excl, T=T, V=V)


def own_gap(M):
    """g[t, k] = smallest l >= 1 with M[t - l, k] = 1 (large if none)."""
    T, V = M.shape; g = np.full((T, V), 10 ** 9); last = np.full(V, -10 ** 9)
    for t in range(T):
        g[t] = t - last; last = np.where(M[t] > 0, t, last)
    return g


def train(ds, p, seed, model):
    S = setup(ds, p, seed); T, V = S["T"], S["V"]; dev = torch.device("cuda")
    cols = [int(model[2:])] if model.startswith("SV") else list(range(V))
    perm = S["P"] if model == "MVshift" else np.tile(np.arange(T)[:, None], (1, V))
    take = lambda A: np.stack([A[perm[:, k], k] for k in range(V)], 1)[:, cols]
    X, M, Hv = take(S["X"]), take(S["M"]), take(S["Hv"])
    torch.manual_seed(seed); np.random.seed(seed)
    mdl = ArmModel(len(cols), list(range(len(cols))), np.ones((len(cols), M_), np.float32)).to(dev)
    opt = torch.optim.Adam(mdl.parameters(), 1e-3); gen = torch.Generator(device=dev).manual_seed(seed + 1000)
    Xt, Mt = torch.tensor(X, device=dev), torch.tensor(M, device=dev); Bv = torch.tensor(M - Hv, device=dev)
    a, b = S["sp"][0]; W = SPAN + 64; best, state, bad, log = np.inf, None, 0, []; t0 = time.time()
    for ep in range(300):
        mdl.train()
        for _ in range(100):
            s = torch.randint(a, b - W + 1, (64,), generator=gen, device=dev); idx = s[:, None] + torch.arange(W, device=dev)[None]
            Xs, Ms = Xt[idx], Mt[idx]; H = Ms * (torch.rand(Ms.shape, generator=gen, device=dev) < 0.2).float(); Bm = Ms - H
            V_, MV = delay_embed(Xs * Bm, Bm, M_, 1); loss, _ = fdr_masked_loss(mdl.forward_full(V_, MV), Xs, H, M_, 1)
            if loss is None or not torch.isfinite(loss):
                continue
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(mdl.parameters(), 5.0); opt.step()
        _, y0 = encode_series(mdl, Xt, Bv); y0 = y0.cpu().numpy(); vs, ve = S["sp"][1]
        vr = float(np.sqrt((((y0 - X) ** 2) * Hv)[vs:ve].sum() / Hv[vs:ve].sum())); log.append(vr)
        if vr < best - 1e-6:
            best, bad, state = vr, 0, {k: q.clone() for k, q in mdl.state_dict().items()}
        else:
            bad += 1
            if bad >= 15:
                break
    mdl.load_state_dict(state); _, y0 = encode_series(mdl, Xt, Mt); y0 = y0.cpu().numpy()
    # map predictions back to original (time, view) coordinates
    pred = np.full((T, V), np.nan)
    for q, k in enumerate(cols):
        pred[perm[:, k], k] = y0[:, q]
    return pred, dict(best_val=best, epochs=len(log), wall=time.time() - t0, n_params=sum(q.numel() for q in mdl.parameters()))


def baselines(ds, p, seed):
    S = setup(ds, p, seed); X, M = S["X"], S["M"]; T, V = X.shape; out = {}
    out["mean"] = np.zeros_like(X); loc = np.zeros_like(X); lin = np.zeros_like(X); t = np.arange(T)
    for k in range(V):
        o = M[:, k] > 0; idx = np.maximum.accumulate(np.where(o, t, -1)); loc[:, k] = np.where(idx >= 0, X[np.maximum(idx, 0), k], 0)
        lin[:, k] = np.interp(t, t[o], X[o, k])
    out["LOCF"] = loc; out["linear_FUTURE"] = lin
    return out


def score(ds, p, seed, pred):
    S = setup(ds, p, seed); X, Mo, M = S["X"], S["Mo"], S["M"]; s, e = S["sp"][2]; T, V = X.shape
    ev = (Mo > 0) & (M == 0) & ~S["excl"]; ev[:s] = False
    g = own_gap(M); nb = (e - s + 167) // 168
    SSE = np.zeros((nb, V, len(GSTRATA))); N = np.zeros_like(SSE)
    for gi, (lo, hi) in enumerate(GSTRATA):
        sel = ev & (g >= lo) & (g <= hi)
        for k in range(V):
            ts = np.where(sel[:, k])[0]; bk = (ts - s) // 168; e2 = (pred[ts, k] - X[ts, k]) ** 2
            np.add.at(SSE[:, k, gi], bk, e2); np.add.at(N[:, k, gi], bk, 1)
    return SSE, N


if __name__ == "__main__":
    if sys.argv[1] == "prep":
        prep(); sys.exit(0)
    os.makedirs(OUT, exist_ok=True)
    ds, p, seed, model = sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    if model == "baselines":
        for nm, pr in baselines(ds, p, seed).items():
            SSE, N = score(ds, p, seed, pr); np.savez(f"{OUT}/{ds}_p{p}_s{seed}_{nm}.npz", SSE=SSE, N=N)
        sys.exit(0)
    pred, info = train(ds, p, seed, model); SSE, N = score(ds, p, seed, pred)
    np.savez(f"{OUT}/{ds}_p{p}_s{seed}_{model}.npz", SSE=SSE, N=N, info=json.dumps(info))
    print("RESULT", ds, p, seed, model, info, flush=True)
