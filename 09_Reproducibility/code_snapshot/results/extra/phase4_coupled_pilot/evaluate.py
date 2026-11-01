"""Phase 4 coupled-Lorenz evaluation (CPU). Autoencoder was trained without access to y1, y2 (or any hidden coordinate); hidden
coordinates are used here only for supervised probes and analysis. Probes: fit on train rows, hyper-parameters on val rows,
test rows predicted once. Imputation = FINAL series (observed values kept, missing = lag-0 past-only output); the raw decoder
output at observed positions is reported separately (it is never trained). FDR bidir lag-mean uses future data -> separate.
Same-state consistency: two fresh MCAR masks at the data rate p (Generator(4242+seed)), lag-0 output on the test split,
both-missing coordinates, stratified by the identifiability level of the coordinate's subsystem
L_j = min over the two masks of the number of observed x_j entries in the window (0 -> Gamma-unresolved, no point
consistency expected; 1; >= 2). Outputs eval_*.csv and per-time arrays in preds_eval/."""
import glob, json, os, sys
import numpy as np, pandas as pd, torch
from multiprocessing import Pool
from experiments.coupled import data as CD
from experiments.fdr import probes
from experiments.aim_autonomous.model_f import ModelF
from experiments.fdr.model import FDRModel
from experiments.fdr.lagweight import FDRLearned
from src.delay_embedding import delay_embed, pad_front
from src.models import build_model

O = "results/extra/phase4_coupled_pilot"; m, tau, span = 8, 1, 7
CTOR = {"C-Scalar": lambda: build_model("C", 4, m, 16, 64), "C-Learned": lambda: ModelF(4, m, 16, 64),
        "FDR-Scalar": lambda: FDRModel(4, m, 16, 64), "FDR-Learned": lambda: FDRLearned(4, m, 16, 64)}


def std3(Ftr, *Fs):
    mu, sd = Ftr.mean(0), Ftr.std(0); sd = np.where(sd > 1e-8, sd, 1.0); return [(F - mu) / sd for F in (Ftr,) + Fs]


def knn_multi(Ftr, Ytr, Fva, Yva, Fte, ks=(1, 3, 5, 10, 20, 50)):
    Ftr, Fva, Fte = std3(Ftr, Fva, Fte); Tt = torch.tensor(Ftr, dtype=torch.float32); Yt = torch.tensor(Ytr, dtype=torch.float32)
    def idx(F, kmax):
        return torch.cat([torch.cdist(torch.tensor(F[s:s + 2048], dtype=torch.float32), Tt).topk(kmax, largest=False).indices for s in range(0, len(F), 2048)])
    Iv = idx(Fva, max(ks)); best = min((float(np.sqrt(((Yt[Iv[:, :k]].mean(1).numpy() - Yva) ** 2).mean(0)).mean()), k) for k in ks)
    return Yt[idx(Fte, best[1])].mean(1).numpy(), best[1]


def ridge_multi(Ftr, Ytr, Fva, Yva, Fte):
    return np.stack([probes.ridge(Ftr, Ytr[:, j], Fva, Yva[:, j], Fte)[0] for j in range(Ytr.shape[1])], 1)


@torch.no_grad()
def outputs(mdl, X, M):
    Xp, Mp = pad_front(X * M, M, m, tau); V, MV = delay_embed(Xp[None], Mp[None], m, tau)
    z = mdl.encoder(V, MV); o = mdl.decode_full(z) if hasattr(mdl, "decode_full") else mdl.decode(z)[..., None]
    return o[0].numpy().astype(np.float64), z[0].numpy(), MV[0].numpy()


def raw_win(X, M, cols):
    Xp, Mp = pad_front(torch.tensor(X * M), torch.tensor(M), m, tau); V, MV = delay_embed(Xp[None], Mp[None], m, tau)
    return torch.cat([V[0][:, cols].flatten(1), MV[0][:, cols].flatten(1)], 1).numpy()


def rows_of(d, s): a, b = d["split"][s]; return np.arange(max(a, span), b)


def rmse(e2): return float(np.sqrt(np.mean(e2)))


def evaluate(path):
    torch.set_num_threads(3)
    ck = torch.load(path, map_location="cpu", weights_only=False); c = ck["cfg"]; eid = os.path.basename(path)[:-3]
    row = pd.read_csv(f"{O}/rows/{eid}.csv").iloc[0]; mo, k, tj, p, s = row.model_name, row.kappa, int(row.traj), row.p, int(row.seed)
    mdl = CTOR[mo](); mdl.load_state_dict(ck["state_dict"]); mdl.eval()
    d, hid = CD.build(CD.load_raw(k, tj), p, s); Sn = hid["state_norm"]
    X, M = torch.tensor(d["X"]), torch.tensor(d["M"]); Xn, Mn, Ev = d["X"].astype(np.float64), d["M"], d["eval_mask"]
    Y, Z, MV = outputs(mdl, X, M); xh = Y[..., 0]; a, b = d["split"]["test"]
    R = dict(exp_id=eid, model=mo, decoder=mo.split("-")[0], fusion=mo.split("-")[1], kappa=k, traj=tj, p=p, seed=s,
             best_epoch=row.best_epoch, epochs_run=row.epochs_run, n_params=row.n_params, recorded_test_rmse=row.test_rmse)
    A = {}
    sq = (xh[a:b] - Xn[a:b]) ** 2; E = Ev[a:b]; Mo = Mn[a:b]
    R["imp_rmse_all"] = float(np.sqrt((sq * E).sum() / E.sum()))
    for j, v in enumerate(d["views"]):
        R[f"imp_rmse_{v}"] = float(np.sqrt((sq[:, j] * E[:, j]).sum() / E[:, j].sum()))
    for sub, cols in (("sub1", [0, 1]), ("sub2", [2, 3])):
        R[f"imp_rmse_{sub}"] = float(np.sqrt((sq[:, cols] * E[:, cols]).sum() / E[:, cols].sum()))
    R["raw_decoder_rmse_at_observed"] = float(np.sqrt((sq * Mo).sum() / Mo.sum()))
    A["imp_num"] = (sq * E).sum(1); A["imp_den"] = E.sum(1)
    if Y.shape[-1] > 1:
        u = np.arange(a, b - (m - 1)); est = np.stack([Y[u + q, :, q] for q in range(m)], -1).mean(-1); mi = 1 - Mn[u]
        R["imp_rmse_bidir_FUTURE"] = float(np.sqrt((((est - Xn[u]) ** 2) * mi).sum() / mi.sum()))
    # ---- same-state two masks ----
    Xs = X[a:b]; g = torch.Generator().manual_seed(4242 + s)
    M1 = (torch.rand(Xs.shape, generator=g) >= p).float(); M2 = (torch.rand(Xs.shape, generator=g) >= p).float(); oo, nx = [], []
    for Mq in (M1, M2):
        V, MVq = delay_embed((Xs * Mq)[None], Mq[None], m, tau)
        with torch.no_grad():
            oo.append(mdl.decode(mdl.encoder(V, MVq)[0]).numpy().astype(np.float64))
        nx.append(MVq[0][:, [0, 2], :].sum(-1).numpy())                     # observed x1, x2 entries in window
    S = Xs[span:].numpy(); m1, m2 = M1[span:].numpy(), M2[span:].numpy(); mm = (m1 == 0) & (m2 == 0)
    e = ((oo[0] - S) ** 2 + (oo[1] - S) ** 2) / 2; dd = (oo[0] - oo[1]) ** 2; L = np.minimum(nx[0], nx[1])   # (N, 2) per subsystem
    for sub, cols, li in (("sub1", [0, 1], 0), ("sub2", [2, 3], 1)):
        for lev, sel in (("all", np.ones(len(L), bool)), ("L0", L[:, li] == 0), ("L1", L[:, li] == 1), ("L>=2", L[:, li] >= 2)):
            q = mm[:, cols] & sel[:, None]
            R[f"ss_bothmiss_err_{sub}_{lev}"] = float(np.sqrt(e[:, cols][q].mean())) if q.any() else np.nan
            R[f"ss_bothmiss_diff_{sub}_{lev}"] = float(np.sqrt(dd[:, cols][q].mean())) if q.any() else np.nan
            R[f"ss_bothmiss_n_{sub}_{lev}"] = int(q.sum())
    R["ss_bothmiss_err_all"] = float(np.sqrt(e[mm].mean())); R["ss_bothmiss_diff_all"] = float(np.sqrt(dd[mm].mean()))
    A["ss_num"] = (e * mm).sum(1); A["ss_den"] = mm.sum(1); A["ssd_num"] = (dd * mm).sum(1)
    # ---- hidden-state probes ----
    tr, va, te = rows_of(d, "train"), rows_of(d, "val"), rows_of(d, "test")
    targets = {"y1": Sn[:, [1]], "y2": Sn[:, [4]], "s1": Sn[:, :3], "s2": Sn[:, 3:], "s6": Sn}
    def probe_set(F, tag, mlp=True):
        for tn, Tg in targets.items():
            pk, kk = knn_multi(F[tr], Tg[tr], F[va], Tg[va], F[te]); R[f"{tag}_knn_{tn}"] = rmse((pk - Tg[te]) ** 2)
            pr = ridge_multi(F[tr], Tg[tr], F[va], Tg[va], F[te]); R[f"{tag}_ridge_{tn}"] = rmse((pr - Tg[te]) ** 2)
            if tag == "lat" and tn in ("y1", "y2"):
                A[f"knn_{tn}_err2"] = ((pk - Tg[te]) ** 2)[:, 0]
        if mlp:
            for tn in ("y1", "y2"):
                r_ = [rmse((probes.mlp(F[tr], targets[tn][tr, 0], F[va], targets[tn][va, 0], F[te], seed=ps)[0] - targets[tn][te, 0]) ** 2) for ps in (0, 1)]
                R[f"{tag}_mlp_{tn}"] = float(np.mean(r_)); R[f"{tag}_mlp_{tn}_probe_sd"] = float(abs(r_[0] - r_[1]) / np.sqrt(2))
    probe_set(Z, "lat")
    # view removal at inference (OOD for models trained on all four views); refit kNN/ridge probes on the same pattern
    for vn, keep in (("sub1only", [0, 1]), ("sub2only", [2, 3])):
        Mr = d["M"].copy(); Mr[:, [q for q in range(4) if q not in keep]] = 0
        _, Zr, _ = outputs(mdl, X, torch.tensor(Mr))
        for tn in ("y1", "y2"):
            pk, _ = knn_multi(Zr[tr], targets[tn][tr], Zr[va], targets[tn][va], Zr[te]); R[f"lat_{vn}_knn_{tn}"] = rmse((pk - targets[tn][te]) ** 2)
    # dynamics: latent 10-NN overlap with the true 6-D state; latent -> y_j(t+1) with split-safe rows (t+1 < split end)
    rng = np.random.default_rng(s); sub = te[rng.choice(len(te), 2000, replace=False)]
    Dz = torch.cdist(torch.tensor(Z[sub]), torch.tensor(Z[sub])); Ds = torch.cdist(torch.tensor(Sn[sub]), torch.tensor(Sn[sub]))
    Dz.fill_diagonal_(np.inf); Ds.fill_diagonal_(np.inf); iz, is_ = Dz.topk(10, largest=False).indices, Ds.topk(10, largest=False).indices
    R["knn10_overlap_latent_vs_state6"] = float(np.mean([len(set(iz[i].tolist()) & set(is_[i].tolist())) / 10 for i in range(len(sub))]))
    for tn, col in (("y1", 1), ("y2", 4)):
        rr = {sp_: v_[v_ + 1 < d["split"][sp_][1]] for sp_, v_ in (("train", tr), ("val", va), ("test", te))}
        pk, _ = knn_multi(Z[rr["train"]], Sn[rr["train"] + 1][:, [col]], Z[rr["val"]], Sn[rr["val"] + 1][:, [col]], Z[rr["test"]])
        R[f"lat_knn_{tn}_t+1"] = rmse((pk - Sn[rr["test"] + 1][:, [col]]) ** 2)
    return R, A, (k, tj, p, s)


def controls(key):
    torch.set_num_threads(3); k, tj, p, s = key
    d, hid = CD.build(CD.load_raw(k, tj), p, s); Sn = hid["state_norm"]; tr, va, te = rows_of(d, "train"), rows_of(d, "val"), rows_of(d, "test")
    R = dict(kappa=k, traj=tj, p=p, seed=s); tg = {"y1": Sn[:, 1], "y2": Sn[:, 4]}
    for tn, y in tg.items():
        R[f"mean_{tn}"] = rmse((y[te] - y[tr].mean()) ** 2)
    for wn, cols in (("raw_all4", [0, 1, 2, 3]), ("raw_sub1", [0, 1]), ("raw_x1", [0]), ("raw_sub2", [2, 3])):
        F = raw_win(d["X"], d["M"], cols)
        for tn, y in tg.items():
            R[f"{wn}_knn_{tn}"] = rmse((probes.knn(F[tr], y[tr], F[va], y[va], F[te])[0] - y[te]) ** 2)
            R[f"{wn}_mlp_{tn}"] = rmse((probes.mlp(F[tr], y[tr], F[va], y[va], F[te], seed=0)[0] - y[te]) ** 2)
    torch.manual_seed(s); rnd = build_model("C", 4, m, 16, 64); _, Zr, _ = outputs(rnd, torch.tensor(d["X"]), torch.tensor(d["M"]))
    for tn, y in tg.items():
        R[f"random_enc_knn_{tn}"] = rmse((probes.knn(Zr[tr], y[tr], Zr[va], y[va], Zr[te])[0] - y[te]) ** 2)
    return R


if __name__ == "__main__":
    pat = sys.argv[1] if len(sys.argv) > 1 else "p4_*"; tag = sys.argv[2] if len(sys.argv) > 2 else "pilot"
    done = set(pd.read_csv(f"{O}/eval_models_{tag}.csv").exp_id) if os.path.exists(f"{O}/eval_models_{tag}.csv") else set()
    paths = [q for q in sorted(glob.glob(f"{O}/checkpoints/{pat}.pt")) if os.path.basename(q)[:-3] not in done]
    with Pool(16) as pool:
        res = pool.map(evaluate, paths)
        keys = sorted({r[2] for r in res})
        cres = pool.map(controls, keys)
    os.makedirs(f"{O}/preds_eval", exist_ok=True)
    for R, A, _ in res:
        np.savez(f"{O}/preds_eval/{R['exp_id']}.npz", **A)
    new = pd.DataFrame([r[0] for r in res]); fn = f"{O}/eval_models_{tag}.csv"
    (pd.concat([pd.read_csv(fn), new]) if os.path.exists(fn) else new).to_csv(fn, index=False)
    fc = f"{O}/eval_controls_{tag}.csv"; newc = pd.DataFrame(cres)
    (pd.concat([pd.read_csv(fc), newc]) if os.path.exists(fc) else newc).to_csv(fc, index=False)
    print("ALL DONE", len(res))
