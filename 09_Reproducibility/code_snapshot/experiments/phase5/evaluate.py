"""Phase 5 evaluation of one (kappa, p, replicate[, seed]) on GPU. Hidden states are used ONLY here, for supervised post-hoc
probes (fit on the replicate's train trajectory, early-stopped on its val trajectory, evaluated once on test trajectories
200..209) and for analysis. Encoders are frozen.
Probe spec (fixed before any test result): features standardised with train statistics; MLP 2 x 64 ReLU, Adam 1e-3,
batch 512, <= 200 epochs, patience 20 on val RMSE, probe seeds {0, 1}; ridge with alpha chosen on val.
Outputs (long format, additive sums so any aggregation is exact): results/extra/phase5/eval/<tag>_{probe,imp,h3,dyn}.csv"""
import argparse, os, time
import numpy as np, pandas as pd, torch
from experiments.phase5.core import build, ArmModel, encode_series, raw_features, load, ARMS, SUB, SPAN, M_

E = "results/extra/phase5/eval"; RUNS = "results/extra/phase5/runs"; NTEST = 10; DEV = torch.device("cuda")


def state_norm(k, r):
    S = load(k, r); return S.mean(0), S.std(0)


def mlp_probe(Ftr, ytr, Fva, yva, seed):
    mu, sd = Ftr.mean(0), Ftr.std(0); sd = np.where(sd > 1e-8, sd, 1.0)
    T = lambda F: torch.tensor((F - mu) / sd, dtype=torch.float32, device=DEV)
    X, Y, Xv = T(Ftr), torch.tensor(ytr, dtype=torch.float32, device=DEV).reshape(len(ytr), -1), T(Fva)
    Yv = torch.tensor(yva, dtype=torch.float32, device=DEV).reshape(len(yva), -1)
    torch.manual_seed(seed); g = torch.Generator(device=DEV).manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 64), torch.nn.ReLU(), torch.nn.Linear(64, 64), torch.nn.ReLU(),
                              torch.nn.Linear(64, Y.shape[1])).to(DEV)
    opt = torch.optim.Adam(net.parameters(), 1e-3); best, state, bad = np.inf, None, 0
    for ep in range(200):
        perm = torch.randperm(len(X), generator=g, device=DEV)
        for s in range(0, len(X), 512):
            i = perm[s:s + 512]; loss = ((net(X[i]) - Y[i]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            v = float(((net(Xv) - Yv) ** 2).mean())
        if v < best - 1e-7:
            best, state, bad = v, {k: q.clone() for k, q in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 20:
                break
    net.load_state_dict(state); net.eval()
    return lambda F: net(T(F)).detach().cpu().numpy()


def ridge_probe(Ftr, ytr, Fva, yva):
    mu, sd = Ftr.mean(0), Ftr.std(0); sd = np.where(sd > 1e-8, sd, 1.0)
    A = np.c_[(Ftr - mu) / sd, np.ones(len(Ftr))]; Av = np.c_[(Fva - mu) / sd, np.ones(len(Fva))]; best = None
    for al in (1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100):
        reg = al * np.eye(A.shape[1]); reg[-1, -1] = 0; w = np.linalg.solve(A.T @ A + reg, A.T @ ytr)
        v = ((Av @ w - yva) ** 2).mean()
        if best is None or v < best[0]:
            best = (v, w)
    return lambda F: np.c_[(F - mu) / sd, np.ones(len(F))] @ best[1]


def strata(M4):
    """per time t (rows >= SPAN): L_j = observed x_j entries in own window, c_j = x_{3-j}(t) observed. M4: (T,4) mask."""
    T = len(M4); out = {}
    for j, xc, xo in ((1, 0, 2), (2, 2, 0)):
        L = np.array([M4[max(0, t - M_ + 1):t + 1, xc].sum() for t in range(T)])
        out[j] = (np.where(L == 0, "L0", np.where(L == 1, "L1", "L2+")), np.where(M4[:, xo] > 0, "c1", "c0"))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--kappa", type=float); ap.add_argument("--p", type=float); ap.add_argument("--rep", type=int)
    ap.add_argument("--seed", type=int, default=None); ap.add_argument("--arms", default=",".join(ARMS)); a = ap.parse_args()
    k, p, r = a.kappa, a.p, a.rep; seed = r if a.seed is None else a.seed; arms = a.arms.split(",")
    tag = f"k{k}_p{p}_r{r}_s{seed}"; os.makedirs(E, exist_ok=True); torch.set_num_threads(2); t0 = time.time()
    mu_s, sd_s = state_norm(k, r)
    lab = lambda j_id: (load(k, j_id) - mu_s) / sd_s
    Str, Sva, Ste = lab(r), lab(100 + r), [lab(200 + i) for i in range(NTEST)]
    tr, va = np.arange(SPAN, len(Str)), np.arange(SPAN, len(Sva)); te = np.arange(SPAN, 5000)
    Mte = {i: {e: None for e in range(5)} for i in range(NTEST)}
    from experiments.phase5.core import test_masks
    for i in range(NTEST):
        for e in range(5):
            Mte[i][e] = test_masks(i, p, e)
    ST = {i: strata(Mte[i][0]) for i in range(NTEST)}
    # ---------------- features ----------------
    feats = {}            # name -> dict(train, val, test[i][e]) ; e only for H3 sets
    models = {}
    for arm in arms:
        ck = torch.load(f"{RUNS}/ckpt/k{k}_p{p}_r{r}_{arm}_s{seed}.pt", map_location=DEV, weights_only=False)
        D = build(k, p, r, arm); mdl = ArmModel(len(D["ins"]), D["outs"], D["lagmask"]).to(DEV); mdl.load_state_dict(ck["state_dict"])
        enc = lambda X, M: [q.cpu().numpy() for q in encode_series(mdl, torch.tensor(X, device=DEV), torch.tensor(M, device=DEV))]
        h3 = arm[0] in "AC" or arm in ("B1", "B2")
        f = {"train": enc(D["Xtr"], D["Mtr"])[0], "val": enc(D["Xva"], D["Mva"])[0], "test": {}, "y0": {}}
        for i in range(NTEST):
            Xi, _ = D["test"](i)
            for e in (range(5) if h3 else [0]):
                zz, y0 = enc(Xi, Mte[i][e][:, D["ins"]]); f["test"][(i, e)] = zz
                if e == 0:
                    f["y0"][i] = (y0, Xi[:, D["outs"]], Mte[i][0][:, D["ins"]][:, D["outs"]])
        feats[arm] = f; models[arm] = D
    if "C" in arms:                     # raw masked-window features (same normalisation as arm C = all four views)
        D = models["C"]; cols_own = {1: [0, 1], 2: [2, 3]}
        def rawset(X, M):
            o = {f"raw_own{j}": raw_features(X, M, cols_own[j]) for j in (1, 2)}
            o["raw_joint"] = raw_features(X, M, [0, 1, 2, 3]); o["raw_cur"] = raw_features(X, M, [0, 1, 2, 3], lags=1)
            for j in (1, 2):
                o[f"raw_owncur{j}"] = np.c_[o[f"raw_own{j}"], raw_features(X, M, cols_own[3 - j], lags=1)]
            return o
        rt, rv = rawset(D["Xtr"], D["Mtr"]), rawset(D["Xva"], D["Mva"])
        rte = {}
        for i in range(NTEST):
            Xi, _ = D["test"](i)
            for e in range(5):
                for nm, F in rawset(Xi, Mte[i][e]).items():
                    rte.setdefault(nm, {})[(i, e)] = F
        for nm in rt:
            feats[nm] = {"train": rt[nm], "val": rv[nm], "test": rte[nm]}
    # ---------------- probes: y_j (MLP x2 seeds, ridge), s_j (ridge) ----------------
    PR, H3, DY, IMP = [], [], [], []
    def feat_for(j):
        L = [n for n in (f"A{j}", f"B{j}", f"Bcur{j}", "C", f"Sham{j}") if n in feats]
        return L + [n for n in (f"raw_own{j}", "raw_joint", f"raw_owncur{j}", "raw_cur") if n in feats]
    for j in (1, 2):
        yc, sc = (1, 4)[j - 1], SUB_STATE[j]
        for fn in feat_for(j):
            F = feats[fn]; fitted = {}
            for ps in (0, 1):
                fitted[f"mlp{ps}"] = mlp_probe(F["train"][tr], Str[tr, yc], F["val"][va], Sva[va, yc], ps)
            fitted["ridge"] = ridge_probe(F["train"][tr], Str[tr, yc], F["val"][va], Sva[va, yc])
            rs = ridge_probe(F["train"][tr], Str[tr][:, sc], F["val"][va], Sva[va][:, sc])
            for i in range(NTEST):
                Lj, cj = ST[i][j]; Fi = F["test"][(i, 0)][te]; y = Ste[i][te, yc]
                for pn, fp in fitted.items():
                    e2 = (fp(Fi).reshape(-1) - y) ** 2
                    for sname, sel in (("all", np.ones(len(te), bool)), ("L0", Lj[te] == "L0"), ("L1", Lj[te] == "L1"), ("L2+", Lj[te] == "L2+"),
                                       ("c0", cj[te] == "c0"), ("c1", cj[te] == "c1"), ("L0c1", (Lj[te] == "L0") & (cj[te] == "c1"))):
                        PR.append(dict(feature=fn, j=j, target=f"y{j}", probe=pn, traj=200 + i, stratum=sname, sse=float(e2[sel].sum()), n=int(sel.sum())))
                e2s = ((rs(Fi) - Ste[i][te][:, sc]) ** 2).sum(1)
                PR.append(dict(feature=fn, j=j, target=f"s{j}", probe="ridge", traj=200 + i, stratum="all", sse=float(e2s.sum()), n=len(te)))
                # ---- H3: shared readout across 5 masks (only feature sets with 5 test masks) ----
                if (i, 4) in F["test"]:
                    R = np.stack([fitted["mlp0"](F["test"][(i, e)][te]).reshape(-1) for e in range(5)], 1)      # (N, 5)
                    Rs = np.stack([rs(F["test"][(i, e)][te]) for e in range(5)], 1)                               # (N, 5, 3)
                    Le = np.stack([strata(Mte[i][e])[j][0][te] for e in range(5)], 1)                             # (N, 5)
                    for lev in ("L0", "L1", "L2+"):
                        q = Le == lev
                        H3.append(dict(feature=fn, j=j, traj=200 + i, kind="readout_err_y", stratum=lev, sse=float((((R - y[:, None]) ** 2) * q).sum()), n=int(q.sum())))
                        H3.append(dict(feature=fn, j=j, traj=200 + i, kind="readout_err_s", stratum=lev,
                                       sse=float((((Rs - Ste[i][te][:, None, sc]) ** 2).sum(-1) * q).sum()), n=int(q.sum())))
                    adm = Le == "L2+"; na = adm.sum(1); ok = na >= 2
                    def disp(V):                                   # within-time variance across admissible masks (unbiased)
                        mu_ = (V * adm[..., None]).sum(1) / np.maximum(na, 1)[:, None]
                        return ((((V - mu_[:, None]) ** 2) * adm[..., None]).sum((1, 2)) / np.maximum(na - 1, 1))[ok]
                    H3.append(dict(feature=fn, j=j, traj=200 + i, kind="dispersion_y_adm", stratum="L2+", sse=float(disp(R[..., None]).sum()), n=int(ok.sum())))
                    H3.append(dict(feature=fn, j=j, traj=200 + i, kind="dispersion_s_adm", stratum="L2+", sse=float(disp(Rs).sum()), n=int(ok.sum())))
                    allm = np.ones_like(adm); na2 = 5
                    mu2 = R.mean(1, keepdims=True)
                    H3.append(dict(feature=fn, j=j, traj=200 + i, kind="dispersion_y_allmasks", stratum="all", sse=float((((R - mu2) ** 2).sum(1) / 4).sum()), n=len(te)))
        # ---- dynamics: y_j(t+5) from features at t (same trajectory, t+5 < T) ----
        for fn in [n for n in (f"A{j}", f"B{j}", "C", f"raw_own{j}", "raw_joint") if n in feats]:
            F = feats[fn]; h = 5
            trh, vah, teh = tr[tr + h < len(Str)], va[va + h < len(Sva)], te[te + h < 5000]
            fp = mlp_probe(F["train"][trh], Str[trh + h, yc], F["val"][vah], Sva[vah + h, yc], 0)
            for i in range(NTEST):
                e2 = (fp(F["test"][(i, 0)][teh]).reshape(-1) - Ste[i][teh + h, yc]) ** 2
                DY.append(dict(feature=fn, j=j, target=f"y{j}(t+5)", traj=200 + i, sse=float(e2.sum()), n=len(teh)))
    # ---------------- imputation (lag-0, own columns, missing positions of the primary test mask) ----------------
    for arm in arms:
        D = models[arm]
        for i in range(NTEST):
            y0, Xo, Mo = feats[arm]["y0"][i]
            for q, oc in enumerate(D["outs"]):
                v = D["ins"][oc]; j = 1 if v in (0, 1) else 2; Lj = ST[i][j][0]
                miss = (Mo[:, q] == 0); miss[:SPAN] = False; e2 = (y0[:, q] - Xo[:, q]) ** 2
                for sname, sel in (("all", miss), ("L0", miss & (Lj == "L0")), ("L1", miss & (Lj == "L1")), ("L2+", miss & (Lj == "L2+"))):
                    IMP.append(dict(arm=arm, view=["x1", "z1", "x2", "z2"][v], j=j, traj=200 + i, stratum=sname, sse=float(e2[sel].sum()), n=int(sel.sum())))
    meta = dict(kappa=k, p=p, rep=r, seed=seed)
    for nm, rows in (("probe", PR), ("h3", H3), ("dyn", DY), ("imp", IMP)):
        df = pd.DataFrame(rows)
        for kk, vv in meta.items():
            df[kk] = vv
        df.to_csv(f"{E}/{tag}_{nm}.csv", index=False)
    print("EVAL DONE", tag, f"{time.time() - t0:.0f}s", flush=True)


SUB_STATE = {1: [0, 1, 2], 2: [3, 4, 5]}

if __name__ == "__main__":
    main()
