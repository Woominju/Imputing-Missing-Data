"""Phase 6 G2/G3 evaluation of one (kappa, p, replicate) on GPU. Specification fixed in results/notes/phase6_g2_g3_protocol.md
BEFORE any test result. Hidden states are used ONLY for supervised post-hoc probes/readouts (fit on the replicate's train
trajectory, early-stopped on its val trajectory, evaluated once on the replicate's own 8 test trajectories) and for the
oracle-only Jacobian strata. Encoders are frozen.
G2: y_j MLP probe (2 x 64 ReLU, Adam 1e-3, batch 512, <= 200 epochs, patience 20 on val, probe seeds {0,1}) + ridge, per arm
    and raw masked-window features (same spec as Phase 5).
G3: ONE shared nonlinear readout per feature set: MLP 2 x 128 ReLU (same training rule, seed 0) from the frozen feature to the
    full 6-D normalised state, fit on the natural train mask distribution; evaluated on 5 independent test masks per test
    trajectory (e = 0..4); per-mask error, same-state pairwise output differences across masks, sign errors; strata:
    mask-only (L1, L2 counts) and oracle-only (sigma_min of the masked local Jacobian).
Outputs (long format, additive sums): results/extra/phase6/eval/k{K}_p{P}_r{R}_{g2,g3,g3pair}.csv"""
import argparse, itertools, os, time
import numpy as np, pandas as pd, torch
from experiments.phase5.core import ArmModel, encode_series, raw_features, SPAN, M_, SUB
from experiments.phase5.evaluate import ridge_probe
from experiments.phase6.core6 import build, load, ids, norm_stats, test_mask, ARMS6, NT6, NMASK

E = "results/extra/phase6/eval"; RUNS = "results/extra/phase6/runs"; JAC = "results/extra/phase6/jac"; DEV = torch.device("cuda")
G3_FEATS = ["A1", "A2", "Aslot1", "Aslot2", "B1", "B2", "raw_own1", "raw_own2", "raw_joint"]
SIG_BINS = [0.0, 1e-3, 1e-2, 1e-1, np.inf]; SIG_NAMES = ["sig<1e-3", "sig1e-3..1e-2", "sig1e-2..1e-1", "sig>=1e-1"]


def mlp_fit(Ftr, Ytr, Fva, Yva, seed, width):
    """same training rule as experiments.phase5.evaluate.mlp_probe; width configurable; Y may be multi-output."""
    mu, sd = Ftr.mean(0), Ftr.std(0); sd = np.where(sd > 1e-8, sd, 1.0)
    T = lambda F: torch.tensor((F - mu) / sd, dtype=torch.float32, device=DEV)
    X, Xv = T(Ftr), T(Fva); Y = torch.tensor(Ytr, dtype=torch.float32, device=DEV).reshape(len(Ytr), -1)
    Yv = torch.tensor(Yva, dtype=torch.float32, device=DEV).reshape(len(Yva), -1)
    torch.manual_seed(seed); g = torch.Generator(device=DEV).manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], width), torch.nn.ReLU(), torch.nn.Linear(width, width), torch.nn.ReLU(),
                              torch.nn.Linear(width, Y.shape[1])).to(DEV)
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
    def pred(F):
        with torch.no_grad():
            return net(T(F)).cpu().numpy()
    return pred


def own_counts(M4):
    """L_j(t) = # observed x_j entries in the 8-lag window ending at t (x1 = col 0, x2 = col 2); c_j(t) = x_{3-j}(t) observed."""
    c = np.cumsum(np.r_[np.zeros((1, 4)), M4], 0); W = c[M_:] - c[:-M_]                    # window sums for t >= M_-1
    W = np.r_[np.full((M_ - 1, 4), np.nan), W]
    return {1: W[:, 0], 2: W[:, 2]}, {1: M4[:, 2], 2: M4[:, 0]}


def lab(L):
    return np.where(L == 0, "L0", np.where(L == 1, "L1", "L2+"))


def joint_lab(L1, L2):
    a, b = L1 >= 2, L2 >= 2
    return np.where(a & b, "both_L2+", np.where(a | b, "one_L2+", "none_L2+"))


def sig_min(J, M4, t, rows):
    """sigma_min of the masked local Jacobian at times t (N,), J (N, 4, 8, 6), M4 test mask (T, 4); rows = view indices used."""
    lag_mask = np.stack([M4[t - q] for q in range(M_)], 2)                                    # (N, 4, 8)
    out = np.zeros(len(t))
    for n in range(len(t)):
        A = J[n][rows][lag_mask[n][rows] > 0]
        out[n] = 0.0 if A.shape[0] < 6 else np.linalg.svd(A, compute_uv=False)[-1]
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--kappa", type=float); ap.add_argument("--p", type=float); ap.add_argument("--rep", type=int)
    a = ap.parse_args(); k, p, r = a.kappa, a.p, a.rep; tag = f"k{k}_p{p}_r{r}"; os.makedirs(E, exist_ok=True); torch.set_num_threads(2); t0 = time.time()
    I = ids(r); _, _, mu_s, sd_s = norm_stats(k); labS = lambda q: ((load(k, q) - mu_s) / sd_s).astype(np.float32)
    Str, Sva = labS(I["train"]), labS(I["val"]); Ste = [labS(q) for q in I["test"]]
    tr, va, te = np.arange(SPAN, len(Str)), np.arange(SPAN, len(Sva)), np.arange(SPAN, 5000)
    M4 = {(i, e): test_mask(k, p, r, i, e) for i in range(NT6) for e in range(NMASK)}
    CNT = {key: own_counts(m) for key, m in M4.items()}
    feats = {}
    for arm in ARMS6:
        ck = torch.load(f"{RUNS}/ckpt/k{k}_p{p}_r{r}_{arm}_s{6000 + r}.pt", map_location=DEV, weights_only=False)
        D = build(k, p, r, arm); mdl = ArmModel(len(D["ins"]), D["outs"], D["lagmask"]).to(DEV); mdl.load_state_dict(ck["state_dict"])
        enc = lambda X, M: encode_series(mdl, torch.tensor(X, device=DEV), torch.tensor(M, device=DEV))[0].cpu().numpy()
        f = {"train": enc(D["Xtr"], D["Mtr"]), "val": enc(D["Xva"], D["Mva"]), "test": {}}
        for i in range(NT6):
            Xi, _ = D["test"](i)
            for e in (range(NMASK) if arm in G3_FEATS else [0]):
                f["test"][(i, e)] = enc(Xi, M4[(i, e)][:, D["ins"]])
        feats[arm] = f
    D4 = build(k, p, r, "B1")                                                                  # canonical real 4-view order x1 z1 x2 z2
    def rawset(X, M):
        o = {f"raw_own{j}": raw_features(X, M, SUB[j]) for j in (1, 2)}; o["raw_joint"] = raw_features(X, M, [0, 1, 2, 3])
        for j in (1, 2):
            o[f"raw_owncur{j}"] = np.c_[o[f"raw_own{j}"], raw_features(X, M, SUB[3 - j], lags=1)]
        return o
    rt, rv = rawset(D4["Xtr"], D4["Mtr"]), rawset(D4["Xva"], D4["Mva"]); rte = {}
    for i in range(NT6):
        Xi, _ = D4["test"](i)
        for e in range(NMASK):
            for nm, F in rawset(Xi, M4[(i, e)]).items():
                if e == 0 or nm in G3_FEATS:
                    rte.setdefault(nm, {})[(i, e)] = F
    for nm in rt:
        feats[nm] = {"train": rt[nm], "val": rv[nm], "test": rte[nm]}
    # ---------------- G2: y_j probes ----------------
    G2 = []
    for j in (1, 2):
        yc = (1, 4)[j - 1]
        for fn in [f"A{j}", f"Aslot{j}", f"B{j}", f"Bcur{j}", f"Sham{j}", f"raw_own{j}", "raw_joint", f"raw_owncur{j}"]:
            F = feats[fn]; fitted = {f"mlp{s}": mlp_fit(F["train"][tr], Str[tr, yc], F["val"][va], Sva[va, yc], s, 64) for s in (0, 1)}
            fitted["ridge"] = ridge_probe(F["train"][tr], Str[tr, yc], F["val"][va], Sva[va, yc])
            for i in range(NT6):
                L, c = CNT[(i, 0)]; Lj, cj = lab(L[j][te]), c[j][te]; y = Ste[i][te, yc]; Fi = F["test"][(i, 0)][te]
                for pn, fp in fitted.items():
                    e2 = (fp(Fi).reshape(-1) - y) ** 2
                    for sn, sel in (("all", np.ones(len(te), bool)), ("L0", Lj == "L0"), ("L1", Lj == "L1"), ("L2+", Lj == "L2+"),
                                    ("c0", cj == 0), ("c1", cj == 1)):
                        G2.append(dict(feature=fn, j=j, probe=pn, traj=I["test"][i], stratum=sn, sse=float(e2[sel].sum()), n=int(sel.sum())))
    # ---------------- G3: shared 6-D readout across 5 masks ----------------
    G3, G3P = [], []
    jf = f"{JAC}/k{float(k)}_r{r}.npz"; JZ = np.load(jf) if os.path.exists(jf) else None
    for fn in G3_FEATS:
        F = feats[fn]; rd = mlp_fit(F["train"][tr], Str[tr], F["val"][va], Sva[va], 0, 128)
        rows_used = [0, 1] if fn in ("A1", "raw_own1") else [2, 3] if fn in ("A2", "raw_own2") else [0, 1, 2, 3]   # views fed (Aslot: own only)
        if fn.startswith("Aslot"):
            rows_used = SUB[int(fn[-1])]
        for i in range(NT6):
            S = Ste[i][te]; P = np.stack([rd(F["test"][(i, e)][te]) for e in range(NMASK)], 1)          # (N, 5, 6)
            ER = (P - S[:, None]) ** 2                                                                    # (N, 5, 6)
            if JZ is not None:
                sel_j = JZ["traj"] == i; tj = JZ["t"][sel_j]; Jj = JZ["J"][sel_j]
            for e in range(NMASK):
                L, _ = CNT[(i, e)]; L1, L2 = L[1][te], L[2][te]; jl = joint_lab(L1, L2)
                groups = {"all": np.ones(len(te), bool)}
                for nm in ("both_L2+", "one_L2+", "none_L2+"):
                    groups[nm] = jl == nm
                for jj, LL in ((1, L1), (2, L2)):
                    for nm in ("L0", "L1", "L2+"):
                        groups[f"L{jj}:{nm}"] = lab(LL) == nm
                for gname, sel in groups.items():
                    for cname, cols in (("s6", slice(0, 6)), ("s1", slice(0, 3)), ("s2", slice(3, 6)), ("y1", slice(1, 2)), ("y2", slice(4, 5))):
                        G3.append(dict(feature=fn, traj=I["test"][i], mask=e, stratum=gname, coord=cname,
                                       sse=float(ER[sel, e, cols].sum()), n=int(sel.sum())))
                    for yc, yn in ((1, "y1"), (4, "y2")):
                        big = sel & (np.abs(S[:, yc]) > 0.5)
                        G3.append(dict(feature=fn, traj=I["test"][i], mask=e, stratum=gname, coord=f"signerr_{yn}",
                                       sse=float((np.sign(P[big, e, yc]) != np.sign(S[big, yc])).sum()), n=int(big.sum())))
                if JZ is not None:                                                                           # oracle-only strata
                    sm = sig_min(Jj, M4[(i, e)], tj, rows_used); idx = tj - SPAN
                    for b, nm in enumerate(SIG_NAMES):
                        sel = (sm >= SIG_BINS[b]) & (sm < SIG_BINS[b + 1])
                        for cname, cols in (("s6", slice(0, 6)), ("y1", slice(1, 2)), ("y2", slice(4, 5))):
                            G3.append(dict(feature=fn, traj=I["test"][i], mask=e, stratum=f"oracle:{nm}", coord=cname,
                                           sse=float(ER[idx[sel], e, cols].sum()), n=int(sel.sum())))
            # same-state stability: pairwise output differences between independent masks, by mask-only stratum of BOTH masks
            JL = np.stack([joint_lab(CNT[(i, e)][0][1][te], CNT[(i, e)][0][2][te]) for e in range(NMASK)], 1)
            for e1, e2 in itertools.combinations(range(NMASK), 2):
                d2 = (P[:, e1] - P[:, e2]) ** 2; same = JL[:, e1] == JL[:, e2]
                for gname in ("all", "both_L2+", "one_L2+", "none_L2+"):
                    sel = np.ones(len(te), bool) if gname == "all" else same & (JL[:, e1] == gname)
                    for cname, cols in (("s6", slice(0, 6)), ("y1", slice(1, 2)), ("y2", slice(4, 5))):
                        G3P.append(dict(feature=fn, traj=I["test"][i], pair=f"{e1}{e2}", stratum=gname, coord=cname,
                                        sse=float(d2[sel, cols].sum()), n=int(sel.sum())))
            # error of the mask-averaged output (accuracy of the common part), all strata pooled
            Pm = P.mean(1); G3.append(dict(feature=fn, traj=I["test"][i], mask=-1, stratum="mask_mean", coord="s6",
                                           sse=float(((Pm - S) ** 2).sum()), n=len(te)))
    meta = dict(kappa=k, p=p, rep=r)
    for nm, rows in (("g2", G2), ("g3", G3), ("g3pair", G3P)):
        df = pd.DataFrame(rows)
        for kk, vv in meta.items():
            df[kk] = vv
        df.to_csv(f"{E}/{tag}_{nm}.csv", index=False)
    print("EVAL DONE", tag, f"{time.time() - t0:.0f}s", "jac" if JZ is not None else "nojac", flush=True)


if __name__ == "__main__":
    main()
