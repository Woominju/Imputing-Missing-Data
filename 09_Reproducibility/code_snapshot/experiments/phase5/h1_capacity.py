"""Phase 5 H1 capacity sensitivity (exploratory, GPU): raw masked-window estimators of y_j trained on the POOLED 8 train
trajectories (each with its replicate training mask), early-stopped on the pooled 8 val trajectories, tested on the 10 test
trajectories with the primary test masks. MLP widths {64, 256, 1024} (2 hidden layers, ReLU), Adam 1e-3, batch 1024,
<= 150 epochs, patience 10. Feature sets: own_j, joint, own_j + other lag-0, current-only (all four views lag 0).
Normalisation: observed entries of the pooled train trajectories. Output: results/extra/phase5/h1/k{k}_p{p}.csv"""
import sys, os
import numpy as np, pandas as pd, torch
from experiments.phase5.core import load, mcar, seed_of, test_masks, raw_features, OBS, SPAN, M_
from experiments.phase5.evaluate import strata

DEV = torch.device("cuda"); O = "results/extra/phase5/h1"


def fit(F, y, Fv, yv, width, seed=0):
    mu, sd = F.mean(0), F.std(0); sd = np.where(sd > 1e-8, sd, 1.0)
    T = lambda A: torch.tensor((A - mu) / sd, dtype=torch.float32, device=DEV)
    X, Y, Xv, Yv = T(F), torch.tensor(y, dtype=torch.float32, device=DEV)[:, None], T(Fv), torch.tensor(yv, dtype=torch.float32, device=DEV)[:, None]
    torch.manual_seed(seed); g = torch.Generator(device=DEV).manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], width), torch.nn.ReLU(), torch.nn.Linear(width, width), torch.nn.ReLU(), torch.nn.Linear(width, 1)).to(DEV)
    opt = torch.optim.Adam(net.parameters(), 1e-3); best, st, bad = np.inf, None, 0
    for ep in range(150):
        perm = torch.randperm(len(X), generator=g, device=DEV)
        for s in range(0, len(X), 1024):
            i = perm[s:s + 1024]; loss = ((net(X[i]) - Y[i]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            v = float(((net(Xv) - Yv) ** 2).mean())
        if v < best - 1e-7:
            best, st, bad = v, {k: q.clone() for k, q in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 10:
                break
    net.load_state_dict(st); net.eval()
    return lambda A: net(T(A)).detach().cpu().numpy()[:, 0]


def main(k, p):
    os.makedirs(O, exist_ok=True)
    tr = [(load(k, r), mcar((20000, 4), p, seed_of("train", k, p, r))) for r in range(8)]
    va = [(load(k, 100 + r), mcar((5000, 4), p, seed_of("val", k, p, r))) for r in range(8)]
    te = [(load(k, 200 + i), test_masks(i, p)) for i in range(10)]
    Xall = np.concatenate([X[:, OBS] for X, _ in tr]); Mall = np.concatenate([M for _, M in tr])
    mu = np.array([Xall[Mall[:, c] > 0, c].mean() for c in range(4)]); sd = np.array([Xall[Mall[:, c] > 0, c].std() for c in range(4)])
    S = np.concatenate([X for X, _ in tr]); smu, ssd = S.mean(0), S.std(0)
    def fs(X, M, j):
        Xn = ((X[:, OBS] - mu) / sd).astype(np.float32); own, oth = ([0, 1], [2, 3]) if j == 1 else ([2, 3], [0, 1])
        o = raw_features(Xn, M, own); return {"own": o, "joint": raw_features(Xn, M, [0, 1, 2, 3]),
                                             "owncur": np.c_[o, raw_features(Xn, M, oth, lags=1)], "cur": raw_features(Xn, M, [0, 1, 2, 3], lags=1)}
    rows = []
    for j in (1, 2):
        yc = 1 if j == 1 else 4
        Ftr = [fs(X, M, j) for X, M in tr]; Fva = [fs(X, M, j) for X, M in va]; Fte = [fs(X, M, j) for X, M in te]
        cat = lambda L, nm: np.concatenate([d[nm][SPAN:] for d in L])
        ytr = np.concatenate([((X[:, yc] - smu[yc]) / ssd[yc])[SPAN:] for X, _ in tr]); yva = np.concatenate([((X[:, yc] - smu[yc]) / ssd[yc])[SPAN:] for X, _ in va])
        for nm in ("own", "joint", "owncur", "cur"):
            for w in (64, 256, 1024):
                f = fit(cat(Ftr, nm), ytr, cat(Fva, nm), yva, w)
                for i, (X, M) in enumerate(te):
                    y = ((X[:, yc] - smu[yc]) / ssd[yc])[SPAN:]; e2 = (f(Fte[i][nm][SPAN:]) - y) ** 2; L = strata(M)[j][0][SPAN:]
                    for st, sel in (("all", np.ones(len(y), bool)), ("L0", L == "L0"), ("L1", L == "L1"), ("L2+", L == "L2+")):
                        rows.append(dict(kappa=k, p=p, j=j, feature=nm, width=w, traj=200 + i, stratum=st, sse=float(e2[sel].sum()), n=int(sel.sum())))
                print(k, p, j, nm, w, flush=True)
    pd.DataFrame(rows).to_csv(f"{O}/k{k}_p{p}.csv", index=False)


if __name__ == "__main__":
    main(float(sys.argv[1]), float(sys.argv[2]))
