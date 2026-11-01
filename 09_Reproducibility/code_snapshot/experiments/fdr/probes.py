"""Phase 3C supervised probes feature -> hidden y. Fit on TRAIN rows only; hyper-parameters chosen on VALIDATION rows
only; TEST rows are only predicted. Features are standardised with train statistics."""
import numpy as np
import torch


def _std(Ftr, *Fs):
    mu, sd = Ftr.mean(0), Ftr.std(0)
    sd = np.where(sd > 1e-8, sd, 1.0)
    return [(F - mu) / sd for F in (Ftr,) + Fs]


def ridge(Ftr, ytr, Fva, yva, Fte, alphas=(1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100)):
    Ftr, Fva, Fte = _std(Ftr, Fva, Fte)
    A = np.c_[Ftr, np.ones(len(Ftr))]
    best = None
    for al in alphas:
        reg = al * np.eye(A.shape[1]); reg[-1, -1] = 0
        w = np.linalg.solve(A.T @ A + reg, A.T @ ytr)
        r = np.sqrt(np.mean((np.c_[Fva, np.ones(len(Fva))] @ w - yva) ** 2))
        if best is None or r < best[0]:
            best = (r, al, w)
    return np.c_[Fte, np.ones(len(Fte))] @ best[2], {"alpha": best[1], "val_rmse": best[0]}


def knn(Ftr, ytr, Fva, yva, Fte, ks=(1, 3, 5, 10, 20, 50)):
    Ftr, Fva, Fte = _std(Ftr, Fva, Fte)
    Tt = torch.tensor(Ftr, dtype=torch.float32); yt = torch.tensor(ytr, dtype=torch.float32)

    def pred(F, k):
        out = []
        for s in range(0, len(F), 2048):
            dd = torch.cdist(torch.tensor(F[s:s + 2048], dtype=torch.float32), Tt)
            out.append(yt[dd.topk(k, largest=False).indices].mean(1))
        return torch.cat(out).numpy()
    best = min(((np.sqrt(np.mean((pred(Fva, k) - yva) ** 2)), k) for k in ks))
    return pred(Fte, best[1]), {"k": best[1], "val_rmse": float(best[0])}


def mlp(Ftr, ytr, Fva, yva, Fte, seed=0, hidden=64, max_epochs=400, patience=30):
    Ftr, Fva, Fte = _std(Ftr, Fva, Fte)
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(Ftr.shape[1], hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, hidden),
                              torch.nn.ReLU(), torch.nn.Linear(hidden, 1))
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    X, Y = torch.tensor(Ftr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32)[:, None]
    Xv = torch.tensor(Fva, dtype=torch.float32)
    g = torch.Generator().manual_seed(seed); best = (np.inf, None, 0); bad = 0
    for ep in range(max_epochs):
        perm = torch.randperm(len(X), generator=g)
        for s in range(0, len(X), 256):
            i = perm[s:s + 256]; loss = ((net(X[i]) - Y[i]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            r = float(((net(Xv)[:, 0].numpy() - yva) ** 2).mean() ** 0.5)
        if r < best[0] - 1e-6:
            best = (r, {k: v.clone() for k, v in net.state_dict().items()}, ep); bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best[1])
    with torch.no_grad():
        return net(torch.tensor(Fte, dtype=torch.float32))[:, 0].numpy(), {"best_epoch": best[2], "val_rmse": best[0]}


PROBES = {"ridge": ridge, "knn": knn, "mlp": mlp}


def run_all(Ftr, ytr, Fva, yva, Fte):
    return {name: f(Ftr, ytr, Fva, yva, Fte) for name, f in PROBES.items()}
