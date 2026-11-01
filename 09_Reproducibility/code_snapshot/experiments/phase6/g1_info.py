"""Phase 6 Gate G1: observation-level conditional information (supervised, ORACLE-LEVEL diagnostics: hidden y_j is the
training target of these estimators; this is NOT the unsupervised AE). Protocol: results/notes/phase6_g1_protocol.md.
One call = one (kappa, p). Own/other symmetry: the law of (y_j, U_j, V_j) is identical for j = 1, 2 (subsystem swap P is an
exact symmetry of the generator law), so both j are pooled as samples of one problem.
Inputs (past-only 8-lag windows, normalised by the G2 pool constants; values*mask and mask):
  U   = own (x_j, z_j) window                    V   = other (x_o, z_o) window
  Vc  = other lag 0 only (current values)        Vs  = other window from an INDEPENDENT donor trajectory (exact null)
Target: y_j / sd(y) (NOT centred: E[y_j] = 0 exactly by the Gamma x Gamma symmetry of the generator law).
Masks: training = fresh i.i.d. Bernoulli(1-p) per window entry per batch (same marginal law as a window cut from the grid
MCAR); val = fixed grid MCAR per trajectory; test = NTM independent grid masks per test trajectory.
Cross-fitting: fold A (train ids 7000..7015) and fold B (7016..7031).
  f_X^A, f_X^B : MLP(3 x 256, GELU) predictors of y from input set X in {U, UV, UVc, UVs}
  witness (for X in {UV, UVc, UVs}): q = f_X^A - f_U^A ; r^B(U) ~ E[q | U] fitted on fold B ; qt = q - r^B(U);
           eps = y - f_U^B(U) ; T = E_n[eps * qt], Dq = E_n[qt^2]
  certificate (L0 stratum): G = E_n[y^2 - (y - fbar_X)^2], fbar_X = (f_X^A + f_X^B)/2 (fixed before test)
Structural audit (kappa, p): 'p5like' regime = one train trajectory, ONE fixed natural mask, width 64 (Phase 5 raw-MLP
setting), and 'fixedmask' regime = fold A with one fixed mask per trajectory, vs the main 'fresh' regime.
Output: results/extra/phase6/g1/k{K}_p{P}.csv (per test trajectory x stratum sums), *_train.json (training log)."""
import json, os, sys, time
import numpy as np, pandas as pd, torch
from experiments.phase5.core import mcar, seed_of, OBS
from experiments.phase6.core6 import load, norm_stats

O = "results/extra/phase6/g1"; DEV = torch.device("cuda"); M_ = 8; NTM = 10
FOLD_A, FOLD_B = [7000 + i for i in range(16)], [7016 + i for i in range(16)]
VAL, TEST = [7100 + i for i in range(8)], [7200 + i for i in range(32)]
DON = lambda q: 1000 + q                                                     # donor id = target id + 1000 (8000.., 8100.., 8200..)


def windows(X4, M4):
    """X4, M4: (T, 4) normalised observed views x1 z1 x2 z2 and mask -> per t >= 7, for own j = 1 and j = 2 (swapped):
    own (N, 2, 8), oth (N, 2, 8) values*mask and masks. lag k at index k."""
    T = len(X4); t = np.arange(M_ - 1, T); L = np.stack([t - k for k in range(M_)], 1)          # (N, 8)
    V = (X4 * M4)[L].transpose(0, 2, 1); MM = M4[L].transpose(0, 2, 1)                     # (N, 4, 8)
    return t, V, MM


class Pool:
    """stacked windows of a set of trajectories (both own-orientations). Full (unmasked) values; masks applied later."""
    def __init__(self, k, idl, donor=False):
        mu, sd, _, sds = norm_stats(k); self.sy = sds[1]; Xs, Ys, Ds = [], [], []
        for q in idl:
            S = load(k, q); X4 = ((S[:, OBS] - mu) / sd).astype(np.float32)
            Dn = ((load(k, DON(q))[:, OBS] - mu) / sd).astype(np.float32) if donor else None
            t, V, _ = windows(X4, np.ones_like(X4))
            for j, (own, oth, yc) in enumerate((([0, 1], [2, 3], 1), ([2, 3], [0, 1], 4))):
                Xs.append(np.concatenate([V[:, own], V[:, oth]], 1))                      # (N, 4, 8): own x, own z, oth x, oth z
                Ys.append((S[t, yc] / self.sy).astype(np.float32))
                if donor:
                    _, Vd, _ = windows(Dn, np.ones_like(Dn)); Ds.append(Vd[:, oth])
        self.X = torch.tensor(np.concatenate(Xs), device=DEV); self.y = torch.tensor(np.concatenate(Ys), device=DEV)
        self.D = torch.tensor(np.concatenate(Ds), device=DEV) if donor else None


def feats(X, M, D, kind):
    """X (B, 4, 8) full values, M (B, 4, 8) mask, D (B, 2, 8) donor other-window values or None -> input vector."""
    if kind == "UVs":
        X = torch.cat([X[:, :2], D], 1)
    Xm = X * M
    if kind == "U":
        return torch.cat([Xm[:, :2].flatten(1), M[:, :2].flatten(1)], 1)
    if kind == "UVc":
        return torch.cat([Xm[:, :2].flatten(1), Xm[:, 2:, 0], M[:, :2].flatten(1), M[:, 2:, 0]], 1)
    return torch.cat([Xm.flatten(1), M.flatten(1)], 1)


def net(din, width, depth=3):
    L, d = [], din
    for _ in range(depth):
        L += [torch.nn.Linear(d, width), torch.nn.GELU()]; d = width
    return torch.nn.Sequential(*L, torch.nn.Linear(d, 1)).to(DEV)


def fit(pool, kind, p, val, seed, width=256, depth=3, target=None, fixed_mask=None, max_ep=40, patience=5, batch=2048):
    """fit a predictor of `target` (default y) from input set `kind`. target: callable(Xb, Mb, Db) -> (B,) for r-regressions.
    fixed_mask: tensor (N, 4, 8) of masks to use instead of fresh draws (structural audit regimes)."""
    g = torch.Generator(device=DEV).manual_seed(seed); torch.manual_seed(seed)
    N = len(pool.X); f = net(feats(pool.X[:2], torch.ones_like(pool.X[:2]), None if pool.D is None else pool.D[:2], kind).shape[1], width, depth)
    opt = torch.optim.Adam(f.parameters(), 1e-3); best, state, bad, log = np.inf, None, 0, []
    vX, vM, vD, vy = val
    vt = vy if target is None else target(vX, vM, vD)
    for ep in range(max_ep):
        f.train(); perm = torch.randperm(N, generator=g, device=DEV)
        for s in range(0, N, batch):
            i = perm[s:s + batch]; X = pool.X[i]; Dd = None if pool.D is None else pool.D[i]
            M = fixed_mask[i] if fixed_mask is not None else (torch.rand(X.shape, generator=g, device=DEV) >= p).float()
            yt = pool.y[i] if target is None else target(X, M, Dd)
            loss = ((f(feats(X, M, Dd, kind)).squeeze(-1) - yt) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        f.eval()
        with torch.no_grad():
            v = float(np.mean([((f(feats(vX[s:s + 65536], vM[s:s + 65536], None if vD is None else vD[s:s + 65536], kind)).squeeze(-1)
                                 - vt[s:s + 65536]) ** 2).mean().item() for s in range(0, len(vX), 65536)]))
        log.append(v)
        if v < best - 1e-7:
            best, bad, state = v, 0, {a: b.clone() for a, b in f.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    f.load_state_dict(state); f.eval()
    for q in f.parameters():
        q.requires_grad_(False)
    return f, dict(kind=kind, best_val=best, epochs=len(log))


def fixed_masks(pool_ids, p, tag, n_per):
    """grid-MCAR masks (one per trajectory) converted to window masks, same orientation stacking as Pool."""
    out = []
    for q in pool_ids:
        Mg = mcar((n_per, 4), p, seed_of(tag, q, p)); _, _, MM = windows(np.zeros((n_per, 4), np.float32), Mg)
        for own, oth in (([0, 1], [2, 3]), ([2, 3], [0, 1])):
            out.append(np.concatenate([MM[:, own], MM[:, oth]], 1))
    return torch.tensor(np.concatenate(out), device=DEV)


def predict(f, X, M, D, kind):
    with torch.no_grad():
        return torch.cat([f(feats(X[s:s + 65536], M[s:s + 65536], None if D is None else D[s:s + 65536], kind)).squeeze(-1)
                          for s in range(0, len(X), 65536)])


def main(k, p):
    os.makedirs(O, exist_ok=True); t0 = time.time(); info = []
    A, B = Pool(k, FOLD_A, donor=True), Pool(k, FOLD_B, donor=True); V = Pool(k, VAL, donor=True)
    vM = fixed_masks(VAL, p, "g1val", 5000) if p > 0 else torch.ones_like(V.X); val = (V.X, vM, V.D, V.y)
    KINDS = ["U", "UV", "UVc", "UVs"]; F = {}
    for fold, P_ in (("A", A), ("B", B)):
        for kind in KINDS:
            F[(kind, fold)], inf = fit(P_, kind, p, val, seed=seed_of("g1fit", k, p, kind, fold) % 2 ** 31); info.append(dict(inf, fold=fold))
    R = {}
    for kind in ("UV", "UVc", "UVs"):                                         # r^B(U) ~ E[f_X^A - f_U^A | U], fitted on fold B
        tgt = lambda X, M, D, kind=kind: (F[(kind, "A")](feats(X, M, D, kind)) - F[("U", "A")](feats(X, M, D, "U"))).squeeze(-1)
        R[kind], inf = fit(B, "U", p, val, seed=seed_of("g1r", k, p, kind) % 2 ** 31, target=tgt); info.append(dict(inf, fold="B", role=f"r_{kind}"))
    AUD = {}
    if p > 0:                                                                # structural audit regimes (U vs UV only)
        P1 = Pool(k, FOLD_A[:1], donor=True); fm1 = fixed_masks(FOLD_A[:1], p, "g1fixed", 20000)
        fmA = fixed_masks(FOLD_A, p, "g1fixed", 20000)
        for kind in ("U", "UV"):
            AUD[(kind, "p5like")], inf = fit(P1, kind, p, val, seed=11, width=64, depth=2, fixed_mask=fm1, max_ep=200, patience=20, batch=512); info.append(dict(inf, regime="p5like"))
            AUD[(kind, "fixedmask")], inf = fit(A, kind, p, val, seed=12, fixed_mask=fmA); info.append(dict(inf, regime="fixedmask"))
    # ---------------- test ----------------
    rows = []
    for q in TEST:
        T_ = Pool(k, [q], donor=True)
        for e in range(NTM if p > 0 else 1):
            Mg = mcar((5000, 4), p, seed_of("g1test", k, p, q, e)) if p > 0 else np.ones((5000, 4), np.float32)
            _, _, MM = windows(np.zeros((5000, 4), np.float32), Mg)
            M = torch.tensor(np.concatenate([np.concatenate([MM[:, o], MM[:, t_]], 1) for o, t_ in (([0, 1], [2, 3]), ([2, 3], [0, 1]))]), device=DEV)
            X, D, y = T_.X, T_.D, T_.y
            pr = {key: predict(f, X, M, D, key[0]) for key, f in F.items()}
            Lc = M[:, 0].sum(1); cur = M[:, 2, 0]                              # own x observed count in window, other x(t) observed
            strata = {"all": torch.ones_like(y, dtype=torch.bool), "L0": Lc == 0, "L1": Lc == 1, "L2+": Lc >= 2,
                      "L0c1": (Lc == 0) & (cur > 0), "L0c0": (Lc == 0) & (cur == 0), "c0": cur == 0, "c1": cur > 0}
            Q = {"y2": y ** 2}
            for kind in KINDS:
                fb = (pr[(kind, "A")] + pr[(kind, "B")]) / 2; Q[f"risk_{kind}_avg"] = (y - fb) ** 2
                Q[f"risk_{kind}_A"] = (y - pr[(kind, "A")]) ** 2; Q[f"cert_{kind}"] = y ** 2 - (y - fb) ** 2
            for kind in ("UV", "UVc", "UVs"):
                qh = pr[(kind, "A")] - pr[("U", "A")]; qt = qh - predict(R[kind], X, M, D, "U"); eps = y - pr[("U", "B")]
                Q[f"T_{kind}"] = eps * qt; Q[f"Dq_{kind}"] = qt ** 2; Q[f"Traw_{kind}"] = eps * qh; Q[f"Dqraw_{kind}"] = qh ** 2
            for (kind, reg), f in AUD.items():
                Q[f"risk_{kind}_{reg}"] = (y - predict(f, X, M, D, kind)) ** 2
            Q["signerr_U"] = ((torch.sign(pr[("U", "A")]) != torch.sign(y)) & (y.abs() > 0.5)).float()
            Q["signerr_UV"] = ((torch.sign((pr[("UV", "A")] + pr[("UV", "B")]) / 2) != torch.sign(y)) & (y.abs() > 0.5)).float()
            Q["big"] = (y.abs() > 0.5).float()
            for sn, sel in strata.items():
                n = int(sel.sum())
                for qn, v in Q.items():
                    rows.append(dict(traj=q, mask=e, stratum=sn, quantity=qn, sum=float(v[sel].sum()), n=n))
    pd.DataFrame(rows).assign(kappa=k, p=p).to_csv(f"{O}/k{k}_p{p}.csv", index=False)
    json.dump(dict(kappa=k, p=p, wall_s=time.time() - t0, fits=info), open(f"{O}/k{k}_p{p}_train.json", "w"), indent=1)
    print("G1 DONE", k, p, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(float(sys.argv[1]), float(sys.argv[2]))
