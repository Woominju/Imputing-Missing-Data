"""Phase 8 k-NN diagnostics (protocol v2 section 2.1 D5a / D5b, 4.2). Reference = N_REF i.i.d. windows from P (one point per
independent trajectory, gen8). Distances use ONLY the entries observed under the query's own mask (masked Euclid on the 32 window
entries); ties are broken by a seeded uniform perturbation of size 1e-9.
D5a  f_kNN: fixed function of W once the reference and k are fixed -> valid upper bound R*_A <= R_A(f_kNN) (D4); k chosen on
     oracle-pool val points from CONST.knn_k_grid; evaluated on test points n = SPAN, SPAN+KNN_STRIDE, ... of every replicate,
     all NMASK masks, together with O_raw on the SAME points.
D5b  1/2 * 1-NN residual-variance statistic within the reference (self excluded), per stratum A, averaged over D5B_MASKS masks
     drawn from P(M | A) by rejection. EXPLORATORY point estimate (consistency conditions not verified); never used for decisions.
usage: knn8.py {eval,d5b} --kappa K --p P"""
import argparse, os
import numpy as np, pandas as pd, torch
from scipy import stats
from experiments.phase5.core import raw_features, seed_of, SPAN, M_
from experiments.phase8.core8 import (EVAL, SZ, CONST, SMOKE, device, load, load_ref, nz_obs, nz_state, norm_stats, ids, pool_ids,
                                      pool_mask, test_mask, strata)
from experiments.phase8.oracle8 import load_oracle, opath

D5B_STRATA = ["all", "H", "one_L2+", "both_L2+", "X0"]


def reference(k, dev):
    S, _ = load_ref(k); N = len(S); _, _, mu_s, sd_s = norm_stats(k)
    RO = nz_obs(k, S.reshape(-1, 6)).reshape(N, M_, 4); RS = ((S[:, -1] - mu_s) / sd_s).astype(np.float32)
    return torch.tensor(RO.reshape(N, -1), device=dev), torch.tensor(RS, device=dev)


def windows(O4, M4, ts):
    idx = ts[:, None] + np.arange(-M_ + 1, 1)[None]                     # chronological rows n-7..n (same layout as the reference)
    return O4[idx].reshape(len(ts), -1).astype(np.float32), M4[idx].reshape(len(ts), -1).astype(np.float32)


def knn_dist(Q, QM, R, gen):
    d = (QM * Q ** 2).sum(1, keepdim=True) - 2 * (QM * Q) @ R.T + QM @ (R ** 2).T
    return d + 1e-9 * torch.rand(d.shape, generator=gen, device=d.device)


def knn_predict(Q, QM, R, RS, kmax_list, gen, chunk=1024):
    out = {kk: [] for kk in kmax_list}; K = max(kmax_list)
    for s in range(0, len(Q), chunk):
        d = knn_dist(Q[s:s + chunk], QM[s:s + chunk], R, gen); nn = torch.topk(d, min(K, R.shape[0]), largest=False).indices
        for kk in kmax_list:
            out[kk].append(RS[nn[:, :kk]].mean(1))
    return {kk: torch.cat(v).cpu().numpy() for kk, v in out.items()}


def run_eval(k, p, dev):
    R, RS = reference(k, dev); gen = torch.Generator(device=dev).manual_seed(seed_of("p8knn", k, p))
    # k selection on pool-val (mask draw d = 0), never on test
    Qs, QMs, Ys = [], [], []
    for q in pool_ids()["val"]:
        X6 = load(k, q); O = nz_obs(k, X6); M = pool_mask(k, p, "val", q, 0, len(O)); ts = np.arange(SPAN, len(O), SZ["KNN_STRIDE"])
        a, b = windows(O, M, ts); Qs.append(a); QMs.append(b); Ys.append(nz_state(k, X6)[ts])
    P = knn_predict(torch.tensor(np.concatenate(Qs), device=dev), torch.tensor(np.concatenate(QMs), device=dev), R, RS,
                    list(CONST["knn_k_grid"]), gen)
    Yv = np.concatenate(Ys); vr = {kk: float(((P[kk] - Yv) ** 2).mean()) for kk in P}; ksel = min(vr, key=vr.get)
    oraw, _ = load_oracle(opath("raw", k, p), dev); rows = []
    for r in range(SZ["R"]):
        I = ids(r)
        for i in range(SZ["NT"]):
            X6 = load(k, I["test"][i]); O = nz_obs(k, X6); S = nz_state(k, X6); ts = np.arange(SPAN, len(O), SZ["KNN_STRIDE"])
            for e in range(SZ["NMASK"]):
                M = test_mask(k, p, r, i, e); a, b = windows(O, M, ts)
                pk = knn_predict(torch.tensor(a, device=dev), torch.tensor(b, device=dev), R, RS, [ksel], gen)[ksel]
                po = oraw(raw_features(O, M, [0, 1, 2, 3])[ts]); G = {g: v[ts] for g, v in strata(M).items()}
                for nm, pr in (("kNN", pk), ("O_raw", po)):
                    E2 = (pr - S[ts]) ** 2
                    for g, sel in G.items():
                        for c in range(6):
                            rows.append(dict(pipeline=nm, rep=r, traj=I["test"][i], mask=e, stratum=g, coord=c, sse=float(E2[sel, c].sum()), n=int(sel.sum())))
    os.makedirs(EVAL, exist_ok=True)
    pd.DataFrame([dict(k=kk, val_mse=v) for kk, v in vr.items()]).to_csv(f"{EVAL}/knn_ksel_k{float(k)}_p{p}.csv", index=False)
    f = f"{EVAL}/knn_k{float(k)}_p{p}.csv"; pd.DataFrame(rows).assign(kappa=k, p=p, k_selected=ksel, smoke=SMOKE).to_csv(f + ".tmp", index=False); os.replace(f + ".tmp", f)


def in_stratum(L1, L2, A):
    a, b = L1 >= 2, L2 >= 2
    return {"all": True, "H": (not a) and (not b), "one_L2+": a ^ b, "both_L2+": a and b, "X0": L1 == 0 and L2 == 0}[A]


def sample_window_mask(p, A, rng):
    """EXACT draw from P(M_window | A) under MCAR(p) (same stratum definitions as core8.strata): (L1, L2) from the product of
    Binomial(8, 1-p) restricted to A and renormalised; given L_j, the observed x_j positions are uniform; z columns i.i.d."""
    pmf = stats.binom(M_, 1 - p).pmf(np.arange(M_ + 1)); cells = [(a, b) for a in range(M_ + 1) for b in range(M_ + 1) if in_stratum(a, b, A)]
    w = np.array([pmf[a] * pmf[b] for a, b in cells]); L1, L2 = cells[rng.choice(len(cells), p=w / w.sum())]
    m = (rng.random((M_, 4)) >= p).astype(np.float32)
    for col, L in ((0, L1), (2, L2)):
        m[:, col] = 0; m[rng.choice(M_, L, replace=False), col] = 1
    return m


def run_d5b(k, p, dev):
    R, RS = reference(k, dev); N = R.shape[0]; rows = []; gen = torch.Generator(device=dev).manual_seed(seed_of("p8d5b", k, p))
    rng = np.random.default_rng(seed_of("p8d5bmask", k, p))
    for A in D5B_STRATA:
        for got in range(SZ["D5B_MASKS"]):
            m = sample_window_mask(p, A, rng)
            QM = torch.tensor(np.tile(m.reshape(1, -1), (N, 1)), device=dev); acc = 0.0
            for s in range(0, N, 2048):
                d = knn_dist(R[s:s + 2048], QM[s:s + 2048], R, gen); d[torch.arange(d.shape[0]), torch.arange(s, s + d.shape[0])] = float("inf")
                nn = d.argmin(1); acc += float(((RS[s:s + 2048] - RS[nn]) ** 2).sum())
            rows.append(dict(stratum=A, mask_idx=got, n_obs=int(m.sum()), d5b=acc / (2 * N * 6)))
    os.makedirs(EVAL, exist_ok=True)
    f = f"{EVAL}/d5b_k{float(k)}_p{p}.csv"
    pd.DataFrame(rows).assign(kappa=k, p=p, label="EXPLORATORY: 1/2 1-NN residual-variance statistic; not a bound", smoke=SMOKE).to_csv(f + ".tmp", index=False)
    os.replace(f + ".tmp", f)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("what", choices=["eval", "d5b"]); ap.add_argument("--kappa", type=float, required=True)
    ap.add_argument("--p", type=float, required=True); a = ap.parse_args(); dev = device()
    (run_eval if a.what == "eval" else run_d5b)(a.kappa, a.p, dev); print("KNN DONE", a.what, a.kappa, a.p, flush=True)


if __name__ == "__main__":
    main()
