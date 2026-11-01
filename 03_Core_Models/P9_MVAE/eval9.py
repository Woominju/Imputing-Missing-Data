"""Phase 9 evaluation of one (stage, kappa, p, replicate) (protocol sections 3-5). Writes additive sums (analysis pools them):
  imp : method, view, traj, mask, stratum, sse, n      missing positions (M = 0, n >= SPAN): AE:P9, AE:Bfull, AE:Fact8, AE:Fact16, zero,
                                                       LOCF, O_imp (frozen Phase 8 oracle, diagnostic)
  acc : pipeline, traj, mask, stratum, coord, sse, n   R1 readout (P7 rule, core8.fit_mlp) on P9, Bfull, Fact8, Fact16; O_raw (frozen P8)
  ret : feat, traj, sum_nn, sum_rand, n_q, exact_hits, d_same, d_rand, + per query-stratum sums   cross-mask state retrieval (whitened)
  mv  : feat, rho_white (mask sensitivity in whitened latent), readout mask variance per pipeline
usage: eval9.py --stage S --kappa K --p P --rep R"""
import argparse, itertools, os, time
import numpy as np, pandas as pd, torch
from experiments.phase5.core import ArmModel, encode_series, raw_features, seed_of, SPAN, SUB
from experiments.phase8.evaluate8 import locf                                    # frozen P8 helper
from experiments.phase8.oracle8 import load_oracle                               # frozen P8 oracles (diagnostic only)
from experiments.phase9.core9 import EVAL, RUNS, SZ, SMOKE, device, ids, load, nz_obs, nz_state, mask_seeds, test_mask, strata, fit_mlp, init_seed
from experiments.phase5.core import mcar

FEATS = {"P9": [("P9", [0, 1, 2, 3])], "Bfull": [("Bfull", [0, 1, 2, 3])], "Fact8": [("A1_8", SUB[1]), ("A2_8", SUB[2])],
         "Fact16": [("A1_16", SUB[1]), ("A2_16", SUB[2])]}
P8ORC = "results/extra/phase8/oracle"


def load_arm(stage, k, p, r, arm, dev):
    ck = torch.load(f"{RUNS}/ckpt/{stage}_k{k}_p{p}_r{r}_{arm}_s{init_seed(arm, stage, r)}.pt", map_location=dev, weights_only=False)
    m = ArmModel(len(ck["ins"]), ck["outs"], np.ones((len(ck["ins"]), 8), np.float32), latent=ck["latent"]).to(dev); m.load_state_dict(ck["state_dict"]); m.eval()
    return m


def featmap(parts, O, M, dev):
    zs, y = [], {}
    for m, cols in parts:
        z, y0 = encode_series(m, torch.tensor(O[:, cols], device=dev), torch.tensor(M[:, cols], device=dev)); zs.append(z.cpu().numpy())
        for q, c in enumerate(cols):
            y[c] = y0[:, q].cpu().numpy()
    return np.concatenate(zs, 1), y


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stage"); ap.add_argument("--kappa", type=float); ap.add_argument("--p", type=float); ap.add_argument("--rep", type=int)
    a = ap.parse_args(); st, k, p, r = a.stage, a.kappa, a.p, a.rep; tag = f"{st}_k{k}_p{p}_r{r}"; dev = device(); os.makedirs(EVAL, exist_ok=True); t0 = time.time()
    I = ids(st, r); ms = mask_seeds(st, k, p, r); NT, NM = SZ["NT"], SZ["NMASK"]
    Otr, Str = nz_obs(k, load(k, I["train"])), nz_state(k, load(k, I["train"])); Ova, Sva = nz_obs(k, load(k, I["val"])), nz_state(k, load(k, I["val"]))
    TE = [(nz_obs(k, load(k, q)), nz_state(k, load(k, q))) for q in I["test"]]
    Mtr = mcar((len(Otr), 4), p, ms["train"]); Mva = mcar((len(Ova), 4), p, ms["val"])
    M4 = {(i, e): test_mask(st, k, p, r, i, e) for i in range(NT) for e in range(NM)}; G = {q: strata(m) for q, m in M4.items()}
    tr = lambda n: np.arange(SPAN, n)
    PARTS = {f: [(load_arm(st, k, p, r, arm, dev), cols) for arm, cols in comp] for f, comp in FEATS.items()}
    PRED, Z, Y0 = {}, {}, {}
    for f, parts in PARTS.items():
        Ftr, _ = featmap(parts, Otr, Mtr, dev); Fva, _ = featmap(parts, Ova, Mva, dev)
        Z[f], Y0[f] = {}, {}
        for q in M4:
            Z[f][q], Y0[f][q] = featmap(parts, TE[q[0]][0], M4[q], dev)
        rd = fit_mlp(Ftr[tr(len(Ftr))], Str[tr(len(Ftr))], Fva[tr(len(Fva))], Sva[tr(len(Fva))], 0, 128, dev)
        PRED[f"R1:{f}"] = {q: rd(Z[f][q]) for q in M4}
    o_raw, _ = load_oracle(f"{P8ORC}/raw_k{float(k)}_p{p}.pt", dev); o_imp, _ = load_oracle(f"{P8ORC}/imp_k{float(k)}_p{p}.pt", dev)
    PRED["O_raw"] = {q: o_raw(raw_features(TE[q[0]][0], M4[q], [0, 1, 2, 3])) for q in M4}
    ACC = []
    for pipe, P in PRED.items():
        for (i, e), Y in P.items():
            E2 = (Y - TE[i][1]) ** 2
            for g, sel in G[(i, e)].items():
                s_ = E2[sel].sum(0)
                for c in range(6):
                    ACC.append(dict(pipeline=pipe, traj=I["test"][i], mask=e, stratum=g, coord=c, sse=float(s_[c]), n=int(sel.sum())))
    IMP = []
    for (i, e), M in M4.items():
        O = TE[i][0]; meth = {"zero": np.zeros_like(O), "LOCF": locf(O, M), "O_imp": o_imp(raw_features(O, M, [0, 1, 2, 3]))}
        for f in FEATS:
            Yf = np.full_like(O, np.nan)
            for c, v in Y0[f][(i, e)].items():
                Yf[:, c] = v
            meth[f"AE:{f}"] = Yf
        for mt, Y in meth.items():
            for c in range(4):
                j = 1 if c in SUB[1] else 2; miss = (M[:, c] == 0); miss[:SPAN] = False
                for g in ("all", f"L{j}:L0", f"L{j}:L1", f"L{j}:L2+", "nobs<6", "nobs6-9", "nobs10-13", "nobs>=14"):
                    sel = miss & G[(i, e)][g]
                    IMP.append(dict(method=mt, view=c, traj=I["test"][i], mask=e, stratum=g, sse=float(((Y[sel, c] - O[sel, c]) ** 2).sum()), n=int(sel.sum())))
    # ---- cross-mask state retrieval in the WHITENED latent (protocol 3)
    RET, MV = [], []; rng = np.random.default_rng(seed_of("p9ret", st, k, p, r)); NQ = SZ["RET_Q"]; Tn = SZ["N_TEST"]
    for f in FEATS:
        allz = np.concatenate([Z[f][q][SPAN:] for q in M4]).astype(np.float64); mu = allz.mean(0); C = np.cov(allz, rowvar=False)
        ev, U = np.linalg.eigh(C); ev = np.maximum(ev, 1e-8 * ev.max()); Wh = U @ np.diag(ev ** -0.5) @ U.T
        wz = {q: ((Z[f][q] - mu) @ Wh).astype(np.float32) for q in M4}
        num = np.mean([((wz[(i, e1)] - wz[(i, e2)]) ** 2)[SPAN:].sum(1).mean() / 2 for i in range(NT) for e1, e2 in itertools.combinations(range(NM), 2)])
        MV.append(dict(feat=f, rho_white=num / wz[(0, 0)].shape[1]))
        for i in range(NT):
            S = TE[i][1]; qt = rng.choice(np.arange(SPAN, Tn), NQ, replace=False); DBt = np.tile(np.arange(SPAN, Tn), NM - 1)
            DB = torch.tensor(np.concatenate([wz[(i, e)][SPAN:] for e in range(1, NM)]), device=dev); Q = torch.tensor(wz[(i, 0)][qt], device=dev)
            nn = torch.topk(torch.cdist(Q, DB), 10, largest=False).indices.cpu().numpy(); rnd = rng.integers(0, len(DBt), (NQ, 10))
            dn = np.linalg.norm(S[DBt[nn]] - S[qt][:, None], axis=2); dr = np.linalg.norm(S[DBt[rnd]] - S[qt][:, None], axis=2)
            same = np.stack([wz[(i, e)][qt] for e in range(1, NM)]); d_same = np.linalg.norm(same - wz[(i, 0)][qt][None], axis=2).mean()
            d_rand = np.linalg.norm(DB[rnd[:, 0]].cpu().numpy() - wz[(i, 0)][qt], axis=1).mean()
            row = dict(feat=f, traj=I["test"][i], sum_nn=float(dn.sum()), sum_rand=float(dr.sum()), n_q=NQ, exact_hits=int((DBt[nn] == qt[:, None]).sum()),
                       d_same=float(d_same), d_rand=float(d_rand))
            for g in ("H", "X0", "both_L2+", "one_L2+"):
                sel = G[(i, 0)][g][qt]; row[f"nn_{g}"] = float(dn[sel].sum()); row[f"rand_{g}"] = float(dr[sel].sum()); row[f"nq_{g}"] = int(sel.sum())
            RET.append(row)
    for pipe, P in PRED.items():
        v = np.mean([((P[(i, e1)] - P[(i, e2)]) ** 2)[SPAN:].mean() / 2 for i in range(NT) for e1, e2 in itertools.combinations(range(NM), 2)])
        MV.append(dict(feat=pipe, readout_mask_var=float(v)))
    meta = dict(stage=st, kappa=k, p=p, rep=r, smoke=SMOKE)
    for nm, rows in (("imp", IMP), ("ret", RET), ("mv", MV), ("acc", ACC)):                 # acc = marker, written LAST
        f = f"{EVAL}/{tag}_{nm}.csv.gz"; pd.DataFrame(rows).assign(**meta).to_csv(f + ".tmp.gz", index=False); os.replace(f + ".tmp.gz", f)
    print("P9 EVAL DONE", tag, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
