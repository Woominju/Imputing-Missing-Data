"""Phase 8 evaluation of one (kappa, p, replicate) (protocol v2 sections 4.4, 5.3, 6, 7). Fixed before any Phase 8 result.
Pipelines (normalised 6-D state; sham pipelines target S^sharp = (real s_j, donor s_o), P8.3 iv):
  R1:<F>   P7 readout rule (MLP 2x128, seed 0, replicate train trajectory, val early stop) on feature set F in feats8.FEATS
  R0:<F>   ridge, lambda on val (descriptive), F in {Bfull, Fact8, Fact16, raw_joint}
  O_raw, O_own1, O_own2   fixed pool oracles;  O_lat:Bfull, O_lat:Fact8   pool oracle on this replicate's frozen latent
  sym0     raw-zero predictor (-mu/sigma): gives the symmetric Bayes predictor on N1 coordinates (LB, T_A, P8.4)
Outputs (additive sums; analysis pools them):
  acc  : pipeline, traj, mask, stratum, coord(0..5), sse, n          (all core8.strata)
  pair : pipeline, traj, pair, coord, sse, n                        (stratum all; section 2.3 decomposition)
  imp  : method, view, traj, mask, stratum, sse, n                  (missing positions only, n >= SPAN; Family III)
  lat  : feat, traj, rho numerator/denominator pieces, latent-NN state distance vs random   (Family IV)
usage: evaluate8.py --kappa K --p P --rep R"""
import argparse, itertools, os, time
import numpy as np, pandas as pd, torch
from experiments.phase5.core import raw_features, seed_of, SPAN, SUB, mcar
from experiments.phase8.core8 import (EVAL, SZ, SMOKE, device, ids, load, nz_obs, nz_state, mask_seeds, test_mask, strata, sym0_pred,
                                      fit_mlp, fit_ridge)
from experiments.phase8.feats8 import FEATS, SHAM_J, FeatureMap
from experiments.phase8.oracle8 import load_oracle, opath

R0_FEATS = ["Bfull", "Fact8", "Fact16", "raw_joint"]; PAIR_PIPES = ["R1:Bfull", "R1:Fact8", "O_lat:Bfull", "O_raw", "R1:raw_joint"]
LAT_FEATS = ["Bfull", "Fact8", "Fact16"]; AE_IMP = {"Bfull": [0, 1, 2, 3], "BfullSham1": SUB[1], "BfullSham2": SUB[2], "Fact8": [0, 1, 2, 3],
                                                    "Fact16": [0, 1, 2, 3]}


def composite(k, real_id, donor_id, sham_j):
    """canonical O4 (normalised) and S target; sham_j: subsystem 3-sham_j replaced by the donor's (inputs and target)."""
    Xr = load(k, real_id); O = nz_obs(k, Xr); S = nz_state(k, Xr)
    if sham_j:
        o = 3 - sham_j; Xd = load(k, donor_id)[:len(Xr)]; Od, Sd = nz_obs(k, Xd), nz_state(k, Xd); sc = [0, 1, 2] if o == 1 else [3, 4, 5]
        O = O.copy(); S = S.copy(); O[:, SUB[o]] = Od[:, SUB[o]]; S[:, sc] = Sd[:, sc]
    return O, S


def locf(O, M):
    """past-only last observation carried forward for every column (0 = normalised mean if never observed)."""
    out = np.zeros_like(O)
    for c in range(O.shape[1]):
        obs = M[:, c] > 0; idx = np.where(obs, np.arange(len(O)), -1); last = np.maximum.accumulate(idx)
        prev = np.r_[-1, last[:-1]]; out[:, c] = np.where(prev >= 0, O[np.maximum(prev, 0), c], 0.0)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--kappa", type=float); ap.add_argument("--p", type=float); ap.add_argument("--rep", type=int)
    a = ap.parse_args(); k, p, r = a.kappa, a.p, a.rep; tag = f"k{k}_p{p}_r{r}"; dev = device(); os.makedirs(EVAL, exist_ok=True); t0 = time.time()
    I = ids(r); ms = mask_seeds(k, p, r); NT, NM = SZ["NT"], SZ["NMASK"]
    tr = lambda n: np.arange(SPAN, n)
    data = {}
    for sj in (None, 1, 2):
        Otr, Str = composite(k, I["train"], I["dtrain"], sj); Ova, Sva = composite(k, I["val"], I["dval"], sj)
        te = [composite(k, I["test"][i], I["dtest"][i], sj) for i in range(NT)]
        data[sj] = dict(Otr=Otr, Str=Str, Ova=Ova, Sva=Sva, te=te)
    Mtr = mcar((len(data[None]["Otr"]), 4), p, ms["train"]); Mva = mcar((len(data[None]["Ova"]), 4), p, ms["val"])
    M4 = {(i, e): test_mask(k, p, r, i, e) for i in range(NT) for e in range(NM)}; G = {q: strata(m) for q, m in M4.items()}
    PRED, FEAT_TEST = {}, {}
    for fn in FEATS:
        sj = SHAM_J.get(fn); D = data[sj]; fm = FeatureMap(k, p, r, fn, dev)
        Ftr, Fva = fm(D["Otr"], Mtr), fm(D["Ova"], Mva); Fte = {q: fm(D["te"][q[0]][0], M4[q]) for q in M4}; FEAT_TEST[fn] = Fte
        rd = fit_mlp(Ftr[tr(len(Ftr))], D["Str"][tr(len(Ftr))], Fva[tr(len(Fva))], D["Sva"][tr(len(Fva))], 0, 128, dev)
        PRED[f"R1:{fn}"] = {q: rd(Fte[q]) for q in M4}
        if fn in R0_FEATS:
            rr = fit_ridge(Ftr[tr(len(Ftr))], D["Str"][tr(len(Ftr))], Fva[tr(len(Fva))], D["Sva"][tr(len(Fva))]); PRED[f"R0:{fn}"] = {q: rr(Fte[q]) for q in M4}
        if fn in ("Bfull", "Fact8"):
            ol, _ = load_oracle(opath("lat", k, p, r, fn), dev); PRED[f"O_lat:{fn}"] = {q: ol(Fte[q]) for q in M4}
        if fn in AE_IMP:
            FEAT_TEST[fn + "#lag0"] = {q: fm.lag0(D["te"][q[0]][0], M4[q]) for q in M4}
    for nm, cols in (("O_raw", [0, 1, 2, 3]), ("O_own1", SUB[1]), ("O_own2", SUB[2])):
        o, _ = load_oracle(opath(nm[2:] if nm != "O_raw" else "raw", k, p), dev)
        PRED[nm] = {q: o(raw_features(data[None]["te"][q[0]][0], M4[q], cols)) for q in M4}
    PRED["sym0"] = {q: sym0_pred(k, SZ["N_TEST"]) for q in M4}
    # ---- acc / pair
    ACC, PAIR = [], []
    for pipe, P in PRED.items():
        sj = SHAM_J.get(pipe.split(":")[-1]) if pipe.startswith(("R1:", "R0:")) else None
        for i in range(NT):
            S = data[sj]["te"][i][1]
            for e in range(NM):
                E2 = (P[(i, e)] - S) ** 2
                for g, sel in G[(i, e)].items():
                    s_ = E2[sel].sum(0)
                    for c in range(6):
                        ACC.append(dict(pipeline=pipe, traj=I["test"][i], mask=e, stratum=g, coord=c, sse=float(s_[c]), n=int(sel.sum())))
            if pipe in PAIR_PIPES:
                for e1, e2 in itertools.combinations(range(NM), 2):
                    d2 = ((P[(i, e1)] - P[(i, e2)]) ** 2)[SPAN:].sum(0)
                    for c in range(6):
                        PAIR.append(dict(pipeline=pipe, traj=I["test"][i], pair=f"{e1}{e2}", coord=c, sse=float(d2[c]), n=SZ["N_TEST"] - SPAN))
    # ---- imputation (Family III): AE lag-0 outputs, zero, LOCF, O_imp at currently missing entries
    IMP = []; oi, _ = load_oracle(opath("imp", k, p), dev)
    for i in range(NT):
        O = data[None]["te"][i][0]
        for e in range(NM):
            M = M4[(i, e)]; base = {"zero": np.zeros_like(O), "LOCF": locf(O, M), "O_imp": oi(raw_features(O, M, [0, 1, 2, 3]))}
            for fn, cols in AE_IMP.items():
                L0 = FEAT_TEST[fn + "#lag0"][(i, e)]
                for c in cols:
                    base.setdefault(f"AE:{fn}", np.full_like(O, np.nan))[:, c] = L0[c]
            for meth, Y in base.items():
                Oref = O
                for c in range(4):
                    if np.isnan(Y[:, c]).all():
                        continue
                    j = 1 if c in SUB[1] else 2; miss = (M[:, c] == 0); miss[:SPAN] = False
                    for g in ("all", f"L{j}:L0", f"L{j}:L1", f"L{j}:L2+", "nobs<6", "nobs6-9", "nobs10-13", "nobs>=14"):
                        sel = miss & G[(i, e)][g]
                        IMP.append(dict(method=meth, view=c, traj=I["test"][i], mask=e, stratum=g, sse=float(((Y[sel, c] - Oref[sel, c]) ** 2).sum()), n=int(sel.sum())))
    # ---- readout-free latent diagnostics (Family IV)
    LAT = []; rng = np.random.default_rng(seed_of("p8latnn", k, p, r))
    for fn in LAT_FEATS:
        F = FEAT_TEST[fn]; allz = np.concatenate([F[q][SPAN:] for q in F]); trc = float(allz.var(0).sum()); sdz = allz.std(0) + 1e-8
        for i in range(NT):
            num = np.mean([((F[(i, e1)] - F[(i, e2)]) ** 2)[SPAN:].sum(1).mean() / 2 for e1, e2 in itertools.combinations(range(NM), 2)])
            row = dict(feat=fn, traj=I["test"][i], rho_num=float(num), rho_den=trc)
            if i < 2:
                S = data[None]["te"][i][1]; qt = rng.choice(np.arange(SPAN, SZ["N_TEST"]), SZ["LATNN_Q"], replace=False)
                DB = np.concatenate([F[(i, e)][SPAN:] for e in range(1, NM)]) / sdz; DBt = np.tile(np.arange(SPAN, SZ["N_TEST"]), NM - 1)
                Q = torch.tensor(F[(i, 0)][qt] / sdz, device=dev); Dt = torch.tensor(DB, device=dev)
                nn = torch.topk(torch.cdist(Q, Dt), 10, largest=False).indices.cpu().numpy()
                dnn = np.linalg.norm(S[DBt[nn]] - S[qt][:, None], axis=2).mean(); rnd = DBt[rng.integers(0, len(DBt), (len(qt), 10))]
                row.update(nn_state_dist=float(dnn), rand_state_dist=float(np.linalg.norm(S[rnd] - S[qt][:, None], axis=2).mean()))
            LAT.append(row)
    meta = dict(kappa=k, p=p, rep=r, smoke=SMOKE)
    for nm, rows in (("pair", PAIR), ("imp", IMP), ("lat", LAT), ("acc", ACC)):          # acc = scheduler marker, written LAST
        f = f"{EVAL}/{tag}_{nm}.csv.gz"; pd.DataFrame(rows).assign(**meta).to_csv(f + ".tmp.gz", index=False); os.replace(f + ".tmp.gz", f)
    print("P8 EVAL DONE", tag, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
