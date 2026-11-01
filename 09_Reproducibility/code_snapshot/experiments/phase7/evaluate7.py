"""P7-B evaluation of one (kappa, p, rep) on GPU (protocol results/notes/phase7_p7b_protocol.md, fixed before any test result).
ONE shared nonlinear readout per feature set: MLP 2 x 128 ReLU (experiments.phase6.evaluate6.mlp_fit: Adam 1e-3, batch 512,
<= 200 epochs, patience 20 on the replicate's val trajectory, seed 0) from the frozen feature to the full 6-D normalised state,
fit on the replicate's train trajectory under its natural train mask; evaluated on its 8 independent test trajectories x 5
independent masks (e = 0..4). Features: Bself1, Bself2, Bfull (frozen AE latents) and raw_joint (raw masked 4-view delay
window) as a reference. Hidden states are used only as readout targets and for oracle-only strata.
Outputs (additive sums) results/extra/phase7/eval/k{K}_p{P}_r{R}_{acc,pair}.csv"""
import argparse, itertools, os, time
import numpy as np, pandas as pd, torch
from experiments.phase5.core import ArmModel, encode_series, raw_features, SPAN
from experiments.phase6.evaluate6 import mlp_fit, own_counts, lab, joint_lab, sig_min, SIG_BINS, SIG_NAMES, DEV
from experiments.phase7.core7 import build, load, ids, norm_stats, test_mask, ARMS7, NT7, NMASK, ROOT, init_seed

E = f"{ROOT}/eval"; RUNS = f"{ROOT}/runs"; JAC = f"{ROOT}/jac"
COORDS = (("s6", slice(0, 6), 6), ("s1", slice(0, 3), 3), ("s2", slice(3, 6), 3), ("y1", slice(1, 2), 1), ("y2", slice(4, 5), 1))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--kappa", type=float); ap.add_argument("--p", type=float); ap.add_argument("--rep", type=int)
    a = ap.parse_args(); k, p, r = a.kappa, a.p, a.rep; tag = f"k{k}_p{p}_r{r}"; os.makedirs(E, exist_ok=True); torch.set_num_threads(2); t0 = time.time()
    I = ids(r); _, _, mu_s, sd_s = norm_stats(k); labS = lambda q: ((load(k, q) - mu_s) / sd_s).astype(np.float32)
    Str, Sva = labS(I["train"]), labS(I["val"]); Ste = [labS(q) for q in I["test"]]
    tr, va, te = np.arange(SPAN, len(Str)), np.arange(SPAN, len(Sva)), np.arange(SPAN, 5000)
    M4 = {(i, e): test_mask(k, p, r, i, e) for i in range(NT7) for e in range(NMASK)}; CNT = {q: own_counts(m) for q, m in M4.items()}
    D = build(k, p, r, "Bfull"); feats = {}
    for arm in ARMS7:
        ck = torch.load(f"{RUNS}/ckpt/k{k}_p{p}_r{r}_{arm}_s{init_seed(r)}.pt", map_location=DEV, weights_only=False)
        Da = build(k, p, r, arm); assert ck["loss_cols"] == Da["loss_cols"]
        mdl = ArmModel(4, Da["outs"], Da["lagmask"]).to(DEV); mdl.load_state_dict(ck["state_dict"])
        enc = lambda X, M: encode_series(mdl, torch.tensor(X, device=DEV), torch.tensor(M, device=DEV))[0].cpu().numpy()
        feats[arm] = {"train": enc(Da["Xtr"], Da["Mtr"]), "val": enc(Da["Xva"], Da["Mva"]),
                      "test": {(i, e): enc(Da["test"](i)[0], M4[(i, e)]) for i in range(NT7) for e in range(NMASK)}}
    feats["raw_joint"] = {"train": raw_features(D["Xtr"], D["Mtr"], [0, 1, 2, 3]), "val": raw_features(D["Xva"], D["Mva"], [0, 1, 2, 3]),
                          "test": {(i, e): raw_features(D["test"](i)[0], M4[(i, e)], [0, 1, 2, 3]) for i in range(NT7) for e in range(NMASK)}}
    jf = f"{JAC}/k{float(k)}_r{r}.npz"; JZ = np.load(jf) if os.path.exists(jf) else None
    ACC, PAIR = [], []
    for fn, F in feats.items():
        rd = mlp_fit(F["train"][tr], Str[tr], F["val"][va], Sva[va], 0, 128)
        for i in range(NT7):
            S = Ste[i][te]; P = np.stack([rd(F["test"][(i, e)][te]) for e in range(NMASK)], 1); ER = (P - S[:, None]) ** 2
            if JZ is not None:
                sj = JZ["traj"] == i; tj, Jj = JZ["t"][sj], JZ["J"][sj]
            for e in range(NMASK):
                L, _ = CNT[(i, e)]; L1, L2 = L[1][te], L[2][te]; jl = joint_lab(L1, L2)
                G = {"all": np.ones(len(te), bool), "both_L2+": jl == "both_L2+", "one_L2+": jl == "one_L2+", "none_L2+": jl == "none_L2+"}
                for jj, LL in ((1, L1), (2, L2)):
                    for nm in ("L0", "L1", "L2+"):
                        G[f"L{jj}:{nm}"] = lab(LL) == nm
                for g, sel in G.items():
                    for cn, cs, _ in COORDS:
                        ACC.append(dict(feature=fn, traj=I["test"][i], mask=e, stratum=g, coord=cn, sse=float(ER[sel, e, cs].sum()), n=int(sel.sum())))
                if JZ is not None:
                    sm = sig_min(Jj, M4[(i, e)], tj, [0, 1, 2, 3]); idx = tj - SPAN
                    for b, nm in enumerate(SIG_NAMES):
                        sel = (sm >= SIG_BINS[b]) & (sm < SIG_BINS[b + 1])
                        for cn, cs, _ in COORDS:
                            ACC.append(dict(feature=fn, traj=I["test"][i], mask=e, stratum=f"oracle:{nm}", coord=cn, sse=float(ER[idx[sel], e, cs].sum()), n=int(sel.sum())))
            JL = np.stack([joint_lab(CNT[(i, e)][0][1][te], CNT[(i, e)][0][2][te]) for e in range(NMASK)], 1)
            for e1, e2 in itertools.combinations(range(NMASK), 2):
                d2 = (P[:, e1] - P[:, e2]) ** 2; same = JL[:, e1] == JL[:, e2]
                for g in ("all", "both_L2+", "one_L2+", "none_L2+"):
                    sel = np.ones(len(te), bool) if g == "all" else same & (JL[:, e1] == g)
                    for cn, cs, _ in COORDS:
                        PAIR.append(dict(feature=fn, traj=I["test"][i], pair=f"{e1}{e2}", stratum=g, coord=cn, sse=float(d2[sel, cs].sum()), n=int(sel.sum())))
    for nm, rows in (("acc", ACC), ("pair", PAIR)):
        pd.DataFrame(rows).assign(kappa=k, p=p, rep=r).to_csv(f"{E}/{tag}_{nm}.csv", index=False)
    print("P7 EVAL DONE", tag, f"{time.time() - t0:.0f}s", "jac" if JZ is not None else "nojac", flush=True)


if __name__ == "__main__":
    main()
