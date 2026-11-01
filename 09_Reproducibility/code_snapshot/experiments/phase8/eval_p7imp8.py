"""Phase 8 Family III on EXISTING Phase 7 checkpoints (protocol v2 section 6; Phase 7 post-hoc evaluation fixed in the Phase 8
protocol before any result). No training. Missing-position (M = 0, n >= SPAN) past-only lag-0 imputation RMSE on the P7 test
trajectories and P7 test masks e = 0..4: P7 Bfull (4 views), P7 Bself_j (OWN views only: the inactive heads were never trained),
baselines zero and LOCF. Phase 7 files are read only. Output: results/extra/phase8/eval/p7imp_k{K}_p{P}_r{R}.csv.gz
usage: eval_p7imp8.py --kappa K --p P --rep R"""
import argparse, os
import numpy as np, pandas as pd, torch
from experiments.phase5.core import ArmModel, encode_series, SPAN, SUB
import experiments.phase7.core7 as C7
from experiments.phase8.core8 import EVAL, SMOKE, device, strata
from experiments.phase8.evaluate8 import locf

P7_ARMS = {"P7:Bfull": [0, 1, 2, 3], "P7:Bself1": SUB[1], "P7:Bself2": SUB[2]}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--kappa", type=float); ap.add_argument("--p", type=float); ap.add_argument("--rep", type=int)
    a = ap.parse_args(); k, p, r = a.kappa, a.p, a.rep; dev = device(); os.makedirs(EVAL, exist_ok=True); rows = []
    models = {}
    for nm in P7_ARMS:
        arm = nm.split(":")[1]; ck = torch.load(f"{C7.ROOT}/runs/ckpt/k{k}_p{p}_r{r}_{arm}_s{C7.init_seed(r)}.pt", map_location=dev, weights_only=False)
        D = C7.build(k, p, r, arm); assert ck["loss_cols"] == D["loss_cols"]
        m = ArmModel(4, D["outs"], D["lagmask"]).to(dev); m.load_state_dict(ck["state_dict"]); m.eval(); models[nm] = (m, D)
    for i in range(C7.NT7):
        for e in range(C7.NMASK):
            O, M = models["P7:Bfull"][1]["test"](i, e); G = strata(M)
            meth = {"zero": np.zeros_like(O), "LOCF": locf(O, M)}
            for nm, (m, D) in models.items():
                _, y0 = encode_series(m, torch.tensor(O, device=dev), torch.tensor(M, device=dev)); y0 = y0.cpu().numpy()
                Y = np.full_like(O, np.nan); Y[:, P7_ARMS[nm]] = y0[:, P7_ARMS[nm]]; meth[nm] = Y
            for mt, Y in meth.items():
                for c in range(4):
                    if np.isnan(Y[:, c]).all():
                        continue
                    j = 1 if c in SUB[1] else 2; miss = (M[:, c] == 0); miss[:SPAN] = False
                    for g in ("all", f"L{j}:L0", f"L{j}:L1", f"L{j}:L2+", "nobs<6", "nobs6-9", "nobs10-13", "nobs>=14"):
                        sel = miss & G[g]
                        rows.append(dict(method=mt, view=c, traj=C7.ids(r)["test"][i], mask=e, stratum=g, sse=float(((Y[sel, c] - O[sel, c]) ** 2).sum()), n=int(sel.sum())))
    f = f"{EVAL}/p7imp_k{k}_p{p}_r{r}.csv.gz"; pd.DataFrame(rows).assign(kappa=k, p=p, rep=r, smoke=SMOKE).to_csv(f + ".tmp.gz", index=False); os.replace(f + ".tmp.gz", f)
    print("P7IMP DONE", k, p, r, flush=True)


if __name__ == "__main__":
    main()
