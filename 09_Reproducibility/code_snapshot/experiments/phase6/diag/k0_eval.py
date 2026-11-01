"""POST-HOC DIAGNOSTIC evaluation: G2 y_j MLP probes (evaluate6.mlp_fit, width 64, seeds 0/1) for B_j and Sham_j, stratum all.
usage: k0_eval.py VERSION REP -> ROOT/eval_VERSION/r{REP}.csv"""
import os, sys
import numpy as np, pandas as pd, torch
from experiments.phase6.diag.k0_common import patch, ROOT, NT
v, r = sys.argv[1], int(sys.argv[2]); C = patch(v)
from experiments.phase5.core import ArmModel, encode_series, SPAN
from experiments.phase6.evaluate6 import mlp_fit, DEV
k, p = 0.0, 0.7; I = C.ids(r); _, _, mu_s, sd_s = C.norm_stats(k)
lab = lambda q: ((C.load(k, q) - mu_s) / sd_s).astype(np.float32)
Str, Sva, Ste = lab(I["train"]), lab(I["val"]), [lab(q) for q in I["test"]]
tr, va, te = np.arange(SPAN, 20000), np.arange(SPAN, 5000), np.arange(SPAN, 5000); rows = []
for arm in ("B1", "B2", "Sham1", "Sham2"):
    j = int(arm[-1]); yc = (1, 4)[j - 1]
    ck = torch.load(f"{ROOT}/runs_{v}/ckpt/k{k}_p{p}_r{r}_{arm}_s{6000 + r}.pt", map_location=DEV, weights_only=False)
    D = C.build(k, p, r, arm); m = ArmModel(len(D["ins"]), D["outs"], D["lagmask"]).to(DEV); m.load_state_dict(ck["state_dict"])
    enc = lambda X, M: encode_series(m, torch.tensor(X, device=DEV), torch.tensor(M, device=DEV))[0].cpu().numpy()
    Ftr, Fva = enc(D["Xtr"], D["Mtr"]), enc(D["Xva"], D["Mva"])
    for s in (0, 1):
        f = mlp_fit(Ftr[tr], Str[tr, yc], Fva[va], Sva[va, yc], s, 64)
        for i in range(NT):
            X, M = D["test"](i); e2 = (f(enc(X, M)[te]).reshape(-1) - Ste[i][te, yc]) ** 2
            rows.append(dict(version=v, rep=r, arm=arm, j=j, probe=f"mlp{s}", traj=I["test"][i], sse=float(e2.sum()), n=len(te)))
os.makedirs(f"{ROOT}/eval_{v}", exist_ok=True); pd.DataFrame(rows).to_csv(f"{ROOT}/eval_{v}/r{r}.csv", index=False); print("DIAG EVAL DONE", v, r)
