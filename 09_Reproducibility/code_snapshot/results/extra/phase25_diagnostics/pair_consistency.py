"""Phase 2.5 diagnostic (CPU, existing checkpoints, no training).
Prediction of P8' (scalar mask-count mean fusion, codes depend only on own view): latent disagreement between
patterns {x} and {x,z} is bounded below via the Gamma symmetry, while {x} vs {x,y} is not constrained.
For each test time t (fully observed windows, m=8, tau=1) we form inputs with views {x}, {x,y}, {x,z}, {x,y,z}
and report median ||zeta_P - zeta_Q|| / median ||zeta_full(t) - zeta_full(t')|| (random t'), plus decoder-output gaps.
Also checks the P8' bound with mirrored pairs: for t, find t* whose state is closest to Gamma s_t."""
import numpy as np, torch, pandas as pd
from src.delay_embedding import delay_embed
from src.models import build_model
from src.missingness import prepare
torch.set_num_threads(4)
Xraw = np.load("results/data/lorenz_20000.npz")["X"]
rows = []
for p in [0.1, 0.3, 0.7]:
    for seed in [0, 1, 2]:
        d = prepare(Xraw, p, seed); a, b = d["split"]["test"]
        ck = torch.load(f"results/checkpoints/lorenz_C_p{p}_s{seed}.pt", map_location="cpu", weights_only=False)
        c = ck["cfg"]; mdl = build_model("C", 3, c["m"], c["latent"], c["hidden"], c["window"], c["direction"])
        mdl.load_state_dict(ck["state_dict"]); mdl.eval()
        X = torch.tensor(d["X"][a:b])[None]; M1 = torch.ones_like(X)
        V, MV = delay_embed(X, M1, c["m"], c["tau"]); V, MV = V[0], MV[0]          # (K, 3, m)
        def enc(views):
            mv = MV.clone(); keep = torch.zeros(3); keep[views] = 1; mv = mv * keep[None, :, None]
            with torch.no_grad():
                z = mdl.encoder(V[None] * mv[None], mv[None])[0]; out = mdl.decode(z[None])[0]
            return z, out
        Z = {k: enc(v) for k, v in {"x": [0], "xy": [0, 1], "xz": [0, 2], "xyz": [0, 1, 2], "z": [2]}.items()}
        perm = torch.randperm(len(V), generator=torch.Generator().manual_seed(0))
        scale = (Z["xyz"][0] - Z["xyz"][0][perm]).norm(dim=1).median()
        r = {"p": p, "seed": seed, "time_scale": float(scale)}
        for P, Q in [("x", "xy"), ("x", "xz"), ("xy", "xyz"), ("xz", "xyz"), ("x", "xyz")]:
            r[f"lat {P}|{Q}"] = float((Z[P][0] - Z[Q][0]).norm(dim=1).median() / scale)
            r[f"out {P}|{Q}"] = float((Z[P][1] - Z[Q][1]).abs().mean())
        # P8' bound check with mirrored states: ||e_x(s)-e_x(Gs)|| where Gs ~ nearest test state to Gamma s
        S = torch.tensor(Xraw[a:b][c["m"] - 1:]); G = S * torch.tensor([-1., -1., 1.])
        dist = torch.cdist(G, S); j = dist.argmin(1); ok = dist.min(1).values < 0.5   # near-exact mirror partners
        ex = Z["x"][0]; ez = Z["z"][0]
        gap = (ex - ex[j]).norm(dim=1)[ok]
        bound = 0.5 * 0.5 * gap                     # w_z/(w_x+w_z) = 1/2 for full windows; max over pair >= gap/2
        dev = torch.maximum((Z["xz"][0] - ex).norm(dim=1), (Z["xz"][0][j] - ex[j]).norm(dim=1))[ok]
        r["n_mirror_pairs"] = int(ok.sum()); r["P8 bound median/scale"] = float(bound.median() / scale)
        r["observed max-dev median/scale"] = float(dev.median() / scale); r["bound violated (should be 0)"] = int((dev < bound - 1e-5).sum())
        rows.append(r)
df = pd.DataFrame(rows); pd.set_option("display.width", 250)
print(df.round(3).to_string(index=False)); df.to_csv("results/extra/phase25_diagnostics/pair_consistency.csv", index=False)
print("\nmean over seeds:"); print(df.groupby("p").mean(numeric_only=True).drop(columns="seed").round(3).T.to_string())
