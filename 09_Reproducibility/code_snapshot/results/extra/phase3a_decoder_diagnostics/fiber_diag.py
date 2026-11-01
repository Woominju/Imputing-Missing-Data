"""Phase 3A CPU diagnostics on EXISTING checkpoints (Model C, Model E Q=0.5). No training, no file in src/ touched.
Unit: test-split window at time t (m=8, tau=1, past-only), true state s_t = normalised (x,y,z)(t).
Patterns (all on the SAME t):
  ART  artificial view patterns, each kept view fully observed (8 lags): x, y, z, xy, xz, yz, xyz
  NAT  natural MCAR input masks drawn fresh on the test segment:
       NATe1, NATe2  : two independent draws at the evaluation rate p (entry observed w.p. 1-p)
       NATt          : one draw at the training-input rate (observed w.p. (1-p)(1-hide_rate))
       NATe1-1       : NATe1 with one extra observed entry removed (nested, B' subset of B)
  NATV natural mask NATe1 restricted to views (other views emptied): x|nat, xy|nat, xz|nat (view removal on natural mask)
Per pattern: e_B = ||R(z_B) - s_t||, n_obs per view, and the probability of that mask TYPE under the training
mask distribution of C (independent Bernoulli(q_tr)) and of E (view-subset dropout with prob Q).
Per pair: d_z/scale, d_R, cases A-D. Decoder Jacobian, latent intrinsic dimension, near-collisions."""
import json, os, numpy as np, pandas as pd, torch
from math import comb
from src.delay_embedding import delay_embed
from src.models import build_model
from src.missingness import prepare
torch.set_num_threads(8); torch.manual_seed(0)
OUT = "results/extra/phase3a_decoder_diagnostics"
Xraw = np.load("results/data/lorenz_20000.npz")["X"]
E_DIR = "results/extra/aim_autonomous_20261027/model_E_viewdrop/checkpoints"
TH = [dict(name="strict", e=0.10, R=0.10, z=0.30), dict(name="loose", e=0.25, R=0.25, z=0.50)]

def load(kind, p, s):
    path = f"results/checkpoints/lorenz_C_p{p}_s{s}.pt" if kind == "C" else f"{E_DIR}/lorenz_E_viewdrop_p{p}_s{s}.pt"
    ck = torch.load(path, map_location="cpu", weights_only=False); c = ck["cfg"]
    mdl = build_model("C", 3, c["m"], c["latent"], c["hidden"], c["window"], c["direction"]); mdl.load_state_dict(ck["state_dict"]); mdl.eval()
    return mdl, c

def train_type_prob(nobs, m, q, kind, Q=0.5):
    """P(view-count vector == nobs) for a window under the training input mask distribution."""
    base = np.prod([comb(m, k) * q ** k * (1 - q) ** (m - k) for k in nobs])
    if kind == "C":
        return base
    # E: w.p. 1-Q standard; w.p. Q keep 1 (1/2) or 2 (1/2) random views, dropped views empty
    n = len(nobs); pr = (1 - Q) * base
    from itertools import combinations
    for kk in (1, 2):
        for keep in combinations(range(n), kk):
            if all(nobs[j] == 0 for j in range(n) if j not in keep):
                pr += Q * 0.5 / comb(n, kk) * np.prod([comb(m, nobs[j]) * q ** nobs[j] * (1 - q) ** (m - nobs[j]) for j in keep])
    return pr

rows_pat, rows_pair, rows_jac, rows_coll = [], [], [], []
for kind in ["C", "E"]:
    for p in [0.3, 0.7]:
        for seed in [0, 1, 2]:
            mdl, c = load(kind, p, seed); m, hr = c["m"], c["hide_rate"]; q_tr = (1 - p) * (1 - hr)
            d = prepare(Xraw, p, seed); a, b = d["split"]["test"]
            X = torch.tensor(d["X"][a:b]); T = len(X); span = m - 1
            S = X[span:]                                            # true states for each window
            g = torch.Generator().manual_seed(1000 + seed)
            def draw(qobs): return (torch.rand(X.shape, generator=g) < qobs).float()
            Mn = {"NATe1": draw(1 - p), "NATe2": draw(1 - p), "NATt": draw(q_tr)}
            # nested: remove one extra observed entry inside each window's span -> do it per window on the windowed mask
            masks = {}
            ones = torch.ones_like(X)
            for nm, views in {"x": [0], "y": [1], "z": [2], "xy": [0, 1], "xz": [0, 2], "yz": [1, 2], "xyz": [0, 1, 2]}.items():
                Mv = torch.zeros_like(X); Mv[:, views] = 1; masks["ART:" + nm] = Mv
            for nm, M_ in Mn.items(): masks["NAT:" + nm] = M_
            for nm, views in {"x": [0], "xy": [0, 1], "xz": [0, 2]}.items():
                Mv = torch.zeros_like(X); Mv[:, views] = Mn["NATe1"][:, views]; masks["NATV:" + nm] = Mv
            Z, R, NO = {}, {}, {}
            for nm, M_ in masks.items():
                V, MV = delay_embed((X * M_)[None], M_[None], m, 1)
                with torch.no_grad():
                    z = mdl.encoder(V, MV)[0]; r = mdl.decode(z[None])[0]
                Z[nm], R[nm], NO[nm] = z, r, MV[0].sum(-1)            # NO: (K, 3) observed lags per view
            # nested pattern: drop one random observed entry of NATe1 per window (window-level edit)
            V, MV = delay_embed((X * Mn["NATe1"])[None], Mn["NATe1"][None], m, 1); V, MV = V[0].clone(), MV[0].clone()
            flat = MV.reshape(len(MV), -1); idx = torch.multinomial(flat + 1e-9, 1, generator=g).squeeze(1)
            flat[torch.arange(len(flat)), idx] = 0; MV2 = flat.reshape(MV.shape)
            with torch.no_grad():
                z = mdl.encoder((V * MV2)[None], MV2[None])[0]; r = mdl.decode(z[None])[0]
            Z["NAT:NATe1-1"], R["NAT:NATe1-1"], NO["NAT:NATe1-1"] = z, r, MV2.sum(-1)
            perm = torch.randperm(len(S), generator=g)
            scale = float((Z["ART:xyz"] - Z["ART:xyz"][perm]).norm(dim=1).median())
            scale_nat = float((Z["NAT:NATe1"] - Z["NAT:NATe1"][perm]).norm(dim=1).median())
            err = {k: (R[k] - S).norm(dim=1) for k in Z}
            for k in Z:
                no = NO[k].numpy().astype(int)
                types = pd.Series([tuple(r) for r in no]).value_counts()
                ptr = sum(cnt * train_type_prob(t, m, q_tr, kind) for t, cnt in types.items()) / len(no)
                rows_pat.append(dict(model=kind, p=p, seed=seed, pattern=k, e_median=float(err[k].median()), e_p90=float(err[k].quantile(.9)),
                                     nobs_x=float(no[:, 0].mean()), nobs_y=float(no[:, 1].mean()), nobs_z=float(no[:, 2].mean()),
                                     train_prob_of_type=ptr, latent_nn_dist_to_NATt=None))
            # OOD in latent space: distance to nearest training-like latent (NATt), normalised by scale_nat
            ref = Z["NAT:NATt"][::3]
            for r_ in rows_pat[-len(Z):]:
                k = r_["pattern"]; dd = torch.cdist(Z[k][::5], ref)
                if k == "NAT:NATt": dd = dd + torch.eye(*dd.shape) * 0 if False else dd
                r_["latent_nn_dist_to_NATt"] = float(dd.min(1).values.median() / scale_nat)
            pairs = [("ART:x", "ART:xy"), ("ART:x", "ART:xz"), ("ART:y", "ART:xy"), ("ART:y", "ART:yz"), ("ART:xy", "ART:xyz"),
                     ("ART:xz", "ART:xyz"), ("ART:z", "ART:xyz"), ("NAT:NATe1", "NAT:NATe2"), ("NAT:NATe1", "NAT:NATe1-1"),
                     ("NAT:NATe1", "NAT:NATt"), ("NATV:x", "NATV:xy"), ("NATV:xy", "NAT:NATe1"), ("NATV:x", "NAT:NATe1")]
            for P, Qn in pairs:
                dz = (Z[P] - Z[Qn]).norm(dim=1); dR = (R[P] - R[Qn]).norm(dim=1); eP, eQ = err[P], err[Qn]
                sc = scale_nat if P.startswith("NAT") and Qn.startswith("NAT") else scale
                rec = dict(model=kind, p=p, seed=seed, pair=f"{P}|{Qn}", dz_scaled_median=float((dz / sc).median()),
                           dR_median=float(dR.median()), eP_median=float(eP.median()), eQ_median=float(eQ.median()))
                for th in TH:
                    acc = (eP <= th["e"]) & (eQ <= th["e"]); simR = dR <= th["R"]; difz = dz / sc >= th["z"]
                    rec[f"A_{th['name']}"] = float((acc & simR & difz).float().mean())     # accurate+same output, latent differs
                    rec[f"B_{th['name']}"] = float((((eP > th["e"]) | (eQ > th["e"])) & difz).float().mean())
                    rec[f"C_{th['name']}"] = float((acc & ~difz).float().mean())
                    nested = Qn in ("ART:xy", "ART:xz", "ART:yz", "ART:xyz", "NAT:NATe1") and P not in ("NAT:NATe2", "NAT:NATt")
                    # D: more observations (Qn superset of P) but larger error
                    rec[f"D_{th['name']}"] = float((eQ > eP + th["e"]).float().mean()) if nested else np.nan
                rows_pair.append(rec)
            # --- decoder local structure on the in-distribution support (NATt and NATe1 latents) ---
            dec = lambda zz: mdl.decode(zz[None, None])[0, 0]
            zs = torch.cat([Z["NAT:NATt"], Z["NAT:NATe1"]])[::7]
            J = torch.func.vmap(torch.func.jacrev(dec))(zs)                      # (N, 3, 16)
            sv = torch.linalg.svdvals(J)
            zs_all = torch.cat([Z["NAT:NATt"], Z["NAT:NATe1"]])
            C_ = torch.cov(zs_all.T); ev = torch.linalg.eigvalsh(C_).flip(0); evr = (ev / ev.sum()).cumsum(0)
            # TwoNN intrinsic dimension (Facco et al. 2017) on the support
            sub = zs_all[torch.randperm(len(zs_all), generator=g)[:3000]]
            D = torch.cdist(sub, sub); D.fill_diagonal_(float("inf")); r12 = D.topk(2, largest=False).values
            mu = (r12[:, 1] / r12[:, 0]).clamp_min(1 + 1e-9); twonn = float(len(mu) / torch.log(mu).sum())
            Rs = mdl.decode(sub[None])[0].detach()
            subS = torch.cat([S, S])[torch.randperm(len(zs_all), generator=g)[:3000]] if False else None
            rows_jac.append(dict(model=kind, p=p, seed=seed, sv1_med=float(sv[:, 0].median()), sv2_med=float(sv[:, 1].median()),
                                 sv3_med=float(sv[:, 2].median()), sv3_p05=float(sv[:, 2].quantile(.05)),
                                 pca_dims_95=int((evr < 0.95).sum() + 1), pca_dims_99=int((evr < 0.99).sum() + 1), twonn_dim=twonn,
                                 latent_dim=int(zs.shape[1])))
            # --- near-collisions on the support: pairs with close decoder outputs but far latents ---
            DR = torch.cdist(Rs, Rs); DR.fill_diagonal_(float("inf")); nn = DR.argmin(1)
            dzz = (sub - sub[nn]).norm(dim=1) / scale_nat; dRR = DR.min(1).values
            # baseline: latent distance of nearest neighbours in latent space, for comparison
            rows_coll.append(dict(model=kind, p=p, seed=seed, nn_out_dist_median=float(dRR.median()),
                                  latent_dist_of_output_nn_median=float(dzz.median()), latent_dist_of_output_nn_p90=float(dzz.quantile(.9)),
                                  frac_collision=float(((dRR < 0.05) & (dzz > 0.3)).float().mean()),
                                  ratio_dz_over_dR_median=float((dzz * scale_nat / dRR.clamp_min(1e-6)).median()),
                                  inv_lip_lower_bound=float((dzz * scale_nat / dRR.clamp_min(1e-6)).max())))
            print(kind, p, seed, "done", flush=True)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
P_ = pd.DataFrame(rows_pat); Q_ = pd.DataFrame(rows_pair); J_ = pd.DataFrame(rows_jac); C2 = pd.DataFrame(rows_coll)
for nm, df in [("patterns", P_), ("pairs", Q_), ("decoder_jacobian", J_), ("near_collisions", C2)]:
    df.to_csv(f"{OUT}/{nm}.csv", index=False)
print("\n## patterns (mean over seeds)"); print(P_.groupby(["model", "p", "pattern"]).mean(numeric_only=True).drop(columns="seed").round(4).to_string())
print("\n## pairs (mean over seeds)"); print(Q_.groupby(["model", "p", "pair"]).mean(numeric_only=True).drop(columns="seed").round(3).to_string())
print("\n## decoder jacobian / dimension"); print(J_.groupby(["model", "p"]).mean(numeric_only=True).drop(columns="seed").round(4).to_string())
print("\n## near collisions"); print(C2.groupby(["model", "p"]).mean(numeric_only=True).drop(columns="seed").round(4).to_string())
