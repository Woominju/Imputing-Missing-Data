"""Phase 3A: evaluate the 4 pilot checkpoints (C3, C16, E3, E16; Lorenz p=0.7 seed 0) on the SAME test windows,
same mask draws (fixed generator seed) and same metrics as fiber_diag.py, plus
 - restricted-input RMSE: RMSE on the variables NOT observed in a pattern (lag 0), and on all 3 (state error),
 - mask-shift sensitivity: test RMSE when the evaluation mask rate p' differs from the training p (CPU, same model).
Latent distances are scaled within each checkpoint (median distance between random times, natural masks)."""
import numpy as np, pandas as pd, torch
from src.delay_embedding import delay_embed
from src.evaluate import impute
from src.models import build_model
from src.missingness import prepare, mcar_mask
torch.set_num_threads(8)
OUT = "results/extra/phase3a_decoder_diagnostics"; P, SEED = 0.7, 0
Xraw = np.load("results/data/lorenz_20000.npz")["X"]; d = prepare(Xraw, P, SEED); a, b = d["split"]["test"]
X = torch.tensor(d["X"][a:b]); m = 8; S = X[m - 1:]
g0 = torch.Generator().manual_seed(4242)
def draw(q): return (torch.rand(X.shape, generator=g0) < q).float()
NAT = {"NATe1": draw(1 - P), "NATe2": draw(1 - P), "NATt": draw((1 - P) * 0.8)}
PATS = {"x": [0], "y": [1], "z": [2], "xy": [0, 1], "xz": [0, 2], "yz": [1, 2], "xyz": [0, 1, 2]}
masks = {}
for nm, v in PATS.items():
    M_ = torch.zeros_like(X); M_[:, v] = 1; masks["ART:" + nm] = M_
for nm, M_ in NAT.items(): masks["NAT:" + nm] = M_
for nm in ["x", "xy", "xz"]:
    M_ = torch.zeros_like(X); M_[:, PATS[nm]] = NAT["NATe1"][:, PATS[nm]]; masks["NATV:" + nm] = M_
PAIRS = [("ART:x", "ART:xy"), ("ART:x", "ART:xz"), ("ART:y", "ART:xy"), ("ART:xy", "ART:xyz"), ("ART:xz", "ART:xyz"), ("ART:z", "ART:xyz"),
         ("NAT:NATe1", "NAT:NATe2"), ("NAT:NATe1", "NAT:NATt"), ("NATV:x", "NATV:xy"), ("NATV:xy", "NAT:NATe1"), ("NATV:x", "NAT:NATe1")]
TH = [dict(name="strict", e=0.10, R=0.10, z=0.30), dict(name="loose", e=0.25, R=0.25, z=0.50)]
rp, rq, rj, rs = [], [], [], []
for name, path in [("C3", "pilot/checkpoints/pilot_C3_p0.7_s0.pt"), ("C16", "pilot/checkpoints/pilot_C16_p0.7_s0.pt"),
                   ("E3", "pilot/checkpoints/pilot_E3_p0.7_s0.pt"), ("E16", "pilot/checkpoints/pilot_E16_p0.7_s0.pt")]:
    ck = torch.load(f"{OUT}/{path}", map_location="cpu", weights_only=False); c = ck["cfg"]
    mdl = build_model("C", 3, c["m"], c["latent"], c["hidden"], c["window"], c["direction"]); mdl.load_state_dict(ck["state_dict"]); mdl.eval()
    Z, R = {}, {}
    for k, M_ in masks.items():
        V, MV = delay_embed((X * M_)[None], M_[None], m, 1)
        with torch.no_grad(): Z[k] = mdl.encoder(V, MV)[0]; R[k] = mdl.decode(Z[k][None])[0]
    perm = torch.randperm(len(S), generator=torch.Generator().manual_seed(0))
    sc = float((Z["NAT:NATe1"] - Z["NAT:NATe1"][perm]).norm(dim=1).median())
    err = {k: (R[k] - S).norm(dim=1) for k in Z}
    ref = Z["NAT:NATt"][::3]
    for k, M_ in masks.items():
        miss0 = (1 - M_[m - 1:]).bool()                                  # lag-0 unobserved variables
        se = ((R[k] - S) ** 2)[miss0]
        rp.append(dict(model=name, pattern=k, state_err_median=float(err[k].median()),
                       rmse_unobserved_lag0=float(se.mean().sqrt()) if se.numel() else np.nan,
                       latent_nn_to_NATt=float(torch.cdist(Z[k][::5], ref).min(1).values.median() / sc)))
    for A_, B_ in PAIRS:
        dz = (Z[A_] - Z[B_]).norm(dim=1) / sc; dR = (R[A_] - R[B_]).norm(dim=1); eA, eB = err[A_], err[B_]
        rec = dict(model=name, pair=f"{A_}|{B_}", dz_scaled_median=float(dz.median()), dR_median=float(dR.median()),
                   eA=float(eA.median()), eB=float(eB.median()))
        for th in TH:
            acc = (eA <= th["e"]) & (eB <= th["e"]); difz = dz >= th["z"]
            rec[f"A_{th['name']}"] = float((acc & (dR <= th["R"]) & difz).float().mean())
            rec[f"B_{th['name']}"] = float((((eA > th["e"]) | (eB > th["e"])) & difz).float().mean())
            rec[f"C_{th['name']}"] = float((acc & ~difz).float().mean())
            rec[f"D_{th['name']}"] = float((eB > eA + th["e"]).float().mean()) if not A_.startswith("NAT:") else np.nan
        rq.append(rec)
    zs = torch.cat([Z["NAT:NATt"], Z["NAT:NATe1"]])
    J = torch.func.vmap(torch.func.jacrev(lambda zz: mdl.decode(zz[None, None])[0, 0]))(zs[::7]); sv = torch.linalg.svdvals(J)
    ev = torch.linalg.eigvalsh(torch.cov(zs.T)).flip(0); evr = (ev / ev.sum()).cumsum(0)
    sub = zs[torch.randperm(len(zs), generator=torch.Generator().manual_seed(1))[:3000]]
    D = torch.cdist(sub, sub); D.fill_diagonal_(float("inf")); r12 = D.topk(2, largest=False).values
    twonn = float(len(r12) / torch.log((r12[:, 1] / r12[:, 0]).clamp_min(1 + 1e-9)).sum())
    with torch.no_grad(): Rs = mdl.decode(sub[None])[0]
    DR = torch.cdist(Rs, Rs); DR.fill_diagonal_(float("inf")); nn = DR.argmin(1); dzz = (sub - sub[nn]).norm(dim=1) / sc; dRR = DR.min(1).values
    rj.append(dict(model=name, latent=c["latent"], sv_min_median=float(sv[:, -1].median()), sv_min_p05=float(sv[:, -1].quantile(.05)),
                   sv_max_median=float(sv[:, 0].median()), pca_dims_95=int((evr < .95).sum() + 1), twonn_dim=twonn,
                   collision_frac=float(((dRR < 0.05) & (dzz > 0.3)).float().mean()), latent_dist_of_output_nn_median=float(dzz.median())))
    # mask-shift sensitivity on the full series with the model's own impute()
    for pe in [0.5, 0.6, 0.7, 0.8]:
        rng = np.random.default_rng(777); M2 = d["M_orig"] * mcar_mask(d["X"].shape, pe, rng); Et = (d["M_orig"] * (1 - M2))[a:b]
        xh = impute(mdl, torch.tensor(d["X"]), torch.tensor(M2), m, 1).numpy()[a:b]
        rs.append(dict(model=name, eval_rate=pe, test_rmse=float(np.sqrt((((xh - d["X"][a:b]) * Et) ** 2).sum() / Et.sum()))))
    print(name, "done", flush=True)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
for nm, df in [("pilot_patterns", rp), ("pilot_pairs", rq), ("pilot_decoder", rj), ("pilot_maskshift", rs)]:
    df = pd.DataFrame(df); df.to_csv(f"{OUT}/{nm}.csv", index=False); print(f"\n## {nm}")
    key = {"pilot_patterns": "pattern", "pilot_pairs": "pair", "pilot_maskshift": "eval_rate"}.get(nm)
    print((df.set_index([key, "model"]).unstack("model").round(3) if key else df.round(4)).to_string())
