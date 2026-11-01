"""Phase 9 training of one AE arm. The loop is experiments/phase8/train8.py (= phase7 train7) line for line; the ONLY addition is
the tower term for arm P9 (design B2-B3): L = FDR(student; hidden observed entries H) + lam * L_tow(student vs no-grad teacher on
the superset input O, at originally missing entries 1 - O). For every other arm (and for lam = 0) the loss is bitwise the Phase 8
loss (tested). Rows record the mean L_tow per epoch and the number of teacher target entries.
usage: train9.py --stage {pilot,main} --kappa K --p P --rep R --arm ARM"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd, torch
from experiments.phase5.core import ArmModel, encode_series, SPAN
from experiments.phase9.core9 import build, code_md5, init_seed, device, AE_CFG, RUNS, SMOKE, ARMS9, CONST, tower_loss
from experiments.fdr.loss import fdr_masked_loss
from src.delay_embedding import delay_embed


def p9_loss(model, Xs, Ms, H, lc, cfg, lam, tower):
    """returns (total loss or None, L_tow value or nan, #teacher target entries)."""
    Bm = Ms - H
    V, MV = delay_embed(Xs * Bm, Bm, cfg["m"], cfg["tau"])                              # masking BEFORE windowing (as Phase 8)
    Ys = model.forward_full(V, MV)
    loss, _ = fdr_masked_loss(Ys[..., lc, :], Xs[..., lc], H[..., lc], cfg["m"], cfg["tau"])
    if not tower or lam == 0 or loss is None:
        return loss, float("nan"), 0
    with torch.no_grad():                                                               # teacher: same network, superset input O
        Vt, MVt = delay_embed(Xs * Ms, Ms, cfg["m"], cfg["tau"]); Yt = model.forward_full(Vt, MVt)
    D = 1.0 - Ms                                                                        # originally missing entries only
    lt, nt = tower_loss(Ys[..., lc, :], Yt[..., lc, :], D[..., lc], cfg["m"], cfg["tau"])
    if lt is None:
        return loss, float("nan"), 0
    return loss + lam * lt, float(lt.item()), int(nt.item())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", choices=["pilot", "main"], required=True)
    ap.add_argument("--kappa", type=float, required=True); ap.add_argument("--p", type=float, required=True)
    ap.add_argument("--rep", type=int, required=True); ap.add_argument("--arm", required=True, choices=ARMS9); a = ap.parse_args()
    CFG = AE_CFG; seed = init_seed(a.arm, a.stage, a.rep); eid = f"{a.stage}_k{a.kappa}_p{a.p}_r{a.rep}_{a.arm}_s{seed}"
    for d in ("ckpt", "logs", "rows", "failed"):
        os.makedirs(os.path.join(RUNS, d), exist_ok=True)
    dev = device(); torch.manual_seed(seed); np.random.seed(seed)
    D = build(a.stage, a.kappa, a.p, a.rep, a.arm); lam = CONST["lam"] if D["tower"] else 0.0
    model = ArmModel(len(D["ins"]), D["outs"], D["lagmask"], CFG["m"], D["latent"], CFG["hidden"]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=CFG["lr"])
    gen = torch.Generator(device=dev).manual_seed(seed + 1000)
    X = torch.tensor(D["Xtr"], device=dev); Mm = torch.tensor(D["Mtr"], device=dev)
    Xv = torch.tensor(D["Xva"], device=dev); Bv = torch.tensor(D["Mva"] - D["Hva"], device=dev); lc = D["loss_cols"]; Hv = D["Hva"][:, lc]
    W = SPAN + CFG["targets"]; T = X.shape[0]
    log, best, best_state, bad, nonfinite = [], np.inf, None, 0, 0; t0 = time.time()
    for ep in range(CFG["epochs"]):
        model.train(); L, LT, NT = [], [], []
        for _ in range(CFG["steps_per_epoch"]):
            s = torch.randint(0, T - W + 1, (CFG["batch"],), generator=gen, device=dev)
            idx = s[:, None] + torch.arange(W, device=dev)[None]
            Xs, Ms = X[idx], Mm[idx]
            H = Ms * (torch.rand(Ms.shape, generator=gen, device=dev) < CFG["hide_rate"]).float()
            loss, lt, nt = p9_loss(model, Xs, Ms, H, lc, CFG, lam, D["tower"])
            if loss is None:
                continue
            opt.zero_grad(); loss.backward(); g = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["clip"])
            if not (torch.isfinite(loss) and torch.isfinite(g)):
                nonfinite += 1; continue
            opt.step(); L.append(loss.item()); LT.append(lt); NT.append(nt)
        _, y0 = encode_series(model, Xv, Bv); y0 = y0.cpu().numpy()
        vr = float(np.sqrt((((y0[:, lc] - D["Xva"][:, lc]) ** 2) * Hv)[SPAN:].sum() / Hv[SPAN:].sum()))
        log.append(dict(epoch=ep, train_loss=float(np.mean(L)) if L else np.nan, tower_loss=float(np.nanmean(LT)) if D["tower"] and LT else np.nan,
                        tower_targets_per_step=float(np.mean(NT)) if NT else 0.0, val_heldout_obs_rmse=vr, time_s=time.time() - t0))
        if vr < best - 1e-6:
            best, bad = vr, 0; best_state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= CFG["patience"]:
                break
    lg = pd.DataFrame(log); lg.to_csv(os.path.join(RUNS, "logs", f"{eid}.csv"), index=False)
    fin = best_state is not None and all(torch.isfinite(v).all().item() for v in best_state.values())
    valid = fin and nonfinite == 0 and np.isfinite(best)
    row = dict(exp_id=eid, stage=a.stage, kappa=a.kappa, p=a.p, rep=a.rep, arm=a.arm, seed=seed, latent=D["latent"], n_in=len(D["ins"]), lam=lam,
               n_params=sum(q.numel() for q in model.parameters()), best_val=best, best_epoch=int(lg.val_heldout_obs_rmse.idxmin()) if valid else -1,
               epochs_run=len(log), hit_cap=len(log) >= CFG["epochs"], nonfinite_steps=nonfinite, finite_params=fin, valid=valid,
               tower_loss_first=float(lg.tower_loss.iloc[0]) if D["tower"] else np.nan, tower_loss_last=float(lg.tower_loss.iloc[-1]) if D["tower"] else np.nan,
               tower_targets_per_step=float(lg.tower_targets_per_step.mean()), wall_s=round(time.time() - t0, 1), code_md5=code_md5(),
               cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"), cfg=json.dumps(CFG), mask_seeds=json.dumps(D["mask_seeds"]), smoke=SMOKE)
    if not valid:
        fr = os.path.join(RUNS, "failed", f"{eid}.csv"); pd.DataFrame([row]).to_csv(fr + ".tmp", index=False); os.replace(fr + ".tmp", fr)
        print("INVALID", eid, flush=True); sys.exit(3)
    fc = os.path.join(RUNS, "ckpt", f"{eid}.pt")
    torch.save({"state_dict": best_state, "cfg": CFG, "arm": a.arm, "stage": a.stage, "kappa": a.kappa, "p": a.p, "rep": a.rep, "seed": seed,
                "ins": D["ins"], "outs": D["outs"], "latent": D["latent"], "lam": lam, "loss_cols": lc, "smoke": SMOKE}, fc + ".tmp")
    os.replace(fc + ".tmp", fc)
    fr = os.path.join(RUNS, "rows", f"{eid}.csv"); pd.DataFrame([row]).to_csv(fr + ".tmp", index=False); os.replace(fr + ".tmp", fr)
    print("RESULT", eid, "best_val", round(best, 5), "epochs", len(log), "wall", row["wall_s"], flush=True); sys.exit(0)


if __name__ == "__main__":
    main()
