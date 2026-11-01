"""Phase 8 training of one AE arm (protocol v2 section 5.2). The loop is experiments/phase7/train7.py unchanged (FDR masked loss,
hiding, Adam, clip, early stopping on the loss-active columns' held-out observed val entries); only the arm table (core8.ARM_SPECS:
input views, real/donor sources, latent width 8 or 16, init seed) and the data ids differ. No new loss, no new architecture.
usage: train8.py --kappa K --p P --rep R --arm ARM"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd, torch
from experiments.phase5.core import ArmModel, encode_series, SPAN
from experiments.phase8.core8 import build, code_md5, init_seed, device, AE_CFG, RUNS, SMOKE, ARMS8
from experiments.fdr.loss import fdr_masked_loss
INJECT = False
from src.delay_embedding import delay_embed


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--kappa", type=float, required=True); ap.add_argument("--p", type=float, required=True)
    ap.add_argument("--rep", type=int, required=True); ap.add_argument("--arm", required=True, choices=ARMS8); a = ap.parse_args()
    CFG = AE_CFG; seed = init_seed(a.arm, a.rep); eid = f"k{a.kappa}_p{a.p}_r{a.rep}_{a.arm}_s{seed}"
    global INJECT; INJECT = SMOKE and os.environ.get("P8_INJECT") == f"nonfinite:{a.kappa}:{a.p}:{a.arm}:{a.rep}"
    for d in ("ckpt", "logs", "rows", "failed"):
        os.makedirs(os.path.join(RUNS, d), exist_ok=True)
    dev = device(); torch.manual_seed(seed); np.random.seed(seed)
    D = build(a.kappa, a.p, a.rep, a.arm)
    model = ArmModel(len(D["ins"]), D["outs"], D["lagmask"], CFG["m"], D["latent"], CFG["hidden"]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=CFG["lr"])
    gen = torch.Generator(device=dev).manual_seed(seed + 1000)
    X = torch.tensor(D["Xtr"], device=dev); Mm = torch.tensor(D["Mtr"], device=dev)
    Xv = torch.tensor(D["Xva"], device=dev); Bv = torch.tensor(D["Mva"] - D["Hva"], device=dev); lc = D["loss_cols"]; Hv = D["Hva"][:, lc]
    W = SPAN + CFG["targets"]; T = X.shape[0]
    log, best, best_state, bad, nonfinite = [], np.inf, None, 0, 0; t0 = time.time()
    for ep in range(CFG["epochs"]):
        model.train(); L = []
        for _ in range(CFG["steps_per_epoch"]):
            s = torch.randint(0, T - W + 1, (CFG["batch"],), generator=gen, device=dev)
            idx = s[:, None] + torch.arange(W, device=dev)[None]
            Xs, Ms = X[idx], Mm[idx]
            H = Ms * (torch.rand(Ms.shape, generator=gen, device=dev) < CFG["hide_rate"]).float()
            Bm = Ms - H
            V, MV = delay_embed(Xs * Bm, Bm, CFG["m"], CFG["tau"])                  # masking BEFORE windowing
            loss, _ = fdr_masked_loss(model.forward_full(V, MV)[..., lc, :], Xs[..., lc], H[..., lc], CFG["m"], CFG["tau"])
            if loss is None:
                continue
            if INJECT:                                                   # smoke-only fault injection (tests)
                loss = loss * float("nan")
            opt.zero_grad(); loss.backward(); g = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["clip"])
            if not (torch.isfinite(loss) and torch.isfinite(g)):
                nonfinite += 1; continue
            opt.step(); L.append(loss.item())
        _, y0 = encode_series(model, Xv, Bv); y0 = y0.cpu().numpy()
        vr = float(np.sqrt((((y0[:, lc] - D["Xva"][:, lc]) ** 2) * Hv)[SPAN:].sum() / Hv[SPAN:].sum()))
        log.append(dict(epoch=ep, train_loss=float(np.mean(L)) if L else np.nan, val_heldout_obs_rmse=vr, time_s=time.time() - t0))
        if vr < best - 1e-6:
            best, bad = vr, 0; best_state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= CFG["patience"]:
                break
    pd.DataFrame(log).to_csv(os.path.join(RUNS, "logs", f"{eid}.csv"), index=False)
    fin = best_state is not None and all(torch.isfinite(v).all().item() for v in best_state.values())
    valid = fin and nonfinite == 0 and np.isfinite(best)
    row = dict(exp_id=eid, kappa=a.kappa, p=a.p, rep=a.rep, arm=a.arm, seed=seed, latent=D["latent"], n_in=len(D["ins"]),
               n_params=sum(q.numel() for q in model.parameters()), best_val=best,
               best_epoch=int(pd.DataFrame(log).val_heldout_obs_rmse.idxmin()) if valid else -1,
               epochs_run=len(log), hit_cap=len(log) >= CFG["epochs"], nonfinite_steps=nonfinite, finite_params=fin, valid=valid,
               wall_s=round(time.time() - t0, 1), code_md5=code_md5(), cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"),
               cfg=json.dumps(CFG), mask_seeds=json.dumps(D["mask_seeds"]), smoke=SMOKE)
    if not valid:
        # INVALID run (non-finite steps / parameters / validation): NO checkpoint and NO completion marker (rows/), so no dependent
        # O_lat / evaluation can start; a failure record is written instead (reported by analyze8; never retried by sched8).
        fr = os.path.join(RUNS, "failed", f"{eid}.csv"); pd.DataFrame([row]).to_csv(fr + ".tmp", index=False); os.replace(fr + ".tmp", fr)
        print("INVALID", eid, "nonfinite_steps", nonfinite, "finite_params", fin, flush=True); sys.exit(3)
    fc = os.path.join(RUNS, "ckpt", f"{eid}.pt")
    torch.save({"state_dict": best_state, "cfg": CFG, "arm": a.arm, "kappa": a.kappa, "p": a.p, "rep": a.rep, "seed": seed,
                "ins": D["ins"], "outs": D["outs"], "latent": D["latent"], "mask_seeds": D["mask_seeds"], "loss_cols": lc, "smoke": SMOKE}, fc + ".tmp")
    os.replace(fc + ".tmp", fc)
    fr = os.path.join(RUNS, "rows", f"{eid}.csv"); pd.DataFrame([row]).to_csv(fr + ".tmp", index=False); os.replace(fr + ".tmp", fr)   # marker LAST, only if valid
    print("RESULT", eid, "best_val", round(best, 5), "epochs", len(log), "wall", row["wall_s"], flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
