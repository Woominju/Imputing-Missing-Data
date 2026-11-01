"""Phase 7 P7-B training of one arm (GPU): derived from experiments/phase6/train6.py (identical loop, CFG, hiding, optimiser);
ONLY change: the decoder always outputs all 4 views and the FDR loss / early-stopping criterion use the arm's loss-active
columns D["loss_cols"] (Bself_j: own 2 views, Bfull: all 4). Seeds: init 7000 + r. Data: experiments.phase7.core7.build.
Original train6 docstring follows. Only the data assembly
(experiments.phase6.core6.build) and the seed convention differ (init seed = 6000 + r, recorded; mask seeds recorded).
Phase 5 docstring follows. Phase 5 training of one arm (GPU). Objective = multiplicity-corrected masked FDR loss (experiments.fdr.loss) restricted
to the arm's TARGET columns; hiding H ~ Bernoulli(hide) on every observed input entry (same batch-start RNG sequence for every arm of a
replicate; the H draws differ between 2-view and 4-view arms because the tensor shapes differ). Early stopping uses NO ground truth of missing values: lag-0 RMSE at a fixed 20% of OBSERVED target entries of the
independent validation trajectory, hidden from the input (protocol section 3).
usage: train.py --kappa K --p P --rep R --arm ARM [--seed S] --out DIR"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd, torch
from experiments.phase5.core import ArmModel, encode_series, M_, TAU, SPAN
from experiments.phase7.core7 import build, code_md5, init_seed
from experiments.fdr.loss import fdr_masked_loss
from src.delay_embedding import delay_embed

CFG = dict(m=M_, tau=TAU, latent=16, hidden=64, batch=64, targets=64, lr=1e-3, epochs=300, steps_per_epoch=100,
           patience=15, hide_rate=0.2, clip=5.0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--kappa", type=float, required=True); ap.add_argument("--p", type=float, required=True)
    ap.add_argument("--rep", type=int, required=True); ap.add_argument("--arm", required=True); ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", required=True); a = ap.parse_args()
    seed = init_seed(a.rep) if a.seed is None else a.seed
    eid = f"k{a.kappa}_p{a.p}_r{a.rep}_{a.arm}_s{seed}"
    for d in ("ckpt", "logs", "rows"):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)
    dev = torch.device("cuda"); torch.manual_seed(seed); np.random.seed(seed)
    D = build(a.kappa, a.p, a.rep, a.arm)
    model = ArmModel(len(D["ins"]), D["outs"], D["lagmask"], CFG["m"], CFG["latent"], CFG["hidden"]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=CFG["lr"])
    gen = torch.Generator(device=dev).manual_seed(seed + 1000)
    X = torch.tensor(D["Xtr"], device=dev); Mm = torch.tensor(D["Mtr"], device=dev)
    Xv = torch.tensor(D["Xva"], device=dev); Bv = torch.tensor(D["Mva"] - D["Hva"], device=dev); Hv = D["Hva"][:, D["loss_cols"]]; lc = D["loss_cols"]
    outs = D["outs"]; W = SPAN + CFG["targets"]; T = X.shape[0]
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
    torch.save({"state_dict": best_state, "cfg": CFG, "arm": a.arm, "kappa": a.kappa, "p": a.p, "rep": a.rep, "seed": seed,
                "ins": D["ins"], "outs": outs, "mask_seeds": D["mask_seeds"], "loss_cols": lc}, os.path.join(a.out, "ckpt", f"{eid}.pt"))
    pd.DataFrame(log).to_csv(os.path.join(a.out, "logs", f"{eid}.csv"), index=False)
    fin = all(torch.isfinite(v).all().item() for v in best_state.values())
    row = dict(exp_id=eid, kappa=a.kappa, p=a.p, rep=a.rep, arm=a.arm, seed=seed, n_params=sum(q.numel() for q in model.parameters()),
               best_val=best, best_epoch=int(pd.DataFrame(log).val_heldout_obs_rmse.idxmin()), epochs_run=len(log),
               hit_cap=len(log) >= CFG["epochs"], nonfinite_steps=nonfinite, finite_params=fin, wall_s=round(time.time() - t0, 1),
               code_md5=code_md5(), cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"), cfg=json.dumps(CFG), mask_seeds=json.dumps(D["mask_seeds"]))
    pd.DataFrame([row]).to_csv(os.path.join(a.out, "rows", f"{eid}.csv"), index=False)
    print("RESULT", eid, "best_val", round(best, 5), "epochs", len(log), "wall", row["wall_s"], flush=True)
    sys.exit(0 if (fin and nonfinite == 0) else 3)


if __name__ == "__main__":
    main()
