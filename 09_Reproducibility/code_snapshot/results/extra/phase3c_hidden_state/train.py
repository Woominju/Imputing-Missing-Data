"""Phase 3C training (max 8 runs): {C (current value), FDR (full delay)} x observed variables {xz, x}, Lorenz dt=0.05, p=0.3.
cfg copied from results/checkpoints/lorenz_C_p0.3_s{seed}.pt (m=8, tau=1, latent 16, hidden 64, lr 1e-3, hide 0.2, Q=0,
scalar fusion). Epoch cap raised a priori, identically for all 8 runs, from 150 to 300 (patience 15 unchanged), because
Phase 3B runs hit the 150 cap. y is never loaded into the AE data (experiments.fdr.xz_data)."""
import argparse, glob, hashlib, json, os, sys, time
import numpy as np, pandas as pd, torch
ap = argparse.ArgumentParser(); ap.add_argument("--decoder", choices=["current", "full"], required=True)
ap.add_argument("--vars", choices=["xz", "x"], required=True); ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--out", required=True); a = ap.parse_args(); out = os.path.abspath(a.out)
assert os.environ.get("IMD_RESULTS_DIR", "").rstrip("/") == out.rstrip("/")
import src.train as T
from src.models import build_model as orig_build
from experiments.fdr.model import FDRModel
from experiments.fdr.loss import make_forward_batch
from experiments.fdr import xz_data
assert torch.cuda.is_available()
files = sorted(glob.glob("src/*.py")) + sorted(glob.glob("experiments/fdr/*.py")) + [__file__]
code_md5 = hashlib.md5(b"".join(open(f, "rb").read() for f in files)).hexdigest()[:8]
ck = torch.load(f"results/checkpoints/lorenz_C_p0.3_s{a.seed}.pt", map_location="cpu", weights_only=False)
cfg = dict(ck["cfg"]); cfg.update(device="cuda", dataset=f"lorenz_dt0.05_{a.vars}", model="FDR" if a.decoder == "full" else "C", epochs=300)
T.build_model = lambda name, n, m, latent, hidden, window, direction: FDRModel(n, m, latent, hidden) if name == "FDR" else orig_build(name, n, m, latent, hidden, window, direction)
T.forward_batch = make_forward_batch(0.0, a.decoder)
tag = ("FDR" if a.decoder == "full" else "C") + "-" + a.vars
exp_id = f"p3c_{tag}_dt0.05_p0.3_s{a.seed}"
X3 = xz_data.load_raw(); d, _ = xz_data.build(X3, 0.3, a.seed, a.vars)
t0 = time.time(); row, xhat, log = T.run(cfg, d["X"], d, exp_id, save_ckpt=True, verbose=True)
lg = pd.DataFrame(log); best_ep = int(lg.val_rmse.idxmin())
row.update(tag=tag, decoder=a.decoder, vars=a.vars, code_md5=code_md5, wall_s=round(time.time() - t0, 1), finite=bool(np.isfinite(xhat).all()),
           best_epoch=best_ep, final_epoch=len(lg) - 1, hit_cap=len(lg) >= cfg["epochs"],
           stop_reason="epoch_cap" if len(lg) >= cfg["epochs"] else "patience",
           val_last15_change=float(lg.val_rmse.iloc[-1] - lg.val_rmse.iloc[max(0, len(lg) - 16)]),
           cfg_json=json.dumps({k: v for k, v in cfg.items() if k != "device"}), cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"))
np.save(os.path.join(out, f"{exp_id}_xhat_past.npy"), xhat.astype(np.float32))
csv = os.path.join(out, "runs.csv"); pd.DataFrame([row]).to_csv(csv, mode="a", header=not os.path.exists(csv), index=False)
print("RESULT", exp_id, "test_rmse", row["test_rmse"], "best_ep", best_ep, "epochs", len(lg), "cap", row["hit_cap"], "nonfinite", row["nonfinite_steps"], "params", row["n_params"], flush=True)
sys.exit(3 if (row["nonfinite_steps"] > 0 or not row["finite"]) else 0)
