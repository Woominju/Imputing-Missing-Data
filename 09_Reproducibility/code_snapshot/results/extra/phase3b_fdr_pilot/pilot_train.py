"""Phase 3B 2x2 pilot (max 4 runs): decoder {current (C), full delay (FDR)} x training mask {Q=0, Q=0.15}.
Lorenz dt=0.05 (archived data results/extra/aim_autonomous_20261027/coarse_dt0.05/data/lorenz_dt0.05_20000.npz), p=0.7, seed 0.
cfg copied from results/checkpoints/lorenz_C_p0.7_s0.pt (as the archived dt=0.05 runs did); latent 16, scalar fusion.
src/ untouched: src.train.run is used with in-process replacements of build_model and forward_batch only."""
import argparse, glob, hashlib, json, os, sys, time
import numpy as np, pandas as pd, torch
ap = argparse.ArgumentParser(); ap.add_argument("--decoder", choices=["current", "full"], required=True)
ap.add_argument("--Q", type=float, required=True); ap.add_argument("--out", required=True)
ap.add_argument("--p", type=float, default=0.7); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args(); out = os.path.abspath(a.out)
assert os.environ.get("IMD_RESULTS_DIR", "").rstrip("/") == out.rstrip("/")
import src.train as T
from src.models import build_model as orig_build
from src.missingness import prepare
from experiments.fdr.model import FDRModel
from experiments.fdr.loss import make_forward_batch
assert torch.cuda.is_available()
files = sorted(glob.glob("src/*.py")) + sorted(glob.glob("experiments/fdr/*.py")) + [__file__]
code_md5 = hashlib.md5(b"".join(open(f, "rb").read() for f in files)).hexdigest()[:8]
ck = torch.load(f"results/checkpoints/lorenz_C_p{a.p}_s{a.seed}.pt", map_location="cpu", weights_only=False)
cfg = dict(ck["cfg"]); cfg.update(device="cuda", dataset="lorenz_dt0.05", model="FDR" if a.decoder == "full" else "C")
T.build_model = lambda name, n, m, latent, hidden, window, direction: FDRModel(n, m, latent, hidden) if name == "FDR" else orig_build(name, n, m, latent, hidden, window, direction)
T.forward_batch = make_forward_batch(a.Q, a.decoder)
tag = {"current": "C", "full": "FDR"}[a.decoder] + {0.0: "0", 0.15: "15"}[a.Q]
exp_id = f"p3b_{tag}_dt0.05_p{a.p}_s{a.seed}"
Xraw = np.load("results/extra/aim_autonomous_20261027/coarse_dt0.05/data/lorenz_dt0.05_20000.npz")["X"]
d = prepare(Xraw, a.p, a.seed); t0 = time.time()
row, xhat, log = T.run(cfg, Xraw, d, exp_id, save_ckpt=True, verbose=True)
row.update(tag=tag, decoder=a.decoder, Q=a.Q, code_md5=code_md5, wall_s=round(time.time() - t0, 1), finite=bool(np.isfinite(xhat).all()),
           cfg_json=json.dumps({k: v for k, v in cfg.items() if k != "device"}), cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"))
np.save(os.path.join(out, f"{exp_id}_xhat_past.npy"), xhat.astype(np.float32))
csv = os.path.join(out, "pilot_runs.csv"); pd.DataFrame([row]).to_csv(csv, mode="a", header=not os.path.exists(csv), index=False)
print("RESULT", exp_id, "test_rmse", row["test_rmse"], "epochs", row["epochs_run"], "nonfinite", row["nonfinite_steps"], "params", row["n_params"], flush=True)
sys.exit(3 if (row["nonfinite_steps"] > 0 or not row["finite"]) else 0)
