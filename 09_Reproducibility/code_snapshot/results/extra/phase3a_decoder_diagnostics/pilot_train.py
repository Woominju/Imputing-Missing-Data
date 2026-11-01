"""Phase 3A pilot (max 4 runs): Model C architecture with latent k in {3,16} x training mask {standard, view-dropout Q=0.5}.
Lorenz p=0.7 seed=0. All settings copied from the original checkpoint results/checkpoints/lorenz_C_p0.7_s0.pt except
`latent`. src/ is not modified; for 'E' the training forward is replaced in-process by the archived Model E function.
Usage: IMD_RESULTS_DIR=<out> python -m results.extra.phase3a_decoder_diagnostics.pilot_train --variant C --latent 3 --out <out>"""
import argparse, hashlib, glob, json, os, sys, time
import numpy as np, pandas as pd, torch
ap = argparse.ArgumentParser(); ap.add_argument("--variant", choices=["C", "E"], required=True)
ap.add_argument("--latent", type=int, required=True); ap.add_argument("--out", required=True)
ap.add_argument("--p", type=float, default=0.7); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args(); out = os.path.abspath(a.out)
assert os.environ.get("IMD_RESULTS_DIR", "").rstrip("/") == out.rstrip("/"), "set IMD_RESULTS_DIR=out before import of src.train"
import src.train as T
import experiments.aim_autonomous.model_e_viewdrop as EV
from src.missingness import prepare
assert torch.cuda.is_available()
code_md5 = hashlib.md5(b"".join(open(f, "rb").read() for f in sorted(glob.glob("src/*.py")) + ["experiments/aim_autonomous/model_e_viewdrop.py", __file__])).hexdigest()[:8]
ck = torch.load(f"results/checkpoints/lorenz_C_p{a.p}_s{a.seed}.pt", map_location="cpu", weights_only=False)
cfg = dict(ck["cfg"]); cfg.update(device="cuda", latent=a.latent)
if a.variant == "E":
    EV.Q = 0.5; T.forward_batch = EV.forward_batch_viewdrop
exp_id = f"pilot_{a.variant}{a.latent}_p{a.p}_s{a.seed}"
os.makedirs(out, exist_ok=True)
Xraw = np.load("results/data/lorenz_20000.npz")["X"]; d = prepare(Xraw, a.p, a.seed)
t0 = time.time()
row, xhat, log = T.run(cfg, Xraw, d, exp_id, save_ckpt=True, verbose=True)
row.update(variant=a.variant, viewdrop_Q=0.5 if a.variant == "E" else 0.0, code_md5=code_md5, wall_s=round(time.time() - t0, 1),
           finite=bool(np.isfinite(xhat).all()), cfg_json=json.dumps({k: v for k, v in cfg.items() if k != "device"}),
           cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"))
np.save(os.path.join(out, f"{exp_id}_xhat.npy"), xhat.astype(np.float32))
csv = os.path.join(out, "pilot_runs.csv")
pd.DataFrame([row]).to_csv(csv, mode="a", header=not os.path.exists(csv), index=False)
print("RESULT", exp_id, "test_rmse", row["test_rmse"], "epochs", row["epochs_run"], "nonfinite", row["nonfinite_steps"], "params", row["n_params"], flush=True)
if row["nonfinite_steps"] > 0 or not row["finite"]:
    print("WARNING non-finite encountered", flush=True); sys.exit(3)
