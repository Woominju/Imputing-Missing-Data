"""Phase 4 coupled-Lorenz training. Views (x1, z1, x2, z2), hidden (y1, y2); all views MCAR at rate p; dt 0.05.
2 x 2 factorial: decoder {C = current value (src loss), FDR = full delay (experiments.fdr.loss, uniform 1/c(u) objective)}
x fusion {Scalar = observation-count mean (src MultiViewEncoder), Learned = Model F mask-aware MLP fusion}.
Same encoder per view, latent 16, hidden 64, optimiser, batch, hide rate 0.2, epochs cap 300, patience 15 for every cell
(cfg = results/checkpoints/lorenz_C_p0.3_s0.pt with seed replaced; no tuning on coupled data)."""
import argparse, glob, hashlib, json, os, sys, time
import numpy as np, pandas as pd, torch
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True, choices=["C-Scalar", "C-Learned", "FDR-Scalar", "FDR-Learned"])
ap.add_argument("--kappa", type=float, required=True); ap.add_argument("--traj", type=int, default=0); ap.add_argument("--p", type=float, required=True)
ap.add_argument("--seed", type=int, required=True); ap.add_argument("--out", required=True); a = ap.parse_args(); out = os.path.abspath(a.out)
assert os.environ.get("IMD_RESULTS_DIR", "").rstrip("/") == out.rstrip("/")
import src.train as T
from src.models import build_model
from experiments.aim_autonomous.model_f import ModelF
from experiments.fdr.model import FDRModel
from experiments.fdr.lagweight import FDRLearned
from experiments.fdr.loss import make_forward_batch
from experiments.coupled import data as CD
assert torch.cuda.is_available()
files = sorted(glob.glob("src/*.py")) + sorted(glob.glob("experiments/fdr/*.py")) + sorted(glob.glob("experiments/coupled/*.py")) + ["experiments/aim_autonomous/model_f.py", __file__]
code_md5 = hashlib.md5(b"".join(open(f, "rb").read() for f in files)).hexdigest()[:8]
cfg = dict(torch.load("results/checkpoints/lorenz_C_p0.3_s0.pt", map_location="cpu", weights_only=False)["cfg"])
cfg.update(seed=a.seed, device="cuda", dataset=f"coupled_k{a.kappa}_traj{a.traj}_dt0.05", epochs=300, model=a.model)
ctor = {"C-Scalar": lambda n, m, l, h: build_model("C", n, m, l, h), "C-Learned": ModelF, "FDR-Scalar": FDRModel, "FDR-Learned": FDRLearned}[a.model]
T.build_model = lambda name, n, m, latent, hidden, window, direction: ctor(n, m, latent, hidden)
if a.model.startswith("FDR"):
    T.forward_batch = make_forward_batch(0.0, "full")
exp_id = f"p4_k{a.kappa}_t{a.traj}_p{a.p}_{a.model}_s{a.seed}"
d, _ = CD.build(CD.load_raw(a.kappa, a.traj), a.p, a.seed)
t0 = time.time(); row, xhat, log = T.run(cfg, d["X"], d, exp_id, save_ckpt=True, verbose=False)
lg = pd.DataFrame(log); best_ep = int(lg.val_rmse.idxmin())
row.update(model_name=a.model, kappa=a.kappa, traj=a.traj, p=a.p, code_md5=code_md5, wall_s=round(time.time() - t0, 1), finite=bool(np.isfinite(xhat).all()),
           best_epoch=best_ep, final_epoch=len(lg) - 1, hit_cap=len(lg) >= cfg["epochs"], stop_reason="epoch_cap" if len(lg) >= cfg["epochs"] else "patience",
           cfg_json=json.dumps({k: v for k, v in cfg.items() if k != "device"}), cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"))
os.makedirs(os.path.join(out, "preds"), exist_ok=True); os.makedirs(os.path.join(out, "rows"), exist_ok=True)
np.save(os.path.join(out, "preds", f"{exp_id}_xhat_past.npy"), xhat.astype(np.float32))
pd.DataFrame([row]).to_csv(os.path.join(out, "rows", f"{exp_id}.csv"), index=False)
print("RESULT", exp_id, "test_rmse", row["test_rmse"], "best_ep", best_ep, "epochs", len(lg), "params", row["n_params"], flush=True)
sys.exit(3 if (row["nonfinite_steps"] > 0 or not row["finite"]) else 0)
