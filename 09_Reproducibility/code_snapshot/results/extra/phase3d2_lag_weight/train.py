"""Phase 3D-2 Track A training: Lorenz observed through (x, z), y hidden; dt=0.05, p=0.3 (Phase 3C data/masks/splits).
Variants: C (current-value decoder, src loss), Ccap (C with decoder width 67, parameter-matched to FDR),
FDR{alpha} (FDR decoder + lag-weighted objective experiments.fdr.lagweight; alpha = lag-0 weight; alpha=1/8 is the original
uniform FDR objective up to float32 rounding of 1/c(u)). cfg = Phase 3C cfg (copied from results/checkpoints/lorenz_C_p0.3_s{seed}.pt;
seeds without such a checkpoint use the s0 cfg with seed replaced; asserted identical otherwise). epochs cap 300, patience 15."""
import argparse, glob, hashlib, json, os, sys, time
import numpy as np, pandas as pd, torch
ap = argparse.ArgumentParser(); ap.add_argument("--variant", required=True); ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--out", required=True); a = ap.parse_args(); out = os.path.abspath(a.out)
assert os.environ.get("IMD_RESULTS_DIR", "").rstrip("/") == out.rstrip("/")
import src.train as T
from src.models import build_model as orig_build
from experiments.fdr.model import FDRModel
from experiments.fdr.lagweight import CapMatchedC, make_forward_batch_lagweight
from experiments.fdr import xz_data
assert torch.cuda.is_available()
files = sorted(glob.glob("src/*.py")) + sorted(glob.glob("experiments/fdr/*.py")) + [__file__]
code_md5 = hashlib.md5(b"".join(open(f, "rb").read() for f in files)).hexdigest()[:8]
p0 = "results/checkpoints/lorenz_C_p0.3_s{}.pt"
base = torch.load(p0.format(0), map_location="cpu", weights_only=False)["cfg"]
if os.path.exists(p0.format(a.seed)):
    cfg = dict(torch.load(p0.format(a.seed), map_location="cpu", weights_only=False)["cfg"]); cfg_src = p0.format(a.seed)
    assert {k: v for k, v in cfg.items() if k != "seed"} == {k: v for k, v in base.items() if k != "seed"}
else:
    cfg = dict(base); cfg["seed"] = a.seed; cfg_src = p0.format(0) + " (seed replaced)"
cfg.update(device="cuda", dataset="lorenz_dt0.05_xz", epochs=300)
v = a.variant
if v == "C":
    cfg["model"] = "C"; T.build_model = lambda name, n, m, latent, hidden, window, direction: orig_build("C", n, m, latent, hidden, window, direction)
elif v == "Ccap":
    cfg["model"] = "Ccap"; T.build_model = lambda name, n, m, latent, hidden, window, direction: CapMatchedC(n, m, latent, hidden)
elif v.startswith("FDR"):
    alpha = float(eval(v[3:])); cfg.update(model="FDR", lag0_alpha=alpha)
    T.build_model = lambda name, n, m, latent, hidden, window, direction: FDRModel(n, m, latent, hidden)
    T.forward_batch = make_forward_batch_lagweight(alpha, cfg["m"])
else:
    raise ValueError(v)
exp_id = f"p3d2_{v.replace('/', 'o')}_s{a.seed}"
X3 = xz_data.load_raw(); d, _ = xz_data.build(X3, 0.3, a.seed, "xz")
t0 = time.time(); row, xhat, log = T.run(cfg, d["X"], d, exp_id, save_ckpt=True, verbose=False)
lg = pd.DataFrame(log); best_ep = int(lg.val_rmse.idxmin())
row.update(variant=v, code_md5=code_md5, cfg_src=cfg_src, wall_s=round(time.time() - t0, 1), finite=bool(np.isfinite(xhat).all()),
           best_epoch=best_ep, final_epoch=len(lg) - 1, hit_cap=len(lg) >= cfg["epochs"],
           stop_reason="epoch_cap" if len(lg) >= cfg["epochs"] else "patience",
           cfg_json=json.dumps({k: v_ for k, v_ in cfg.items() if k != "device"}), cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"))
np.save(os.path.join(out, f"{exp_id}_xhat_past.npy"), xhat.astype(np.float32))
os.makedirs(os.path.join(out, "rows"), exist_ok=True)            # one file per run (parallel lanes); merged later
pd.DataFrame([row]).to_csv(os.path.join(out, "rows", f"{exp_id}.csv"), index=False)
print("RESULT", exp_id, "test_rmse", row["test_rmse"], "best_ep", best_ep, "epochs", len(lg), "nonfinite", row["nonfinite_steps"], "params", row["n_params"], flush=True)
sys.exit(3 if (row["nonfinite_steps"] > 0 or not row["finite"]) else 0)
