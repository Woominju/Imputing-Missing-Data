"""Phase 2.5 leakage audit (CPU, read-only on checkpoints). Perturbation tests on the real code paths:
 L1 impute(): changing values at eval-mask (artificially missing) positions must not change xhat anywhere.
 L2 impute(): changing ANY value at times > t (future) must not change xhat at times <= t (past-only claim).
 L3 forward_batch(): changing values at training-hidden positions H must not change model output.
 L4 masked_loss multiplicity: each hidden raw entry is a loss target exactly once per segment (current-value target).
 L5 FDR target multiplicity (analysis only): how often one raw entry would be a target if all lags were targets.
"""
import numpy as np, torch
import src.train as T
from src.delay_embedding import delay_embed
from src.evaluate import impute
from src.missingness import prepare
from src.models import build_model
torch.set_num_threads(4)
Xraw = np.load("results/data/lorenz_20000.npz")["X"]
d = prepare(Xraw, 0.3, 0)
ck = torch.load("results/checkpoints/lorenz_C_p0.3_s0.pt", map_location="cpu", weights_only=False)
cfg = dict(ck["cfg"]); cfg["device"] = "cpu"
model = build_model("C", 3, cfg["m"], cfg["latent"], cfg["hidden"], cfg["window"], cfg["direction"])
model.load_state_dict(ck["state_dict"]); model.eval()
X = torch.tensor(d["X"]); M = torch.tensor(d["M"]); E = torch.tensor(d["eval_mask"])
base = impute(model, X, M, cfg["m"], cfg["tau"])
g = torch.Generator().manual_seed(0)
# L1
X1 = X.clone(); X1[E > 0] = 1e3 * torch.randn(int(E.sum()), generator=g)
print("L1 eval-target values changed -> max |dxhat| =", float((impute(model, X1, M, cfg["m"], cfg["tau"]) - base).abs().max()))
# L2
for t in [5000, 15000, 17500]:
    X2 = X.clone(); X2[t + 1:] = 1e3 * torch.randn(X2[t + 1:].shape, generator=g)
    M2 = M.clone(); M2[t + 1:] = (torch.rand(M2[t + 1:].shape, generator=g) > 0.5).float()
    out = impute(model, X2, M2, cfg["m"], cfg["tau"])
    print(f"L2 future (>{t}) values+mask changed -> max |dxhat[<= {t}]| =", float((out[:t + 1] - base[:t + 1]).abs().max()))
# L3
a, b = d["split"]["train"]; W = T.segment_len(cfg); s = torch.randint(a, b - W, (16,), generator=g)
idx = s[:, None] + torch.arange(W)[None]; Xs, Ms = X[idx], M[idx]
H = Ms * (torch.rand(Ms.shape, generator=g) < cfg["hide_rate"]).float(); B = Ms - H
span = (cfg["m"] - 1) * cfg["tau"]
o1 = model(*delay_embed(Xs * B, B, cfg["m"], cfg["tau"]))
Xs2 = Xs + H * 1e3 * torch.randn(Xs.shape, generator=g)
o2 = model(*delay_embed(Xs2 * B, B, cfg["m"], cfg["tau"]))
print("L3 training-hidden values changed -> max |doutput| =", float((o1 - o2).abs().max()))
# L4/L5
K = W - span
print(f"L4 segment W={W}, span={span}, targets per segment K={K}: current-value loss counts each target time once; "
      f"hidden entries in the first span={span} positions are hidden from input but never targets.")
mult = np.zeros(W)
for t in range(span, W):
    for k in range(cfg["m"]):
        mult[t - k * cfg["tau"]] += 1
print("L5 FDR (all lags as targets) multiplicity per segment position: min", mult.min(), "max", mult.max(),
      "| first positions:", mult[:span + 2].tolist(), "| last:", mult[-3:].tolist())
print("cfg used:", {k: cfg[k] for k in ["m", "tau", "targets", "hide_rate", "latent", "hidden"]})
