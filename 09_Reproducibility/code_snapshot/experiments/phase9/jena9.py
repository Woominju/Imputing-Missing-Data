"""Phase 9 real-data experiment (protocol section 7; SECONDARY, descriptive only). Jena climate (70,127 x 5 hourly: p, T, rh, u, v).
Chronological split 70/15/15; normalisation from OBSERVED train entries; natural NaN = never observed (inputs) and never a target.
Artificial MCAR p in {0.3, 0.7} on top of the natural observations; replicate r = (init seed 9800 + r, mask seeds seed_of('p9jena', ...)),
r = 0..7 -- the SAME series, so replicates are not independent data (no inference). Arms: P9 (5 views, latent 16, L_rec + L_tow),
Bfull (same, lam = 0), SelfV (one AE per variable, latent 4 -> 20). Training loop = train9.p9_loss + the Phase 8 loop.
usage: jena9.py train --p P --rep R --arm {P9,Bfull,SelfV1..SelfV5} | jena9.py eval --p P --rep R | jena9.py run NPAR (GPUS env)"""
import argparse, json, os, subprocess, sys, time
import numpy as np, pandas as pd, torch
from experiments.phase5.core import ArmModel, encode_series, seed_of, mcar, SPAN
from experiments.phase8.evaluate8 import locf
from experiments.phase9.core9 import AE_CFG, CONST, device, SMOKE
from experiments.phase9.train9 import p9_loss

ROOT = os.environ.get("P9_ROOT", "results/extra/phase9") + "/jena"; SRC = "Data/external/jena/processed/jena.npz"; R_J = 2 if SMOKE else 8
ARMS = ["P9", "Bfull"] + [f"SelfV{v}" for v in range(1, 6)]


def data():
    X = np.load(SRC, allow_pickle=True)["X"].astype(np.float64)
    if SMOKE:
        X = X[:3000]
    n = len(X); a, b = int(.7 * n), int(.85 * n); mu = np.nanmean(X[:a], 0); sd = np.nanstd(X[:a], 0)
    Z = ((X - mu) / sd).astype(np.float32); nat = (~np.isnan(Z)).astype(np.float32); Z = np.nan_to_num(Z)
    return dict(tr=(Z[:a], nat[:a]), va=(Z[a:b], nat[a:b]), te=(Z[b:], nat[b:]))


def masks(p, r, split, shape, nat):
    return mcar(shape, p, seed_of("p9jena", p, r, split)) * nat


def spec(arm):
    return ([0, 1, 2, 3, 4], 16, arm == "P9") if arm in ("P9", "Bfull") else ([int(arm[-1]) - 1], 4, False)


def train(p, r, arm):
    ins, lat, tower = spec(arm); CFG = dict(AE_CFG); seed = 9800 + r; eid = f"jena_p{p}_r{r}_{arm}"; dev = device()
    torch.manual_seed(seed); np.random.seed(seed); Dd = data(); os.makedirs(f"{ROOT}/ckpt", exist_ok=True); os.makedirs(f"{ROOT}/rows", exist_ok=True)
    (Ztr, ntr), (Zva, nva) = Dd["tr"], Dd["va"]; Mtr = masks(p, r, "train", Ztr.shape, ntr)[:, ins]; Mva = masks(p, r, "val", Zva.shape, nva)[:, ins]
    Hva = Mva * (np.random.default_rng(seed_of("p9jenavh", p, r)).random(Mva.shape) < 0.2).astype(np.float32)
    model = ArmModel(len(ins), list(range(len(ins))), np.ones((len(ins), 8), np.float32), CFG["m"], lat, CFG["hidden"]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=CFG["lr"]); gen = torch.Generator(device=dev).manual_seed(seed + 1000)
    X = torch.tensor(Ztr[:, ins], device=dev); Mm = torch.tensor(Mtr, device=dev); Xv = torch.tensor(Zva[:, ins], device=dev); Bv = torch.tensor(Mva - Hva, device=dev)
    lc = list(range(len(ins))); W = SPAN + CFG["targets"]; T = X.shape[0]; best, best_state, bad, nonf, log = np.inf, None, 0, 0, []
    for ep in range(CFG["epochs"]):
        model.train(); L = []
        for _ in range(CFG["steps_per_epoch"]):
            s = torch.randint(0, T - W + 1, (CFG["batch"],), generator=gen, device=dev); idx = s[:, None] + torch.arange(W, device=dev)[None]
            Xs, Ms = X[idx], Mm[idx]; H = Ms * (torch.rand(Ms.shape, generator=gen, device=dev) < CFG["hide_rate"]).float()
            loss, lt, nt = p9_loss(model, Xs, Ms, H, lc, CFG, CONST["lam"] if tower else 0.0, tower)
            if loss is None:
                continue
            opt.zero_grad(); loss.backward(); g = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["clip"])
            if not (torch.isfinite(loss) and torch.isfinite(g)):
                nonf += 1; continue
            opt.step(); L.append(loss.item())
        _, y0 = encode_series(model, Xv, Bv); y0 = y0.cpu().numpy(); vr = float(np.sqrt((((y0 - Zva[:, ins]) ** 2) * Hva)[SPAN:].sum() / Hva[SPAN:].sum()))
        log.append(vr)
        if vr < best - 1e-6:
            best, bad, best_state = vr, 0, {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= CFG["patience"]:
                break
    valid = best_state is not None and nonf == 0 and np.isfinite(best)
    if valid:
        torch.save(dict(state_dict=best_state, ins=ins, latent=lat), f"{ROOT}/ckpt/{eid}.pt.tmp"); os.replace(f"{ROOT}/ckpt/{eid}.pt.tmp", f"{ROOT}/ckpt/{eid}.pt")
    pd.DataFrame([dict(exp_id=eid, p=p, rep=r, arm=arm, best_val=best, epochs=len(log), hit_cap=len(log) >= CFG["epochs"], nonfinite=nonf, valid=valid, smoke=SMOKE,
                       cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"))]).to_csv(f"{ROOT}/rows/{eid}.csv.tmp", index=False)
    os.replace(f"{ROOT}/rows/{eid}.csv.tmp", f"{ROOT}/rows/{eid}.csv"); sys.exit(0 if valid else 3)


def evaluate(p, r):
    dev = device(); Dd = data(); Zte, nte = Dd["te"]; M = masks(p, r, "test", Zte.shape, nte); out = []
    def lag0(arm):
        ck = torch.load(f"{ROOT}/ckpt/jena_p{p}_r{r}_{arm}.pt", map_location=dev, weights_only=False); ins = ck["ins"]
        m = ArmModel(len(ins), list(range(len(ins))), np.ones((len(ins), 8), np.float32), latent=ck["latent"]).to(dev); m.load_state_dict(ck["state_dict"])
        _, y = encode_series(m, torch.tensor(Zte[:, ins], device=dev), torch.tensor(M[:, ins], device=dev)); return ins, y.cpu().numpy()
    P = {"zero": np.zeros_like(Zte), "LOCF": locf(Zte, M)}
    for arm in ("P9", "Bfull"):
        P[f"AE:{arm}"] = lag0(arm)[1]
    S = np.zeros_like(Zte)
    for v in range(1, 6):
        ins, y = lag0(f"SelfV{v}"); S[:, ins[0]] = y[:, 0]
    P["AE:SelfV"] = S
    tgt = (M == 0) & (nte == 1); tgt[:SPAN] = False                     # artificially masked positions with a known truth
    for mt, Y in P.items():
        for c in range(5):
            sel = tgt[:, c]; out.append(dict(method=mt, view=c, sse=float(((Y[sel, c] - Zte[sel, c]) ** 2).sum()), n=int(sel.sum())))
    os.makedirs(f"{ROOT}/eval", exist_ok=True); f = f"{ROOT}/eval/jena_p{p}_r{r}.csv"
    pd.DataFrame(out).assign(p=p, rep=r, smoke=SMOKE).to_csv(f + ".tmp", index=False); os.replace(f + ".tmp", f)


def run(npar):
    gpus = os.environ.get("GPUS", "").split(",") if not SMOKE else [""]; env = dict(os.environ, PYTHONPATH=".pydeps:.", CUDA_DEVICE_ORDER="PCI_BUS_ID", OMP_NUM_THREADS="2")
    T = [("train", p, r, a) for p in (0.7, 0.3) for r in range(R_J) for a in ARMS]; os.makedirs(f"{ROOT}/logs", exist_ok=True); running, gi = [], 0
    def go(cmd):
        nonlocal gi
        g = gpus[gi % len(gpus)]; gi += 1; e = dict(env, CUDA_VISIBLE_DEVICES=g) if g else env
        return subprocess.Popen([sys.executable, "experiments/phase9/jena9.py"] + cmd, env=e, stdout=open(f"{ROOT}/logs/{'_'.join(cmd)}.log", "w"), stderr=subprocess.STDOUT)
    q = [t for t in T if not os.path.exists(f"{ROOT}/rows/jena_p{t[1]}_r{t[2]}_{t[3]}.csv")]
    while q or running:
        running = [x for x in running if x.poll() is None]
        while q and len(running) < npar:
            _, p, r, a = q.pop(0); running.append(go(["train", "--p", str(p), "--rep", str(r), "--arm", a]))
        time.sleep(1 if SMOKE else 10)
    ev = [go(["eval", "--p", str(p), "--rep", str(r)]) for p in (0.7, 0.3) for r in range(R_J) if not os.path.exists(f"{ROOT}/eval/jena_p{p}_r{r}.csv")]
    [x.wait() for x in ev]; print("JENA DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode"); ap.add_argument("npar", nargs="?", type=int, default=16)
    ap.add_argument("--p", type=float); ap.add_argument("--rep", type=int); ap.add_argument("--arm"); a = ap.parse_args()
    {"train": lambda: train(a.p, a.rep, a.arm), "eval": lambda: evaluate(a.p, a.rep), "run": lambda: run(a.npar)}[a.mode]()
