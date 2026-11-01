"""Phase 8 supervised DIAGNOSTIC oracles (protocol v2 section 4.2). They are trained on the independent oracle pool only
(ids 60000+; never on replicate, donor or test trajectories) and are fixed functions afterwards (D4: R*_A <= R_A(f)).
Class: MLP with 4 Linear layers, ReLU, width in {256, 512, 1024}; Adam 1e-3, batch 1024, epoch = ORC_EPOCH_STEPS steps,
<= ORC_MAX_EPOCHS epochs, patience ORC_PATIENCE on pool-val risk (coordinate-mean weighted MSE); seed 0.
Kinds:  raw   : raw masked 4-view window (64) -> S (6)           [width selected on pool-val]
        own{j}: raw masked own window W_j (32) -> S (6)           [width selected on pool-val]
        imp   : raw 4-view window (64) -> current 4 view values, loss only on currently MISSING entries   [selected]
        lat   : frozen latent of feature set FEAT of replicate R -> S; width FIXED to the width selected for 'raw' at (k, p)
        sat   : 'raw' at the selected width with the first {10, 20} pool train trajectories (saturation curve, descriptive)
usage: oracle8.py KIND --kappa K --p P [--j J] [--rep R --feat FEAT]"""
import argparse, json, os
import numpy as np, torch
from experiments.phase5.core import raw_features, seed_of, SPAN, SUB
from experiments.phase8.core8 import (ORC, SZ, CONST, SMOKE, device, load, nz_obs, nz_state, pool_ids, pool_mask)
from experiments.phase8.feats8 import FeatureMap


def mlp4(din, dout, w):
    return torch.nn.Sequential(torch.nn.Linear(din, w), torch.nn.ReLU(), torch.nn.Linear(w, w), torch.nn.ReLU(),
                               torch.nn.Linear(w, w), torch.nn.ReLU(), torch.nn.Linear(w, dout))


def pool_arrays(k, p, kind, split, featmap=None, ntraj=None):
    """features, targets, loss weights for the pool split (all K mask draws; rows n >= SPAN)."""
    F, Y, Wt = [], [], []
    qs = pool_ids()[split][:ntraj] if ntraj else pool_ids()[split]
    for q in qs:
        X6 = load(k, q); O = nz_obs(k, X6); S = nz_state(k, X6)
        for d in range(SZ["KPOOL"]):
            M = pool_mask(k, p, split, q, d, len(O))
            if kind == "raw" or kind == "imp":
                f = raw_features(O, M, [0, 1, 2, 3])
            elif kind.startswith("own"):
                f = raw_features(O, M, SUB[int(kind[-1])])
            else:
                f = featmap(O, M)
            if kind == "imp":
                y, w = O, 1.0 - M                       # current values; loss only where currently missing
            else:
                y, w = S, np.ones_like(S)
            F.append(f[SPAN:]); Y.append(y[SPAN:]); Wt.append(w[SPAN:])
    return np.concatenate(F).astype(np.float32), np.concatenate(Y).astype(np.float32), np.concatenate(Wt).astype(np.float32)


def fit(Ftr, Ytr, Wtr, Fva, Yva, Wva, width, seed, dev):
    mu, sd = Ftr.mean(0), Ftr.std(0); sd = np.where(sd > 1e-8, sd, 1.0)
    T = lambda A: torch.tensor(A, dtype=torch.float32, device=dev)
    X, Y, Wt = T((Ftr - mu) / sd), T(Ytr), T(Wtr); Xv, Yv, Wv = T((Fva - mu) / sd), T(Yva), T(Wva)
    torch.manual_seed(seed); g = torch.Generator(device=dev).manual_seed(seed)
    net = mlp4(X.shape[1], Y.shape[1], width).to(dev); opt = torch.optim.Adam(net.parameters(), CONST["orc_lr"])
    def vrisk():
        with torch.no_grad():
            num = sum(float((((net(Xv[s:s + 65536]) - Yv[s:s + 65536]) ** 2) * Wv[s:s + 65536]).sum()) for s in range(0, len(Xv), 65536))
        return num / float(Wv.sum())
    best, state, bad, log = np.inf, None, 0, []
    for ep in range(SZ["ORC_MAX_EPOCHS"]):
        net.train()
        for _ in range(SZ["ORC_EPOCH_STEPS"]):
            i = torch.randint(0, len(X), (CONST["orc_batch"],), generator=g, device=dev)
            loss = (((net(X[i]) - Y[i]) ** 2) * Wt[i]).sum() / Wt[i].sum().clamp_min(1.0)
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval(); v = vrisk(); log.append(v)
        if v < best - 1e-7:
            best, state, bad = v, {a: b.detach().clone() for a, b in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= SZ["ORC_PATIENCE"]:
                break
    return dict(state=state, mu=mu, sd=sd, din=X.shape[1], dout=Y.shape[1], width=width, val_risk=best, val_log=log)


def load_oracle(path, dev):
    o = torch.load(path, map_location=dev, weights_only=False); net = mlp4(o["din"], o["dout"], o["width"]).to(dev)
    net.load_state_dict(o["state"]); net.eval(); mu, sd = o["mu"], o["sd"]
    def pred(F):
        with torch.no_grad():
            return np.concatenate([net(torch.tensor((F[s:s + 65536] - mu) / sd, dtype=torch.float32, device=dev)).cpu().numpy()
                                   for s in range(0, len(F), 65536)]) if len(F) else np.zeros((0, o["dout"]), np.float32)
    return pred, o


def save_atomic(obj, path):
    torch.save(obj, path + ".tmp"); os.replace(path + ".tmp", path)


def opath(name, k, p, r=None, feat=None):
    return f"{ORC}/{name}_k{float(k)}_p{p}" + (f"_r{r}_{feat}" if r is not None else "") + ".pt"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("kind", choices=["raw", "own1", "own2", "imp", "lat", "sat"])
    ap.add_argument("--kappa", type=float, required=True); ap.add_argument("--p", type=float, required=True)
    ap.add_argument("--rep", type=int); ap.add_argument("--feat", choices=["Bfull", "Fact8"]); a = ap.parse_args()
    dev = device(); os.makedirs(ORC, exist_ok=True); k, p = a.kappa, a.p
    if a.kind in ("raw", "own1", "own2", "imp"):
        tr, va = pool_arrays(k, p, a.kind, "train"), pool_arrays(k, p, a.kind, "val"); res = {}
        for w in CONST["orc_widths"]:
            res[w] = fit(*tr, *va, w, seed_of("p8orc", a.kind, k, p), dev); print("ORACLE", a.kind, k, p, w, res[w]["val_risk"], flush=True)
        wsel = min(res, key=lambda w: res[w]["val_risk"])
        save_atomic({**res[wsel], "kind": a.kind, "selected_on": "pool-val", "val_risks": {w: res[w]["val_risk"] for w in res}, "smoke": SMOKE},
                    opath(a.kind, k, p))
    elif a.kind == "sat":
        wsel = torch.load(opath("raw", k, p), weights_only=False)["width"]; va = pool_arrays(k, p, "raw", "val"); out = {}
        for nt in [n for n in CONST["sat_ntraj"] if n < SZ["NPOOL_TR"]] or [1]:
            out[nt] = fit(*pool_arrays(k, p, "raw", "train", ntraj=nt), *va, wsel, seed_of("p8orc", "raw", k, p), dev)["val_risk"]
        out[SZ["NPOOL_TR"]] = torch.load(opath("raw", k, p), weights_only=False)["val_risk"]
        f = f"{ORC}/sat_k{float(k)}_p{p}.json"; json.dump({"width": wsel, "val_risk_by_ntraj": out, "smoke": SMOKE}, open(f + ".tmp", "w")); os.replace(f + ".tmp", f)
    else:
        wsel = torch.load(opath("raw", k, p), weights_only=False)["width"]; fm = FeatureMap(k, p, a.rep, a.feat, dev)
        res = fit(*pool_arrays(k, p, "lat", "train", fm), *pool_arrays(k, p, "lat", "val", fm), wsel, seed_of("p8orc", "lat", a.feat, k, p, a.rep), dev)
        save_atomic({**res, "kind": "lat", "feat": a.feat, "rep": a.rep, "width_rule": "fixed = selected raw width", "smoke": SMOKE},
                    opath("lat", k, p, a.rep, a.feat))
    print("ORACLE DONE", a.kind, k, p, a.rep, a.feat, flush=True)


if __name__ == "__main__":
    main()
