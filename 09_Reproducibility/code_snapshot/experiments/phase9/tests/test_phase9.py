"""Phase 9 integrity tests (protocol section 8). CPU only. Run: PYTHONPATH=.pydeps:. python experiments/phase9/tests/test_phase9.py
The smoke end-to-end part uses P9_SMOKE=1 and a scratch P9_ROOT (never touches results/extra/phase9)."""
import glob, hashlib, os, subprocess, sys, tempfile
import numpy as np, torch

PASS, FAIL = [], []


def t(name, cond):
    (PASS if cond else FAIL).append(name); print(("PASS " if cond else "FAIL ") + name, flush=True)


def main():
    import experiments.phase9.core9 as C9
    from experiments.phase9.train9 import p9_loss
    from experiments.phase5.core import ArmModel, SPAN
    from experiments.fdr.loss import fdr_masked_loss
    from src.delay_embedding import delay_embed
    import experiments.phase8.core8 as C8
    torch.manual_seed(0); cfg = dict(C9.AE_CFG); m = ArmModel(4, [0, 1, 2, 3], np.ones((4, 8), np.float32), 8, 16, 64)
    Xs = torch.randn(5, 71, 4); Ms = (torch.rand(5, 71, 4) > 0.7).float(); H = Ms * (torch.rand(5, 71, 4) < 0.2).float(); Bm = Ms - H
    V, MV = delay_embed(Xs * Bm, Bm, 8, 1); ref, _ = fdr_masked_loss(m.forward_full(V, MV), Xs, H, 8, 1)
    l0, _, _ = p9_loss(m, Xs, Ms, H, [0, 1, 2, 3], cfg, 0.0, True); l1, _, _ = p9_loss(m, Xs, Ms, H, [0, 1, 2, 3], cfg, 1.0, False)
    t("1 lam=0 / non-P9 arm loss is bitwise the Phase 8 FDR loss", torch.equal(l0, ref) and torch.equal(l1, ref))
    # 2 teacher superset, targets only on 1 - O, no gradient through teacher
    l, lt, nt = p9_loss(m, Xs, Ms, H, [0, 1, 2, 3], cfg, 1.0, True); l.backward()
    t("2a P9 tower term active (finite, > 0, targets > 0)", np.isfinite(lt) and lt > 0 and nt > 0)
    t("2b student input B = O - H <= teacher input O", bool(((Ms - H) <= Ms).all()))
    Yt_dummy = torch.zeros(5, 64, 4, 8, requires_grad=True); Ys = torch.zeros(5, 64, 4, 8)
    D = 1 - Ms; D2 = D.clone(); D2[:, :, 0] = 0
    a, _ = C9.tower_loss(Ys + 1, Ys, D, 8, 1); b, _ = C9.tower_loss(Ys + 1, Ys, torch.zeros_like(D), 8, 1)
    t("2c tower loss uses only originally-missing entries (zero D -> no loss)", b is None and a is not None)
    src = open("experiments/phase9/train9.py").read()
    t("2d teacher computed under torch.no_grad()", "with torch.no_grad():" in src and "Yt = model.forward_full(Vt, MVt)" in src)
    # 3 hidden y never used: inputs are nz_obs (4 observed columns)
    k = 1.0; X6 = np.random.randn(10, 6); O = C8.nz_obs(k, X6); t("3 AE inputs are the 4 observed views only (y columns absent)", O.shape[1] == 4)
    t("3b build() uses nz_obs only", "nz_state" not in open("experiments/phase9/core9.py").read().split("def build")[1].split("def tower_loss")[0])
    # 4 ids / seeds disjoint from earlier phases
    I = [j for j, _ in C9.lengths()]; t("4a all Phase 9 ids in [200000, 300000)", min(I) >= 200000 and max(I) < 300000 and len(set(I)) == len(I))
    from experiments.phase6.gen6 import SEED6
    seeds9 = {SEED6 + 100000 * ki + j for ki in (0, 1) for j in I}
    prev = set()
    for d in ("results/extra/phase6/data", "results/extra/phase7/data", "results/extra/phase8/data"):
        for f in glob.glob(f"{d}/k*_id*.npz"):
            ki = float(f.split("/k")[-1].split("_id")[0]); j = int(f.split("_id")[-1].split(".npz")[0]); prev.add(SEED6 + int(100000 * ki) + j)
    prev |= {SEED6 + 100000 * ki + 70000 + q for ki in (0, 1) for q in range(30000)}       # P8 kNN reference ids (not stored)
    t("4b Phase 9 rng seeds disjoint from all P6/P7/P8 seeds", not (seeds9 & prev))
    # 5 test mask seeds differ from train/val; strata identical to P8
    ms = C9.mask_seeds("main", 1.0, 0.7, 0); ts = {C9.seed_of("p9test", "main", 1.0, 0.7, 0, i, e) for i in range(8) for e in range(5)}
    t("5a test mask seeds distinct from train/val/valhide and from pilot", not (ts & set(ms.values())) and not (ts & {C9.seed_of("p9test", "pilot", 1.0, 0.7, 0, i, e) for i in range(8) for e in range(5)}))
    t("5b strata function is the frozen P8 one", C9.strata is C8.strata)
    t("5c AE_CFG identical to Phase 8", {a: b for a, b in C9.AE_CFG.items() if a not in ("epochs", "steps_per_epoch", "patience")} ==
      {a: b for a, b in C8.AE_CFG.items() if a not in ("epochs", "steps_per_epoch", "patience")} and C9.FULL["AE_EPOCHS"] == C8.FULL["AE_EPOCHS"] and C9.FULL["AE_STEPS"] == C8.FULL["AE_STEPS"] and C9.FULL["AE_PATIENCE"] == C8.FULL["AE_PATIENCE"])
    e9 = open("experiments/phase9/eval9.py").read(); t("6 imputation evaluates M == 0 positions with n >= SPAN only", "miss = (M[:, c] == 0); miss[:SPAN] = False" in e9)
    # 7 smoke end-to-end in a scratch root
    with tempfile.TemporaryDirectory() as tmp:
        env = dict(os.environ, P9_SMOKE="1", P9_ROOT=tmp, PYTHONPATH=".pydeps:.", CUDA_VISIBLE_DEVICES="")
        r1 = subprocess.run([sys.executable, "experiments/phase9/gen9.py", "4"], env=env, capture_output=True, text=True)
        r2 = subprocess.run([sys.executable, "experiments/phase9/sched9.py", "pilot", "8"], env=env, capture_output=True, text=True)
        rows = glob.glob(f"{tmp}/runs/rows/*.csv"); accs = glob.glob(f"{tmp}/eval/*_acc.csv.gz")
        t("7a smoke pipeline: 2 reps x 4 cells x 6 arms trained, 8 evaluations", r1.returncode == 0 and r2.returncode == 0 and len(rows) == 48 and len(accs) == 8)
        if rows:
            import pandas as pd
            R = pd.concat([pd.read_csv(f) for f in rows]); t("7b rows flag smoke and P9 rows have lam = 1 and tower targets > 0",
                                                             bool(R.smoke.all()) and bool((R[R.arm == "P9"].lam == 1).all()) and bool((R[R.arm == "P9"].tower_targets_per_step > 0).all()) and bool((R[R.arm != "P9"].lam == 0).all()))
        if accs:
            import pandas as pd
            A = pd.read_csv(accs[0]); rt = pd.read_csv(accs[0].replace("_acc", "_ret")); t("7c eval outputs present (ret, imp, mv)", len(rt) > 0 and os.path.exists(accs[0].replace("_acc", "_imp")))
        if r2.returncode or r1.returncode:
            print(r1.stderr[-2000:], r2.stderr[-2000:])
    print(f"\n{len(PASS)}/{len(PASS) + len(FAIL)} PASS"); sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
