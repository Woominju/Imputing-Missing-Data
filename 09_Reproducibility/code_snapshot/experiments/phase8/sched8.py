"""Phase 8 evidence-based scheduler (Gate D only; protocol v2 section 11). Orchestration only: no effect on any estimand.
* A task starts only when all dependency FILES exist, its output marker does not, and it has no permanent failure marker.
* Every task writes its output marker LAST and atomically; invalid AE runs write no marker at all (train8).
* A task that ends with rc != 0 gets a permanent failure marker ROOT/failed/<task>.txt and is NEVER retried (not in this run,
  not after a restart); its dependants never start. A task terminated at the hard deadline is NOT a failure: no marker, it is
  resumed by the next scheduler run (outputs of finished tasks are never recomputed -> no duplication).
* Reservation guards: DEADLINE_EPOCH (no new task) < HARD_DEADLINE_EPOCH (terminate running tasks), both required in real runs.
* Exclusive lock ROOT/sched8.lock (fcntl): a second scheduler on the same ROOT is refused.
* Priority (list order = preference among READY tasks), tiers:
    T1  p = 0.7 tasks required by the confirmatory evaluation: pool oracles (raw, own1, own2, imp), O_lat, evaluation, AE training
    T2  p = 0.7 exploratory / descriptive extras not needed by evaluation: saturation, kNN (D5a), D5b
    T3  p = 0.3 required tasks (all p = 0.3 analyses are descriptive)      T4  p = 0.3 extras      T5  Phase 7 imputation (Family III)
  Within a tier: oracles, then downstream (O_lat, eval), then training, so downstream takes free slots as soon as it is ready.
* On every exit the (CPU) analysis analyze8 runs; it writes a NEW partial_* or complete_* directory when its inputs changed and
  never overwrites an earlier one.
GPUs from env GPUS (default "0,1"), CUDA_DEVICE_ORDER=PCI_BUS_ID; P8_SMOKE=1 forces CPU.
usage: sched8.py NPAR [--dry]"""
import fcntl, os, signal, subprocess, sys, time
from experiments.phase8.core8 import ROOT, RUNS, ORC, EVAL, DDIR, SZ, CONST, ARMS8, init_seed
from experiments.phase8.oracle8 import opath

PY = sys.executable; P_ORDER = (0.7, 0.3); FAILED = f"{ROOT}/failed"


def row(k, p, r, arm):
    return f"{RUNS}/rows/k{k}_p{p}_r{r}_{arm}_s{init_seed(arm, r)}.csv"


def tasks_tiered():
    """-> list of (tier, task) with task = (name, argv, dependency files, output marker)."""
    out = []; R = range(SZ["R"])
    for pi, p in enumerate(P_ORDER):
        t_req, t_ext = 1 + 2 * pi, 2 + 2 * pi; orc, down, train, ext = [], [], [], []
        for k in CONST["kappas"]:
            for kind in ("raw", "own1", "own2", "imp"):
                orc.append((f"orc:{kind}:{k}:{p}", ["experiments/phase8/oracle8.py", kind, "--kappa", str(k), "--p", str(p)], [], opath(kind, k, p)))
            ref = f"{DDIR}/ref_k{float(k)}.npz"
            ext.append((f"orc:sat:{k}:{p}", ["experiments/phase8/oracle8.py", "sat", "--kappa", str(k), "--p", str(p)], [opath("raw", k, p)], f"{ORC}/sat_k{float(k)}_p{p}.json"))
            ext.append((f"knn:eval:{k}:{p}", ["experiments/phase8/knn8.py", "eval", "--kappa", str(k), "--p", str(p)], [opath("raw", k, p), ref], f"{EVAL}/knn_k{float(k)}_p{p}.csv"))
            ext.append((f"knn:d5b:{k}:{p}", ["experiments/phase8/knn8.py", "d5b", "--kappa", str(k), "--p", str(p)], [ref], f"{EVAL}/d5b_k{float(k)}_p{p}.csv"))
        for r in R:                                      # replicate-major: complete replicates appear early
            for k in CONST["kappas"]:
                for feat, arms in (("Bfull", ["Bfull"]), ("Fact8", ["A1_8", "A2_8"])):
                    down.append((f"orc:lat:{k}:{p}:{r}:{feat}", ["experiments/phase8/oracle8.py", "lat", "--kappa", str(k), "--p", str(p), "--rep", str(r), "--feat", feat],
                                 [opath("raw", k, p)] + [row(k, p, r, a) for a in arms], opath("lat", k, p, r, feat)))
                deps = [row(k, p, r, a) for a in ARMS8] + [opath(x, k, p) for x in ("raw", "own1", "own2", "imp")] + [opath("lat", k, p, r, f) for f in ("Bfull", "Fact8")]
                down.append((f"eval:{k}:{p}:{r}", ["experiments/phase8/evaluate8.py", "--kappa", str(k), "--p", str(p), "--rep", str(r)], deps, f"{EVAL}/k{k}_p{p}_r{r}_acc.csv.gz"))
                for arm in ARMS8:
                    train.append((f"train:{k}:{p}:{r}:{arm}", ["experiments/phase8/train8.py", "--kappa", str(k), "--p", str(p), "--rep", str(r), "--arm", arm], [], row(k, p, r, arm)))
        out += [(t_req, t) for t in orc + down + train] + [(t_ext, t) for t in ext]
    for p in P_ORDER:
        for k in CONST["kappas"]:
            for r in range(0 if os.environ.get("P8_SMOKE") == "1" else 24):
                out.append((5, (f"p7imp:{k}:{p}:{r}", ["experiments/phase8/eval_p7imp8.py", "--kappa", str(k), "--p", str(p), "--rep", str(r)], [], f"{EVAL}/p7imp_k{k}_p{p}_r{r}.csv.gz")))
    return sorted(out, key=lambda z: z[0])               # stable sort: keeps the within-tier order


def tasks():
    return [t for _, t in tasks_tiered()]


def fmarker(nm):
    return f"{FAILED}/{nm.replace(':', '_')}.txt"


def run_analysis(env, log):
    r = subprocess.run([PY, "experiments/phase8/analyze8.py"], env=dict(env, CUDA_VISIBLE_DEVICES=""), capture_output=True, text=True)
    log(f"analysis rc={r.returncode} {r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ''}")
    if r.returncode != 0:
        log("analysis stderr: " + r.stderr[-1500:].replace("\n", " | "))


def main():
    npar = int(sys.argv[1]); dry = "--dry" in sys.argv; smoke = os.environ.get("P8_SMOKE") == "1"
    gpus = [""] if smoke else os.environ.get("GPUS", "0,1").split(",")
    dl = float(os.environ.get("DEADLINE_EPOCH", "inf")); hard = float(os.environ.get("HARD_DEADLINE_EPOCH", "inf"))
    T = tasks()
    if dry:                                                   # no side effects
        print(len(T), "tasks"); [print(n, "|", " ".join(c), "| deps", len(d), "| out", o) for n, c, d, o in T]; return
    if not smoke and (dl == float("inf") or hard == float("inf") or hard <= dl):
        sys.exit("REFUSED: real runs need DEADLINE_EPOCH < HARD_DEADLINE_EPOCH (both before the GPU reservation end)")
    os.makedirs(f"{ROOT}/logs", exist_ok=True); os.makedirs(FAILED, exist_ok=True)
    lockf = open(f"{ROOT}/sched8.lock", "w")
    try:
        fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit("REFUSED: another sched8 holds the Phase 8 run lock")
    lockf.write(str(os.getpid())); lockf.flush()
    LOG = open(f"{ROOT}/logs/sched8.log", "a"); log = lambda s: (LOG.write(f"[{time.strftime('%m-%d %H:%M:%S')}] {s}\n"), LOG.flush())
    env = dict(os.environ, PYTHONPATH=".pydeps:.", CUDA_DEVICE_ORDER="PCI_BUS_ID", OMP_NUM_THREADS="2"); running, gi = {}, 0
    prev_failed = sorted(nm for nm, _, _, _ in T if os.path.exists(fmarker(nm)))
    log(f"START {len(T)} tasks npar={npar} gpus={gpus} deadline={dl} hard={hard} done={sum(os.path.exists(t[3]) for t in T)} permanently_failed={len(prev_failed)}")
    while True:
        for nm, (pr, fh) in list(running.items()):
            if pr.poll() is not None:
                fh.close(); del running[nm]; log(f"end {nm} rc={pr.returncode}")
                if pr.returncode != 0:
                    open(fmarker(nm), "w").write(f"{time.strftime('%F %T')} rc={pr.returncode} (permanent; dependants will not start)\n")
        if time.time() >= hard:
            for nm, (pr, fh) in running.items():
                pr.send_signal(signal.SIGTERM); log(f"killed {nm} (hard deadline; not a failure, resumable)")
            t0 = time.time()
            while any(pr.poll() is None for pr, _ in running.values()) and time.time() - t0 < 30:
                time.sleep(1)
            for nm, (pr, fh) in running.items():
                if pr.poll() is None:
                    pr.kill()
                fh.close()
            log(f"EXIT hard deadline; killed={sorted(running)}"); running = {}; break
        todo = [t for t in T if t[0] not in running and not os.path.exists(t[3]) and not os.path.exists(fmarker(t[0]))]
        ready = [t for t in todo if all(os.path.exists(d) for d in t[2])]
        if time.time() < dl:
            for nm, cmd, _, _ in ready[:max(0, npar - len(running))]:
                g = gpus[gi % len(gpus)]; gi += 1; fh = open(f"{ROOT}/logs/{nm.replace(':', '_')}.log", "w")
                running[nm] = (subprocess.Popen([PY] + cmd, stdout=fh, stderr=subprocess.STDOUT, env=dict(env, CUDA_VISIBLE_DEVICES=g)), fh); log(f"start {nm} gpu {g}")
        if not running and (not ready or time.time() >= dl):
            failed = sorted(nm for nm, _, _, _ in T if os.path.exists(fmarker(nm)))
            log(f"EXIT todo={len(todo)} failed={failed} deadline_passed={time.time() >= dl}"); break
        time.sleep(1 if smoke else 20)
    run_analysis(env, log)


if __name__ == "__main__":
    main()
