"""Phase 9 evidence-based scheduler (same principles as sched8): a task runs when its dependency files exist and its output marker
does not; rc != 0 -> permanent failure marker (never retried, dependents never start); DEADLINE_EPOCH (no new task) and
HARD_DEADLINE_EPOCH (terminate running tasks; not failures, resumable) are required for real runs; exclusive lock; GPUs from env
GPUS (CUDA_DEVICE_ORDER=PCI_BUS_ID set by the caller). Priority: p = 0.7 before p = 0.3; evaluations before further training.
usage: sched9.py STAGE NPAR"""
import fcntl, os, signal, subprocess, sys, time
from experiments.phase9.core9 import ROOT, RUNS, EVAL, SZ, ARMS9, CONST, init_seed, SMOKE


def tasks(stage):
    T = []
    for p in sorted(CONST["ps"], reverse=True):
        for k in CONST["kappas"]:
            for r in range(SZ["R"][stage]):
                rows = [f"{RUNS}/rows/{stage}_k{k}_p{p}_r{r}_{a}_s{init_seed(a, stage, r)}.csv" for a in ARMS9]
                T.append((0, f"eval:{stage}:{k}:{p}:{r}", ["experiments/phase9/eval9.py", "--stage", stage, "--kappa", str(k), "--p", str(p), "--rep", str(r)],
                          rows, f"{EVAL}/{stage}_k{k}_p{p}_r{r}_acc.csv.gz"))
                for a, f in zip(ARMS9, rows):
                    T.append((1, f"train:{stage}:{k}:{p}:{r}:{a}", ["experiments/phase9/train9.py", "--stage", stage, "--kappa", str(k), "--p", str(p), "--rep", str(r),
                                                                     "--arm", a], [], f))
    return T


def main():
    stage, npar = sys.argv[1], int(sys.argv[2]); gpus = [""] if SMOKE else os.environ.get("GPUS", "0").split(",")
    dl = float(os.environ.get("DEADLINE_EPOCH", "inf")); hard = float(os.environ.get("HARD_DEADLINE_EPOCH", "inf"))
    if not SMOKE and (dl == float("inf") or hard <= dl):
        sys.exit("REFUSED: real runs need DEADLINE_EPOCH < HARD_DEADLINE_EPOCH")
    FD = f"{ROOT}/failed"; os.makedirs(FD, exist_ok=True); os.makedirs(f"{ROOT}/logs", exist_ok=True)
    lk = open(f"{ROOT}/sched9.lock", "w")
    try:
        fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit("REFUSED: another sched9 holds the lock")
    LOG = open(f"{ROOT}/logs/sched9_{stage}.log", "a"); log = lambda s: (LOG.write(f"[{time.strftime('%m-%d %H:%M:%S')}] {s}\n"), LOG.flush())
    fm = lambda nm: f"{FD}/{nm.replace(':', '_')}.failed"
    T = tasks(stage); env = dict(os.environ, PYTHONPATH=".pydeps:.", CUDA_DEVICE_ORDER="PCI_BUS_ID", OMP_NUM_THREADS="2"); running, gi = {}, 0
    log(f"START {stage} {len(T)} tasks npar={npar} gpus={gpus} done={sum(os.path.exists(t[4]) for t in T)} failed={sum(os.path.exists(fm(t[1])) for t in T)}")
    while True:
        now = time.time()
        for nm, (pr, fh) in list(running.items()):
            if now > hard and pr.poll() is None:
                pr.send_signal(signal.SIGTERM); log(f"killed {nm} (hard deadline; resumable)")
            if pr.poll() is not None:
                rc = pr.returncode; fh.close(); del running[nm]; log(f"end {nm} rc={rc}")
                if rc not in (0, -15) and not (now > hard):
                    open(fm(nm), "w").write(f"rc={rc}\n")
        todo = [t for t in T if not os.path.exists(t[4]) and not os.path.exists(fm(t[1])) and t[1] not in running]
        blocked = lambda t: any(os.path.exists(fm(f"train:{t[1].split(':', 1)[1]}:{a}")) for a in ARMS9) if t[1].startswith("eval") else False
        ready = sorted([t for t in todo if all(os.path.exists(d) for d in t[3]) and not blocked(t)], key=lambda t: t[0])
        if not running and (not todo or not ready or now > dl):        # never exit while a child is still running
            break
        if now <= dl:
            for pri, nm, cmd, _, _ in ready[:max(0, npar - len(running))]:
                g = gpus[gi % len(gpus)]; gi += 1; e = dict(env, CUDA_VISIBLE_DEVICES=g) if g != "" else env
                fh = open(f"{ROOT}/logs/{nm.replace(':', '_')}.log", "a"); running[nm] = (subprocess.Popen([sys.executable] + cmd, env=e, stdout=fh, stderr=subprocess.STDOUT), fh)
                log(f"start {nm} gpu {g}")
        time.sleep(1 if SMOKE else 10)
    left = [t[1] for t in T if not os.path.exists(t[4])]; failed = [t[1] for t in T if os.path.exists(fm(t[1]))]
    log(f"EXIT {stage} todo={len(left)} failed={failed} deadline_passed={time.time() > dl}")


if __name__ == "__main__":
    main()
