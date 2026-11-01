"""Evidence-based evaluation scheduler for Phase 6 G2/G3: runs evaluate6.py for every (kappa, p, rep) whose 10 checkpoints
exist and whose g2 csv does not, with at most NPAR concurrent evaluations (alternating GPUs 0/1). Exits when all 96 are done
or when no training lane is alive and nothing remains runnable. Log: results/extra/phase6/eval/scheduler.log"""
import os, subprocess, sys, time, glob
from experiments.phase6.core6 import ARMS6

NPAR = int(sys.argv[1]) if len(sys.argv) > 1 else 4; E = "results/extra/phase6/eval"; CK = "results/extra/phase6/runs/ckpt"
CELLS = [(k, p, r) for k, p in ((1.0, 0.7), (0.0, 0.7), (1.0, 0.3), (0.0, 0.3)) for r in range(24)]
LOG = open(f"{E}/scheduler.log", "a"); env = dict(os.environ, PYTHONPATH=".pydeps:.", CUDA_DEVICE_ORDER="PCI_BUS_ID", OMP_NUM_THREADS="2")
log = lambda s: (LOG.write(f"[{time.strftime('%H:%M:%S')}] {s}\n"), LOG.flush())
running, done, gi = {}, set(), 0
while True:
    for c, (pr, f) in list(running.items()):
        if pr.poll() is not None:
            log(f"end {c} rc={pr.returncode}"); f.close(); del running[c]; done.add(c)
    todo = [c for c in CELLS if c not in running and c not in done and not os.path.exists(f"{E}/k{c[0]}_p{c[1]}_r{c[2]}_g2.csv")]
    ready = [c for c in todo if all(os.path.exists(f"{CK}/k{c[0]}_p{c[1]}_r{c[2]}_{a}_s{6000 + c[2]}.pt") for a in ARMS6)]
    for c in ready[:max(0, NPAR - len(running))]:
        e = dict(env, CUDA_VISIBLE_DEVICES=str(gi % 2)); gi += 1
        f = open(f"{E}/log_k{c[0]}_p{c[1]}_r{c[2]}.txt", "w")
        running[c] = (subprocess.Popen([sys.executable, "experiments/phase6/evaluate6.py", "--kappa", str(c[0]), "--p", str(c[1]), "--rep", str(c[2])], stdout=f, stderr=subprocess.STDOUT, env=e), f)
        log(f"start {c} gpu {e['CUDA_VISIBLE_DEVICES']}")
    lanes_alive = subprocess.run(["pgrep", "-f", "run_lane6.sh"], capture_output=True).stdout.strip() != b""
    if not todo and not running:
        log(f"SCHEDULER EXIT all done"); break
    if not lanes_alive and not running and not ready:
        log(f"SCHEDULER EXIT lanes finished, {len(todo)} cells not runnable"); break
    time.sleep(30)
