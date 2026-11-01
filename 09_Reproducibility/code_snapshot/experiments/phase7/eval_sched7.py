"""Evidence-based scheduler for P7-B evaluation: runs evaluate7.py for each (kappa, p, rep) whose 3 checkpoints and oracle
Jacobian file exist and whose acc csv does not; <= NPAR concurrent (GPUs alternate). Log: results/extra/phase7/eval/scheduler.log"""
import os, subprocess, sys, time
from experiments.phase7.core7 import ARMS7, init_seed, ROOT

NPAR = int(sys.argv[1]) if len(sys.argv) > 1 else 4; E = f"{ROOT}/eval"; CK = f"{ROOT}/runs/ckpt"; os.makedirs(E, exist_ok=True)
CELLS = [(k, p, r) for k, p in ((1.0, 0.7), (1.0, 0.3), (0.0, 0.7), (0.0, 0.3)) for r in range(24)]
LOG = open(f"{E}/scheduler.log", "a"); env = dict(os.environ, PYTHONPATH=".pydeps:.", CUDA_DEVICE_ORDER="PCI_BUS_ID", OMP_NUM_THREADS="2")
log = lambda s: (LOG.write(f"[{time.strftime('%H:%M:%S')}] {s}\n"), LOG.flush())
running, done, gi = {}, set(), 0
while True:
    for c, (pr, f) in list(running.items()):
        if pr.poll() is not None:
            log(f"end {c} rc={pr.returncode}"); f.close(); del running[c]; done.add(c)
    todo = [c for c in CELLS if c not in running and c not in done and not os.path.exists(f"{E}/k{c[0]}_p{c[1]}_r{c[2]}_acc.csv")]
    ready = [c for c in todo if os.path.exists(f"{ROOT}/jac/k{c[0]}_r{c[2]}.npz") and
             all(os.path.exists(f"{CK}/k{c[0]}_p{c[1]}_r{c[2]}_{a}_s{init_seed(c[2])}.pt") for a in ARMS7)]
    for c in ready[:max(0, NPAR - len(running))]:
        e = dict(env, CUDA_VISIBLE_DEVICES=str(gi % 2)); gi += 1; f = open(f"{E}/log_k{c[0]}_p{c[1]}_r{c[2]}.txt", "w")
        running[c] = (subprocess.Popen([sys.executable, "experiments/phase7/evaluate7.py", "--kappa", str(c[0]), "--p", str(c[1]), "--rep", str(c[2])],
                                       stdout=f, stderr=subprocess.STDOUT, env=e), f); log(f"start {c} gpu {e['CUDA_VISIBLE_DEVICES']}")
    lanes = subprocess.run(["pgrep", "-f", "run_lane7.sh"], capture_output=True).stdout.strip() != b""
    if not todo and not running:
        log("SCHEDULER EXIT all done"); break
    if not lanes and not running and not ready:
        log(f"SCHEDULER EXIT lanes finished, {len(todo)} cells not runnable"); break
    time.sleep(30)
