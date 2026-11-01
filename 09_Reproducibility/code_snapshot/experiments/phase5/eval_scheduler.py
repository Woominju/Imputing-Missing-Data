"""Launches experiments/phase5/evaluate.py for every (kappa, p, rep[, seed]) whose training rows all exist (evidence-based,
session-independent; at most MAXC concurrent evaluations, alternating GPUs 0/1). Stops when all targets are evaluated or after
deadline. Log: results/extra/phase5/eval/scheduler.log"""
import os, subprocess, time, datetime
R = "/home/minju/faicpaston/Imputing_Missing_Data"; os.chdir(R)
PY = os.path.expanduser("~/.conda/envs/reef_gpu/bin/python"); E = "results/extra/phase5/eval"; RUNS = "results/extra/phase5/runs"
ARMS = ["A1", "A2", "B1", "B2", "Bcur1", "Bcur2", "C", "Sham1", "Sham2"]; MAXC = 10; DEADLINE = "15:30"
T = [(k, p, r, r, ARMS) for r in range(8) for k in (1.0, 0.0) for p in (0.7, 0.3)] + \
    [(k, 0.7, 0, s, ["A1", "A2", "B1", "B2"]) for k in (1.0, 0.0) for s in (1, 2)]
os.makedirs(E, exist_ok=True); running = {}; done = set(); n = 0
def log(m):
    open(f"{E}/scheduler.log", "a").write(f"[{datetime.datetime.now():%H:%M:%S}] {m}\n")
while len(done) < len(T) and datetime.datetime.now().strftime("%H:%M") < DEADLINE:
    for key, pr in list(running.items()):
        if pr.poll() is not None:
            log(f"end {key} rc={pr.returncode}"); done.add(key); del running[key]
    for (k, p, r, s, arms) in T:
        key = (k, p, r, s)
        if key in done or key in running or len(running) >= MAXC:
            continue
        if os.path.exists(f"{E}/k{k}_p{p}_r{r}_s{s}_imp.csv"):
            done.add(key); continue
        if all(os.path.exists(f"{RUNS}/rows/k{k}_p{p}_r{r}_{a}_s{s}.csv") for a in arms):
            env = dict(os.environ, PYTHONPATH=".pydeps:.", CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(n % 2), OMP_NUM_THREADS="2"); n += 1
            cmd = [PY, "experiments/phase5/evaluate.py", "--kappa", str(k), "--p", str(p), "--rep", str(r), "--seed", str(s), "--arms", ",".join(arms)]
            running[key] = subprocess.Popen(cmd, env=env, stdout=open(f"{E}/eval_k{k}_p{p}_r{r}_s{s}.log", "w"), stderr=subprocess.STDOUT)
            log(f"start {key}")
    time.sleep(20)
log(f"SCHEDULER EXIT done={len(done)}/{len(T)}")
