#!/bin/bash
# Overnight queue (start with setsid nohup; independent of SSH / Claude Code).
#  1. Exp A (GPU 0): Lorenz tau in {5,10,20} x {B,C} x p {0.3,0.7} x seeds 0-2 -> results/extra/expA_tau
#     (completed exp_ids are skipped; no run starts if it could pass the GPU deadline;
#      the independent watchdog terminates anything still running at the deadline)
#  2. Exp B, C, D (CPU only, evaluation of saved checkpoints) -> results/extra/exp{B,C,D}_*/overnight
#     (the scripts refuse to overwrite existing outputs)
#  3. waits for every training process of this project to finish (main grids on GPU 1 + Exp A)
#  4. CPU post-processing: Exp A analysis, checkpoint recomputation, verify_results, figures,
#     overnight_status.txt; then creates results/logs/QUEUE_DONE
# Each step logs its exit code; a failing step does not stop later independent steps.
DEADLINE_STR="${1:-2026-10-25 05:55}"
DEADLINE=$(date -d "$DEADLINE_STR" +%s) || exit 1
PROJ="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ"
LOG=results/logs/overnight_queue.log
PY="$HOME/.conda/envs/reef_gpu/bin/python"
export PYTHONPATH=.pydeps CUDA_DEVICE_ORDER=PCI_BUS_ID
log() { echo "$(date '+%F %T') [queue $$] $*" >> "$LOG"; }
status() { CUDA_VISIBLE_DEVICES="" nice -n 10 "$PY" -W ignore experiments/extra/overnight_status.py > /dev/null 2>> "$LOG"; }
cpu_step() {  # cpu_step <name> <script> [args]
  local name=$1; shift
  log "== $name start"
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 IMD_EXTRA_RUN=overnight nice -n 10 "$PY" -W ignore "$@" >> "$LOG" 2>&1
  log "== $name exit=$?"
  status
}
training_pids() {  # same detection as the watchdog
  local d p exe arg1
  for d in /proc/[0-9]*; do
    p=${d#/proc/}
    [ "$(stat -c %u "$d" 2>/dev/null)" = "$(id -u)" ] || continue
    exe=$(readlink "$d/exe" 2>/dev/null) || continue
    case "$(basename "$exe")" in python|python3|python3.*) ;; *) continue ;; esac
    [ "$(readlink "$d/cwd" 2>/dev/null)" = "$PROJ" ] || continue
    arg1=$(tr '\0' '\n' < "$d/cmdline" 2>/dev/null | sed -n 2p)
    case "$arg1" in experiments/run_lorenz.py|experiments/run_coupled_lorenz.py|experiments/run_real_data.py) echo "$p" ;; esac
  done
}

# refuse to start twice
exec 9> results/logs/overnight_queue.lock
if ! flock -n 9; then echo "$(date '+%F %T') another overnight_queue is running; exit" >> "$LOG"; exit 1; fi
log "start; GPU deadline $DEADLINE_STR"
# keep a copy of derived files that post-processing regenerates (parts/checkpoints/logs are never modified)
BK="results/backup_pre_overnight_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BK"
cp -p results/*.csv "$BK"/ 2>/dev/null
cp -rp results/figures results/verification "$BK"/ 2>/dev/null
log "backup of derived files -> $BK ($(find "$BK" -type f | wc -l) files)"
status

# ---- 1. Exp A on GPU 0
EA=results/extra/expA_tau
mkdir -p "$EA"; [ -e "$EA/data" ] || ln -s ../../data "$EA/data"
APIDS=""
if [ -e results/logs/STOP_GPU ]; then
  log "STOP_GPU present: Exp A not started"
elif [ $(( $(date +%s) + 3600 )) -ge "$DEADLINE" ]; then
  log "less than 1 h to the GPU deadline: Exp A not started"
elif ! labres-check 2>/dev/null | grep -q "GPU 0 · .*06:00"; then
  log "GPU 0 reservation until 06:00 not confirmed by labres-check: Exp A not started"
else
  for tau in 5 10 20; do
    IMD_RESULTS_DIR="$PROJ/$EA" IMD_GPU_DEADLINE=$DEADLINE IMD_RUN_BUDGET_S=1500 \
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
      "$PY" experiments/run_lorenz.py --tau $tau --m 8 --epochs 150 --patience 15 --models B,C \
      --rates 0.3,0.7 --seeds 0,1,2 --baselines "" --tag _tau$tau --out lorenz_tau$tau.csv \
      > "$EA/run_tau$tau.txt" 2>&1 < /dev/null &
    APIDS="$APIDS $!"
  done
  log "Exp A started on GPU 0, pids:$APIDS"
fi

# ---- 2. CPU experiments (evaluation only)
cpu_step "Exp B contiguous gaps" experiments/extra/exp_b_contiguous.py
cpu_step "Exp C cross-view ablation" experiments/extra/exp_c_crossview.py
cpu_step "Exp D observability sensitivity" experiments/extra/exp_d_observability.py

# ---- 3. wait for all training (Exp A children + main grids started earlier)
for p in $APIDS; do wait "$p"; log "Exp A process $p exit=$?"; done
while [ -n "$(training_pids)" ]; do
  [ -z "$waited" ] && log "waiting for training pids: $(training_pids | tr '\n' ' ')" && waited=1
  sleep 60
done
log "no training processes left"

# ---- 4. post-processing (CPU)
cpu_step "Exp A analysis" experiments/extra/analyze_exp_a.py
cpu_step "recompute main results from checkpoints" experiments/recompute_from_checkpoints.py
cpu_step "verify_results" experiments/verify_results.py
cpu_step "make_figures" experiments/make_figures.py
status
touch results/logs/QUEUE_DONE
log "done"
