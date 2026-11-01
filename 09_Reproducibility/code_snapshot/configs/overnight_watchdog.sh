#!/bin/bash
# Independent GPU-deadline guard (start with setsid nohup; survives SSH / Claude Code exit).
# At DEADLINE it creates results/logs/STOP_GPU (queue stops launching GPU work) and terminates
# every training process of THIS project: a python executable whose argv[1] is
# experiments/run_*.py (or experiments/extra/*.py with --gpu) and whose cwd is this project.
# SIGTERM first, SIGKILL after 60 s. Completed CSV rows / checkpoints / logs are untouched.
# Exits after the deadline is handled, or earlier once results/logs/QUEUE_DONE exists and no
# training process is left.
# Usage: setsid nohup configs/overnight_watchdog.sh "YYYY-mm-dd HH:MM" >> results/logs/watchdog.log 2>&1 < /dev/null &
DEADLINE=$(date -d "$1" +%s) || exit 1
PROJ="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ"
log() { echo "$(date '+%F %T') [watchdog $$] $*"; }

training_pids() {
  local d p exe arg1
  for d in /proc/[0-9]*; do
    p=${d#/proc/}
    [ "$(stat -c %u "$d" 2>/dev/null)" = "$(id -u)" ] || continue
    exe=$(readlink "$d/exe" 2>/dev/null) || continue
    case "$(basename "$exe")" in python|python3|python3.*) ;; *) continue ;; esac
    [ "$(readlink "$d/cwd" 2>/dev/null)" = "$PROJ" ] || continue
    arg1=$(tr '\0' '\n' < "$d/cmdline" 2>/dev/null | sed -n 2p)
    case "$arg1" in experiments/run_lorenz.py|experiments/run_coupled_lorenz.py|experiments/run_real_data.py|experiments/extra/run_exp_a.py) echo "$p" ;; esac
  done
}

log "start; deadline $1 ($DEADLINE); project $PROJ; current training pids: $(training_pids | tr '\n' ' ')"
last=0
while true; do
  now=$(date +%s)
  pids=$(training_pids | tr '\n' ' ')
  if [ "$now" -ge "$DEADLINE" ]; then
    touch results/logs/STOP_GPU
    log "DEADLINE reached; STOP_GPU created; SIGTERM: [$pids]"
    [ -n "$pids" ] && kill -TERM $pids 2>/dev/null
    sleep 60
    pids=$(training_pids | tr '\n' ' ')
    if [ -n "$pids" ]; then log "still alive, SIGKILL: [$pids]"; kill -KILL $pids 2>/dev/null; sleep 5; fi
    log "after deadline: training pids [$(training_pids | tr '\n' ' ')]; gpu apps: $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>&1 | tr '\n' ' ')"
    break
  fi
  if [ -e results/logs/QUEUE_DONE ] && [ -z "$pids" ]; then
    log "queue done and no training processes left; exiting before deadline"
    break
  fi
  if [ $((now - last)) -ge 900 ]; then log "alive; training pids [$pids]"; last=$now; fi
  sleep 30
done
log "exit"
