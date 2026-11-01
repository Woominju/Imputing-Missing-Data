#!/bin/bash
# usage: run_lane.sh GPU JOBFILE LANELOG   (each job line: "<train args>"); guard: no job starts after DEADLINE (HH:MM)
GPU=$1; JOBS=$2; LOG=$3; DEADLINE=${DEADLINE:-15:20}
cd /home/minju/faicpaston/Imputing_Missing_Data
export PYTHONPATH=.pydeps:. CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=2
PY=/home/minju/.conda/envs/reef_gpu/bin/python
while IFS= read -r job; do
  [ -z "$job" ] && continue
  if [[ "$(date +%H:%M)" > "$DEADLINE" ]]; then echo "[$(date +%T)] skip $job: past deadline" >> "$LOG"; continue; fi
  echo "[$(date +%T)] start $job" >> "$LOG"
  $PY experiments/phase5/train.py $job --out results/extra/phase5/runs >> "$LOG" 2>&1; rc=$?
  echo "[$(date +%T)] end $job rc=$rc" >> "$LOG"   # bug fix 10:37: rc captured before $(date) resets $?
done < "$JOBS"
echo "[$(date +%T)] LANE DONE" >> "$LOG"
