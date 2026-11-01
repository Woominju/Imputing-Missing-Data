#!/bin/bash
GPU=$1; JOBS=$2; LOG=$3; DEADLINE=${DEADLINE:-23:50}
cd /home/minju/faicpaston/Imputing_Missing_Data
export PYTHONPATH=.pydeps:. CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=2
PY=/home/minju/.conda/envs/reef_gpu/bin/python
while IFS= read -r job; do
  [ -z "$job" ] && continue
  if [[ "$(date +%H:%M)" > "$DEADLINE" ]]; then echo "[$(date +%T)] skip $job" >> "$LOG"; continue; fi
  echo "[$(date +%T)] start $job" >> "$LOG"
  $PY experiments/phase6/s1r/train.py $job >> "$LOG" 2>&1; rc=$?
  echo "[$(date +%T)] end $job rc=$rc" >> "$LOG"
done < "$JOBS"
echo "[$(date +%T)] LANE DONE" >> "$LOG"
