#!/bin/bash
# post-hoc diagnostic: evaluate each (version, rep) once its 4 checkpoints exist; up to 4 concurrent
cd /home/minju/faicpaston/Imputing_Missing_Data
export PYTHONPATH=.pydeps:. CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=2
PY=/home/minju/.conda/envs/reef_gpu/bin/python; R=results/extra/phase6/diag_k0; LOG=$R/eval_loop.log; g=0
while true; do
  left=0
  for v in J S; do for r in $(seq 0 23); do
    out=$R/eval_$v/r$r.csv; [ -f $out ] && continue; left=$((left+1))
    [ -f $R/started_${v}_$r ] && continue
    n=$(ls $R/runs_$v/ckpt/k0.0_p0.7_r${r}_*_s$((6000+r)).pt 2>/dev/null | wc -l); [ "$n" -lt 4 ] && continue
    while [ $(pgrep -f k0_eval.py | wc -l) -ge 4 ]; do sleep 10; done
    touch $R/started_${v}_$r; echo "[$(date +%T)] start $v $r" >> $LOG
    (CUDA_VISIBLE_DEVICES=$g $PY experiments/phase6/diag/k0_eval.py $v $r >> $LOG 2>&1; echo "[$(date +%T)] end $v $r rc=$?" >> $LOG) &
    g=$(( (g+1) % 2 ))
  done; done
  [ $left -eq 0 ] && { echo "[$(date +%T)] EVAL LOOP DONE" >> $LOG; break; }
  sleep 30
done
