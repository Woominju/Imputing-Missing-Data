#!/bin/bash
# usage: run_lane.sh GPU MODULE OUTDIR job1 job2 ...   (job = "arg string" passed to the module)
cd /home/minju/faicpaston/Imputing_Missing_Data; G=$1; MOD=$2; O=$3; shift 3
END=$(TZ=Asia/Seoul date -d '2026-10-27 18:30' +%s)
for job in "$@"; do
  LEFT=$(( END - $(date +%s) )); [ $LEFT -lt 240 ] && { echo "[$(TZ=Asia/Seoul date '+%T')] skip $job: near reservation end"; continue; }
  echo "[$(TZ=Asia/Seoul date '+%T')] start $job GPU $G"; labres-check 2>&1 | grep -E "경고|WARN" | head -2
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$G IMD_RESULTS_DIR=$PWD/$O PYTHONPATH=.pydeps OMP_NUM_THREADS=3 \
    timeout $LEFT ~/.conda/envs/reef_gpu/bin/python -m $MOD $job --out $PWD/$O
  echo "[$(TZ=Asia/Seoul date '+%T')] end $job rc=$?"
done
