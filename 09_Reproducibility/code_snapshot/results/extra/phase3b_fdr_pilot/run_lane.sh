#!/bin/bash
cd /home/minju/faicpaston/Imputing_Missing_Data; G=$1; shift; O=results/extra/phase3b_fdr_pilot
for job in "$@"; do dec=${job%%:*}; Q=${job##*:}
  echo "[$(TZ=Asia/Seoul date '+%F %T')] start $dec Q=$Q GPU $G"; labres-check 2>&1 | head -4
  END=$(TZ=Asia/Seoul date -d '2026-10-27 18:15' +%s); [ $(date +%s) -ge $END ] && { echo "skip: near reservation end"; continue; }
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$G IMD_RESULTS_DIR=$PWD/$O PYTHONPATH=.pydeps OMP_NUM_THREADS=4 \
    timeout 2400 ~/.conda/envs/reef_gpu/bin/python -m results.extra.phase3b_fdr_pilot.pilot_train --decoder $dec --Q $Q --out $PWD/$O
  echo "[$(TZ=Asia/Seoul date '+%F %T')] end $dec Q=$Q rc=$?"
done
