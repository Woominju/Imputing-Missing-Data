#!/bin/bash
cd /home/minju/faicpaston/Imputing_Missing_Data; G=$1; shift; O=results/extra/phase3c_hidden_state
for job in "$@"; do IFS=: read dec vars seed <<< "$job"
  echo "[$(TZ=Asia/Seoul date '+%F %T')] start $dec $vars s$seed GPU $G"; labres-check 2>&1 | head -4
  END=$(TZ=Asia/Seoul date -d '2026-10-27 18:00' +%s); [ $(date +%s) -ge $END ] && { echo "skip: near reservation end"; continue; }
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$G IMD_RESULTS_DIR=$PWD/$O PYTHONPATH=.pydeps OMP_NUM_THREADS=4 \
    timeout 3000 ~/.conda/envs/reef_gpu/bin/python -m results.extra.phase3c_hidden_state.train --decoder $dec --vars $vars --seed $seed --out $PWD/$O
  echo "[$(TZ=Asia/Seoul date '+%F %T')] end $dec $vars s$seed rc=$?"
done
