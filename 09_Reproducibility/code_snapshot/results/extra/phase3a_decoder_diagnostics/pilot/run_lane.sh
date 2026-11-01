#!/bin/bash
# lane runner: $1 = GPU index, rest = "variant:latent" jobs, run sequentially
cd /home/minju/faicpaston/Imputing_Missing_Data
G=$1; shift; O=results/extra/phase3a_decoder_diagnostics/pilot
for job in "$@"; do v=${job%%:*}; k=${job##*:}
  echo "[$(TZ=Asia/Seoul date '+%F %T')] start $v$k on GPU $G"; labres-check 2>&1 | head -4
  END=$(TZ=Asia/Seoul date -d '2026-10-27 14:15' +%s); [ $(date +%s) -ge $END ] && { echo "skip: too close to reservation end"; continue; }
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$G IMD_RESULTS_DIR=$PWD/$O PYTHONPATH=.pydeps OMP_NUM_THREADS=4 \
    timeout 3600 ~/.conda/envs/reef_gpu/bin/python -m results.extra.phase3a_decoder_diagnostics.pilot_train --variant $v --latent $k --out $PWD/$O
  echo "[$(TZ=Asia/Seoul date '+%F %T')] end $v$k rc=$?"
done
