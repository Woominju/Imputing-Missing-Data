#!/bin/bash
# Phase 9 launcher. usage: experiments/phase9/launch_p9.sh STAGE(pilot|main) "YYYY-MM-DD HH:MM"(reservation END) [NPAR]  ; env P9_GPUS (default 0,1,2,3)
# Refuses unless: freeze file verifies; for main, the pilot gate file says PILOT_PASS true; labres-check shows every GPU in P9_GPUS reserved
# until END ("~HH:MM" or "~M/D HH:MM" when END is another day) and no [GPU 예약 경고]; no other Phase 9 run holds the lock.
# Deadlines: no new task after END - 35 min, running tasks terminated at END - 5 min (resumable). Detached (setsid); analyze9 runs at the end.
set -u
cd /home/minju/faicpaston/Imputing_Missing_Data
ROOT="${P9_ROOT:-results/extra/phase9}"; STAGE="${1:?stage}"; END="${2:?reservation end}"; NPAR="${3:-32}"; GP="${P9_GPUS:-0,1,2,3}"
PY=/home/minju/.conda/envs/reef_gpu/bin/python; LOCK="$ROOT/run.lock"; mkdir -p "$ROOT/logs" "$ROOT/analysis"
[ "$STAGE" = pilot ] || [ "$STAGE" = main ] || { echo "REFUSED: stage"; exit 2; }
if [ -e "$LOCK" ] && ! flock -n "$LOCK" true; then echo "REFUSED: another Phase 9 run holds $LOCK"; exit 3; fi
FZ="$ROOT/PROTOCOL_FREEZE_p9.txt"; [ -f "$FZ" ] || { echo "REFUSED: no freeze file"; exit 2; }
grep -E '^[0-9a-f]{64}  ' "$FZ" | sha256sum -c --quiet || { echo "REFUSED: freeze hashes do not verify"; exit 2; }
if [ "$STAGE" = main ]; then
  G=$(ls -t $ROOT/analysis/pilot_*/p9_pilot_gate.json 2>/dev/null | head -1)
  [ -n "$G" ] && grep -q '"PILOT_PASS": true' "$G" || { echo "REFUSED: no passing pilot gate"; exit 2; }; fi
LC=$(labres-check 2>&1); echo "$LC"; HM=$(date -d "$END" +%H:%M) || exit 2
[ "$(date -d "$END" +%F)" = "$(date +%F)" ] || HM="$(date -d "$END" +%-m/%-d) $HM"
for g in ${GP//,/ }; do echo "$LC" | grep -q "GPU $g · [0-9:]*~$HM" || { echo "REFUSED: GPU $g not reserved until $HM"; exit 2; }; done
echo "$LC" | grep -q "GPU 예약 경고" && { echo "REFUSED: [GPU 예약 경고]"; exit 2; }
E=$(date -d "$END" +%s); DL=$((E - 2100)); HD=$((E - 300)); [ "$(date +%s)" -lt "$DL" ] || { echo "REFUSED: too close to END"; exit 2; }
echo "launch $(date '+%F %T') stage=$STAGE END=$END NPAR=$NPAR GPUS=$GP" | tee -a "$ROOT/logs/launch.log"
setsid nohup flock -n "$LOCK" bash -c "
  export PYTHONPATH=.pydeps:. CUDA_DEVICE_ORDER=PCI_BUS_ID GPUS=$GP DEADLINE_EPOCH=$DL HARD_DEADLINE_EPOCH=$HD P9_ROOT=$ROOT
  $PY experiments/phase9/gen9.py 48 >> $ROOT/logs/gen9.log 2>&1 || { echo \"[\$(date '+%F %T')] gen9 FAILED\" >> $ROOT/logs/launch.log; exit 3; }
  $PY experiments/phase9/sched9.py $STAGE $NPAR >> $ROOT/logs/sched9.stdout 2>&1
  $PY experiments/phase9/analyze9.py $STAGE >> $ROOT/logs/analyze9_$STAGE.log 2>&1
  echo \"[\$(date '+%F %T')] PIPELINE END stage=$STAGE\" >> $ROOT/logs/launch.log
" > /dev/null 2>&1 < /dev/null &
sleep 2; echo "started detached"
