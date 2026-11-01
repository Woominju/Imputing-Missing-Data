#!/bin/bash
# detached watcher: when the P7-B evaluation scheduler exits, run the frozen pre-registered analysis once
cd /home/minju/faicpaston/Imputing_Missing_Data
until grep -q "SCHEDULER EXIT" results/extra/phase7/eval/scheduler.log 2>/dev/null; do sleep 60; done
PYTHONPATH=.pydeps:. /home/minju/.conda/envs/reef_gpu/bin/python experiments/phase7/analyze7.py > results/extra/phase7/auto_analyze.log 2>&1
echo "[$(date '+%F %T')] analyze7 rc=$?" >> results/extra/phase7/auto_analyze.log
