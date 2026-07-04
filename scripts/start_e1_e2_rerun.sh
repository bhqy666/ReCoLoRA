#!/usr/bin/env bash
# Detached launcher for run_e1_e2_baseline_seeds.sh (E1 + E2, ~38 runs, multi-day).
# Usage: bash scripts/start_e1_e2_rerun.sh
set -u
cd /home/bhqy/Documents/project/HiLoRA
mkdir -p logs/e1_e2_rerun
LOG_FILE=logs/e1_e2_rerun/master_$(date +%Y%m%d_%H%M%S).log
PID_FILE=logs/e1_e2_rerun/master.pid

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "Already running with PID $(cat "$PID_FILE"). Log: $(ls -t logs/e1_e2_rerun/master_*.log | head -1)"
  exit 1
fi

nohup bash scripts/run_e1_e2_baseline_seeds.sh >"$LOG_FILE" 2>&1 &
echo $! >"$PID_FILE"
echo "Started E1+E2 rerun. PID $(cat "$PID_FILE")"
echo "Follow with: tail -f $LOG_FILE"
