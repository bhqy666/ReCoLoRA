#!/usr/bin/env bash
set -u

cd /home/bhqy/Documents/project/HiLoRA

RESULT="outputs/continual_llm/recursive_hilora/qwen3_8b/seed_42/continual_results.json"
LOG="logs/recursive_hilora_suite/shutdown_after_qwen8.log"

mkdir -p "$(dirname "$LOG")"
echo "[$(date "+%F %T")] watcher started; waiting for $RESULT" >> "$LOG"

while [ ! -s "$RESULT" ]; do
  sleep 10
done

echo "[$(date "+%F %T")] Qwen8 result detected; stopping remaining recursive suite before shutdown" >> "$LOG"
sleep 5

pkill -TERM -f "train_continual_llm.py.*recursive_hilora" 2>/dev/null || true
pkill -TERM -f "run_recursive_hilora_llm_suite.sh" 2>/dev/null || true
sleep 3
pkill -KILL -f "train_continual_llm.py.*recursive_hilora" 2>/dev/null || true

sync
echo "[$(date "+%F %T")] invoking poweroff" >> "$LOG"
(systemctl poweroff -i || shutdown -h now || poweroff) >> "$LOG" 2>&1
