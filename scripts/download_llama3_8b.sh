#!/usr/bin/env bash
set -euo pipefail
CONDA_ENV=${CONDA_ENV:-open_manus}
MODEL_ID=${MODEL_ID:-NousResearch/Meta-Llama-3-8B}
LOG_DIR=${LOG_DIR:-logs}
mkdir -p "$LOG_DIR"
PYTHONUNBUFFERED=1 MODEL_ID="$MODEL_ID" conda run --no-capture-output -n "$CONDA_ENV" python - <<'PY'
import os
from huggingface_hub import snapshot_download
model_id = os.environ.get("MODEL_ID", "NousResearch/Meta-Llama-3-8B")
path = snapshot_download(repo_id=model_id, resume_download=True, local_files_only=False)
print(path)
PY
