#!/usr/bin/env bash

set -euo pipefail

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    echo "[!] OPENROUTER_API_KEY is not set"
    exit 1
fi

while IFS= read -r model; do
    [[ -z "$model" ]] && continue

    echo "[*] Running model: ${model}"
    env \
        SEGGSBENCH_MODEL="$model" \
        SEGGSBENCH_ENFORCE_REASONING=true \
        OPENROUTER_API_KEY="${OPENROUTER_API_KEY}" \
        python model_run.py
done < models_list.txt