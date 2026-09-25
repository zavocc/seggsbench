#!/usr/bin/env bash

set -euo pipefail

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    echo "[!] OPENROUTER_API_KEY is not set"
    exit 1
fi

while IFS= read -r model; do
    [[ -z "$model" ]] && continue

    echo "[*] Scoring model: ${model}"
    env \
        SEGGSBENCH_MODEL="$model" \
        OPENROUTER_API_KEY="${OPENROUTER_API_KEY}" \
        python jev_score.py
done < models_list.txt