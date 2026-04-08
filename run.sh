#!/usr/bin/env bash
# ---------------------------------------------------------------
# Prompt Injection Tester — quick-start helper
# Supports both Docker and bare-metal execution.
# ---------------------------------------------------------------
set -euo pipefail

# --- Configuration (edit these or pass as env vars) -------------

# Uncomment ONE target block, or export URL/PRESET before running.

# Ollama (local — if running on host)
# export URL="http://localhost:11434/api/generate"
# export PRESET="ollama"

# Ollama (from inside Docker — uses host-gateway)
# export URL="http://host.docker.internal:11434/api/generate"
# export PRESET="ollama"

# Claude
# export URL="https://api.anthropic.com/v1/messages"
# export PRESET="claude"
# export API_KEY="sk-ant-..."

# Grok (xAI)
# export URL="https://api.x.ai/v1/chat/completions"
# export PRESET="grok"
# export API_KEY="xai-..."

# Gemini
# export URL="https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
# export PRESET="gemini"
# export API_KEY="AIza..."

# Defaults
export TARGET_URL="${URL:-http://localhost:8080/v1/chat/completions}"
export PRESET="${PRESET:-default}"
export API_KEY="${API_KEY:-}"

# --- Detect mode ------------------------------------------------

MODE="${1:-}"

if [ "$MODE" = "docker" ]; then
    shift  # consume "docker" arg, pass the rest through
    echo "==> [Docker] Target: ${TARGET_URL}  (preset: ${PRESET})"
    mkdir -p results

    docker compose run --rm \
        -e API_KEY \
        -e TARGET_URL \
        -e PRESET \
        tester "$@"

    echo "==> Results saved to ./results/"

elif [ "$MODE" = "docker-with-ollama" ]; then
    shift
    echo "==> [Docker + bundled Ollama] Starting..."
    mkdir -p results

    # Start Ollama in the background, wait for it
    docker compose --profile with-ollama up -d ollama
    echo "    Waiting for Ollama to be ready..."
    for i in $(seq 1 30); do
        if docker compose exec ollama ollama list &>/dev/null; then
            break
        fi
        sleep 1
    done

    # Pull a model if none exist
    MODEL_COUNT=$(docker compose exec ollama ollama list 2>/dev/null | tail -n +2 | wc -l)
    if [ "$MODEL_COUNT" -eq 0 ]; then
        echo "    Pulling llama3 (first run only)..."
        docker compose exec ollama ollama pull llama3
    fi

    # Run tests targeting the Ollama container
    export TARGET_URL="http://ollama:11434/api/generate"
    export PRESET="ollama"

    docker compose --profile with-ollama run --rm \
        -e API_KEY \
        -e TARGET_URL \
        -e PRESET \
        tester "$@"

    echo "==> Results saved to ./results/"
    echo "    Ollama is still running. Stop with: docker compose --profile with-ollama down"

else
    # Bare-metal (no Docker)
    echo "==> [Local] Target: ${TARGET_URL}  (preset: ${PRESET})"
    echo ""

    python3 "$(dirname "$0")/tester.py" \
        --url "${TARGET_URL}" \
        --api-key "${API_KEY}" \
        --preset "${PRESET}" \
        --output results.json \
        --delay 1.5 \
        --verbose \
        "$@"

    echo ""
    echo "==> Results saved to results.json"
fi
