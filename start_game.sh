#!/usr/bin/env bash
# start_game.sh — start the Ultimate Texas Hold'em game interface
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── 1. pick the right Python interpreter ────────────────────────────────────
if [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
    PIP=".venv/bin/pip"
else
    PYTHON="$(command -v python3 2>/dev/null || command -v python)"
    PIP="$(command -v pip3 2>/dev/null || command -v pip)"
fi

# ── 2. make sure FastAPI / uvicorn are installed ─────────────────────────────
if ! "$PYTHON" -c "import fastapi, uvicorn" 2>/dev/null; then
    echo "[start_game] Installing web dependencies..."
    "$PIP" install -q -r requirements-web.txt
fi

# ── 3. make sure the game engine can be imported ─────────────────────────────
if ! "$PYTHON" -c "from src.envs.uth_env import UTHState" 2>/dev/null; then
    echo "[start_game] Installing project (editable mode)..."
    "$PIP" install -q -e .
fi

# ── 4. launch ────────────────────────────────────────────────────────────────
HOST="${UTH_HOST:-0.0.0.0}"
PORT="${UTH_PORT:-8000}"

echo ""
echo "  Ultimate Texas Hold'em"
echo "  ─────────────────────────────────────"
echo "  URL:  http://localhost:${PORT}"
echo "  Stop: Ctrl+C"
echo ""

"$PYTHON" -m uvicorn web.uth_server:app \
    --host "$HOST" \
    --port "$PORT" \
    --reload
