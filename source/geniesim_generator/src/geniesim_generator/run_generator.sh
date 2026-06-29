#!/usr/bin/env bash
#
# Run the scene generator (app.py) to compile the current LLM_RESULT.py into a
# scene.usda. Invoked by scene_viewer.py on every LLM_RESULT.py change, and
# usable standalone.
#
# Resolves paths relative to this script (lives next to app.py), so it works in
# any checkout without hardcoded paths. Overridable via env vars:
#   GENERATOR_PYTHON  interpreter to run app.py with (default: python3 on PATH)
#   GENERATOR_RUN_AS  if set (e.g. "1000" or "#1000"), re-run as that uid via sudo
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../src/geniesim_generator
SRC_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"                  # .../source/geniesim_generator/src
PY_PATH="${SCRIPT_DIR}/app.py"
PYTHON="${GENERATOR_PYTHON:-python3}"

# Fail early if app.py is missing
[[ -f "$PY_PATH" ]] || { echo "ERROR: $PY_PATH not found" >&2; exit 1; }

# app.py uses script-relative imports (from helper / from LLM_RESULT), so it must
# run with its own dir as cwd; PYTHONPATH=src makes `geniesim_generator.*` importable.
cd "$SCRIPT_DIR"

if [[ -n "${GENERATOR_RUN_AS:-}" ]]; then
    # Re-run as a specific user (e.g. when the caller is root inside a container)
    sudo -u "#${GENERATOR_RUN_AS#\#}" -g "#${GENERATOR_RUN_AS#\#}" \
        env PYTHONPATH="${PYTHONPATH:-}${PYTHONPATH:+:}${SRC_DIR}" "$PYTHON" "$PY_PATH" "$@"
else
    PYTHONPATH="${PYTHONPATH:-}${PYTHONPATH:+:}${SRC_DIR}" "$PYTHON" "$PY_PATH" "$@"
fi
