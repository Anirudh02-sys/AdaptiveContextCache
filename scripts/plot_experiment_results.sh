#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="${PYTHON}"
elif [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
else
  echo "error: python3 not found. Set PYTHON to your venv interpreter." >&2
  exit 1
fi

if ! env -u APPIMAGE "${PYTHON_BIN}" -c "import matplotlib, numpy" >/dev/null 2>&1; then
  echo "error: ${PYTHON_BIN} cannot import matplotlib and numpy (needed for plotting)." >&2
  echo "  Install dependencies from the repo root (see INSTALL.md), e.g.:" >&2
  echo "    python3 -m venv .venv && . .venv/bin/activate" >&2
  echo "    python -m pip install -r requirements.txt && python -m pip install -e ." >&2
  exit 1
fi

exec env -u APPIMAGE "${PYTHON_BIN}" "${SCRIPT_DIR}/plot_experiment_results.py" "$@"
