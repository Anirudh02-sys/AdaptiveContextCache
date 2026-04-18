#!/usr/bin/env bash
set -euo pipefail

# Reproduce a dry-run adaptive load experiment where context window transitions
# are visible after delayed load activation.
#
# Exact tuned parameters used:
# - load_multiplier: 10
# - load_multiplier_start_seconds: 70
# - experiment_duration_seconds: 190
# - load_adaptive_ratio: 1.2
# - load_adaptive_token_ratio: 1.2
# - load_adaptive_shrink_min_rps: 0.0
# - load_adaptive_grow_max_rps: 9999
# - GPTCACHE_DRY_RUN_SLEEP_S: 0.01
#
# This script writes:
# - run artifacts under data/test_apps/load_mult_delayed_compare/<RUN_TAG>/
# - plots under data/test_apps/load_mult_delayed_compare/plots/compare_<RUN_TAG>/
# - a summary with window transition timestamps relative to first request.

if ! command -v jq >/dev/null 2>&1; then
  echo "error: jq is required" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
else
  echo "error: python3 not found" >&2
  exit 1
fi

RUN_TAG="${RUN_TAG:-lm10_delayed70_drycheck_v3_repro}"
export RUN_TAG
PORT="${PORT:-8012}"
CACHE_DIR="${CACHE_DIR:-/tmp/contextcache_data_load_mult_delayed}"
DATA_ROOT="data/test_apps/load_mult_delayed_compare"
RUN_DIR="${DATA_ROOT}/${RUN_TAG}"
PLOT_DIR="${DATA_ROOT}/plots/compare_${RUN_TAG}"
TMP_CONFIG="/tmp/request_gen_${RUN_TAG}.json"

LOAD_MULT="${LOAD_MULT:-10}"
LOAD_MULT_START_S="${LOAD_MULT_START_S:-70}"
EXPERIMENT_DURATION_S="${EXPERIMENT_DURATION_S:-190}"
LOAD_ADAPTIVE_RATIO="${LOAD_ADAPTIVE_RATIO:-1.2}"
LOAD_ADAPTIVE_TOKEN_RATIO="${LOAD_ADAPTIVE_TOKEN_RATIO:-1.2}"
LOAD_ADAPTIVE_SHRINK_MIN_RPS="${LOAD_ADAPTIVE_SHRINK_MIN_RPS:-0.0}"
LOAD_ADAPTIVE_GROW_MAX_RPS="${LOAD_ADAPTIVE_GROW_MAX_RPS:-9999}"
DRY_SLEEP_S="${DRY_SLEEP_S:-0.01}"

mkdir -p "${RUN_DIR}" "${PLOT_DIR}"

jq \
  --arg metrics "${RUN_DIR}/request_metrics_adaptive_load.json" \
  --arg log "${RUN_DIR}/request_log_adaptive_load.jsonl" \
  --argjson lm "${LOAD_MULT}" \
  --argjson lm_start "${LOAD_MULT_START_S}" \
  --argjson dur "${EXPERIMENT_DURATION_S}" \
  '.experiment_duration_seconds = $dur
    | .warmup_enabled = false
    | .output_metrics_path = $metrics
    | .output_request_log_path = $log
    | .load_multiplier = $lm
    | .load_multiplier_start_seconds = $lm_start
    | .accuracy_baseline_reference_run = false
    | .accuracy_baseline_log_path = ""' \
  "config/request_gen.example.json" > "${TMP_CONFIG}"

SERVER_LOG="${RUN_DIR}/server_adaptive_load.log"
pkill -f "gptcache_server.server.*:${PORT}" 2>/dev/null || true
pkill -f "gptcache_server.server.*-p ${PORT}" 2>/dev/null || true

env -u APPIMAGE \
  PYTHONUNBUFFERED=1 \
  GPTCACHE_DRY_RUN_SLEEP_S="${DRY_SLEEP_S}" \
  "${PYTHON_BIN}" -m gptcache_server.server \
    -s 127.0.0.1 \
    -p "${PORT}" \
    -d "${CACHE_DIR}" \
    -o True \
    --server-mode adaptivecontextcache \
    --load-adaptive \
    --load-adaptive-ratio "${LOAD_ADAPTIVE_RATIO}" \
    --load-adaptive-token-ratio "${LOAD_ADAPTIVE_TOKEN_RATIO}" \
    --load-adaptive-shrink-min-rps "${LOAD_ADAPTIVE_SHRINK_MIN_RPS}" \
    --load-adaptive-grow-max-rps "${LOAD_ADAPTIVE_GROW_MAX_RPS}" \
    --dry-run yes > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

cleanup() {
  kill "${SERVER_PID}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

ready="no"
for _i in $(seq 1 120); do
  if curl -sSf "http://127.0.0.1:${PORT}/" >/dev/null 2>&1; then
    ready="yes"
    break
  fi
  sleep 1
done
if [[ "${ready}" != "yes" ]]; then
  echo "error: server did not become ready on 127.0.0.1:${PORT}" >&2
  exit 1
fi

"${PYTHON_BIN}" scripts/generate_requests.py --config "${TMP_CONFIG}"

kill "${SERVER_PID}" >/dev/null 2>&1 || true
wait "${SERVER_PID}" 2>/dev/null || true
trap - EXIT

"${PYTHON_BIN}" scripts/plot_delayed_experiment_timeseries.py \
  --root-dir "${DATA_ROOT}" \
  --lm-tag "${RUN_TAG}" \
  --suffix adaptive_load \
  --output-dir "${PLOT_DIR}" \
  --prefix delayed_timeseries

echo
echo "Window transition summary (relative to first successful request):"
"${PYTHON_BIN}" - <<'PY'
import json
import os
import re
from datetime import datetime, timedelta

run_tag = os.environ["RUN_TAG"]
base = os.path.join("data/test_apps/load_mult_delayed_compare", run_tag)
server_log = os.path.join(base, "server_adaptive_load.log")
req_log = os.path.join(base, "request_log_adaptive_load.jsonl")
metrics_json = os.path.join(base, "request_metrics_adaptive_load.json")

pat = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d+).*effective window (\d+) → (\d+)"
)
events = []
with open(server_log, "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        m = pat.match(line.strip())
        if not m:
            continue
        ts = (
            datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
            + timedelta(milliseconds=int((m.group(2) + "000")[:3]))
        ).timestamp()
        events.append((ts, int(m.group(3)), int(m.group(4))))

t0 = None
with open(req_log, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        if obj.get("ok") is True and obj.get("timestamp") is not None:
            t0 = float(obj["timestamp"])
            break

with open(metrics_json, "r", encoding="utf-8") as f:
    metrics = json.load(f)

print(f"  load_multiplier_start_seconds={metrics.get('load_multiplier_start_seconds')}")
print(f"  experiment_duration_seconds={metrics.get('experiment_duration_seconds')}")
if t0 is None:
    print("  no successful request timestamps found")
elif not events:
    print("  no 'effective window X -> Y' transitions found in server log")
else:
    for ts, w0, w1 in events:
        print(f"  t={ts - t0:.3f}s : {w0} -> {w1}")
PY

echo
echo "Done."
echo "  metrics: ${RUN_DIR}/request_metrics_adaptive_load.json"
echo "  server log: ${RUN_DIR}/server_adaptive_load.log"
echo "  window plot: ${PLOT_DIR}/delayed_timeseries_${RUN_TAG}_window_timeseries.png"
