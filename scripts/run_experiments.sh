#!/usr/bin/env bash
set -euo pipefail

if ! command -v jq >/dev/null 2>&1; then
  echo "error: jq is required (this script patches JSON configs). Install it, e.g.:" >&2
  echo "  Debian/Ubuntu: sudo apt-get install -y jq" >&2
  echo "  See INSTALL.md (system prerequisites)." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Prefer repo venv so numpy / gptcache match INSTALL.md (avoid bare /usr/bin/python3).
if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="${PYTHON}"
elif [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
else
  echo "error: python3 not found. Install Python or set PYTHON to your venv interpreter." >&2
  exit 1
fi

if ! env -u APPIMAGE "${PYTHON_BIN}" -c "import numpy, gptcache" >/dev/null 2>&1; then
  echo "error: ${PYTHON_BIN} cannot import numpy and gptcache." >&2
  echo "  From the repo root, create a venv and install the package (see INSTALL.md sections 3–4), e.g.:" >&2
  echo "    python3 -m venv .venv && . .venv/bin/activate" >&2
  echo "    python -m pip install -r requirements.txt && python -m pip install -e ." >&2
  echo "  Or set PYTHON=/path/to/a/venv/bin/python that already has those installed." >&2
  exit 1
fi

# Orchestrate three experiments against gptcache_server:
# 1) no-cache
# 2) gptcache
# 3) contextcache
# 4) adaptivecontextcache with --slo-adaptive
#
# Usage:
#   DRY_RUN=yes ./scripts/run_experiments.sh
#   DRY_RUN=no  ./scripts/run_experiments.sh
#
# Uses .venv/bin/python when present, else python3 on PATH. Override with PYTHON=...
#
# DRY_RUN controls the server's upstream dry-run flag (--dry-run yes).
# Optional: EXPERIMENT_DURATION_SECONDS=120 overrides request_gen experiment_duration_seconds
# for quicker iteration (default: use value from config/request_gen.example.json).
# Optional: SERVER_READY_TIMEOUT_S=90 — seconds to wait for the server HTTP probe (default: 90).

DRY_RUN="${DRY_RUN:-no}"          # yes|no -> passed to server
CACHE_DIR="${CACHE_DIR:-/tmp/contextcache_data}"
EXAMPLE_CONFIG="config/request_gen.example.json"
METRICS_BASE_DIR="data/test_apps/example"
PLOTS_BASE_DIR="${METRICS_BASE_DIR}/plots"
# Latency SLO anchors (served p99 ms, prior contextcache run): app0=1932 app1=1824 app2=1978 app3=2009 app4=1900.
# application_slo_expectations in config/request_gen.example.json uses strict targets for apps 2-3 and loose for 0,1,4.

# API keys must come from the environment — never commit real credentials.
# Example before running:
#   export OPENAI_API_KEY=...
#   export OPENAI_API_BASE=https://api.openai.com/v1
#   export VOCAREUM_API_KEY=...
#   export VOCAREUM_BASE_URL=https://genai.vocareum.com/v1

OPENAI_API_KEY_VALUE="${OPENAI_API_KEY:-}"
OPENAI_API_BASE_VALUE="${OPENAI_API_BASE:-https://api.openai.com/v1}"
VOCAREUM_API_KEY_VALUE="${VOCAREUM_API_KEY:-}"
VOCAREUM_API_BASE_VALUE="${VOCAREUM_BASE_URL:-https://genai.vocareum.com/v1}"

# export OPENAI_API_KEY="${OPENAI_API_KEY_VALUE}"
# export OPENAI_API_BASE="${OPENAI_API_BASE_VALUE}"
export VOCAREUM_API_KEY=
export VOCAREUM_BASE_URL="${VOCAREUM_API_BASE_VALUE}"

mkdir -p "${METRICS_BASE_DIR}"
mkdir -p "${PLOTS_BASE_DIR}"

server_cmd_base=(
  env -u APPIMAGE PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -m gptcache_server.server
  -s 127.0.0.1
  -p 8012
  -d "${CACHE_DIR}"
  -o True
)

if [[ "${DRY_RUN}" == "yes" ]]; then
  server_cmd_base+=("--dry-run" "yes")
fi

wait_for_server() {
  local timeout_s="${SERVER_READY_TIMEOUT_S:-90}"
  local start
  start=$(date +%s)
  while true; do
    if curl -sSf "http://127.0.0.1:8012/" >/dev/null 2>&1; then
      break
    fi
    local now
    now=$(date +%s)
    if (( now - start > timeout_s )); then
      echo "Server did not become ready within ${timeout_s}s" >&2
      return 1
    fi
    sleep 1
  done
}

run_experiment() {
  local mode="$1"         # no-cache | gptcache | contextcache | adaptivecontextcache
  local suffix="$2"       # nocache | gptcache | contextcache | adaptive_slo | ...
  shift 2
  local extra_server_args=("$@")   # e.g. --slo-adaptive

  echo "=== Running experiment: mode=${mode}, suffix=${suffix}, DRY_RUN=${DRY_RUN} ==="
  if ((${#extra_server_args[@]} > 0)); then
    echo "    extra server args: ${extra_server_args[*]}"
  fi

  rm -rf "${CACHE_DIR}"
  mkdir -p "${CACHE_DIR}"

  local tmp_config
  # Use a temp config in /tmp so we don't pollute the repo with many configs.
  tmp_config="$(mktemp "/tmp/request_gen_example.${suffix}.XXXX.json")"

  local metrics_path="${METRICS_BASE_DIR}/request_metrics_${suffix}.json"
  local log_path="${METRICS_BASE_DIR}/request_log_${suffix}.jsonl"
  local nocache_baseline_log="${METRICS_BASE_DIR}/request_log_nocache.jsonl"
  local model="gpt-3.5-turbo" # or "@azure-1/gpt-4o"

  # Nocache run records the reference log; other modes compare accuracy to that log
  # (same thread layout as this script’s nocache run).
  if [[ "${suffix}" == "nocache" ]]; then
    jq \
      --arg metrics_path "${metrics_path}" \
      --arg log_path "${log_path}" \
      '.output_metrics_path = $metrics_path
        | .output_request_log_path = $log_path
        | .accuracy_baseline_reference_run = true
        | .accuracy_baseline_log_path = ""' \
      "${EXAMPLE_CONFIG}" > "${tmp_config}"
  else
    jq \
      --arg metrics_path "${metrics_path}" \
      --arg log_path "${log_path}" \
      --arg baseline_log "${nocache_baseline_log}" \
      '.output_metrics_path = $metrics_path
        | .output_request_log_path = $log_path
        | .accuracy_baseline_reference_run = false
        | .accuracy_baseline_log_path = $baseline_log' \
      "${EXAMPLE_CONFIG}" > "${tmp_config}"
  fi

  if [[ -n "${EXPERIMENT_DURATION_SECONDS:-}" ]]; then
    jq --argjson dur "${EXPERIMENT_DURATION_SECONDS}" '.experiment_duration_seconds = $dur' "${tmp_config}" > "${tmp_config}.dur"
    mv "${tmp_config}.dur" "${tmp_config}"
  fi

  local server_cmd=("${server_cmd_base[@]}" "--server-mode" "${mode}" "${extra_server_args[@]}")
  "${server_cmd[@]}" &
  local server_pid=$!

  trap 'kill "${server_pid}" >/dev/null 2>&1 || true' EXIT

  wait_for_server

  "${PYTHON_BIN}" scripts/generate_requests.py --config "${tmp_config}"

  kill "${server_pid}" >/dev/null 2>&1 || true
  wait "${server_pid}" 2>/dev/null || true
  trap - EXIT

  rm -f "${tmp_config}"

  local plots_dir="${PLOTS_BASE_DIR}/${suffix}"
  mkdir -p "${plots_dir}"

  "${PYTHON_BIN}" scripts/plot_request_metrics.py \
    --metrics "${metrics_path}" \
    --output-dir "${plots_dir}" \
    --prefix "${suffix}"

  echo "Experiment ${mode} complete."
  echo "  Metrics: ${metrics_path}"
  echo "  Log:     ${log_path}"
  echo "  Plots:   ${plots_dir}"
  echo
}

# Baseline experiments (uncomment when you want to regenerate nocache / gptcache / contextcache metrics):
# Temporarily skipped to save time; uncomment to regenerate nocache / gptcache metrics.
run_experiment "no-cache" "nocache"
run_experiment "gptcache" "gptcache"
run_experiment "contextcache" "contextcache"

# SLO-adaptive: --slo-adaptive-* flags tune window deltas (see gptcache_server --help).
run_experiment "adaptivecontextcache" "adaptive_slo" \
  --slo-adaptive \
  --slo-adaptive-alpha "${SLO_ADAPTIVE_ALPHA:-0.8}" \
  --slo-adaptive-beta "${SLO_ADAPTIVE_BETA:-0.2}" \
  --slo-adaptive-baseline-center "${SLO_ADAPTIVE_BASELINE_CENTER:--0.25}"

compare_output_dir="${PLOTS_BASE_DIR}/compare"
mkdir -p "${compare_output_dir}"

# Optional nocache/gptcache: add more --run lines here (do not put # inside a \-continued command).
"${PYTHON_BIN}" scripts/compare_experiments.py \
  --run "nocache:${METRICS_BASE_DIR}/request_metrics_nocache.json" \
  --run "gptcache:${METRICS_BASE_DIR}/request_metrics_gptcache.json" \
  --run "contextcache:${METRICS_BASE_DIR}/request_metrics_contextcache.json" \
  --run "adaptive_slo:${METRICS_BASE_DIR}/request_metrics_adaptive_slo.json" \
  --output-dir "${compare_output_dir}" \
  --prefix "compare"
