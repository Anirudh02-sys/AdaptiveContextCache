#!/usr/bin/env bash
set -euo pipefail

# Orchestrate three experiments against gptcache_server:
# 1) no-cache
# 2) gptcache
# 3) contextcache
#
# Usage:
#   DRY_RUN=yes ./scripts/run_experiments.sh
#   DRY_RUN=no  ./scripts/run_experiments.sh
#
# DRY_RUN controls the server's upstream dry-run flag (--dry-run yes).

DRY_RUN="${DRY_RUN:-no}"          # yes|no -> passed to server
CACHE_DIR="${CACHE_DIR:-/tmp/contextcache_data}"
EXAMPLE_CONFIG="config/request_gen.example.json"
METRICS_BASE_DIR="data/test_apps/example"
PLOTS_BASE_DIR="${METRICS_BASE_DIR}/plots"

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
export VOCAREUM_API_KEY="${VOCAREUM_API_KEY_VALUE}"
export VOCAREUM_BASE_URL="${VOCAREUM_API_BASE_VALUE}"

mkdir -p "${METRICS_BASE_DIR}"
mkdir -p "${PLOTS_BASE_DIR}"

server_cmd_base=(
  env -u APPIMAGE /usr/bin/python3 -m gptcache_server.server
  -s 127.0.0.1
  -p 8012
  -d "${CACHE_DIR}"
  -o True
)

if [[ "${DRY_RUN}" == "yes" ]]; then
  server_cmd_base+=("--dry-run" "yes")
fi

wait_for_server() {
  local timeout_s=30
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
  local mode="$1"         # no-cache | gptcache | contextcache
  local suffix="$2"       # nocache | gptcache | contextcache

  echo "=== Running experiment: mode=${mode}, suffix=${suffix}, DRY_RUN=${DRY_RUN} ==="

  rm -rf "${CACHE_DIR}"
  mkdir -p "${CACHE_DIR}"

  local tmp_config
  # Use a temp config in /tmp so we don't pollute the repo with many configs.
  tmp_config="$(mktemp "/tmp/request_gen_example.${suffix}.XXXX.json")"

  local metrics_path="${METRICS_BASE_DIR}/request_metrics_${suffix}.json"
  local log_path="${METRICS_BASE_DIR}/request_log_${suffix}.jsonl"
  local model="gpt-3.5-turbo" # or "@azure-1/gpt-4o"

  jq \
    --arg metrics_path "${metrics_path}" \
    --arg log_path "${log_path}" \
    '.output_metrics_path = $metrics_path | .output_request_log_path = $log_path' \
    "${EXAMPLE_CONFIG}" > "${tmp_config}"

  local server_cmd=("${server_cmd_base[@]}" "--server-mode" "${mode}")
  "${server_cmd[@]}" &
  local server_pid=$!

  trap 'kill "${server_pid}" >/dev/null 2>&1 || true' EXIT

  wait_for_server

  python3 scripts/generate_requests.py --config "${tmp_config}"

  kill "${server_pid}" >/dev/null 2>&1 || true
  wait "${server_pid}" 2>/dev/null || true
  trap - EXIT

  rm -f "${tmp_config}"

  local plots_dir="${PLOTS_BASE_DIR}/${suffix}"
  mkdir -p "${plots_dir}"

  python3 scripts/plot_request_metrics.py \
    --metrics "${metrics_path}" \
    --output-dir "${plots_dir}" \
    --prefix "${suffix}"

  echo "Experiment ${mode} complete."
  echo "  Metrics: ${metrics_path}"
  echo "  Log:     ${log_path}"
  echo "  Plots:   ${plots_dir}"
  echo
}

run_experiment "no-cache" "nocache"
run_experiment "gptcache" "gptcache"
run_experiment "contextcache" "contextcache"

compare_output_dir="${PLOTS_BASE_DIR}/compare"
mkdir -p "${compare_output_dir}"

python3 scripts/compare_experiments.py \
  --metrics-nocache "${METRICS_BASE_DIR}/request_metrics_nocache.json" \
  --metrics-gptcache "${METRICS_BASE_DIR}/request_metrics_gptcache.json" \
  --metrics-contextcache "${METRICS_BASE_DIR}/request_metrics_contextcache.json" \
  --output-dir "${compare_output_dir}" \
  --prefix "compare"

# Presentation figures: per-application SLOs + request generation stats.
SLO_WORKLOAD_DIR="${PLOTS_BASE_DIR}/slo_workload"
mkdir -p "${SLO_WORKLOAD_DIR}"
python3 scripts/visualize_app_slos_and_workload.py \
  --config "${EXAMPLE_CONFIG}" \
  --output-dir "${SLO_WORKLOAD_DIR}" \
  --prefix "request_gen_example"
