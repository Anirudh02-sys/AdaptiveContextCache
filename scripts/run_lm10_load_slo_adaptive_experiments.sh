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
  echo "  From the repo root, create a venv and install the package (see INSTALL.md sections 3-4), e.g.:" >&2
  echo "    python3 -m venv .venv && . .venv/bin/activate" >&2
  echo "    python -m pip install -r requirements.txt && python -m pip install -e ." >&2
  echo "  Or set PYTHON=/path/to/a/venv/bin/python that already has those installed." >&2
  exit 1
fi

# Single load-multiplier experiment at lm=10.
DRY_RUN="${DRY_RUN:-yes}"
LOAD_MULTIPLIER="${LOAD_MULTIPLIER:-10}"
CACHE_DIR="${CACHE_DIR:-/tmp/contextcache_data_lm10_load_slo_adaptive}"
EXAMPLE_CONFIG="${EXAMPLE_CONFIG:-config/request_gen.example.json}"
PREWARM_ENABLED="${PREWARM_ENABLED:-no}"
BASE_WINDOW="${BASE_WINDOW:-7}"
LOAD_ADAPTIVE_RATIO="${LOAD_ADAPTIVE_RATIO:-2.0}"
LOAD_ADAPTIVE_FORCE_SHRINK_RPS="${LOAD_ADAPTIVE_FORCE_SHRINK_RPS:-0.05}"

# Optional global config overrides. Leave unset to use EXAMPLE_CONFIG defaults.
EXPERIMENT_DURATION_S="${EXPERIMENT_DURATION_S:-}"
MAX_TURNS_PER_CONV="${MAX_TURNS_PER_CONV:-}"
MAX_CONVS_PER_APP="${MAX_CONVS_PER_APP:-}"

DATA_ROOT="data/test_apps/lm10_load_slo_adaptive_compare"
PLOTS_ROOT="${DATA_ROOT}/plots"

export VOCAREUM_API_KEY="${VOCAREUM_API_KEY:-}"
export VOCAREUM_BASE_URL="${VOCAREUM_BASE_URL:-https://genai.vocareum.com/v1}"

mkdir -p "${DATA_ROOT}"
mkdir -p "${PLOTS_ROOT}"

server_cmd_base=(
  env -u APPIMAGE PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" -m gptcache_server.server
  -s 127.0.0.1
  -p 8012
  -d "${CACHE_DIR}"
  -o True
  --context-cache-window-len "${BASE_WINDOW}"
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
  local mode="$1"
  local suffix="$2"
  shift 2
  local extra_server_args=("$@")

  echo "=== Running experiment: LM=${LOAD_MULTIPLIER}, mode=${mode}, suffix=${suffix}, DRY_RUN=${DRY_RUN} ==="
  if ((${#extra_server_args[@]} > 0)); then
    echo "    extra server args: ${extra_server_args[*]}"
  fi

  rm -rf "${CACHE_DIR}"
  mkdir -p "${CACHE_DIR}"

  local tmp_config
  tmp_config="$(mktemp "/tmp/request_gen_lm${LOAD_MULTIPLIER}.${suffix}.XXXX.json")"

  local metrics_path="${DATA_ROOT}/request_metrics_${suffix}.json"
  local log_path="${DATA_ROOT}/request_log_${suffix}.jsonl"

  local baseline_abs="${PWD}/${DATA_ROOT}/request_log_nocache.jsonl"
  local baseline_arg=""
  if [[ -f "${baseline_abs}" ]]; then
    baseline_arg="${baseline_abs}"
  fi

  local exp_dur_json="${EXPERIMENT_DURATION_S:-null}"
  local max_turns_json="${MAX_TURNS_PER_CONV:-null}"
  local max_convs_json="${MAX_CONVS_PER_APP:-null}"
  local warmup_enabled_json="false"
  if [[ "${PREWARM_ENABLED}" == "yes" ]]; then
    warmup_enabled_json="true"
  fi
  local jq_overrides='
    | .warmup_enabled = $warmup_enabled
    | (if $exp_dur != null then .experiment_duration_seconds = $exp_dur else . end)
    | (if $max_turns != null then .max_turns_per_conversation = $max_turns else . end)
    | (if $max_convs != null then .max_conversations_per_app = $max_convs else . end)'

  if [[ "${suffix}" == "nocache" ]]; then
    jq \
      --arg metrics_path "${metrics_path}" \
      --arg log_path "${log_path}" \
      --argjson load_multiplier "${LOAD_MULTIPLIER}" \
      --argjson warmup_enabled "${warmup_enabled_json}" \
      --argjson exp_dur "${exp_dur_json}" \
      --argjson max_turns "${max_turns_json}" \
      --argjson max_convs "${max_convs_json}" \
      '.output_metrics_path = $metrics_path
        | .output_request_log_path = $log_path
        | .load_multiplier = $load_multiplier
        | .accuracy_baseline_reference_run = true
        | .accuracy_baseline_log_path = ""'"${jq_overrides}" \
      "${EXAMPLE_CONFIG}" > "${tmp_config}"
  else
    jq \
      --arg metrics_path "${metrics_path}" \
      --arg log_path "${log_path}" \
      --arg baseline "${baseline_arg}" \
      --argjson load_multiplier "${LOAD_MULTIPLIER}" \
      --argjson warmup_enabled "${warmup_enabled_json}" \
      --argjson exp_dur "${exp_dur_json}" \
      --argjson max_turns "${max_turns_json}" \
      --argjson max_convs "${max_convs_json}" \
      '.output_metrics_path = $metrics_path
        | .output_request_log_path = $log_path
        | .load_multiplier = $load_multiplier
        | .accuracy_baseline_reference_run = false
        | .accuracy_baseline_log_path = $baseline'"${jq_overrides}" \
      "${EXAMPLE_CONFIG}" > "${tmp_config}"
  fi

  local server_cmd=("${server_cmd_base[@]}" "--server-mode" "${mode}" "${extra_server_args[@]}")
  local server_log="${DATA_ROOT}/server_${suffix}.log"
  "${server_cmd[@]}" >"${server_log}" 2>&1 &
  local server_pid=$!

  trap 'kill "${server_pid}" >/dev/null 2>&1 || true' EXIT

  wait_for_server

  "${PYTHON_BIN}" scripts/generate_requests.py --config "${tmp_config}"

  kill "${server_pid}" >/dev/null 2>&1 || true
  wait "${server_pid}" 2>/dev/null || true
  trap - EXIT

  rm -f "${tmp_config}"

  local plots_dir="${PLOTS_ROOT}/${suffix}"
  mkdir -p "${plots_dir}"
  local plot_prefix="lm${LOAD_MULTIPLIER}_${suffix}"

  "${PYTHON_BIN}" scripts/plot_request_metrics.py \
    --metrics "${metrics_path}" \
    --output-dir "${plots_dir}" \
    --prefix "${plot_prefix}"

  echo "Experiment LM=${LOAD_MULTIPLIER} ${mode} complete."
  echo "  Metrics: ${metrics_path}"
  echo "  Log:     ${log_path}"
  echo "  Plots:   ${plots_dir}"
  echo
}

run_experiment "no-cache" "nocache"
run_experiment "gptcache" "gptcache"
run_experiment "contextcache" "contextcache"
run_experiment "adaptivecontextcache" "adaptive_load_slo" \
  --load-adaptive \
  --load-adaptive-ratio "${LOAD_ADAPTIVE_RATIO}" \
  --load-adaptive-force-shrink-rps "${LOAD_ADAPTIVE_FORCE_SHRINK_RPS}" \
  --slo-adaptive

compare_output_dir="${PLOTS_ROOT}/compare"
mkdir -p "${compare_output_dir}"

"${PYTHON_BIN}" scripts/compare_experiments.py \
  --run "nocache:${DATA_ROOT}/request_metrics_nocache.json" \
  --run "gptcache:${DATA_ROOT}/request_metrics_gptcache.json" \
  --run "contextcache:${DATA_ROOT}/request_metrics_contextcache.json" \
  --run "adaptive_load_slo:${DATA_ROOT}/request_metrics_adaptive_load_slo.json" \
  --label "adaptive_load_slo:adaptive context (load+slo)" \
  --output-dir "${compare_output_dir}" \
  --prefix "compare_lm${LOAD_MULTIPLIER}_load_slo"

ablation_prefix="compare_lm${LOAD_MULTIPLIER}_load_slo"
"${PYTHON_BIN}" scripts/plot_window_factor_ablation.py \
  --metrics "${DATA_ROOT}/request_metrics_adaptive_load_slo.json" \
  --server-log "${DATA_ROOT}/server_adaptive_load_slo.log" \
  --output-dir "${compare_output_dir}" \
  --prefix "${ablation_prefix}"

echo "All LM=${LOAD_MULTIPLIER} load+slo adaptive comparison experiments complete."
echo "  Metrics/logs: ${DATA_ROOT}"
echo "  Compare plots: ${compare_output_dir}"
echo "  Window ablation JSON: ${compare_output_dir}/${ablation_prefix}_window_factor_ablation.json"
echo "  Window ablation plot: ${compare_output_dir}/${ablation_prefix}_window_factor_ablation.png"
