#!/usr/bin/env bash
# Delayed load-multiplier sweep: same cache modes as run_load_multiplier_experiments.sh,
# but the configured load_multiplier only takes effect after LOAD_MULT_START_S seconds
# have elapsed. Before that, every thread paces as if load_multiplier=1.0. This lets the
# controllers (including --load-adaptive) observe a stable warm phase and then react to
# a mid-run load surge.
#
# For the single load_multiplier=10 sweep:
#   1) no-cache          (TODO-gated, commented out like the existing sweep)
#   2) gptcache          (TODO-gated, commented out like the existing sweep)
#   3) contextcache
#   4) adaptivecontextcache + --load-adaptive
#
# Artifacts are written ONLY under data/test_apps/load_mult_delayed_compare/ so they do
# NOT overwrite runs from scripts/run_load_multiplier_experiments.sh or
# scripts/run_experiments.sh.
#
# Layout:
#   data/test_apps/load_mult_delayed_compare/lm10_delayed240/request_metrics_<suffix>.json
#   data/test_apps/load_mult_delayed_compare/plots/lm10_delayed240/<suffix>/   (per-run plots)
#   data/test_apps/load_mult_delayed_compare/plots/compare_lm10_delayed240/    (compare plot + timeseries)
#
# Usage:
#   ./scripts/run_delayed_load_multiplier_experiments.sh
#   DRY_RUN=no ./scripts/run_delayed_load_multiplier_experiments.sh   # real upstream LLM on cache miss
#
# Environment:
#   PYTHON              Python interpreter (default: .venv/bin/python or python3)
#   DRY_RUN             yes|no — server --dry-run (stub LLM on miss); default: yes
#   CACHE_DIR           Server cache directory cleared each run (default: /tmp/contextcache_data_load_mult_delayed)
#   LOAD_ADAPTIVE_RATIO Passed to --load-adaptive-ratio (default: 2.0, must be > 1)
#   EXAMPLE_CONFIG      Request-gen template (default: config/request_gen.example.json)
#   LOAD_MULT           Target load multiplier after the delay (default: 10)
#   LOAD_MULT_START_S   Seconds after experiment start when LOAD_MULT engages (default: 240 = 4 min)
#
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
  echo "  From the repo root, create a venv and install the package (see INSTALL.md sections 3–4), e.g.:" >&2
  echo "    python3 -m venv .venv && . .venv/bin/activate" >&2
  echo "    python -m pip install -r requirements.txt && python -m pip install -e ." >&2
  echo "  Or set PYTHON=/path/to/a/venv/bin/python that already has those installed." >&2
  exit 1
fi

DRY_RUN="${DRY_RUN:-yes}"
CACHE_DIR="${CACHE_DIR:-/tmp/contextcache_data_load_mult_delayed}"
EXAMPLE_CONFIG="${EXAMPLE_CONFIG:-config/request_gen.example.json}"
LOAD_ADAPTIVE_RATIO="${LOAD_ADAPTIVE_RATIO:-2.0}"
LOAD_ADAPTIVE_TOKEN_RATIO="${LOAD_ADAPTIVE_TOKEN_RATIO:-4.0}"
LOAD_MULT="${LOAD_MULT:-10}"
LOAD_MULT_START_S="${LOAD_MULT_START_S:-240}"

# This sweep intentionally does NOT override experiment_duration_seconds,
# max_turns_per_conversation, or max_conversations_per_app — they come from
# EXAMPLE_CONFIG as-is so the LM=10 phase has enough time after the delayed switch.

# Isolated from run_experiments.sh and run_load_multiplier_experiments.sh.
DATA_ROOT="data/test_apps/load_mult_delayed_compare"
PLOTS_ROOT="${DATA_ROOT}/plots"

OPENAI_API_KEY_VALUE="${OPENAI_API_KEY:-}"
OPENAI_API_BASE_VALUE="${OPENAI_API_BASE:-https://api.openai.com/v1}"
VOCAREUM_API_KEY_VALUE="${VOCAREUM_API_KEY:-}"
VOCAREUM_API_BASE_VALUE="${VOCAREUM_BASE_URL:-https://genai.vocareum.com/v1}"

export VOCAREUM_API_KEY=
export VOCAREUM_BASE_URL=https://genai.vocareum.com/v1


mkdir -p "${DATA_ROOT}"
mkdir -p "${PLOTS_ROOT}"

# Subdir name encodes both the target LM and the delay so multiple sweeps don't collide.
LM_TAG="lm${LOAD_MULT}_delayed${LOAD_MULT_START_S}"

server_cmd_base=(
  env -u APPIMAGE "${PYTHON_BIN}" -m gptcache_server.server
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

append_if_set() {
  local -n out_ref="$1"
  local flag="$2"
  local value="${3-}"
  if [[ -n "${value}" ]]; then
    out_ref+=("${flag}" "${value}")
  fi
}

run_experiment() {
  local mode="$1"
  local suffix="$2"
  shift 2
  local extra_server_args=("$@")

  local lm_metrics_dir="${DATA_ROOT}/${LM_TAG}"
  mkdir -p "${lm_metrics_dir}"

  echo "=== Running experiment: LM=${LOAD_MULT} (delayed start ${LOAD_MULT_START_S}s), mode=${mode}, suffix=${suffix}, DRY_RUN=${DRY_RUN} ==="
  if ((${#extra_server_args[@]} > 0)); then
    echo "    extra server args: ${extra_server_args[*]}"
  fi

  rm -rf "${CACHE_DIR}"
  mkdir -p "${CACHE_DIR}"

  local tmp_config
  tmp_config="$(mktemp "/tmp/request_gen_${LM_TAG}.${suffix}.XXXX.json")"

  local metrics_path="${lm_metrics_dir}/request_metrics_${suffix}.json"
  local log_path="${lm_metrics_dir}/request_log_${suffix}.jsonl"

  # Cached runs use the nocache baseline log for accuracy comparison when available.
  # When nocache runs are skipped, fall back to an empty path so generate_requests.py
  # doesn't raise FileNotFoundError.
  local baseline_abs="${PWD}/${lm_metrics_dir}/request_log_nocache.jsonl"
  local baseline_arg=""
  if [[ -f "${baseline_abs}" ]]; then
    baseline_arg="${baseline_abs}"
  fi

  if [[ "${suffix}" == "nocache" ]]; then
    jq \
      --arg metrics_path "${metrics_path}" \
      --arg log_path "${log_path}" \
      --argjson load_multiplier "${LOAD_MULT}" \
      --argjson lm_start_s "${LOAD_MULT_START_S}" \
      '.output_metrics_path = $metrics_path
        | .output_request_log_path = $log_path
        | .load_multiplier = $load_multiplier
        | .load_multiplier_start_seconds = $lm_start_s
        | .accuracy_baseline_reference_run = true
        | .accuracy_baseline_log_path = ""' \
      "${EXAMPLE_CONFIG}" > "${tmp_config}"
  else
    jq \
      --arg metrics_path "${metrics_path}" \
      --arg log_path "${log_path}" \
      --arg baseline "${baseline_arg}" \
      --argjson load_multiplier "${LOAD_MULT}" \
      --argjson lm_start_s "${LOAD_MULT_START_S}" \
      '.output_metrics_path = $metrics_path
        | .output_request_log_path = $log_path
        | .load_multiplier = $load_multiplier
        | .load_multiplier_start_seconds = $lm_start_s
        | .accuracy_baseline_reference_run = false
        | .accuracy_baseline_log_path = $baseline' \
      "${EXAMPLE_CONFIG}" > "${tmp_config}"
  fi

  local server_cmd=("${server_cmd_base[@]}" "--server-mode" "${mode}" "${extra_server_args[@]}")
  local server_log="${lm_metrics_dir}/server_${suffix}.log"
  "${server_cmd[@]}" >"${server_log}" 2>&1 &
  local server_pid=$!

  trap 'kill "${server_pid}" >/dev/null 2>&1 || true' EXIT

  wait_for_server

  "${PYTHON_BIN}" scripts/generate_requests.py --config "${tmp_config}"

  kill "${server_pid}" >/dev/null 2>&1 || true
  wait "${server_pid}" 2>/dev/null || true
  trap - EXIT

  rm -f "${tmp_config}"

  local plot_prefix="${LM_TAG}_${suffix}"
  local plots_dir="${PLOTS_ROOT}/${LM_TAG}/${suffix}"
  mkdir -p "${plots_dir}"

  "${PYTHON_BIN}" scripts/plot_request_metrics.py \
    --metrics "${metrics_path}" \
    --output-dir "${plots_dir}" \
    --prefix "${plot_prefix}"

  echo "Experiment LM=${LOAD_MULT} (delayed ${LOAD_MULT_START_S}s) ${mode} complete."
  echo "  Metrics: ${metrics_path}"
  echo "  Log:     ${log_path}"
  echo "  Plots:   ${plots_dir}"
  echo
}

D_ABS="${PWD}/${DATA_ROOT}"

# RUN_ONLY_SUFFIXES (space-separated list) limits the sweep to the specified suffixes
# (nocache, gptcache, contextcache, adaptive_load). Default: run all four.
RUN_ONLY_SUFFIXES="${RUN_ONLY_SUFFIXES:-nocache gptcache contextcache adaptive_load}"
should_run() {
  local want="$1"
  [[ " ${RUN_ONLY_SUFFIXES} " == *" ${want} "* ]]
}

# TODO(re-enable): nocache baseline run (needed for accuracy comparisons and compare_all plot).
if should_run "nocache"; then
  run_experiment "no-cache" "nocache"
fi
# TODO(re-enable): gptcache mode.
if should_run "gptcache"; then
  run_experiment "gptcache" "gptcache"
fi
if should_run "contextcache"; then
  run_experiment "contextcache" "contextcache"
fi
if should_run "adaptive_load"; then
  adaptive_args=(
    --load-adaptive
    --load-adaptive-ratio "${LOAD_ADAPTIVE_RATIO}"
    --load-adaptive-token-ratio "${LOAD_ADAPTIVE_TOKEN_RATIO}"
  )
  run_experiment "adaptivecontextcache" "adaptive_load" "${adaptive_args[@]}"
fi

compare_dir="${PLOTS_ROOT}/compare_${LM_TAG}"
mkdir -p "${compare_dir}"

"${PYTHON_BIN}" scripts/compare_experiments.py \
  --run "nocache:${D_ABS}/${LM_TAG}/request_metrics_nocache.json" \
  --run "gptcache:${D_ABS}/${LM_TAG}/request_metrics_gptcache.json" \
  --run "contextcache:${D_ABS}/${LM_TAG}/request_metrics_contextcache.json" \
  --run "adaptive_load:${D_ABS}/${LM_TAG}/request_metrics_adaptive_load.json" \
  --output-dir "${compare_dir}" \
  --prefix "compare_${LM_TAG}"

timeseries_suffix_args=()
for suf in ${RUN_ONLY_SUFFIXES}; do
  timeseries_suffix_args+=(--suffix "${suf}")
done

if ((${#timeseries_suffix_args[@]} > 0)); then
  echo "Writing delayed experiment latency/accuracy time-series plots..."
  "${PYTHON_BIN}" scripts/plot_delayed_experiment_timeseries.py \
    --root-dir "${D_ABS}" \
    --lm-tag "${LM_TAG}" \
    "${timeseries_suffix_args[@]}" \
    --output-dir "${compare_dir}" \
    --prefix "delayed_timeseries"
fi

echo "All delayed-load-multiplier experiments complete."
echo "  Metrics/logs: ${DATA_ROOT}/${LM_TAG}/"
echo "  Compare:      ${compare_dir}"
