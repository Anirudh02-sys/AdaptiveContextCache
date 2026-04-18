#!/usr/bin/env bash
# Load-multiplier sweep: same four cache modes as run_experiments.sh, but experiment 4 uses
# load-aware adaptive context (--load-adaptive) instead of SLO-adaptive (--slo-adaptive).
#
# For each load_multiplier in {1, 10} (see config/request_gen.example.json "load_multiplier"):
#   1) no-cache
#   2) gptcache
#   3) contextcache
#   4) adaptivecontextcache + --load-adaptive
#
# Artifacts are written ONLY under data/test_apps/load_mult_compare/ (not data/test_apps/example/),
# so they do not overwrite runs from scripts/run_experiments.sh.
#
# Layout:
#   data/test_apps/load_mult_compare/lm1|lm10/request_metrics_<suffix>.json
#   data/test_apps/load_mult_compare/plots/lm1|lm10/<suffix>/   (per-run plots)
#   data/test_apps/load_mult_compare/plots/compare_all/         (8-way compare, both loads)
#   data/test_apps/load_mult_compare/plots/compare_lm1/         (4 runs, LM=1 only)
#   data/test_apps/load_mult_compare/plots/compare_lm10/        (4 runs, LM=10 only)
#
# Usage:
#   ./scripts/run_load_multiplier_experiments.sh
#   DRY_RUN=yes ./scripts/run_load_multiplier_experiments.sh
#
# Environment:
#   PYTHON              Python interpreter (default: .venv/bin/python or python3)
#   DRY_RUN             yes|no — server upstream dry-run (default: no)
#   CACHE_DIR           Server cache directory cleared each run (default: /tmp/contextcache_data_load_mult)
#   LOAD_ADAPTIVE_RATIO Passed to --load-adaptive-ratio (default: 2.0, must be > 1)
#   EXAMPLE_CONFIG      Request-gen template (default: config/request_gen.example.json)
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

DRY_RUN="${DRY_RUN:-no}"
CACHE_DIR="${CACHE_DIR:-/tmp/contextcache_data_load_mult}"
EXAMPLE_CONFIG="${EXAMPLE_CONFIG:-config/request_gen.example.json}"
LOAD_ADAPTIVE_RATIO="${LOAD_ADAPTIVE_RATIO:-2.0}"

# Workload characteristics (threads_per_application, app_delay_ms, etc.) come from
# EXAMPLE_CONFIG unchanged. With the current request_gen.example.json (7 threads across
# 5 apps with per-app delays 1600/4500/2800/1400/3800 ms), offered rates are
# ~3.52 req/s at LM=1 and ~35.2 req/s at LM=10.

# Load-adaptive controller thresholds. `curr_rps` at the cache is thread-bottlenecked by
# upstream server latency (threads sleep AFTER each call), so arrival rates are below the
# offered rates: LM=1 lands in ~2.25-3.52 rps, LM=10 in ~5.7-35 rps. The 4.0 rps gate
# below sits in the separation band, firing force-shrink only at LM=10 while leaving the
# LM=1 window untouched at the base size.
LOAD_ADAPTIVE_SHRINK_MIN_RPS="${LOAD_ADAPTIVE_SHRINK_MIN_RPS:-4.0}"
LOAD_ADAPTIVE_FORCE_SHRINK_RPS="${LOAD_ADAPTIVE_FORCE_SHRINK_RPS:-4.0}"
# Growth ceiling: a grow step is gated by curr_rps <= this value. LM=1's floor is
# ~2.25 rps (well above 1.0), so warmup->steady ratio drops at LM=1 never grow the window.
LOAD_ADAPTIVE_GROW_MAX_RPS="${LOAD_ADAPTIVE_GROW_MAX_RPS:-1.0}"

# Optional overrides to keep the 8-run sweep under tight time budgets. When set, they
# override the corresponding fields in EXAMPLE_CONFIG for every run. Leave unset to use
# whatever EXAMPLE_CONFIG already specifies.
LOAD_MULT_CMP_EXPERIMENT_DURATION_S="${LOAD_MULT_CMP_EXPERIMENT_DURATION_S:-}"
LOAD_MULT_CMP_MAX_TURNS_PER_CONV="${LOAD_MULT_CMP_MAX_TURNS_PER_CONV:-}"
LOAD_MULT_CMP_MAX_CONVS_PER_APP="${LOAD_MULT_CMP_MAX_CONVS_PER_APP:-}"

# Isolated from run_experiments.sh (data/test_apps/example).
DATA_ROOT="data/test_apps/load_mult_compare"
PLOTS_ROOT="${DATA_ROOT}/plots"

OPENAI_API_KEY_VALUE="${OPENAI_API_KEY:-}"
OPENAI_API_BASE_VALUE="${OPENAI_API_BASE:-https://api.openai.com/v1}"
VOCAREUM_API_KEY_VALUE="${VOCAREUM_API_KEY:-}"
VOCAREUM_API_BASE_VALUE="${VOCAREUM_BASE_URL:-https://genai.vocareum.com/v1}"

export VOCAREUM_API_KEY=
export VOCAREUM_BASE_URL=https://genai.vocareum.com/v1


mkdir -p "${DATA_ROOT}"
mkdir -p "${PLOTS_ROOT}"

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

run_experiment() {
  local load_mult="$1"
  local mode="$2"
  local suffix="$3"
  shift 3
  local extra_server_args=("$@")

  local lm_metrics_dir="${DATA_ROOT}/lm${load_mult}"
  mkdir -p "${lm_metrics_dir}"

  echo "=== Running experiment: LM=${load_mult}, mode=${mode}, suffix=${suffix}, DRY_RUN=${DRY_RUN} ==="
  if ((${#extra_server_args[@]} > 0)); then
    echo "    extra server args: ${extra_server_args[*]}"
  fi

  rm -rf "${CACHE_DIR}"
  mkdir -p "${CACHE_DIR}"

  local tmp_config
  tmp_config="$(mktemp "/tmp/request_gen_lm${load_mult}.${suffix}.XXXX.json")"

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

  # Optional overrides default to null (leave original value) when the env var is unset.
  local exp_dur_json="${LOAD_MULT_CMP_EXPERIMENT_DURATION_S:-null}"
  local max_turns_json="${LOAD_MULT_CMP_MAX_TURNS_PER_CONV:-null}"
  local max_convs_json="${LOAD_MULT_CMP_MAX_CONVS_PER_APP:-null}"
  local jq_overrides='
        | (if $exp_dur != null then .experiment_duration_seconds = $exp_dur else . end)
        | (if $max_turns != null then .max_turns_per_conversation = $max_turns else . end)
        | (if $max_convs != null then .max_conversations_per_app = $max_convs else . end)'

  if [[ "${suffix}" == "nocache" ]]; then
    jq \
      --arg metrics_path "${metrics_path}" \
      --arg log_path "${log_path}" \
      --argjson load_multiplier "${load_mult}" \
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
      --argjson load_multiplier "${load_mult}" \
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

  local plot_prefix="lm${load_mult}_${suffix}"
  local plots_dir="${PLOTS_ROOT}/lm${load_mult}/${suffix}"
  mkdir -p "${plots_dir}"

  "${PYTHON_BIN}" scripts/plot_request_metrics.py \
    --metrics "${metrics_path}" \
    --output-dir "${plots_dir}" \
    --prefix "${plot_prefix}"

  echo "Experiment LM=${load_mult} ${mode} complete."
  echo "  Metrics: ${metrics_path}"
  echo "  Log:     ${log_path}"
  echo "  Plots:   ${plots_dir}"
  echo
}

D_ABS="${PWD}/${DATA_ROOT}"

# RUN_ONLY_SUFFIXES (space-separated list) limits the sweep to the specified suffixes
# (nocache, gptcache, contextcache, adaptive_load). Default: run all four per LM.
RUN_ONLY_SUFFIXES="${RUN_ONLY_SUFFIXES:-nocache gptcache contextcache adaptive_load}"
should_run() {
  local want="$1"
  [[ " ${RUN_ONLY_SUFFIXES} " == *" ${want} "* ]]
}

for LOAD_MULT in 1 10; do
  # TODO(re-enable): nocache baseline run (needed for accuracy comparisons and compare_all plot).
  # if should_run "nocache"; then
  #   run_experiment "${LOAD_MULT}" "no-cache" "nocache"
  # fi
  # TODO(re-enable): gptcache mode.
  # if should_run "gptcache"; then
  #   run_experiment "${LOAD_MULT}" "gptcache" "gptcache"
  # fi
  if should_run "contextcache"; then
    run_experiment "${LOAD_MULT}" "contextcache" "contextcache"
  fi
  if should_run "adaptive_load"; then
    run_experiment "${LOAD_MULT}" "adaptivecontextcache" "adaptive_load" \
      --load-adaptive \
      --load-adaptive-ratio "${LOAD_ADAPTIVE_RATIO}" \
      --load-adaptive-shrink-min-rps "${LOAD_ADAPTIVE_SHRINK_MIN_RPS}" \
      --load-adaptive-force-shrink-rps "${LOAD_ADAPTIVE_FORCE_SHRINK_RPS}" \
      --load-adaptive-grow-max-rps "${LOAD_ADAPTIVE_GROW_MAX_RPS}"
  fi
done

# TODO(re-enable): compare_all (8-run) plot — requires nocache and gptcache runs.
# compare_all_dir="${PLOTS_ROOT}/compare_all"
# mkdir -p "${compare_all_dir}"
#
# "${PYTHON_BIN}" scripts/compare_experiments.py \
#   --run "nocache_lm1:${D_ABS}/lm1/request_metrics_nocache.json" \
#   --label "nocache_lm1:no-cache (LM=1)" \
#   --run "gptcache_lm1:${D_ABS}/lm1/request_metrics_gptcache.json" \
#   --label "gptcache_lm1:gptcache (LM=1)" \
#   --run "contextcache_lm1:${D_ABS}/lm1/request_metrics_contextcache.json" \
#   --label "contextcache_lm1:contextcache (LM=1)" \
#   --run "adaptive_load_lm1:${D_ABS}/lm1/request_metrics_adaptive_load.json" \
#   --label "adaptive_load_lm1:adaptive context (load) (LM=1)" \
#   --run "nocache_lm10:${D_ABS}/lm10/request_metrics_nocache.json" \
#   --label "nocache_lm10:no-cache (LM=10)" \
#   --run "gptcache_lm10:${D_ABS}/lm10/request_metrics_gptcache.json" \
#   --label "gptcache_lm10:gptcache (LM=10)" \
#   --run "contextcache_lm10:${D_ABS}/lm10/request_metrics_contextcache.json" \
#   --label "contextcache_lm10:contextcache (LM=10)" \
#   --run "adaptive_load_lm10:${D_ABS}/lm10/request_metrics_adaptive_load.json" \
#   --label "adaptive_load_lm10:adaptive context (load) (LM=10)" \
#   --output-dir "${compare_all_dir}" \
#   --prefix "compare"

compare_lm1_dir="${PLOTS_ROOT}/compare_lm1"
mkdir -p "${compare_lm1_dir}"

"${PYTHON_BIN}" scripts/compare_experiments.py \
  `# TODO(re-enable): --run "nocache:${D_ABS}/lm1/request_metrics_nocache.json"` \
  `# TODO(re-enable): --run "gptcache:${D_ABS}/lm1/request_metrics_gptcache.json"` \
  --run "contextcache:${D_ABS}/lm1/request_metrics_contextcache.json" \
  --run "adaptive_load:${D_ABS}/lm1/request_metrics_adaptive_load.json" \
  --output-dir "${compare_lm1_dir}" \
  --prefix "compare_lm1"

compare_lm10_dir="${PLOTS_ROOT}/compare_lm10"
mkdir -p "${compare_lm10_dir}"

"${PYTHON_BIN}" scripts/compare_experiments.py \
  `# TODO(re-enable): --run "nocache:${D_ABS}/lm10/request_metrics_nocache.json"` \
  `# TODO(re-enable): --run "gptcache:${D_ABS}/lm10/request_metrics_gptcache.json"` \
  --run "contextcache:${D_ABS}/lm10/request_metrics_contextcache.json" \
  --run "adaptive_load:${D_ABS}/lm10/request_metrics_adaptive_load.json" \
  --output-dir "${compare_lm10_dir}" \
  --prefix "compare_lm10"

echo "All load-multiplier experiments complete."
echo "  Metrics/logs: ${DATA_ROOT}/lm1/ and ${DATA_ROOT}/lm10/"
echo "  Compare LM=1:  ${compare_lm1_dir}"
echo "  Compare LM=10: ${compare_lm10_dir}"
