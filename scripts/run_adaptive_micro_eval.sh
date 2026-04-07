#!/usr/bin/env bash
set -euo pipefail

# Micro evaluation for adaptive context cache.
# Runs two experiments:
#   1) baseline contextcache (no load-adaptive)
#   2) adaptivecontextcache with --load-adaptive enabled
#
# Each experiment has two phases against the same live server:
#   - low load (minute ~1)
#   - high load burst (minute ~2)
#
# The script then plots key metrics and prints a concise observation.
#
# Usage:
#   chmod +x scripts/run_adaptive_micro_eval.sh
#   ./scripts/run_adaptive_micro_eval.sh
# Optional knobs:
#   DRY_RUN=yes|no (default: no)
#   PORT=8013
#   LOW_DURATION_S=70
#   HIGH_DURATION_S=70
#   LOW_MULTIPLIER=0.6
#   HIGH_MULTIPLIER=4.0

DRY_RUN="${DRY_RUN:-no}"                     # yes|no
PORT="${PORT:-8013}"
HOST="${HOST:-127.0.0.1}"
CACHE_ROOT="${CACHE_ROOT:-/tmp/adaptive_micro_eval_cache}"
OUT_DIR="${OUT_DIR:-data/test_apps/adaptive_micro_eval}"
EXAMPLE_CONFIG="${EXAMPLE_CONFIG:-config/request_gen.example.json}"
LOW_DURATION_S="${LOW_DURATION_S:-70}"
HIGH_DURATION_S="${HIGH_DURATION_S:-70}"
LOW_MULTIPLIER="${LOW_MULTIPLIER:-0.6}"
HIGH_MULTIPLIER="${HIGH_MULTIPLIER:-4.0}"
LOAD_ADAPTIVE_RATIO="${LOAD_ADAPTIVE_RATIO:-2.0}"
CONTEXT_WINDOW_LEN="${CONTEXT_WINDOW_LEN:-8}"
MODEL="${MODEL:-@azure-1/gpt-4o}"
API_KEY_ENV="${API_KEY_ENV:-VOCAREUM_API_KEY}"

mkdir -p "${OUT_DIR}"

if ! python3 -c "import matplotlib" >/dev/null 2>&1; then
  echo "Missing dependency: matplotlib"
  echo "Install it with: python3 -m pip install matplotlib"
  exit 1
fi

wait_for_server() {
  local timeout_s=40
  local start
  start="$(date +%s)"
  while true; do
    if curl -sSf "http://${HOST}:${PORT}/" >/dev/null 2>&1; then
      return 0
    fi
    local now
    now="$(date +%s)"
    if (( now - start > timeout_s )); then
      echo "Server did not become ready within ${timeout_s}s on ${HOST}:${PORT}" >&2
      return 1
    fi
    sleep 1
  done
}

build_phase_config() {
  local mode_tag="$1"     # baseline | adaptive
  local phase_tag="$2"    # low | high
  local duration_s="$3"
  local load_multiplier="$4"
  local out_config="$5"

  local metrics_path="${OUT_DIR}/request_metrics_${mode_tag}_${phase_tag}.json"
  local request_log_path="${OUT_DIR}/request_log_${mode_tag}_${phase_tag}.jsonl"

  jq \
    --arg base_url "http://${HOST}:${PORT}" \
    --arg model "${MODEL}" \
    --arg api_key_env "${API_KEY_ENV}" \
    --arg metrics_path "${metrics_path}" \
    --arg log_path "${request_log_path}" \
    --argjson duration_s "${duration_s}" \
    --argjson load_multiplier "${load_multiplier}" \
    --argjson dry_run "$( [[ "${DRY_RUN}" == "yes" ]] && echo "true" || echo "false" )" \
    '
      .base_url = $base_url
      | .model = $model
      | .api_key_env = $api_key_env
      | .warmup_enabled = false
      | .accuracy_enabled = false
      | .accuracy_baseline_reference_run = false
      | .accuracy_baseline_log_path = ""
      | .applications = [0, 1]
      | .default_threads_per_app = 1
      | .threads_per_application = {"0": 1, "1": 1}
      | .default_delay_ms = 700
      | .app_delay_ms = {"0": 700, "1": 700}
      | .max_conversations_per_app = 4
      | .max_turns_per_conversation = 4
      | .retry_count = 1
      | .retry_backoff_ms = 0
      | .timeout_seconds = 30
      | .max_workers = 8
      | .experiment_duration_seconds = $duration_s
      | .load_multiplier = $load_multiplier
      | .output_metrics_path = $metrics_path
      | .output_request_log_path = $log_path
      | .warmup_log_path = "'"${OUT_DIR}"'/warmup_log_unused.jsonl"
      | .dry_run = $dry_run
    ' \
    "${EXAMPLE_CONFIG}" > "${out_config}"
}

run_phase() {
  local mode_tag="$1"
  local phase_tag="$2"
  local duration_s="$3"
  local load_multiplier="$4"

  local cfg
  cfg="$(mktemp "/tmp/adaptive_micro_eval.${mode_tag}.${phase_tag}.XXXX.json")"
  build_phase_config "${mode_tag}" "${phase_tag}" "${duration_s}" "${load_multiplier}" "${cfg}"

  echo "Running phase: mode=${mode_tag}, phase=${phase_tag}, duration=${duration_s}s, load_multiplier=${load_multiplier}"
  python3 scripts/generate_requests.py --config "${cfg}"
  rm -f "${cfg}"
}

run_experiment() {
  local mode_tag="$1"          # baseline | adaptive
  local server_mode="$2"       # contextcache | adaptivecontextcache
  local enable_adaptive="$3"   # yes | no

  local cache_dir="${CACHE_ROOT}/${mode_tag}"
  local server_log="${OUT_DIR}/server_${mode_tag}.log"

  rm -rf "${cache_dir}"
  mkdir -p "${cache_dir}"

  local server_cmd=(
    env -u APPIMAGE /usr/bin/python3
    -c "import logging; logging.basicConfig(level=logging.INFO); from gptcache_server.server import main; main()"
    -s "${HOST}"
    -p "${PORT}"
    -d "${cache_dir}"
    -o True
    --server-mode "${server_mode}"
    --context-cache-window-len "${CONTEXT_WINDOW_LEN}"
    --load-adaptive-ratio "${LOAD_ADAPTIVE_RATIO}"
    -dr "${DRY_RUN}"
  )
  if [[ "${enable_adaptive}" == "yes" ]]; then
    server_cmd+=("--load-adaptive")
  fi

  echo "=== Start ${mode_tag} (${server_mode}, load_adaptive=${enable_adaptive}) ==="
  "${server_cmd[@]}" > "${server_log}" 2>&1 &
  local server_pid=$!

  cleanup_server() {
    kill "${server_pid}" >/dev/null 2>&1 || true
    wait "${server_pid}" 2>/dev/null || true
  }
  trap cleanup_server EXIT

  wait_for_server

  # Minute 1: low load. Minute 2: burst load.
  run_phase "${mode_tag}" "low" "${LOW_DURATION_S}" "${LOW_MULTIPLIER}"
  run_phase "${mode_tag}" "high" "${HIGH_DURATION_S}" "${HIGH_MULTIPLIER}"

  cleanup_server
  trap - EXIT

  echo "=== Done ${mode_tag}; server log: ${server_log} ==="
}

run_experiment "baseline" "contextcache" "no"
run_experiment "adaptive" "adaptivecontextcache" "yes"

python3 - <<'PY'
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out_dir = Path(os.environ.get("OUT_DIR", "data/test_apps/adaptive_micro_eval"))

def load_metrics(mode, phase):
    p = out_dir / f"request_metrics_{mode}_{phase}.json"
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def hit_rate(m):
    req = float(m.get("requests_sent", 0) or 0)
    hits = float(m.get("cache_hits", 0) or 0)
    return (hits / req) if req > 0 else 0.0

metrics = {
    ("baseline", "low"): load_metrics("baseline", "low"),
    ("baseline", "high"): load_metrics("baseline", "high"),
    ("adaptive", "low"): load_metrics("adaptive", "low"),
    ("adaptive", "high"): load_metrics("adaptive", "high"),
}

lat_key = "served_p95_latency_ms"
thr_key = "overall_throughput_rps"

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

labels = ["baseline", "adaptive"]
x = [0, 1]

for phase, color in [("low", "#8ab6d6"), ("high", "#ff9f80")]:
    y = [float(metrics[(m, phase)].get(lat_key, 0.0)) for m in labels]
    axes[0].plot(x, y, marker="o", linewidth=2, color=color, label=f"{phase} load")
axes[0].set_xticks(x)
axes[0].set_xticklabels(labels)
axes[0].set_title("served p95 latency (ms)")
axes[0].grid(alpha=0.3)
axes[0].legend()

for phase, color in [("low", "#8ab6d6"), ("high", "#ff9f80")]:
    y = [float(metrics[(m, phase)].get(thr_key, 0.0)) for m in labels]
    axes[1].plot(x, y, marker="o", linewidth=2, color=color, label=f"{phase} load")
axes[1].set_xticks(x)
axes[1].set_xticklabels(labels)
axes[1].set_title("throughput (rps)")
axes[1].grid(alpha=0.3)
axes[1].legend()

for phase, color in [("low", "#8ab6d6"), ("high", "#ff9f80")]:
    y = [hit_rate(metrics[(m, phase)]) for m in labels]
    axes[2].plot(x, y, marker="o", linewidth=2, color=color, label=f"{phase} load")
axes[2].set_xticks(x)
axes[2].set_xticklabels(labels)
axes[2].set_ylim(0.0, 1.0)
axes[2].set_title("cache hit rate")
axes[2].grid(alpha=0.3)
axes[2].legend()

fig.tight_layout()
plot_path = out_dir / "adaptive_micro_eval_compare.png"
fig.savefig(plot_path, dpi=160)
plt.close(fig)

def pct_delta(new, old):
    old = float(old)
    new = float(new)
    if abs(old) < 1e-9:
        return None
    return (new - old) / old * 100.0

b_hi = metrics[("baseline", "high")]
a_hi = metrics[("adaptive", "high")]

b_hi_p95 = float(b_hi.get(lat_key, 0.0))
a_hi_p95 = float(a_hi.get(lat_key, 0.0))
b_hi_thr = float(b_hi.get(thr_key, 0.0))
a_hi_thr = float(a_hi.get(thr_key, 0.0))
b_hi_hit = hit_rate(b_hi)
a_hi_hit = hit_rate(a_hi)

lat_delta = pct_delta(a_hi_p95, b_hi_p95)
thr_delta = pct_delta(a_hi_thr, b_hi_thr)
hit_delta = pct_delta(a_hi_hit, b_hi_hit)

def fmt_pct(v):
    return "n/a" if v is None else f"{v:+.2f}%"

print("")
print("=== Micro-eval observation ===")
print(f"Plot: {plot_path}")
print(f"High-load served p95 latency: baseline={b_hi_p95:.2f} ms, adaptive={a_hi_p95:.2f} ms, delta={fmt_pct(lat_delta)}")
print(f"High-load throughput: baseline={b_hi_thr:.3f} rps, adaptive={a_hi_thr:.3f} rps, delta={fmt_pct(thr_delta)}")
print(f"High-load cache-hit-rate: baseline={b_hi_hit:.3f}, adaptive={a_hi_hit:.3f}, delta={fmt_pct(hit_delta)}")

if lat_delta is not None and thr_delta is not None:
    if lat_delta < 0 and thr_delta >= 0:
        print("Conclusion: adaptive mode improved tail latency under burst load without throughput regression.")
    elif lat_delta > 0 and thr_delta < 0:
        print("Conclusion: adaptive mode regressed both latency and throughput in this micro workload.")
    else:
        print("Conclusion: mixed result; inspect cache-hit-rate and server log for adaptive-window behavior.")
else:
    print("Conclusion: insufficient non-zero baseline values to compute percentage deltas.")
PY

if command -v rg >/dev/null 2>&1; then
  adaptive_events="$( (rg -n "load_adaptive:" "${OUT_DIR}/server_adaptive.log" || true) | wc -l | tr -d ' ' )"
  echo "Adaptive resize log events detected: ${adaptive_events}"
fi

echo ""
echo "Artifacts written to: ${OUT_DIR}"
echo "- request_metrics_baseline_low.json"
echo "- request_metrics_baseline_high.json"
echo "- request_metrics_adaptive_low.json"
echo "- request_metrics_adaptive_high.json"
echo "- adaptive_micro_eval_compare.png"
echo "- server_baseline.log / server_adaptive.log"
