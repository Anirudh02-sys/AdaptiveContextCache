# README

## Usage

**Run the server:** From the repository root (where `setup.py` lives), start the process with any of the flags in the table below:

```bash
cd /path/to/AdaptiveContextCache
python -m gptcache_server.server -s 127.0.0.1 -p 8012 -d /tmp/contextcache_data -o True
```

Append any additional flags from the table below (for example `--load-adaptive`, `-m adaptivecontextcache`, `--dry-run yes`).

### Cache server CLI

| Flag | Meaning |
|------|--------|
| `-s` / `--host` | Bind host (default `localhost`). |
| `-p` / `--port` | Listen port (default `8000`). |
| `-d` / `--cache-dir` | On-disk cache data directory. |
| `-k` / `--cache-file-key` | Optional cache file key. |
| `-f` / `--cache-config-file` | YAML/JSON cache init file (optional). |
| `-o` / `--openai` | Enable OpenAI-compatible routes (e.g. `/v1/chat/completions`). |
| `-of` / `--openai-cache-config-file` | Separate cache config for the OpenAI proxy cache instance. |
| `-dr` / `--dry-run` | `yes` or `no`: if `yes`, upstream LLM calls can be suppressed for testing. |
| `-cw` / `--context-cache-window-len` | Base dialogue embedding window length for context matching (default `5`; overrides config from `-f` when set). |
| `-m` / `--server-mode` | `no-cache`, `gptcache`, `contextcache`, or `adaptivecontextcache`. |
| `--load-adaptive` | Enable per-minute load tracking and adjust the effective context window via `context_cache_overall_factor` (see **Source code map**). |
| `--load-adaptive-ratio` | `R` must be greater than 1: shrink window when load ≥ R× the previous minute; grow when load ≤ 1/R× (default `2.0`). |
| `--slo-adaptive` | When set, enables SLO-driven per-application context-window factors when application SLOs are registered (see HTTP API below). |

**Core networking and storage:** `--host` / `--port` control where **uvicorn** listens. `--cache-dir` is the working directory for persisted cache state; `--cache-file-key` and `--cache-config-file` let you use file-based or YAML-driven initialization instead of all-defaults. **`--dry-run yes`** avoids calling real upstream LLMs where the adapter supports it—useful when only cache behavior matters.

**OpenAI proxy:** `-o` / `--openai` turns on routes such as **`/v1/chat/completions`**. If you need different storage or evaluation stacks for the “raw” cache API versus the OpenAI shim, pass **`-of`** with a second cache config.

**Context window:** **`-cw`** sets the **base** `context_cache_window_len` used everywhere the effective window is computed (before per-app and load factors). It overrides the value embedded in a config file from **`-f`**.

**Load adaptation:** With **`--load-adaptive`**, each request (via the OpenAI adapter) feeds **token and request counts** into **`LoadAdaptiveContextController`**. Once per minute, the controller compares the sliding window to the **previous** minute; if traffic grew by at least **`--load-adaptive-ratio`** it **shrinks** the effective window (favoring cheaper, cache-friendly paths), and if traffic dropped by the inverse ratio it **grows** the window (up to **`context_cache_window_max`** in config). This keeps the dialogue window responsive to load without manual restarts.

**SLO adaptation:** With **`--slo-adaptive`**, **`on_application_registry_changed()`** recomputes **per-application additive offsets** `context_cache_window_delta_by_app` from the **latency** and **accuracy** targets stored for each server-issued application id. Stricter latency goals bias toward **negative** deltas (shorter effective windows, more aggressive caching); stricter accuracy goals bias toward **positive** deltas (longer windows, richer context). When the flag is off, every app delta stays at **0** regardless of registrations.

Clients register targets with:

- **`POST /v1/applications`** — JSON body with **`latency_p99_ms`** and **`accuracy_slo`** (same semantics as `application_slo_expectations` in the request generator). The server returns a new **`application_id`** (UUID string).
- **`DELETE /v1/applications/{application_id}`** — drop that app’s SLO record and recompute factors.

(Request/response **Pydantic** models live in [`gptcache_server/server.py`](gptcache_server/server.py).)

## Major Functions and Source Code Organization

This section describes the major functions of our source code and where they are implemented. The primary focus of this project is the load-adaptive and SLO-adaptive extensions to ContextCache.

### 1. Core Adaptive Cache Implementation (Main Contribution)

- **`gptcache_server/server.py`** — Main server entry point and adaptive feature control.
  - `main()` parses command-line arguments, initializes the cache, selects the server mode, and starts the HTTP server.
  - `on_application_registry_changed()` recomputes per-application context-window offsets when SLO registrations change.
  - `register_application_slo` adds a new application's latency and accuracy targets.
  - `deregister_application` removes an application's registration and refreshes active offsets.

- **`gptcache/utils/adaptive_window.py`** — Implements load-adaptive context-window logic.
  - `LoadAdaptiveMinuteWindow` tracks recent request counts and token counts over a sliding time window.
  - `LoadAdaptiveContextController` compares recent load against the previous interval and shrinks or expands the effective context window.

- **`gptcache/config.py`** — Defines adaptive configuration and runtime window computation.
  - `Config` stores the base context window, global adaptive factor, per-application offsets, and adaptive feature flags.
  - `effective_context_window_len(application_id)` computes the final context window used for a request by combining the base window, load-adaptive global factor, and per-application SLO offset.

- **`gptcache/adapter/adapter.py`** — Existing ContextCache retrieval path; in this fork, it was modified to apply `effective_context_window_len(application_id)` during cache matching.

- **`gptcache/adapter/openai.py`** — Existing model-compatible request path; in this fork, it was modified to forward request load statistics to the adaptive controller when enabled.

### 2. Experiment Drivers (Evaluation of Load / SLO Changes)

- **`scripts/generate_requests.py`** — Main workload generator.
  - Loads application workload files.
  - Replays multi-turn requests against the server.
  - Records latency, cache hits, accuracy metrics, and SLO attainment.

- **`scripts/run_experiments.sh`** — Runs baseline comparisons across cache modes.

- **`scripts/run_load_multiplier_experiments.sh`** — Main load-scaling evaluation script that compares system behavior under different workload intensities.

- **`scripts/run_lm10_load_slo_adaptive_experiments.sh`** — Runs focused experiments combining load-adaptive and SLO-adaptive behavior.

- **`scripts/repro_dry_window_activation.sh`** — Reproducibility script for a delayed-load dry-run experiment that highlights adaptive context-window transitions.

### 3. Plotting and Analysis

- **`scripts/plot_request_metrics.py`** — Plots latency, accuracy, and SLO attainment from one run.

- **`scripts/compare_experiments.py`** — Compares multiple experiment runs.

- **`scripts/compare_experiments_avg.py`** — Produces averaged comparisons across applications or load levels.

- **`scripts/plot_delayed_experiment_timeseries.py`** — Plots time-series metrics such as effective context window, requests per minute, tokens per minute, and accuracy.

- **`scripts/plot_latency_breakdown_ablation.py`** — Plots latency component breakdowns from timing logs.

- **`scripts/plot_window_factor_ablation.py`** — Plots how the final context window is composed from base window, load adaptation, and SLO offsets.

- **`scripts/plot_experiment_results.sh`** — Shell wrapper for regenerating saved plots.

### 4. Data Preparation

- **`scripts/extract_sharegpt.py`** — Cleans and extracts ShareGPT conversations into normalized datasets.

- **`scripts/preprocess.py`** — Builds warmup, evaluation, and application-partitioned datasets used by experiments.

### 5. Additional Topic-Drift Scripts

These scripts support an optional stretch-goal analysis and are secondary to the main adaptive cache contribution.

- **`scripts/drift_utils.py`** — Shared utilities for drift experiments.
- **`scripts/mark_drift.py`** — Labels conversations by drift level.
- **`scripts/build_candidate_pool.py`** — Builds retrieval candidate pools.
- **`scripts/drift_window_eval.py`** — Evaluates retrieval across different history windows.
- **`scripts/plot_window_metrics.py`** — Plots drift evaluation results.





















<!--
 ### Experiment and evaluation scripts

#### [`scripts/run_experiments.sh`](scripts/run_experiments.sh)

Orchestrates one or more **cache-mode experiments** against a locally started `gptcache_server`. For each run, `run_experiment` clears `CACHE_DIR`, builds a **temporary JSON** from [`config/request_gen.example.json`](config/request_gen.example.json) with `jq` (injecting `output_metrics_path`, `output_request_log_path`, and accuracy-baseline fields), starts the server in the background with `--server-mode` and any extra flags (for example `--slo-adaptive`), waits until the HTTP port responds, then invokes **`scripts/generate_requests.py`** with that config. After the run it stops the server, runs **`plot_request_metrics.py`** on the new metrics, and writes per-run plots under `data/test_apps/example/plots/<suffix>/` by default. A final step can call **`compare_experiments.py`** to merge multiple `request_metrics_*.json` files into comparison charts. **Environment:** `DRY_RUN`, `CACHE_DIR`, `PYTHON`, plus API credentials for the backend (see below). Individual baseline modes may be commented or uncommented in the script depending on which experiments you want to regenerate.

#### [`scripts/run_load_multiplier_experiments.sh`](scripts/run_load_multiplier_experiments.sh)

Same overall pattern as `run_experiments.sh`, but **sweeps `load_multiplier`** (for example 1 and 10) in the patched request-gen config, keeps artifacts under **`data/test_apps/load_mult_compare/`**, and uses **`--load-adaptive`** on the fourth mode instead of `--slo-adaptive`. **Environment:** `LOAD_ADAPTIVE_RATIO`, `EXAMPLE_CONFIG`, plus the same variables as the main experiment driver.

#### [`scripts/generate_requests.py`](scripts/generate_requests.py) — `main()` and `run(config)`

**`main()`** parses **`--config`**, loads the JSON, calls **`run(config)`**, and writes the returned metrics object to `output_metrics_path` when set, also printing JSON to stdout. **`run(config)`** is the core workload: it loads per-application conversation JSONL from `data_dir`, optionally runs a **warmup** phase that sends chat requests to the server to populate the cache, then spawns worker threads per **application** with configurable delays, jitter, concurrency, and optional time limits. It records per-request latency, cache hit/miss flags, accuracy metrics (token F1, sequence ratio, exact match) against gold or baseline logs, and **SLO attainment** against `application_slo_expectations` in the config. Aggregates are written into a single metrics structure (including `per_application` and overall throughput/latency percentiles).

#### Plotting and comparison

- **[`scripts/plot_request_metrics.py`](scripts/plot_request_metrics.py)** — Reads one **metrics JSON** produced by `generate_requests.py` and writes PNG charts (latency distributions, accuracy, SLO bars, etc.) under `--output-dir` with an optional filename `--prefix`. Use this to visualize a **single** experimental run.

- **[`scripts/compare_experiments.py`](scripts/compare_experiments.py)** — Loads **several** runs (via repeated `--run KEY:path/to/request_metrics.json[:optional_log.jsonl]` or legacy `--metrics-*` arguments), aligns them by application id, and emits **comparison** figures (for example latency and SLO attainment side-by-side). Request logs can be inferred from metrics filenames when the naming convention `request_metrics_<suffix>.json` / `request_log_<suffix>.jsonl` is followed.

- **[`scripts/plot_experiment_results.py`](scripts/plot_experiment_results.py)** — Convenience wrapper that regenerates **both** per-run plots (delegating to `plot_request_metrics.py`) and comparison plots (delegating to `compare_experiments.py`) from existing artifacts, without starting the server. Supports `--include` to filter which run keys appear on comparison charts and optional `--request-gen-config` for supplementary SLO/workload figures when the corresponding helper script is present.

#### Data preparation (optional)

- **[`scripts/extract_sharegpt.py`](scripts/extract_sharegpt.py)** — Filters and cleans raw ShareGPT-style dumps into normalized JSONL suitable for downstream splitting (language, turn counts, length limits).

- **[`scripts/preprocess.py`](scripts/preprocess.py)** — Splits cleaned JSONL into warmup/eval portions, **embeds** conversations, **clusters** them into synthetic “applications,” and writes **`application_*.jsonl`** plus manifests under `data/` for use with `generate_requests.py`.

### Small sample data under `test/`

The smallest runnable **application JSONL** bundle for quick experiments lives under **[`test/`](test/)**. 

| Path | Contents |
|------|----------|
| [`test/`](test/) | Five apps: `application_00.jsonl` … `application_04.jsonl`, warmup `mt10_warmup_5.jsonl`, `manifest.json`. Matches [`config/request_gen.test.json`](config/request_gen.test.json) (`data_dir`: `test/`, `warmup_file`: `mt10_warmup_5.jsonl`, `file_glob`: `application_*.jsonl`). |


[`config/request_gen.test.json`](config/request_gen.test.json) matches the `test/` layout above (five apps; experiment length and limits are set in that JSON).

Below is an **example** to run our code: run the server in one shell, then the request generator in another.

- **Server command:**

```bash
python3 -m gptcache_server.server --server-mode contextcache --slo-adaptive --load-adaptive
```

- **Requests command:**

```bash
python3 scripts/generate_requests.py --config config/request_gen.test.json
```


### API keys and backends

Request-gen JSON uses `api_key_env` (name of an environment variable holding the API key) and `base_url` for the chat backend. Export those variables in your shell before running scripts; do not commit real keys. Align `base_url` and model names with your provider.

---

## Source code map

The entries below focus on **new or substantially changed** pieces in this fork (baseline commit `a0fc5a9`, message `demo_update`). For shell-based orchestration and `generate_requests.py` behavior, see **Experiment and evaluation scripts** under Usage.

### [`gptcache/utils/adaptive_window.py`](gptcache/utils/adaptive_window.py)

#### `LoadAdaptiveMinuteWindow`

Maintains a **thread-safe sliding time window** (default 60 seconds) of `(timestamp, request_count, input_token_count)` events. **`record_request(input_tokens)`** appends an event for the current time and drops events older than the window. **`counts_in_window()`** returns the total number of requests and total input tokens still inside the window after pruning. This is the low-level signal used to approximate “current minute” load.

#### `LoadAdaptiveContextController`

Owns a `LoadAdaptiveMinuteWindow` and a **`Cache`** instance. **`record_request(input_tokens)`** forwards counts into the window; **once per window interval** (aligned with `window_seconds`), it compares the **current** window totals to **previous** minute snapshots (`_prev_req`, `_prev_tok`) and calls **`_maybe_resize_context_window`**. That helper uses `load_adaptive_ratio` (R) from config: if requests or tokens grew by at least **R×** versus the prior minute it treats load as **high** and tries to **shrink** the effective dialogue window by one step (down to 1); if both signals fell to at most **1/R×** it treats load as **low** and tries to **grow** by one step (up to `context_cache_window_max`). It implements shrink/grow by updating **`context_cache_overall_factor`** so that `effective_context_window_len(None)` matches the new integer length relative to the base `context_cache_window_len`. **`minute_counts()`** exposes the latest sliding-window totals for diagnostics.

---

### [`gptcache_server/server.py`](gptcache_server/server.py)

#### `main()`

Parses CLI arguments, initializes **`cache`** (and optionally a second **`openai_cache`**) via `init_similar_cache` / `init_similar_cache_from_config`, copies **`Config`** fields from flags (`context_cache_window_len`, `load_adaptive`, `load_adaptive_ratio`, `slo_adaptive`), sets globals **`server_mode`** and **`dry_run`**, and starts **uvicorn** on `--host` / `--port`. When `--openai` is set, wires the OpenAI-compatible chat route against the appropriate cache instance.

#### `on_application_registry_changed()`

Runs whenever the in-memory registry of application SLOs changes. If **`slo_adaptive`** is off, it sets **per-app deltas to 0** (neutral). If on, it reads **`_application_slos`** (latency and accuracy targets per server-issued application id), **normalizes** strictness scores, and computes a **bounded integer turn offset per app** (latency-tight apps get negative deltas for shorter context windows, accuracy-tight apps get positive deltas for longer ones). It writes the result into **`cache.config.context_cache_window_delta_by_app`** (and `openai_cache` when present).

#### `register_application_slo` (`POST /v1/applications`)

Validates **`latency_p99_ms`** (positive) and **`accuracy_slo`** in `[0, 1]`, assigns a new **`uuid4`** string as **`application_id`**, stores the record under **`_application_slos`**, and calls **`on_application_registry_changed()`**. Clients use the returned id when tagging traffic if the workload supports it.

#### `deregister_application` (`DELETE /v1/applications/{application_id}`)

Removes an entry from **`_application_slos`** and recomputes factors.

#### Legacy cache routes

**`hello`**, **`put_cache`**, **`get_cache`**, **`flush_cache`** expose the original simple put/get/flush API on the primary **`cache`** object.

---

### [`gptcache/config.py`](gptcache/config.py) — `Config` and `effective_context_window_len`

**`Config`** adds fields for **base** window length **`context_cache_window_len`**, **overall** multiplier **`context_cache_overall_factor`**, optional **per-application additive-offset** map **`context_cache_window_delta_by_app`**, cap **`context_cache_window_max`**, and booleans **`load_adaptive`**, **`load_adaptive_ratio`**, **`slo_adaptive`**. **`effective_context_window_len(application_id)`** computes `round(N * overall) + app_delta` with `app_delta` defaulting to 0 when the id is missing, then **clamps** the result to `[1, context_cache_window_max]`. Dialogue-window truncation in the cache path uses this value rather than raw `N`.

---

### [`gptcache/core.py`](gptcache/core.py) — `Cache`

#### `init(...)`

If **`config.load_adaptive`** is true, constructs **`LoadAdaptiveContextController(self)`** at init time so load statistics are ready before the first request.

#### `record_load_adaptive_request(input_tokens)`

No-op when `load_adaptive` is false; otherwise lazily creates **`LoadAdaptiveContextController`** if needed and forwards to **`record_request`**, updating minute load and possibly **`context_cache_overall_factor`**.

#### `load_adaptive_minute_stats()`

Returns **`(requests, tokens)`** for the current sliding minute, or **`None`** if load adaptation is disabled.

#### `import_data(...)`

For list-shaped “dialogue” questions, trims stored context embedding lists to **`effective_context_window_len(None)`** so bulk imports stay consistent with runtime behavior.

---

### Adapters and client

#### [`gptcache/adapter/adapter.py`](gptcache/adapter/adapter.py)

Where dialogue turns are turned into embedding sequences for context matching, the code uses **`effective_context_window_len(application_id)`** (when an application id is available) so **hit/miss decisions** respect the same dynamic window as storage and load adaptation.

#### [`gptcache/adapter/openai.py`](gptcache/adapter/openai.py)

After estimating **input token count** for an incoming chat request, if **`load_adaptive`** is enabled on the active cache it calls **`record_load_adaptive_request`** so the **LoadAdaptiveContextController** observes production traffic.

#### [`gptcache/adapter/api.py`](gptcache/adapter/api.py)

Small changes so **`init_similar_cache`** / config loading paths pass through the extended **`Config`** fields used by this fork.

#### [`gptcache/adapter/adapter_bac.py`](gptcache/adapter/adapter_bac.py)

Alternate or backup adapter implementations kept for **experiments and A/B** against the main **`adapt()`** path.

#### [`gptcache/client.py`](gptcache/client.py)

Adjustments to the lightweight HTTP client helpers so they remain compatible with **server modes**, headers, and response shapes used in evaluation.

---

### Embeddings and context preprocessing (delta)

#### [`gptcache/embedding/onnx.py`](gptcache/embedding/onnx.py) / [`gptcache/similarity_evaluation/onnx.py`](gptcache/similarity_evaluation/onnx.py)

Updates to **ONNX Runtime** embedding and evaluation paths (model loading, tensor layout, error handling) so semantic cache and similarity stages work reliably with the versions pinned in **`requirements.txt`**.

#### [`gptcache/processor/context/summarization_context.py`](gptcache/processor/context/summarization_context.py)

Tweaks to **summarization** helpers used when long turns are compressed before embedding, keeping behavior aligned with longer multi-turn traces in experiments.

---

### Additional automation scripts (reference)

#### [`scripts/run_adaptive_micro_eval.sh`](scripts/run_adaptive_micro_eval.sh)

Runs a **paired micro-benchmark**: same workload twice against a live server—**contextcache** without `--load-adaptive`, then **`adaptivecontextcache`** with **`--load-adaptive`**. Each arm typically includes a **low-load** phase and a **high-load** burst so **`LoadAdaptiveContextController`** can react. Outputs land under `data/test_apps/adaptive_micro_eval/` (configurable). Tunables include **`PORT`**, **`LOW_DURATION_S` / `HIGH_DURATION_S`**, **`LOW_MULTIPLIER` / `HIGH_MULTIPLIER`**, **`LOAD_ADAPTIVE_RATIO`**, **`CONTEXT_WINDOW_LEN`**, and **`DRY_RUN`**.

#### [`scripts/run_app_window_factor_micro_eval.sh`](scripts/run_app_window_factor_micro_eval.sh)

A heavier **async Python harness** (embedded heredoc) that compares **no-cache**, **contextcache without SLO adaptation**, and **contextcache with SLO adaptation**, using **asymmetric** synthetic workloads (**latency-sensitive** vs **accuracy-sensitive** templates). Optional **grid search** over base window, overall factor, and max window tests how **`context_cache_window_delta_by_app`** interacts with accuracy and latency; results can be written under a configurable **`RESULTS_DIR`**.

#### [`scripts/plot_experiment_results.sh`](scripts/plot_experiment_results.sh)

Thin shell wrapper around **`plot_experiment_results.py`** for the same arguments as in interactive use.

#### [`scripts/compare_experiments.py`](scripts/compare_experiments.py) — `parse_args`, `load_experiment_runs`, `write_comparison_plots`

**`load_experiment_runs`** resolves each **`--run`** into metrics plus optional request logs, loads JSON, and builds **`ExperimentRun`** rows. **`write_comparison_plots`** renders the cross-run **latency / accuracy / SLO** figures. Legacy flags map fixed filenames for older three-way comparisons.

#### [`scripts/plot_request_metrics.py`](scripts/plot_request_metrics.py)

Loads a single metrics JSON, extracts **`per_application`** statistics, and saves multiple **matplotlib** figures (latency percentiles, SLO attainment, etc.).

#### [`scripts/plot_experiment_results.py`](scripts/plot_experiment_results.py) — `main`

Subprocess-invokes **`plot_request_metrics.py`** once per run and **`compare_experiments.py`** once for the combined set; optionally invokes **`visualize_app_slos_and_workload.py`** when present.

#### [`scripts/preprocess.py`](scripts/preprocess.py) — `main` pipeline

Loads cleaned JSONL, performs **train/warmup/eval splits**, **embeds** text with sentence-transformers, **clusters** rows into **`num_applications`** buckets, and writes **`application_<id>.jsonl`** files plus **`manifest.json`** for the request generator.

#### [`scripts/extract_sharegpt.py`](scripts/extract_sharegpt.py) — CLI pipeline

Filters ShareGPT dumps (language, turns, length), normalizes fields, and emits **cleaned JSONL** for **`preprocess.py`**.

---

### Configuration examples

#### [`config/request_gen.example.json`](config/request_gen.example.json)

Full **multi-app** experiment template: **`data_dir`**, **`warmup_file`**, threading, **`load_multiplier`**, duration, per-app SLO targets, accuracy mode, and output paths for metrics and logs.

#### [`config/request_gen.live_test.json`](config/request_gen.live_test.json)

Short **smoke-test** profile: fewer apps, shorter **`experiment_duration_seconds`**, reduced concurrency—use with a **`data_dir`** that already contains **`application_*.jsonl`** (see Usage).

---

### Packaging and repo metadata

#### [`INSTALL.md`](INSTALL.md)

Step-by-step **OS packages**, **venv**, **`pip install`**, and sanity checks so `gptcache_server` and `scripts/` run in a consistent environment.

#### [`requirements.txt`](requirements.txt)

Pins and adds dependencies (**numpy**, **torch**, **transformers**, **FastAPI**, etc.) required by this branch.

#### [`.gitignore`](.gitignore)

Excludes local secrets and generated config patterns (for example **`config/*.local.json`**).
-->