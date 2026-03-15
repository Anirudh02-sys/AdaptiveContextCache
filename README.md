# ContextCache

ContextCache is a context-aware semantic caching system built on top of GPTCache-like primitives.
It is designed for multi-turn conversations where query meaning depends on dialogue history.

This repository currently exposes:

- a local cache API server (`/put`, `/get`, `/flush`, `/cache_file`),
- an optional OpenAI-compatible proxy route (`/v1/chat/completions`),
- context-aware matching logic in the adapter layer.

Demo video: <https://youtu.be/R3NByaQS7Ws>

---

## Table of Contents

- [What this project does](#what-this-project-does)
- [Repository structure](#repository-structure)
- [How request processing works](#how-request-processing-works)
- [Requirements](#requirements)
- [Install](#install)
- [Run the server](#run-the-server)
- [API usage](#api-usage)
- [OpenAI proxy mode](#openai-proxy-mode)
- [Python API usage](#python-api-usage)
- [Worker script for turn-by-turn conversations](#worker-script-for-turn-by-turn-conversations)
- [ShareGPT preprocessing for experiments](#sharegpt-preprocessing-for-experiments)
- [Multithreaded request generator](#multithreaded-request-generator)
- [Configuration details](#configuration-details)
- [Context input format](#context-input-format)
- [Observability and reporting](#observability-and-reporting)
- [Troubleshooting](#troubleshooting)
- [Known limitations of this branch](#known-limitations-of-this-branch)

---

## What this project does

At a high level:

1. Preprocess incoming prompt/messages.
2. Build embeddings and search top-k candidate cache items.
3. Filter by current-query similarity threshold.
4. Re-rank with dialogue context similarity (`mean` mode by default).
5. Return cached answer on hit, otherwise call miss path and store result.

The core orchestration lives in `gptcache/adapter/adapter.py` (`adapt()`).

---

## Repository structure

- `gptcache/`
  - `core.py` - `Cache` object initialization and lifecycle.
  - `config.py` - runtime config (thresholds, context state, flags).
  - `adapter/`
    - `adapter.py` - main hit/miss orchestration.
    - `api.py` - helper APIs (`init_similar_cache`, `put`, `get`).
    - `openai.py` - OpenAI wrapper (`ChatCompletion`, etc.).
  - `manager/` - scalar/vector/object storage + eviction.
  - `processor/` - pre/post/context processing.
  - `similarity_evaluation/` - ranking/evaluation strategies.
  - `report.py` - per-operation counters/timing.
- `gptcache_server/server.py` - FastAPI server and CLI entrypoint.
- `setup.py` - package metadata and `gptcache_server` console command.
- `requirements.txt` - base minimal dependencies.

---

## How request processing works

### 1) Preprocess
`pre_embedding_func` extracts a cache key source from request payload.  
Examples:
- `get_prompt` for simple `{"prompt": "..."}`
- `last_content` for chat messages

### 2) Embed + retrieve candidates
`embedding_func` builds vectors; `data_manager.search()` returns nearest candidates.

### 3) Score candidates
The branch uses:
- query similarity threshold (`similarity_threshold`)
- dialogue/context threshold (`dialuoge_threshold`)
- context combination mode (`method`, default `mean`)

### 4) Hit path
If a candidate passes thresholds:
- post-process selected answer,
- return cached response,
- update report counters.

### 5) Miss path
If no candidate qualifies:
- call `llm_handler` (or no-op handler depending on API path),
- store answer + embedding + context in cache.

---

## Requirements

- Python `>= 3.8.1` (`setup.py`)
- Linux/macOS recommended
- Network access for optional OpenAI mode

Notes:

- `requirements.txt` is intentionally minimal (`numpy`, `cachetools`, `requests`).
- Several optional packages are loaded lazily and can be auto-installed on first use (for example FastAPI/uvicorn/openai/onnxruntime/faiss).

---

## Install

From repo root:

```bash
cd ContextCache
python3 -m pip install -U pip
python3 -m pip install -e .
```

Alternative (package-only usage):

```bash
python3 -m pip install gptcache
```

---

## Run the server

### Basic cache server

```bash
python3 -m gptcache_server.server -s 127.0.0.1 -p 8011 -d /tmp/contextcache_data
```

Equivalent console entrypoint after install:

```bash
gptcache_server -s 127.0.0.1 -p 8011 -d /tmp/contextcache_data
```

### Health check

```bash
curl -s http://127.0.0.1:8011/
```

Expected:

```json
"hello gptcache server"
```

---

## API usage

### Insert (`/put`)

```bash
curl -s -X POST http://127.0.0.1:8011/put \
  -H "Content-Type: application/json" \
  -d '{"prompt":"hello","answer":"world"}'
```

Expected:

```json
"successfully update the cache"
```

### Retrieve (`/get`)

```bash
curl -s -X POST http://127.0.0.1:8011/get \
  -H "Content-Type: application/json" \
  -d '{"prompt":"hello"}'
```

Expected:

```json
{"prompt":"hello","answer":"world"}
```

### Flush

```bash
curl -s -X POST http://127.0.0.1:8011/flush
```

### Download cache files (if key set at startup)

```bash
curl -L "http://127.0.0.1:8011/cache_file?key=<your-key>" -o cache.zip
```

---

## OpenAI proxy mode

Enable OpenAI-compatible chat completion endpoint:

```bash
env -u APPIMAGE /usr/bin/python3 -m gptcache_server.server \
  -s 127.0.0.1 -p 8012 \
  -d /tmp/contextcache_data \
  -o True
```

### Server modes

The chat proxy supports a `--server-mode` runtime option:

- `contextcache` (default): current context-aware two-stage filtering.
- `gptcache`: query-only semantic cache behavior, no dialogue-history gate.
- `no-cache`: bypass cache entirely and forward every request to OpenAI.

ContextCache mode example:

```bash
env -u APPIMAGE /usr/bin/python3 -m gptcache_server.server \
  -s 127.0.0.1 -p 8012 \
  -d /tmp/contextcache_data \
  -o True \
  --server-mode contextcache
```

GPTCache mode example:

```bash
env -u APPIMAGE /usr/bin/python3 -m gptcache_server.server \
  -s 127.0.0.1 -p 8012 \
  -d /tmp/contextcache_data \
  -o True \
  --server-mode gptcache
```

No-cache mode example:

```bash
env -u APPIMAGE /usr/bin/python3 -m gptcache_server.server \
  -s 127.0.0.1 -p 8012 \
  -d /tmp/contextcache_data \
  -o True \
  --server-mode no-cache
```

Route exposed:

- `POST /v1/chat/completions`

Example request:

```bash
curl -s -X POST http://127.0.0.1:8012/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model":"gpt-3.5-turbo",
    "messages":[{"role":"user","content":"What is a relational database?"}],
    "temperature":0
  }'
```

Verify key is set before calling:

```bash
python3 - <<'PY'
import os
k = os.getenv("OPENAI_API_KEY")
print("OPENAI_API_KEY", "set" if k else "missing")
PY
```

Behavior:

- Cache hit: returns cached response.
- Cache miss: routed through OpenAI adapter path.
- You can force miss with `"cache_skip": true` or include `/cache_skip` marker in message content.

Important:

- This mode requires a valid OpenAI key and working OpenAI SDK environment.
- The server extracts key from `Authorization: Bearer ...`.

---

## Python API usage

Minimal script:

```python
from gptcache.adapter.api import init_similar_cache, put, get
from gptcache.processor.pre import get_prompt

init_similar_cache(data_dir="/tmp/api_cache", pre_func=get_prompt)
put("hello", "world")
print(get("hello"))  # world
```

---

## Worker script for turn-by-turn conversations

Use `scripts/conversation_cache_worker.py` to send a conversation turn-by-turn while carrying prior turns automatically.

### Mode 1: Cache API (`/put` + optional `/get`)

```bash
env -u APPIMAGE /usr/bin/python3 scripts/conversation_cache_worker.py \
  --mode cache-api \
  --base-url http://127.0.0.1:8011 \
  --verify-get
```

### Mode 2: OpenAI proxy backend

The worker supports both OpenAI-compatible proxy styles:

- `--proxy-api chat` -> `/v1/chat/completions`
- `--proxy-api completion` -> `/v1/completions`

Chat endpoint example:

```bash
env -u APPIMAGE /usr/bin/python3 scripts/conversation_cache_worker.py \
  --mode chat-proxy \
  --proxy-api chat \
  --base-url http://127.0.0.1:8012 \
  --model gpt-3.5-turbo \
  --api-key "$OPENAI_API_KEY"
```

Completion endpoint example:

```bash
env -u APPIMAGE /usr/bin/python3 scripts/conversation_cache_worker.py \
  --mode chat-proxy \
  --proxy-api completion \
  --base-url http://127.0.0.1:8012 \
  --model gpt-3.5-turbo \
  --api-key "$OPENAI_API_KEY"
```

Dry-run (print payloads without sending HTTP requests):

```bash
env -u APPIMAGE /usr/bin/python3 scripts/conversation_cache_worker.py \
  --mode chat-proxy \
  --proxy-api completion \
  --dry-run
```

Use a custom conversation file:

```bash
env -u APPIMAGE /usr/bin/python3 scripts/conversation_cache_worker.py \
  --conversation-file /path/to/conversation.json
```

Conversation JSON format:

```json
[
  {"user":"What is a relational database?","assistant":"..."},
  {"user":"What are its key features?","assistant":"..."}
]
```

---

## ShareGPT preprocessing for experiments

Use `scripts/preprocess_sharegpt.py` to download a ShareGPT-style dataset, cluster
conversations into applications, and annotate topic drift.

Default behavior:

- pulls `RyokoAI/ShareGPT52K` from Hugging Face,
- processes up to `1000` normalized conversations,
- groups into `--num-applications` clusters,
- writes `application_XX.jsonl` files and `manifest.json`.

Run for 10 applications:

```bash
python3 scripts/preprocess_sharegpt.py \
  --num-applications 10 \
  --output-dir data
```

Key output fields per conversation record:

- `application_id`
- `topic_label`
- `topic_drift_score`
- `topic_drift_bucket`
- `turns`
- `turn_pairs`

---

## Multithreaded request generator

Use `scripts/generate_requests.py` to replay application data with multiple threads
per application and different per-application request speeds.

Configs:

- `config/request_gen.example.json`
- `config/request_gen.live_test.json`

Generate synthetic test files for 10 apps:

```bash
python3 scripts/gen_test_app_data.py --num-apps 10 --out-dir data/test_apps
```

Start cache proxy first (required for non-dry-run requests):

```bash
env -u APPIMAGE /usr/bin/python3 -m gptcache_server.server \
  -s 127.0.0.1 -p 8012 \
  -d /tmp/contextcache_data \
  -o True
```

Quick smoke run (real HTTP requests, shorter duration):

```bash
python3 scripts/generate_requests.py --config config/request_gen.live_test.json
```

Experiment/template run (dry-run by default):

```bash
python3 scripts/generate_requests.py --config config/request_gen.example.json
```

Notes on run behavior:

- workers randomly sample conversations from each app dataset (sampling with replacement),
- when `experiment_duration_seconds > 0`, all apps/threads stop at the same global wall-clock end time,
- if `experiment_duration_seconds` is omitted or `<= 0`, workers fall back to draw-count mode based on per-thread shard size.

Config reference (`config/request_gen.example.json`):

- **Endpoint/model**
  - `base_url`, `endpoint`, `model`, `api_key_env`
- **Input data selection**
  - `data_dir`, `file_glob`, `applications`
- **Concurrency**
  - `default_threads_per_app`, `threads_per_application`, `max_workers`
- **Rate/speed control**
  - `default_delay_ms`, `app_delay_ms`, `thread_jitter_ms`, `load_multiplier`
- **Experiment window**
  - `experiment_duration_seconds` (all workers stop together when duration expires)
- **Replay limits**
  - `max_conversations_per_app`, `max_turns_per_conversation`
- **Reliability**
  - `retry_count`, `retry_backoff_ms`, `timeout_seconds`
- **Evaluation and SLO**
  - `accuracy_enabled`
  - `accuracy_slo_metric` (`exact_match_rate`, `avg_token_f1`, or `avg_sequence_ratio`)
  - `application_slo_expectations` per app:
    - `latency_p99_ms`
    - `accuracy_slo`
- **Outputs**
  - `output_metrics_path`, `output_request_log_path`

SLO reporting behavior:

- per app, the runner reports `slo_target`, `slo_observed`, and `slo_attainment`,
- latency SLO uses observed `served_p99_latency_ms` (or `p99_latency_ms` if no successful responses),
- accuracy SLO uses the metric chosen by `accuracy_slo_metric`,
- top-level `slo_summary` includes total apps with targets, passed apps, and missed apps.

Load multiplier behavior:

- `load_multiplier` scales request pacing globally for all applications and threads,
- `1.0` keeps configured delays unchanged,
- values `> 1.0` increase load (faster request generation),
- values between `0` and `1.0` decrease load (slower request generation),
- example: `load_multiplier=2.0` makes effective inter-request sleep roughly half.

Notes:

- thread counts are controlled by `default_threads_per_app` and `threads_per_application`,
- per-app pacing is controlled by `default_delay_ms` and `app_delay_ms`,
- app conversation selection is random with replacement per worker,
- each turn includes prior turns from the same conversation only (history resets on next conversation),
- metrics and optional per-request logs are written to configured output paths,
- metrics include latency percentiles, cache hits/misses (when returned by proxy), and response-accuracy scores against expected dataset answers (`exact_match_rate`, `avg_token_f1`, `avg_sequence_ratio`),
- accuracy scoring can be toggled with `accuracy_enabled` in config.

---

## Configuration details

Two common initialization styles:

1. Programmatic via `init_similar_cache(...)`
2. YAML via `init_similar_cache_from_config(...)` and `--cache-config-file`

Key config fields in `Config` (`gptcache/config.py`):

- `similarity_threshold` (default `0.75`) - current query similarity gate.
- `dialuoge_threshold` (default `0.6`) - context similarity gate.
- `method` (default `"mean"`) - context ranking method.
- `auto_flush` - flush period for cache persistence.
- `enable_token_counter`, `input_summary_len`, `data_check`, `disable_report`.

Server CLI flags (`gptcache_server/server.py`):

- `-s, --host`
- `-p, --port`
- `-d, --cache-dir`
- `-k, --cache-file-key`
- `-f, --cache-config-file`
- `-o, --openai`
- `-of, --openai-cache-config-file`

---

## Context input format

This branch supports two styles at preprocessing time:

- simple prompt (`prompt`)
- chat message content

For context-aware multi-turn matching, the preprocessors can return either:

- a single value, or
- a tuple `(current_query, context_history)`

The adapter now handles both return shapes safely.

### Important request-shape note for `/v1/chat/completions` on this branch

Current `adapt()` logic expects `messages[-1]["content"]` to be list-like in some paths.
If you pass a plain string, prompt handling can degrade (for example only the last character may be treated as the current question).

Recommended payload for this branch:

```json
{
  "messages": [
    {
      "role": "user",
      "content": ["Your current question here"]
    }
  ]
}
```

Example with context:

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        "Previous user question: ...",
        "Previous model response: ...",
        "Current question: ..."
      ]
    }
  ]
}
```

---

## Observability and reporting

`gptcache/report.py` tracks timing/counts for:

- pre-process
- embedding
- search
- data fetch
- evaluation
- post-process
- llm
- save
- cache hit count (`hint_cache_count`)

You can integrate custom logging with `Config(log_time_func=...)`.

---

## Troubleshooting

### Server starts but `/put` or `/get` fails

- Ensure you are using this updated branch state.
- Check server traceback output for adapter/runtime errors.
- Verify writable cache directory path (`-d`).

### OpenAI proxy returns 500

- Confirm `OPENAI_API_KEY`/Authorization header is valid.
- Verify network access from runtime environment.
- Test with a minimal non-stream request first.

### Response is irrelevant and cache hit is `true`

Symptom example:

- Response content does not match current prompt.
- Return tuple includes hit flag `true`.
- Response object shows `"gptcache": true`.

Cause:

- A prior malformed request shape can cache an irrelevant answer.
- Later requests may semantically match that stale entry and return it as a cache hit.

Immediate recovery options:

1. Force miss for verification:

```bash
curl -s -X POST http://127.0.0.1:8012/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model":"gpt-3.5-turbo",
    "cache_skip": true,
    "messages":[{"role":"user","content":["Say no in 2 words."]}],
    "temperature":0
  }'
```

2. Clear cache data directory and restart server:

```bash
rm -rf /tmp/contextcache_data
mkdir -p /tmp/contextcache_data
env -u APPIMAGE /usr/bin/python3 -m gptcache_server.server -s 127.0.0.1 -p 8012 -d /tmp/contextcache_data -o True
```

Note: `/flush` persists cache to storage; it does not delete entries.

### OpenAI error mentions `Cursor-...AppImage` path

Symptom example:

`openai error: [Errno 2] No such file or directory: '/.../Cursor-2.4.37-x86_64.AppImage'`

Cause:

- In some Cursor-launched shells, `APPIMAGE` can make Python report an invalid `sys.executable`.
- OpenAI Python SDK (`0.28.x`) reads platform metadata and fails when that executable path does not exist.

Checks:

```bash
env | grep -E "APPIMAGE|SSL_CERT_FILE|REQUESTS_CA_BUNDLE|CURL_CA_BUNDLE|OPENAI_API_URL|OPENAI_API_BASE|HTTPS_PROXY|HTTP_PROXY"
python3 -c 'import sys; print(sys.executable)'
```

Workaround for run commands:

```bash
env -u APPIMAGE /usr/bin/python3 -m gptcache_server.server -s 127.0.0.1 -p 8012 -d /tmp/contextcache_data -o True
```

Optional shell-level mitigation:

```bash
unset APPIMAGE
```

### Dependency errors

- Run `python3 -m pip install -e .` first.
- If optional modules still fail, install explicitly (for example `fastapi`, `uvicorn`, `openai`, `onnxruntime`, `faiss-cpu`).

---

## Known limitations of this branch

- `ContextMatchEvaluation` in `gptcache/similarity_evaluation/context_match.py` contains a hardcoded local checkpoint path (`/data/home/.../checkpoint_epoch_20.pth`). It will fail unless you provide/edit that model path.
- There are multiple pathways (sync/async, API/server/OpenAI) inherited from GPTCache plus custom branch logic; behavior may differ by entrypoint.

---

## License / upstream note

This repository is based on GPTCache-style architecture (`setup.py` still references GPTCache metadata). Treat this branch as a customized ContextCache variant and validate behavior in your target deployment path.

