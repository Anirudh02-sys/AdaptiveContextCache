#!/usr/bin/env python3
"""
add optional cache prewarm that populates our cache before experiment begins
prewarm uses the same request format as experiment

Multithreaded request generator for application-grouped conversation files.

Reads `application_*.jsonl`, spawns multiple threads per application, and sends
requests to `/v1/chat/completions` while preserving history only within each
conversation.

`load_multiplier` scales **how many threads** run per application (config
`threads_per_application` × `load_multiplier`, at least one thread per app). Per-thread
pacing uses the configured delays unchanged; it does **not** divide sleep by LM.
Compare against a nocache baseline only when the baseline was produced with the same
effective thread layout (same LM), or accuracy keys may not line up.

`load_multiplier_start_seconds` is kept in the config for compatibility but no longer
changes pacing (thread count is fixed at process start from `load_multiplier`).
"""

from __future__ import annotations

import argparse
import difflib
import glob
import json
import os
import random
import re
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import median
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multithreaded application-aware request generation."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to request generator JSON config.",
    )
    return parser.parse_args()


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _load_app_records(config: Dict[str, Any]) -> Dict[int, List[Dict[str, Any]]]:
    data_dir = config["data_dir"]
    file_glob = config.get("file_glob", "application_*.jsonl")
    applications = set(config.get("applications", []))
    paths = sorted(glob.glob(os.path.join(data_dir, file_glob)))
    app_records: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

    for path in paths:
        fname = os.path.basename(path)
        app_id = None
        if fname.startswith("application_") and fname.endswith(".jsonl"):
            try:
                app_id = int(fname[len("application_") : -len(".jsonl")])
            except ValueError:
                app_id = None
        if app_id is None:
            continue
        if applications and app_id not in applications:
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                app_records[app_id].append(json.loads(line))
    return app_records

def _load_warmup_records(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    data_dir = config["data_dir"]
    warmup_file = config.get("warmup_file")
    if not warmup_file:
        raise ValueError("config must specify warmup_file")

    path = os.path.join(data_dir, warmup_file)
    if not os.path.exists(path):
        raise FileNotFoundError(f"warmup file not found at: {path}")
     
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _pairs_from_record(record: Dict[str, Any]) -> List[Tuple[str, str]]:
    if isinstance(record.get("turn_pairs"), list) and record["turn_pairs"]:
        pairs = []
        for p in record["turn_pairs"]:
            if not isinstance(p, dict):
                continue
            user = p.get("user", "")
            assistant = p.get("assistant", "")
            if isinstance(user, str) and user.strip():
                pairs.append((user.strip(), assistant.strip() if isinstance(assistant, str) else ""))
        if pairs:
            return pairs

    turns = record.get("turns", [])
    if not isinstance(turns, list):
        return []
    pairs = []
    pending_user: Optional[str] = None
    for t in turns:
        if not isinstance(t, dict):
            continue
        role = str(t.get("role", "")).strip().lower()
        text = t.get("text", "")
        if not isinstance(text, str):
            continue
        text = text.strip()
        if not text:
            continue
        if role == "user":
            pending_user = text
        elif role == "assistant" and pending_user is not None:
            pairs.append((pending_user, text))
            pending_user = None
    return pairs


def _extract_response_text(resp_json: Any) -> str:
    if isinstance(resp_json, dict):
        choices = resp_json.get("choices")
        if isinstance(choices, list) and choices:
            c0 = choices[0]
            if isinstance(c0, dict):
                message = c0.get("message", {})
                if isinstance(message, dict):
                    content = message.get("content", "")
                    if isinstance(content, str):
                        return content
                    return str(content)
    if isinstance(resp_json, list) and resp_json:
        return _extract_response_text(resp_json[0])
    if isinstance(resp_json, str):
        return resp_json
    return str(resp_json)


def _extract_response_details(resp_json: Any) -> Tuple[str, Optional[bool]]:
    # Expected branch shape on this repo can be:
    # [answer_payload, is_hit, query_flags, context_flags]
    if isinstance(resp_json, list) and resp_json:
        text = _extract_response_text(resp_json)
        is_hit: Optional[bool] = None
        if len(resp_json) > 1 and isinstance(resp_json[1], bool):
            is_hit = resp_json[1]
        return text, is_hit

    if isinstance(resp_json, dict):
        text = _extract_response_text(resp_json)
        is_hit = resp_json.get("is_hit")
        if isinstance(is_hit, bool):
            return text, is_hit
        return text, None

    return _extract_response_text(resp_json), None


def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _tokenize_for_f1(text: str) -> List[str]:
    normalized = _normalize_text(text)
    if not normalized:
        return []
    word_tokens = re.findall(r"\w+", normalized, flags=re.UNICODE)
    if word_tokens:
        return word_tokens
    return list(normalized.replace(" ", ""))


def _token_f1(pred: str, gold: str) -> float:
    p_tokens = _tokenize_for_f1(pred)
    g_tokens = _tokenize_for_f1(gold)
    if not p_tokens or not g_tokens:
        return 0.0
    p_counts: Dict[str, int] = defaultdict(int)
    g_counts: Dict[str, int] = defaultdict(int)
    for t in p_tokens:
        p_counts[t] += 1
    for t in g_tokens:
        g_counts[t] += 1
    overlap = 0
    for t, c in p_counts.items():
        overlap += min(c, g_counts.get(t, 0))
    if overlap == 0:
        return 0.0
    precision = overlap / len(p_tokens)
    recall = overlap / len(g_tokens)
    return (2 * precision * recall) / (precision + recall)


def _accuracy_baseline_key(
    app_id: int, thread_idx: int, conversation_id: str, turn_index: int
) -> Tuple[int, int, str, int]:
    return (int(app_id), int(thread_idx), str(conversation_id), int(turn_index))


def _load_accuracy_baseline_from_log(path: str) -> Dict[Tuple[int, int, str, int], str]:
    """Map (application_id, thread_id, conversation_id, turn_index) -> response text from a prior run."""
    out: Dict[Tuple[int, int, str, int], str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = _accuracy_baseline_key(
                int(row["application_id"]),
                int(row["thread_id"]),
                str(row["conversation_id"]),
                int(row["turn_index"]),
            )
            text = row.get("response_text")
            if text is None:
                text = row.get("response_snippet", "")
            out[key] = str(text) if text is not None else ""
    return out


def _perfect_accuracy_metrics() -> Dict[str, float]:
    return {"exact_match": 1.0, "token_f1": 1.0, "sequence_ratio": 1.0}


def _accuracy_metrics(predicted: str, expected: str) -> Dict[str, float]:
    p = predicted or ""
    e = expected or ""
    exact = 1.0 if _normalize_text(p) == _normalize_text(e) and _normalize_text(e) else 0.0
    token_f1 = _token_f1(p, e)
    ratio = difflib.SequenceMatcher(a=_normalize_text(p), b=_normalize_text(e)).ratio()
    return {
        "exact_match": exact,
        "token_f1": float(token_f1),
        "sequence_ratio": float(ratio),
    }


def _is_placeholder_response(text: str) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False
    return "[dry-run]" in normalized or normalized == "dry-run-response"


def _percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    idx = int(round((p / 100.0) * (len(sorted_vals) - 1)))
    idx = max(0, min(idx, len(sorted_vals) - 1))
    return float(sorted_vals[idx])


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _scaled_thread_count(base_threads: int, load_multiplier: float) -> int:
    """Per-app thread count = base_threads × load_multiplier (minimum 1)."""
    lm = float(load_multiplier)
    if lm <= 0:
        lm = 1.0
    scaled = int(round(float(base_threads) * lm))
    return max(1, scaled)


class RunState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.started_at = time.time()
        self.request_count = 0
        self.success_count = 0
        self.error_count = 0
        self.cache_hit_count = 0
        self.cache_miss_count = 0
        self.placeholder_response_count = 0
        self.latencies_ms: List[float] = []
        self.served_latencies_ms: List[float] = []
        self.accuracy_count = 0
        self.exact_match_count = 0
        self.accuracy_token_f1_sum = 0.0
        self.accuracy_sequence_ratio_sum = 0.0
        self.per_app: Dict[int, Dict[str, Any]] = defaultdict(
            lambda: {
                "requests": 0,
                "success": 0,
                "errors": 0,
                "cache_hits": 0,
                "cache_misses": 0,
                "placeholder_responses": 0,
                "latencies_ms": [],
                "served_latencies_ms": [],
                "accuracy_count": 0,
                "exact_match_count": 0,
                "accuracy_token_f1_sum": 0.0,
                "accuracy_sequence_ratio_sum": 0.0,
            }
        )

    def add_result(
        self,
        app_id: int,
        ok: bool,
        latency_ms: float,
        is_cache_hit: Optional[bool] = None,
        is_placeholder_response: bool = False,
        accuracy: Optional[Dict[str, float]] = None,
    ) -> None:
        with self.lock:
            self.request_count += 1
            self.per_app[app_id]["requests"] += 1
            if ok:
                self.success_count += 1
                self.per_app[app_id]["success"] += 1
                self.served_latencies_ms.append(latency_ms)
                self.per_app[app_id]["served_latencies_ms"].append(latency_ms)
            else:
                self.error_count += 1
                self.per_app[app_id]["errors"] += 1
            self.latencies_ms.append(latency_ms)
            self.per_app[app_id]["latencies_ms"].append(latency_ms)
            if is_cache_hit is True:
                self.cache_hit_count += 1
                self.per_app[app_id]["cache_hits"] += 1
            elif is_cache_hit is False:
                self.cache_miss_count += 1
                self.per_app[app_id]["cache_misses"] += 1
            if is_placeholder_response:
                self.placeholder_response_count += 1
                self.per_app[app_id]["placeholder_responses"] += 1
            if accuracy:
                self.accuracy_count += 1
                self.per_app[app_id]["accuracy_count"] += 1
                self.accuracy_token_f1_sum += float(accuracy["token_f1"])
                self.accuracy_sequence_ratio_sum += float(accuracy["sequence_ratio"])
                self.per_app[app_id]["accuracy_token_f1_sum"] += float(accuracy["token_f1"])
                self.per_app[app_id]["accuracy_sequence_ratio_sum"] += float(accuracy["sequence_ratio"])
                if float(accuracy["exact_match"]) >= 1.0:
                    self.exact_match_count += 1
                    self.per_app[app_id]["exact_match_count"] += 1


def _request_once(
    session: requests.Session,
    base_url: str,
    endpoint: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    timeout_s: float,
    dry_run: bool,
) -> Tuple[bool, str, float, int, Optional[bool]]:
    start = time.time()
    if dry_run:
        time.sleep(0.01)
        latency_ms = (time.time() - start) * 1000.0
        return True, "dry-run-response", latency_ms, 200, None

    url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
    resp = session.post(url, headers=headers, json=payload, timeout=timeout_s)
    latency_ms = (time.time() - start) * 1000.0
    if resp.status_code >= 400:
        return False, resp.text[:500], latency_ms, resp.status_code, None
    response_text, is_hit = _extract_response_details(resp.json())
    return True, response_text, latency_ms, resp.status_code, is_hit


def _thread_worker(
    app_id: int,
    thread_idx: int,
    conversations: Sequence[Dict[str, Any]],
    conversation_draw_count: int,
    experiment_end_time: Optional[float],
    config: Dict[str, Any],
    state: RunState,
    log_path: str,
    log_lock: threading.Lock,
) -> None:
    default_delay_ms = int(config.get("default_delay_ms", 150))
    app_delay_ms = int(config.get("app_delay_ms", {}).get(str(app_id), default_delay_ms))
    jitter_ms = int(config.get("thread_jitter_ms", 40))
    retry_count = int(config.get("retry_count", 1))
    retry_backoff_ms = int(config.get("retry_backoff_ms", 200))
    timeout_s = float(config.get("timeout_seconds", 30))
    dry_run = bool(config.get("dry_run", False))
    endpoint = str(config.get("endpoint", "/v1/chat/completions"))
    base_url = str(config.get("base_url", "http://127.0.0.1:8012"))
    model = str(config.get("model", "gpt-3.5-turbo"))
    max_turns = int(config.get("max_turns_per_conversation", 0))
    accuracy_enabled = bool(config.get("accuracy_enabled", True))

    api_key_env = str(config.get("api_key_env", "OPENAI_API_KEY"))
    api_key = os.getenv(api_key_env, "")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    thread_rng = random.Random((app_id + 1) * 1000 + thread_idx)
    session = requests.Session()

    if not conversations:
        return

    draw_count = max(0, int(conversation_draw_count))
    remaining_draws = draw_count
    while True:
        if experiment_end_time is not None:
            if time.time() >= experiment_end_time:
                break
        else:
            if remaining_draws <= 0:
                break
            remaining_draws -= 1

        conversation = conversations[thread_rng.randrange(len(conversations))]
        pairs = _pairs_from_record(conversation)
        if max_turns > 0:
            pairs = pairs[:max_turns]
        if not pairs:
            continue

        history: List[str] = []
        conversation_id = str(conversation.get("conversation_id", "unknown"))
        for turn_idx, (user_prompt, expected_answer) in enumerate(pairs, start=1):
            if experiment_end_time is not None and time.time() >= experiment_end_time:
                break

            content_list = list(history) + [user_prompt]
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        # Adapter-side preprocessing (`gptcache/processor/pre.py::last_content`)
                        # treats list-like content as: [history..., current_user_prompt].
                        # This enables true two-stage retrieval: stage-1 uses only the
                        # current prompt, stage-2 uses extracted `context_res`.
                        "content": content_list,
                    }
                ],
                "temperature": 0,
                "max_tokens": 50,
            }
            app_server_id_map = config.get("_app_id_to_server_id", {})
            if app_id in app_server_id_map:
                payload["application_id"] = app_server_id_map[app_id]

            ok = False
            response_text = ""
            latency_ms = 0.0
            status_code = 0
            last_err = ""
            is_cache_hit: Optional[bool] = None
            is_placeholder_response = False

            for attempt in range(retry_count + 1):
                attempt_start = time.time()
                try:
                    ok, response_text, latency_ms, status_code, is_cache_hit = _request_once(
                        session=session,
                        base_url=base_url,
                        endpoint=endpoint,
                        headers=headers,
                        payload=payload,
                        timeout_s=timeout_s,
                        dry_run=dry_run,
                    )
                    if ok:
                        is_placeholder_response = _is_placeholder_response(response_text)
                        break
                    last_err = response_text
                except Exception as exc:
                    last_err = str(exc)
                    ok = False
                    latency_ms = (time.time() - attempt_start) * 1000.0
                    status_code = 0
                if attempt < retry_count:
                    backoff = (retry_backoff_ms / 1000.0) * (attempt + 1)
                    time.sleep(backoff)

            accuracy = None
            if ok and not is_placeholder_response and accuracy_enabled:
                if bool(config.get("accuracy_baseline_reference_run", False)):
                    accuracy = _perfect_accuracy_metrics()
                else:
                    baseline_map = config.get("_accuracy_baseline_map")
                    if baseline_map is not None:
                        bkey = _accuracy_baseline_key(
                            app_id, thread_idx, conversation_id, turn_idx
                        )
                        gold = baseline_map.get(bkey, "")
                        if isinstance(gold, str) and gold.strip():
                            accuracy = _accuracy_metrics(response_text, gold)
                    elif isinstance(expected_answer, str) and expected_answer.strip():
                        accuracy = _accuracy_metrics(response_text, expected_answer)

            state.add_result(
                app_id=app_id,
                ok=ok,
                latency_ms=latency_ms,
                is_cache_hit=is_cache_hit,
                is_placeholder_response=is_placeholder_response,
                accuracy=accuracy,
            )
            used_answer = response_text if ok else f"request_failed: {last_err}"

            # Preserve context only within this conversation.
            history.append(f"Previous user question: {user_prompt}")
            history.append(f"Previous model response: {used_answer}")

            if log_path:
                entry = {
                    "timestamp": time.time(),
                    "application_id": app_id,
                    "thread_id": thread_idx,
                    "conversation_id": conversation_id,
                    "turn_index": turn_idx,
                    "ok": ok,
                    "status_code": status_code,
                    "latency_ms": round(latency_ms, 3),
                    "cache_hit": is_cache_hit,
                    "placeholder_response": is_placeholder_response,
                    "request_prompt": user_prompt,
                    "expected_response": expected_answer[:240] if isinstance(expected_answer, str) else "",
                    "response_snippet": used_answer[:240],
                    "response_text": used_answer,
                    "request_content_count": len(content_list),
                }
                if accuracy:
                    entry["accuracy"] = {
                        "exact_match": round(float(accuracy["exact_match"]), 6),
                        "token_f1": round(float(accuracy["token_f1"]), 6),
                        "sequence_ratio": round(float(accuracy["sequence_ratio"]), 6),
                    }
                with log_lock:
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            # Inter-request pacing (load_multiplier does not shorten sleep; it scales
            # thread count in `run()` instead).
            raw_delay_s = (app_delay_ms + thread_rng.randint(0, max(0, jitter_ms))) / 1000.0
            delay_s = raw_delay_s
            if experiment_end_time is not None:
                remaining_s = experiment_end_time - time.time()
                if remaining_s <= 0:
                    break
                time.sleep(min(delay_s, remaining_s))
            else:
                time.sleep(delay_s)


def _split_for_threads(
    conversations: Sequence[Dict[str, Any]],
    thread_count: int,
) -> List[List[Dict[str, Any]]]:
    buckets: List[List[Dict[str, Any]]] = [[] for _ in range(max(1, thread_count))]
    for idx, conv in enumerate(conversations):
        buckets[idx % len(buckets)].append(conv)
    return buckets

def _run_warmup_conversation(
    conversation: Dict[str, Any],
    config: Dict[str, Any],
    session: requests.Session,
    headers: Dict[str, str],
    warmup_log_path: str,
    log_lock: threading.Lock,
) -> Dict[str, int]:
    retry_count = int(config.get("retry_count", 1))
    retry_backoff_ms = int(config.get("retry_backoff_ms", 200))
    timeout_s = float(config.get("timeout_seconds", 30))
    dry_run = bool(config.get("dry_run", False))
    endpoint = str(config.get("endpoint", "/v1/chat/completions"))
    base_url = str(config.get("base_url", "http://127.0.0.1:8012"))
    model = str(config.get("model", "gpt-3.5-turbo"))
    max_turns = int(config.get("max_turns_per_conversation", 0))

    pairs = _pairs_from_record(conversation)
    if max_turns > 0:
        pairs = pairs[:max_turns]

    if not pairs:
        return {
            "requests_sent": 0,
            "success": 0,
            "errors": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

    history: List[str] = []
    conversation_id = str(conversation.get("conversation_id", "unknown"))

    request_count = 0
    success_count = 0
    error_count = 0
    cache_hit_count = 0
    cache_miss_count = 0

    for turn_idx, (user_prompt, expected_answer) in enumerate(pairs, start=1):
        content_list = list(history) + [user_prompt]
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    # Same shape as `_thread_worker`: list content so `last_content` can
                    # split current query vs `context_res` (contextcache / gptcache baselines).
                    "content": content_list,
                }
            ],
            "temperature": 0,
            "max_tokens": 50,
        }

        ok = False
        response_text = ""
        latency_ms = 0.0
        status_code = 0
        last_err = ""
        is_cache_hit: Optional[bool] = None

        for attempt in range(retry_count + 1):
            attempt_start = time.time()
            try:
                ok, response_text, latency_ms, status_code, is_cache_hit = _request_once(
                    session=session,
                    base_url=base_url,
                    endpoint=endpoint,
                    headers=headers,
                    payload=payload,
                    timeout_s=timeout_s,
                    dry_run=dry_run,
                )
                if ok:
                    break
                last_err = response_text
            except Exception as exc:
                last_err = str(exc)
                ok = False
                latency_ms = (time.time() - attempt_start) * 1000.0
                status_code = 0

            if attempt < retry_count:
                backoff = (retry_backoff_ms / 1000.0) * (attempt + 1)
                time.sleep(backoff)

        request_count += 1
        if ok:
            success_count += 1
        else:
            error_count += 1

        if is_cache_hit is True:
            cache_hit_count += 1
        elif is_cache_hit is False:
            cache_miss_count += 1

        used_answer = response_text if ok else f"request_failed: {last_err}"

        history.append(f"Previous user question: {user_prompt}")
        history.append(f"Previous model response: {used_answer}")

        if warmup_log_path:
            entry = {
                "entry_type": "warmup_request",
                "timestamp": time.time(),
                "conversation_id": conversation_id,
                "turn_index": turn_idx,
                "ok": ok,
                "status_code": status_code,
                "latency_ms": round(latency_ms, 3),
                "cache_hit": is_cache_hit,
                "request_prompt": user_prompt,
                "expected_response": expected_answer[:240] if isinstance(expected_answer, str) else "",
                "response_snippet": used_answer[:240],
                "request_content_count": len(content_list),
            }
            with log_lock:
                with open(warmup_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return {
        "requests_sent": request_count,
        "success": success_count,
        "errors": error_count,
        "cache_hits": cache_hit_count,
        "cache_misses": cache_miss_count,
    }


def _run_warmup_phase(
    warmup_records: Sequence[Dict[str, Any]],
    config: Dict[str, Any],
    warmup_log_path: str,
) -> Dict[str, Any]:
    if not warmup_records:
        print("[warmup] no conversations to run; skipping warmup phase.", flush=True)
        return {
            "entry_type": "warmup_summary",
            "conversations_seen": 0,
            "requests_sent": 0,
            "success": 0,
            "errors": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "elapsed_seconds": 0.0,
        }

    api_key_env = str(config.get("api_key_env", "OPENAI_API_KEY"))
    api_key = os.getenv(api_key_env, "")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    session = requests.Session()
    log_lock = threading.Lock()

    started_at = time.time()
    request_count = 0
    success_count = 0
    error_count = 0
    cache_hit_count = 0
    cache_miss_count = 0

    n_warmup = len(warmup_records)
    # Frequent lines for small runs; ~10 milestones for large runs.
    progress_step = max(1, n_warmup // 10)
    print(
        f"[warmup] sending requests for {n_warmup} conversation(s)...",
        flush=True,
    )

    for i, conversation in enumerate(warmup_records, start=1):
        conv_id = conversation.get("conversation_id", "?")
        conv_result = _run_warmup_conversation(
            conversation=conversation,
            config=config,
            session=session,
            headers=headers,
            warmup_log_path=warmup_log_path,
            log_lock=log_lock,
        )
        request_count += int(conv_result["requests_sent"])
        success_count += int(conv_result["success"])
        error_count += int(conv_result["errors"])
        cache_hit_count += int(conv_result["cache_hits"])
        cache_miss_count += int(conv_result["cache_misses"])

        if (
            n_warmup <= 25
            or i == 1
            or i == n_warmup
            or i % progress_step == 0
        ):
            elapsed = time.time() - started_at
            print(
                f"[warmup] progress {i}/{n_warmup} "
                f"conversation_id={conv_id!r} "
                f"(+{conv_result['requests_sent']} req this conv; "
                f"cumulative req={request_count} ok={success_count} err={error_count}; "
                f"{elapsed:.1f}s elapsed)",
                flush=True,
            )

    elapsed_s = max(0.001, time.time() - started_at)

    summary = {
        "entry_type": "warmup_summary",
        "conversations_seen": len(warmup_records),
        "requests_sent": request_count,
        "success": success_count,
        "errors": error_count,
        "cache_hits": cache_hit_count,
        "cache_misses": cache_miss_count,
        "elapsed_seconds": round(elapsed_s, 3),
    }

    if warmup_log_path:
        with log_lock:
            with open(warmup_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(summary, ensure_ascii=False) + "\n")

    print(
        f"[warmup] finished: {request_count} requests, "
        f"{success_count} success, {error_count} errors, "
        f"cache_hits={cache_hit_count} cache_misses={cache_miss_count}, "
        f"{elapsed_s:.1f}s total",
        flush=True,
    )

    return summary

def _register_applications_with_server(
    config: Dict[str, Any],
) -> Dict[int, str]:
    """Register application SLO targets with the cache server via POST /v1/applications.

    Returns a mapping of integer app_id -> server-issued UUID string.
    If no SLO expectations are configured, returns an empty dict.
    """
    slo_expectations = config.get("application_slo_expectations", {})
    if not slo_expectations:
        return {}

    base_url = str(config.get("base_url", "http://127.0.0.1:8012")).rstrip("/")
    url = f"{base_url}/v1/applications"
    mapping: Dict[int, str] = {}

    for app_id_str, slo in sorted(slo_expectations.items()):
        app_id = int(app_id_str)
        latency_target = slo.get("latency_p99_ms")
        accuracy_target = slo.get("accuracy_slo")
        if latency_target is None or accuracy_target is None:
            continue
        body = {
            "latency_p99_ms": float(latency_target),
            "accuracy_slo": float(accuracy_target),
        }
        try:
            resp = requests.post(url, json=body, timeout=10)
            resp.raise_for_status()
            server_app_id = resp.json().get("application_id", "")
            if server_app_id:
                mapping[app_id] = str(server_app_id)
                print(
                    f"[slo-register] app {app_id} -> server_id={server_app_id} "
                    f"(latency_p99_ms={latency_target}, accuracy_slo={accuracy_target})",
                    flush=True,
                )
        except Exception as exc:
            print(
                f"[slo-register] WARNING: failed to register app {app_id}: {exc}",
                flush=True,
            )

    return mapping


def run(config: Dict[str, Any]) -> Dict[str, Any]:
    warmup_log_path = str(config.get("warmup_log_path", "")).strip()
    if warmup_log_path:
        warmup_log_parent = os.path.dirname(warmup_log_path)
        if warmup_log_parent:
            os.makedirs(warmup_log_parent, exist_ok=True)
        with open(warmup_log_path, "w", encoding="utf-8") as f:
            f.write("")
    warmup_summary = None
    warmup_enabled = bool(config.get("warmup_enabled", False))
    if warmup_enabled:
        warmup_records = _load_warmup_records(config)
        warmup_rel = config.get("warmup_file", "")
        data_dir = config.get("data_dir", "")
        warmup_path = os.path.join(str(data_dir), str(warmup_rel)) if warmup_rel else ""
        print(
            f"[warmup] loaded {len(warmup_records)} conversation(s)"
            + (f" from {warmup_path}" if warmup_path else ""),
            flush=True,
        )
        warmup_summary = _run_warmup_phase(warmup_records, config, warmup_log_path)

    if warmup_summary is not None:
        print("[warmup] summary (JSON):", flush=True)
        print(json.dumps(warmup_summary, ensure_ascii=False, indent=2))

    app_id_to_server_id = _register_applications_with_server(config)
    config["_app_id_to_server_id"] = app_id_to_server_id
    if app_id_to_server_id:
        print(
            f"[slo-register] registered {len(app_id_to_server_id)} application(s) with server",
            flush=True,
        )

    app_records = _load_app_records(config)
    if not app_records:
        raise RuntimeError("No application data files matched config.")

    max_conversations = int(config.get("max_conversations_per_app", 0))
    default_threads = int(config.get("default_threads_per_app", 1))
    thread_overrides = config.get("threads_per_application", {})
    slo_expectations = config.get("application_slo_expectations", {})
    accuracy_slo_metric = str(config.get("accuracy_slo_metric", "avg_sequence_ratio"))
    if accuracy_slo_metric not in {"exact_match_rate", "avg_token_f1", "avg_sequence_ratio"}:
        accuracy_slo_metric = "avg_sequence_ratio"
    experiment_duration_s = _safe_float(config.get("experiment_duration_seconds", 0.0), default=0.0)
    if experiment_duration_s < 0:
        experiment_duration_s = 0.0

    log_path = str(config.get("output_request_log_path", "")).strip()
    if log_path:
        log_parent = os.path.dirname(log_path)
        if log_parent:
            os.makedirs(log_parent, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("")
    log_lock = threading.Lock()

    baseline_path = str(config.get("accuracy_baseline_log_path", "")).strip()
    if baseline_path:
        if not os.path.isfile(baseline_path):
            raise FileNotFoundError(
                f"accuracy_baseline_log_path does not exist: {baseline_path}"
            )
        config["_accuracy_baseline_map"] = _load_accuracy_baseline_from_log(baseline_path)
    else:
        config["_accuracy_baseline_map"] = None

    load_mult = _safe_float(config.get("load_multiplier", 1.0), default=1.0)
    if load_mult <= 0:
        load_mult = 1.0
    if baseline_path and abs(load_mult - 1.0) > 1e-9:
        print(
            "[accuracy] note: accuracy_baseline_log_path is set and load_multiplier != 1.0. "
            "Baseline keys are (app, thread_id, conv, turn); use the same load_multiplier "
            "when recording the baseline or comparisons may be sparse.",
            flush=True,
        )

    state = RunState()
    experiment_end_time = (
        state.started_at + experiment_duration_s if experiment_duration_s > 0 else None
    )
    futures = []
    threads_per_app_effective: Dict[str, int] = {}
    worker_jobs: List[Tuple[int, int, List[Dict[str, Any]], int]] = []
    for app_id in sorted(app_records.keys()):
        app_convs = app_records[app_id]
        if max_conversations > 0:
            app_convs = app_convs[:max_conversations]
        base_threads = int(thread_overrides.get(str(app_id), default_threads))
        base_threads = max(1, base_threads)
        scaled_threads = _scaled_thread_count(base_threads, load_mult)
        threads_per_app_effective[str(app_id)] = scaled_threads
        split = _split_for_threads(app_convs, scaled_threads)
        for thread_idx, shard in enumerate(split):
            conversation_draw_count = len(shard)
            if conversation_draw_count <= 0:
                continue
            worker_jobs.append((app_id, thread_idx, app_convs, conversation_draw_count))

    max_workers_cfg = int(config.get("max_workers", 64))
    pool_size = max(max_workers_cfg, len(worker_jobs))
    total_threads = len(worker_jobs)

    with ThreadPoolExecutor(max_workers=pool_size) as executor:
        for app_id, thread_idx, app_convs, conversation_draw_count in worker_jobs:
            futures.append(
                executor.submit(
                    _thread_worker,
                    app_id,
                    thread_idx,
                    app_convs,
                    conversation_draw_count,
                    experiment_end_time,
                    config,
                    state,
                    log_path,
                    log_lock,
                )
            )
        for fut in as_completed(futures):
            fut.result()

    elapsed_s = max(0.001, time.time() - state.started_at)
    per_app_summary: Dict[str, Any] = {}
    slo_eval_total = 0
    slo_eval_passed = 0
    slo_eval_failed = 0
    for app_id, stats in sorted(state.per_app.items()):
        lat = stats["latencies_ms"]
        served_lat = stats["served_latencies_ms"]
        app_acc_count = int(stats["accuracy_count"])
        app_exact = int(stats["exact_match_count"])
        app_token_f1_avg = (
            float(stats["accuracy_token_f1_sum"]) / app_acc_count if app_acc_count else 0.0
        )
        app_seq_ratio_avg = (
            float(stats["accuracy_sequence_ratio_sum"]) / app_acc_count if app_acc_count else 0.0
        )
        p99_latency_all = round(_percentile(lat, 99), 3)
        served_p99_latency = round(_percentile(served_lat, 99), 3)
        observed_latency_for_slo = served_p99_latency if served_lat else p99_latency_all
        observed_accuracy_for_slo = {
            "exact_match_rate": round((app_exact / app_acc_count) if app_acc_count else 0.0, 6),
            "avg_token_f1": round(app_token_f1_avg, 6),
            "avg_sequence_ratio": round(app_seq_ratio_avg, 6),
        }[accuracy_slo_metric]

        app_slo = slo_expectations.get(str(app_id), {})
        target_latency = app_slo.get("latency_p99_ms")
        target_accuracy = app_slo.get("accuracy_slo")
        latency_slo_met = None
        accuracy_slo_met = None
        app_slo_attained = None
        if target_latency is not None:
            latency_slo_met = observed_latency_for_slo <= _safe_float(target_latency, default=float("inf"))
        if target_accuracy is not None:
            accuracy_slo_met = observed_accuracy_for_slo >= _safe_float(target_accuracy, default=0.0)
        if latency_slo_met is not None and accuracy_slo_met is not None:
            app_slo_attained = bool(latency_slo_met and accuracy_slo_met)
            slo_eval_total += 1
            if app_slo_attained:
                slo_eval_passed += 1
            else:
                slo_eval_failed += 1

        per_app_summary[str(app_id)] = {
            "requests": stats["requests"],
            "success": stats["success"],
            "errors": stats["errors"],
            "cache_hits": stats["cache_hits"],
            "cache_misses": stats["cache_misses"],
            "placeholder_responses": stats["placeholder_responses"],
            "p50_latency_ms": round(_percentile(lat, 50), 3),
            "p95_latency_ms": round(_percentile(lat, 95), 3),
            "p99_latency_ms": p99_latency_all,
            "served_p50_latency_ms": round(_percentile(served_lat, 50), 3),
            "served_p95_latency_ms": round(_percentile(served_lat, 95), 3),
            "served_p99_latency_ms": served_p99_latency,
            "throughput_rps": round(stats["requests"] / elapsed_s, 3),
            "accuracy_count": app_acc_count,
            "exact_match_rate": round((app_exact / app_acc_count) if app_acc_count else 0.0, 6),
            "avg_token_f1": round(app_token_f1_avg, 6),
            "avg_sequence_ratio": round(app_seq_ratio_avg, 6),
            "slo_target": {
                "latency_p99_ms": target_latency,
                "accuracy_slo": target_accuracy,
                "accuracy_metric": accuracy_slo_metric,
            },
            "slo_observed": {
                "latency_p99_ms": observed_latency_for_slo,
                accuracy_slo_metric: observed_accuracy_for_slo,
            },
            "slo_attainment": {
                "latency_slo_met": latency_slo_met,
                "accuracy_slo_met": accuracy_slo_met,
                "attained": app_slo_attained,
            },
        }

    lat_all = state.latencies_ms
    served_lat_all = state.served_latencies_ms
    accuracy_count = state.accuracy_count
    exact_match_rate = (state.exact_match_count / accuracy_count) if accuracy_count else 0.0
    avg_token_f1 = (state.accuracy_token_f1_sum / accuracy_count) if accuracy_count else 0.0
    avg_sequence_ratio = (
        state.accuracy_sequence_ratio_sum / accuracy_count if accuracy_count else 0.0
    )
    if bool(config.get("accuracy_baseline_reference_run", False)):
        accuracy_eval_mode = "baseline_reference_run"
    elif baseline_path:
        accuracy_eval_mode = "vs_nocache_log"
    else:
        accuracy_eval_mode = "vs_dataset"

    result = {
        "dry_run": bool(config.get("dry_run", False)),
        "accuracy_eval": {
            "mode": accuracy_eval_mode,
            "baseline_log_path": baseline_path or None,
            "baseline_reference_run": bool(
                config.get("accuracy_baseline_reference_run", False)
            ),
        },
        "load_multiplier": _safe_float(config.get("load_multiplier", 1.0), default=1.0),
        "load_multiplier_scales": "threads_per_application",
        "load_multiplier_start_seconds": _safe_float(
            config.get("load_multiplier_start_seconds", 0), default=0.0
        ),
        "experiment_duration_seconds": round(experiment_duration_s, 3),
        "time_limited_run": bool(experiment_end_time is not None),
        "threads_per_application_effective": threads_per_app_effective,
        "threads_spawned": total_threads,
        "applications_seen": sorted(list(app_records.keys())),
        "requests_sent": state.request_count,
        "success": state.success_count,
        "errors": state.error_count,
        "cache_hits": state.cache_hit_count,
        "cache_misses": state.cache_miss_count,
        "placeholder_response_count": state.placeholder_response_count,
        "placeholder_response_detected": bool(state.placeholder_response_count),
        "elapsed_seconds": round(elapsed_s, 3),
        "overall_throughput_rps": round(state.request_count / elapsed_s, 3),
        "overall_p50_latency_ms": round(_percentile(lat_all, 50), 3),
        "overall_p95_latency_ms": round(_percentile(lat_all, 95), 3),
        "overall_p99_latency_ms": round(_percentile(lat_all, 99), 3),
        "overall_median_latency_ms": round(median(lat_all), 3) if lat_all else 0.0,
        "served_p50_latency_ms": round(_percentile(served_lat_all, 50), 3),
        "served_p95_latency_ms": round(_percentile(served_lat_all, 95), 3),
        "served_p99_latency_ms": round(_percentile(served_lat_all, 99), 3),
        "served_median_latency_ms": round(median(served_lat_all), 3) if served_lat_all else 0.0,
        "accuracy_count": accuracy_count,
        "exact_match_rate": round(exact_match_rate, 6),
        "avg_token_f1": round(avg_token_f1, 6),
        "avg_sequence_ratio": round(avg_sequence_ratio, 6),
        "slo_summary": {
            "accuracy_metric": accuracy_slo_metric,
            "applications_with_targets": slo_eval_total,
            "applications_attained": slo_eval_passed,
            "applications_missed": slo_eval_failed,
        },
        "per_application": per_app_summary,
    }
    return result


def main() -> None:
    args = parse_args()
    config = _load_json(args.config)
    metrics = run(config)
    metrics_path = str(config.get("output_metrics_path", "")).strip()
    if metrics_path:
        _write_json(metrics_path, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
