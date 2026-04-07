#!/usr/bin/env bash
set -euo pipefail

# Micro-evaluation for SLO-adaptive per-application window factors.
# Runs an in-process A/B:
#   A) slo_adaptive = off (app factors forced to 1.0)
#   B) slo_adaptive = on  (app factors computed from SLO targets)
#
# For each scenario, it sends app-tagged chat requests to `/v1/chat/completions`,
# measures latency + no-cache-referenced accuracy, and reports per-app SLO attainment.
#
# App roles:
# - latency_sensitive: should trend toward HIGHER hit rate with slo_adaptive on
#   (smaller context-window factor).
# - accuracy_sensitive: should trend toward LOWER hit rate and HIGHER baseline-fidelity
#   with slo_adaptive on (larger context-window factor).
#
# Usage:
#   chmod +x scripts/run_app_window_factor_micro_eval.sh
#   ./scripts/run_app_window_factor_micro_eval.sh
#
# Optional env knobs:
#   BASE_WINDOW=4
#   WINDOW_MAX=32
#   OVERALL_FACTOR=1.0
#   LOOPS=2
#   TEMPLATES=4
#   MODEL=@azure-1/gpt-4o

BASE_WINDOW="${BASE_WINDOW:-4}"
WINDOW_MAX="${WINDOW_MAX:-32}"
OVERALL_FACTOR="${OVERALL_FACTOR:-1.0}"
LOOPS="${LOOPS:-2}"
TEMPLATES="${TEMPLATES:-4}"
MODEL="${MODEL:-@azure-1/gpt-4o}"

python3 - <<'PY'
import asyncio
import json
import os
import statistics
import tempfile
import time
from difflib import SequenceMatcher
from pathlib import Path

import httpx
from httpx import ASGITransport

from gptcache import cache
from gptcache import Cache
from gptcache.adapter.api import init_similar_cache
from gptcache.config import Config
from gptcache.processor.pre import last_content
from gptcache_server import server
from gptcache_server.server import (
    ApplicationSloIn,
    register_application_slo,
)


BASE_WINDOW = int(os.environ.get("BASE_WINDOW", "8"))
WINDOW_MAX = int(os.environ.get("WINDOW_MAX", "32"))
OVERALL_FACTOR = float(os.environ.get("OVERALL_FACTOR", "1.0"))
LOOPS = int(os.environ.get("LOOPS", "2"))
TEMPLATES = int(os.environ.get("TEMPLATES", "4"))
MODEL = os.environ.get("MODEL", "@azure-1/gpt-4o")

# Must use real upstream (no dry-run).
if not (os.getenv("VOCAREUM_API_KEY") or os.getenv("OPENAI_API_KEY")):
    raise SystemExit(
        "Missing API key. Set VOCAREUM_API_KEY (preferred) or OPENAI_API_KEY to run remote evaluation."
    )

def _init_cache_state(slo_adaptive: bool) -> None:
    # Use persistent temp dirs to avoid faiss flush errors at process exit.
    root = Path(tempfile.mkdtemp(prefix="acc_slo_micro_"))
    main_dir = root / "main_cache"
    openai_dir = root / "openai_cache"
    main_dir.mkdir(parents=True, exist_ok=True)
    openai_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config(
        context_cache_window_len=BASE_WINDOW,
        context_cache_window_max=WINDOW_MAX,
        context_cache_overall_factor=OVERALL_FACTOR,
        slo_adaptive=slo_adaptive,
    )
    init_similar_cache(data_dir=str(main_dir), config=cfg)

    server.openai_cache = Cache()
    oai_cfg = Config(
        context_cache_window_len=BASE_WINDOW,
        context_cache_window_max=WINDOW_MAX,
        context_cache_overall_factor=OVERALL_FACTOR,
        slo_adaptive=slo_adaptive,
    )
    init_similar_cache(
        data_dir=str(openai_dir),
        pre_func=last_content,
        cache_obj=server.openai_cache,
        config=oai_cfg,
    )

    server.server_mode = "contextcache"
    server.dry_run = False
    with server._application_slos_lock:
        server._application_slos.clear()


def _extract_answer_and_hit(resp_json):
    is_hit = False
    data = resp_json
    if isinstance(data, list) and data:
        answer = data[0]
        if len(data) > 1:
            is_hit = bool(data[1])
        data = answer
    elif isinstance(data, dict):
        is_hit = bool(data.get("is_hit", False))

    text = ""
    if isinstance(data, dict):
        choices = data.get("choices") or []
        if choices and isinstance(choices[0], dict):
            msg = choices[0].get("message") or {}
            if isinstance(msg, dict):
                text = str(msg.get("content", ""))
    return text, is_hit


def _p99(values):
    if not values:
        return float("inf")
    vs = sorted(values)
    idx = int(round(0.99 * (len(vs) - 1)))
    return vs[max(0, min(idx, len(vs) - 1))]


def _seq_ratio(a: str, b: str) -> float:
    return float(SequenceMatcher(None, a or "", b or "").ratio())


def _summarize(rows, app_targets):
    out = {}
    for app_name in ("latency_sensitive", "accuracy_sensitive"):
        rs = [r for r in rows if r["app"] == app_name]
        lats = [r["latency_ms"] for r in rs]
        accs = [r["accuracy_seq_ratio"] for r in rs]
        oks = [r["ok"] for r in rs]
        p99 = _p99(lats)
        avg_acc = statistics.mean(accs) if accs else 0.0
        hit_rate = sum(1 for r in rs if r["is_hit"]) / float(len(rs)) if rs else 0.0
        success_rate = sum(1 for ok in oks if ok) / float(len(oks)) if oks else 0.0
        t = app_targets[app_name]
        lat_ok = p99 <= t["latency_p99_ms"]
        acc_ok = avg_acc >= t["accuracy_slo"]
        out[app_name] = {
            "requests": len(rs),
            "success_rate": round(success_rate, 6),
            "hit_rate": round(hit_rate, 6),
            "p99_latency_ms": round(p99, 3),
            "avg_accuracy_seq_ratio": round(avg_acc, 6),
            "latency_slo_met": bool(lat_ok),
            "accuracy_slo_met": bool(acc_ok),
            "attained": bool(lat_ok and acc_ok),
        }
    out["attained_count"] = sum(
        1 for a in ("latency_sensitive", "accuracy_sensitive") if out[a]["attained"]
    )
    return out


def _request_plan():
    # Deterministic order so no-cache baseline can be keyed exactly.
    plan = []
    for lp in range(LOOPS):
        for t in range(TEMPLATES):
            for app_name in ("latency_sensitive", "accuracy_sensitive"):
                plan.append((lp, t, app_name))
    return plan


def _make_payload(template_idx: int, app_id: str):
    shared_tail = [
        "RULE: answer using ONLY the first token in the conversation history.",
        "Output must be exactly that token, no extra words.",
        "COMMON_A",
        "COMMON_B",
        "COMMON_C",
    ]
    return {
        "model": MODEL,
        "application_id": app_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    f"ID_{template_idx}",
                    *shared_tail,
                    "What is the first token? Return exactly that token.",
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": 6,
    }


async def _run_scenario(slo_adaptive: bool, mode: str, baseline_map=None):
    _init_cache_state(slo_adaptive=slo_adaptive)
    server.server_mode = mode
    transport = ASGITransport(app=server.app)
    rows = []
    observed_map = {}

    app_targets = {
        # Latency-sensitive app: tight latency, loose accuracy target.
        "latency_sensitive": {"latency_p99_ms": 1800.0, "accuracy_slo": 0.20},
        # Accuracy-sensitive app: loose latency, strict baseline-fidelity target.
        "accuracy_sensitive": {"latency_p99_ms": 3200.0, "accuracy_slo": 0.70},
    }

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        latency_out = await register_application_slo(
            ApplicationSloIn(
                latency_p99_ms=app_targets["latency_sensitive"]["latency_p99_ms"],
                accuracy_slo=app_targets["latency_sensitive"]["accuracy_slo"],
            )
        )
        accuracy_out = await register_application_slo(
            ApplicationSloIn(
                latency_p99_ms=app_targets["accuracy_sensitive"]["latency_p99_ms"],
                accuracy_slo=app_targets["accuracy_sensitive"]["accuracy_slo"],
            )
        )
        app_ids = {
            "latency_sensitive": latency_out.application_id,
            "accuracy_sensitive": accuracy_out.application_id,
        }

        for lp, t, app_name in _request_plan():
            app_id = app_ids[app_name]
            payload = _make_payload(t, app_id)
            t0 = time.time()
            resp = await client.post("/v1/chat/completions", json=payload)
            lat_ms = (time.time() - t0) * 1000.0
            key = (lp, t, app_name)
            if resp.status_code >= 400:
                observed_map[key] = ""
                rows.append(
                    {
                        "app": app_name,
                        "ok": False,
                        "is_hit": False,
                        "latency_ms": lat_ms,
                        "accuracy_seq_ratio": 0.0,
                    }
                )
                continue
            text, is_hit = _extract_answer_and_hit(resp.json())
            observed_map[key] = text
            if baseline_map is None:
                acc = 1.0
            else:
                acc = _seq_ratio(text, baseline_map.get(key, ""))
            rows.append(
                {
                    "app": app_name,
                    "ok": True,
                    "is_hit": bool(is_hit),
                    "latency_ms": lat_ms,
                    "accuracy_seq_ratio": acc,
                }
            )

    factors = dict(cache.config.context_cache_window_factor_by_app)
    summary = _summarize(rows, app_targets)
    return summary, factors, observed_map


async def main() -> None:
    # 1) no-cache reference run (same request keys) => baseline_map for accuracy.
    nocache_summary, _, baseline_map = await _run_scenario(
        slo_adaptive=False, mode="no-cache", baseline_map=None
    )
    # 2) contextcache with slo_adaptive off (all factors=1.0)
    off_summary, off_factors, _ = await _run_scenario(
        slo_adaptive=False, mode="contextcache", baseline_map=baseline_map
    )
    # 3) contextcache with slo_adaptive on (SLO-derived factors)
    on_summary, on_factors, _ = await _run_scenario(
        slo_adaptive=True, mode="contextcache", baseline_map=baseline_map
    )

    print("\n=== Scenario REF: no-cache baseline ===")
    print("metrics:")
    print(json.dumps(nocache_summary, indent=2, sort_keys=True))

    print("\n=== Scenario A: contextcache, slo_adaptive = OFF ===")
    print("factor_map:")
    print(json.dumps(off_factors, indent=2, sort_keys=True))
    print("metrics:")
    print(json.dumps(off_summary, indent=2, sort_keys=True))

    print("\n=== Scenario B: contextcache, slo_adaptive = ON ===")
    print("factor_map:")
    print(json.dumps(on_factors, indent=2, sort_keys=True))
    print("metrics:")
    print(json.dumps(on_summary, indent=2, sort_keys=True))

    print("\n=== Comparison ===")
    print(
        "attained_count:",
        f"off={off_summary['attained_count']}",
        f"on={on_summary['attained_count']}",
        f"delta={on_summary['attained_count'] - off_summary['attained_count']}",
    )
    for app_name in ("latency_sensitive", "accuracy_sensitive"):
        b = off_summary[app_name]
        f = on_summary[app_name]
        print(
            f"{app_name}:",
            f"p99_latency_ms off={b['p99_latency_ms']} on={f['p99_latency_ms']};",
            f"avg_accuracy_seq_ratio off={b['avg_accuracy_seq_ratio']} on={f['avg_accuracy_seq_ratio']};",
            f"attained off={b['attained']} on={f['attained']}",
        )

    print("\n=== Directional Feature Checks ===")
    lat_hit_delta = on_summary["latency_sensitive"]["hit_rate"] - off_summary["latency_sensitive"]["hit_rate"]
    acc_hit_delta = on_summary["accuracy_sensitive"]["hit_rate"] - off_summary["accuracy_sensitive"]["hit_rate"]
    acc_acc_delta = (
        on_summary["accuracy_sensitive"]["avg_accuracy_seq_ratio"]
        - off_summary["accuracy_sensitive"]["avg_accuracy_seq_ratio"]
    )
    print(
        "latency_sensitive hit-rate delta (want > 0):",
        round(lat_hit_delta, 6),
    )
    print(
        "accuracy_sensitive hit-rate delta (want < 0):",
        round(acc_hit_delta, 6),
    )
    print(
        "accuracy_sensitive accuracy delta (want > 0):",
        round(acc_acc_delta, 6),
    )


asyncio.run(main())
PY

