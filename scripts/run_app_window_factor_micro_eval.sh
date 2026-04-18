#!/usr/bin/env bash
set -euo pipefail

# Remote-dependent micro evaluation + tuning harness for SLO-adaptive app factors.
#
# The harness executes three arms per setting:
# 1) no-cache reference run (accuracy baseline)
# 2) contextcache with slo_adaptive OFF
# 3) contextcache with slo_adaptive ON
#
# Accuracy is measured against no-cache responses for the same request key.
# Workload is intentionally asymmetric:
# - latency_sensitive: shorter prompts, more frequent requests
# - accuracy_sensitive: longer prompts, less frequent requests
#
# Optional env (core):
#   BASE_WINDOW=6
#   WINDOW_MAX=16
#   OVERALL_FACTOR=1.0
#   MODEL=@azure-1/gpt-4o
#   API_KEY_ENV=VOCAREUM_API_KEY
#
# Optional env (workload shape):
#   LOOPS=1
#   LATENCY_TEMPLATES=6
#   LATENCY_REPEAT=2
#   LATENCY_PAUSE_MS=20
#   ACCURACY_TEMPLATES=2
#   ACCURACY_REPEAT=1
#   ACCURACY_PAUSE_MS=180
#
# Optional env (tuning grid):
#   GRID_BASE_WINDOWS=4,6,8
#   GRID_OVERALL_FACTORS=0.8,1.0,1.2
#   GRID_WINDOW_MAX=12,16
#   MAX_GRID_CONFIGS=9
#   STABILITY_REPEATS=2
#   RESULTS_DIR=/tmp/slo_adaptive_remote_eval

BASE_WINDOW="${BASE_WINDOW:-6}"
WINDOW_MAX="${WINDOW_MAX:-16}"
OVERALL_FACTOR="${OVERALL_FACTOR:-1.0}"
MODEL="${MODEL:-@azure-1/gpt-4o}"
API_KEY_ENV="${API_KEY_ENV:-VOCAREUM_API_KEY}"

python3 - <<'PY'
import asyncio
import json
import os
import statistics
import tempfile
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import httpx
from httpx import ASGITransport

from gptcache import cache, Cache
from gptcache.adapter import openai as cache_openai
from gptcache.adapter.api import init_similar_cache
from gptcache.config import Config
from gptcache.processor.pre import last_content
from gptcache_server import server
from gptcache_server.server import ApplicationSloIn, register_application_slo


BASE_WINDOW_DEFAULT = int(os.environ.get("BASE_WINDOW", "6"))
WINDOW_MAX_DEFAULT = int(os.environ.get("WINDOW_MAX", "16"))
OVERALL_FACTOR_DEFAULT = float(os.environ.get("OVERALL_FACTOR", "1.0"))

MODEL = os.environ.get("MODEL", "@azure-1/gpt-4o")
API_KEY_ENV = os.environ.get("API_KEY_ENV", "VOCAREUM_API_KEY")

LOOPS = int(os.environ.get("LOOPS", "1"))
LATENCY_TEMPLATES = int(os.environ.get("LATENCY_TEMPLATES", "6"))
LATENCY_REPEAT = int(os.environ.get("LATENCY_REPEAT", "2"))
LATENCY_PAUSE_MS = float(os.environ.get("LATENCY_PAUSE_MS", "20"))
ACCURACY_TEMPLATES = int(os.environ.get("ACCURACY_TEMPLATES", "2"))
ACCURACY_REPEAT = int(os.environ.get("ACCURACY_REPEAT", "1"))
ACCURACY_PAUSE_MS = float(os.environ.get("ACCURACY_PAUSE_MS", "180"))
REQUEST_TIMEOUT_SECONDS = float(os.environ.get("REQUEST_TIMEOUT_SECONDS", "45"))
REQUEST_MAX_RETRIES = int(os.environ.get("REQUEST_MAX_RETRIES", "2"))
RETRY_BACKOFF_MS = float(os.environ.get("RETRY_BACKOFF_MS", "250"))

GRID_BASE_WINDOWS = os.environ.get("GRID_BASE_WINDOWS", "4,6,8")
GRID_OVERALL_FACTORS = os.environ.get("GRID_OVERALL_FACTORS", "0.8,1.0,1.2")
GRID_WINDOW_MAX = os.environ.get("GRID_WINDOW_MAX", "12,16")
MAX_GRID_CONFIGS = int(os.environ.get("MAX_GRID_CONFIGS", "9"))
STABILITY_REPEATS = int(os.environ.get("STABILITY_REPEATS", "2"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "/tmp/slo_adaptive_remote_eval"))

if not os.getenv(API_KEY_ENV):
    raise SystemExit(f"Missing API key env: {API_KEY_ENV}")

# Ensure wrapper uses real upstream, not any prior injected local LLM.
cache_openai.ChatCompletion.llm = None


def _parse_csv_ints(raw: str) -> List[int]:
    out = []
    for p in raw.split(","):
        s = p.strip()
        if not s:
            continue
        out.append(int(s))
    return out


def _parse_csv_floats(raw: str) -> List[float]:
    out = []
    for p in raw.split(","):
        s = p.strip()
        if not s:
            continue
        out.append(float(s))
    return out


def _targets() -> Dict[str, Dict[str, float]]:
    return {
        "latency_sensitive": {"latency_p99_ms": 2500.0, "accuracy_slo": 0.20},
        "accuracy_sensitive": {"latency_p99_ms": 3000.0, "accuracy_slo": 0.70},
    }


def _init_cache_state(
    slo_adaptive: bool, base_window: int, window_max: int, overall_factor: float
) -> None:
    root = Path(tempfile.mkdtemp(prefix="acc_slo_remote_"))
    main_dir = root / "main_cache"
    openai_dir = root / "openai_cache"
    main_dir.mkdir(parents=True, exist_ok=True)
    openai_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config(
        context_cache_window_len=base_window,
        context_cache_window_max=window_max,
        context_cache_overall_factor=overall_factor,
        slo_adaptive=slo_adaptive,
        dialuoge_threshold=0.95,
    )
    init_similar_cache(data_dir=str(main_dir), config=cfg)

    server.openai_cache = Cache()
    oai_cfg = Config(
        context_cache_window_len=base_window,
        context_cache_window_max=window_max,
        context_cache_overall_factor=overall_factor,
        slo_adaptive=slo_adaptive,
        dialuoge_threshold=0.95,
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
        p99 = _p99(lats)
        avg_acc = statistics.mean(accs) if accs else 0.0
        hit_rate = sum(1 for r in rs if r["is_hit"]) / float(len(rs)) if rs else 0.0
        t = app_targets[app_name]
        lat_ok = p99 <= t["latency_p99_ms"]
        acc_ok = avg_acc >= t["accuracy_slo"]
        out[app_name] = {
            "requests": len(rs),
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
    plan = []
    for lp in range(LOOPS):
        for t in range(LATENCY_TEMPLATES):
            for rep in range(LATENCY_REPEAT):
                plan.append(
                    {
                        "loop": lp,
                        "template_idx": t,
                        "repeat_idx": rep,
                        "app": "latency_sensitive",
                    }
                )
        for t in range(ACCURACY_TEMPLATES):
            for rep in range(ACCURACY_REPEAT):
                plan.append(
                    {
                        "loop": lp,
                        "template_idx": t,
                        "repeat_idx": rep,
                        "app": "accuracy_sensitive",
                    }
                )
    return plan


def _make_payload(template_idx: int, app_id: str, app_name: str):
    if app_name == "latency_sensitive":
        content = [
            f"LAT_REQ_{template_idx}",
            "PING",
            "Return the first token exactly as written.",
        ]
        max_tokens = 8
    else:
        long_context = (
            f"Document {template_idx}: "
            "Adaptive caches reduce repeated context transfer. "
            "The service has latency and accuracy SLOs. "
            "Explain why preserving key details matters for correctness. "
            "Include references to prior context windows and retrieval precision."
        )
        content = [
            f"ACC_REQ_{template_idx}",
            long_context,
            "Provide a concise explanation with two numbered points and keep factual wording.",
        ]
        max_tokens = 80
    return {
        "model": MODEL,
        "application_id": app_id,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }


def _pause_for_app_ms(app_name: str) -> float:
    if app_name == "latency_sensitive":
        return LATENCY_PAUSE_MS
    return ACCURACY_PAUSE_MS


def _gap_from_summary(summary: Dict[str, Dict[str, float]], app_targets: Dict[str, Dict[str, float]]):
    app_gaps: Dict[str, Dict[str, float]] = {}
    total_gap = 0.0
    for app_name in ("latency_sensitive", "accuracy_sensitive"):
        target = app_targets[app_name]
        observed = summary[app_name]
        lat_target = target["latency_p99_ms"]
        acc_target = target["accuracy_slo"]
        lat_gap = max(0.0, observed["p99_latency_ms"] - lat_target) / lat_target
        acc_gap = max(0.0, acc_target - observed["avg_accuracy_seq_ratio"]) / max(acc_target, 1e-9)
        total = lat_gap + acc_gap
        total_gap += total
        app_gaps[app_name] = {
            "latency_gap": round(lat_gap, 6),
            "accuracy_gap": round(acc_gap, 6),
            "total_gap": round(total, 6),
        }
    return {
        "per_app": app_gaps,
        "avg_total_gap": round(total_gap / 2.0, 6),
        "sum_total_gap": round(total_gap, 6),
    }


async def _run_scenario(
    slo_adaptive: bool,
    mode: str,
    baseline_map=None,
    *,
    base_window: int,
    window_max: int,
    overall_factor: float,
):
    _init_cache_state(
        slo_adaptive=slo_adaptive,
        base_window=base_window,
        window_max=window_max,
        overall_factor=overall_factor,
    )
    server.server_mode = mode
    transport = ASGITransport(app=server.app)
    rows = []
    observed_map = {}

    app_targets = _targets()

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

        for request_index, req in enumerate(_request_plan()):
            lp = req["loop"]
            t = req["template_idx"]
            rep = req["repeat_idx"]
            app_name = req["app"]
            app_id = app_ids[app_name]
            payload = _make_payload(t, app_id, app_name)
            t0 = time.perf_counter()
            resp = None
            for attempt in range(REQUEST_MAX_RETRIES + 1):
                try:
                    resp = await asyncio.wait_for(
                        client.post(
                            "/v1/chat/completions",
                            json=payload,
                            timeout=None,
                        ),
                        timeout=REQUEST_TIMEOUT_SECONDS,
                    )
                    break
                except Exception:
                    if attempt >= REQUEST_MAX_RETRIES:
                        resp = None
                        break
                    await asyncio.sleep((RETRY_BACKOFF_MS / 1000.0) * (attempt + 1))

            lat_ms = (time.perf_counter() - t0) * 1000.0
            key = (lp, app_name, t, rep, request_index)
            if resp is None or resp.status_code >= 400:
                observed_map[key] = ""
                rows.append(
                    {
                        "app": app_name,
                        "is_hit": False,
                        "latency_ms": lat_ms,
                        "accuracy_seq_ratio": 0.0,
                    }
                )
                continue
            text, is_hit = _extract_answer_and_hit(resp.json())
            observed_map[key] = text
            acc = 1.0 if baseline_map is None else _seq_ratio(text, baseline_map.get(key, ""))
            rows.append(
                {
                    "app": app_name,
                    "is_hit": bool(is_hit),
                    "latency_ms": lat_ms,
                    "accuracy_seq_ratio": acc,
                }
            )

            pause_ms = _pause_for_app_ms(app_name)
            if pause_ms > 0:
                await asyncio.sleep(pause_ms / 1000.0)

    factors = dict(cache.config.context_cache_window_delta_by_app)
    summary = _summarize(rows, app_targets)
    gap = _gap_from_summary(summary, app_targets)
    return summary, factors, observed_map, gap


async def _run_three_arm(setting: Dict[str, float]):
    ref_summary, _, baseline_map, ref_gap = await _run_scenario(
        slo_adaptive=False,
        mode="no-cache",
        baseline_map=None,
        base_window=setting["base_window"],
        window_max=setting["window_max"],
        overall_factor=setting["overall_factor"],
    )
    off_summary, off_factors, _, off_gap = await _run_scenario(
        slo_adaptive=False,
        mode="contextcache",
        baseline_map=baseline_map,
        base_window=setting["base_window"],
        window_max=setting["window_max"],
        overall_factor=setting["overall_factor"],
    )
    on_summary, on_factors, _, on_gap = await _run_scenario(
        slo_adaptive=True,
        mode="contextcache",
        baseline_map=baseline_map,
        base_window=setting["base_window"],
        window_max=setting["window_max"],
        overall_factor=setting["overall_factor"],
    )

    gap_improvement = round(off_gap["avg_total_gap"] - on_gap["avg_total_gap"], 6)
    attained_delta = int(on_summary["attained_count"] - off_summary["attained_count"])
    return {
        "setting": setting,
        "reference": {"summary": ref_summary, "gap": ref_gap},
        "adaptive_off": {"summary": off_summary, "factor_map": off_factors, "gap": off_gap},
        "adaptive_on": {"summary": on_summary, "factor_map": on_factors, "gap": on_gap},
        "compare": {
            "gap_improvement_off_minus_on": gap_improvement,
            "attained_delta_on_minus_off": attained_delta,
        },
    }


def _grid_settings() -> List[Dict[str, float]]:
    base_windows = _parse_csv_ints(GRID_BASE_WINDOWS)
    overall_factors = _parse_csv_floats(GRID_OVERALL_FACTORS)
    window_maxes = _parse_csv_ints(GRID_WINDOW_MAX)
    settings = []
    for bw in base_windows:
        for of in overall_factors:
            for wm in window_maxes:
                settings.append(
                    {
                        "name": f"bw{bw}_of{of:.2f}_wm{wm}",
                        "base_window": int(bw),
                        "overall_factor": float(of),
                        "window_max": int(wm),
                    }
                )
    if MAX_GRID_CONFIGS > 0:
        settings = settings[:MAX_GRID_CONFIGS]
    return settings


def _rank_key(record: Dict) -> Tuple[float, float, float]:
    # Rank by lower adaptive gap, then larger improvement and attained delta.
    on_gap = record["adaptive_on"]["gap"]["avg_total_gap"]
    improvement = record["compare"]["gap_improvement_off_minus_on"]
    attained_delta = record["compare"]["attained_delta_on_minus_off"]
    return (on_gap, -improvement, -float(attained_delta))


def _aggregate_stability(records: Iterable[Dict]) -> Dict[str, float]:
    recs = list(records)
    if not recs:
        return {}
    on_gaps = [r["adaptive_on"]["gap"]["avg_total_gap"] for r in recs]
    off_gaps = [r["adaptive_off"]["gap"]["avg_total_gap"] for r in recs]
    improvements = [r["compare"]["gap_improvement_off_minus_on"] for r in recs]
    attained_deltas = [r["compare"]["attained_delta_on_minus_off"] for r in recs]
    return {
        "runs": len(recs),
        "mean_on_gap": round(statistics.mean(on_gaps), 6),
        "mean_off_gap": round(statistics.mean(off_gaps), 6),
        "mean_gap_improvement": round(statistics.mean(improvements), 6),
        "mean_attained_delta": round(statistics.mean(attained_deltas), 6),
    }


async def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== Workload Shape ===")
    print(
        json.dumps(
            {
                "loops": LOOPS,
                "latency_sensitive": {
                    "templates": LATENCY_TEMPLATES,
                    "repeat": LATENCY_REPEAT,
                    "pause_ms": LATENCY_PAUSE_MS,
                },
                "accuracy_sensitive": {
                    "templates": ACCURACY_TEMPLATES,
                    "repeat": ACCURACY_REPEAT,
                    "pause_ms": ACCURACY_PAUSE_MS,
                },
            },
            indent=2,
            sort_keys=True,
        )
    )

    settings = _grid_settings()
    if not settings:
        raise SystemExit("No grid settings resolved from GRID_* env values.")

    print("\n=== Grid Settings ===")
    print(json.dumps(settings, indent=2))

    grid_results = []
    for idx, setting in enumerate(settings, start=1):
        print(f"\n--- Running grid setting {idx}/{len(settings)}: {setting['name']} ---")
        record = await _run_three_arm(setting)
        grid_results.append(record)
        out_file = RESULTS_DIR / f"grid_{setting['name']}.json"
        out_file.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
        print(f"saved: {out_file}")
        print(
            "gap_improvement_off_minus_on:",
            record["compare"]["gap_improvement_off_minus_on"],
            "| attained_delta_on_minus_off:",
            record["compare"]["attained_delta_on_minus_off"],
        )

    ranked = sorted(grid_results, key=_rank_key)
    print("\n=== Ranked Grid (best first) ===")
    for pos, rec in enumerate(ranked, start=1):
        name = rec["setting"]["name"]
        on_gap = rec["adaptive_on"]["gap"]["avg_total_gap"]
        off_gap = rec["adaptive_off"]["gap"]["avg_total_gap"]
        impr = rec["compare"]["gap_improvement_off_minus_on"]
        attained_delta = rec["compare"]["attained_delta_on_minus_off"]
        print(
            f"{pos:>2}. {name} | on_gap={on_gap:.6f} off_gap={off_gap:.6f} "
            f"improvement={impr:.6f} attained_delta={attained_delta:+d}"
        )

    default_name = f"bw{BASE_WINDOW_DEFAULT}_of{OVERALL_FACTOR_DEFAULT:.2f}_wm{WINDOW_MAX_DEFAULT}"
    best = ranked[0]
    control = next((r for r in ranked if r["setting"]["name"] == default_name), None)
    if control is None:
        control = ranked[-1]
    if control["setting"]["name"] == best["setting"]["name"] and len(ranked) > 1:
        control = ranked[-1]

    print("\n=== Stability Check Selection ===")
    print("best:", best["setting"]["name"])
    print("control:", control["setting"]["name"])

    stability_records = {"best": [], "control": []}
    for label, rec in (("best", best), ("control", control)):
        setting = rec["setting"]
        for rep in range(STABILITY_REPEATS):
            print(f"\n--- Stability {label} run {rep + 1}/{STABILITY_REPEATS}: {setting['name']} ---")
            run_rec = await _run_three_arm(setting)
            stability_records[label].append(run_rec)
            out_file = RESULTS_DIR / f"stability_{label}_{rep + 1}_{setting['name']}.json"
            out_file.write_text(json.dumps(run_rec, indent=2, sort_keys=True), encoding="utf-8")
            print(f"saved: {out_file}")

    stability_summary = {
        "best": _aggregate_stability(stability_records["best"]),
        "control": _aggregate_stability(stability_records["control"]),
    }
    summary_file = RESULTS_DIR / "stability_summary.json"
    summary_file.write_text(json.dumps(stability_summary, indent=2, sort_keys=True), encoding="utf-8")

    ranked_file = RESULTS_DIR / "ranked_grid_results.json"
    ranked_file.write_text(json.dumps(ranked, indent=2, sort_keys=True), encoding="utf-8")

    print("\n=== Stability Summary ===")
    print(json.dumps(stability_summary, indent=2, sort_keys=True))
    print(f"saved: {summary_file}")
    print(f"saved: {ranked_file}")

    if stability_summary["best"] and stability_summary["control"]:
        best_impr = stability_summary["best"]["mean_gap_improvement"]
        ctrl_impr = stability_summary["control"]["mean_gap_improvement"]
        print("\n=== Final Signal ===")
        print("mean gap improvement (best):", best_impr)
        print("mean gap improvement (control):", ctrl_impr)
        print("best_is_closer_to_slo_than_control:", bool(best_impr >= ctrl_impr))


asyncio.run(main())
PY

