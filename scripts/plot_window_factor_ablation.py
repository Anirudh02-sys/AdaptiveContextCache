#!/usr/bin/env python3
"""
Ablation for final effective context window per application.

Decomposes each app's final window into:
- base window
- load-adaptive contribution (global overall-factor component)
- SLO contribution (per-app additive delta)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


KV_RE = re.compile(r"([a-zA-Z0-9_]+)=([^\s]+)")
OVERALL_FACTOR_RE = re.compile(r"overall_factor=([0-9eE+.\-]+)")


@dataclass
class AppWindowEvent:
    app_id: str
    base_window: int
    overall_factor: float
    window_min: int
    window_max: int
    target_latency_p99_ms: Optional[float]
    target_accuracy_slo: Optional[float]
    new_slo_delta: int
    new_effective_window: int


@dataclass
class GlobalWindowEvent:
    base_window: int
    window_min: int
    window_max: int
    new_overall_factor: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot load-vs-SLO contribution to final effective window per app."
    )
    p.add_argument("--metrics", required=True, help="request_metrics_*.json path")
    p.add_argument("--server-log", required=True, help="server_*.log path")
    p.add_argument("--output-dir", required=True, help="output directory")
    p.add_argument("--prefix", default="window_factor_ablation", help="output file prefix")
    return p.parse_args()


def _to_int(v: Optional[str], default: int) -> int:
    if v is None:
        return default
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def _to_float(v: Optional[str], default: float) -> float:
    if v is None:
        return default
    try:
        cleaned = str(v).strip().rstrip(",)")
        x = float(cleaned)
        if math.isnan(x):
            return default
        return x
    except (TypeError, ValueError):
        return default


def _to_optional_float(v: Optional[str]) -> Optional[float]:
    if v is None:
        return None
    if v in {"None", "null", "nan", "NaN"}:
        return None
    try:
        x = float(v)
        if math.isnan(x):
            return None
        return x
    except (TypeError, ValueError):
        return None


def parse_kv_line(line: str) -> Dict[str, str]:
    return {k: v for k, v in KV_RE.findall(line)}


def parse_server_log(path: str) -> Tuple[Dict[str, AppWindowEvent], Optional[GlobalWindowEvent], int]:
    latest_per_app: Dict[str, AppWindowEvent] = {}
    latest_global: Optional[GlobalWindowEvent] = None
    load_resize_events = 0

    if not os.path.isfile(path):
        return latest_per_app, latest_global, load_resize_events

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if "window_update " in line:
                kv = parse_kv_line(line)
                app_id = kv.get("app_id", "")
                if not app_id:
                    continue
                latest_per_app[app_id] = AppWindowEvent(
                    app_id=app_id,
                    base_window=_to_int(kv.get("base_window"), 5),
                    overall_factor=_to_float(kv.get("overall_factor"), 1.0),
                    window_min=_to_int(kv.get("window_min"), 2),
                    window_max=_to_int(kv.get("window_max"), 32),
                    target_latency_p99_ms=_to_optional_float(kv.get("target_latency_p99_ms")),
                    target_accuracy_slo=_to_optional_float(kv.get("target_accuracy_slo")),
                    new_slo_delta=_to_int(kv.get("new_slo_delta"), 0),
                    new_effective_window=_to_int(kv.get("new_effective_window"), 0),
                )
            elif "window_update_global " in line:
                kv = parse_kv_line(line)
                latest_global = GlobalWindowEvent(
                    base_window=_to_int(kv.get("base_window"), 5),
                    window_min=_to_int(kv.get("window_min"), 1),
                    window_max=_to_int(kv.get("window_max"), 32),
                    new_overall_factor=_to_float(kv.get("new_overall_factor"), 1.0),
                )
            elif "load_adaptive:" in line and "overall_factor=" in line:
                m = OVERALL_FACTOR_RE.search(line)
                if m:
                    # Fallback path: derive final load factor from legacy resize logs.
                    inferred_overall = _to_float(m.group(1), 1.0)
                    load_resize_events += 1
                    fallback_base = next((ev.base_window for ev in latest_per_app.values()), 5)
                    fallback_min = next((ev.window_min for ev in latest_per_app.values()), 2)
                    fallback_max = next((ev.window_max for ev in latest_per_app.values()), 32)
                    latest_global = GlobalWindowEvent(
                        base_window=fallback_base,
                        window_min=fallback_min,
                        window_max=fallback_max,
                        new_overall_factor=inferred_overall,
                    )

    return latest_per_app, latest_global, load_resize_events


def _close(a: Optional[float], b: Optional[float], tol: float = 1e-6) -> bool:
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) <= tol


def map_metric_apps_to_log_apps(
    metric_apps: List[Dict[str, Any]],
    app_events: Dict[str, AppWindowEvent],
) -> Dict[str, str]:
    by_uuid = list(app_events.values())
    used: set[str] = set()
    out: Dict[str, str] = {}

    for m in metric_apps:
        m_app_id = str(m["app_id"])
        target = m.get("slo_target", {}) or {}
        m_lat = _to_optional_float(str(target.get("latency_p99_ms"))) if target.get("latency_p99_ms") is not None else None
        m_acc = _to_optional_float(str(target.get("accuracy_slo"))) if target.get("accuracy_slo") is not None else None
        candidates = [
            e for e in by_uuid
            if e.app_id not in used
            and _close(e.target_latency_p99_ms, m_lat)
            and _close(e.target_accuracy_slo, m_acc)
        ]
        if candidates:
            chosen = candidates[0]
            out[m_app_id] = chosen.app_id
            used.add(chosen.app_id)

    remaining_metrics = [str(m["app_id"]) for m in metric_apps if str(m["app_id"]) not in out]
    remaining_logs = [e.app_id for e in by_uuid if e.app_id not in used]
    for m_app_id, log_app_id in zip(sorted(remaining_metrics), sorted(remaining_logs)):
        out[m_app_id] = log_app_id
        used.add(log_app_id)

    return out


def read_metric_apps(metrics_path: str) -> List[Dict[str, Any]]:
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    per_app = metrics.get("per_application", {}) or {}
    rows: List[Dict[str, Any]] = []
    for app_id, stats in per_app.items():
        rows.append(
            {
                "app_id": str(app_id),
                "stats": stats if isinstance(stats, dict) else {},
                "slo_target": (stats or {}).get("slo_target", {}) if isinstance(stats, dict) else {},
            }
        )
    return sorted(rows, key=lambda r: int(r["app_id"]) if str(r["app_id"]).isdigit() else str(r["app_id"]))


def clamp(x: int, lo: int, hi: int) -> int:
    return max(lo, min(x, hi))


def build_ablation(
    metric_apps: List[Dict[str, Any]],
    app_events: Dict[str, AppWindowEvent],
    global_event: Optional[GlobalWindowEvent],
    load_resize_events: int,
) -> Dict[str, Any]:
    app_map = map_metric_apps_to_log_apps(metric_apps, app_events)

    fallback_base = next((ev.base_window for ev in app_events.values()), 5)
    fallback_overall = next((ev.overall_factor for ev in app_events.values()), 1.0)
    base_window = global_event.base_window if global_event else fallback_base
    final_overall = global_event.new_overall_factor if global_event else fallback_overall
    fallback_w_min = next((ev.window_min for ev in app_events.values()), 2)
    fallback_w_max = next((ev.window_max for ev in app_events.values()), 32)
    w_min = global_event.window_min if global_event else fallback_w_min
    w_max = global_event.window_max if global_event else fallback_w_max

    base_component = int(round(base_window * final_overall))
    per_app_rows: List[Dict[str, Any]] = []
    for m in metric_apps:
        m_app_id = str(m["app_id"])
        mapped_log_app_id = app_map.get(m_app_id)
        ev = app_events.get(mapped_log_app_id or "")
        slo_delta = int(ev.new_slo_delta) if ev else 0
        unclamped = base_component + slo_delta
        final_window = clamp(unclamped, w_min, w_max)
        row = {
            "application_id": m_app_id,
            "mapped_server_application_id": mapped_log_app_id,
            "base_window": base_window,
            "window_min": w_min,
            "window_max": w_max,
            "final_overall_factor": final_overall,
            "base_component_after_load": base_component,
            "load_component_delta_vs_base": base_component - base_window,
            "slo_component_delta": slo_delta,
            "final_window_unclamped": unclamped,
            "final_window": final_window,
            "logged_final_window": ev.new_effective_window if ev else None,
            "slo_target": m.get("slo_target", {}),
        }
        per_app_rows.append(row)

    return {
        "base_window": base_window,
        "window_min": w_min,
        "window_max": w_max,
        "final_overall_factor": final_overall,
        "load_resize_events_observed": int(load_resize_events),
        "app_mapping": app_map,
        "per_application": per_app_rows,
    }


def write_plot(ablation: Dict[str, Any], output_dir: str, prefix: str) -> str:
    rows = ablation.get("per_application", [])
    labels = [str(r["application_id"]) for r in rows]
    base_window = [float(r["base_window"]) for r in rows]
    load_component_delta = [float(r["load_component_delta_vs_base"]) for r in rows]
    slo_component = [float(r["slo_component_delta"]) for r in rows]
    final_window = [float(r["final_window"]) for r in rows]

    x = list(range(len(labels)))
    w = 0.2
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axhline(0.0, color="#666666", linewidth=1, linestyle="--")
    ax.bar([i - 1.5 * w for i in x], base_window, width=w, label="base_window")
    load_bars = ax.bar([i - 0.5 * w for i in x], load_component_delta, width=w, label="load_component_delta")
    slo_bars = ax.bar([i + 0.5 * w for i in x], slo_component, width=w, label="slo_component_delta")
    ax.bar([i + 1.5 * w for i in x], final_window, width=w, label="final_window")

    for bar, value in zip(load_bars, load_component_delta):
        y = bar.get_height()
        va = "bottom" if y >= 0 else "top"
        offset = 0.15 if y >= 0 else -0.15
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            y + offset,
            f"{value:+.0f}",
            ha="center",
            va=va,
            fontsize=8,
        )

    for bar, value in zip(slo_bars, slo_component):
        y = bar.get_height()
        va = "bottom" if y >= 0 else "top"
        offset = 0.15 if y >= 0 else -0.15
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            y + offset,
            f"{value:+.0f}",
            ha="center",
            va=va,
            fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Application ID")
    ax.set_ylabel("Turns")
    ax.set_title("Window Ablation: load vs SLO contribution to final window")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    os.makedirs(output_dir, exist_ok=True)
    out_png = os.path.join(output_dir, f"{prefix}_window_factor_ablation.png")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    metric_apps = read_metric_apps(args.metrics)
    app_events, global_event, load_resize_events = parse_server_log(args.server_log)
    if not app_events:
        print(
            f"WARNING: no per-app window_update lines found in {args.server_log}; "
            "falling back to neutral SLO deltas.",
        )
    if global_event is None:
        print(
            f"WARNING: no window_update_global line found in {args.server_log}; "
            "falling back to last known/default overall factor.",
        )
    if load_resize_events == 0:
        print(
            "WARNING: no load_adaptive resize events observed; load delta may be zero.",
        )
    ablation = build_ablation(metric_apps, app_events, global_event, load_resize_events)

    out_json = os.path.join(args.output_dir, f"{args.prefix}_window_factor_ablation.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(ablation, f, indent=2, sort_keys=True)

    out_png = write_plot(ablation, args.output_dir, args.prefix)
    print("Wrote window-factor ablation:")
    print(f"- {out_json}")
    print(f"- {out_png}")


if __name__ == "__main__":
    main()
