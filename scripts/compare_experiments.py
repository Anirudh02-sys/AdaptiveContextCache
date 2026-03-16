#!/usr/bin/env python3
"""
Compare latency, accuracy, and SLO attainment across multiple experiments.

Example:
  python3 scripts/compare_experiments.py \
    --metrics-nocache data/test_apps/example/request_metrics_nocache.json \
    --metrics-gptcache data/test_apps/example/request_metrics_gptcache.json \
    --metrics-contextcache data/test_apps/example/request_metrics_contextcache.json \
    --output-dir data/test_apps/example/plots/compare \
    --prefix compare
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare latency, accuracy, and SLO attainment across experiments."
    )
    parser.add_argument("--metrics-nocache", required=True, help="Metrics JSON for no-cache mode")
    parser.add_argument("--metrics-gptcache", required=True, help="Metrics JSON for gptcache mode")
    parser.add_argument(
        "--metrics-contextcache", required=True, help="Metrics JSON for contextcache mode"
    )
    parser.add_argument(
        "--output-dir",
        default="plots/compare",
        help="Directory to write comparison plot PNG files (default: plots/compare)",
    )
    parser.add_argument(
        "--prefix",
        default="compare",
        help="Filename prefix for generated comparison plot files",
    )
    return parser.parse_args()


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _sorted_per_app(metrics: Dict[str, Any]) -> List[Tuple[int, Dict[str, Any]]]:
    per_app = metrics.get("per_application", {})
    rows: List[Tuple[int, Dict[str, Any]]] = []
    for app_id_str, app_stats in per_app.items():
        try:
            app_id = int(app_id_str)
        except (TypeError, ValueError):
            continue
        if isinstance(app_stats, dict):
            rows.append((app_id, app_stats))
    return sorted(rows, key=lambda x: x[0])


def _ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _save(fig: Any, output_dir: str, filename: str) -> str:
    out_path = os.path.join(output_dir, filename)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_latency_compare(
    nocache: Dict[str, Any],
    gptcache: Dict[str, Any],
    contextcache: Dict[str, Any],
    output_dir: str,
    prefix: str,
) -> str:
    rows0 = _sorted_per_app(nocache)
    app_ids = [app_id for app_id, _ in rows0]

    def p99(m: Dict[str, Any]) -> List[float]:
        rows = _sorted_per_app(m)
        return [float(stats.get("p99_latency_ms", 0.0)) for _, stats in rows]

    p0 = p99(nocache)
    p1 = p99(gptcache)
    p2 = p99(contextcache)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(app_ids, p0, marker="o", label="no-cache p99 (ms)")
    ax.plot(app_ids, p1, marker="o", label="gptcache p99 (ms)")
    ax.plot(app_ids, p2, marker="o", label="contextcache p99 (ms)")
    ax.set_title("P99 Latency by Application (All Experiments)")
    ax.set_xlabel("Application ID")
    ax.set_ylabel("Latency p99 (ms)")
    ax.grid(alpha=0.3)
    ax.legend()
    return _save(fig, output_dir, f"{prefix}_latency_compare.png")


def plot_accuracy_compare(
    nocache: Dict[str, Any],
    gptcache: Dict[str, Any],
    contextcache: Dict[str, Any],
    output_dir: str,
    prefix: str,
) -> str:
    rows0 = _sorted_per_app(nocache)
    app_ids = [app_id for app_id, _ in rows0]

    def acc(m: Dict[str, Any]) -> List[float]:
        rows = _sorted_per_app(m)
        return [float(stats.get("avg_sequence_ratio", 0.0)) for _, stats in rows]

    a0 = acc(nocache)
    a1 = acc(gptcache)
    a2 = acc(contextcache)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(app_ids, a0, marker="o", label="no-cache avg_sequence_ratio")
    ax.plot(app_ids, a1, marker="o", label="gptcache avg_sequence_ratio")
    ax.plot(app_ids, a2, marker="o", label="contextcache avg_sequence_ratio")
    ax.set_title("Accuracy (avg_sequence_ratio) by Application (All Experiments)")
    ax.set_xlabel("Application ID")
    ax.set_ylabel("avg_sequence_ratio")
    ax.grid(alpha=0.3)
    ax.legend()
    return _save(fig, output_dir, f"{prefix}_accuracy_compare.png")


def plot_slo_compare(
    nocache: Dict[str, Any],
    gptcache: Dict[str, Any],
    contextcache: Dict[str, Any],
    output_dir: str,
    prefix: str,
) -> Tuple[str, str]:
    # Use app IDs from no-cache as reference (assumed same across experiments).
    rows0 = _sorted_per_app(nocache)
    app_ids = [app_id for app_id, _ in rows0]

    def latency_margins(m: Dict[str, Any]) -> List[float]:
        rows = _sorted_per_app(m)
        margins: List[float] = []
        for _, stats in rows:
            target = (stats.get("slo_target", {}) or {}).get("latency_p99_ms")
            observed = (stats.get("slo_observed", {}) or {}).get("latency_p99_ms")
            margin = 0.0
            if target is not None and observed is not None:
                try:
                    t = float(target)
                    o = float(observed)
                    # Positive margin means observed latency is better (lower) than target.
                    margin = t - o
                except (TypeError, ValueError):
                    margin = 0.0
            margins.append(margin)
        return margins

    def accuracy_margins(m: Dict[str, Any]) -> List[float]:
        rows = _sorted_per_app(m)
        margins: List[float] = []
        # Use this experiment's configured accuracy metric name, defaulting to avg_sequence_ratio.
        accuracy_metric = str(
            m.get("slo_summary", {}).get("accuracy_metric", "avg_sequence_ratio")
        )
        for _, stats in rows:
            target = (stats.get("slo_target", {}) or {}).get("accuracy_slo")
            observed = (stats.get("slo_observed", {}) or {}).get(accuracy_metric)
            margin = 0.0
            if target is not None and observed is not None:
                try:
                    t = float(target)
                    o = float(observed)
                    # Positive margin means observed accuracy is better (higher) than target.
                    margin = o - t
                except (TypeError, ValueError):
                    margin = 0.0
            margins.append(margin)
        return margins

    lat0 = latency_margins(nocache)
    lat1 = latency_margins(gptcache)
    lat2 = latency_margins(contextcache)

    acc0 = accuracy_margins(nocache)
    acc1 = accuracy_margins(gptcache)
    acc2 = accuracy_margins(contextcache)

    # Multi-bar latency SLO margin per app.
    x = list(range(len(app_ids)))
    width = 0.25

    fig_lat, ax_lat = plt.subplots(figsize=(12, 5))
    ax_lat.axhline(0.0, color="#666666", linewidth=1, linestyle="--")
    ax_lat.bar([xi - width for xi in x], lat0, width=width, label="no-cache")
    ax_lat.bar(x, lat1, width=width, label="gptcache")
    ax_lat.bar([xi + width for xi in x], lat2, width=width, label="contextcache")
    ax_lat.set_xticks(x)
    ax_lat.set_xticklabels(app_ids)
    ax_lat.set_xlabel("Application ID")
    ax_lat.set_ylabel("Latency SLO margin (target - observed, ms)")
    ax_lat.set_title("Latency SLO Margin by Application (All Experiments)")
    ax_lat.grid(axis="y", alpha=0.3)
    ax_lat.legend()
    lat_path = _save(fig_lat, output_dir, f"{prefix}_latency_slo_compare.png")

    # Multi-bar accuracy SLO margin per app.
    fig_acc, ax_acc = plt.subplots(figsize=(12, 5))
    ax_acc.axhline(0.0, color="#666666", linewidth=1, linestyle="--")
    ax_acc.bar([xi - width for xi in x], acc0, width=width, label="no-cache")
    ax_acc.bar(x, acc1, width=width, label="gptcache")
    ax_acc.bar([xi + width for xi in x], acc2, width=width, label="contextcache")
    ax_acc.set_xticks(x)
    ax_acc.set_xticklabels(app_ids)
    ax_acc.set_xlabel("Application ID")
    ax_acc.set_ylabel("Accuracy SLO margin (observed - target)")
    ax_acc.set_title("Accuracy SLO Margin by Application (All Experiments)")
    ax_acc.grid(axis="y", alpha=0.3)
    ax_acc.legend()
    acc_path = _save(fig_acc, output_dir, f"{prefix}_accuracy_slo_compare.png")

    return lat_path, acc_path


def main() -> None:
    args = parse_args()
    _ensure_output_dir(args.output_dir)

    nocache = _load_json(args.metrics_nocache)
    gptcache = _load_json(args.metrics_gptcache)
    contextcache = _load_json(args.metrics_contextcache)

    latency_plot = plot_latency_compare(nocache, gptcache, contextcache, args.output_dir, args.prefix)
    accuracy_plot = plot_accuracy_compare(
        nocache, gptcache, contextcache, args.output_dir, args.prefix
    )
    lat_slo_plot, acc_slo_plot = plot_slo_compare(
        nocache, gptcache, contextcache, args.output_dir, args.prefix
    )

    print("Wrote comparison plots:")
    print(f"- {latency_plot}")
    print(f"- {accuracy_plot}")
    print(f"- {lat_slo_plot}")
    print(f"- {acc_slo_plot}")


if __name__ == "__main__":
    main()

