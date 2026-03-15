#!/usr/bin/env python3
"""
Plot latency, accuracy, and SLO attainment from request generator metrics JSON.

Example:
  python3 scripts/plot_request_metrics.py \
    --metrics data/test_apps/request_metrics_live_test.json \
    --output-dir data/test_apps/plots
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
        description="Plot latency, accuracy, and SLO attainment from metrics JSON."
    )
    parser.add_argument(
        "--metrics",
        required=True,
        help="Path to request metrics JSON from scripts/generate_requests.py",
    )
    parser.add_argument(
        "--output-dir",
        default="plots",
        help="Directory to write plot PNG files (default: plots)",
    )
    parser.add_argument(
        "--prefix",
        default="request_metrics",
        help="Filename prefix for generated plot files",
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


def plot_latency(metrics: Dict[str, Any], output_dir: str, prefix: str) -> str:
    rows = _sorted_per_app(metrics)
    app_ids = [app_id for app_id, _ in rows]
    p50 = [float(stats.get("p50_latency_ms", 0.0)) for _, stats in rows]
    p95 = [float(stats.get("p95_latency_ms", 0.0)) for _, stats in rows]
    p99 = [float(stats.get("p99_latency_ms", 0.0)) for _, stats in rows]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(app_ids, p50, marker="o", label="p50 latency (ms)")
    ax.plot(app_ids, p95, marker="o", label="p95 latency (ms)")
    ax.plot(app_ids, p99, marker="o", label="p99 latency (ms)")
    ax.set_title("Latency by Application")
    ax.set_xlabel("Application ID")
    ax.set_ylabel("Latency (ms)")
    ax.grid(alpha=0.3)
    ax.legend()
    return _save(fig, output_dir, f"{prefix}_latency_by_app.png")


def plot_accuracy(metrics: Dict[str, Any], output_dir: str, prefix: str) -> str:
    rows = _sorted_per_app(metrics)
    app_ids = [app_id for app_id, _ in rows]
    exact = [float(stats.get("exact_match_rate", 0.0)) for _, stats in rows]
    token_f1 = [float(stats.get("avg_token_f1", 0.0)) for _, stats in rows]
    seq_ratio = [float(stats.get("avg_sequence_ratio", 0.0)) for _, stats in rows]
    acc_counts = [int(stats.get("accuracy_count", 0)) for _, stats in rows]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax1.plot(app_ids, exact, marker="o", label="exact_match_rate")
    ax1.plot(app_ids, token_f1, marker="o", label="avg_token_f1")
    ax1.plot(app_ids, seq_ratio, marker="o", label="avg_sequence_ratio")
    ax1.set_ylim(0.0, 1.05)
    ax1.set_ylabel("Score")
    ax1.set_title("Accuracy by Application")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.bar(app_ids, acc_counts, color="#6699cc")
    ax2.set_xlabel("Application ID")
    ax2.set_ylabel("Accuracy sample count")
    ax2.grid(axis="y", alpha=0.3)

    return _save(fig, output_dir, f"{prefix}_accuracy_by_app.png")


def _plot_slo_metric(
    metrics: Dict[str, Any],
    output_dir: str,
    prefix: str,
    metric_key: str,
    title_prefix: str,
    filename_suffix: str,
) -> str:
    rows = _sorted_per_app(metrics)
    app_ids = [app_id for app_id, _ in rows]

    attained = 0
    missed = 0
    no_target = 0
    per_app_code: List[int] = []
    for _, stats in rows:
        slo = stats.get("slo_attainment", {})
        val = slo.get(metric_key) if isinstance(slo, dict) else None
        if val is True:
            attained += 1
            per_app_code.append(1)
        elif val is False:
            missed += 1
            per_app_code.append(-1)
        else:
            no_target += 1
            per_app_code.append(0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(
        ["Attained", "Missed", "No Target"],
        [attained, missed, no_target],
        color=["#4caf50", "#f44336", "#9e9e9e"],
    )
    ax1.set_title(f"{title_prefix} SLO Summary")
    ax1.set_ylabel("Application count")
    ax1.grid(axis="y", alpha=0.3)

    colors = []
    for code in per_app_code:
        if code == 1:
            colors.append("#4caf50")
        elif code == -1:
            colors.append("#f44336")
        else:
            colors.append("#9e9e9e")
    y_vals = [1 if c == 1 else (0 if c == 0 else -1) for c in per_app_code]
    ax2.bar(app_ids, y_vals, color=colors)
    ax2.set_yticks([-1, 0, 1])
    ax2.set_yticklabels(["Missed", "No Target", "Attained"])
    ax2.set_xlabel("Application ID")
    ax2.set_title(f"Per-Application {title_prefix} SLO Status")
    ax2.grid(axis="y", alpha=0.3)

    return _save(fig, output_dir, f"{prefix}_{filename_suffix}.png")


def main() -> None:
    args = parse_args()
    metrics = _load_json(args.metrics)
    _ensure_output_dir(args.output_dir)

    latency_plot = plot_latency(metrics, args.output_dir, args.prefix)
    accuracy_plot = plot_accuracy(metrics, args.output_dir, args.prefix)
    latency_slo_plot = _plot_slo_metric(
        metrics=metrics,
        output_dir=args.output_dir,
        prefix=args.prefix,
        metric_key="latency_slo_met",
        title_prefix="Latency",
        filename_suffix="latency_slo_attainment",
    )
    accuracy_slo_plot = _plot_slo_metric(
        metrics=metrics,
        output_dir=args.output_dir,
        prefix=args.prefix,
        metric_key="accuracy_slo_met",
        title_prefix="Accuracy",
        filename_suffix="accuracy_slo_attainment",
    )

    print("Wrote plots:")
    print(f"- {latency_plot}")
    print(f"- {accuracy_plot}")
    print(f"- {latency_slo_plot}")
    print(f"- {accuracy_slo_plot}")


if __name__ == "__main__":
    main()
