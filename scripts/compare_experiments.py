#!/usr/bin/env python3
"""
Compare latency, accuracy, and SLO attainment across multiple experiments.

Legacy (exactly three runs, fixed keys):
  python3 scripts/compare_experiments.py \
    --metrics-nocache data/test_apps/example/request_metrics_nocache.json \
    --metrics-gptcache data/test_apps/example/request_metrics_gptcache.json \
    --metrics-contextcache data/test_apps/example/request_metrics_contextcache.json \
    --output-dir data/test_apps/example/plots/compare \
    --prefix compare

Flexible (any number of runs; optional request log path for avg latency):
  python3 scripts/compare_experiments.py \
    --run nocache:/path/request_metrics_nocache.json \
    --run gptcache:/path/request_metrics_gptcache.json \
    --output-dir plots/compare --prefix compare
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_RUN_LABELS: Dict[str, str] = {
    "nocache": "no-cache",
    "gptcache": "gptcache",
    "contextcache": "contextcache",
    "adaptive_load": "adaptive context (load)",
    "adaptive_slo": "adaptive context (SLO)",
}


@dataclass(frozen=True)
class ExperimentRun:
    """One experiment: stable key, legend label, metrics JSON path, resolved log path, loaded metrics."""

    key: str
    label: str
    metrics_path: str
    log_path: str
    metrics: Dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare latency, accuracy, and SLO attainment across experiments."
    )
    parser.add_argument(
        "--run",
        action="append",
        default=None,
        metavar="KEY:METRICS_JSON[:LOG_JSONL]",
        help=(
            "Repeat for each experiment. KEY is a short id (e.g. nocache, adaptive_load). "
            "If LOG_JSONL is omitted, it is inferred from the metrics filename "
            "(request_metrics_<suffix>.json -> request_log_<suffix>.jsonl)."
        ),
    )
    parser.add_argument(
        "--label",
        action="append",
        default=None,
        metavar="KEY:Display name",
        help="Override legend label for KEY (repeatable).",
    )
    parser.add_argument("--metrics-nocache", default=None, help="(legacy) Metrics JSON for no-cache mode")
    parser.add_argument("--metrics-gptcache", default=None, help="(legacy) Metrics JSON for gptcache mode")
    parser.add_argument(
        "--metrics-contextcache", default=None, help="(legacy) Metrics JSON for contextcache mode"
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


def parse_label_flags(label_args: Optional[Sequence[str]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not label_args:
        return out
    for raw in label_args:
        if ":" not in raw:
            continue
        key, display = raw.split(":", 1)
        key = key.strip()
        display = display.strip()
        if key:
            out[key] = display
    return out


def infer_request_log_path(metrics_path: str) -> str:
    """
    Infer request log path from metrics path when naming follows:
      request_metrics_<suffix>.json -> request_log_<suffix>.jsonl
    """
    base_dir = os.path.dirname(os.path.abspath(metrics_path))
    base = os.path.basename(metrics_path)
    if base.startswith("request_metrics_") and base.endswith(".json"):
        mid = base[len("request_metrics_") : -len(".json")]
        return os.path.join(base_dir, f"request_log_{mid}.jsonl")
    if "request_metrics" in base and base.endswith(".json"):
        return os.path.join(
            base_dir,
            base.replace("request_metrics", "request_log").replace(".json", ".jsonl"),
        )
    stem, _ext = os.path.splitext(base)
    return os.path.join(base_dir, f"{stem}_log.jsonl")


def parse_run_arg(line: str) -> Tuple[str, str, Optional[str]]:
    """Parse KEY:METRICS[:LOG]."""
    parts = line.split(":")
    if len(parts) < 2:
        raise ValueError(f"Invalid --run value (need KEY:METRICS[:LOG]): {line!r}")
    key = parts[0].strip()
    if not key:
        raise ValueError(f"Invalid --run (empty key): {line!r}")
    if len(parts) == 2:
        return key, parts[1].strip(), None
    metrics_path = parts[1].strip()
    log_path = ":".join(parts[2:]).strip()
    return key, metrics_path, log_path or None


def load_experiment_runs(
    specs: Sequence[Tuple[str, str, Optional[str]]],
    label_overrides: Optional[Dict[str, str]] = None,
) -> List[ExperimentRun]:
    """Load metrics JSON (and infer or attach log paths) for each (key, metrics_path, log_override)."""
    label_over = label_overrides or {}
    runs: List[ExperimentRun] = []
    for key, metrics_path, log_override in specs:
        if not metrics_path or not os.path.isfile(metrics_path):
            raise SystemExit(f"error: metrics file not found for key {key!r}: {metrics_path}")
        log_path = log_override if log_override else infer_request_log_path(metrics_path)
        label = label_over.get(key) or DEFAULT_RUN_LABELS.get(key) or key
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        runs.append(
            ExperimentRun(
                key=key,
                label=label,
                metrics_path=os.path.abspath(metrics_path),
                log_path=os.path.abspath(log_path),
                metrics=metrics,
            )
        )
    return runs


def build_runs_from_args(args: argparse.Namespace) -> List[ExperimentRun]:
    label_over = parse_label_flags(args.label)

    if args.run:
        specs = [parse_run_arg(x) for x in args.run]
    else:
        missing = [
            n
            for n, p in (
                ("--metrics-nocache", args.metrics_nocache),
                ("--metrics-gptcache", args.metrics_gptcache),
                ("--metrics-contextcache", args.metrics_contextcache),
            )
            if not p
        ]
        if missing:
            raise SystemExit(
                "error: provide either --run KEY:METRICS[:LOG] (repeatable) or all three legacy flags: "
                + ", ".join(
                    [
                        "--metrics-nocache",
                        "--metrics-gptcache",
                        "--metrics-contextcache",
                    ]
                )
            )
        specs = [
            ("nocache", args.metrics_nocache, None),
            ("gptcache", args.metrics_gptcache, None),
            ("contextcache", args.metrics_contextcache, None),
        ]

    return load_experiment_runs(specs, label_over)


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


def plot_latency_compare_multi(runs: Sequence[ExperimentRun], output_dir: str, prefix: str) -> str:
    if not runs:
        raise ValueError("runs must be non-empty")
    rows0 = _sorted_per_app(runs[0].metrics)
    app_ids = [app_id for app_id, _ in rows0]

    def p99(m: Dict[str, Any]) -> List[float]:
        rows = _sorted_per_app(m)
        return [float(stats.get("p99_latency_ms", 0.0)) for _, stats in rows]

    fig, ax = plt.subplots(figsize=(12, 5))
    for run in runs:
        ax.plot(app_ids, p99(run.metrics), marker="o", label=f"{run.label} p99 (ms)")
    ax.set_title("P99 Latency by Application")
    ax.set_xlabel("Application ID")
    ax.set_ylabel("Latency p99 (ms)")
    ax.grid(alpha=0.3)
    ax.legend()
    return _save(fig, output_dir, f"{prefix}_latency_compare.png")


def _load_avg_latency_ms_from_log(log_path: str) -> Dict[int, float]:
    sums: Dict[int, float] = {}
    counts: Dict[int, int] = {}

    if not os.path.exists(log_path):
        return {}

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("ok") is not True:
                continue
            app_id = obj.get("application_id")
            latency_ms = obj.get("latency_ms")
            if app_id is None or latency_ms is None:
                continue
            app_id_int = int(app_id)
            sums[app_id_int] = sums.get(app_id_int, 0.0) + float(latency_ms)
            counts[app_id_int] = counts.get(app_id_int, 0) + 1

    out: Dict[int, float] = {}
    for app_id_int, total in sums.items():
        c = counts.get(app_id_int, 0)
        out[app_id_int] = (total / c) if c else 0.0
    return out


def plot_avg_latency_compare_multi(runs: Sequence[ExperimentRun], output_dir: str, prefix: str) -> str:
    if not runs:
        raise ValueError("runs must be non-empty")
    rows0 = _sorted_per_app(runs[0].metrics)
    app_ids = [app_id for app_id, _ in rows0]

    fig, ax = plt.subplots(figsize=(12, 5))
    for run in runs:
        avg = _load_avg_latency_ms_from_log(run.log_path)
        series = [float(avg.get(app_id, 0.0)) for app_id in app_ids]
        ax.plot(app_ids, series, marker="o", label=f"{run.label} avg (ms)")
    ax.set_title("Average Latency by Application (from request logs)")
    ax.set_xlabel("Application ID")
    ax.set_ylabel("Average latency (ms)")
    ax.grid(alpha=0.3)
    ax.legend()
    return _save(fig, output_dir, f"{prefix}_latency_avg_compare.png")


def plot_accuracy_compare_multi(runs: Sequence[ExperimentRun], output_dir: str, prefix: str) -> str:
    if not runs:
        raise ValueError("runs must be non-empty")
    rows0 = _sorted_per_app(runs[0].metrics)
    app_ids = [app_id for app_id, _ in rows0]

    def accuracy_metric_name(m: Dict[str, Any]) -> str:
        metric = str(m.get("slo_summary", {}).get("accuracy_metric", "avg_token_f1"))
        if metric not in {"exact_match_rate", "avg_token_f1", "avg_sequence_ratio"}:
            return "avg_token_f1"
        return metric

    chosen_metric = accuracy_metric_name(runs[-1].metrics)

    def acc(m: Dict[str, Any], metric: str) -> List[float]:
        rows = _sorted_per_app(m)
        return [float(stats.get(metric, 0.0)) for _, stats in rows]

    fig, ax = plt.subplots(figsize=(12, 5))
    for run in runs:
        ax.plot(
            app_ids,
            acc(run.metrics, chosen_metric),
            marker="o",
            label=f"{run.label} {chosen_metric}",
        )
    ax.set_title(f"Accuracy ({chosen_metric}) by Application")
    ax.set_xlabel("Application ID")
    ax.set_ylabel(chosen_metric)
    ax.grid(alpha=0.3)
    ax.legend()
    return _save(fig, output_dir, f"{prefix}_accuracy_compare.png")


def plot_slo_compare_multi(runs: Sequence[ExperimentRun], output_dir: str, prefix: str) -> Tuple[str, str]:
    if not runs:
        raise ValueError("runs must be non-empty")
    rows0 = _sorted_per_app(runs[0].metrics)
    app_ids = [app_id for app_id, _ in rows0]
    n = len(runs)
    bar_width = min(0.8 / max(n, 1), 0.28)
    offsets = [(i - (n - 1) / 2.0) * bar_width for i in range(n)]
    x = list(range(len(app_ids)))

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
                    margin = t - o
                except (TypeError, ValueError):
                    margin = 0.0
            margins.append(margin)
        return margins

    def accuracy_margins(m: Dict[str, Any]) -> List[float]:
        rows = _sorted_per_app(m)
        margins: List[float] = []
        accuracy_metric = str(m.get("slo_summary", {}).get("accuracy_metric", "avg_sequence_ratio"))
        for _, stats in rows:
            target = (stats.get("slo_target", {}) or {}).get("accuracy_slo")
            observed = (stats.get("slo_observed", {}) or {}).get(accuracy_metric)
            margin = 0.0
            if target is not None and observed is not None:
                try:
                    t = float(target)
                    o = float(observed)
                    margin = o - t
                except (TypeError, ValueError):
                    margin = 0.0
            margins.append(margin)
        return margins

    fig_lat, ax_lat = plt.subplots(figsize=(12, 5))
    ax_lat.axhline(0.0, color="#666666", linewidth=1, linestyle="--")
    for off, run in zip(offsets, runs):
        lat = latency_margins(run.metrics)
        ax_lat.bar([xi + off for xi in x], lat, width=bar_width, label=run.label)
    ax_lat.set_xticks(x)
    ax_lat.set_xticklabels(app_ids)
    ax_lat.set_xlabel("Application ID")
    ax_lat.set_ylabel("Latency SLO margin (target - observed, ms)")
    ax_lat.set_title("Latency SLO Margin by Application")
    ax_lat.grid(axis="y", alpha=0.3)
    ax_lat.legend()
    lat_path = _save(fig_lat, output_dir, f"{prefix}_latency_slo_compare.png")

    fig_acc, ax_acc = plt.subplots(figsize=(12, 5))
    ax_acc.axhline(0.0, color="#666666", linewidth=1, linestyle="--")
    for off, run in zip(offsets, runs):
        accm = accuracy_margins(run.metrics)
        ax_acc.bar([xi + off for xi in x], accm, width=bar_width, label=run.label)
    ax_acc.set_xticks(x)
    ax_acc.set_xticklabels(app_ids)
    ax_acc.set_xlabel("Application ID")
    ax_acc.set_ylabel("Accuracy SLO margin (observed - target)")
    ax_acc.set_title("Accuracy SLO Margin by Application")
    ax_acc.grid(axis="y", alpha=0.3)
    ax_acc.legend()
    acc_path = _save(fig_acc, output_dir, f"{prefix}_accuracy_slo_compare.png")

    return lat_path, acc_path


def write_comparison_plots(runs: Sequence[ExperimentRun], output_dir: str, prefix: str) -> List[str]:
    """Generate all comparison figures; returns list of written paths."""
    _ensure_output_dir(output_dir)
    paths: List[str] = []
    paths.append(plot_latency_compare_multi(runs, output_dir, prefix))
    paths.append(plot_avg_latency_compare_multi(runs, output_dir, prefix))
    paths.append(plot_accuracy_compare_multi(runs, output_dir, prefix))
    lat_slo, acc_slo = plot_slo_compare_multi(runs, output_dir, prefix)
    paths.extend([lat_slo, acc_slo])
    return paths


def main() -> None:
    args = parse_args()
    runs = build_runs_from_args(args)
    paths = write_comparison_plots(runs, args.output_dir, args.prefix)
    print("Wrote comparison plots:")
    for p in paths:
        print(f"- {p}")


if __name__ == "__main__":
    main()
