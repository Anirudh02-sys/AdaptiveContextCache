#!/usr/bin/env python3
"""
Compare experiment latency and accuracy using per-application averages.

This keeps the CLI shape of ``compare_experiments.py`` but aggregates each run
down to a single bar per metric by averaging across applications.
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
    """One experiment: stable key, label, metrics path, log path, and parsed metrics."""

    key: str
    label: str
    metrics_path: str
    log_path: str
    metrics: Dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare latency and accuracy across experiments using app-averaged bars."
    )
    parser.add_argument(
        "--run",
        action="append",
        default=None,
        metavar="KEY:METRICS_JSON[:LOG_JSONL]",
        help=(
            "Repeat for each experiment. KEY is a short id (e.g. contextcache_lm1). "
            "If LOG_JSONL is omitted, it is inferred from the metrics filename."
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


def _ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _save(fig: Any, output_dir: str, filename: str) -> str:
    out_path = os.path.join(output_dir, filename)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


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


def _mean(values: Sequence[float]) -> float:
    vals = [float(v) for v in values]
    return (sum(vals) / len(vals)) if vals else 0.0


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


def _accuracy_metric_name(metrics: Dict[str, Any]) -> str:
    metric = str(metrics.get("slo_summary", {}).get("accuracy_metric", "avg_token_f1"))
    if metric not in {"exact_match_rate", "avg_token_f1", "avg_sequence_ratio"}:
        return "avg_token_f1"
    return metric


def _average_latency_across_apps(run: ExperimentRun) -> float:
    avg_by_app = _load_avg_latency_ms_from_log(run.log_path)
    if avg_by_app:
        return _mean(list(avg_by_app.values()))
    rows = _sorted_per_app(run.metrics)
    if rows:
        return _mean([float(stats.get("served_p50_latency_ms", 0.0)) for _, stats in rows])
    return float(run.metrics.get("served_p50_latency_ms", 0.0))


def _average_accuracy_across_apps(run: ExperimentRun, metric: str) -> float:
    rows = _sorted_per_app(run.metrics)
    if rows:
        return _mean([float(stats.get(metric, 0.0)) for _, stats in rows])
    return float(run.metrics.get(metric, 0.0))


def _bar_plot(
    runs: Sequence[ExperimentRun],
    values: Sequence[float],
    title: str,
    ylabel: str,
    output_dir: str,
    filename: str,
) -> str:
    labels = [run.label for run in runs]
    x = list(range(len(runs)))
    fig_width = max(10.0, 1.8 * len(runs))
    fig, ax = plt.subplots(figsize=(fig_width, 5.5))
    bars = ax.bar(x, values, width=0.68)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.3)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{value:.3f}" if abs(value) < 10 else f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    return _save(fig, output_dir, filename)


def plot_avg_latency_bar_multi(runs: Sequence[ExperimentRun], output_dir: str, prefix: str) -> str:
    values = [_average_latency_across_apps(run) for run in runs]
    return _bar_plot(
        runs,
        values,
        "Average Latency Across Applications",
        "Average latency (ms)",
        output_dir,
        f"{prefix}_latency_avg_compare.png",
    )


def plot_avg_accuracy_bar_multi(runs: Sequence[ExperimentRun], output_dir: str, prefix: str) -> str:
    chosen_metric = _accuracy_metric_name(runs[-1].metrics if runs else {})
    values = [_average_accuracy_across_apps(run, chosen_metric) for run in runs]
    return _bar_plot(
        runs,
        values,
        f"Average Accuracy Across Applications ({chosen_metric})",
        chosen_metric,
        output_dir,
        f"{prefix}_accuracy_avg_compare.png",
    )


def write_comparison_plots(runs: Sequence[ExperimentRun], output_dir: str, prefix: str) -> List[str]:
    _ensure_output_dir(output_dir)
    return [
        plot_avg_latency_bar_multi(runs, output_dir, prefix),
        plot_avg_accuracy_bar_multi(runs, output_dir, prefix),
    ]


def main() -> None:
    args = parse_args()
    runs = build_runs_from_args(args)
    paths = write_comparison_plots(runs, args.output_dir, args.prefix)
    print("Wrote comparison plots:")
    for p in paths:
        print(f"- {p}")


if __name__ == "__main__":
    main()
