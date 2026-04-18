#!/usr/bin/env python3
"""
Compare experiment latency and accuracy using per-application averages.

This keeps the CLI shape of ``compare_experiments.py`` but aggregates each run
down to a single bar per metric by averaging across applications.

Also writes ``{prefix}_avg_metrics_row.png``: one row (accuracy, then latency) with the
same charts as the standalone PNGs, for a single combined figure.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from matplotlib.axes import Axes

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
    baseline_key: str = ""
    load_multiplier: Optional[int] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare latency and accuracy across experiments using app-averaged bars."
    )
    parser.add_argument(
        "--root-dir",
        default=None,
        help=(
            "Root directory containing lm1/ and lm10/ subfolders with request_metrics_*.json "
            "(e.g. data/test_apps/load_mult_compare). When set, use --suffix to select baselines."
        ),
    )
    parser.add_argument(
        "--suffix",
        action="append",
        default=None,
        metavar="SUFFIX",
        help=(
            "Baseline suffix to compare when using --root-dir "
            "(repeatable; examples: nocache, gptcache, contextcache, adaptive_load)."
        ),
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


def _load_mult_runs_from_root(
    root_dir: str,
    suffixes: Sequence[str],
    label_overrides: Optional[Dict[str, str]] = None,
    load_multipliers: Sequence[int] = (1, 10),
) -> List[ExperimentRun]:
    root = os.path.abspath(root_dir)
    label_over = label_overrides or {}
    runs: List[ExperimentRun] = []
    for suffix in suffixes:
        suffix = str(suffix).strip()
        if not suffix:
            continue
        for lm in load_multipliers:
            metrics_path = os.path.join(root, f"lm{lm}", f"request_metrics_{suffix}.json")
            if not os.path.isfile(metrics_path):
                raise SystemExit(f"error: metrics file not found: {metrics_path}")
            log_path = infer_request_log_path(metrics_path)
            key = f"{suffix}_lm{lm}"
            label = label_over.get(suffix) or DEFAULT_RUN_LABELS.get(suffix) or suffix
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            runs.append(
                ExperimentRun(
                    key=key,
                    label=label,
                    metrics_path=metrics_path,
                    log_path=log_path,
                    metrics=metrics,
                    baseline_key=suffix,
                    load_multiplier=int(lm),
                )
            )
    return runs


def build_runs_from_args(args: argparse.Namespace) -> List[ExperimentRun]:
    label_over = parse_label_flags(args.label)

    if args.root_dir:
        suffixes = args.suffix or []
        if not suffixes:
            raise SystemExit("error: --root-dir requires at least one --suffix")
        return _load_mult_runs_from_root(args.root_dir, suffixes, label_overrides=label_over)

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


def _grouped_bar_baselines(runs: Sequence[ExperimentRun]) -> List[str]:
    baselines: List[str] = []
    seen = set()
    for r in runs:
        b = r.baseline_key or r.key
        if b not in seen:
            baselines.append(b)
            seen.add(b)
    return baselines


def _grouped_bar_geometry(
    baselines: Sequence[str], load_multipliers: Sequence[int]
) -> Tuple[List[int], List[int], float, float, Dict[str, Any], Dict[int, Dict[str, Any]]]:
    lm_list = [int(x) for x in load_multipliers]
    series_count = len(lm_list)
    group_x = list(range(len(baselines)))
    group_width = 0.76
    bar_w = group_width / max(1, series_count)
    cmap = plt.get_cmap("tab10")
    baseline_color: Dict[str, Any] = {b: cmap(i % 10) for i, b in enumerate(baselines)}
    lm_styles = {
        lm_list[0]: {"alpha": 0.75, "hatch": ""},
        lm_list[1] if len(lm_list) > 1 else lm_list[0]: {"alpha": 1.0, "hatch": "//"},
    }
    return lm_list, group_x, group_width, bar_w, baseline_color, lm_styles


def _draw_grouped_bars_on_ax(
    ax: Axes,
    baselines: Sequence[str],
    value_by_key: Dict[str, float],
    title: str,
    ylabel: str,
    load_multipliers: Sequence[int] = (1, 10),
    *,
    show_legend: bool = True,
    value_fontsize: float = 8.5,
) -> None:
    lm_list, group_x, group_width, bar_w, baseline_color, lm_styles = _grouped_bar_geometry(
        baselines, load_multipliers
    )
    for j, lm in enumerate(lm_list):
        offsets = [gx - (group_width / 2.0) + (j + 0.5) * bar_w for gx in group_x]
        vals = []
        for b in baselines:
            k = f"{b}_lm{lm}"
            vals.append(float(value_by_key.get(k, 0.0)))
        style = lm_styles.get(lm, {"alpha": 1.0, "hatch": ""})
        bars = ax.bar(
            offsets,
            vals,
            width=bar_w * 0.92,
            color=[baseline_color[b] for b in baselines],
            edgecolor="black",
            linewidth=0.3,
            alpha=style["alpha"],
            hatch=style["hatch"],
            label=f"LM={lm}",
        )
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{v:.3f}" if abs(v) < 10 else f"{v:.1f}",
                ha="center",
                va="bottom",
                fontsize=value_fontsize,
                rotation=0,
            )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(list(group_x))
    ax.set_xticklabels([DEFAULT_RUN_LABELS.get(b, b) for b in baselines], rotation=15, ha="right")
    ax.grid(axis="y", alpha=0.25)
    if show_legend:
        ax.legend(loc="upper left", frameon=True)


def _grouped_bar_plot_by_baseline(
    runs: Sequence[ExperimentRun],
    value_by_key: Dict[str, float],
    title: str,
    ylabel: str,
    output_dir: str,
    filename: str,
    load_multipliers: Sequence[int] = (1, 10),
) -> str:
    baselines = _grouped_bar_baselines(runs)
    fig_width = max(10.5, 1.7 * len(baselines))
    fig, ax = plt.subplots(figsize=(fig_width, 5.8))
    _draw_grouped_bars_on_ax(
        ax,
        baselines,
        value_by_key,
        title,
        ylabel,
        load_multipliers=load_multipliers,
        show_legend=True,
    )
    return _save(fig, output_dir, filename)


def plot_avg_metrics_row_grouped(
    runs: Sequence[ExperimentRun],
    output_dir: str,
    prefix: str,
    load_multipliers: Sequence[int] = (1, 10),
) -> str:
    """One row: same grouped charts as ``*_accuracy_avg_compare`` then ``*_latency_avg_compare`` (left to right)."""
    chosen_metric = _accuracy_metric_name(runs[-1].metrics if runs else {})
    baselines = _grouped_bar_baselines(runs)
    panel_w = max(3.2, 1.15 * len(baselines))
    fig, axes = plt.subplots(1, 2, figsize=(panel_w * 2, 5.9), sharey=False)
    ax_acc, ax_lat = axes

    val_avg_lat = {r.key: _average_latency_across_apps(r) for r in runs}
    val_acc = {r.key: _average_accuracy_across_apps(r, chosen_metric) for r in runs}

    _draw_grouped_bars_on_ax(
        ax_acc,
        baselines,
        val_acc,
        f"Average Accuracy Across Applications ({chosen_metric})",
        chosen_metric,
        load_multipliers=load_multipliers,
        show_legend=True,
        value_fontsize=8.0,
    )
    _draw_grouped_bars_on_ax(
        ax_lat,
        baselines,
        val_avg_lat,
        "Average Latency Across Applications",
        "Average latency (ms)",
        load_multipliers=load_multipliers,
        show_legend=False,
        value_fontsize=8.0,
    )

    fig.suptitle("Average accuracy and latency (app-averaged)", y=1.02, fontsize=12)
    fig.tight_layout()
    out_path = os.path.join(output_dir, f"{prefix}_avg_metrics_row.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _simple_bars_on_ax(
    ax: Axes,
    runs: Sequence[ExperimentRun],
    values: Sequence[float],
    title: str,
    ylabel: str,
    *,
    value_fontsize: float = 8.0,
) -> None:
    labels = [run.label for run in runs]
    x = list(range(len(runs)))
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
            fontsize=value_fontsize,
        )


def plot_avg_metrics_row_simple(runs: Sequence[ExperimentRun], output_dir: str, prefix: str) -> str:
    """Non-``--root-dir`` runs: one row matching the two standalone bar charts."""
    chosen_metric = _accuracy_metric_name(runs[-1].metrics if runs else {})
    fig, axes = plt.subplots(1, 2, figsize=(max(14.0, 3.6 * len(runs)), 5.6), sharey=False)
    ax_acc, ax_lat = axes

    _simple_bars_on_ax(
        ax_acc,
        runs,
        [_average_accuracy_across_apps(r, chosen_metric) for r in runs],
        f"Average Accuracy Across Applications ({chosen_metric})",
        chosen_metric,
        value_fontsize=8.5,
    )
    _simple_bars_on_ax(
        ax_lat,
        runs,
        [_average_latency_across_apps(r) for r in runs],
        "Average Latency Across Applications",
        "Average latency (ms)",
        value_fontsize=8.5,
    )

    fig.suptitle("Average accuracy and latency (app-averaged)", y=1.02, fontsize=12)
    fig.tight_layout()
    out_path = os.path.join(output_dir, f"{prefix}_avg_metrics_row.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_avg_latency_bar_multi(runs: Sequence[ExperimentRun], output_dir: str, prefix: str) -> str:
    # If runs came from --root-dir, we have baseline_key and load_multiplier populated.
    if runs and all(r.baseline_key and r.load_multiplier is not None for r in runs):
        value_by_key = {r.key: _average_latency_across_apps(r) for r in runs}
        return _grouped_bar_plot_by_baseline(
            runs,
            value_by_key,
            "Average Latency Across Applications",
            "Average latency (ms)",
            output_dir,
            f"{prefix}_latency_avg_compare.png",
            load_multipliers=(1, 10),
        )
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
    if runs and all(r.baseline_key and r.load_multiplier is not None for r in runs):
        value_by_key = {r.key: _average_accuracy_across_apps(r, chosen_metric) for r in runs}
        return _grouped_bar_plot_by_baseline(
            runs,
            value_by_key,
            f"Average Accuracy Across Applications ({chosen_metric})",
            chosen_metric,
            output_dir,
            f"{prefix}_accuracy_avg_compare.png",
            load_multipliers=(1, 10),
        )
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
    paths: List[str] = [
        plot_avg_latency_bar_multi(runs, output_dir, prefix),
        plot_avg_accuracy_bar_multi(runs, output_dir, prefix),
    ]
    if not runs:
        return paths
    if all(r.baseline_key and r.load_multiplier is not None for r in runs):
        paths.append(plot_avg_metrics_row_grouped(runs, output_dir, prefix, load_multipliers=(1, 10)))
    else:
        paths.append(plot_avg_metrics_row_simple(runs, output_dir, prefix))
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
