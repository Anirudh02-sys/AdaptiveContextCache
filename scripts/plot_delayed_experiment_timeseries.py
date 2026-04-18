#!/usr/bin/env python3
"""
Time-series plots for delayed load-multiplier experiments.

Reads per-request JSONL logs for multiple baselines (suffixes) under:
  <root-dir>/<lm-tag>/request_log_<suffix>.jsonl

For each baseline, aggregates *all applications together* (no per-app split) and plots:
  - latency vs elapsed time since the first logged request in that baseline
  - accuracy vs elapsed time (uses `accuracy.token_f1` when present; falls back to exact_match)

Each baseline becomes one smoothed line (raw points are shown faintly underneath).
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_LABELS: Dict[str, str] = {
    "nocache": "no-cache",
    "gptcache": "gptcache",
    "contextcache": "contextcache",
    "adaptive_load": "adaptive context (load)",
}


@dataclass(frozen=True)
class Series:
    suffix: str
    label: str
    t_s: np.ndarray  # seconds since series start
    y: np.ndarray


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot latency/accuracy vs time for delayed LM experiments.")
    p.add_argument(
        "--root-dir",
        required=True,
        help="Experiment root containing the LM_TAG folder (e.g. data/test_apps/load_mult_delayed_compare).",
    )
    p.add_argument(
        "--lm-tag",
        required=True,
        help="Subfolder name under root-dir (e.g. lm10_delayed240).",
    )
    p.add_argument(
        "--suffix",
        action="append",
        default=None,
        help="Baseline suffix to plot (repeatable). Default: nocache,gptcache,contextcache,adaptive_load.",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write PNG outputs.",
    )
    p.add_argument(
        "--prefix",
        default="delayed_timeseries",
        help="Filename prefix for generated PNGs.",
    )
    p.add_argument(
        "--smooth-window",
        type=int,
        default=51,
        help="Odd rolling-median window (requests). Use 1 to disable smoothing.",
    )
    p.add_argument(
        "--max-points",
        type=int,
        default=200_000,
        help="Safety cap on points read per baseline (default: 200000).",
    )
    return p.parse_args()


def _odd_window(n: int) -> int:
    n = int(n)
    if n <= 1:
        return 1
    if n % 2 == 0:
        return n + 1
    return n


def _rolling_nanmedian(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or y.size == 0:
        return y.astype(float)
    if y.size < window:
        window = max(1, (y.size // 2) * 2 + 1)
    pad = window // 2
    ypad = np.pad(y.astype(float), (pad, pad), mode="edge")
    out = np.empty_like(y, dtype=float)
    for i in range(y.size):
        seg = ypad[i : i + window]
        out[i] = float(np.nanmedian(seg))
    return out


def _read_series(log_path: str, max_points: int, include_accuracy: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ts: List[float] = []
    lat: List[float] = []
    acc: List[float] = []

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if len(ts) >= max_points:
                break
            obj = json.loads(line)
            if obj.get("ok") is not True:
                continue
            t = obj.get("timestamp")
            lm = obj.get("latency_ms")
            if t is None or lm is None:
                continue
            ts.append(float(t))
            lat.append(float(lm))

            if include_accuracy:
                aobj = obj.get("accuracy")
                a = None
                if isinstance(aobj, dict):
                    if "token_f1" in aobj:
                        a = aobj.get("token_f1")
                    elif "exact_match" in aobj:
                        a = aobj.get("exact_match")
                if a is None:
                    a = float("nan")
                acc.append(float(a))

    if not ts:
        return np.array([]), np.array([]), (np.array([]) if include_accuracy else np.array([]))

    order = np.argsort(np.asarray(ts, dtype=float))
    ts_arr = np.asarray(ts, dtype=float)[order]
    lat_arr = np.asarray(lat, dtype=float)[order]
    acc_arr = np.asarray(acc, dtype=float)[order] if include_accuracy else np.array([])

    t0 = float(ts_arr[0])
    t_rel = ts_arr - t0
    return t_rel, lat_arr, acc_arr


def _plot_lines(
    series_list: Sequence[Series],
    title: str,
    ylabel: str,
    outfile: str,
    smooth_window: int,
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    cmap = plt.get_cmap("tab10")

    w = _odd_window(smooth_window)
    for idx, s in enumerate(series_list):
        color = cmap(idx % 10)
        if s.t_s.size == 0:
            continue
        ax.plot(s.t_s, s.y, color=color, alpha=0.12, linewidth=1.0)
        y_sm = _rolling_nanmedian(s.y, w)
        ax.plot(s.t_s, y_sm, color=color, linewidth=2.2, label=s.label)

    ax.set_title(title)
    ax.set_xlabel("Elapsed time since first request in run (s)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(loc="best", frameon=True)

    os.makedirs(os.path.dirname(os.path.abspath(outfile)) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = os.path.abspath(args.root_dir)
    lm_tag = str(args.lm_tag).strip()
    if not lm_tag:
        raise SystemExit("error: empty --lm-tag")

    suffixes = args.suffix or ["nocache", "gptcache", "contextcache", "adaptive_load"]

    base_dir = os.path.join(root, lm_tag)
    if not os.path.isdir(base_dir):
        raise SystemExit(f"error: experiment dir not found: {base_dir}")

    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    series_list: List[Series] = []
    for suf in suffixes:
        suf = str(suf).strip()
        if not suf:
            continue
        log_path = os.path.join(base_dir, f"request_log_{suf}.jsonl")
        if not os.path.isfile(log_path):
            print(f"[skip] missing log for suffix={suf}: {log_path}")
            continue
        t, lat, _acc = _read_series(log_path, max_points=int(args.max_points), include_accuracy=False)
        label = DEFAULT_LABELS.get(suf, suf)
        series_list.append(Series(suffix=suf, label=label, t_s=t, y=lat))

    if not series_list:
        raise SystemExit("error: no request logs found for provided suffixes")

    prefix = str(args.prefix).strip() or "delayed_timeseries"
    lat_path = os.path.join(out_dir, f"{prefix}_{lm_tag}_latency_timeseries.png")
    acc_path = os.path.join(out_dir, f"{prefix}_{lm_tag}_accuracy_timeseries.png")

    # Latency plot
    _plot_lines(
        series_list,
        title=f"Latency vs time ({lm_tag}) — all applications combined",
        ylabel="Latency (ms)",
        outfile=lat_path,
        smooth_window=args.smooth_window,
        ylim=None,
    )

    acc_series: List[Series] = []
    for suf in suffixes:
        suf = str(suf).strip()
        if not suf:
            continue
        log_path = os.path.join(base_dir, f"request_log_{suf}.jsonl")
        if not os.path.isfile(log_path):
            continue
        t, _lat, acc = _read_series(log_path, max_points=int(args.max_points), include_accuracy=True)
        label = DEFAULT_LABELS.get(suf, suf)
        acc_series.append(Series(suffix=suf, label=label, t_s=t, y=acc))

    _plot_lines(
        acc_series,
        title=f"Accuracy vs time ({lm_tag}) — token_f1 when available — all applications combined",
        ylabel="Accuracy (token_f1 / exact_match fallback)",
        outfile=acc_path,
        smooth_window=args.smooth_window,
        ylim=(0.0, 1.05),
    )

    print(f"Wrote: {lat_path}")
    print(f"Wrote: {acc_path}")


if __name__ == "__main__":
    main()
