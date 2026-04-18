#!/usr/bin/env python3
"""
Time-series plots for delayed load-multiplier experiments.

Reads per-request JSONL logs for multiple baselines under:
  <root-dir>/<lm-tag>/request_log_<suffix>.jsonl

Plots:
  - A column (stacked) time-series figure from adaptive server logs with aligned x-axis:
      1) effective context window,
      2) requests/minute window count,
      3) tokens/minute window count.
    The figure includes a vertical line when load_multiplier engages — timing and LM value
    from the run's request_metrics JSON (same keys as request_gen config:
    load_multiplier_start_seconds, load_multiplier).
  - Accuracy vs elapsed time (token_f1 when present; falls back to exact_match).

Latency vs time is optional (--plot-latency); the default emphasizes window dynamics.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
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

# load_adaptive: ... effective window 5 → 4 (overall_factor=...)
_RE_EFFECTIVE_WINDOW = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d+).*effective window (\d+) → (\d+)"
)
_RE_LOAD_COUNTS = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d+).*load_adaptive:.*req\s+(\d+)\s*(?:→|->)\s*(\d+),\s*tok\s+(\d+)\s*(?:→|->)\s*(\d+)"
)


@dataclass(frozen=True)
class Series:
    suffix: str
    label: str
    t_s: np.ndarray  # seconds since series start
    y: np.ndarray


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot window / optional latency / accuracy vs time for delayed LM experiments."
    )
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
        help="Baseline suffix for accuracy plots (repeatable). Default: nocache,gptcache,contextcache,adaptive_load.",
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
        "--window-suffix",
        default="adaptive_load",
        help="Suffix whose server log is parsed for load_adaptive window events (default: adaptive_load).",
    )
    p.add_argument(
        "--metrics-json",
        default=None,
        help=(
            "Run metrics JSON with load_multiplier_start_seconds and load_multiplier "
            "(default: <root-dir>/<lm-tag>/request_metrics_<window-suffix>.json)."
        ),
    )
    p.add_argument(
        "--server-log",
        default=None,
        help="Override path to server stderr log (default: <lm-tag>/server_<window-suffix>.log).",
    )
    p.add_argument(
        "--plot-latency",
        action="store_true",
        help="Also write latency vs time (all suffixes), in addition to window and accuracy.",
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


def _parse_log_line_ts_from_match(m: re.Match) -> float:
    base = m.group(1)
    ms_str = m.group(2)
    ms = int(ms_str[:3]) if len(ms_str) >= 3 else int((ms_str + "000")[:3])
    dt = datetime.strptime(base, "%Y-%m-%d %H:%M:%S") + timedelta(milliseconds=ms)
    return float(dt.timestamp())


def _parse_window_events(server_log_path: str) -> List[Tuple[float, int, int]]:
    """Return list of (unix_ts, w_from, w_to) for each effective window transition."""
    out: List[Tuple[float, int, int]] = []
    if not os.path.isfile(server_log_path):
        return out
    with open(server_log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = _RE_EFFECTIVE_WINDOW.match(line.strip())
            if not m:
                continue
            ts = _parse_log_line_ts_from_match(m)
            w0 = int(m.group(3))
            w1 = int(m.group(4))
            out.append((ts, w0, w1))
    out.sort(key=lambda x: x[0])
    return out


def _parse_load_minute_points(server_log_path: str) -> List[Tuple[float, int, int, int, int]]:
    """Return list of (unix_ts, prev_req, curr_req, prev_tok, curr_tok) for load_adaptive evaluations."""
    out: List[Tuple[float, int, int, int, int]] = []
    if not os.path.isfile(server_log_path):
        return out
    with open(server_log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = _RE_LOAD_COUNTS.match(line.strip())
            if not m:
                continue
            ts = _parse_log_line_ts_from_match(m)
            out.append(
                (
                    ts,
                    int(m.group(3)),
                    int(m.group(4)),
                    int(m.group(5)),
                    int(m.group(6)),
                )
            )
    out.sort(key=lambda x: x[0])
    return out


def _read_request_time_and_token_est(log_path: str, max_points: int) -> Tuple[np.ndarray, np.ndarray]:
    """Read per-request relative timestamps and token estimate from request log."""
    ts: List[float] = []
    tok_est: List[float] = []
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
            if t is None:
                continue
            ts.append(float(t))
            if obj.get("input_tokens") is not None:
                try:
                    tok_est.append(float(obj.get("input_tokens")))
                    continue
                except (TypeError, ValueError):
                    pass
            # Fallback for older logs: estimate token load from prompt words scaled by
            # request_content_count (proxy for context depth).
            prompt = str(obj.get("request_prompt", "") or "")
            prompt_tok = max(1.0, float(len(prompt.split())))
            content_n = max(1.0, float(obj.get("request_content_count", 1) or 1))
            tok_est.append(prompt_tok * content_n)
    if not ts:
        return np.array([]), np.array([])
    order = np.argsort(np.asarray(ts, dtype=float))
    ts_arr = np.asarray(ts, dtype=float)[order]
    tok_arr = np.asarray(tok_est, dtype=float)[order]
    t0 = float(ts_arr[0])
    return ts_arr - t0, tok_arr


def _rolling_minute_series(
    t_req_s: np.ndarray, token_est: np.ndarray, window_seconds: float = 60.0, sample_step_s: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Continuous rolling 60s request count and token sum sampled over time."""
    if t_req_s.size == 0:
        return np.array([]), np.array([]), np.array([])
    t_end = float(t_req_s[-1])
    if t_end <= 0:
        grid = np.array([0.0], dtype=float)
    else:
        n = max(1, int(np.floor(t_end / sample_step_s)) + 1)
        grid = np.linspace(0.0, t_end, n, dtype=float)

    req_pm = np.zeros_like(grid)
    tok_pm = np.zeros_like(grid)
    left = 0
    right = -1
    n_req = int(t_req_s.size)
    run_req = 0
    run_tok = 0.0
    for i, g in enumerate(grid):
        while right + 1 < n_req and t_req_s[right + 1] <= g:
            right += 1
            run_req += 1
            run_tok += float(token_est[right])
        cutoff = g - window_seconds
        while left <= right and t_req_s[left] < cutoff:
            run_req -= 1
            run_tok -= float(token_est[left])
            left += 1
        req_pm[i] = float(max(0, run_req))
        tok_pm[i] = float(max(0.0, run_tok))
    return grid, req_pm, tok_pm


def _first_request_t0_unix(request_log_path: str, max_points: int) -> Optional[float]:
    with open(request_log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("ok") is not True:
                continue
            t = obj.get("timestamp")
            if t is not None:
                return float(t)
            max_points -= 1
            if max_points <= 0:
                break
    return None


def _read_series(
    log_path: str, max_points: int, include_accuracy: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def _load_lm_from_metrics(metrics_path: str) -> Tuple[float, float]:
    """(load_multiplier_start_seconds, load_multiplier) from metrics JSON."""
    with open(metrics_path, "r", encoding="utf-8") as f:
        m = json.load(f)
    start = float(m.get("load_multiplier_start_seconds", 0.0))
    lm = float(m.get("load_multiplier", 1.0))
    return start, lm


def _plot_lines(
    series_list: Sequence[Series],
    title: str,
    ylabel: str,
    outfile: str,
    smooth_window: int,
    ylim: Optional[Tuple[float, float]] = None,
    vline_x: Optional[float] = None,
    vline_label: Optional[str] = None,
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

    if vline_x is not None and vline_x >= 0:
        ax.axvline(
            vline_x,
            color="red",
            linewidth=2.0,
            linestyle="-",
            label=vline_label or f"load multiplier engages ({vline_x:.0f} s)",
        )

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


def _plot_window_step(
    t_rel_s: np.ndarray,
    y_window: np.ndarray,
    t_load_s: np.ndarray,
    y_req_per_min: np.ndarray,
    y_tok_per_min: np.ndarray,
    outfile: str,
    title: str,
) -> None:
    fig, (ax0, ax1, ax2) = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(12.5, 10.5),
        sharex=True,
        gridspec_kw={"hspace": 0.08},
        constrained_layout=True,
    )
    if t_rel_s.size:
        ax0.step(
            t_rel_s,
            y_window,
            where="post",
            color="#1f77b4",
            linewidth=2.2,
            label="effective window (turns)",
        )
        ax0.fill_between(t_rel_s, y_window, step="post", alpha=0.12, color="#1f77b4")
    if t_load_s.size:
        ax1.plot(
            t_load_s,
            y_req_per_min,
            marker="o",
            markersize=3.0,
            linewidth=1.8,
            color="#2ca02c",
            label="requests in last 60s (rolling)",
        )
        ax2.plot(
            t_load_s,
            y_tok_per_min,
            marker="o",
            markersize=3.0,
            linewidth=1.8,
            color="#ff7f0e",
            label="tokens in last 60s (rolling, estimated)",
        )
    for ax in (ax0, ax1, ax2):
        ax.grid(alpha=0.25)
        ax.legend(loc="best", frameon=True)

    ax0.set_title(title)
    ax0.set_ylabel("Window (turns)")
    ax0.set_ylim(bottom=0)
    ax1.set_ylabel("Requests/min")
    ax1.set_ylim(bottom=0)
    ax2.set_ylabel("Tokens/min")
    ax2.set_xlabel("Elapsed time since first request in run (s)")
    ax2.set_ylim(bottom=0)

    os.makedirs(os.path.dirname(os.path.abspath(outfile)) or ".", exist_ok=True)
    fig.savefig(outfile, dpi=150)
    plt.close(fig)


def _build_window_step_series(
    events: List[Tuple[float, int, int]],
    t0_unix: float,
    experiment_end_rel: Optional[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Piecewise-constant effective window from transition events; time axis in seconds since t0."""
    if not events:
        return np.array([]), np.array([])

    w_initial = events[0][1]
    times: List[float] = [0.0]
    values: List[float] = [float(w_initial)]

    for ts, _w_from, w_to in events:
        rel = ts - t0_unix
        if rel < 0:
            continue
        times.append(rel)
        values.append(float(w_to))

    # extend to end of experiment if known
    if experiment_end_rel is not None and experiment_end_rel > times[-1]:
        times.append(float(experiment_end_rel))
        values.append(values[-1])

    return np.asarray(times, dtype=float), np.asarray(values, dtype=float)


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

    prefix = str(args.prefix).strip() or "delayed_timeseries"
    window_suffix = str(args.window_suffix).strip() or "adaptive_load"

    metrics_path = args.metrics_json or os.path.join(base_dir, f"request_metrics_{window_suffix}.json")
    if not os.path.isfile(metrics_path):
        raise SystemExit(f"error: metrics JSON not found (needed for LM line): {metrics_path}")

    lm_start_s, lm_value = _load_lm_from_metrics(metrics_path)

    server_log = args.server_log or os.path.join(base_dir, f"server_{window_suffix}.log")
    events = _parse_window_events(server_log)

    req_log_adaptive = os.path.join(base_dir, f"request_log_{window_suffix}.jsonl")
    t0_unix = _first_request_t0_unix(req_log_adaptive, max_points=int(args.max_points))
    if t0_unix is None:
        raise SystemExit(f"error: could not get first request timestamp from {req_log_adaptive}")

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics_obj = json.load(f)
    exp_dur = float(metrics_obj.get("experiment_duration_seconds", 0.0) or 0.0)
    experiment_end_rel = exp_dur if exp_dur > 0 else None

    t_w, y_w = _build_window_step_series(events, t0_unix, experiment_end_rel)
    if t_w.size == 0 and experiment_end_rel is not None:
        # No parseable transitions; still draw LM line and a flat reference (base window_len typical default).
        t_w = np.array([0.0, float(experiment_end_rel)], dtype=float)
        y_w = np.array([5.0, 5.0], dtype=float)

    window_path = os.path.join(out_dir, f"{prefix}_{lm_tag}_window_timeseries.png")
    note = "" if events else " (no window transitions matched in server log; flat line is illustrative)"
    title_win = (
        f"Window / requests-min / tokens-min vs time ({lm_tag}) — from server log ({window_suffix}){note}\n"
        f"load_multiplier_start_seconds={lm_start_s:g} s, load_multiplier={lm_value:g} (from metrics JSON)"
    )
    t_req, tok_est = _read_request_time_and_token_est(
        req_log_adaptive, max_points=int(args.max_points)
    )
    t_load, y_req, y_tok = _rolling_minute_series(t_req, tok_est, window_seconds=60.0, sample_step_s=1.0)
    _plot_window_step(
        t_w,
        y_w,
        t_load,
        y_req,
        y_tok,
        window_path,
        title_win,
    )
    print(f"Wrote: {window_path}")

    if args.plot_latency:
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

        if series_list:
            lat_path = os.path.join(out_dir, f"{prefix}_{lm_tag}_latency_timeseries.png")
            _plot_lines(
                series_list,
                title=f"Latency vs time ({lm_tag}) — all applications combined",
                ylabel="Latency (ms)",
                outfile=lat_path,
                smooth_window=args.smooth_window,
                ylim=None,
                vline_x=lm_start_s,
                vline_label=f"LM={lm_value:g} engages ({lm_start_s:.0f} s)",
            )
            print(f"Wrote: {lat_path}")

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

    acc_path = os.path.join(out_dir, f"{prefix}_{lm_tag}_accuracy_timeseries.png")
    _plot_lines(
        acc_series,
        title=f"Accuracy vs time ({lm_tag}) — token_f1 when available — all applications combined",
        ylabel="Accuracy (token_f1 / exact_match fallback)",
        outfile=acc_path,
        smooth_window=args.smooth_window,
        ylim=(0.0, 1.05),
        vline_x=lm_start_s,
        vline_label=f"LM={lm_value:g} engages ({lm_start_s:.0f} s)",
    )
    print(f"Wrote: {acc_path}")


if __name__ == "__main__":
    main()
