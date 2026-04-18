#!/usr/bin/env python3
"""
Plot an ablation-style latency breakdown from per-request JSONL logs.

Expected log rows include an optional dict field:
  timing_breakdown: { pre_ms, embed_query_ms, search_ms, embed_context_ms, ... }

This script aggregates successful requests and plots stacked bars comparing
LM=1 vs LM=10 for a single suffix (default: adaptive_load).
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_COMPONENTS: Tuple[str, ...] = (
    "pre_ms",
    "embed_query_ms",
    "search_ms",
    "embed_context_ms",
    "get_scalar_ms",
    "post_ms",
    "llm_ms",
    "save_ms",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot LM=1 vs LM=10 latency breakdown ablation from JSONL logs.")
    p.add_argument(
        "--root-dir",
        required=True,
        help="Root directory containing lm1/ and lm10/ subfolders (e.g. data/test_apps/load_mult_compare).",
    )
    p.add_argument(
        "--suffix",
        default="adaptive_load",
        help="Run suffix used in request_log_<suffix>.jsonl (default: adaptive_load).",
    )
    p.add_argument(
        "--components",
        default=",".join(DEFAULT_COMPONENTS),
        help="Comma-separated timing_breakdown keys to include in the stacked plot.",
    )
    p.add_argument(
        "--output-dir",
        default="plots/compare",
        help="Directory to write PNG output.",
    )
    p.add_argument(
        "--prefix",
        default="latency_breakdown_ablation",
        help="Filename prefix for generated PNG.",
    )
    return p.parse_args()


def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if v != v:  # NaN
        return None
    return v


def _load_breakdown_means(log_path: str, components: Sequence[str]) -> Tuple[Dict[str, float], int, int]:
    comp_sums: DefaultDict[str, float] = defaultdict(float)
    comp_counts: DefaultDict[str, int] = defaultdict(int)

    rows_seen = 0
    rows_with_breakdown = 0

    if not os.path.isfile(log_path):
        raise SystemExit(f"error: log not found: {log_path}")

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows_seen += 1
            obj = json.loads(line)
            if obj.get("ok") is not True:
                continue
            tb = obj.get("timing_breakdown")
            if not isinstance(tb, dict):
                continue
            rows_with_breakdown += 1

            for k in components:
                v = _safe_float(tb.get(k))
                if v is None:
                    continue
                comp_sums[k] += v
                comp_counts[k] += 1

    means: Dict[str, float] = {}
    for k in components:
        c = comp_counts.get(k, 0)
        means[k] = (comp_sums[k] / c) if c else 0.0

    return means, rows_with_breakdown, rows_seen


def _write_placeholder(out_path: str, title: str, body: str) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ax.axis("off")
    ax.set_title(title, pad=20)
    ax.text(
        0.5,
        0.5,
        body,
        ha="center",
        va="center",
        fontsize=12,
        wrap=True,
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    root = os.path.abspath(args.root_dir)
    suffix = str(args.suffix).strip()
    if not suffix:
        raise SystemExit("error: empty --suffix")

    components = [c.strip() for c in str(args.components).split(",") if c.strip()]
    if not components:
        raise SystemExit("error: empty --components")

    log_lm1 = os.path.join(root, "lm1", f"request_log_{suffix}.jsonl")
    log_lm10 = os.path.join(root, "lm10", f"request_log_{suffix}.jsonl")

    means1, n1, rows1 = _load_breakdown_means(log_lm1, components)
    means10, n10, rows10 = _load_breakdown_means(log_lm10, components)

    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.prefix}_{suffix}.png")

    if n1 == 0 and n10 == 0:
        msg = (
            "No per-request timing_breakdown found in logs.\n\n"
            f"lm1 log:  {log_lm1} (rows={rows1})\n"
            f"lm10 log: {log_lm10} (rows={rows10})\n\n"
            "Re-run experiments with the updated client/server so request logs include timing_breakdown."
        )
        _write_placeholder(
            out_path,
            title=f"Latency breakdown ablation ({suffix})",
            body=msg,
        )
        print(f"Wrote placeholder (no breakdown data): {out_path}")
        return

    groups = [f"LM=1\n(n={n1})", f"LM=10\n(n={n10})"]

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    cmap = plt.get_cmap("tab20")

    centers = [0.0, 1.0]
    means_list = [means1, means10]
    group_w = 0.72

    for center_x, means in zip(centers, means_list):
        bottom = 0.0
        for i, comp in enumerate(components):
            h = float(means.get(comp, 0.0))
            if h <= 0:
                continue
            ax.bar(
                [center_x],
                [h],
                width=group_w,
                bottom=[bottom],
                color=cmap(i % 20),
                edgecolor="black",
                linewidth=0.25,
                label=comp,
            )
            ax.text(
                center_x,
                bottom + h / 2.0,
                f"{h:.0f}",
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )
            bottom += h
        ax.text(center_x, bottom, f"{bottom:.1f}", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(centers)
    ax.set_xticklabels(groups, rotation=0, ha="center")
    ax.set_title(f"Latency breakdown ablation ({suffix}): mean timing_breakdown components")
    ax.set_ylabel("Mean time (ms)")
    ax.grid(axis="y", alpha=0.25)

    handles, labels = ax.get_legend_handles_labels()
    uniq: Dict[str, Any] = {}
    for h, lab in zip(handles, labels):
        uniq.setdefault(str(lab), h)
    ax.legend(
        list(uniq.values()),
        list(uniq.keys()),
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
