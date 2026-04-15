#!/usr/bin/env python3
"""
Regenerate the same plots as scripts/run_experiments.sh from existing metrics JSON
(and optional request logs), without re-running the server or generate_requests.

Per-run plots (same as plot_request_metrics.py):
  <output-dir>/per_run/<KEY>/...

Comparison plots (same as compare_experiments.py):
  <output-dir>/compare/...

Optional presentation figures (same as run_experiments.sh when the script exists):
  <output-dir>/slo_workload/...

Examples:
  python3 scripts/plot_experiment_results.py \\
    --output-dir data/my_run/plots \\
    --run nocache:data/my_run/request_metrics_nocache.json \\
    --run gptcache:data/my_run/request_metrics_gptcache.json \\
    --run contextcache:data/my_run/request_metrics_contextcache.json

  # Only nocache + contextcache on the comparison charts (still need --run for paths):
  python3 scripts/plot_experiment_results.py \\
    --output-dir out/plots \\
    --include nocache,contextcache \\
    --run nocache:.../request_metrics_nocache.json \\
    --run gptcache:.../request_metrics_gptcache.json \\
    --run contextcache:.../request_metrics_contextcache.json

  # Explicit request log for average-latency comparison (otherwise inferred from filename):
  python3 scripts/plot_experiment_results.py \\
    --output-dir out/plots \\
    --run adaptive_load:.../request_metrics_adaptive_load.json:.../request_log_adaptive_load.jsonl

  python3 scripts/plot_experiment_results.py \\
    --output-dir out/plots \\
    --request-gen-config config/request_gen.example.json \\
    --run ...
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Set, Tuple

from compare_experiments import (
    load_experiment_runs,
    parse_label_flags,
    parse_run_arg,
    write_comparison_plots,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _parse_include(raw: Optional[str]) -> Optional[Set[str]]:
    if raw is None or not str(raw).strip():
        return None
    return {x.strip() for x in raw.split(",") if x.strip()}


def _filter_specs(
    specs: Sequence[Tuple[str, str, Optional[str]]],
    include: Optional[Set[str]],
) -> List[Tuple[str, str, Optional[str]]]:
    if include is None:
        return list(specs)
    unknown = include - {s[0] for s in specs}
    if unknown:
        known = ", ".join(sorted({s[0] for s in specs}))
        raise SystemExit(f"error: --include keys not present in --run: {sorted(unknown)} (known: {known})")
    order = [s[0] for s in specs]
    inc_list = [k for k in order if k in include]
    spec_by_key = {s[0]: s for s in specs}
    return [spec_by_key[k] for k in inc_list]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Plot per-run metrics, comparison charts, and optional SLO/workload figures "
            "from existing experiment outputs (same figures as run_experiments.sh)."
        )
    )
    p.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="KEY:METRICS_JSON[:LOG_JSONL]",
        help="Repeat per experiment. Log path is optional if metrics follow request_metrics_<s>.json naming.",
    )
    p.add_argument(
        "--include",
        default=None,
        help="Comma-separated subset of KEYs to include in plots. Default: all keys from --run.",
    )
    p.add_argument(
        "--label",
        action="append",
        default=None,
        metavar="KEY:Display name",
        help="Override legend label for KEY (repeatable). Passed through to comparison plots.",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Base directory for per_run/, compare/, and optional slo_workload/ subdirectories.",
    )
    p.add_argument(
        "--prefix",
        default="compare",
        help="Filename prefix for comparison plot PNGs (default: compare).",
    )
    p.add_argument(
        "--request-gen-config",
        default=None,
        help=(
            "Path to request generator JSON (e.g. config/request_gen.example.json). "
            "If scripts/visualize_app_slos_and_workload.py exists, writes SLO/workload figures "
            "under <output-dir>/slo_workload/."
        ),
    )
    p.add_argument(
        "--python",
        default=None,
        help="Python interpreter for subprocesses (default: same as this script).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo = _repo_root()
    py = args.python or sys.executable
    include = _parse_include(args.include)
    label_over = parse_label_flags(args.label)

    try:
        all_specs = [parse_run_arg(x) for x in args.run]
    except ValueError as e:
        raise SystemExit(f"error: {e}") from e

    filtered = _filter_specs(all_specs, include)
    if not filtered:
        raise SystemExit("error: after --include, no experiments remain.")

    runs = load_experiment_runs(filtered, label_over)

    out_base = Path(args.output_dir).resolve()
    per_run_dir = out_base / "per_run"
    compare_dir = out_base / "compare"
    slo_dir = out_base / "slo_workload"

    plot_script = repo / "scripts" / "plot_request_metrics.py"
    if not plot_script.is_file():
        raise SystemExit(f"error: missing {plot_script}")

    per_run_dir.mkdir(parents=True, exist_ok=True)
    for run in runs:
        sub = per_run_dir / run.key
        sub.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(
            [
                py,
                str(plot_script),
                "--metrics",
                run.metrics_path,
                "--output-dir",
                str(sub),
                "--prefix",
                run.key,
            ],
            cwd=str(repo),
        )

    compare_dir.mkdir(parents=True, exist_ok=True)
    paths = write_comparison_plots(runs, str(compare_dir), args.prefix)
    print("Comparison plots:")
    for pth in paths:
        print(f"- {pth}")

    vis_script = repo / "scripts" / "visualize_app_slos_and_workload.py"
    if args.request_gen_config:
        cfg = Path(args.request_gen_config)
        if not cfg.is_file():
            raise SystemExit(f"error: --request-gen-config not found: {cfg}")
        if vis_script.is_file():
            slo_dir.mkdir(parents=True, exist_ok=True)
            prefix = cfg.stem
            subprocess.check_call(
                [
                    py,
                    str(vis_script),
                    "--config",
                    str(cfg.resolve()),
                    "--output-dir",
                    str(slo_dir),
                    "--prefix",
                    prefix,
                ],
                cwd=str(repo),
            )
            print(f"SLO / workload plots written under: {slo_dir}")
        else:
            print(
                f"Note: skipping SLO/workload figures ({vis_script.name} not found in this checkout).",
                file=sys.stderr,
            )
    print(f"Per-run plots under: {per_run_dir}")


if __name__ == "__main__":
    main()
