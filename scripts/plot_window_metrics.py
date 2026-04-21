import os
import argparse
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def read_metric_file(path: str) -> List[Tuple[str, float]]:
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) != 2:
                continue
            conv_id = parts[0].strip()
            score = float(parts[1].strip())
            rows.append((conv_id, score))
    return rows


def main():
    parser = argparse.ArgumentParser(description='Plot window metric files.')
    parser.add_argument('--metrics_dir', required=True, help='Folder containing window0.txt ... window5.txt')
    parser.add_argument('--output_dir', required=True, help='Folder to save image')
    parser.add_argument('--title', default='Window Evaluation Scores')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    data: Dict[int, List[Tuple[str, float]]] = {}
    all_conv_ids: List[str] = []

    for w in range(6):
        path = os.path.join(args.metrics_dir, f'window{w}.txt')
        rows = read_metric_file(path)
        data[w] = rows
        for conv_id, _ in rows:
            if conv_id not in all_conv_ids:
                all_conv_ids.append(conv_id)


    # filtered_conv_ids: List[str] = []

    # for conv_id in all_conv_ids:
    #     scores = []
    #     for w in range(6):
    #         score_map = {cid: score for cid, score in data[w]}
    #         if conv_id in score_map:
    #             scores.append(score_map[conv_id])

    #     if scores and (max(scores) - min(scores) > 1e-6):
    #         filtered_conv_ids.append(conv_id)
    # filtered_conv_ids = filtered_conv_ids[:10]
    x_positions = list(range(len(all_conv_ids)))
    # print(
    #     f"Showing {len(filtered_conv_ids)} differing "
    #     f"conversations out of {len(all_conv_ids)} total"
    # )

    # x_positions = list(range(len(filtered_conv_ids)))
    plt.figure(figsize=(14, 7))

    for w in range(6):
        rows = data[w]
        score_map = {cid: score for cid, score in rows}
        y_vals = [score_map.get(cid, None) for cid in all_conv_ids]
        # y_vals = [score_map.get(cid, None) for cid in filtered_conv_ids]
        plt.plot(x_positions, y_vals, marker='o', linewidth=2, label=f'window{w}')

    plt.xticks(x_positions, all_conv_ids, rotation=45, ha='right')
    # plt.xticks(
    #     ticks=range(0, len(all_conv_ids), max(1, len(all_conv_ids)//10))
    # )
    plt.xlabel('Conversation ID')
    # plt.xlabel('Conversations')

    # plt.xticks(
    #     ticks=range(0, len(filtered_conv_ids),max(1, len(filtered_conv_ids)//10))
    # )
    # plt.xlabel('Conversations with score differences')
    plt.ylabel('Similarity Score')
    plt.title(args.title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = os.path.join(args.output_dir, 'window_metrics_plot.png')
    # out_path = os.path.join(args.output_dir, 'window_metrics_diff_plot_6.png')
    plt.savefig(out_path, dpi=200)
    print(f'Saved: {out_path}')
# def compute_mean(values: List[float]) -> float:
#     if not values:
#         return 0.0
#     return sum(values) / len(values)


# def compute_median(values: List[float]) -> float:
#     if not values:
#         return 0.0

#     sorted_vals = sorted(values)
#     n = len(sorted_vals)
#     mid = n // 2

#     if n % 2 == 1:
#         return sorted_vals[mid]
#     return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0


# def main():
#     parser = argparse.ArgumentParser(description="Plot summary metrics for windows 0-5.")
#     parser.add_argument(
#         "--metrics_dir",
#         required=True,
#         help="Folder containing window0.txt ... window5.txt",
#     )
#     parser.add_argument(
#         "--output_dir",
#         required=True,
#         help="Folder to save output plots",
#     )
#     parser.add_argument(
#         "--title_prefix",
#         default="Window Evaluation",
#         help="Prefix for plot titles",
#     )
#     args = parser.parse_args()

#     os.makedirs(args.output_dir, exist_ok=True)

#     data: Dict[int, List[Tuple[str, float]]] = {}
#     score_maps: Dict[int, Dict[str, float]] = {}

#     for w in range(6):
#         path = os.path.join(args.metrics_dir, f"window{w}.txt")
#         rows = read_metric_file(path)
#         data[w] = rows
#         score_maps[w] = {conv_id: score for conv_id, score in rows}

#     all_conv_ids = sorted(
#         set().union(*[set(score_maps[w].keys()) for w in range(6)])
#     )

#     mean_scores: List[float] = []
#     median_scores: List[float] = []
#     win_counts: List[int] = [0] * 6
#     mean_improvements: List[float] = []

#     for w in range(6):
#         vals = [score_maps[w][cid] for cid in all_conv_ids if cid in score_maps[w]]
#         mean_scores.append(compute_mean(vals))
#         median_scores.append(compute_median(vals))

#     for cid in all_conv_ids:
#         available = []
#         for w in range(6):
#             if cid in score_maps[w]:
#                 available.append((w, score_maps[w][cid]))

#         if not available:
#             continue

#         best_score = max(score for _, score in available)

#         for w, score in available:
#             if score == best_score:
#                 win_counts[w] += 1

#     baseline_map = score_maps[0]

#     for w in range(6):
#         improvements = []

#         for cid in all_conv_ids:
#             if cid in baseline_map and cid in score_maps[w]:
#                 improvements.append(score_maps[w][cid] - baseline_map[cid])

#         mean_improvements.append(compute_mean(improvements))

#     window_labels = [f"window{w}" for w in range(6)]
#     x_positions = list(range(6))

#     print("Mean scores by window:")
#     for w in range(6):
#         print(f"window{w}: {mean_scores[w]:.6f}")

#     print("\nMedian scores by window:")
#     for w in range(6):
#         print(f"window{w}: {median_scores[w]:.6f}")

#     print("\nWin counts by window:")
#     for w in range(6):
#         print(f"window{w}: {win_counts[w]}")

#     print("\nMean improvement over window0:")
#     for w in range(6):
#         print(f"window{w}: {mean_improvements[w]:.6f}")

#     best_mean_window = max(range(6), key=lambda w: mean_scores[w])
#     best_median_window = max(range(6), key=lambda w: median_scores[w])
#     best_win_window = max(range(6), key=lambda w: win_counts[w])
#     best_improvement_window = max(range(6), key=lambda w: mean_improvements[w])

#     print(f"\nBest mean score: window{best_mean_window}")
#     print(f"Best median score: window{best_median_window}")
#     print(f"Most wins: window{best_win_window}")
#     print(f"Best mean improvement over window0: window{best_improvement_window}")

#     plt.figure(figsize=(8, 5))
#     plt.bar(x_positions, mean_scores)
#     plt.xticks(x_positions, window_labels)
#     plt.xlabel("Window")
#     plt.ylabel("Mean Similarity Score")
#     plt.title(f"{args.title_prefix}: Mean Score by Window")
#     plt.tight_layout()
#     out_path = os.path.join(args.output_dir, "mean_score_by_window.png")
#     plt.savefig(out_path, dpi=200)
#     print(f"Saved: {out_path}")
#     plt.close()

#     plt.figure(figsize=(8, 5))
#     plt.bar(x_positions, median_scores)
#     plt.xticks(x_positions, window_labels)
#     plt.xlabel("Window")
#     plt.ylabel("Median Similarity Score")
#     plt.title(f"{args.title_prefix}: Median Score by Window")
#     plt.tight_layout()
#     out_path = os.path.join(args.output_dir, "median_score_by_window.png")
#     plt.savefig(out_path, dpi=200)
#     print(f"Saved: {out_path}")
#     plt.close()

#     plt.figure(figsize=(8, 5))
#     plt.bar(x_positions, win_counts)
#     plt.xticks(x_positions, window_labels)
#     plt.xlabel("Window")
#     plt.ylabel("Win Count")
#     plt.title(f"{args.title_prefix}: Win Count by Window")
#     plt.tight_layout()
#     out_path = os.path.join(args.output_dir, "win_count_by_window.png")
#     plt.savefig(out_path, dpi=200)
#     print(f"Saved: {out_path}")
#     plt.close()

#     plt.figure(figsize=(8, 5))
#     plt.bar(x_positions, mean_improvements)
#     plt.xticks(x_positions, window_labels)
#     plt.xlabel("Window")
#     plt.ylabel("Mean Improvement vs window0")
#     plt.title(f"{args.title_prefix}: Mean Improvement over window0")
#     plt.tight_layout()
#     out_path = os.path.join(args.output_dir, "mean_improvement_over_window0.png")
#     plt.savefig(out_path, dpi=200)
#     print(f"Saved: {out_path}")
#     plt.close()



if __name__ == '__main__':
    main()