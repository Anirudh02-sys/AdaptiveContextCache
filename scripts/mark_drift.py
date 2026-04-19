import argparse
from typing import Any, Dict, List, Sequence

import numpy as np

from drift_utils import Embedder, load_jsonl, write_jsonl, cosine_sim, make_history_unit

"""
This script marks each conversation as high, med, low drift and 
gives the indices of the turns that have drift over the threshold.

If  turn_pairs = [{"user": q1, "assistant": a1},{"user": q2, "assistant": a2}]
then each unit is like   "Previous user question: q1\nPrevious model response: a1"

1. embeds the units
2. computes drift = 1 -cos sim(current unit, mean of previous units)
3. mark the turns in the turn_pairs with drift_score >= drift_threshold
4. keep only the eligible drift turns for the later experiments 
    if w=0 means no history so query only 
    w=1 means 1 previous turn 
    so w=5 means 5 previous turns, and we toss the conversations that have drift index less than 5
5.buecketize the conversation into high, med, low based on the max_drift_score in the turns
6. bucketization is using p33 and p66
7. Save 3 files: full annotated file, low drift eligible conversations, high drift eligible conversations
"""

def compute_drift(
    turn_pairs: List[Dict[str, Any]],
    embedder: Embedder,
    drift_threshold: float,
    max_history_units: int = 5,
    min_eligible_turn_idx: int = 5,
    max_user_chars: int = 1500,
) -> Dict[str, Any]:
    if not turn_pairs:
        return {
            "drift_scores": [],
            "drift_turn_idxs": [],
            "eligible_drift_turn_idxs": [],
            "has_eligible_drift_turn": False,
            "max_drift_score": 0.0,
            "turn_of_max_drift": None,
        }

    turn_units = [make_history_unit(tp) for tp in turn_pairs]
    embeddings = embedder.encode(turn_units)

    drift_scores: List[float] = []

    for t in range(len(embeddings)):
        if t == 0:
            drift_scores.append(0.0)
            continue

        history = embeddings[:t]
        if max_history_units > 0:
            history = history[-max_history_units:]

        history_mean = np.mean(history, axis=0)
        current = embeddings[t]

        sim = cosine_sim(current, history_mean)
        drift = 1.0 - sim
        drift_scores.append(float(drift))

        # if sim =0.55, drift =0.45
        # so threshold is set at drift >= 0.45 means sim <= 0.55)

    drift_turn_idxs = [
        i for i, score in enumerate(drift_scores)
        if i > 0 and score >= drift_threshold
    ]
    eligible_drift_turn_idxs = [
        t for t in drift_turn_idxs
        if t >= min_eligible_turn_idx
    ]


    drift_turn_numbers = [i + 1 for i in drift_turn_idxs]
    drift_turn_queries = [
        turn_pairs[i].get("user", "")
        for i in drift_turn_idxs
    ]

    if drift_scores:
        turn_of_max_drift = int(np.argmax(drift_scores))
        max_drift_score = float(drift_scores[turn_of_max_drift])
    else:
        turn_of_max_drift = None
        max_drift_score = 0.0

    return {
        "drift_scores": drift_scores,
        "drift_turn_idxs": drift_turn_idxs,
        "eligible_drift_turn_idxs": eligible_drift_turn_idxs,
        "has_eligible_drift_turn": len(eligible_drift_turn_idxs) > 0,
        "max_drift_score": max_drift_score,
        "turn_of_max_drift": turn_of_max_drift,
    }


def bucketize(values: Sequence[float]) -> List[str]:
    if not values:
        return []
    arr = np.asarray(values, dtype=np.float32)
    p33, p66 = np.percentile(arr, [33, 66]).tolist()
    labels: List[str] = []
    for v in values:
        if v <= p33:
            labels.append("low")
        elif v <= p66:
            labels.append("medium")
        else:
            labels.append("high")
    return labels


# conversation_drift_score = [0.1, 0.2, 0.8]
#bucketize(conversation_drift_score) -> ["low", "low", "high"]


def filter_bucket_rows(
    rows: List[Dict[str, Any]],
    target_bucket: str,
) -> List[Dict[str, Any]]:
    output_rows: List[Dict[str, Any]] = []

    for row in rows:
        if row.get("conversation_drift_bucket") != target_bucket:
            continue
        if not row.get("has_eligible_drift_turn", False):
            continue
        output_rows.append(row)

    return output_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="mark conversations with drift scores.")
    parser.add_argument("--input_file", required=True, help="path to input jsonl file")

    parser.add_argument("--output_file", required=True)
    parser.add_argument("--low_output_file", required=True)
    parser.add_argument("--high_output_file", required=True)
    
    parser.add_argument("--model_name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch_size",type=int, default=128)
    parser.add_argument("--max_history_units", type=int, default=5, help="Max number of previous turns to use when computing drift")
    parser.add_argument("--drift_threshold", type=float, default=0.45, help="Threshold for marking a turn as drifted")
    parser.add_argument("--min_eligible_turn_idx",type=int, default=5, help="min drift turn index needed for window experiment. Windows 0 to 5, use 5.",)
    args = parser.parse_args()

    rows = load_jsonl(args.input_file)
    embedder = Embedder(
        model_name=args.model_name,
        batch_size=args.batch_size,
    )

    output_rows: List[Dict[str, Any]] = []

    for row in rows:
        turn_pairs = row.get("turn_pairs", [])

        drift_result = compute_drift(
            turn_pairs=turn_pairs,
            embedder=embedder,
            drift_threshold=args.drift_threshold,
            max_history_units=args.max_history_units,
            min_eligible_turn_idx=args.min_eligible_turn_idx,
        )

        output_row = dict(row)
        output_row["drift_scores"] = drift_result["drift_scores"]
        output_row["drift_turn_idxs"] = drift_result["drift_turn_idxs"]
        output_row["eligible_drift_turn_idxs"] = drift_result["eligible_drift_turn_idxs"]
        output_row["has_eligible_drift_turn"] = drift_result["has_eligible_drift_turn"]
        output_row["max_drift_score"] = drift_result["max_drift_score"]
        output_row["turn_of_max_drift"] = drift_result["turn_of_max_drift"]

        output_rows.append(output_row)

    max_drift_scores = [r["max_drift_score"] for r in output_rows]
    drift_buckets = bucketize(max_drift_scores)

    for row, bucket in zip(output_rows, drift_buckets):
        row["conversation_drift_bucket"] = bucket

    write_jsonl(args.output_file, output_rows)

    low_rows = filter_bucket_rows(
        rows=output_rows,
        target_bucket="low",
    )
    high_rows = filter_bucket_rows(
        rows=output_rows,
        target_bucket="high",
    )

    write_jsonl(args.low_output_file, low_rows)
    write_jsonl(args.high_output_file, high_rows)

    print(f"Wrote full annotated file to {args.output_file}")
    print(f"Wrote {len(low_rows)} low-drift eligible conversations to {args.low_output_file}")
    print(f"Wrote {len(high_rows)} high-drift eligible conversations to {args.high_output_file}")

    

if __name__ == "__main__":
    main()