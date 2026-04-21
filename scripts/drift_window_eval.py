import argparse
import math
import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from drift_utils import Embedder, load_jsonl, make_history_unit, cosine_sim

"""
Stage1 uses the test query only. 
Compares test query embedding to all candidatate query embeddings
Retrieves the raw-top-k candidates based on sim

Stage2 uses the same attention mechanism from adapter
to 
1. build one vector for the previous turns + test query
2. build a vector for each candidate using the context_sequence from pickle
3. compares the vectors, reranks 
"""

# CONVERSATION_IDS: List[str] = [
#     "conv_0006102",
#     "conv_0002423",
#     "conv_0004008",
#     "conv_0001750",
#     "conv_0003249",
#     "conv_0003467"
# ]

CONVERSATION_IDS: List[str] = []


@dataclass
class CandidateEntry:
    conversation_id: Any
    pair_index: int
    query_text: str
    answer_text: str
    full_prior_context: List[Dict[str, Any]]
    query_embedding: np.ndarray
    context_sequence: np.ndarray


class RemapUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if name == "CandidateEntry":
            return CandidateEntry
        return super().find_class(module, name)


def load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return RemapUnpickler(f).load()


def write_text(path: str, text: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def semantic_similarity(a: str, b: str, embedder: Embedder) -> float:
    emb = embedder.encode([a, b])
    return cosine_sim(emb[0], emb[1])


def attention_pool_last_query(context_seq: np.ndarray) -> np.ndarray:
    seq = np.asarray(context_seq, dtype=np.float32)

    if seq.ndim != 2 or seq.shape[0] == 0:
        if seq.ndim == 2 and seq.shape[1] > 0:
            return np.zeros((seq.shape[1],), dtype=np.float32)
        return np.zeros((0,), dtype=np.float32)

    q = seq[-1]
    k = seq
    v = seq

    scale = float(math.sqrt(max(1, q.shape[0])))
    logits = (k @ q) / (scale + 1e-12)
    logits = logits - float(np.max(logits))
    weights = np.exp(logits)
    weights = weights / (float(np.sum(weights)) + 1e-12)

    return (weights[:, None] * v).sum(axis=0)


def retrieve_top_k_indices(
    query_embedding: np.ndarray,
    pool_matrix: np.ndarray,
    top_k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if pool_matrix.shape[0] == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

    scores = pool_matrix @ query_embedding
    top_k = min(top_k, scores.shape[0])
    top_indices = np.argsort(scores)[::-1][:top_k]
    top_scores = scores[top_indices]
    return top_indices, top_scores


def build_current_context_sequence(
    history_turn_pairs: List[Dict[str, Any]],
    current_query: str,
    embedder: Embedder,
    window_size: int,
) -> np.ndarray:
    history_units = [make_history_unit(tp) for tp in history_turn_pairs]

    if window_size > 0:
        history_units = history_units[-window_size:]
    else:
        history_units = []

    texts = list(history_units) + [current_query]
    return embedder.encode(texts).astype(np.float32)


def get_first_eligible_case(row: Dict[str, Any]) -> Dict[str, Any]:
    conversation_id = row.get("conversation_id", "")
    turn_pairs = row.get("turn_pairs", [])
    eligible = row.get("eligible_drift_turn_idxs", [])

    if not eligible:
        raise ValueError(f"{conversation_id} has no eligible_drift_turn_idxs")

    turn_idx = int(eligible[0])
    if turn_idx >= len(turn_pairs):
        raise ValueError(f"{conversation_id} eligible drift turn {turn_idx} out of range")

    current_turn = turn_pairs[turn_idx]
    history_turn_pairs = turn_pairs[:turn_idx]
    drift_scores = row.get("drift_scores", [])

    drift_score = None
    if turn_idx < len(drift_scores):
        drift_score = drift_scores[turn_idx]

    return {
        "conversation_id": conversation_id,
        "turn_idx": turn_idx,
        "history_turn_pairs": history_turn_pairs,
        "current_query": current_turn.get("user", ""),
        "ground_truth_answer": current_turn.get("assistant", ""),
        "drift_score": drift_score,
    }


def build_stage1_candidates(
    case: Dict[str, Any],
    candidate_pool: List[CandidateEntry],
    pool_matrix: np.ndarray,
    embedder: Embedder,
    top_k: int,
    raw_top_k: int = 50,
) -> List[Dict[str, Any]]:
    current_query = case["current_query"]
    source_conversation_id = case["conversation_id"]

    query_emb = embedder.encode([current_query])[0]
    top_indices, top_scores = retrieve_top_k_indices(query_emb, pool_matrix, top_k=raw_top_k)

    kept: List[Dict[str, Any]] = []

    for idx, score in zip(top_indices.tolist(), top_scores.tolist()):
        cand = candidate_pool[idx]

        if cand.conversation_id == source_conversation_id:
            continue

        kept.append({
            "pool_index": idx,
            "conversation_id": cand.conversation_id,
            "pair_index": cand.pair_index,
            "query_similarity": float(score),
            "query_text": cand.query_text,
            "answer_text": cand.answer_text,
        })

        if len(kept) >= top_k:
            break

    return kept


def rerank_for_window(
    case: Dict[str, Any],
    stage1_candidates: List[Dict[str, Any]],
    candidate_pool: List[CandidateEntry],
    embedder: Embedder,
    window_size: int,
) -> List[Dict[str, Any]]:
    current_context_seq = build_current_context_sequence(
        history_turn_pairs=case["history_turn_pairs"],
        current_query=case["current_query"],
        embedder=embedder,
        window_size=window_size,
    )
    current_repr = attention_pool_last_query(current_context_seq)

    reranked: List[Dict[str, Any]] = []

    for cand_meta in stage1_candidates:
        cand = candidate_pool[cand_meta["pool_index"]]
        candidate_repr = attention_pool_last_query(cand.context_sequence)

        context_score = cosine_sim(current_repr, candidate_repr)
        sem_sim = semantic_similarity(cand.answer_text, case["ground_truth_answer"], embedder)

        reranked.append({
            **cand_meta,
            "context_score": float(context_score),
            "semantic_similarity": float(sem_sim),
        })

    reranked.sort(key=lambda x: x["context_score"], reverse=True)

    for rank, item in enumerate(reranked, start=1):
        item["reranked_rank"] = rank

    return reranked[:5]


def write_candidate_files(
    output_root: str,
    group_label: str,
    case: Dict[str, Any],
    window_size: int,
    candidates: List[Dict[str, Any]],
) -> None:
    conv_dir = os.path.join(
        output_root,
        group_label,
        case["conversation_id"],
        f"window{window_size}",
    )
    os.makedirs(conv_dir, exist_ok=True)

    for i, cand in enumerate(candidates, start=1):
        text = (
            f"source_conversation_id: {case['conversation_id']}\n"
            f"source_turn_idx: {case['turn_idx']}\n"
            f"window_size: {window_size}\n"
            f"source_drift_score: {case['drift_score']}\n\n"
            f"current_query:\n{case['current_query']}\n\n"
            f"ground_truth_answer:\n{case['ground_truth_answer']}\n\n"
            f"candidate_rank: {cand['reranked_rank']}\n"
            f"candidate_conversation_id: {cand['conversation_id']}\n"
            f"candidate_pair_index: {cand['pair_index']}\n"
            f"query_similarity: {cand['query_similarity']:.6f}\n"
            f"context_score: {cand['context_score']:.6f}\n"
            f"semantic_similarity: {cand['semantic_similarity']:.6f}\n\n"
            f"candidate_query:\n{cand['query_text']}\n\n"
            f"candidate_answer:\n{cand['answer_text']}\n"
        )
        write_text(os.path.join(conv_dir, f"candidate_{i:02d}.txt"), text)


def write_metrics_files(
    output_root: str,
    group_label: str,
    metrics_by_window: Dict[int, List[Tuple[str, float]]],
) -> None:
    metrics_dir = os.path.join(output_root, group_label, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    for window_size, rows in metrics_by_window.items():
        lines = [f"{conversation_id},{score:.6f}" for conversation_id, score in rows]
        write_text(
            os.path.join(metrics_dir, f"window{window_size}.txt"),
            "\n".join(lines) + ("\n" if lines else ""),
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run windowed retrieval/reranking inspection for chosen conversation_ids."
    )
    parser.add_argument(
        "--input_jsonl",
        required=True,
        help="input the file containing turn_pairs and eligible_drift_turn_idxs",
    )
    parser.add_argument(
        "--candidate_pool_pickle",
        required=True,
        help="path to to candidate pool pickle",
    )
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--group_label", required=True, help="folder label like high/ low/")
    parser.add_argument("--model_name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch_size",type=int, default=128,)
    parser.add_argument("--top_k",type=int,default=5, help="num candidates")
    parser.add_argument("--raw_top_k",type=int,default=50, help="larger num of stage1 hits")
    args = parser.parse_args()

    # if not CONVERSATION_IDS:
        # raise ValueError("mke sure CONVERSATION_IDS at the top of the scriptis set")

    window_sizes = [0, 1, 2, 3, 4, 5]

    print("Loading annotated JSONL...")
    rows = load_jsonl(args.input_jsonl)
    for row in rows:
        conversation_id = str(row.get("conversation_id", ""))
        if conversation_id:
            CONVERSATION_IDS.append(conversation_id)
    row_map: Dict[str, Dict[str, Any]] = {
        str(row.get("conversation_id", "")): row for row in rows
    }

    print("Loading candidate pool pickle...")
    payload = load_pickle(args.candidate_pool_pickle)
    candidate_pool: List[CandidateEntry] = payload["candidate_pool"]
    pool_matrix: np.ndarray = payload["pool_matrix"]

    print("Loading embedder...")
    embedder = Embedder(
        model_name=args.model_name,
        batch_size=args.batch_size,
    )

    metrics_by_window: Dict[int, List[Tuple[str, float]]] = {w: [] for w in window_sizes}

    for conversation_id in CONVERSATION_IDS:
        if conversation_id not in row_map:
            print(f"Skipping {conversation_id}: not found in input JSONL")
            continue

        row = row_map[conversation_id]
        case = get_first_eligible_case(row)

        print(f"Processing {conversation_id} at drift turn {case['turn_idx']}")

        stage1_candidates = build_stage1_candidates(
            case=case,
            candidate_pool=candidate_pool,
            pool_matrix=pool_matrix,
            embedder=embedder,
            top_k=args.top_k,
            raw_top_k=args.raw_top_k,
        )

        for window_size in window_sizes:
            top_candidates = rerank_for_window(
                case=case,
                stage1_candidates=stage1_candidates,
                candidate_pool=candidate_pool,
                embedder=embedder,
                window_size=window_size,
            )

            write_candidate_files(
                output_root=args.output_root,
                group_label=args.group_label,
                case=case,
                window_size=window_size,
                candidates=top_candidates,
            )

            top1_score = top_candidates[0]["semantic_similarity"] if top_candidates else 0.0
            metrics_by_window[window_size].append((conversation_id, top1_score))

    write_metrics_files(
        output_root=args.output_root,
        group_label=args.group_label,
        metrics_by_window=metrics_by_window,
    )

    print("Done.")


if __name__ == "__main__":
    main()