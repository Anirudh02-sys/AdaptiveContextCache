import argparse
import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np

from drift_utils import Embedder, load_jsonl, make_history_unit

"""
This is build the candidate pool (simulates the cache)for the drift experiment.
Because we don't actually need to invoke the llm.
Dataset to use: mt10_prewarm_2220.jsonl
What script does:
1. Read in the dataset
2. Convert each conversation turn into a candidate entry
    like if turn_pairs: [{user: q1, assistant: a1}, {user: q2, assistant: a2}, {user: q3, assistant: a3}] 
    then conceptually 
        candidate 1: query = q1, context_sequence = [q1]
        candidate 2: query = q2, context_sequence = [q1+a1, q2]
        candidate 3: query = q3, context_sequence = [q1+a1, q2+a2, q3]
        And by q1+a1 we mean:
        "Previous user question: q1\nPrevious model response: a1" just like in the other experiments
3. Compute the embeddings 
4. What is saved to a pickle:
    candidate_pool: every context_sequence embeddings and metadata (used for stage2 later)
    pool_matrix: every query embedding (used for stage1 later)
"""

@dataclass
class CandidateEntry:
    conversation_id: Any
    pair_index: int
    query_text: str
    answer_text: str
    full_prior_context: List[Dict[str, Any]]
    query_embedding: np.ndarray
    context_sequence: np.ndarray


def build_candidate_pool(
    warmup_rows: List[Dict[str, Any]],
    embedder: Embedder,
    max_candidate_history_units: int = 5,
) -> List[CandidateEntry]:
    raw_entries: List[Tuple[Any, int, str, str, List[Dict[str, Any]]]] = []
    query_texts: List[str] = []
    history_unit_texts: List[str] = []

    for row in warmup_rows:
        conversation_id = row.get("conversation_id")
        turn_pairs = row.get("turn_pairs", [])

        for idx, pair in enumerate(turn_pairs):
            query_text = str(pair.get("user", "") or "").strip()
            answer_text = str(pair.get("assistant", "") or "").strip()

            if not query_text or not answer_text:
                continue

            full_prior_context = turn_pairs[:idx]
            raw_entries.append(
                (conversation_id, idx, query_text, answer_text, full_prior_context)
            )
            query_texts.append(query_text)

            history_units = [make_history_unit(tp) for tp in full_prior_context]
            if max_candidate_history_units > 0:
                history_units = history_units[-max_candidate_history_units:]
            history_unit_texts.extend(history_units)

    print(f"Embedding {len(query_texts)} candidate queries...")
    query_embs = embedder.encode(query_texts)

    history_emb_map: Dict[str, np.ndarray] = {}
    if history_unit_texts:
        unique_history_units = list(dict.fromkeys(history_unit_texts))
        print(f"Embedding {len(unique_history_units)} unique history units...")
        unique_history_embs = embedder.encode(unique_history_units)
        for text, emb in zip(unique_history_units, unique_history_embs):
            history_emb_map[text] = emb

    candidate_pool: List[CandidateEntry] = []
    for i, (conversation_id, idx, query_text, answer_text, full_prior_context) in enumerate(raw_entries):
        query_emb = query_embs[i]

        history_units = [make_history_unit(tp) for tp in full_prior_context]
        if max_candidate_history_units > 0:
            history_units = history_units[-max_candidate_history_units:]

        history_embs = [history_emb_map[text] for text in history_units] if history_units else []
        context_seq = history_embs + [query_emb]
        context_array = np.asarray(context_seq, dtype=np.float32)

        candidate_pool.append(
            CandidateEntry(
                conversation_id=conversation_id,
                pair_index=idx,
                query_text=query_text,
                answer_text=answer_text,
                full_prior_context=full_prior_context,
                query_embedding=query_emb,
                context_sequence=context_array,
            )
        )

    return candidate_pool


def get_pool_query_matrix(candidate_pool: List[CandidateEntry]) -> np.ndarray:
    if not candidate_pool:
        return np.zeros((0, 384), dtype=np.float32)
    return np.vstack([c.query_embedding for c in candidate_pool]).astype(np.float32)


def save_pickle(path: str, obj: Any) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute and save turn-level warmup candidate embeddings."
    )
    parser.add_argument("--warmup_file", required=True)
    parser.add_argument("--output_pickle", required=True)
    parser.add_argument("--model_name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_candidate_history_units", type=int, default=0, help="max prior turns. 0=full history")
    args = parser.parse_args()

    print("Loading warmup rows...")
    warmup_rows = load_jsonl(args.warmup_file)

    print("Loading embedder...")
    embedder = Embedder(
        model_name=args.model_name,
        batch_size=args.batch_size,
    )

    print("Building candidate pool...")
    candidate_pool = build_candidate_pool(
        warmup_rows=warmup_rows,
        embedder=embedder,
        max_candidate_history_units=args.max_candidate_history_units,
    )

    print("Building pool query matrix...")
    pool_matrix = get_pool_query_matrix(candidate_pool)

    payload = {
        "model_name": args.model_name,
        "max_candidate_history_units": args.max_candidate_history_units,
        "candidate_pool": candidate_pool,
        "pool_matrix": pool_matrix,
    }

    print(f"Saving to {args.output_pickle} ...")
    save_pickle(args.output_pickle, payload)

    print("Done.")
    print(f"Saved {len(candidate_pool)} candidates.")


if __name__ == "__main__":
    main()