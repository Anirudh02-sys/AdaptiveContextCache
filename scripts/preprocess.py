#!/usr/bin/env python3
"""split cleaned sharegpt jsonl into warmup (30%) and eval (1000 rows)
then run the embedding / clustering pipeline on the eval only
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


class Embedder:
    """Sentence-transformers wrapper with sklearn fallback."""

    def __init__(self, model_name: str, batch_size: int):
        self.batch_size = batch_size
        self._mode = "sentence_transformers"
        self._st_model = None
        self._tfidf = None
        self._svd = None
        self.dimension = None
        try:
            from sentence_transformers import SentenceTransformer

            self._st_model = SentenceTransformer(model_name)
        except Exception:
            self._mode = "tfidf"

    def fit(self, texts: Sequence[str]) -> None:
        if self._mode == "sentence_transformers":
            if not texts:
                self.dimension = 384
                return
            sample = self.encode(texts[:1])
            self.dimension = int(sample.shape[1])
            return

        from sklearn.decomposition import TruncatedSVD

        self._tfidf = TfidfVectorizer(max_features=15000, ngram_range=(1, 2))
        sparse = self._tfidf.fit_transform(texts)
        max_components = max(2, min(256, sparse.shape[1] - 1))
        if sparse.shape[1] <= 2:
            max_components = sparse.shape[1]
        self._svd = TruncatedSVD(n_components=max_components, random_state=42)
        dense = self._svd.fit_transform(sparse)
        dense = _l2_normalize(dense)
        self.dimension = int(dense.shape[1]) if dense.ndim == 2 else 1

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension or 1), dtype=np.float32)

        if self._mode == "sentence_transformers":
            arr = self._st_model.encode(
                list(texts),
                batch_size=self.batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return arr.astype(np.float32)

        sparse = self._tfidf.transform(texts)
        dense = self._svd.transform(sparse)
        dense = _l2_normalize(dense)
        return dense.astype(np.float32)

    @property
    def mode(self) -> str:
        return self._mode


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return vectors / norms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a cleaned ShareGPT JSONL and run clustering on the eval split."
    )
    parser.add_argument(
        "--input-file",
        required=True,
        help="Path to cleaned JSONL, e.g. data/min_turns_4.jsonl",
    )
    parser.add_argument(
        "--dataset-label",
        default="",
        help="Prefix for output names. Defaults to input filename stem.",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory to write warmup/eval split files and app outputs.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.30,
        help="Warmup ratio. Default 0.30.",
    )
    parser.add_argument(
        "--eval-count",
        type=int,
        default=1000,
        help="Number of non-overlapping eval conversations after warmup.",
    )
    parser.add_argument(
        "--num-applications",
        type=int,
        default=10,
        help="Number of application groups to generate.",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-transformers model name.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Embedding batch size.",
    )
    parser.add_argument(
        "--max-text-chars",
        type=int,
        default=2400,
        help="Max chars used per conversation for clustering text.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--min-turns",
        type=int,
        default=4,
        help="Metadata only, for manifest tracking.",
    )
    parser.add_argument(
        "--max-user-chars",
        type=int,
        default=1000,
        help="Metadata only, for manifest tracking.",
    )
    return parser.parse_args()


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"Line {line_num} in {path} is not a JSON object")
            rows.append(obj)
    return rows


def write_jsonl(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_warmup_pool(
    rows: Sequence[Dict[str, Any]],
    warmup_ratio: float,
    rng,
) -> Tuple[List[Dict[str, Any]], int]:
    total = len(rows)
    if total == 0:
        raise ValueError("Input JSONL has no rows")

    warmup_n = math.ceil(total * warmup_ratio)
    warmup_rows = rng.sample(rows, warmup_n)

    return warmup_rows, warmup_n


def _build_pairs(turns: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    pairs: List[Dict[str, str]] = []
    pending_user = None
    for t in turns:
        role = t["role"]
        txt = t["text"]
        if role == "user":
            pending_user = txt
        elif role == "assistant" and pending_user is not None:
            pairs.append({"user": pending_user, "assistant": txt})
            pending_user = None
    return pairs


def cluster_topics(
    embeddings: np.ndarray,
    num_applications: int,
    seed: int,
) -> np.ndarray:
    n_samples = embeddings.shape[0]
    n_clusters = max(1, min(num_applications, n_samples))
    model = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed)
    labels = model.fit_predict(embeddings)
    return labels.astype(int)


def build_topic_labels(
    rows: Sequence[Dict[str, Any]],
    labels: np.ndarray,
    num_terms: int = 5,
) -> Dict[int, str]:
    cluster_docs: Dict[int, List[str]] = defaultdict(list)
    for row, lbl in zip(rows, labels):
        cluster_docs[int(lbl)].append(str(row.get("conversation_text", "")))

    app_labels: Dict[int, str] = {}
    for cluster_id, docs in cluster_docs.items():
        vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            ngram_range=(1, 2),
        )
        mat = vectorizer.fit_transform(docs)
        scores = np.asarray(mat.mean(axis=0)).ravel()
        terms = np.asarray(vectorizer.get_feature_names_out())
        top_idx = np.argsort(scores)[::-1][:num_terms]
        top_terms = [terms[i] for i in top_idx if scores[i] > 0]
        if top_terms:
            app_labels[cluster_id] = ", ".join(top_terms)
        else:
            app_labels[cluster_id] = "general dialogue"
    return app_labels


def compute_topic_drift(
    rows: Sequence[Dict[str, Any]],
    embedder: Embedder,
) -> List[float]:
    scores: List[float] = []
    for row in rows:
        turns = row["turns"]
        turn_texts = [t["text"] for t in turns]
        if len(turn_texts) < 3:
            scores.append(0.0)
            continue
        emb = embedder.encode(turn_texts)
        if emb.shape[0] < 3:
            scores.append(0.0)
            continue
        anchor_count = min(2, emb.shape[0])
        anchor = np.mean(emb[:anchor_count], axis=0)
        anchor_norm = np.linalg.norm(anchor)
        if anchor_norm == 0.0:
            scores.append(0.0)
            continue
        anchor = anchor / anchor_norm
        later = emb[anchor_count:]
        sims = later @ anchor
        dists = 1.0 - sims
        drift = float(np.mean(dists)) if dists.size else 0.0
        if math.isnan(drift) or math.isinf(drift):
            drift = 0.0
        scores.append(max(0.0, drift))
    return scores


def bucketize_drift(scores: Sequence[float]) -> List[str]:
    if not scores:
        return []
    arr = np.asarray(scores, dtype=np.float32)
    p33, p66 = np.percentile(arr, [33, 66]).tolist()
    buckets: List[str] = []
    for value in arr:
        if value <= p33:
            buckets.append("low")
        elif value <= p66:
            buckets.append("medium")
        else:
            buckets.append("high")
    return buckets


def export_grouped_data(
    rows: Sequence[Dict[str, Any]],
    labels: np.ndarray,
    topic_labels: Dict[int, str],
    drift_scores: Sequence[float],
    drift_buckets: Sequence[str],
    output_dir: str,
    dataset_name: str,
    split: str,
    embedding_mode: str,
    embedding_model: str,
    min_turns: int,
    max_conversations: int,
    max_text_chars: int,
    max_user_chars: int,
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)

    grouped_records: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row, app_id, drift_score, drift_bucket in zip(
        rows, labels, drift_scores, drift_buckets
    ):
        turns = row["turns"]
        pair_turns = _build_pairs(turns)
        grouped_records[int(app_id)].append(
            {
                "conversation_id": row["conversation_id"],
                "source_id": row["source_id"],
                "application_id": int(app_id),
                "topic_label": topic_labels.get(int(app_id), "general dialogue"),
                "topic_drift_score": round(float(drift_score), 6),
                "topic_drift_bucket": drift_bucket,
                "num_turns": row.get("num_turns", len(turns)),
                "turns": turns,
                "turn_pairs": pair_turns,
            }
        )

    files: List[Dict[str, Any]] = []
    for app_id in sorted(grouped_records.keys()):
        path = os.path.join(output_dir, f"application_{app_id:02d}.jsonl")
        records = grouped_records[app_id]
        with open(path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        drift_counts = Counter(r["topic_drift_bucket"] for r in records)
        files.append(
            {
                "application_id": app_id,
                "topic_label": topic_labels.get(app_id, "general dialogue"),
                "num_conversations": len(records),
                "file_path": path,
                "topic_drift_distribution": dict(drift_counts),
            }
        )

    manifest = {
        "dataset": dataset_name,
        "split": split,
        "num_applications": len(files),
        "num_conversations": len(rows),
        "embedding_backend": embedding_mode,
        "embedding_model": embedding_model,
        "preprocessing": {
            "min_turns": min_turns,
            "max_conversations": max_conversations,
            "max_text_chars": max_text_chars,
            "max_user_chars": max_user_chars,
        },
        "output_files": files,
    }

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return manifest


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    import random
    rng = random.Random(args.seed)

    label = args.dataset_label or os.path.splitext(os.path.basename(args.input_file))[0]
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading cleaned JSONL: {args.input_file}")
    rows = load_jsonl(args.input_file)
    print(f"Loaded rows: {len(rows)}")

    warmup_rows, warmup_n = split_warmup_pool(
        rows,
        warmup_ratio=args.warmup_ratio,
        rng=rng
    )
    warmup_rows_with_pairs = []
    for row in warmup_rows:
        new_row = dict(row)
        new_row["turn_pairs"] = _build_pairs(row.get("turns", []))
        warmup_rows_with_pairs.append(new_row)

    if len(warmup_rows_with_pairs) < args.eval_count:
        raise ValueError(
            f"Warmup pool smaller than eval_count. "
            f"warmup={len(warmup_rows_with_pairs)}, eval_count={args.eval_count}"
        )

    eval_rows = rng.sample(warmup_rows_with_pairs, args.eval_count)

    warmup_path = os.path.join(args.output_dir, f"{label}_warmup_{len(warmup_rows)}.jsonl")
    eval_path = os.path.join(args.output_dir, f"{label}_eval_{len(eval_rows)}.jsonl")
    apps_output_dir = os.path.join(args.output_dir, f"{label}_eval_{len(eval_rows)}_apps")
    split_manifest_path = os.path.join(args.output_dir, f"{label}_split_manifest.json")

    write_jsonl(warmup_path, warmup_rows_with_pairs)
    write_jsonl(eval_path, eval_rows)

    print(f"Wrote warmup file: {warmup_path}")
    print(f"Wrote eval file: {eval_path}")
    print(f"Warmup count: {len(warmup_rows)} | Eval count: {len(eval_rows)}")

    print("Building embeddings...")
    convo_texts = [str(row["conversation_text"])[: args.max_text_chars] for row in eval_rows]
    embedder = Embedder(args.embedding_model, batch_size=args.batch_size)
    embedder.fit(convo_texts)
    convo_embeddings = embedder.encode(convo_texts)
    print(f"Embedding backend: {embedder.mode}")

    print("Clustering into application groups...")
    labels = cluster_topics(convo_embeddings, args.num_applications, args.seed)
    n_apps_actual = len(set(labels.tolist()))
    print(f"Created {n_apps_actual} application groups.")

    print("Extracting topic labels...")
    topic_labels = build_topic_labels(eval_rows, labels)

    print("Scoring topic drift...")
    drift_scores = compute_topic_drift(eval_rows, embedder)
    drift_buckets = bucketize_drift(drift_scores)

    print(f"Exporting grouped files to: {apps_output_dir}")
    export_grouped_data(
        rows=eval_rows,
        labels=labels,
        topic_labels=topic_labels,
        drift_scores=drift_scores,
        drift_buckets=drift_buckets,
        output_dir=apps_output_dir,
        dataset_name=label,
        split=f"eval_{len(eval_rows)}",
        embedding_mode=embedder.mode,
        embedding_model=args.embedding_model,
        min_turns=args.min_turns,
        max_conversations=len(eval_rows),
        max_text_chars=args.max_text_chars,
        max_user_chars=args.max_user_chars,
    )

    split_manifest = {
        "label": label,
        "input_file": args.input_file,
        "total_rows": len(rows),
        "warmup_ratio": args.warmup_ratio,
        "warmup_count": len(warmup_rows),
        "eval_count": len(eval_rows),
        "sampling": {
            "warmup_random_sample": True,
            "eval_random_sample_from_warmup": True,
            "overlap": True,
            "seed": args.seed,
        },
        "warmup_file": warmup_path,
        "eval_file": eval_path,
        "apps_output_dir": apps_output_dir,
        "apps_manifest": os.path.join(apps_output_dir, "manifest.json"),
    }
    with open(split_manifest_path, "w", encoding="utf-8") as f:
        json.dump(split_manifest, f, ensure_ascii=False, indent=2)

    print("Done.")
    print(json.dumps(split_manifest, indent=2))


if __name__ == "__main__":
    main()