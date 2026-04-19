import json
import os
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np

from sentence_transformers import SentenceTransformer


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

def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_history_unit(turn_pair: Dict[str, Any]) -> str:
    user_text = str(turn_pair.get("user", "") or "")
    assistant_text = str(turn_pair.get("assistant", "") or "")
    return f"Previous user question: {user_text}\nPrevious model response: {assistant_text}"


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a <= 0 or norm_b <= 0:
        return 0.0
    return float(np.dot(a,b) / (norm_a * norm_b))


class Embedder:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 128,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        texts = [str(t) for t in texts]
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)

        embeddings = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype(np.float32)



#helpers for mnanully inspecting the marked drift turns
def truncate_text(s: str, max_chars: int = 1500) -> str:
    if s is None:
        return ""
    return s[:max_chars]

def wrap_text(s: str, width: int = 70) -> str:
    text = str(s or "")
    if not text:
        return ""

    lines = text.splitlines()
    wrapped_lines: List[str] = []

    for line in lines:
        if len(line) <= width:
            wrapped_lines.append(line)
        else:
            wrapped_lines.extend(
                textwrap.wrap(
                    line,
                    width=width,
                    break_long_words=False,
                    break_on_hyphens=False,
                )
            )

    return "\n".join(wrapped_lines)


def format_turn_text(turn_idx: int, turn_pair: Dict[str, Any], width: int = 70) -> str:
    user_text = wrap_text(turn_pair.get("user", ""), width=width)
    assistant_text = wrap_text(turn_pair.get("assistant", ""), width=width)

    return (
        f"Turn: {turn_idx}\n\n"
        f"User:\n{user_text}\n\n"
        f"Assistant:\n{assistant_text}\n"
    )

def write_temp_analysis_tree(
    rows: List[Dict[str, Any]],
    analysis_root: str,
    dataset_label: str,
    wrap_width: int = 70,
) -> None:
    dataset_root = os.path.join(analysis_root, dataset_label)

    for bucket in ["low", "medium", "high"]:
        os.makedirs(os.path.join(dataset_root, bucket), exist_ok=True)

    for row in rows:
        conversation_id = str(row.get("conversation_id", "unknown_conversation"))
        bucket = row.get("conversation_drift_bucket", "unknown")
        turn_pairs = row.get("turn_pairs", [])

        conversation_dir = os.path.join(dataset_root, bucket, conversation_id)
        os.makedirs(conversation_dir, exist_ok=True)

        # one file per turn
        for turn_idx, turn_pair in enumerate(turn_pairs):
            turn_path = os.path.join(conversation_dir, f"{turn_idx}.txt")
            with open(turn_path, "w", encoding="utf-8") as f:
                f.write(format_turn_text(turn_idx, turn_pair, width=wrap_width))

        # drift summary file
        drift_summary_path = os.path.join(conversation_dir, "drift_scores.txt")
        with open(drift_summary_path, "w", encoding="utf-8") as f:
            f.write(format_drift_summary(row, width=wrap_width))
