#!/usr/bin/env python3


from __future__ import annotations

import argparse
import json
import os
import re
import html
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Iterable
import time

import warnings
from bs4 import MarkupResemblesLocatorWarning, XMLParsedAsHTMLWarning
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

from bs4 import BeautifulSoup, Tag, NavigableString
import ijson
from pathlib import Path
from langdetect import detect, LangDetectException, DetectorFactory

DetectorFactory.seed = 0

SUPPORTED_USER_ROLES = {"human", "user"}
SUPPORTED_ASSISTANT_ROLES = {"gpt", "assistant", "chatgpt", "bot"}


@dataclass
class Conversation:
    conversation_id: str
    source_id: str
    turns: List[Dict[str, str]]
    conversation_text: str
    num_turns: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract cleaned English-only ShareGPT conversations into one file."
    )
    parser.add_argument(
        "--output-file",
        default="sharegpt_english_clean.json",
        help="Output file path (.json or .jsonl).",
    )
    parser.add_argument(
        "--min-turns",
        type=int,
        default=10,
        help="Minimum number of turns to keep a conversation.",
    )
    parser.add_argument(
        "--max-conversations",
        type=int,
        default=0,
        help="Maximum normalized conversations to keep (0 means all).",
    )
    parser.add_argument(
        "--max-text-chars",
        type=int,
        default=2400,
        help="Max chars used for stored conversation_text preview.",
    )
    parser.add_argument(
        "--max-user-chars",
        type=int,
        default=1500,
        help="Max chars allowed for any user turn.",
    )
    parser.add_argument(
        "--format",
        choices=["json", "jsonl"],
        default="json",
        help="Output format.",
    )

    parser.add_argument(
    "--input-dir",
    default="sharegpt_raw",
    help="Directory containing downloaded ShareGPT json files.",
    )
    return parser.parse_args()



def _normalize_role(raw: Any) -> Optional[str]:
    if not isinstance(raw, str):
        return None
    r = raw.strip().lower()
    if r in SUPPORTED_USER_ROLES:
        return "user"
    if r in SUPPORTED_ASSISTANT_ROLES:
        return "assistant"
    return None



def _user_turn_too_long(text: str, max_user_chars: int) -> bool:
    return len(text) > max_user_chars



def _strip_code_for_language(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"`[^`]*`", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[{}\[\]();=<>\\/_*#+|~]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text



def _is_mostly_non_linguistic(text: str) -> bool:
    if not text:
        return True

    letters = sum(ch.isalpha() for ch in text)
    digits = sum(ch.isdigit() for ch in text)
    spaces = sum(ch.isspace() for ch in text)
    others = len(text) - letters - digits - spaces

    if letters < 20:
        return True
    if others > letters:
        return True
    return False



def _contains_non_ascii_letters(text: str) -> bool:
    for ch in text:
        if ch.isalpha() and ord(ch) > 127:
            return True
    return False



def _is_english(text: str) -> bool:
    prose = _strip_code_for_language(text)
    if len(prose) < 20:
        return False
    if _is_mostly_non_linguistic(prose):
        return False
    if _contains_non_ascii_letters(prose):
        return False
    try:
        return detect(prose) == "en"
    except LangDetectException:
        return False



def _is_full_english_conversation(turns: Sequence[Dict[str, str]]) -> bool:
    if not turns:
        return False
    return all(_is_english(t.get("text", "")) for t in turns)



def _clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    if not text:
        return ""
    if "<" not in text and "&" not in text:
        return " ".join(text.split())

    soup = BeautifulSoup(text, "html.parser")

    for tag in soup(["script", "style", "svg", "noscript"]):
        tag.decompose()

    parts: List[str] = []

    def normalize_prose(s: str) -> str:
        s = html.unescape(s)
        s = re.sub(r"\s+", " ", s)
        return s.strip()

    def normalize_code(s: str) -> str:
        s = html.unescape(s)
        s = s.replace("\r\n", "\n").replace("\r", "\n")
        lines = [line.rstrip() for line in s.split("\n")]
        return "\n".join(lines).strip()

    def push_prose(s: str) -> None:
        s = normalize_prose(s)
        if not s or s.lower() == "copy code":
            return
        if parts and not parts[-1].endswith("\n\n"):
            parts.append(" ")
        parts.append(s)

    def push_code(s: str) -> None:
        s = normalize_code(s)
        if not s or s.lower() == "copy code":
            return
        if parts and parts[-1] != "\n\n":
            parts.append("\n\n")
        parts.append(s)
        parts.append("\n\n")

    def walk(node: Any) -> None:
        if isinstance(node, NavigableString):
            push_prose(str(node))
            return

        if not isinstance(node, Tag):
            return

        name = (node.name or "").lower()

        if name in {"script", "style", "svg", "noscript"}:
            return

        if name in {"pre", "code"}:
            push_code(node.get_text("\n"))
            return

        if name in {"p", "div", "section", "article", "li", "ul", "ol", "br"}:
            if parts and parts[-1] not in {" ", "\n\n"}:
                parts.append(" ")

        for child in node.children:
            walk(child)

        if name in {"p", "div", "section", "article", "li"}:
            if parts and parts[-1] not in {" ", "\n\n"}:
                parts.append(" ")

    for child in soup.children:
        walk(child)

    cleaned = "".join(parts)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r" *\n\n *", "\n\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"(?im)^\s*copy code\s*$", "", cleaned)
    cleaned = cleaned.strip()

    if cleaned:
        return cleaned

    fallback = html.unescape(soup.get_text(" "))
    fallback = re.sub(r"\s+", " ", fallback).strip()
    return fallback



def _extract_turns(record: Dict[str, Any]) -> List[Dict[str, str]]:
    raw_turns = None
    for key in ("conversations", "messages", "dialog", "chat"):
        value = record.get(key)
        if isinstance(value, list):
            raw_turns = value
            break
    if raw_turns is None:
        return []

    turns: List[Dict[str, str]] = []
    for item in raw_turns:
        if not isinstance(item, dict):
            continue
        role = _normalize_role(item.get("from", item.get("role", item.get("speaker"))))
        if role is None:
            continue
        text = item.get("value", item.get("content", item.get("text", "")))
        if not isinstance(text, str):
            continue
        clean = _clean_text(text)
        if not clean:
            continue
        turns.append({"role": role, "text": clean})
    return turns

def iter_sharegpt_files(input_dir: str) -> List[str]:
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    preferred = [
        root / "sg_90k_part1.json",
        root / "sg_90k_part2.json",
    ]

    files: List[str] = []
    for path in preferred:
        if path.exists():
            files.append(str(path))

    if not files:
        raise FileNotFoundError(
            f"Did not find sg_90k_part1.json / sg_90k_part2.json in {input_dir}"
        )

    return files


def stream_records_from_json_file(path: str):
    print(f"Reading file: {path}")
    try:
        with open(path, "rb") as f:
            for record in ijson.items(f, "item"):
                if isinstance(record, dict):
                    yield record
    except Exception as e:
        print(f"Skipping bad file {path}: {e}")


def stream_records(input_dir: str):
    for path in iter_sharegpt_files(input_dir):
        yield from stream_records_from_json_file(path)

def normalize_conversations(
    dataset_rows: Iterable[Dict[str, Any]],
    min_turns: int,
    max_conversations: int,
    max_text_chars: int,
    max_user_chars: int,
) -> List[Conversation]:
    conversations: List[Conversation] = []
    start = time.time()
    for idx, record in enumerate(dataset_rows):
        if idx % 1000 == 0:
            elapsed = time.time() - start
            print(f"{idx} rows scanned | {len(conversations)} kept | {elapsed:.1f}s")
        turns = _extract_turns(record)
        if len(turns) < min_turns:
            continue

        if any(
            t["role"] == "user" and _user_turn_too_long(t["text"], max_user_chars)
            for t in turns
        ):
            continue

        if not _is_full_english_conversation(turns):
            continue

        text = "\n".join(f"{t['role']}: {t['text']}" for t in turns)
        source_id = str(
            record.get("id")
            or record.get("conversation_id")
            or record.get("chat_id")
            or idx
        )
        conversations.append(
            Conversation(
                conversation_id=f"conv_{len(conversations):07d}",
                source_id=source_id,
                turns=turns,
                conversation_text=text[:max_text_chars],
                num_turns=len(turns),
            )
        )

        if max_conversations > 0 and len(conversations) >= max_conversations:
            break

    return conversations



def export_conversations(
    conversations: Sequence[Conversation],
    output_file: str,
    output_format: str,
    input_dir: str,
    min_turns: int,
    max_conversations: int,
    max_text_chars: int,
    max_user_chars: int,
) -> None:
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    records = [
        {
            "conversation_id": conv.conversation_id,
            "source_id": conv.source_id,
            "num_turns": conv.num_turns,
            "conversation_text": conv.conversation_text,
            "turns": conv.turns,
        }
        for conv in conversations
    ]

    if output_format == "jsonl":
        with open(output_file, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return

    payload = {
        "input_dir": input_dir,
        "num_conversations": len(records),
        "preprocessing": {
            "min_turns": min_turns,
            "max_conversations": max_conversations,
            "max_text_chars": max_text_chars,
            "max_user_chars": max_user_chars,
            "english_only": True,
            "clean_html": True,
        },
        "conversations": records,
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)



def main() -> None:
    args = parse_args()

    print(f"Reading local ShareGPT files from: {args.input_dir}")
    rows = stream_records(args.input_dir)
    print("Local streaming started")

    print("Cleaning + filtering English conversations...")
    conversations = normalize_conversations(
        rows,
        min_turns=args.min_turns,
        max_conversations=args.max_conversations,
        max_text_chars=args.max_text_chars,
        max_user_chars=args.max_user_chars,
    )

    if not conversations:
        raise RuntimeError("No conversations remained after cleaning/filtering.")

    print(f"Kept conversations: {len(conversations)}")
    print(f"Writing output to: {args.output_file}")

    export_conversations(
        conversations=conversations,
        output_file=args.output_file,
        output_format=args.format,
        input_dir=args.input_dir,
        min_turns=args.min_turns,
        max_conversations=args.max_conversations,
        max_text_chars=args.max_text_chars,
        max_user_chars=args.max_user_chars,
    )

    print("Done.")


if __name__ == "__main__":
    main()