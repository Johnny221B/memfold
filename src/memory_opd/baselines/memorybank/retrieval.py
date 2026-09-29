from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Iterable

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?", re.UNICODE)


def tokenize(text: str) -> list[str]:
    tokens = TOKEN_RE.findall(text.lower())
    # Lightweight English normalization for PersonaMem's lexical retriever.
    return [token[:-1] if len(token) > 3 and token.endswith("s") and not token.endswith("ss") else token for token in tokens]


@dataclass(frozen=True)
class MemoryEntry:
    memory_id: str
    kind: str
    text: str
    session_index: int
    strength: int = 1

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class MemoryBank:
    """Small deterministic BM25 index over MemoryBank text memories.

    Upstream used FAISS plus old LangChain APIs. BM25 keeps the adaptation
    dependency-light and reproducible; the memory construction remains the
    method-defining session-summary/personality representation.
    """

    def __init__(self, entries: Iterable[MemoryEntry], k1: float = 1.5, b: float = 0.75):
        self.entries = list(entries)
        self.k1 = k1
        self.b = b
        self._tokens = [tokenize(entry.text) for entry in self.entries]
        self._tf = [Counter(tokens) for tokens in self._tokens]
        self._avgdl = sum(map(len, self._tokens)) / max(1, len(self._tokens))
        self._df = Counter()
        for tokens in self._tokens:
            self._df.update(set(tokens))

    def search(self, query: str, top_k: int = 3) -> list[tuple[MemoryEntry, float]]:
        terms = set(tokenize(query))
        n_docs = len(self.entries)
        scored: list[tuple[MemoryEntry, float]] = []
        for entry, tokens, frequencies in zip(self.entries, self._tokens, self._tf):
            score = 0.0
            for term in terms:
                tf = frequencies[term]
                if not tf:
                    continue
                idf = math.log(1.0 + (n_docs - self._df[term] + 0.5) / (self._df[term] + 0.5))
                norm = tf + self.k1 * (1.0 - self.b + self.b * len(tokens) / max(self._avgdl, 1.0))
                score += idf * tf * (self.k1 + 1.0) / norm
            scored.append((entry, score))
        scored.sort(key=lambda item: (-item[1], item[0].memory_id))
        return scored[:top_k]


def session_blocks(messages: list[dict[str, str]], turns_per_session: int) -> list[list[dict[str, str]]]:
    if turns_per_session < 1:
        raise ValueError("turns_per_session must be positive")
    system = [m for m in messages if m.get("role") == "system"]
    dialogue = [m for m in messages if m.get("role") in {"user", "assistant"}]
    width = 2 * turns_per_session
    blocks = [dialogue[i : i + width] for i in range(0, len(dialogue), width)]
    if system and blocks:
        blocks[0] = system + blocks[0]
    elif system:
        blocks = [system]
    return blocks


def render_messages(messages: list[dict[str, str]]) -> str:
    return "\n".join(f"{m.get('role', 'unknown').upper()}: {m.get('content', '').strip()}" for m in messages)


def extract_persona(messages: list[dict[str, str]]) -> str:
    return "\n".join(m.get("content", "").strip() for m in messages if m.get("role") == "system").strip()
