from __future__ import annotations

from collections.abc import Callable


def chunk_messages(
    messages: list[dict[str, str]],
    token_count: Callable[[str], int],
    *,
    max_tokens: int = 384,
) -> list[tuple[int, str]]:
    """Pack messages chronologically, splitting an over-budget message safely."""
    chunks: list[tuple[int, str]] = []
    current: list[str] = []
    current_start = 0
    for index, message in enumerate(messages):
        rendered = f"[{str(message.get('role', 'unknown')).upper()}]\n{message.get('content', '')}"
        if token_count(rendered) > max_tokens:
            if current:
                chunks.append((current_start, "\n\n".join(current)))
                current = []
            words = rendered.split()
            piece: list[str] = []
            for word in words:
                candidate_piece = " ".join(piece + [word])
                if piece and token_count(candidate_piece) > max_tokens:
                    chunks.append((index, " ".join(piece)))
                    piece = [word]
                else:
                    piece.append(word)
            if piece:
                if token_count(" ".join(piece)) > max_tokens:
                    raise ValueError(f"single token-like span in message {index} exceeds chunk budget")
                chunks.append((index, " ".join(piece)))
            current_start = index + 1
            continue
        candidate = "\n\n".join(current + [rendered])
        if current and token_count(candidate) > max_tokens:
            chunks.append((current_start, "\n\n".join(current)))
            current, current_start = [rendered], index
        else:
            if not current:
                current_start = index
            current.append(rendered)
    if current:
        chunks.append((current_start, "\n\n".join(current)))
    return chunks


def select_top_k(
    chunks: list[tuple[int, str]], scores: list[float], *, top_k: int = 5
) -> list[tuple[int, str]]:
    if len(chunks) != len(scores):
        raise ValueError("one retrieval score is required per chunk")
    ranked = sorted(range(len(chunks)), key=lambda i: (-scores[i], chunks[i][0]))[:top_k]
    # Preserve original chronology after retrieval to avoid inventing a new timeline.
    return sorted((chunks[i] for i in ranked), key=lambda item: item[0])
