"""Minimal context-to-soft-token memory reconstruction components."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from memory_opd.rq2_baselines.personamem import load_contexts, render_context


@dataclass(frozen=True)
class ReconstructionExample:
    """One question-conditioned reconstruction view of a context state."""

    question_id: str
    context_id: str
    prefix_end: int
    state_id: str
    question: str
    memory_text: str


def different_context_example(
    examples: list[ReconstructionExample], index: int
) -> ReconstructionExample:
    """Return the next deterministic negative drawn from a different context."""
    if not 0 <= index < len(examples):
        raise IndexError(index)
    anchor = examples[index]
    for offset in range(1, len(examples)):
        candidate = examples[(index + offset) % len(examples)]
        if candidate.context_id != anchor.context_id:
            return candidate
    raise ValueError("at least two distinct contexts are required")

def same_context_example(examples: list[ReconstructionExample], index: int) -> ReconstructionExample:
    if not 0 <= index < len(examples):
        raise IndexError(index)
    anchor = examples[index]
    for offset in range(1, len(examples)):
        candidate = examples[(index + offset) % len(examples)]
        if candidate.context_id == anchor.context_id:
            return candidate
    raise ValueError("at least two examples from one context are required")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def context_state_id(context_id: str, prefix_end: int) -> str:
    digest = hashlib.sha256(f"{context_id}\0{prefix_end}".encode()).hexdigest()[:20]
    return f"{context_id[:12]}-{prefix_end:06d}-{digest}"


def text_memory_state_id(question_id: str, memory_text: str) -> str:
    """Return a stable cache key for one question-conditioned text memory."""

    digest = hashlib.sha256(f"{question_id}\0{memory_text}".encode()).hexdigest()[:20]
    return f"memory-{question_id[:12]}-{digest}"


def serialize_memory(memory: Mapping[str, Any]) -> str:
    """Serialize the evidence-v1 object deterministically for causal-LM SFT."""

    required = ("evidence", "temporal_relations", "derived_facts")
    if set(memory) != set(required):
        raise ValueError(f"memory must contain exactly {required}; found {sorted(memory)}")
    normalized: dict[str, list[str]] = {}
    for key in required:
        values = memory[key]
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            raise ValueError(f"memory.{key} must be a list of strings")
        normalized[key] = values
    return json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))


def load_reconstruction_examples(
    questions_path: Path,
    contexts_path: Path,
    memories_path: Path,
) -> tuple[list[ReconstructionExample], dict[str, str]]:
    """Join prepared PersonaMem rows to one-memory-per-question annotations.

    Answers and options are deliberately not copied into the returned examples.
    Context text is returned once per unique state for offline feature caching.
    """

    questions = _jsonl(questions_path)
    contexts = load_contexts(contexts_path)
    memories = _jsonl(memories_path)
    memory_by_id: dict[str, Mapping[str, Any]] = {}
    for row in memories:
        question_id = str(row["question_id"])
        if question_id in memory_by_id:
            raise ValueError(f"duplicate memory question_id: {question_id}")
        memory_by_id[question_id] = row["memory"]

    examples: list[ReconstructionExample] = []
    state_text: dict[str, str] = {}
    seen_questions: set[str] = set()
    for row in questions:
        question_id = str(row["question_id"])
        if question_id in seen_questions:
            raise ValueError(f"duplicate question_id: {question_id}")
        seen_questions.add(question_id)
        if question_id not in memory_by_id:
            raise KeyError(f"missing memory for question {question_id}")
        context_id = str(row["shared_context_id"])
        messages = contexts[context_id]
        raw_end = int(row["end_index_in_shared_context"])
        prefix_end = raw_end if raw_end >= 0 else len(messages) + raw_end
        if not 0 < prefix_end <= len(messages):
            raise ValueError(f"invalid legal prefix end {raw_end} for {question_id}")
        state_id = context_state_id(context_id, prefix_end)
        text = render_context(messages[:prefix_end])
        existing = state_text.setdefault(state_id, text)
        if existing != text:
            raise AssertionError(f"state ID collision: {state_id}")
        examples.append(
            ReconstructionExample(
                question_id=question_id,
                context_id=context_id,
                prefix_end=prefix_end,
                state_id=state_id,
                question=str(row["question"]),
                memory_text=serialize_memory(memory_by_id[question_id]),
            )
        )
    extra = set(memory_by_id) - seen_questions
    if extra:
        raise ValueError(f"memories contain unknown question IDs: {sorted(extra)[:3]}")
    return examples, state_text


def load_text_memory_examples(
    questions_path: Path,
    memories_path: Path,
) -> tuple[list[ReconstructionExample], dict[str, str]]:
    """Join questions to extracted memories used as the compressor input.

    Each question has its own state even when several questions share the same
    original PersonaMem session. The raw context is never loaded or returned.
    """

    questions = _jsonl(questions_path)
    memories = _jsonl(memories_path)
    memory_by_id: dict[str, Mapping[str, Any]] = {}
    for row in memories:
        question_id = str(row["question_id"])
        if question_id in memory_by_id:
            raise ValueError(f"duplicate memory question_id: {question_id}")
        memory_by_id[question_id] = row["memory"]

    examples: list[ReconstructionExample] = []
    state_text: dict[str, str] = {}
    seen_questions: set[str] = set()
    for row in questions:
        question_id = str(row["question_id"])
        if question_id in seen_questions:
            raise ValueError(f"duplicate question_id: {question_id}")
        seen_questions.add(question_id)
        if question_id not in memory_by_id:
            raise KeyError(f"missing memory for question {question_id}")
        memory_text = serialize_memory(memory_by_id[question_id])
        state_id = text_memory_state_id(question_id, memory_text)
        state_text[state_id] = memory_text
        examples.append(
            ReconstructionExample(
                question_id=question_id,
                context_id=str(row["shared_context_id"]),
                prefix_end=-1,
                state_id=state_id,
                question=str(row["question"]),
                memory_text=memory_text,
            )
        )
    extra = set(memory_by_id) - seen_questions
    if extra:
        raise ValueError(f"memories contain unknown question IDs: {sorted(extra)[:3]}")
    return examples, state_text


def split_examples_by_context(
    examples: Sequence[ReconstructionExample], *, validation_contexts: int = 4, seed: int = 42
) -> tuple[list[ReconstructionExample], list[ReconstructionExample]]:
    """Hold out complete shared contexts, never individual questions."""

    context_ids = sorted({example.context_id for example in examples})
    if not 0 < validation_contexts < len(context_ids):
        raise ValueError("validation_contexts must hold out at least one but not every context")
    ranked = sorted(
        context_ids,
        key=lambda value: hashlib.sha256(f"{seed}\0{value}".encode()).digest(),
    )
    validation_ids = set(ranked[:validation_contexts])
    train = [example for example in examples if example.context_id not in validation_ids]
    validation = [example for example in examples if example.context_id in validation_ids]
    if {item.context_id for item in train} & {item.context_id for item in validation}:
        raise AssertionError("context leakage across reconstruction train/validation")
    return train, validation


class ContextResampler(nn.Module):
    """Perceiver-style learned queries over cached frozen context states."""

    def __init__(
        self,
        context_dim: int,
        *,
        latent_dim: int = 768,
        token_count: int = 128,
        layers: int = 2,
        heads: int = 12,
        dropout: float = 0.0,
        context_residual: bool = False,
    ) -> None:
        super().__init__()
        if min(context_dim, latent_dim, token_count, layers, heads) <= 0:
            raise ValueError("resampler dimensions must be positive")
        if latent_dim % heads:
            raise ValueError("latent_dim must be divisible by heads")
        self.token_count = token_count
        self.context_residual = context_residual
        self.queries = nn.Parameter(torch.empty(token_count, latent_dim))
        nn.init.normal_(self.queries, std=0.02)
        self.context_projection = nn.Linear(context_dim, latent_dim)
        layer = nn.TransformerDecoderLayer(
            d_model=latent_dim,
            nhead=heads,
            dim_feedforward=4 * latent_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=layers)
        self.output_norm = nn.LayerNorm(latent_dim)
        if context_residual:
            self.context_gate = nn.Parameter(torch.ones(()))

    def forward(self, states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if states.ndim != 3 or mask.shape != states.shape[:2]:
            raise ValueError("cached states/mask shapes must be [batch, length, hidden]/[batch, length]")
        if torch.any(mask.long().sum(-1) == 0):
            raise ValueError("every context state must contain at least one cached vector")
        memory = self.context_projection(states)
        queries = self.queries.unsqueeze(0).expand(states.shape[0], -1, -1)
        if self.context_residual:
            valid = mask.bool().unsqueeze(-1)
            context_summary = (memory * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
            queries = queries + self.context_gate * context_summary.unsqueeze(1)
        hidden = self.decoder(
            tgt=queries,
            memory=memory,
            memory_key_padding_mask=~mask.bool(),
        )
        return self.output_norm(hidden)


class SoftTokenProjector(nn.Module):
    """Backbone-specific map from shared latents to the LM embedding space."""

    def __init__(self, latent_dim: int, lm_dim: int, *, initial_scale: float = 0.02) -> None:
        super().__init__()
        if min(latent_dim, lm_dim) <= 0 or initial_scale <= 0:
            raise ValueError("projector dimensions and initial_scale must be positive")
        self.projection = nn.Linear(latent_dim, lm_dim)
        self.norm = nn.LayerNorm(lm_dim)
        self.log_scale = nn.Parameter(torch.tensor(initial_scale).log())

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.norm(self.projection(latent)) * self.log_scale.exp()


class ContextToSoftTokens(nn.Module):
    def __init__(self, resampler: ContextResampler, projector: SoftTokenProjector) -> None:
        super().__init__()
        self.resampler = resampler
        self.projector = projector

    def forward(self, states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.projector(self.resampler(states, mask))


def soft_memory_vector(soft_tokens: torch.Tensor) -> torch.Tensor:
    """Return a normalized context-level vector for collapse diagnostics."""
    if soft_tokens.ndim != 3:
        raise ValueError("soft tokens must have shape [batch, tokens, hidden]")
    return F.normalize(soft_tokens.float().mean(dim=1), dim=-1)


def cross_context_separation_loss(
    own_soft: torch.Tensor,
    negative_soft: torch.Tensor,
    *,
    maximum_cosine: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Penalize different-context soft memories that remain too similar."""
    if own_soft.shape != negative_soft.shape:
        raise ValueError("own and negative soft tokens must have identical shapes")
    if not -1.0 <= maximum_cosine <= 1.0:
        raise ValueError("maximum cosine must be in [-1, 1]")
    own_vector = soft_memory_vector(own_soft)
    negative_vector = soft_memory_vector(negative_soft)
    cosine = (own_vector * negative_vector).sum(-1)
    # A cosine hinge has vanishing angular gradient near cosine=1, precisely
    # where a collapsed representation starts. The equivalent unit-sphere
    # distance threshold keeps a useful gradient for near-collapsed pairs.
    minimum_distance = (2.0 - 2.0 * maximum_cosine) ** 0.5
    distance = torch.linalg.vector_norm(own_vector - negative_vector, dim=-1)
    return torch.relu(minimum_distance - distance).mean(), cosine.mean()


def reconstruction_prompt(question: str) -> str:
    return (
        "Question:\n"
        f"{question}\n\n"
        "Reconstruct the compact evidence memory supported by the compressed context. "
        "Do not answer the question. Return only the memory JSON.\nMemory:\n"
    )
