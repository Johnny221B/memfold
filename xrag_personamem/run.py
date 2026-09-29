#!/usr/bin/env python3
"""Train and evaluate a compact xRAG-style bridge on PersonaMem."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup


LETTERS = ["(a)", "(b)", "(c)", "(d)"]


class Projector(nn.Module):
    """Official xRAG mlp2x_gelu bridge shape."""
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x.float())


def prompt_messages(question: str, options: list[str], k: int) -> list[dict]:
    memory = " ".join(["<xRAG>"] * k)
    rendered = "\n".join(f"{LETTERS[i]} {value}" for i, value in enumerate(options))
    return [
        {"role": "system", "content": "You answer multiple-choice questions using compressed retrieved memory. The xRAG tokens contain relevant information from the user's prior conversation."},
        {"role": "user", "content": f"Compressed retrieved memory: {memory}\n\nQuestion: {question}\n\nOptions:\n{rendered}\n\nAnswer with only (a), (b), (c), or (d)."},
    ]


def encode_qa(tokenizer, row: dict, k: int, answer: str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    messages = prompt_messages(row["question"], row["options"], k)
    prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
    if answer is None:
        return torch.tensor(prompt, dtype=torch.long), torch.full((len(prompt),), -100, dtype=torch.long)
    target = tokenizer.encode(answer, add_special_tokens=False) + [tokenizer.eos_token_id]
    ids = torch.tensor(prompt + target, dtype=torch.long)
    labels = torch.tensor([-100] * len(prompt) + target, dtype=torch.long)
    return ids, labels


def inject(model, projector, ids: torch.Tensor, vectors: torch.Tensor, xrag_id: int) -> torch.Tensor:
    embeddings = model.get_input_embeddings()(ids)
    mask = ids.eq(xrag_id)
    assert int(mask.sum()) == len(vectors), (int(mask.sum()), len(vectors))
    projected = projector(vectors.to(embeddings.device)).to(embeddings.dtype)
    return embeddings.masked_scatter(mask.unsqueeze(-1), projected.reshape(-1))


def batch_forward(model, projector, tokenizer, rows: list[dict], query_mode: str, k: int, device: str) -> torch.Tensor:
    encoded = [encode_qa(tokenizer, row, k, row["answer"]) for row in rows]
    width = max(len(x[0]) for x in encoded)
    all_embeds, all_labels, masks = [], [], []
    for row, (ids, labels) in zip(rows, encoded):
        ids, labels = ids.to(device), labels.to(device)
        vectors = row[f"retrieved_{query_mode}_vectors"][:k]
        embeddings = inject(model, projector, ids, vectors, tokenizer.convert_tokens_to_ids("<xRAG>"))
        pad = width - len(ids)
        all_embeds.append(F.pad(embeddings, (0, 0, 0, pad)))
        all_labels.append(F.pad(labels, (0, pad), value=-100))
        masks.append(F.pad(torch.ones(len(ids), device=device, dtype=torch.long), (0, pad)))
    output = model(inputs_embeds=torch.stack(all_embeds), attention_mask=torch.stack(masks), labels=torch.stack(all_labels), use_cache=False)
    return output.loss


def permute_training_row(row: dict, rng: random.Random) -> dict:
    """Remove answer-position shortcuts while preserving retrieval vectors."""
    row = dict(row)
    permutation = list(range(4)); rng.shuffle(permutation)
    gold = LETTERS.index(row["answer"])
    row["options"] = [row["options"][i] for i in permutation]
    row["answer"] = LETTERS[permutation.index(gold)]
    return row


def pretrain_batch(model, projector, tokenizer, docs: list[dict], device: str, max_target_tokens: int) -> torch.Tensor:
    encoded = []
    for doc in docs:
        messages = [
            {"role": "user", "content": "Background: <xRAG> means the same as"},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
        target = tokenizer.encode(doc["text"], add_special_tokens=False)[:max_target_tokens] + [tokenizer.eos_token_id]
        encoded.append((torch.tensor(prompt + target), torch.tensor([-100] * len(prompt) + target), doc["vector"].unsqueeze(0)))
    width = max(len(x[0]) for x in encoded)
    embeds, labels, masks = [], [], []
    xrag_id = tokenizer.convert_tokens_to_ids("<xRAG>")
    for ids, target, vector in encoded:
        ids, target = ids.to(device), target.to(device)
        value = inject(model, projector, ids, vector, xrag_id)
        pad = width - len(ids)
        embeds.append(F.pad(value, (0, 0, 0, pad)))
        labels.append(F.pad(target, (0, pad), value=-100))
        masks.append(F.pad(torch.ones(len(ids), device=device, dtype=torch.long), (0, pad)))
    return model(inputs_embeds=torch.stack(embeds), attention_mask=torch.stack(masks), labels=torch.stack(labels), use_cache=False).loss


@torch.inference_mode()
def evaluate(model, projector, tokenizer, rows: list[dict], query_mode: str, k: int, device: str, output: Path) -> dict:
    projector.eval(); model.eval(); results = []
    for number, row in enumerate(rows, 1):
        ids, _ = encode_qa(tokenizer, row, k)
        ids = ids.to(device)
        vectors = row[f"retrieved_{query_mode}_vectors"][:k]
        embeddings = inject(model, projector, ids, vectors, tokenizer.convert_tokens_to_ids("<xRAG>")).unsqueeze(0)
        generated = model.generate(inputs_embeds=embeddings, attention_mask=torch.ones(1, len(ids), device=device, dtype=torch.long), max_new_tokens=8, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
        predicted = next((letter for letter in LETTERS if letter in text.lower()), None)
        results.append({
            "question_id": row["question_id"], "gold": row["answer"], "predicted": predicted,
            "correct": predicted == row["answer"], "generation": text,
            "answer_prompt_tokens": len(ids), "answer_output_tokens": len(generated[0]),
            "query_encoder_tokens": row[f"query_tokens_{query_mode}"],
            "context_encoder_tokens": row["context_encoder_tokens"],
            "retrieved_texts": row[f"retrieved_{query_mode}_texts"][:k],
            "retrieved_scores": row[f"retrieved_{query_mode}_scores"][:k],
        })
        if number % 25 == 0:
            print(json.dumps({"eval": number, "total": len(rows), "correct": sum(x["correct"] for x in results)}), flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in results), encoding="utf-8")
    unique_context_cost = {}
    for row, result in zip(rows, results):
        unique_context_cost[row["shared_context_id"]] = max(unique_context_cost.get(row["shared_context_id"], 0), result["context_encoder_tokens"])
    online = [x["answer_prompt_tokens"] + x["answer_output_tokens"] + x["query_encoder_tokens"] for x in results]
    strict_end_to_end = [cost + result["context_encoder_tokens"] for cost, result in zip(online, results)]
    return {
        "questions": len(results), "correct": sum(x["correct"] for x in results),
        "accuracy": sum(x["correct"] for x in results) / len(results),
        "parsed": sum(x["predicted"] is not None for x in results),
        "mean_online_tokens": sum(online) / len(online),
        "mean_amortized_end_to_end_tokens": (sum(online) + sum(unique_context_cost.values())) / len(online),
        "mean_strict_end_to_end_tokens": sum(strict_end_to_end) / len(strict_end_to_end),
    }


def train_phase(model, projector, tokenizer, items: list[dict], args, phase: str, epochs: int, lr: float) -> None:
    optimizer = torch.optim.AdamW(projector.parameters(), lr=lr)
    batch_size = args.pretrain_batch_size if phase == "pretrain" else args.qa_batch_size
    steps = math.ceil(len(items) / batch_size) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, max(1, int(steps * 0.03)), steps)
    rng = random.Random(args.seed)
    projector.train(); model.eval(); step = 0
    for epoch in range(epochs):
        order = list(range(len(items))); rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            batch = [items[i] for i in order[start : start + batch_size]]
            if phase == "pretrain":
                loss = pretrain_batch(model, projector, tokenizer, batch, args.device, args.max_pretrain_target_tokens)
            else:
                batch = [permute_training_row(row, rng) for row in batch]
                loss = batch_forward(model, projector, tokenizer, batch, args.query_mode, args.top_k, args.device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(projector.parameters(), 5.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True); step += 1
            if step % args.log_steps == 0:
                print(json.dumps({"phase": phase, "epoch": epoch + 1, "step": step, "steps": steps, "loss": float(loss), "lr": scheduler.get_last_lr()[0]}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--query-mode", choices=["question", "question_options"], default="question")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--pretrain-epochs", type=int, default=1)
    parser.add_argument("--qa-epochs", type=int, default=3)
    parser.add_argument("--pretrain-lr", type=float, default=6e-3)
    parser.add_argument("--qa-lr", type=float, default=2e-5)
    parser.add_argument("--pretrain-batch-size", type=int, default=12)
    parser.add_argument("--qa-batch-size", type=int, default=8)
    parser.add_argument("--max-pretrain-docs", type=int, default=20000)
    parser.add_argument("--max-pretrain-target-tokens", type=int, default=180)
    parser.add_argument("--seed", type=int, default=980406)
    parser.add_argument("--log-steps", type=int, default=50)
    parser.add_argument("--initial-projector", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n")
    random.seed(args.seed); torch.manual_seed(args.seed)

    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.add_special_tokens({"additional_special_tokens": ["<xRAG>"]})
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").to(args.device)
    model.resize_token_embeddings(len(tokenizer))
    for parameter in model.parameters(): parameter.requires_grad_(False)
    model.eval(); model.config.use_cache = False
    projector = Projector(cache["retriever_hidden_size"], model.config.hidden_size).to(args.device)
    if args.initial_projector:
        projector.load_state_dict(torch.load(args.initial_projector, map_location=args.device, weights_only=True))

    docs = cache["pretrain_docs"]
    if len(docs) > args.max_pretrain_docs:
        docs = random.Random(args.seed).sample(docs, args.max_pretrain_docs)
    started = time.time()
    train_phase(model, projector, tokenizer, docs, args, "pretrain", args.pretrain_epochs, args.pretrain_lr)
    torch.save(projector.state_dict(), args.output / "projector-pretrain.pt")
    train = [x for x in cache["questions"] if x["split"] == "train"]
    train_phase(model, projector, tokenizer, train, args, "qa", args.qa_epochs, args.qa_lr)
    torch.save(projector.state_dict(), args.output / "projector-final.pt")
    validation = evaluate(model, projector, tokenizer, [x for x in cache["questions"] if x["split"] == "validation"], args.query_mode, args.top_k, args.device, args.output / "validation.jsonl")
    test = evaluate(model, projector, tokenizer, [x for x in cache["questions"] if x["split"] == "test"], args.query_mode, args.top_k, args.device, args.output / "test.jsonl")
    summary = {"schema": "xrag-personamem-result-v1", "dataset": cache["dataset"], "backbone": args.model.name, "validation": validation, "test": test, "elapsed_seconds": time.time() - started, "official_delta": f"{args.model.name} and PersonaMem adapter; official mlp2x_gelu bridge and reconstruction-to-QA schedule retained."}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
