#!/usr/bin/env python3
"""Train and evaluate a RA-DIT LM-ft (RA-IT) baseline on PersonaMem."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup


LETTERS = ["(a)", "(b)", "(c)", "(d)"]


def cap_words(text: str, maximum: int) -> str:
    return " ".join(text.split()[:maximum])


def permute_row(row: dict, rng: random.Random) -> dict:
    result = dict(row)
    order = list(range(len(row["options"])))
    rng.shuffle(order)
    gold = LETTERS.index(row["answer"])
    result["options"] = [row["options"][i] for i in order]
    result["answer"] = LETTERS[order.index(gold)]
    return result


def messages(row: dict, passage: str) -> list[dict]:
    options = "\n".join(f"{LETTERS[i]} {value}" for i, value in enumerate(row["options"]))
    return [
        {
            "role": "system",
            "content": (
                "Answer the multiple-choice question using the retrieved background when it is relevant. "
                "Ignore irrelevant or misleading background. Answer with only (a), (b), (c), or (d)."
            ),
        },
        {
            "role": "user",
            "content": f"Background: {passage}\n\nQuestion: {row['question']}\n\nOptions:\n{options}",
        },
    ]


def encode_prompt(tokenizer, row: dict, passage: str) -> list[int]:
    return tokenizer.apply_chat_template(
        messages(row, passage), tokenize=True, add_generation_prompt=True, enable_thinking=False
    )


class PassageDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer, top_k: int, max_words: int, seed: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.top_k = top_k
        self.max_words = max_words
        self.seed = seed
        self.epoch = 0
        self.examples = [(i, j) for i, row in enumerate(rows) for j in range(min(top_k, len(row["retrieved_question_texts"])))]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[list[int], list[int]]:
        row_index, passage_index = self.examples[index]
        rng = random.Random(self.seed + self.epoch * 1_000_003 + index)
        row = permute_row(self.rows[row_index], rng)
        passage = cap_words(row["retrieved_question_texts"][passage_index], self.max_words)
        prompt = encode_prompt(self.tokenizer, row, passage)
        target = self.tokenizer.encode(row["answer"], add_special_tokens=False) + [self.tokenizer.eos_token_id]
        return prompt + target, [-100] * len(prompt) + target


class Collator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, examples):
        width = max(len(ids) for ids, _ in examples)
        input_ids, labels, masks = [], [], []
        for ids, target in examples:
            pad = width - len(ids)
            input_ids.append(ids + [self.pad_id] * pad)
            labels.append(target + [-100] * pad)
            masks.append([1] * len(ids) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }


def sequence_logprob(model, prompt: list[int], targets: list[list[int]], device: str) -> tuple[torch.Tensor, int]:
    sequences = [prompt + target for target in targets]
    width = max(len(x) for x in sequences)
    ids = torch.full((len(sequences), width), model.config.pad_token_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, sequence in enumerate(sequences):
        ids[i, : len(sequence)] = torch.tensor(sequence, device=device)
        mask[i, : len(sequence)] = 1
    logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits.float()
    values = []
    for i, target in enumerate(targets):
        start = len(prompt)
        token_logits = logits[i, start - 1 : start + len(target) - 1]
        token_ids = torch.tensor(target, device=device)
        values.append(F.log_softmax(token_logits, dim=-1).gather(1, token_ids[:, None]).sum())
    return torch.stack(values), sum(len(sequence) for sequence in sequences)


@torch.inference_mode()
def evaluate(
    model, tokenizer, rows: list[dict], top_k: int, max_words: int, device: str,
    output: Path, memory_mode: str = "own",
) -> dict:
    model.eval()
    targets = [tokenizer.encode(letter, add_special_tokens=False) for letter in LETTERS]
    results = []
    started = time.time()
    for number, row in enumerate(rows, 1):
        passage_logprobs, retriever_scores = [], []
        logical_prompt_tokens = 0
        scored_model_tokens = 0
        if memory_mode == "own":
            passages = row["retrieved_question_texts"][:top_k]
            scores = row["retrieved_question_scores"][:top_k]
        elif memory_mode == "null":
            passages = ["No retrieved background is available."]
            scores = [0.0]
        elif memory_mode == "shuffled":
            donor = rows[(number % len(rows))]
            passages = donor["retrieved_question_texts"][:top_k]
            scores = donor["retrieved_question_scores"][:top_k]
        else:
            raise ValueError(memory_mode)
        for passage, score in zip(passages, scores):
            prompt = encode_prompt(tokenizer, row, cap_words(passage, max_words))
            logprob, model_tokens = sequence_logprob(model, prompt, targets, device)
            passage_logprobs.append(F.log_softmax(logprob, dim=0))
            retriever_scores.append(score)
            logical_prompt_tokens += len(prompt) + max(len(x) for x in targets)
            scored_model_tokens += model_tokens
        retrieval_weights = F.softmax(torch.tensor(retriever_scores, device=device), dim=0)
        mixture = torch.logsumexp(
            torch.stack(passage_logprobs) + retrieval_weights.log().unsqueeze(1), dim=0
        )
        predicted = LETTERS[int(mixture.argmax())]
        online_logical = row["query_tokens_question"] + logical_prompt_tokens
        online_scored = row["query_tokens_question"] + scored_model_tokens
        results.append({
            "question_id": row["question_id"],
            "gold": row["answer"],
            "predicted": predicted,
            "correct": predicted == row["answer"],
            "choice_logprob": {letter: float(value) for letter, value in zip(LETTERS, mixture)},
            "retrieved_texts": passages,
            "retrieved_scores": scores,
            "memory_mode": memory_mode,
            "query_encoder_tokens": row["query_tokens_question"],
            "context_encoder_tokens": row["context_encoder_tokens"],
            "logical_answer_tokens": logical_prompt_tokens,
            "actual_scoring_tokens": scored_model_tokens,
            "online_logical_tokens": online_logical,
            "online_actual_scoring_tokens": online_scored,
        })
        if number % 25 == 0 or number == len(rows):
            print(json.dumps({
                "eval": number,
                "total": len(rows),
                "correct": sum(x["correct"] for x in results),
                "elapsed_seconds": round(time.time() - started, 1),
            }), flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in results), encoding="utf-8")
    unique_context_cost = {}
    for row, result in zip(rows, results):
        unique_context_cost[row["shared_context_id"]] = max(
            unique_context_cost.get(row["shared_context_id"], 0), result["context_encoder_tokens"]
        )
    logical = [x["online_logical_tokens"] for x in results]
    actual = [x["online_actual_scoring_tokens"] for x in results]
    return {
        "questions": len(results),
        "correct": sum(x["correct"] for x in results),
        "accuracy": sum(x["correct"] for x in results) / len(results),
        "mean_online_logical_tokens": sum(logical) / len(logical),
        "mean_online_actual_scoring_tokens": sum(actual) / len(actual),
        "mean_amortized_end_to_end_logical_tokens": (
            sum(logical) + sum(unique_context_cost.values())
        ) / len(logical),
        "mean_strict_end_to_end_logical_tokens": sum(
            value + result["context_encoder_tokens"] for value, result in zip(logical, results)
        ) / len(logical),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--train-top-k", type=int, default=3)
    parser.add_argument("--eval-top-k", type=int, default=10)
    parser.add_argument("--max-passage-words", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=980406)
    parser.add_argument("--log-steps", type=int, default=20)
    parser.add_argument("--max-train-questions", type=int, default=0)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--eval-memory-mode", choices=["own", "null", "shuffled"], default="own")
    parser.add_argument("--eval-only-adapter", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(args.device)
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id

    if args.eval_only_adapter:
        model = PeftModel.from_pretrained(model, args.eval_only_adapter).to(args.device)
    elif not args.skip_train:
        lora = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora)
        model.print_trainable_parameters()

        train_rows = [x for x in cache["questions"] if x["split"] == "train"]
        if args.max_train_questions:
            train_rows = train_rows[: args.max_train_questions]
        dataset = PassageDataset(train_rows, tokenizer, args.train_top_k, args.max_passage_words, args.seed)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=Collator(tokenizer.pad_token_id),
            num_workers=2,
            pin_memory=True,
        )
        update_steps = math.ceil(len(loader) / args.grad_accum) * args.epochs
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        scheduler = get_linear_schedule_with_warmup(optimizer, max(1, int(update_steps * 0.03)), update_steps)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        update = 0
        for epoch in range(args.epochs):
            dataset.set_epoch(epoch)
            for micro_step, batch in enumerate(loader, 1):
                batch = {key: value.to(args.device, non_blocking=True) for key, value in batch.items()}
                loss = model(**batch, use_cache=False).loss / args.grad_accum
                loss.backward()
                if micro_step % args.grad_accum == 0 or micro_step == len(loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    update += 1
                    if update % args.log_steps == 0 or update == update_steps:
                        print(json.dumps({
                            "epoch": epoch + 1,
                            "update": update,
                            "updates": update_steps,
                            "loss": float(loss) * args.grad_accum,
                            "lr": scheduler.get_last_lr()[0],
                        }), flush=True)
        model.save_pretrained(args.output / "adapter")

    started = time.time()
    validation = evaluate(
        model, tokenizer,
        [x for x in cache["questions"] if x["split"] == "validation"],
        args.eval_top_k, args.max_passage_words, args.device, args.output / "validation.jsonl",
        args.eval_memory_mode,
    )
    test = evaluate(
        model, tokenizer,
        [x for x in cache["questions"] if x["split"] == "test"],
        args.eval_top_k, args.max_passage_words, args.device, args.output / "test.jsonl",
        args.eval_memory_mode,
    )
    summary = {
        "schema": "rait-personamem-result-v1",
        "method": "RA-DIT LM-ft / RA-IT adaptation",
        "dataset": cache["dataset"],
        "backbone": args.model.name,
        "retriever": "fixed GTE-large",
        "memory_mode": args.eval_memory_mode,
        "validation": validation,
        "test": test,
        "evaluation_elapsed_seconds": time.time() - started,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
