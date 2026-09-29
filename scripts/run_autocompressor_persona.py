#!/usr/bin/env python3
"""Leakage-safe PersonaMem training/evaluation for the Qwen AutoCompressor port."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoConfig, AutoTokenizer

from memory_opd.baselines.autocompressor import autocompressor_class, attach_autocompressor_lora
from memory_opd.shared.personamem.data import load_contexts, load_questions, prior_context
from memory_opd.shared.personamem.evaluation import parse_strict_option
from memory_opd.shared.personamem.prompts import render_memory, render_prompt


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("train", "eval"), required=True)
    p.add_argument("--questions", type=Path, required=True)
    p.add_argument("--contexts", type=Path, required=True)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--summary-length", type=int, default=32)
    p.add_argument("--segment-length", type=int, default=1536)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--disable-thinking", action="store_true")
    p.add_argument("--limit", type=int)
    p.add_argument("--eval-split", choices=("val", "test"), default="test")
    return p.parse_args()


def load_model(args, trainable: bool):
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    config.summary_length = args.summary_length
    config.accumulate_summary = True
    cls = autocompressor_class(config)
    base = cls.from_pretrained(
        args.model, config=config, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True,
    )
    base.config.use_cache = False
    if args.checkpoint:
        model = PeftModel.from_pretrained(base, args.checkpoint, is_trainable=trainable)
    elif trainable:
        model = attach_autocompressor_lora(base, rank=16, alpha=16, dropout=0.05)
    else:
        raise ValueError("evaluation requires --checkpoint")
    if trainable:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model.cuda()


def legal_context_ids(example, contexts, tokenizer):
    text = render_memory(prior_context(example, contexts))
    return tokenizer(text, add_special_tokens=False).input_ids


def compress(model, token_ids, segment_length: int, train_last: bool):
    softprompt = None
    chunks = [token_ids[i:i + segment_length] for i in range(0, len(token_ids), segment_length)]
    if not chunks:
        return next(model.parameters()).new_zeros((1, 0, model.config.hidden_size))
    for index, chunk in enumerate(chunks):
        ids = torch.tensor(chunk, dtype=torch.long, device="cuda").unsqueeze(0)
        grad = train_last and index == len(chunks) - 1
        with torch.set_grad_enabled(grad):
            out = model(input_ids=ids, softprompt=softprompt, output_softprompt=True)
        softprompt = out.softprompt if grad else out.softprompt.detach()
    return softprompt


def qa_ids(example, tokenizer, disable_thinking: bool, answer: str | None = None):
    # The visible marker preserves the common semantic prompt; actual memory is latent.
    prompt = render_prompt(example, "(compressed into AutoCompressor latent summary vectors)")
    kwargs = {"enable_thinking": False} if disable_thinking else {}
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True,
        return_tensors="pt", **kwargs,
    )[0].tolist()
    if answer is None:
        return encoded, 0
    answer_ids = tokenizer(" " + answer, add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
    return encoded + answer_ids, len(encoded)


def train(args, model, tokenizer, examples, contexts, ids):
    by_id = {x.question_id: x for x in examples}
    order = list(ids)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.learning_rate, weight_decay=0.01
    )
    args.output.mkdir(parents=True, exist_ok=True)
    metrics = args.output / "metrics.jsonl"
    step = 0
    started = time.time()
    model.train()
    for epoch in range(args.epochs):
        random.Random(args.seed + epoch).shuffle(order)
        for qid in order:
            ex = by_id[qid]
            if ex.correct_answer is None:
                raise RuntimeError("training rows must include labels")
            optimizer.zero_grad(set_to_none=True)
            context_ids = legal_context_ids(ex, contexts, tokenizer)
            softprompt = compress(model, context_ids, args.segment_length, train_last=True)
            packed, prompt_length = qa_ids(ex, tokenizer, args.disable_thinking, ex.correct_answer)
            input_ids = torch.tensor(packed, dtype=torch.long, device="cuda").unsqueeze(0)
            out = model(input_ids=input_ids, softprompt=softprompt)
            labels = input_ids[:, 1:].clone()
            logits = out.logits[:, :-1]
            labels[:, : max(0, prompt_length - 1)] = -100
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            step += 1
            row = {"step": step, "epoch": epoch, "question_id": qid, "loss": float(loss.detach()),
                   "grad_norm": float(grad_norm), "context_tokens": len(context_ids),
                   "summary_tokens": int(softprompt.size(1)), "label_fields_read": True}
            with metrics.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
            if step % args.save_steps == 0:
                model.save_pretrained(args.output / f"checkpoint-{step:06d}")
    final = args.output / "checkpoint-final"
    model.save_pretrained(final)
    result = {"status": "completed", "steps": step, "checkpoint": str(final),
              "seconds": time.time() - started, "split": "train", "test_rows_read": False}
    (args.output / "train_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


@torch.inference_mode()
def evaluate(args, model, tokenizer, examples, contexts, ids):
    by_id = {x.question_id: x for x in examples}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    started = time.time()
    with args.output.open("w", encoding="utf-8") as handle:
        for ordinal, qid in enumerate(ids, 1):
            ex = by_id[qid]
            context_ids = legal_context_ids(ex, contexts, tokenizer)
            softprompt = compress(model, context_ids, args.segment_length, train_last=False)
            prompt_ids, _ = qa_ids(ex, tokenizer, args.disable_thinking)
            generated = []
            for _ in range(5):
                ids_tensor = torch.tensor(prompt_ids + generated, dtype=torch.long, device="cuda").unsqueeze(0)
                out = model(input_ids=ids_tensor, softprompt=softprompt)
                next_id = int(out.logits[0, -1].argmax())
                generated.append(next_id)
                partial = tokenizer.decode(generated, skip_special_tokens=True)
                if next_id == tokenizer.eos_token_id or parse_strict_option(partial) is not None:
                    break
            prediction = tokenizer.decode(generated, skip_special_tokens=True)
            row = {"question_id": qid, "method": "autocompressor_qwen_adapted", "status": "ok",
                   "input_tokens": len(prompt_ids), "source_context_tokens": len(context_ids),
                   "latent_memory_tokens": int(softprompt.size(1)), "generated_tokens": len(generated),
                   "prediction": prediction, "parsed_prediction": parse_strict_option(prediction),
                   "label_fields_read": False}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"{ordinal}/{len(ids)} {qid}", flush=True)
    print(json.dumps({"status": "completed", "questions": len(ids), "seconds": time.time() - started}), flush=True)


def main():
    args = arguments()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    split = json.loads(args.split.read_text())
    split_name = "train" if args.mode == "train" else args.eval_split
    ids = split["question_ids"][split_name]
    if args.limit is not None:
        ids = ids[:args.limit]
    # Labels are deliberately inaccessible to the evaluation process.
    examples = load_questions(args.questions, include_labels=args.mode == "train")
    contexts = load_contexts(args.contexts)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = load_model(args, trainable=args.mode == "train")
    if args.mode == "train":
        train(args, model, tokenizer, examples, contexts, ids)
    else:
        evaluate(args, model, tokenizer, examples, contexts, ids)


if __name__ == "__main__":
    main()
