#!/usr/bin/env python3
"""Anti-collapse PersonaMem adaptation for recurrent Qwen AutoCompressor."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoConfig, AutoTokenizer, get_cosine_schedule_with_warmup

from memory_opd.baselines.autocompressor import autocompressor_class, attach_autocompressor_lora
from memory_opd.shared.personamem.data import load_contexts, load_questions, prior_context
from memory_opd.shared.personamem.prompts import render_memory, render_prompt


LABELS = ("(a)", "(b)", "(c)", "(d)")
PREFIX_RE = re.compile(r"^\s*\([abcd]\)\s*", re.IGNORECASE)


def clean_options(options):
    return tuple(PREFIX_RE.sub("", option, count=1) for option in options)


def permuted_example(example, seed: int):
    if example.correct_answer not in LABELS:
        raise ValueError("anti-collapse training requires canonical option labels")
    order = list(range(4))
    digest = hashlib.sha256(f"{seed}:{example.question_id}".encode()).digest()
    random.Random(int.from_bytes(digest[:8], "big")).shuffle(order)
    original_gold = LABELS.index(example.correct_answer)
    new_gold = order.index(original_gold)
    cleaned = clean_options(example.options)
    return replace(
        example,
        options=tuple(cleaned[index] for index in order),
        correct_answer=LABELS[new_gold],
    )


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("train", "eval"), required=True)
    p.add_argument("--questions", type=Path, required=True)
    p.add_argument("--contexts", type=Path, required=True)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--eval-split", choices=("val", "test"), default="val")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--summary-length", type=int, default=32)
    p.add_argument("--segment-length", type=int, default=1536)
    p.add_argument("--tbptt-segments", type=int, default=4)
    p.add_argument("--compression-lm-weight", type=float, default=0.1)
    p.add_argument("--learning-rate", type=float, default=3e-5)
    p.add_argument("--gradient-accumulation", type=int, default=8)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-samples", type=int)
    p.add_argument("--save-samples", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--disable-thinking", action="store_true")
    return p.parse_args()


def load_model(args, trainable):
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    config.summary_length = args.summary_length
    config.accumulate_summary = True
    base = autocompressor_class(config).from_pretrained(
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


def context_ids(example, contexts, tokenizer):
    return tokenizer(render_memory(prior_context(example, contexts)), add_special_tokens=False).input_ids


def compress(model, ids, segment_length, *, tbptt_segments=0):
    chunks = [ids[i:i + segment_length] for i in range(0, len(ids), segment_length)]
    if not chunks:
        empty = next(model.parameters()).new_zeros((1, 0, model.config.hidden_size))
        return empty, []
    softprompt = None
    aux_losses = []
    grad_start = max(0, len(chunks) - tbptt_segments)
    for index, chunk in enumerate(chunks):
        token_ids = torch.tensor(chunk, dtype=torch.long, device="cuda").unsqueeze(0)
        grad = tbptt_segments > 0 and index >= grad_start
        with torch.set_grad_enabled(grad):
            out = model(
                input_ids=token_ids,
                labels=token_ids if grad else None,
                softprompt=softprompt,
                output_softprompt=True,
            )
        if grad:
            aux_losses.append(out.loss)
            softprompt = out.softprompt
        else:
            softprompt = out.softprompt.detach()
    return softprompt, aux_losses


def prompt_ids(example, tokenizer, disable_thinking):
    cleaned = replace(example, options=clean_options(example.options))
    prompt = render_prompt(cleaned, "(compressed into AutoCompressor latent summary vectors)")
    kwargs = {"enable_thinking": False} if disable_thinking else {}
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True,
        return_tensors="pt", **kwargs,
    )[0].tolist()


def candidate_scores(model, softprompt, prompt, tokenizer, candidates):
    """Length-normalized semantic likelihood for the four full option texts."""
    scores = []
    for candidate in candidates:
        answer = tokenizer(" " + candidate, add_special_tokens=False).input_ids
        if not answer:
            raise ValueError("empty candidate option")
        packed = torch.tensor(prompt + answer, dtype=torch.long, device="cuda").unsqueeze(0)
        out = model(input_ids=packed, softprompt=softprompt)
        log_probs = F.log_softmax(out.logits[0], dim=-1)
        positions = torch.arange(len(prompt) - 1, len(prompt) + len(answer) - 1, device="cuda")
        targets = torch.tensor(answer, dtype=torch.long, device="cuda")
        scores.append(log_probs[positions, targets].mean())
    return torch.stack(scores)


def train(args, model, tokenizer, examples, contexts, ids):
    by_id = {x.question_id: x for x in examples}
    order = list(ids)
    random.Random(args.seed).shuffle(order)
    if args.max_samples is not None:
        order = order[:args.max_samples]
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.learning_rate, weight_decay=0.01
    )
    update_steps = math.ceil(len(order) * args.epochs / args.gradient_accumulation)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, max(1, int(update_steps * args.warmup_ratio)), update_steps
    )
    args.output.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output / "metrics.jsonl"
    model.train()
    optimizer.zero_grad(set_to_none=True)
    sample_step = optimizer_step = 0
    started = time.time()
    for epoch in range(args.epochs):
        for ordinal, qid in enumerate(order, 1):
            example = permuted_example(by_id[qid], args.seed + epoch)
            ids_context = context_ids(example, contexts, tokenizer)
            softprompt, aux_losses = compress(
                model, ids_context, args.segment_length, tbptt_segments=args.tbptt_segments
            )
            scores = candidate_scores(
                model, softprompt, prompt_ids(example, tokenizer, args.disable_thinking), tokenizer,
                example.options,
            )
            target = torch.tensor([LABELS.index(example.correct_answer)], device="cuda")
            qa_loss = F.cross_entropy(scores.unsqueeze(0), target)
            lm_loss = torch.stack(aux_losses).mean() if aux_losses else qa_loss.new_zeros(())
            loss = qa_loss + args.compression_lm_weight * lm_loss
            (loss / args.gradient_accumulation).backward()
            sample_step += 1
            should_update = sample_step % args.gradient_accumulation == 0 or ordinal == len(order)
            grad_norm = None
            if should_update:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
            row = {
                "sample_step": sample_step, "optimizer_step": optimizer_step,
                "question_id": qid, "qa_loss": float(qa_loss.detach()),
                "compression_lm_loss": float(lm_loss.detach()), "loss": float(loss.detach()),
                "gold": example.correct_answer, "predicted": LABELS[int(scores.detach().argmax())],
                "context_tokens": len(ids_context), "summary_tokens": int(softprompt.size(1)),
                "tbptt_segments": args.tbptt_segments,
                "grad_norm": None if grad_norm is None else float(grad_norm),
                "learning_rate": scheduler.get_last_lr()[0], "label_fields_read": True,
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
            if sample_step % args.save_samples == 0:
                model.save_pretrained(args.output / f"checkpoint-{sample_step:06d}")
    final = args.output / "checkpoint-final"
    model.save_pretrained(final)
    result = {
        "status": "completed", "samples": sample_step, "optimizer_steps": optimizer_step,
        "checkpoint": str(final), "seconds": time.time() - started,
        "prediction_distribution": Counter(
            json.loads(line)["predicted"] for line in metrics_path.read_text().splitlines()
        ),
    }
    result["prediction_distribution"] = dict(result["prediction_distribution"])
    (args.output / "train_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


@torch.inference_mode()
def evaluate(args, model, tokenizer, examples, contexts, ids):
    by_id = {x.question_id: x for x in examples}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    with args.output.open("w", encoding="utf-8") as handle:
        for ordinal, qid in enumerate(ids, 1):
            example = replace(by_id[qid], options=clean_options(by_id[qid].options))
            ids_context = context_ids(example, contexts, tokenizer)
            softprompt, _ = compress(model, ids_context, args.segment_length)
            scores = candidate_scores(
                model, softprompt, prompt_ids(example, tokenizer, args.disable_thinking), tokenizer,
                example.options,
            )
            prediction = LABELS[int(scores.argmax())]
            row = {
                "question_id": qid, "method": "autocompressor_qwen_anticollapse",
                "prediction": prediction, "parsed_prediction": prediction,
                "candidate_log_scores": [float(x) for x in scores],
                "source_context_tokens": len(ids_context),
                "latent_memory_tokens": int(softprompt.size(1)), "label_fields_read": False,
            }
            handle.write(json.dumps(row) + "\n"); handle.flush()
            print(f"{ordinal}/{len(ids)} {qid} {prediction}", flush=True)


def main():
    args = arguments()
    if args.tbptt_segments <= 0 or args.gradient_accumulation <= 0:
        raise ValueError("TBPTT and gradient accumulation must be positive")
    random.seed(args.seed); torch.manual_seed(args.seed)
    split = json.loads(args.split.read_text())
    split_name = "train" if args.mode == "train" else args.eval_split
    ids = split["question_ids"][split_name]
    examples = load_questions(args.questions, include_labels=args.mode == "train")
    contexts = load_contexts(args.contexts)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = load_model(args, trainable=args.mode == "train")
    if args.mode == "train": train(args, model, tokenizer, examples, contexts, ids)
    else: evaluate(args, model, tokenizer, examples, contexts, ids)


if __name__ == "__main__":
    main()
