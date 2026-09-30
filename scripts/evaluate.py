"""Evaluate MemFold on PersonaMem with greedy decoding and fixed option permutations."""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer
from memory_opd.compressed_opd import (
    deterministic_option_order, load_compressed_opd_examples,
    permute_options, soft_reader_messages,
)
from memory_opd.data.rewards import parse_choice
from policy_utils import cached_soft_memory, chat_ids, load_bridge, load_policy, rollout_student

FINAL_ANSWER = re.compile(r"(?i)final\s+answer\s*:\s*(\([a-d]\))")
LEADING_OPTION = re.compile(r"(?i)^\s*(\([a-d]\))")


def parsed_choice(text: str) -> str | None:
    # Same explicit answer-format normalization as the historical 128K protocol.
    from answer_parser import normalized_choice
    return parse_choice(text) or normalized_choice(text)

def soft_prefix(
    model: Any,
    tokenizer: Any,
    soft: torch.Tensor,
    question: str,
    options: tuple[str, str, str, str],
    device: torch.device,
    response_format: str,
) -> torch.Tensor:
    messages = soft_reader_messages(question, options)
    if response_format == "reasoning-answer":
        rendered = "\n".join(
            "({}) {}".format(chr(97 + index), text)
            for index, text in enumerate(options)
        )
        messages = [
            {"role": "system", "content": (
                "Continuous soft-memory tokens precede this conversation. "
                "Use them as the only memory evidence."
            )},
            {"role": "user", "content": (
                "Give concise evidence-grounded reasoning and end with "
                "`Final answer: (a)`, `(b)`, `(c)`, or `(d)`.\n\n"
                f"QUESTION:\n{question}\n\nOPTIONS:\n{rendered}"
            )},
        ]
    prompt = chat_ids(tokenizer, messages, device)
    return torch.cat((soft, model.get_input_embeddings()(prompt)), dim=1)

def decode_response(tokenizer: Any, response: torch.Tensor, mask: torch.Tensor) -> str:
    return tokenizer.decode(response[0][mask[0].bool()], skip_special_tokens=True).strip()

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--compressor', type=Path, required=True)
    parser.add_argument('--questions', type=Path, required=True)
    parser.add_argument('--memories', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', default='test')
    parser.add_argument('--trials', type=int, default=5)
    parser.add_argument('--max-tokens', type=int, default=5)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--response-format', choices=['label', 'reasoning-answer'], default='label')
    args = parser.parse_args()
    if min(args.trials, args.max_tokens) <= 0:
        parser.error('trials and max-tokens must be positive')
    if args.output.exists():
        parser.error('output already exists; use a new directory')
    examples = load_compressed_opd_examples(args.questions, args.memories, expected_split=args.split)
    if not examples:
        parser.error('no evaluation examples')
    torch.set_num_threads(4)
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_policy(args.model, args.adapter, device, trainable=False)
    compressor, config = load_bridge(args.compressor, device)
    args.output.mkdir(parents=True)
    (args.output / 'config.json').write_text(json.dumps(vars(args), default=str, indent=2) + '\n')
    start = time.monotonic()
    rows = []
    with (args.output / 'rows.jsonl').open('w') as out, torch.inference_mode():
        for index, example in enumerate(examples, 1):
            soft = cached_soft_memory(args.cache, example, compressor, device)
            for trial in range(args.trials):
                order = deterministic_option_order(example.question_id, trial)
                options, expected = permute_options(example.options, example.gold_label, order)
                prefix = soft_prefix(model, tokenizer, soft, example.question, options, device, args.response_format)
                response, mask = rollout_student(model, prefix, tokenizer, args.max_tokens)
                text = decode_response(tokenizer, response, mask)
                parsed = parsed_choice(text)
                row = dict(question_id=example.question_id, trial=trial, option_order=order,
                           expected=expected, output=text, parsed=parsed, correct=parsed == expected,
                           output_tokens=int(mask.sum()))
                rows.append(row)
                out.write(json.dumps(row) + '\n')
                out.flush()
            print(f'{index}/{len(examples)}', flush=True)
    trials = []
    for trial in range(args.trials):
        group = [row for row in rows if row['trial'] == trial]
        correct = sum(row['correct'] for row in group)
        trials.append(dict(trial=trial, correct=correct, total=len(group), accuracy=correct / len(group)))
    summary = dict(examples=len(examples), trials=trials,
                   mean_accuracy=sum(t['accuracy'] for t in trials) / len(trials),
                   best_accuracy=max(t['accuracy'] for t in trials),
                   parsed_rate=sum(row['parsed'] is not None for row in rows) / len(rows),
                   soft_tokens=config['token_count'], seconds=time.monotonic() - start,
                   decoding='greedy; trials permute answer options')
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
