#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

from memory_opd.shared.personamem.data import load_contexts, load_questions, prior_context
from memory_opd.shared.personamem.evaluation import parse_strict_option
from memory_opd.shared.personamem.prompts import render_memory, render_prompt
from memory_opd.shared.personamem.retrieval import chunk_messages, select_top_k


def mean_pool(model_output, attention_mask):
    hidden = model_output.last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


@torch.inference_mode()
def dense_memory(messages, question, tokenizer, model, device, chunk_tokens, top_k, query_prefix):
    chunks = chunk_messages(
        messages,
        lambda text: len(tokenizer(text, add_special_tokens=False).input_ids),
        max_tokens=chunk_tokens,
    )
    texts = [text for _, text in chunks]
    encoded = tokenizer(
        texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
    ).to(device)
    passage_vectors = F.normalize(mean_pool(model(**encoded), encoded.attention_mask), p=2, dim=1)
    query = tokenizer(
        query_prefix + question, truncation=True, max_length=512, return_tensors="pt"
    ).to(device)
    query_vector = F.normalize(mean_pool(model(**query), query.attention_mask), p=2, dim=1)
    scores = (passage_vectors @ query_vector.T).squeeze(1).float().cpu().tolist()
    return "\n\n".join(text for _, text in select_top_k(chunks, scores, top_k=top_k))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["no_memory", "full_context", "retrieval_only_rag"], required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--retriever")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-input-tokens", type=int, default=32763)
    parser.add_argument("--chunk-tokens", type=int, default=384)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--query-prefix", default="Represent this sentence for searching relevant passages: ")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--yarn-factor", type=float)
    args = parser.parse_args()

    # Deliberate label firewall: this process never loads correct_answer values.
    examples = load_questions(args.questions, include_labels=False)
    by_id = {item.question_id: item for item in examples}
    split = json.loads(args.split.read_text(encoding="utf-8"))
    test_ids = split["question_ids"]["test"]
    if args.limit is not None:
        test_ids = test_ids[: args.limit]
    contexts = load_contexts(args.contexts)
    device = torch.device(args.device)

    retrieval_tokenizer = retrieval_model = None
    if args.method == "retrieval_only_rag":
        if not args.retriever:
            raise ValueError("--retriever is required for retrieval_only_rag")
        retrieval_tokenizer = AutoTokenizer.from_pretrained(args.retriever, local_files_only=True)
        retrieval_model = AutoModel.from_pretrained(args.retriever, local_files_only=True).to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    if args.yarn_factor is not None:
        original = int(config.max_position_embeddings)
        config.rope_scaling = {
            "rope_type": "yarn", "factor": args.yarn_factor,
            "original_max_position_embeddings": original,
        }
        config.max_position_embeddings = int(original * args.yarn_factor)
        # Avoid transformers constructing a quadratic 4-D sliding-window mask.
        config.sliding_window = None
    model = AutoModelForCausalLM.from_pretrained(
        args.model, config=config, local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(device).eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with args.output.open("w", encoding="utf-8") as handle:
        for ordinal, question_id in enumerate(test_ids):
            example = by_id[question_id]
            history = prior_context(example, contexts)
            if args.method == "no_memory":
                memory = "(none)"
            elif args.method == "full_context":
                memory = render_memory(history)
            else:
                memory = dense_memory(
                    history, example.question, retrieval_tokenizer, retrieval_model, device,
                    args.chunk_tokens, args.top_k, args.query_prefix,
                )
            prompt = render_prompt(example, memory)
            messages = [{"role": "user", "content": prompt}]
            template_kwargs = {"enable_thinking": False} if args.disable_thinking else {}
            encoded_prompt = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                **template_kwargs,
            ).to(device)
            input_ids = encoded_prompt.input_ids
            input_tokens = input_ids.shape[1]
            if input_tokens > args.max_input_tokens:
                record = {
                    "question_id": question_id, "method": args.method, "status": "overflow",
                    "input_tokens": input_tokens, "prediction": None, "parsed_prediction": None,
                }
            else:
                output = model.generate(
                    input_ids,
                    attention_mask=encoded_prompt.attention_mask,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=5,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    pad_token_id=tokenizer.eos_token_id,
                )
                generated_ids = output[0, input_tokens:]
                prediction = tokenizer.decode(generated_ids, skip_special_tokens=True)
                record = {
                    "question_id": question_id, "method": args.method, "status": "ok",
                    "input_tokens": input_tokens, "generated_tokens": len(generated_ids),
                    "prediction": prediction, "parsed_prediction": parse_strict_option(prediction),
                }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"{ordinal + 1}/{len(test_ids)} {question_id} {record['status']}", flush=True)
    print(json.dumps({"method": args.method, "questions": len(test_ids), "seconds": time.time() - started}))


if __name__ == "__main__":
    main()
