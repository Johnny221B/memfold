#!/usr/bin/env python3
"""Train a full-history QA Vanilla GRPO baseline with the pinned stack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from memory_opd.rq2_baselines.trl_adapter import load_prepared_records, trl_grpo_reward


LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--validation-jsonl", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-prompt-length", type=int, required=True)
    parser.add_argument("--max-completion-length", type=int, default=5)
    parser.add_argument("--num-train-epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument(
        "--vllm-mode",
        choices=("colocate", "server"),
        default="colocate",
        help=(
            "colocate: the vLLM engine shares every training GPU. server: generation runs "
            "on a separate `trl vllm-serve` process, which is what 128K contexts need on "
            "80GB cards -- the colocated engine alone costs ~24 GiB per training GPU."
        ),
    )
    parser.add_argument(
        "--vllm-server-base-url",
        default=None,
        help="e.g. http://node-0:8765 . Required when --vllm-mode server.",
    )
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--save-steps", type=int, default=0)
    parser.add_argument("--save-total-limit", type=int, default=0)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    train_rows = load_prepared_records(args.train_jsonl, allowed_splits={"train"})
    validation_rows = load_prepared_records(args.validation_jsonl, allowed_splits={"validation"})
    summary = {
        "method": "vanilla_grpo", "train": len(train_rows), "validation": len(validation_rows),
        "datasets": sorted({str(row.get("dataset", "personamem")) for row in train_rows}),
        "reward_types": sorted({str(row.get("reward_type", "strict_choice")) for row in train_rows}),
    }
    if args.preflight_only:
        print(json.dumps(summary, indent=2))
        return

    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def convert(rows):
        converted = [
            {
                "prompt": tokenizer.apply_chat_template(
                        [{"role": "user", "content": row["prompt"]}],
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    ),
                "answer": row["answer"],
                "question_id": row["question_id"],
                "reward_type": row.get("reward_type", "strict_choice"),
            }
            for row in rows
        ]
        lengths = [
            len(tokenizer(item["prompt"], add_special_tokens=False, truncation=False)["input_ids"])
            for item in converted
        ]
        if max(lengths) > args.max_prompt_length:
            raise ValueError(
                f"prompt length {max(lengths)} exceeds --max-prompt-length={args.max_prompt_length}; "
                "silent truncation is forbidden"
            )
        return Dataset.from_list(converted)

    if args.vllm_mode == "server" and not args.vllm_server_base_url:
        raise SystemExit("--vllm-mode server requires --vllm-server-base-url")

    training_args = GRPOConfig(
        output_dir=str(args.output),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        gradient_checkpointing=True,
        bf16=True,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        temperature=args.temperature,
        beta=0.0,
        loss_type="grpo",
        scale_rewards="group",
        use_vllm=True,
        vllm_mode=args.vllm_mode,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        **(
            {"vllm_server_base_url": args.vllm_server_base_url}
            if args.vllm_mode == "server"
            else {}
        ),
        logging_steps=10,
        save_steps=args.save_steps if args.save_steps > 0 else (1 if args.max_steps > 0 else 20),
        # A ZeRO-3 checkpoint carries the partitioned frozen base model, not just
        # the adapter, so each one is ~17 GB for a 7B backbone. Ten of those per
        # cell is 170 GB on a /home that is already 96% full and shared.
        **({"save_total_limit": args.save_total_limit} if args.save_total_limit > 0 else {}),
        report_to=args.report_to,
        run_name=args.run_name,
    )
    training_args.model_init_kwargs = {
        "attn_implementation": "flash_attention_2",
        "torch_dtype": "bfloat16",
        "use_cache": False,
    }
    peft_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=LORA_TARGETS,
        task_type="CAUSAL_LM",
    )
    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=trl_grpo_reward,
        args=training_args,
        train_dataset=convert(train_rows),
        eval_dataset=convert(validation_rows),
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    checkpoints = list(args.output.glob("checkpoint-*")) if args.output.exists() else []
    trainer.train(resume_from_checkpoint=True if checkpoints else None)
    trainer.save_model(str(args.output))


if __name__ == "__main__":
    main()
