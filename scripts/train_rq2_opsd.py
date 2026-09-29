#!/usr/bin/env python3
"""Train a full-history QA OPSD baseline with the pinned official trainer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from memory_opd.rq2_baselines.trl_adapter import PersonaMemOPSDDataCollator, load_prepared_records
from memory_opd.rq2_baselines.memory_efficient_opsd import make_memory_efficient_opsd_trainer
from memory_opd.rq2_baselines.runtime import verify_opsd_source


PINNED_OPSD_COMMIT = "7448751f307a9cdbcc1246dd1565a1a605b443df"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--opsd-source", type=Path, required=True)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--max-completion-length", type=int, default=5)
    parser.add_argument("--num-train-epochs", type=float, default=30.0)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--temperature", type=float, default=1.1)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.3)
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--run-name")
    parser.add_argument("--save-steps", type=int, default=0)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    train_rows = load_prepared_records(args.train_jsonl, allowed_splits={"train"})
    # OPSDTrainer subclasses TRL's SFTTrainer.  The parent performs its own
    # prompt/completion preprocessing before invoking our long-context
    # collator, so retain the benchmark answer under TRL's required key too.
    for row in train_rows:
        row["completion"] = row["answer"]
    summary = {
        "method": "opsd", "train": len(train_rows), "opsd_source": str(args.opsd_source),
        "datasets": sorted({str(row.get("dataset", "personamem")) for row in train_rows}),
        "reward_types": sorted({str(row.get("reward_type", "strict_choice")) for row in train_rows}),
    }
    if args.preflight_only:
        print(json.dumps(summary, indent=2))
        return

    if not (args.opsd_source / "opsd_trainer.py").is_file():
        raise FileNotFoundError(f"missing official opsd_trainer.py under {args.opsd_source}")
    verify_opsd_source(args.opsd_source, PINNED_OPSD_COMMIT)
    sys.path.insert(0, str(args.opsd_source))
    from datasets import Dataset
    from opsd_trainer import OPSDTrainer as OfficialOPSDTrainer
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl.experimental.gold import GOLDConfig

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    training_args = GOLDConfig(
        output_dir=str(args.output),
        learning_rate=args.learning_rate,
        max_grad_norm=0.1,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        gradient_checkpointing=True,
        bf16=True,
        max_completion_length=args.max_completion_length,
        max_length=args.max_length,
        beta=0.0,
        lmbda=1.0,
        temperature=args.temperature,
        top_p=0.95,
        top_k=20,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_tensor_parallel_size=1,
        logging_steps=2,
        save_steps=args.save_steps if args.save_steps > 0 else (1 if args.max_steps > 0 else 25),
        report_to=args.report_to,
        run_name=args.run_name,
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
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
    OPSDTrainer = make_memory_efficient_opsd_trainer(OfficialOPSDTrainer)
    trainer = OPSDTrainer(
        model=args.model,
        args=training_args,
        train_dataset=Dataset.from_list(train_rows),
        eval_dataset=None,
        processing_class=tokenizer,
        data_collator=PersonaMemOPSDDataCollator(tokenizer, max_length=args.max_length),
        peft_config=peft_config,
        use_thinking_machines_loss=False,
        fixed_teacher=True,
        reason_first=False,
        top_k_loss=None,
        jsd_token_clip=0.05,
        use_ema_teacher=False,
        student_thinking=False,
        teacher_thinking=False,
    )
    checkpoints = list(args.output.glob("checkpoint-*")) if args.output.exists() else []
    trainer.train(resume_from_checkpoint=True if checkpoints else None)
    trainer.save_model(str(args.output))


if __name__ == "__main__":
    main()
