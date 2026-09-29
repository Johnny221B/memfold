#!/usr/bin/env python3
"""Train the official MemGen Weaver on trajectory-level ALFWorld demonstrations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import torch

from datasets import Dataset
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from memgen.model.configuration_memgen import MemGenConfig
from memgen.model.modeling_memgen import MemGenModel
from memgen.model.modeling_utils import MemGenOutputWithPast


LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    "lora_dropout": 0.1,
    "bias": "none",
    "task_type": "CAUSAL_LM",
}


class RecoverableMemGenModel(MemGenModel):
    """Upstream MemGen serialization plus a Trainer-readable trainable state."""

    def save_pretrained(self, save_directory: str, **kwargs) -> None:
        super().save_pretrained(save_directory, **kwargs)
        trainable_state = {
            name: parameter.detach().cpu()
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }
        torch.save(trainable_state, Path(save_directory) / "pytorch_model.bin")


class FusedAssistantLossMemGenModel(RecoverableMemGenModel):
    """MemGen SFT with an exact fused CE over assistant positions only.

    Upstream materializes logits for every long-context token before ignoring all
    prompt labels. PersonaMem supervises only the final option, so this equivalent
    path keeps hidden states and applies the LM head inside fused CE only where the
    shifted label is not -100.
    """

    def _assistant_hidden_states(self, input_ids, attention_mask, labels):
        tokenizer, reasoner, weaver = self.tokenizer, self.reasoner, self.weaver
        augmentation_indices = self._select_augment_points_after_delimiter(
            input_ids, labels, self.delimiters, tokenizer,
            self.config.max_inference_aug_num,
        )
        inputs_embeds = reasoner.get_input_embeddings()(input_ids)
        batch, _, hidden_size = inputs_embeds.shape
        current_start = 0
        current_embeds = torch.empty(
            (batch, 0, hidden_size), device=self.device, dtype=inputs_embeds.dtype,
        )
        current_attention = torch.empty(
            (batch, 0), device=self.device, dtype=attention_mask.dtype,
        )
        current_latent_mask = torch.empty(
            (batch, 0), device=self.device, dtype=torch.bool,
        )
        for point in augmentation_indices:
            segment = inputs_embeds[:, current_start:point]
            segment_attention = attention_mask[:, current_start:point]
            current_embeds = torch.cat([current_embeds, segment], dim=1)
            current_attention = torch.cat([current_attention, segment_attention], dim=1)
            current_latent_mask = torch.cat([
                current_latent_mask,
                torch.zeros_like(segment_attention, dtype=torch.bool),
            ], dim=1)
            positions = self._generate_position_ids(current_attention)
            weaver_inputs = self.reasoner_to_weaver(current_embeds)
            is_prompt = (
                (labels[:, point] != -100).all()
                and (labels[:, point - 1] == -100).all().item()
            )
            if is_prompt:
                latent_hidden, latent_attention, _ = weaver.augment_prompt(
                    weaver_inputs, current_attention, positions,
                )
            else:
                latent_hidden, latent_attention, _ = weaver.augment_inference(
                    weaver_inputs, current_attention, positions,
                )
            latent_embeds = self.weaver_to_reasoner(latent_hidden)
            current_embeds = torch.cat([current_embeds, latent_embeds], dim=1)
            current_attention = torch.cat([current_attention, latent_attention], dim=1)
            current_latent_mask = torch.cat([
                current_latent_mask,
                torch.ones_like(latent_attention, dtype=torch.bool),
            ], dim=1)
            current_start = point
        remaining = inputs_embeds[:, current_start:]
        remaining_attention = attention_mask[:, current_start:]
        current_embeds = torch.cat([current_embeds, remaining], dim=1)
        current_attention = torch.cat([current_attention, remaining_attention], dim=1)
        current_latent_mask = torch.cat([
            current_latent_mask,
            torch.zeros_like(remaining_attention, dtype=torch.bool),
        ], dim=1)
        outputs = reasoner.model(
            inputs_embeds=current_embeds,
            attention_mask=current_attention,
            position_ids=self._generate_position_ids(current_attention),
        )
        shifted = torch.zeros_like(current_latent_mask)
        shifted[:, :-1] = current_latent_mask[:, 1:]
        return outputs.last_hidden_state[~shifted].view(batch, input_ids.size(1), -1)

    def forward(self, input_ids, attention_mask, labels, **kwargs):
        if input_ids.size(0) != 1:
            raise ValueError("fused PersonaMem path currently requires batch size 1")
        labels = self._postprocess_assistant_labels(input_ids, labels, self.tokenizer)
        hidden = self._assistant_hidden_states(input_ids, attention_mask, labels)
        shifted_hidden = hidden[:, :-1].reshape(-1, hidden.size(-1))
        shifted_labels = labels[:, 1:].reshape(-1)
        supervised = shifted_labels != -100
        if not supervised.any():
            raise ValueError("no shifted assistant labels")
        loss = LigerFusedLinearCrossEntropyLoss(ignore_index=-100)(
            self.reasoner.lm_head.weight,
            shifted_hidden[supervised],
            shifted_labels[supervised],
        )
        output = MemGenOutputWithPast(loss=loss, logits=None)
        output.supervised_labels = labels
        return output


class RecoverableTrainer(Trainer):
    """Compatibility shim for trusted local RNG checkpoints on Torch 2.4."""

    def _load_rng_state(self, checkpoint) -> None:
        original_load = torch.load

        def trusted_local_load(*args, **kwargs):
            kwargs["weights_only"] = False
            return original_load(*args, **kwargs)

        torch.load = trusted_local_load
        try:
            super()._load_rng_state(checkpoint)
        finally:
            torch.load = original_load


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--valid-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--valid-limit", type=int, default=284)
    parser.add_argument("--stress-longest", action="store_true")
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deepspeed", type=Path)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--use-liger-kernel", action="store_true",
        help="Use the installed fused Liger kernels for long-context training.",
    )
    parser.add_argument(
        "--share-dormant-trigger-with-reasoner",
        action="store_true",
        help=(
            "During Weaver SFT only, attach the frozen inactive Trigger adapter to the "
            "frozen Reasoner backbone instead of loading a third identical backbone."
        ),
    )
    return parser.parse_args()


def read_records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select_records(records: list[dict], limit: int | None, seed: int, longest: bool) -> list[dict]:
    if longest:
        records = sorted(records, key=lambda row: int(row["token_length"]), reverse=True)
    else:
        records = list(records)
        random.Random(seed).shuffle(records)
    return records if limit is None else records[:limit]


def model_config(model: Path) -> dict:
    name = str(model)
    return {
        "model_name": name,
        "load_model_path": None,
        "max_prompt_aug_num": 1,
        "max_inference_aug_num": 5,
        "weaver": {
            "model_name": name,
            "prompt_latents_len": 8,
            "inference_latents_len": 8,
            "lora_config": LORA_CONFIG,
        },
        "trigger": {"model_name": name, "active": False, "lora_config": LORA_CONFIG},
    }


def build_model(
    model_path: Path, share_dormant_trigger: bool, fused_assistant_loss: bool,
) -> MemGenModel:
    """Build upstream MemGen, optionally sharing the two frozen dormant backbones.

    The Trigger is inactive throughout upstream Weaver SFT.  Its LoRA B matrices are
    initialized to zero and then frozen, so attaching that dormant adapter to the
    already-frozen Reasoner has identical Weaver-SFT outputs while avoiding a third
    3B backbone allocation.  The saved Trigger adapter remains load-compatible with
    the normal three-backbone upstream model used in the later Trigger stage.
    """
    config_dict = model_config(model_path)
    if not share_dormant_trigger:
        return RecoverableMemGenModel.from_config(config_dict)

    name = str(model_path)
    config = MemGenConfig.from_pretrained(
        name,
        max_prompt_aug_num=config_dict["max_prompt_aug_num"],
        max_inference_aug_num=config_dict["max_inference_aug_num"],
        prompt_latents_len=config_dict["weaver"]["prompt_latents_len"],
        inference_latents_len=config_dict["weaver"]["inference_latents_len"],
        weaver_lora_config=config_dict["weaver"]["lora_config"],
        trigger_active=False,
        trigger_lora_config=config_dict["trigger"]["lora_config"],
    )
    tokenizer = AutoTokenizer.from_pretrained(name)
    reasoner = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    )
    weaver = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    )
    model_class = FusedAssistantLossMemGenModel if fused_assistant_loss else RecoverableMemGenModel
    model = model_class(
        config=config,
        base_tokenizer=tokenizer,
        reasoner_base_model=reasoner,
        weaver_base_model=weaver,
        trigger_base_model=reasoner,
    )
    # PEFT otherwise executes the frozen zero-initialized Trigger LoRA branch on
    # every Reasoner token and retains its dropout activations.  The upstream
    # three-backbone construction has no adapter on the Reasoner, so disabling it
    # here is both output-equivalent and necessary for memory equivalence.
    model.trigger.model.disable_adapter_layers()
    return model


def tokenize_assistant_only(tokenizer, rows: list[dict], maximum: int) -> Dataset:
    encoded_rows = []
    for row in rows:
        encoded = tokenizer.apply_chat_template(
            row["messages"], tokenize=True, return_dict=True,
            return_assistant_tokens_mask=True, truncation=True, max_length=maximum,
        )
        labels = [
            token if mask else -100
            for token, mask in zip(encoded["input_ids"], encoded["assistant_masks"])
        ]
        if not any(label != -100 for label in labels):
            raise ValueError(f"no assistant labels for {row['trajectory_id']}")
        encoded_rows.append({
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": labels,
        })
    return Dataset.from_list(encoded_rows)


class AssistantOnlyCollator:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        maximum = max(len(feature["input_ids"]) for feature in features)
        pad = int(self.tokenizer.pad_token_id)
        input_ids, attention_mask, labels = [], [], []
        for feature in features:
            padding = maximum - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [pad] * padding)
            attention_mask.append(feature["attention_mask"] + [0] * padding)
            labels.append(feature["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def main() -> int:
    args = arguments()
    train_rows = select_records(
        read_records(args.train_data), args.train_limit, args.seed, args.stress_longest
    )
    valid_rows = select_records(read_records(args.valid_data), args.valid_limit, args.seed + 1, False)
    if not train_rows or not valid_rows:
        raise ValueError("train and validation datasets must be non-empty")
    model = build_model(
        args.model, args.share_dormant_trigger_with_reasoner, args.use_liger_kernel,
    )
    model.fix_component("trigger")
    if args.gradient_checkpointing:
        model.reasoner.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.weaver.model.base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.reasoner.config.use_cache = False
        model.weaver.model.base_model.config.use_cache = False

    train_dataset = tokenize_assistant_only(model.tokenizer, train_rows, args.max_length)
    valid_dataset = tokenize_assistant_only(model.tokenizer, valid_rows, args.max_length)
    config = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        bf16=True,
        gradient_checkpointing=False,
        remove_unused_columns=False,
        eval_strategy="no" if args.max_steps > 0 else "epoch",
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        seed=args.seed,
        data_seed=args.seed,
        report_to="none",
        dataloader_num_workers=0,
        # Upstream conversational SFT samples only max_prompt_aug_num turns from
        # each trajectory. Prompt/inference latent branches can therefore be
        # conditionally unused on an individual DDP rank and step.
        ddp_find_unused_parameters=True,
        deepspeed=str(args.deepspeed) if args.deepspeed else None,
        # MemGen is a composite model, so the generic Trainer monkey-patch cannot
        # identify its inner Qwen. The fused assistant loss is applied above.
        use_liger_kernel=False,
    )
    trainer = RecoverableTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=AssistantOnlyCollator(model.tokenizer),
    )
    trainer.train(
        resume_from_checkpoint=str(args.resume_from_checkpoint)
        if args.resume_from_checkpoint else None
    )
    trainer.save_model(str(args.output_dir / "final"))
    if trainer.is_world_process_zero():
        summary = {
            "schema_version": "1.0",
            "train_records": len(train_rows),
            "valid_records": len(valid_rows),
            "global_step": trainer.state.global_step,
            "epoch": trainer.state.epoch,
            "best_model_checkpoint": trainer.state.best_model_checkpoint,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
            "log_history": trainer.state.log_history,
        }
        (args.output_dir / "run_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
