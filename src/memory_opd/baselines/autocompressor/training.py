"""Training helpers for the Qwen2.5 AutoCompressor adaptation."""

from __future__ import annotations

from peft import LoraConfig, TaskType, get_peft_model


OFFICIAL_LLAMA_LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")


def attach_autocompressor_lora(
    model,
    rank: int = 16,
    alpha: int = 16,
    dropout: float = 0.05,
):
    """Attach the LoRA boundary used by upstream ``run/train_llama.sh``.

    ``embed_summary`` is kept as a fully trainable and separately saved module.
    """

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=list(OFFICIAL_LLAMA_LORA_TARGETS),
        modules_to_save=["embed_summary"],
    )
    return get_peft_model(model, config)


def trainable_parameter_names(model) -> set[str]:
    return {name for name, parameter in model.named_parameters() if parameter.requires_grad}
