"""Qwen2.5 adaptation of the AutoCompressor summary-vector interface."""

from .modeling_qwen2 import AutoCompressorOutput, Qwen2AutoCompressorForCausalLM
from .modeling_qwen3 import Qwen3AutoCompressorForCausalLM
from .training import attach_autocompressor_lora, trainable_parameter_names

__all__ = [
    "AutoCompressorOutput",
    "Qwen2AutoCompressorForCausalLM",
    "Qwen3AutoCompressorForCausalLM",
    "attach_autocompressor_lora",
    "trainable_parameter_names",
]


def autocompressor_class(config):
    if config.model_type == "qwen2":
        return Qwen2AutoCompressorForCausalLM
    if config.model_type == "qwen3":
        return Qwen3AutoCompressorForCausalLM
    raise ValueError(f"unsupported AutoCompressor backbone: {config.model_type}")

__all__.append("autocompressor_class")
