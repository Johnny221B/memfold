"""Qwen3 policy for the No-Memory baseline."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList


def build_prefix_map(
    sequences: list[tuple[int, ...]], eos_token_ids: list[int]
) -> dict[tuple[int, ...], list[int]]:
    """Build allowed-next-token sets for an exact finite action trie."""
    allowed: dict[tuple[int, ...], set[int]] = {}
    for sequence in sequences:
        if not sequence:
            raise ValueError("constrained action token sequence must be non-empty")
        for index, token in enumerate(sequence):
            allowed.setdefault(sequence[:index], set()).add(token)
        allowed.setdefault(sequence, set()).update(eos_token_ids)
    return {prefix: sorted(tokens) for prefix, tokens in allowed.items()}


class QwenNoMemoryPolicy:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        max_new_tokens: int = 16,
        enable_thinking: bool = False,
        do_sample: bool = False,
        seed: int = 42,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=False,
        ).to(device)
        self.model.eval()
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = enable_thinking
        self.do_sample = do_sample
        self.seed = seed
        if self.do_sample:
            torch.manual_seed(self.seed)

    @torch.inference_mode()
    def __call__(self, system: str, user: str) -> str:
        text = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        generation = {
            "do_sample": self.do_sample,
            "temperature": 0.6 if self.do_sample else None,
            "top_p": 0.95 if self.do_sample else None,
            "top_k": 20 if self.do_sample else None,
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        output = self.model.generate(**inputs, **generation)
        generated = output[0, inputs.input_ids.shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    @torch.inference_mode()
    def constrained(self, system: str, user: str, admissible_actions: list[str]) -> str:
        """Greedily decode exactly one current admissible action."""
        if self.enable_thinking:
            raise ValueError("strict action decoding requires thinking=false")
        if not admissible_actions:
            raise ValueError("admissible_actions must not be empty")
        text = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        action_tokens: dict[tuple[int, ...], str] = {}
        for action in admissible_actions:
            sequence = tuple(self.tokenizer(action, add_special_tokens=False).input_ids)
            if sequence in action_tokens and action_tokens[sequence] != action:
                raise ValueError(f"distinct admissible actions share token sequence: {action!r}")
            action_tokens[sequence] = action
        eos = self.model.generation_config.eos_token_id
        eos_ids = [int(item) for item in (eos if isinstance(eos, list) else [eos])]
        prefix_map = build_prefix_map(list(action_tokens), eos_ids)
        longest = max(map(len, action_tokens))
        if longest + 1 > self.max_new_tokens:
            raise ValueError(
                f"admissible action requires {longest + 1} tokens including EOS, "
                f"above max_new_tokens={self.max_new_tokens}"
            )
        prompt_length = inputs.input_ids.shape[1]

        def allowed_tokens(_batch_id: int, input_ids: torch.Tensor) -> list[int]:
            prefix = tuple(int(item) for item in input_ids[prompt_length:].tolist())
            if prefix not in prefix_map:
                raise RuntimeError(f"generation left admissible-action trie at prefix {prefix}")
            return prefix_map[prefix]

        output = self.model.generate(
            **inputs, do_sample=False, temperature=None, top_p=None, top_k=None,
            max_new_tokens=self.max_new_tokens, pad_token_id=self.tokenizer.eos_token_id,
            prefix_allowed_tokens_fn=allowed_tokens,
        )
        generated = [int(item) for item in output[0, prompt_length:].tolist()]
        while generated and generated[-1] in eos_ids:
            generated.pop()
        sequence = tuple(generated)
        if sequence not in action_tokens:
            raise RuntimeError(f"constrained generation did not finish an admissible action: {sequence}")
        return action_tokens[sequence]

    @torch.inference_mode()
    def raw_prompt(self, prompt: str) -> str:
        """Generate from an already-rendered prompt, as used by MCMA."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_length = inputs.input_ids.shape[1]
        tokenizer = self.tokenizer

        class CompleteActionLine(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                generated = tokenizer.decode(input_ids[0, prompt_length:], skip_special_tokens=True)
                if "Action:" not in generated:
                    return False
                return "\n" in generated.split("Action:")[-1]

        output = self.model.generate(
            **inputs, do_sample=False, temperature=None, top_p=None, top_k=None,
            max_new_tokens=self.max_new_tokens, pad_token_id=self.tokenizer.eos_token_id,
            stopping_criteria=StoppingCriteriaList([CompleteActionLine()]),
        )
        return self.tokenizer.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
