from __future__ import annotations


class TransformersGenerator:
    def __init__(self, model_path: str, max_input_tokens: int = 32768):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype="auto", device_map="auto", local_files_only=True
        ).eval()
        self.max_input_tokens = max_input_tokens

    def generate(self, prompt: str, max_new_tokens: int) -> tuple[str, int]:
        messages = [{"role": "user", "content": prompt}]
        rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = self.tokenizer(
            rendered,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        ).to(self.model.device)
        input_tokens = int(encoded.input_ids.shape[1])
        with self.torch.inference_mode():
            output = self.model.generate(
                **encoded,
                do_sample=False,
                num_beams=1,
                temperature=None,
                top_p=None,
                top_k=None,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(output[0, input_tokens:], skip_special_tokens=True).strip()
        return text, input_tokens
