"""Long-context, algorithm-equivalent adapter for the official OPSD trainer."""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn.functional as F
from accelerate.utils import is_peft_model
from trl.trainer.utils import empty_cache


def make_memory_efficient_opsd_trainer(base_class):
    """Return an OPSDTrainer that projects logits only at completion positions.

    The pinned upstream trainer projects every prompt hidden state through the
    vocabulary head and slices the prompt logits afterwards.  At 128K this
    creates a ~36 GiB tensor for Qwen although OPSD consumes only the final
    completion positions.  Qwen's ``logits_to_keep`` performs the identical
    slice before the vocabulary projection.
    """

    class MemoryEfficientOPSDTrainer(base_class):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            student_prompt_len = inputs["student_prompt_length"]
            teacher_prompt_len = inputs["teacher_prompt_length"]
            sampled_token_ids = inputs["student_input_ids"][:, student_prompt_len:]
            shifted_labels = inputs["labels"][:, student_prompt_len:]
            completion_length = sampled_token_ids.shape[1]
            # Keep one extra causal position: logits at positions [prompt-1, ..., -2]
            # predict the completion tokens [0, ..., completion_length-1].
            logits_to_keep = completion_length + 1

            outputs_student = model(
                input_ids=inputs["student_input_ids"],
                attention_mask=inputs["student_attention_mask"],
                logits_to_keep=logits_to_keep,
            )
            student_logits_for_loss = outputs_student.logits[:, :-1, :]
            if student_logits_for_loss.shape[1] != completion_length:
                raise RuntimeError("student completion-logit slice has an unexpected length")

            if self.use_thinking_machines_loss:
                student_log_probs = F.log_softmax(student_logits_for_loss / self.temperature, dim=-1)
                student_log_probs_sampled = torch.gather(
                    student_log_probs, dim=-1, index=sampled_token_ids.unsqueeze(-1)
                ).squeeze(-1)
                del student_logits_for_loss, student_log_probs
            del outputs_student
            empty_cache()

            if self.use_ema_teacher:
                adapter_context = self._ema_teacher_context(model)
            elif self.fixed_teacher and is_peft_model(model):
                adapter_context = self.accelerator.unwrap_model(model).disable_adapter()
            else:
                adapter_context = nullcontext()

            with torch.no_grad(), adapter_context:
                outputs_teacher = model(
                    input_ids=inputs["teacher_input_ids"],
                    attention_mask=inputs["teacher_attention_mask"],
                    logits_to_keep=logits_to_keep,
                )
                teacher_logits_for_loss = outputs_teacher.logits[:, :-1, :]
                if teacher_logits_for_loss.shape[1] != completion_length:
                    raise RuntimeError("teacher completion-logit slice has an unexpected length")
                if self.use_thinking_machines_loss:
                    teacher_log_probs = F.log_softmax(teacher_logits_for_loss / self.temperature, dim=-1)
                    teacher_log_probs_sampled = torch.gather(
                        teacher_log_probs, dim=-1, index=sampled_token_ids.unsqueeze(-1)
                    ).squeeze(-1)
                    del teacher_logits_for_loss, teacher_log_probs
                del outputs_teacher
                empty_cache()

            if self.use_thinking_machines_loss:
                advantage = (teacher_log_probs_sampled - student_log_probs_sampled).detach()
                mask = shifted_labels != -100
                loss = -(advantage[mask] * student_log_probs_sampled[mask]).mean()
            else:
                loss = self.generalized_jsd_loss(
                    student_logits=student_logits_for_loss,
                    teacher_logits=teacher_logits_for_loss,
                    labels=shifted_labels,
                    beta=self.beta,
                    temperature=self.temperature,
                    top_k=self.top_k_loss,
                    token_clip=self.jsd_token_clip,
                )
            empty_cache()

            if return_outputs:
                class MinimalOutput:
                    pass

                output = MinimalOutput()
                output.loss = loss
                return loss, output
            return loss

    MemoryEfficientOPSDTrainer.__name__ = "MemoryEfficientOPSDTrainer"
    return MemoryEfficientOPSDTrainer
