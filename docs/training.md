# Training and inputs

Use the functional procedure names in [the naming standard](naming.md). Start with `python memfold.py --help`; each command exposes `--help`. The dispatcher calls the scripts below directly. Train/validation/test splits must remain separate; all input artifacts are supplied locally.

| Procedure | Inputs and outputs | Trainable modules |
| --- | --- | --- |
| `memory_writer_initialization` | Prepared history/query → API-memory targets; outputs writer adapter | Writer LoRA |
| `compressor_reconstruction` | Encoder-state cache and textual-memory targets; outputs compressor weights | Resampler and projector |
| `representation_warmup` | Cached context views and reconstructed compressor; separation, alignment, Gram regularization | Resampler and projector |
| `auxiliary_reasoning_adaptation` | Compressor, questions and evidence-grounded reasoning targets; reasoning, ranking and separation losses | Temporary reader LoRA, resampler and projector |
| `reader_initialization` | Writer-generated memory, compressor and writer adapter; outputs reader and adapted compressor | Reader LoRA, final resampler layer, resampler output normalization, projector |
| `on_policy_optimization` | Initialized reader/compressor and textual teacher memories; outputs optimized reader | Student LoRA only |

## Memory extraction

1. Prepare splits with `scripts/split_data.py` and `scripts/prepare_data.py`.
2. Extract API supervision with `scripts/extract_memory.py`. It accepts `--base-url`, `--model`, and either `OPENAI_API_KEY` or `--api-key-file`. The default endpoint is `https://api.openai.com/v1`.
3. Build writer supervision with `scripts/prepare_writer_targets.py` and the `prepare_writer_inputs.py`, `prepare_memory_data.py`, and `prepare_generated_data.py` utilities as appropriate for your memory format.
4. After writer initialization, use `scripts/generate_memory.py` to obtain its own memory outputs.

The alternative `memory_extraction/extract_question_blind_memory.py` extracts context-only memory without sending questions/options/answers to the API. LoCoMo extraction is in the same directory. Extraction commands make API requests only when explicitly run; credentials and generated data are not included in this repository.

## Compressor pretraining

`train_compressor.py` has `cache`, `train`, `evaluate`, and `dry-run` modes. Use the same input representation and cache configuration across dependent procedures. The default budget is 256 soft tokens; `--token-count` supports other budgets. Reconstruction supports explicit separation, ranking and alignment weights.

`warmup_compressor.py` applies `separation_loss + alignment_weight * alignment_loss + gram_regularization_weight * gram_regularization`; alignment and Gram weights default to 0.1. Supply a reconstruction checkpoint using `--checkpoint`.

Generate evidence-grounded targets with `prepare_reasoning.py`, then run `train_reasoning.py --compressor-checkpoint ...`. Its temporary LoRA is saved for diagnostics/resuming this procedure. **Do not pass that temporary LoRA as the reader's initial adapter.** Transfer the compressor and initialize the reader from the memory-writer adapter.

The scripts retain configurable research controls. Explicitly pass learning rate, training duration and loss weights for the experiment being reproduced; defaults across scripts are not a complete paper configuration.

## Reader initialization and on-policy optimization

Reader initialization defaults to `--compressor-trainable-scope last-resampler-projector` and `--soft-output-anchor-weight 0.1`, alongside the trainable reader LoRA. The scope includes output normalization. `none` and `projector` remain explicit ablation controls. Supply matching questions, memories, reasoning/answer targets and encoder-state cache.

[The on-policy launch template](../examples/on_policy_optimization.sh) exposes artifact paths through environment variables. Its objective is `opd_weight * L_OPD + grpo_weight * L_GRPO + reference_kl_weight * L_ref`. Reference KL defaults to zero and is skipped when disabled. The textual-memory teacher scores student-generated responses; it does not sample an independent rollout. Zero-variance reward groups have zero GRPO advantage while OPD remains active.

## Dataset-specific recipes

LoCoMo entrypoints are launched as modules:

```bash
python -m locomo_pipeline.train_compressor --help
python -m locomo_pipeline.train_joint_reader_initialization --help
python -m locomo_pipeline.train_session512_on_policy_optimization --help
```

`train_compressor` uses functional procedure names in its positional argument. The separate joint reader-initialization entrypoint implements the LoRA + final resampler layer/norm/projector update. The session512 recipes use their own compressor architecture and retain dataset-specific coverage checks. The on-policy entrypoint consumes `--reader-initialization`, `--selection`, and an optional `--teacher`, and exposes `--reference-kl-weight`. The LoCoMo shaping reward is 0.75 token F1 + 0.25 exact match.

## Artifact compatibility

Public entrypoints, CLI options and new procedure metadata use functional names. The tensor serialization key `bridge`, filenames such as `bridge.pt`, and internal loading helpers remain compatible with saved compressors. They denote the complete compressor, not only its resampler. Tensor parameter names including `resampler.decoder.layers` and `resampler.output_norm` are preserved; renaming them would break weight loading. LoCoMo protocol metadata now uses `procedure`; archived numbered-procedure protocol JSON needs conversion before reuse.

The package namespaces `memory_opd` and `locomo_pipeline` remain compatible with the HF loader. The public release's zero reference-KL default and reader-initialization/warmup defaults are explicit release settings, not retroactive statements about all archived runs.

## Inference

Run `examples/evaluate.sh` from the repository root with `MODEL`, `BUNDLE`, `QUESTIONS`, `WRITER_INPUTS`, and `OUTPUT` set. The example generates memory, computes fresh encoder states, and evaluates only MemFold. For PersonaMem-32K it defaults to 50 examples and a 40960-token writer context. For 128K, set `EXAMPLES=233` and `MAX_MODEL_LEN` to the model/runtime's supported context budget; do not silently truncate histories.

The evaluator writes `rows.jsonl`, `config.json`, and `summary.json`. Five trials are deterministic option permutations under greedy decoding, not five stochastic samples.

For PrefEval, use `python prefeval/prepare.py --help` and `python prefeval/evaluate.py --help`. One evaluation implementation accepts all supported backbones through `--model` and `--checkpoint`; it accepts the public `reader/` + `bridge.pt` bundle layout.
