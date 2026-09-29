# Repository map

| Directory | Purpose |
| --- | --- |
| `src/memory_opd/` | Shared model components, soft-memory losses, dataset/reward utilities and baseline implementations |
| `scripts/` | PersonaMem preparation, memory extraction, training, inference and scoring entrypoints |
| `baselines/sdpo/` | Synchronous eight-GPU SDPO recipe: one teacher, three student ranks, four rollout workers |
| `baselines/text_opd_grpo/` | Same allocation for the text OPD + GRPO comparison |
| `locomo_pipeline/` | Session-based compressor/reader training, selection, on-policy optimization and open-answer evaluation |
| `prefeval_pipeline/` | Preference-task preparation, generation and token accounting |
| `xrag_personamem/`, `radit_personamem/` | Dataset-specific adapted retrieval baselines |
| `memory_extraction/` | Question-blind training-memory extraction tools |
| `tests/` | CPU tests for memory construction, compressor components and on-policy optimization losses |
| `examples/` | Portable launch templates; supply local artifact paths |

Training entrypoints use the canonical functional names. The package namespace and tensor serialization layout remain compatible with saved weights. Intermediate outputs, run logs, generated memories, weights and historical Git commits are excluded.

## Dataset-specific recipes

PersonaMem core scripts accept explicit artifact paths. LoCoMo modules are launched as `python -m locomo_pipeline.<module>` and consume the `protocol.json`, `data.json` and selection files emitted by earlier procedures. They validate fixed dataset sizes and split membership; do not remove these checks to mix training and test data. Some auxiliary LoCoMo and retrieval recipes retain relative research-run input layouts. Read their path constants before use and supply the corresponding prepared artifacts; the code release does not recreate an old server filesystem.

PrefEval scripts are launched by path, for example `python prefeval_pipeline/generate_ourmethod.py --help`. Its MemFold generation entrypoint accepts `--model`, `--checkpoint`, and `--inputs`, and records input/code/checkpoint hashes. Other baseline adapters expect checkpoint folders under `prefeval_pipeline/checkpoints/`. `backbone_config.py` documents supported backbone names. Token auditing consumes previously generated response files.

`locomo_pipeline/evaluate_longmemeval.py` is a stratified 50-question LongMemEval-S recipe, not a full leaderboard evaluation. Its full-bounded fusion variant shares these utilities. Download original benchmark data separately and respect its license.

## Moving to another server

Clone this repository, install the dependencies, download the required base models and adapter/compressor artifacts separately, and prepare benchmark data using the entrypoints in the training guide. Update local paths in launch arguments or recipe constants. API keys belong in environment variables or explicitly supplied files outside Git. No private artifact store is required by the core PersonaMem implementation.
