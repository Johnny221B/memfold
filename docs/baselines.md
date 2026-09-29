# Baselines

| Method | Entrypoint |
| --- | --- |
| GRPO | `scripts/train_rq2_grpo.py`; generic QA variant `train_qa_grpo.py` |
| OPSD | `scripts/train_rq2_opsd.py`; generic QA variant `train_qa_opsd.py` |
| SDPO | `baselines/sdpo/train_8gpu.py` |
| Text OPD + GRPO | `baselines/text_opd_grpo/train_8gpu.py` |
| No-memory/full-context and fast baselines | `scripts/run_personamem_fast_baselines.py` |
| MemoryBank | `scripts/run_memorybank_persona.py` |
| MemP | `scripts/build_memp_persona_memory.py`, `prepare_memp_persona_retrieval.py`, `eval_memp_persona.py` |
| AutoCompressor | `scripts/run_autocompressor_persona.py`, `score_autocompressor_persona.py` |
| MemGen | `scripts/build_memgen_persona_chunked.py`, `train_memgen_persona_chunked_weaver.py`, `eval_memgen_persona.py` |
| xRAG-style | `xrag_personamem/prepare.py`, `run.py`, `train_openqa.py`, `eval_ood.py` |
| RA-IT-style | `radit_personamem/run.py` |

OPSD requires a separate checkout of `https://github.com/siyan-zhao/OPSD` at commit `7448751f307a9cdbcc1246dd1565a1a605b443df`, passed with `--opsd-source`. Its loader verifies the pinned source. MemGen integration likewise expects an external official MemGen checkout, including `memgen/model/weaver.py`; pass its source path through the entrypoint. Upstream projects are not vendored.

## Eight-GPU comparison recipes

```bash
python baselines/sdpo/train_8gpu.py --config baselines/sdpo/config.json --output outputs/sdpo
python baselines/text_opd_grpo/train_8gpu.py --config baselines/text_opd_grpo/config.json --output outputs/text-opd-grpo
```

Edit artifact paths in a copy of the configuration first. Relative paths are resolved from the launch directory. These are fixed PersonaMem-32K curve recipes: 489 training questions, 369 updates, 2 questions × 8 rollouts per update, 10 saved checkpoints, one teacher GPU, three student GPUs and four rollout GPUs. All eight GPUs must be free. A `--smoke --steps 2` run exercises the same worker topology on a shorter schedule. These recipes use a Qwen2.5-3B default model path; they are not automatically Qwen3-4B configurations.

Some hyperparameters are intentionally fixed in `roles.py`/`train_8gpu.py` to match the comparison protocol; `config.json` is also a protocol record, not a generic override for every field. For a new setting, change the implementation and its recorded config together. Text OPD + GRPO has no reference-policy KL branch; its zero-valued config field records that fact. The configurable reference-KL interface is implemented in MemFold on-policy optimization.

Retrieval/continuous-memory adapters are dataset/backbone adaptations. In particular xRAG-style and RA-IT-style should not be described as exact official checkpoint reproductions; consult their directory READMEs.
