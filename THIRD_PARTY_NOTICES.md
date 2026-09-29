# Third-party notices

- `scripts/prepare_longmemeval_judge_batch.py` uses LongMemEval's task-specific judge prompt templates. Upstream: https://github.com/xiaowu0162/LongMemEval. Its MIT license is included in `docs/third_party/LongMemEval-LICENSE`.
- OPSD wrappers load an external checkout of https://github.com/siyan-zhao/OPSD; the upstream source and its license are not replaced by this repository's license.
- MemGen wrappers load an external official MemGen checkout. Obtain it under its upstream terms.
- SDPO, AutoCompressor, MemoryBank, MemP, xRAG-style and RA-IT-style modules are research integrations/adaptations. Their entrypoints document departures from the corresponding methods; inclusion does not imply an official release from those authors.
- Qwen models, embedding models and the PersonaMem, LoCoMo, PrefEval and LongMemEval datasets are not redistributed here. Their original licenses and access conditions apply separately.
