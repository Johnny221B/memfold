# Repository map

Start with `README.md`, then `python memfold.py --help`.

- `memfold.py` selects a procedure; it forwards arguments directly to its implementation.
- `scripts/` contains the shared PersonaMem workflow. Use the same commands for 32K and 128K.
- `src/memory_opd/` contains compressor modules, objectives, writer supervision, and data utilities.
- `memory_extraction/` contains context-only training-memory extraction.
- `prefeval/` and `locomo_pipeline/` handle dataset-specific inputs and session memory.
- `examples/` provides inference and distributed optimization launch templates.

Baseline integrations and duplicated backbone-specific PrefEval runners have been removed. Checkpoints, datasets, caches, and experimental outputs are external artifacts.

The internal namespaces `memory_opd` and `locomo_pipeline` are preserved because the published HF loading helper imports them. Compressor tensor names and `bridge.pt` serialization are unchanged. The public CLI uses short procedure names; checkpoint procedure metadata keeps its original functional names.

On a new server, install the repository, download the base model and matching HF bundle, prepare the benchmark inputs, and follow `examples/evaluate.sh`. Dataset-specific LoCoMo recipes still require their prepared protocol artifacts; they are not interchangeable with the PersonaMem compressor. See `training.md` for details.
