# MemFold

MemFold learns a fixed-budget soft memory and trains a reader to answer from it. This repository includes memory extraction, compressor pretraining, reader training, inference, and baseline implementations.

[Model weights on Hugging Face](https://huggingface.co/Johnny221B/memfold) · [Code](https://github.com/Johnny221B/memfold)

The weight release includes selected PersonaMem-32K, PersonaMem-128K and LoCoMo-trained reader adapters with matching memory components for Qwen3-4B and Qwen2.5-3B/7B. See the model card for the checkpoint layout, loading instructions and upstream model licenses.

## Installation

Use Python 3.11 and a CUDA-compatible PyTorch installation (the training stack used PyTorch 2.8.0).

```bash
git clone https://github.com/Johnny221B/memfold.git
cd memfold
pip install -e '.[test]'
# Optional baseline dependencies
pip install -e '.[baselines]'
```

## Training pipeline

| Procedure | Entrypoint in `scripts/` |
| --- | --- |
| Memory-writer initialization | `train_memory_writer_initialization.py` |
| Compressor reconstruction | `train_compressor_reconstruction.py` |
| Representation warmup | `train_representation_warmup.py` |
| Auxiliary reasoning adaptation | `train_auxiliary_reasoning_adaptation.py` |
| Reader initialization | `train_reader_initialization.py` |
| On-policy optimization | `train_on_policy_optimization.py` |

To extract training memories with an API, use `scripts/extract_personamem_api_memory.py`; context-only extraction tools are in `memory_extraction/`. Supply `OPENAI_API_KEY` or an explicit key file. Generate the trained writer's own memories with `scripts/generate_personamem_self_memory.py`.

Compressor pretraining includes reconstruction, representation warmup, and auxiliary reasoning adaptation. The temporary reasoning LoRA is discarded before reader initialization. Reader initialization updates the reader LoRA, final resampler layer, resampler output normalization, and projector. On-policy optimization updates only the student LoRA; reference KL defaults to **0**, with the interface retained.

```bash
python scripts/extract_personamem_api_memory.py --help
python scripts/train_compressor_reconstruction.py --help
python scripts/train_on_policy_optimization.py --help
pytest -q
```

See [training and inputs](docs/training.md), [baselines](docs/baselines.md), and [repository structure](docs/repository.md). Download datasets, base models and checkpoints separately. Commands run from the repository root; the Python import namespace remains `memory_opd`.

[MIT license](LICENSE) · [Third-party notices](THIRD_PARTY_NOTICES.md)
