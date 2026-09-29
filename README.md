<h1 align="center">
  <img src="docs/assets/memfold-logo.png" width="64" alt="MemFold icon">
  MemFold
</h1>
<h3 align="center">Learning Compact Soft Memory for Long-Context Personalization<br>via On-Policy Optimization</h3>

<p align="center"><b>Compact memory. Personalized answers.</b></p>

<p align="center">
  <a href="https://memfold.github.io/">
    <img src="https://img.shields.io/badge/Project-Page-1F6FEB?style=for-the-badge&amp;logo=googlechrome&amp;logoColor=white" alt="Project Page">
  </a>
  <!-- TODO: Replace only the paper href below when the arXiv URL becomes available. -->
  <a href="https://drive.google.com/file/d/1WgRcUQ7mfzxNVd74B5kLYAUUQzVIea5v/view" title="Paper manuscript; temporary Google Drive link">
    <img src="https://img.shields.io/badge/arXiv-Paper-B31B1B?style=for-the-badge&amp;logo=arxiv&amp;logoColor=white" alt="arXiv Paper">
  </a>
  <a href="https://huggingface.co/Johnny221B/memfold">
    <img src="https://img.shields.io/badge/Model-Checkpoints-FFD21E?style=for-the-badge&amp;logo=huggingface&amp;logoColor=111827" alt="Model Checkpoints">
  </a>
  <a href="https://github.com/Johnny221B/memfold">
    <img src="https://img.shields.io/badge/GitHub-Code-181717?style=for-the-badge&amp;logo=github&amp;logoColor=white" alt="GitHub Code">
  </a>
</p>

<p align="center">
  <a href="#overview">Overview</a> ·
  <a href="#results">Results</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#training-pipeline">Training</a> ·
  <a href="#citation">Citation</a>
</p>

<p align="center">
  Jingxuan Wu<sup>2,*</sup>, Yuzhe Yang<sup>1,*</sup>, Yiqiao Huang<sup>3</sup>, Chengzhi Liu<sup>1</sup>, Qingni Wang<sup>1</sup>,<br>
  Chengxuan Qian<sup>1</sup>, Shutong Wu<sup>4</sup>, Jiawei Zhang<sup>4</sup>, Xin Eric Wang<sup>1</sup><br>
  <sup>1</sup>UC Santa Barbara · <sup>2</sup>UNC Chapel Hill · <sup>3</sup>Harvard University · <sup>4</sup>UW–Madison<br>
  <sup>*</sup>Equal contribution
</p>

## News

- **2026-09-29**: The paper manuscript, code, and model checkpoints are available through the links above. The README now includes results, paper figures, and a checkpoint quick start.

## Overview

**MemFold learns compact memories through the answers they support.** It converts a user's interaction history into textual evidence, compresses that evidence into soft memory vectors, and trains a reader on responses it generates from those vectors. Task rewards and a frozen textual-memory teacher provide complementary feedback on the same student rollouts.

- **A compact reader interface.** PersonaMem experiments use 256 soft memory vectors for both 32K and 128K histories.
- **Training on the reader's own answers.** GRPO rewards successful responses; confidence-gated token supervision guides the same sampled tokens without extra teacher rollouts.
- **Personalization and transfer.** Evaluated with three Qwen backbones on PersonaMem-32K/128K, PrefEval, and LongMemEval.
- **Code and weights.** Memory extraction, compressor pretraining, reader training, evaluation, and baseline implementations are included. Nine selected reader bundles and their memory components are available on [Hugging Face](https://huggingface.co/Johnny221B/memfold).

## How it works

<p align="center">
  <img src="docs/assets/method.png" width="100%" alt="MemFold on-policy training: the student generates answers from soft memory; a frozen textual-memory teacher scores those same tokens; GRPO and gated token supervision update the student.">
</p>

1. **Build a memory.** A writer extracts useful evidence from the visible interaction history. A Perceiver-style compressor maps the textual memory into continuous vectors.
2. **Initialize the reader.** Supervised training teaches the reader to use the compressed representation to answer queries.
3. **Optimize on-policy.** The student samples responses from soft memory. GRPO supplies outcome-level rewards, while the frozen textual-memory teacher scores the student's tokens through a detached confidence gate. Only the student LoRA is updated during this phase; the teacher and compressor stay frozen.

The teacher is removed at inference. The fixed budget bounds the **reader-side memory input**; building the memory still requires processing the history. The LoCoMo-to-LongMemEval recipe uses a separate session interface: **512 soft tokens per session plus bounded textual memory**, rather than a single 256-token block for the entire history.

## Results

Accuracy (%) reported on the [project page](https://memfold.github.io/#results); higher is better. These are the reported main results, not a new evaluation of this checkout.

| Backbone | PersonaMem-32K | PersonaMem-128K | PrefEval | LongMemEval |
|---|---:|---:|---:|---:|
| Qwen2.5-3B-Instruct | **70.0** | **88.4** | **19.9** | **32.4** |
| Qwen2.5-7B-Instruct | **88.0** | **94.4** | **14.1** | **36.8** |
| Qwen3-4B | **84.0** | **89.4** | **15.2** | **38.6** |

MemFold has the highest reported accuracy across these comparisons, with a tie on Qwen2.5-7B PrefEval. On PersonaMem-128K, its advantage over the best listed baseline is **22.3, 15.9, and 23.7 percentage points** for Qwen2.5-3B, Qwen2.5-7B, and Qwen3-4B, respectively.

PrefEval uses PersonaMem-32K-trained checkpoints; LongMemEval uses LoCoMo-trained checkpoints. Neither transfer setting trains on the target dataset. Main results use greedy decoding; sampled component-ablation results use a different protocol and should not be mixed with this table. “—” denotes no valid outputs in the project's comparison.

<details>
<summary><b>Qwen2.5-3B-Instruct: full comparison</b></summary>

| Method | PersonaMem-32K | PersonaMem-128K | PrefEval | LongMemEval |
|---|---:|---:|---:|---:|
| Full Text | 46.0 | 21.9 | 12.9 | 26.6 |
| xRAG | 36.0 | 55.8 | 8.5 | 10.2 |
| AutoCompressor | 32.0 | 30.5 | — | — |
| MemGen | 54.0 | 66.1 | 13.3 | 3.8 |
| GRPO | 68.0 | 58.4 | 11.3 | 26.0 |
| OPSD | 54.0 | 29.6 | 12.8 | 26.2 |
| **MemFold** | **70.0** | **88.4** | **19.9** | **32.4** |

</details>

<details>
<summary><b>Qwen2.5-7B-Instruct: full comparison</b></summary>

| Method | PersonaMem-32K | PersonaMem-128K | PrefEval | LongMemEval |
|---|---:|---:|---:|---:|
| Full Text | 60.0 | 24.0 | 14.0 | 25.4 |
| xRAG | 62.0 | 64.9 | 10.9 | 12.2 |
| AutoCompressor | 66.0 | 30.7 | — | 9.6 |
| MemGen | 76.0 | 78.5 | 14.1 | 11.0 |
| GRPO | 70.0 | 62.2 | 14.1 | 25.0 |
| OPSD | 64.0 | 47.2 | 13.2 | 25.0 |
| **MemFold** | **88.0** | **94.4** | **14.1** | **36.8** |

</details>

<details>
<summary><b>Qwen3-4B: full comparison</b></summary>

| Method | PersonaMem-32K | PersonaMem-128K | PrefEval | LongMemEval |
|---|---:|---:|---:|---:|
| Full Text | 56.0 | 4.3 | 13.6 | 27.8 |
| xRAG | 66.0 | 65.2 | 2.8 | 6.6 |
| AutoCompressor | 58.0 | 30.8 | — | 14.2 |
| MemGen | 56.0 | 65.7 | 12.3 | 10.4 |
| GRPO | 62.0 | 65.7 | 13.8 | 27.4 |
| OPSD | 74.0 | 39.5 | 13.8 | 26.0 |
| **MemFold** | **84.0** | **89.4** | **15.2** | **38.6** |

</details>
## Training efficiency and memory analysis

<p align="center">
  <img src="docs/assets/training-efficiency.png" width="100%" alt="Qwen2.5-3B PersonaMem-32K accuracy versus optimizer updates and cumulative student rollouts for MemFold, GRPO, GRPO plus OPD, SDPO, and OPSD.">
</p>

**Learning from shared rollouts.** On PersonaMem-32K with Qwen2.5-3B, MemFold improves accuracy faster per optimizer update and reaches comparable accuracy with fewer student rollouts. These curves compare full recipes with different inputs and initializations. They measure updates and student rollouts, **not wall-clock time**; initialization and teacher forward-pass costs are not included in the rollout count.

<p align="center">
  <img src="docs/assets/memory-analysis.png" width="100%" alt="Qwen3-4B analysis: accuracy peaks at 256 memory tokens, token cost is dominated by history processing, and matched memory outperforms shuffled or null memory.">
</p>

**The memory content matters.** Qwen3-4B peaks at a 256-token budget in the tested recipe. Shuffling or removing its memory sharply reduces accuracy on both PersonaMem settings. Changing the soft-token budget has a comparatively small effect on end-to-end token-equivalent cost because the history-reading pass remains necessary.

<details>
<summary><b>A personalization example: respecting a user's preference</b></summary>

<p align="center">
  <img src="docs/assets/personalization-case.svg" width="100%" alt="PrefEval case: GRPO and OPSD recommend wearables despite the user's implied dislike of fitness trackers; MemFold suggests a journal, measurements, and progress photos.">
</p>

In this PrefEval example with Qwen3-4B, the user's history implies a dislike of wearable technology. GRPO and OPSD still recommend fitness trackers; MemFold offers compatible alternatives. The displayed preference summarizes the history and is not supplied explicitly to the models. Answer-token counts describe this example only and do not include memory construction or establish a dataset-wide efficiency gain.

</details>

## Quick start

### 1. Install

Use Python 3.11 and a CUDA-compatible PyTorch installation. The training stack used PyTorch 2.8.0; pinned library dependencies are listed in [pyproject.toml](pyproject.toml).

```bash
git clone https://github.com/Johnny221B/memfold.git
cd memfold
pip install -e '.[test]'
# Optional dependencies for baseline integrations:
pip install -e '.[baselines]'
```

### 2. Download a checkpoint bundle

The [weight repository](https://huggingface.co/Johnny221B/memfold) contains the following selected main-result bundles for **Qwen3-4B, Qwen2.5-3B-Instruct, and Qwen2.5-7B-Instruct**:

| HF directory | Contents |
|---|---|
| `personamem-32k/<backbone>/` | Reader LoRA + 256-token bridge |
| `personamem-128k/<backbone>/` | Reader LoRA + 256-token bridge |
| `locomo/<backbone>/` | Reader LoRA + session compressor + mapper |
| `shared/locomo-writer-qwen3-4b/` | Shared writer used by the released LongMemEval configurations |

Backbone directory names are `qwen3-4b`, `qwen2.5-3b`, and `qwen2.5-7b`. For example:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Johnny221B/memfold",
    revision="a5548705e97e137942268e92c1209fb8517bc316",
    allow_patterns=[
        "personamem-32k/qwen3-4b/*",
        "load_components.py", "manifest.json", "VALIDATION.json",
        "LICENSE.md", "NOTICE", "licenses/*",
    ],
    local_dir="checkpoints/memfold",
)
```

### 3. Check the memory components

From the repository root:

```bash
python checkpoints/memfold/load_components.py   --bundle checkpoints/memfold/personamem-32k/qwen3-4b   --code .
```

This checks **CPU loading of the memory components**, not end-to-end answer generation. The HF helper also exposes `load_reader(bundle, **model_kwargs)` for PEFT loading. Use the matching base-model revision and tokenizer in each `bundle.json`, then follow the memory-generation, encoding, and evaluation steps in the [training and input guide](docs/training.md). A reader LoRA alone does not supply the soft-memory prefix.

Base models, benchmark data, generated memories, and private experiment outputs are not bundled with this code. API credentials are supplied through environment variables or explicit key files.

## Training pipeline

The release uses functional procedure names throughout the public entrypoints.

| Procedure | Purpose | Entrypoint in `scripts/` |
|---|---|---|
| Memory-writer initialization | Learn to produce useful textual memories | `train_memory_writer_initialization.py` |
| Compressor reconstruction | Learn the soft-memory interface from text | `train_compressor_reconstruction.py` |
| Representation warmup | Shape memory representations with separation, alignment, and Gram regularization | `train_representation_warmup.py` |
| Auxiliary reasoning adaptation | Adapt the compressor using evidence-grounded reasoning | `train_auxiliary_reasoning_adaptation.py` |
| Reader initialization | Teach the reader to consume the soft-memory interface | `train_reader_initialization.py` |
| On-policy optimization | Improve student-generated answers with task rewards and token guidance | `train_on_policy_optimization.py` |

Extract API training memories with `scripts/extract_personamem_api_memory.py`; context-only and LoCoMo extraction tools are in `memory_extraction/`. Set `OPENAI_API_KEY` or pass a key file. Generate the trained writer's own memories with `scripts/generate_personamem_self_memory.py`.

Compressor pretraining comprises reconstruction, representation warmup, and auxiliary reasoning adaptation. The temporary reasoning LoRA is discarded before reader initialization. Reader initialization updates the reader LoRA, final resampler layer, output normalization, and projector. On-policy optimization updates only the student LoRA; **reference KL defaults to 0**, with the interface retained. Current defaults are not a substitute for the configuration of a particular historical experiment.

```bash
python scripts/extract_personamem_api_memory.py --help
python scripts/train_compressor_reconstruction.py --help
python scripts/train_reader_initialization.py --help
python scripts/train_on_policy_optimization.py --help
pytest -q
```

For the required artifacts and dataset-specific recipes, see [training and inputs](docs/training.md). A portable launch template is available in [examples/on_policy_optimization.sh](examples/on_policy_optimization.sh).

## Repository guide

| Directory / guide | What to find |
|---|---|
| [`src/memory_opd/`](src/memory_opd/) | Shared memory components, objectives, and data utilities |
| [`scripts/`](scripts/) | PersonaMem preparation, training, inference, and scoring |
| [`memory_extraction/`](memory_extraction/) | Training-memory extraction tools |
| [`locomo_pipeline/`](locomo_pipeline/) | Session-memory training and LongMemEval evaluation |
| [`prefeval_pipeline/`](prefeval_pipeline/) | Preference-task generation and token accounting |
| [`baselines/`](baselines/) · [baseline guide](docs/baselines.md) | Baseline integrations and comparison recipes |
| [Repository map](docs/repository.md) | Artifact compatibility, dataset-specific layouts, and migration notes |

Commands run from the repository root. The Python import namespace remains `memory_opd` for checkpoint and code compatibility. Some dataset-specific research recipes require prepared artifacts and local path configuration; they are not a one-command reproduction of every reported experiment.

## Citation

```bibtex
@misc{wu2026memfold,
  title  = {{MemFold}: Learning Compact Soft Memory for Long-Context Personalization via On-Policy Optimization},
  author = {Jingxuan Wu and Yuzhe Yang and Yiqiao Huang and Chengzhi Liu and Qingni Wang and Chengxuan Qian and Shutong Wu and Jiawei Zhang and Xin Eric Wang},
  year   = {2026},
  url    = {https://memfold.github.io/}
}
```

## License and acknowledgments

Code is released under the [MIT license](LICENSE). See [third-party notices](THIRD_PARTY_NOTICES.md) for upstream implementations and dataset terms. Model weights remain subject to their upstream licenses: Qwen2.5-3B uses the Qwen Research License, while Qwen3-4B and Qwen2.5-7B use Apache-2.0. Details are included in the [weight release](https://huggingface.co/Johnny221B/memfold/blob/main/LICENSE.md).

Figures come from the supplied paper artwork and the project website; their sources and unchanged-file hashes are recorded in [figure provenance](docs/assets/README.md).
