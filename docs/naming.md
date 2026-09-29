# MemFold Open-Source Naming and Repository Standard

**Status:** Canonical naming specification for the public MemFold repository
**Source of truth:** The current MemFold paper and the final experiment configuration
**Scope:** Source files, Python modules, scripts, configuration keys, command-line interfaces, checkpoints, logs, documentation, and release artifacts

## 1. Purpose

The public repository must describe MemFold by the function of each training procedure, not by historical execution order. Names such as `stage1`, `stage2`, and `stage3` were useful during internal development but are ambiguous, become incorrect when procedures are inserted or reordered, and no longer match the paper.

The terms **MUST**, **SHOULD**, and **MAY** below are normative:

- **MUST**: required for the first public release.
- **SHOULD**: strongly recommended unless an implementation constraint is documented.
- **MAY**: optional.

## 2. Canonical MemFold pipeline

The canonical procedure order is:

```text
memory_writer_initialization
    -> compressor_reconstruction
    -> representation_warmup
    -> auxiliary_reasoning_adaptation
    -> reader_initialization
    -> on_policy_optimization
```

The first five procedures establish the memory interface. The final procedure optimizes how the reader uses that interface on its own generations.

| Public name | Machine identifier | Purpose | Trainable modules in the released configuration |
|---|---|---|---|
| Memory-writer initialization | `memory_writer_initialization` | Train the writer to produce query-conditioned textual memory. | Writer policy LoRA |
| Compressor reconstruction | `compressor_reconstruction` | Make compressed history states support textual-memory reconstruction. | Compressor and projector |
| Representation warmup | `representation_warmup` | Apply cross-context separation, same-context alignment, and Gram-matrix regularization. | Compressor and projector |
| Auxiliary reasoning adaptation | `auxiliary_reasoning_adaptation` | Adapt the compressor with evidence-grounded reasoning and ranking objectives. | Temporary reader LoRA, compressor, and projector; the temporary LoRA is discarded afterward |
| Reader initialization | `reader_initialization` | Initialize answer generation from self-generated textual memory and fixed-budget soft memory. | Reader policy LoRA, final resampler layer, resampler output normalization, and projector |
| On-policy optimization | `on_policy_optimization` | Jointly optimize GRPO and confidence-gated OPD on student rollouts. | Student policy LoRA only |

The public implementation MUST NOT present these procedures as numbered stages.

## 3. Mapping legacy names

Legacy names must be mapped by their behavior, objective, and trainable modules--not by the stage number alone. Different internal branches may have used the same number for different procedures.

| Legacy name or pattern | Canonical replacement |
|---|---|
| `text_sft`, `writer_sft`, `stage1_writer` | `memory_writer_initialization` |
| `initial_reconstruction`, `reconstruction_stage`, reconstruction of textual memory from compressed history | `compressor_reconstruction` |
| `representation_pretrain`, `contrastive_warmup`, separation/alignment/Gram training | `representation_warmup` |
| `reasoning_sft`, `aux_reasoning`, reasoning/ranking with a temporary LoRA | `auxiliary_reasoning_adaptation` |
| `latent_sft`, `reader_sft`, `soft_sft`, answer training from soft memory | `reader_initialization` |
| `stage3`, `stage_iii`, `rl_stage`, joint OPD-GRPO student-rollout training | `on_policy_optimization` |

If an old path is named only `stage1`, `stage2`, or `stage3`, inspect the following before renaming it:

1. the loss it computes;
2. the input representation it consumes;
3. the parameters it updates;
4. the checkpoint it loads and produces.

Do not infer the canonical name from the numeral.

## 4. Component terminology

Use the following public terms consistently.

| Concept | Canonical term | Definition |
|---|---|---|
| Text-memory generator | `memory_writer` | Maps visible history and query to textual memory, $M=e_\psi(C,x)$. |
| Fixed-budget memory component | `compressor` | Functional component that maps textual memory to $K$ soft vectors. |
| Internal latent-query network | `resampler` | The Perceiver-style module inside the compressor. Use this name when identifying exact layers or parameters. |
| Output mapping | `projector` | Maps resampler states into the reader input-embedding space. |
| Downstream policy | `reader` or `student_policy` | Generates responses from the query and soft memory. |
| Text-conditioned scoring model | `textual_memory_teacher` | Frozen model that assigns likelihoods to student-generated tokens under textual memory. It does not generate separate rollouts. |

### Compressor versus resampler

- Use **compressor** when referring to the functional memory component, its inputs/outputs, or a configuration group.
- Use **resampler** when referring to an exact internal layer or normalization module.
- Do not mechanically rename every occurrence of `resampler` to `compressor`.
- The exact reader-initialization update set is: `final resampler layer`, `resampler output normalization`, and `projector`.

## 5. Objective terminology

Use these names in code, configuration, logging, and documentation:

| Paper term | Machine identifier |
|---|---|
| Reconstruction loss | `reconstruction_loss` |
| Separation loss | `separation_loss` |
| Alignment loss | `alignment_loss` |
| Gram-matrix regularization | `gram_regularization` |
| Auxiliary reasoning loss | `auxiliary_reasoning_loss` |
| Ranking loss | `ranking_loss` |
