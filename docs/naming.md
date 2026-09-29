# Names used in MemFold

Use one workflow, independent of context length or research question.

| CLI | Training procedure |
|---|---|
| `train writer` | Memory-writer initialization |
| `train compressor` | Compressor reconstruction |
| `train warmup` | Representation warmup |
| `train reasoning` | Auxiliary reasoning adaptation |
| `train reader` | Reader initialization |
| `train optimize` | On-policy optimization |

The **writer** generates textual memory. The **compressor** maps it to soft tokens using a resampler and projector. The **reader** answers from those tokens. During optimization, a frozen text-memory teacher scores the reader's sampled answers.

Short command names do not change objectives, parameter update scopes, tensor names, or checkpoint metadata. Internal package names remain compatible with the published HF loader.
