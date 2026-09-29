# xRAG-style PersonaMem pilot

This is a Qwen3/PersonaMem adaptation of the official NeurIPS 2024 xRAG method.
It retains the frozen retriever, frozen LLM, `mlp2x_gelu` modality bridge, and
reconstruction-pretraining then QA-finetuning schedule. It changes the official
Mistral/SFR and open-domain QA stack to Qwen3-4B, local GTE-large embeddings, and
PersonaMem message-level retrieval. Consequently, results must be labelled
**xRAG-style**, not an exact official checkpoint reproduction.

The source-train contexts are disjoint from validation and test contexts. Target
context embeddings are built by the frozen retriever only and never used for
gradient updates.
