# RA-IT text-space baseline for PersonaMem

This directory implements the language-model fine-tuning (`LM-ft`, also called
`RA-IT`) stage of RA-DIT for the local PersonaMem 32K and 128K splits.

The implementation deliberately keeps the existing GTE-large retriever frozen
so that the experiment isolates whether a source-trained reasoner can use and
generalize with **text-space memory**.  Following RA-DIT, each retrieved passage
is prepended as a `Background` field and becomes an independent training
instance.  At inference, predictions from the top-k passages are mixed using
the normalized retrieval scores.

This is an adaptation rather than an exact reproduction: Qwen3-4B replaces
Llama, PersonaMem replaces the paper's multitask corpus, and LoRA is used for
efficient language-model fine-tuning.

Default protocol:

- fixed GTE-large question-only retrieval;
- top-3 passages during training;
- top-10 passages during validation/test;
- passages capped at 200 whitespace-delimited words, matching the paper's
  maximum passage size;
- answer-only next-token loss;
- shuffled option order during training to remove answer-position shortcuts;
- LoRA on all attention and MLP projections.

Run both datasets with:

```bash
bash radit_personamem/launch_qwen3_4b.sh
```
