# Token accounting

Record per-example counts for each actual model call. Include writer input and output, encoder/compressor input, soft-memory length presented to the reader, reader question/prompt input, and final answer output where these calls exist. Do not amortize a generated memory across different questions unless the reported protocol explicitly shares and caches that memory.

Keep textual token counts and continuous-memory positions separately identifiable. Full-context/text baselines do not incur a soft-memory term. Report the sum only together with the included components. Use each run's tokenizer; record whether generated EOS is counted and how padding is excluded. Generation scripts preserve token IDs/counts and manifests so an audit can follow the actual run rather than re-tokenizing a displayed answer.
