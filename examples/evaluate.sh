#!/usr/bin/env bash
set -euo pipefail
# Run from the repository root. Download the HF bundle and base model first.
: "${MODEL:?Set MODEL to the base-model directory}"
: "${BUNDLE:?Set BUNDLE to the HF personamem bundle containing reader/ and bridge.pt}"
: "${QUESTIONS:?Set QUESTIONS to the test question JSONL}"
: "${WRITER_INPUTS:?Set WRITER_INPUTS to prepared writer-message JSONL}"
: "${OUTPUT:?Set OUTPUT to a fresh output directory}"
python memfold.py generate \
  --model "$MODEL" --adapter "$BUNDLE/reader" --writer-inputs "$WRITER_INPUTS" \
  --output "$OUTPUT/memory" --expected-split test --expected-rows "${EXAMPLES:-50}" \
  --max-model-len "${MAX_MODEL_LEN:-40960}" --maximum-memory-tokens 2048 \
  --gpu-memory-utilization 0.75 --seed 42
python memfold.py encode \
  --model "$MODEL" --questions "$QUESTIONS" --memories "$OUTPUT/memory/memories.jsonl" \
  --cache-dir "$OUTPUT/cache" --split-name test --encoder-chunk-tokens 2048 \
  --pool-tokens 32 --encoder-batch-size 4
python memfold.py evaluate \
  --model "$MODEL" --adapter "$BUNDLE/reader" --compressor "$BUNDLE/bridge.pt" \
  --questions "$QUESTIONS" --memories "$OUTPUT/memory/memories.jsonl" \
  --cache "$OUTPUT/cache" --output "$OUTPUT/evaluation" --split test --trials 5
