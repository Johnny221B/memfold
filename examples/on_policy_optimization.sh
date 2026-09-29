#!/usr/bin/env bash
set -euo pipefail
# Set these to prepared artifacts; no server-specific paths are assumed.
: "${MODEL:?Set MODEL to the local base model}"
: "${INITIAL_ADAPTER:?Set INITIAL_ADAPTER to the Reader initialization adapter}"
: "${QUESTIONS:?Set QUESTIONS to the training question JSONL}"
: "${SELF_MEMORIES:?Set SELF_MEMORIES to the policy memory JSONL}"
: "${TEACHER_MEMORIES:?Set TEACHER_MEMORIES to teacher memory JSONL}"
: "${CACHE_DIR:?Set CACHE_DIR to the encoder cache}"
: "${COMPRESSOR_CHECKPOINT:?Set COMPRESSOR_CHECKPOINT to the trained compressor}"
: "${OUTPUT:?Set OUTPUT to a fresh output directory}"
python -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE:-4}" \
  memfold.py train optimize \
  --model "$MODEL" --initial-adapter "$INITIAL_ADAPTER" \
  --questions "$QUESTIONS" --self-memories "$SELF_MEMORIES" \
  --teacher-memories "$TEACHER_MEMORIES" --cache-dir "$CACHE_DIR" \
  --compressor-checkpoint "$COMPRESSOR_CHECKPOINT" --output "$OUTPUT" \
  --mode opd_grpo --expected-token-count 256 --reference-kl-weight 0 "$@"
