#!/bin/bash
# Evaluate CLIPSIM & CLIP-T for sdpa vs sage3 on CogVideoX-2b

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VIDEO_BASE="/mnt/disk1/yiliu7/SageAttention-Fork/example/videos/cogvideox-2b/open_sora_t2v_samples"
PROMPT_FILE="/mnt/disk1/yiliu7/SageAttention-Fork/example/videos/open_sora_t2v_samples.txt"
VIDEO_BASE="/mnt/disk1/yiliu7/SageAttention-Fork/example/videos/cogvideox-2b/open_sora_prompts"
VIDEO_BASE="/mnt/disk1/yiliu7/SageAttention-Fork/example/videos/cogvideox-2b/open_sora_prompts"
PROMPT_FILE="/mnt/disk1/yiliu7/SageAttention-Fork/example/videos/open_sora_prompts.txt"

    # "$VIDEO_BASE/sdpa" \

attn_variant="sage3_triton"
attn_variant="sage3"

python "$SCRIPT_DIR/eval_clip_metrics.py" \
  --video_dirs \
    "$VIDEO_BASE/$attn_variant" \
  --prompt_file "$PROMPT_FILE" \
  --output_path "$SCRIPT_DIR/eval_clip_results_$attn_variant.json"
