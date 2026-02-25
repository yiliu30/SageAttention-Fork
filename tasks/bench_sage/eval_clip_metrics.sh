#!/bin/bash
# Evaluate CLIPSIM & CLIP-T for sdpa vs sage3 on CogVideoX-2b

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VIDEO_BASE="/mnt/disk1/yiliu7/SageAttention-Fork/example/videos/cogvideox-2b/open_sora_t2v_samples"
PROMPT_FILE="/mnt/disk1/yiliu7/SageAttention-Fork/example/videos/open_sora_t2v_samples.txt"

python "$SCRIPT_DIR/eval_clip_metrics.py" \
  --video_dirs \
    "$VIDEO_BASE/sdpa" \
    "$VIDEO_BASE/sage3" \
  --prompt_file "$PROMPT_FILE" \
  --output_path "$SCRIPT_DIR/eval_clip_results.json"
