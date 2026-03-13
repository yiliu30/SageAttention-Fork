#!/usr/bin/env bash
# Quick end-to-end regression test: sage3_standalone (triton) vs sage3 (CUTE kernel)
#
# Generates one frame with each kernel using CogVideoX-2b and compares via PSNR
# and cosine similarity. Exits 0 on PASS, 1 on FAIL.
#
# Usage:
#   cd example
#   bash test_quick_e2e.sh
#
# Environment:
#   PSNR_THRESHOLD  — minimum PSNR in dB (default: 20)
#   COS_THRESHOLD   — minimum cosine similarity (default: 0.95)
#   NUM_FRAMES      — number of frames to generate (default: 1)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

THRESHOLD="${PSNR_THRESHOLD:-20}"
COS_THRESH="${COS_THRESHOLD:-0.95}"
NFRAMES="${NUM_FRAMES:-1}"

MODEL="cogvideox-2b"
REF_TYPE="sage3"
TEST_TYPE="sage3_standalone"
TEST_TYPE="sage3_standalone_mxfp8_s1"

REF_FRAME="videos/${MODEL}/smoke/quick_e2e_${REF_TYPE}/0_frames/frame_000.png"
TEST_FRAME="videos/${MODEL}/smoke/quick_e2e_${TEST_TYPE}/0_frames/frame_000.png"

echo "========================================"
echo " Quick E2E Regression Test"
echo " Reference : ${REF_TYPE} (CUTE kernel)"
echo " Under test: ${TEST_TYPE} (triton kernel)"
echo " PSNR thresh: ${THRESHOLD} dB"
echo " CosSim thresh: ${COS_THRESH}"
echo "========================================"
echo

# ── Step 1: Generate test frame ─────────────────────────────────
echo "[1/3] Generating ${TEST_TYPE} (triton kernel) frame..."
python cogvideox_infer.py \
    --model "$MODEL" \
    --attention_type "$TEST_TYPE" \
    -q -i -n "$NFRAMES"
echo "     -> ${TEST_FRAME}"
echo

# ── Step 2: Generate reference frame ────────────────────────────
echo "[2/3] Generating ${REF_TYPE} (CUTE kernel) reference frame..."
python cogvideox_infer.py \
    --model "$MODEL" \
    --attention_type "$REF_TYPE" \
    -q -i -n "$NFRAMES"
echo "     -> ${REF_FRAME}"
echo

# ── Step 3: Compare ─────────────────────────────────────────────
echo "[3/3] Comparing frames..."
python compare_frames.py "$REF_FRAME" "$TEST_FRAME" \
    --threshold "$THRESHOLD" --cos-threshold "$COS_THRESH"

# real e4m3
# [3/3] Comparing frames...
# PSNR:              18.21 dB
# Cosine Similarity: 0.978975
# ❌ FAIL  PSNR 18.21 < 20.00 dB
# ✅ PASS  CosSim 0.978975 >= 0.95

# approximate e4m3
# [3/3] Comparing frames...
# PSNR:              17.77 dB
# Cosine Similarity: 0.976606
# ❌ FAIL  PSNR 17.77 < 20.00 dB
# ✅ PASS  CosSim 0.976606 >= 0.95