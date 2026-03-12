#!/usr/bin/env bash
# Bisect E2E accuracy across commits.
# Generates sage3_standalone frame at each commit and compares to the existing sage3 reference.
set -uo pipefail

cd /mnt/disk1/yiliu7/SageAttention-Fork/example

REF_FRAME="videos/cogvideox-2b/smoke/quick_e2e_sage3/0_frames/frame_000.png"
TEST_FRAME="videos/cogvideox-2b/smoke/quick_e2e_sage3_standalone/0_frames/frame_000.png"
MODEL="cogvideox-2b"

# Commits to test (oldest first)
COMMITS=(
    "4c3a170"
    "44f348d"
    "6723d86"
    "e4cca81"
    "f6490d7"
    "e931710"
    "a7506c0"
)

ORIGINAL_REF=$(git rev-parse HEAD)

echo "========================================================"
echo " Bisect E2E: sage3_standalone accuracy per commit"
echo " Reference: ${REF_FRAME}"
echo "========================================================"
echo

for COMMIT in "${COMMITS[@]}"; do
    MSG=$(git log --oneline -1 "$COMMIT" 2>/dev/null)

    echo "──────────────────────────────────────────────────────"
    echo "  ${MSG}"
    echo "──────────────────────────────────────────────────────"

    git checkout "$COMMIT" --quiet 2>/dev/null

    # Generate sage3_standalone frame (1 frame)
    echo "  Generating sage3_standalone frame..."
    python cogvideox_infer.py \
        --model "$MODEL" \
        --attention_type sage3_standalone \
        -q -i -n 1 2>&1 | tail -3

    if [ ! -f "$TEST_FRAME" ]; then
        echo "  ❌ Frame not generated!"
        echo
        continue
    fi

    # Compare
    python compare_frames.py "$REF_FRAME" "$TEST_FRAME" 2>&1 | grep -E "PSNR|Cosine|PASS|FAIL"
    echo
done

# Restore original HEAD
git checkout "$ORIGINAL_REF" --quiet 2>/dev/null
echo "Done. Restored to ${ORIGINAL_REF}."
