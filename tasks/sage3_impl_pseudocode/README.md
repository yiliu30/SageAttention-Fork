## Task Description

### Goal:
Add the pseudocode for sage attention3, with details commets and add tensor shape if possible.

## Source:
- Paper:
    - /mnt/disk1/yiliu7/SageAttention-Fork/papers/sageattention/SageAttention3-2505.11594.pdf
    - /mnt/disk1/yiliu7/SageAttention-Fork/papers/sageattention/sage_v3
    - /mnt/disk1/yiliu7/SageAttention-Fork/papers/sageattention/paper_summaries.md
- Codebase:
    - All Sage (v1, v2, v3): /mnt/disk1/yiliu7/SageAttention-Fork
    - Sage v3: /mnt/disk1/yiliu7/SageAttention-Fork/sageattention3_blackwell

## Note:
- Keep code consistant and simple, may ignore some simple helper function
- Heavy comment with tensor shape for study purpose
- Keep the data flow consistant with the real kernel, and align core functions signature if possible
- No need e2e verification
- Don't dequant q/k/v before gemm, but use kernel-like instruction/api 