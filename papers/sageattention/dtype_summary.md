# SageAttention Quantization and GEMM Data Types

This note summarizes how each public SageAttention release handles the dtypes of the query/key/value (Q/K/V) tensors and the attention GEMMs, based on the official paper sources under `papers/sageattention/`.

## SageAttention (v1, "Accurate 8-bit attention")
- **Q/K dtypes.** Q and K are quantized to INT8 with either per-token or per-block scales after smoothing K by subtracting the token mean (see Section 3.2 of the v1 Method). The INT8 tiles feed the `QK^T` GEMM on INT8 tensor cores before being rescaled in FP16/FP32.
- **\~P/V dtypes.** Two execution modes exist:
  - `SAGEAttn-B` (default fast kernel) keeps \~P and V in FP16 with an FP16 accumulator so the `\tilde PV` GEMM runs as FP16×FP16→FP16-accum (Section 3.4). This avoids the accuracy loss seen with INT8 outputs while still saving registers vs FP32.
  - `SAGEAttn-vB`/`SAGEAttn-vT` fully quantize \~P and V to INT8 (per-block/per-channel), trading a small accuracy drop for slightly higher throughput (Table 3.5).
- **GEMM summary.** `QK^T`: INT8×INT8 with FP16/FP32 accumulation. `\tilde PV`: FP16×FP16 with FP16 accumulation (mainline) or INT8×INT8 when using the all-INT8 variants.

## SageAttention2 (v2, "INT4 per-thread + FP8")
- **Q/K dtypes.** Q and K are first smoothed by subtracting per-block means, then quantized to INT4 with per-thread scales aligned to the `mma.m16n8k64` layout (Section 2.2). The tensor-core GEMM uses INT4×INT4 operands and adds a low-rank \(\Delta S\) correction computed in FP16.
- **V dtype.** V is quantized per-channel to FP8 (E4M3) so each channel keeps its own scale (Section 2.4).
- **\~P dtype.** \~P uses a static E4M3 FP8 scale (`1/448`) plus a two-level accumulation strategy: the hardware MMA accumulates in FP22, and results are immediately flushed into an FP32 buffer to recover accuracy (Section 2.5).
- **GEMM summary.** `QK^T`: INT4×INT4 tensor-core GEMM + FP16 GEMV bias restore. `\tilde PV`: FP8(E4M3)×FP8(E4M3) GEMM using `mma(f32f8f8f32)` (FP22 accumulator) followed by FP32 accumulation.

## SageAttention3 (v3, "Microscaling FP4 + 8-bit training")
- **Q/K dtypes.** Adopts NVFP4 microscaling (E2M1 data, block size 1×16) with per-block FP8 (E4M3) scale factors. Q is first mean-centered per block (SageAttention2 smoothing) before quantization; K is mean-centered globally (Section 1.1).
- **\~P dtype.** Uses a two-level scheme: each row of \~P is first scaled into `[0, 448×6]` via an FP32 row scale `s_{P1}`, then microscaled again into FP4 with FP8 `s_{P2}` (Section 1.2). This maximizes the effective FP8 scale range.
- **V dtype.** V is microscaled to NVFP4 with FP8 scales (Section 1.1). Channel-wise mean subtraction is optional when large biases appear (Appendix E).
- **GEMM summary.** Both `QK^T` and `\tilde PV` run on Blackwell FP4 tensor cores (`FP4MM`) that consume FP4 operands plus their FP8 scale tensors, delivering the published 1038 TOPS on RTX5090. A fused GEMV term adds back the smoothed means for `QK^T`.
- **Training path.** The same paper also proposes an 8-bit forward/backward attention for training; both passes quantize activations to 8-bit (details in Section 3), but the inference-focused FP4 path above is the default for SageAttention3.

## Quick Reference Table

| Version | Q/K dtype & granularity | V dtype | \~P dtype | GEMM dtype (QK / \~PV) |
|---------|------------------------|---------|-----------|-------------------------|
| SageAttention v1 | INT8 per-token or per-block (K smoothed) | INT8 per-channel **or** FP16 (kernel-dependent) | INT8 per-block **or** FP16 | INT8×INT8→FP16/FP32 (QK); FP16×FP16→FP16 (default) or INT8×INT8→FP32 (fast variant) |
| SageAttention2 | INT4 per-thread with Q/K smoothing + FP16 \(\Delta S\) | FP8 (E4M3) per-channel | FP8 (E4M3) with static scale + FP22→FP32 accumulation | INT4×INT4 tensor core + FP16 GEMV correction (QK); FP8×FP8 tensor core with FP22 accumulator + FP32 buffer (\~PV) |
| SageAttention3 | NVFP4 microscaling (E2M1) with FP8 scales, per-block smoothing | NVFP4 microscaling with FP8 scales (optional mean subtraction) | Two-level scaling (FP32 row scale + NVFP4 microscaling) | FP4MM (FP4×FP4 with FP8 scales) for both QK and \~PV on Blackwell |

These settings are all derived from the LaTeX sources in `papers/sageattention/sage_v*/src/Method*.tex`.
