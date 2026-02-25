# SageAttention v1 → v2 → v3: Paper Summaries

This document summarizes the three SageAttention papers from Tsinghua University, extracted from the PDFs under `papers/sageattention/`.

---

## SageAttention v1 — "Accurate 8-Bit Attention for Plug-and-Play Inference Acceleration"

**Venue:** ICLR 2025 | **arXiv:** 2410.02367  
**Authors:** Jintao Zhang, Jia Wei, Haofeng Huang, Pengle Zhang, Jun Zhu, Jianfei Chen

### Problem

Attention has O(N²) complexity and dominates latency at long sequence lengths (8K–128K). Existing quantization methods focus on linear layers, leaving attention unaccelerated at FP16. Direct 8-bit quantization of attention tensors (Q, K, P, V) causes severe accuracy degradation (blurry images, random-guessing LLM accuracy).

### Key Challenges

- **(C1)** Matrix K has significant channel-wise outliers, causing large quantization error.
- **(C2)** Quantizing P and V to INT8 does not consistently preserve accuracy of PV across models.

### Method

1. **Smooth K:** γ(K) = K − mean(K). Lossless because softmax is shift-invariant: σ(qKᵀ − q·mean(K)) = σ(qKᵀ). Overhead <0.2%.
2. **INT8 for Q·Kᵀ:** Q, K quantized to INT8 (per-token or per-block granularity). INT8 is more accurate than FP8 (E4M3/E5M2) for QKᵀ and 2–4x faster than FP16 on RTX 4090/3090.
3. **FP16 P·V with FP16 accumulator:** Instead of quantizing P/V to 8-bit, keeps them in FP16 with an FP16 accumulator (2x faster than FP32 accum on consumer GPUs, no accuracy loss).
4. **Adaptive kernel selection:** 4 kernel variants with per-layer selection based on cosine similarity threshold (99.8%):
   - `SAGEAttn-T`: per-token INT8 Q/K, FP16 P/V (FP16 accum)
   - `SAGEAttn-B`: per-block INT8 Q/K, FP16 P/V (FP16 accum)
   - `SAGEAttn-vT`: per-token INT8 Q/K, INT8 P/V
   - `SAGEAttn-vB`: per-block INT8 Q/K, INT8 P/V
5. **Fusion:** Quantization of Q, K fused with RoPE to eliminate I/O overhead.

### Results

- **~2.1x** faster than FlashAttention2, **~2.7x** faster than xformers
- **340 TOPS** on RTX4090 (52% of theoretical INT8 peak), vs FlashAttention2's peak of 165 TOPS
- At headdim=64, approaches FlashAttention3's 490 TOPS on much more expensive Hopper GPUs

**End-to-end accuracy (virtually no loss):**

| Model | Task | Metric | Full Precision | SageAttention |
|-------|------|--------|---------------|---------------|
| Llama2-7B | LLM (WikiText) | Perplexity↓ | 5.4721 | 5.4729 |
| CogVideoX | Video Gen | FScore↑ | 3.768 | 3.834 |
| Unidiffuser | Image Gen | FID↓ | 163.33 | 166.49 |
| UltraPixel | Image Gen | FID↓ | 179.78 | ~179.79 |
| TIMM (ViT) | ImageNet | Accuracy↑ | 84.79% | 84.74% |

**GEMM dtypes:** `QKᵀ`: INT8×INT8→FP16/FP32 | `PV`: FP16×FP16→FP16 (default) or INT8×INT8 (fast variant)

**Implementation:** OpenAI Triton, using Tensor Core instructions `u8.u8.s32` (INT8) and `f16.f16.f16` (FP16 accum). Block sizes: bq=128, bkv=64.

---

## SageAttention2 — "Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization"

**Venue:** ICML 2025 | **arXiv:** 2411.10958  
**Authors:** Jintao Zhang\*, Haofeng Huang\*, Pengle Zhang, Jia Wei, Jun Zhu, Jianfei Chen

### Problem

Push quantization further to 4-bit for Q/K while handling new precision challenges from narrower numeric ranges and hardware accumulator limitations.

### Key Challenges

- **(C1)** INT4's extremely limited range (only 15 levels) causes large quantization errors when Q/K have outliers. Per-block granularity from v1 is insufficient for INT4.
- **(C2)** The FP8 tensor core accumulator (`mma.f32.f8.f8.f32`) is actually **FP22** (1 sign + 8 exponent + 13 mantissa bits), not true FP32, degrading P·V accuracy.

### Method

1. **Smooth Q+K:** Extends v1's smooth-K to also smooth Q by subtracting q̄ᵢ = mean(Qᵢ) per block, then adds back q̄ᵢKᵀ via a GEMV correction after the matmul. Combined smoothing: 99.46% CosSim vs 80% without any smoothing (Table 4).
2. **Per-thread INT4 quantization:** Groups tokens by their GPU thread mapping in the PTX `mma` instruction layout (`mma.m16n8k64`). Each thread corresponds to exactly one quantization scale, achieving per-token-level accuracy with zero extra dequantization overhead. Much better than per-block (99.45% vs 98.03% CosSim, Table 6).
3. **FP8 (E4M3) for P and V:** P quantized per-block with static scale (1/448), V per-channel. E4M3 chosen over E5M2 and INT8 for best accuracy (Table 7).
4. **Two-level FP32 accumulation for P·V:** Maintains two register sets: Rᵢⱼ (FP22 from `mma`) and Oᵢⱼ (FP32 buffer). After each block matmul, FP22 result is accumulated into FP32 buffer, confining precision loss to block-level. Same strategy as CUTLASS and DeepGemm.

### Kernel Variants

| Kernel | Q, K | P, V |
|--------|------|------|
| SageAttn2-4b | INT4 per-thread | FP8 per-block/per-channel |
| SageAttn2-8b | INT8 per-thread (for Hopper GPUs lacking INT4 cores) | FP8 per-block/per-channel |

### Results

- **~3x** faster than FlashAttention2, **~4.5x** faster than xformers (RTX4090)
- Matches FlashAttention3(fp8) speed on Hopper GPUs with much higher accuracy
- Tested on 10 models: Llama2/3.1, GLM4, CogVideoX (2B & 1.5-5B), HunyuanVideo, Mochi, Flux, SD3.5, TIMM

**End-to-end highlights:**

| Model | FlashAttn3-fp8 | SageAttn2-8b | SageAttn2-4b |
|-------|---------------|-------------|-------------|
| CogVideoX 1.5-5B (VQA-t↑) | 2.181 (broken) | 74.415 | 52.989 |
| HunyuanVideo (VQA-a↑) | 4.433 (broken) | 81.786 | 81.478 |
| Mochi (VQA-a↑) | 14.964 | 46.760 | 35.955 |
| Flux (FID↓) | — | 10.927 | 10.577 |

**End-to-end latency (RTX4090):**
- CogVideoX 1.5-5B: 1040s → 577s (SageAttn2-8b) → 555s (SageAttn2-4b)
- Llama3.1 100K tokens (L20): 39.9s → 25.4s → 23.2s

**GEMM dtypes:** `QKᵀ`: INT4×INT4 tensor core + FP16 GEMV correction | `PV`: FP8(E4M3)×FP8(E4M3) with FP22 accum → FP32 buffer

**Implementation:** CUDA (not Triton). Uses PTX `mma` instructions directly.

---

## SageAttention3 — "Microscaling FP4 Attention for Inference and An Exploration of 8-bit Training"

**Venue:** NeurIPS 2025 | **arXiv:** 2505.11594  
**Authors:** Jintao Zhang\*, Jia Wei\*, Haoxu Wang, Pengle Zhang, Xiaoming Xu, Haofeng Huang, Kai Jiang, Jianfei Chen, Jun Zhu

### Problem

Exploit Blackwell's new FP4 tensor cores for even faster inference; pioneer low-bit attention for training (no prior work has explored this).

### Key Challenges

- **(C1)** FP4 has only 15 representable values — per-tensor/per-token quantization is insufficient.
- **(C2)** P's values lie in [0,1], forcing FP8 scale factors into a tiny range [0, 0.167] → large E4M3 representation error.
- **(C3)** In backward pass, attention-map gradients (especially via dO·Vᵀ) are highly sensitive to quantization errors, which accumulate along the sequence length.

### Method (Inference — SageAttention3)

1. **NVFP4 microscaling:** Uses E2M1 FP4 with block size 1×16 and E4M3 FP8 scale factors (chosen over MXFP4: 99.52% vs 98.37% CosSim). Applied to **both** QKᵀ and PV GEMMs via the `FP4MM` instruction.
2. **Two-level scaling for P:** First scales each row of P̃ to [0, 448×6] via per-token FP32 scale s_{P1}, then applies NVFP4 microscaling with FP8 s_{P2}. This maximizes E4M3 range utilization (93.32% → 99.52% CosSim, Table 1b).
3. **Smoothing Q+K** from SageAttention2 is retained.
4. **Hardware optimizations:**
   - K column permutation to match FP4MM accumulator layout (avoids thread shuffles)
   - Reuse of softmax max-reduction in quantization (~10% kernel speedup)
   - Producer-warp epilogue: ping-pong between producer warps for overlapping compute and global memory stores within register constraints

### Method (Training — SageBwd)

1. **INT8 attention for both forward and backward** — quantizes 6 of 7 matmuls to INT8 per-block.
2. **Key insight:** The dO·Vᵀ matmul is the most accuracy-sensitive (errors accumulate into dQ and dK along sequence length) — kept in FP16. All other matmuls (QKᵀ, Pᵀ·dO, dS·K, dSᵀ·Q) use INT8.
3. **INT8 > FP8 for training:** INT8 SageBwd yields lower L1 error and higher cosine similarity for gradients than FP8 SageBwd (Table 6, 7), and has wider hardware support (A100, AMD MI250, Ascend 910B).

### Results

**SageAttention3 (inference):**
- **1038 TOPS** on RTX5090, **5x** faster than FlashAttention2, **11x** faster than xformers
- End-to-end: **3x** speedup on HunyuanVideo (489s→164s), **2.4x** on CogVideoX (64s→27s) on RTX5090
- Negligible quality loss across CogVideoX, HunyuanVideo, Mochi, Flux, SD3.5

| Model | Full Precision | SageAttention2 (8bit) | SageAttention3 (4bit) |
|-------|---------------|----------------------|----------------------|
| CogVideoX (VQA-a↑) | 70.476 | 69.414 | 69.860 |
| HunyuanVideo (VQA-a↑) | 68.998 | 69.497 | 70.552 |
| Flux (FID↓) | 162.812 | 163.107 | 162.121 |

**SageBwd (training):**
- 1.67x faster than FlashAttention2 on RTX4090
- Lossless fine-tuning on GSM8K, DROP, MMLU, HellaSwag (Table 3)
- Slower convergence in pretraining (current limitation)
- Combined SageBwd fine-tuning → SageAttention3 inference actually **improves** accuracy (INT8/FP4 distribution alignment, Table 5)

**GEMM dtypes:** `QKᵀ`: FP4×FP4 (FP4MM with FP8 scales) + FP16 GEMV | `PV`: FP4×FP4 (two-level FP8 scales) on Blackwell

**Implementation:** CUTLASS + CUDA (inference), OpenAI Triton (SageBwd training kernels).

---

## Evolution Summary Table

| | **v1 (ICLR'25)** | **v2 (ICML'25)** | **v3 (NeurIPS'25)** |
|---|---|---|---|
| **Q/K dtype** | INT8 per-token/block | INT4 per-thread | NVFP4 microscaling (1×16) |
| **Q/K smoothing** | Smooth K only | Smooth K + Smooth Q | Smooth K + Smooth Q |
| **P/V dtype** | FP16 (default) or INT8 | FP8 E4M3 | NVFP4 with two-level scaling |
| **QKᵀ GEMM** | INT8×INT8 | INT4×INT4 + FP16 GEMV | FP4MM (FP4+FP8 scales) |
| **PV GEMM** | FP16×FP16 (FP16 accum) | FP8×FP8 (FP22→FP32 buffer) | FP4MM (FP4+FP8 scales) |
| **Target GPU** | RTX 4090/3090 | RTX4090, L20, H100 | RTX5090 (Blackwell) |
| **Speedup vs FA2** | ~2.1x | ~3x | ~5x |
| **Training support** | No | No | Yes (SageBwd, INT8 fwd+bwd) |
| **Implementation** | Triton | CUDA | CUTLASS+CUDA (infer), Triton (train) |
| **Code** | [github.com/thu-ml/SageAttention](https://github.com/thu-ml/SageAttention) | same repo | same repo |

---

## Key Recurring Themes

1. **Smoothing is critical.** Subtracting channel-wise means from K (v1) and Q (v2+) is the single most impactful technique, enabling aggressive quantization without accuracy loss. It's mathematically lossless for K (softmax shift-invariance) and corrected via GEMV for Q.

2. **Accumulator precision matters.** v1 discovered FP16 accum is sufficient for PV; v2 discovered the "FP22 accumulator" bug in FP8 tensor cores and worked around it with two-level accumulation; v3 uses two-level scaling to fit P's narrow range into FP8 scale factors.

3. **Quantization granularity tracks hardware.** per-block (v1) → per-thread aligned to MMA layout (v2) → microscaling 1×16 blocks (v3). Each step matches the quantization group to the hardware's native dequantization path for zero overhead.

4. **Orthogonal to other techniques.** All versions are plug-and-play post-training methods, composable with weight quantization (AWQ, GPTQ), sparse attention, and linear attention.
