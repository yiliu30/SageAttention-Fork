"""
SageAttention3 Pseudocode — Kernel-Aligned Implementation
==========================================================

This pseudocode mirrors the actual SageAttention3 CUDA/CUTLASS kernel data flow
on Blackwell GPUs (SM_100+). It uses kernel-like FP4MM instructions instead of
dequantizing Q/K/V before GEMM, matching how the hardware actually executes.

References:
  - Paper: SageAttention3 — Microscaling FP4 Attention (arXiv 2505.11594)
  - Source: sageattention3_blackwell/sageattn3/

Architecture Overview (Warp-Specialized Persistent Kernel):
  ┌──────────────────────────────────────────────────────┐
  │  WarpGroup 0 (Producer)                              │
  │    Warp 0: TMA loads Q, K, V, SFQ, SFK, SFV, DS     │
  │    Warp 1: TMA store O (epilogue)                    │
  │    Warp 2-3: idle                                    │
  ├──────────────────────────────────────────────────────┤
  │  WarpGroup 1-2 (Consumer MMA)                        │
  │    Phase 1: QK^T via FP4MM (with scale factors)      │
  │    Phase 2: Online softmax + fused P quantization    │
  │    Phase 3: PV via FP4MM (with two-level P scales)   │
  ├──────────────────────────────────────────────────────┤
  │  Communication: TMA async pipeline (3 stages K/V)    │
  └──────────────────────────────────────────────────────┘

Tensor Naming Conventions:
  B  = batch_size
  H  = num_heads  (H_k for KV heads in GQA)
  L  = seq_len
  D  = head_dim   (64 or 128)
  G  = num_groups = L // 128 (for per-block Q mean subtraction)

NVFP4 E2M1 representable magnitudes:
  {0, 0.5, 1, 2, 3, 4, 6}   (4 bits: 1 sign + 2 exponent + 1 mantissa)

FP8 E4M3 is used for all microscaling scale factors.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import Tuple, Optional, NamedTuple


# =============================================================================
# Data Structures (mirrors csrc params.h)
# =============================================================================

class QuantizedTensor(NamedTuple):
    """Packed FP4 data + FP8 E4M3 scale factors (1 scale per 16 elements)."""
    data: torch.Tensor       # uint8, packed 2×E2M1 per byte
    scale: torch.Tensor      # float8_e4m3fn


# =============================================================================
# Top-Level API (mirrors sageattn3/api.py :: sageattn3_blackwell)
# =============================================================================

def sageattn3_blackwell(
    q: torch.Tensor,            # [B, H, L, D] fp16/bf16
    k: torch.Tensor,            # [B, H_k, L, D] fp16/bf16
    v: torch.Tensor,            # [B, H_k, L, D] fp16/bf16
    attn_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    per_block_mean: bool = True,
    **kwargs,
) -> torch.Tensor:
    """
    SageAttention3 entry point.

    Data flow:
      q,k,v [B,H,L,D] fp16/bf16
        │
        ├─ preprocess_qkv()  →  q,k,v [B,H,L_pad,D], delta_s [B,H,G,L_pad] fp32
        │
        ├─ scale_and_quant_fp4(q)            →  QuantizedTensor [B,H,L_pad,D//2], [B,H,L_pad,D//16]
        ├─ scale_and_quant_fp4_permute(k)    →  QuantizedTensor [B,H,L_pad,D//2], [B,H,L_pad,D//16]
        ├─ scale_and_quant_fp4_transpose(v)  →  QuantizedTensor [B,H,D,L_pad//2], [B,H,D,L_pad//16]
        │
        └─ blockscaled_fp4_attn()            →  output [B,H,L,D] fp16/bf16
    """
    assert q.size(-1) < 256, f"Unsupported head_dim={q.size(-1)}, max 192"

    QL = q.size(2)       # original query seq_len (before padding)
    KL = k.size(2)       # original key seq_len
    is_bf16 = (q.dtype == torch.bfloat16)

    # ── Step 1: Preprocessing ──────────────────────────────────────────────
    # Smooth K (subtract seq-mean), smooth Q (per-block or global mean),
    # pad to multiple of 128, compute delta correction.
    q, k, v, delta_s = preprocess_qkv(q, k, v, per_block_mean)
    # q: [B,H,L_pad,D] fp16/bf16  — Q with mean subtracted
    # k: [B,H,L_pad,D] fp16/bf16  — K centered
    # v: [B,H,L_pad,D] fp16/bf16  — V (only padded)
    # delta_s: [B,H,G,L_pad] fp32 — Q_mean @ K^T correction

    # ── Step 2: FP4 Quantization (three variants) ─────────────────────────
    # Each produces (packed_e2m1_uint8, fp8_e4m3_scales).
    # Quantization uses NVFP4 microscaling: 1×16 block, E4M3 scale factor.

    # Q — standard quantization
    q_quant = scale_and_quant_fp4(q)
    # q_quant.data:  [B,H,L_pad,D//2]  uint8
    # q_quant.scale: [B,H,L_pad,D//16] fp8_e4m3fn

    # K — with column permutation for FP4MM accumulator layout
    k_quant = scale_and_quant_fp4_permute(k)
    # k_quant.data:  [B,H,L_pad,D//2]  uint8 (permuted token order within 32-token groups)
    # k_quant.scale: [B,H,L_pad,D//16] fp8_e4m3fn

    # V — with transposition (seq↔head_dim) for efficient PV matmul
    v_quant = scale_and_quant_fp4_transpose(v)
    # v_quant.data:  [B,H,D,L_pad//2]  uint8  (transposed layout)
    # v_quant.scale: [B,H,D,L_pad//16] fp8_e4m3fn

    # ── Step 3: Attention kernel ───────────────────────────────────────────
    output = blockscaled_fp4_attn(
        q_quant, k_quant, v_quant,
        delta_s,
        KL,
        is_causal=is_causal,
        per_block_mean=per_block_mean,
        is_bf16=is_bf16,
    )
    # output: [B,H,L_pad,D] fp16/bf16

    # ── Step 4: Crop padding ───────────────────────────────────────────────
    return output[:, :, :QL, :].contiguous()


# =============================================================================
# Preprocessing (mirrors sageattn3/api.py :: preprocess_qkv + triton_group_mean)
# =============================================================================

def preprocess_qkv(
    q: torch.Tensor,    # [B, H, L, D]
    k: torch.Tensor,    # [B, H, L, D]
    v: torch.Tensor,    # [B, H, L, D]
    per_block_mean: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Preprocessing pipeline for SageAttention3.

    1. Smooth K: K ← K − mean(K, dim=seq).
       Mathematically lossless because softmax is shift-invariant:
         softmax(Q @ K^T) = softmax(Q @ (K − μ_K)^T + Q @ μ_K^T)
       The Q @ μ_K^T term is constant per query and cancels in softmax.

    2. Pad L to next multiple of 128 (required by FP4MM tile size).

    3. Smooth Q (per_block_mean=True):
         For each 128-token block g:
           Q_mean[g] = mean(Q[g*128 : (g+1)*128], dim=0)  # [D]
           Q[g*128 : (g+1)*128] -= Q_mean[g]
       OR (per_block_mean=False):
           Q_mean = mean(Q, dim=seq)  # [1, D]
           Q -= Q_mean

    4. Delta correction: delta_s = Q_mean @ K_padded^T  →  [B,H,G,L_pad] fp32
       Added back to QK^T logits inside the attention kernel.

    Returns:
        q:       [B, H, L_pad, D] fp16/bf16  — smoothed Q
        k:       [B, H, L_pad, D] fp16/bf16  — smoothed K
        v:       [B, H, L_pad, D] fp16/bf16  — padded V
        delta_s: [B, H, G, L_pad] fp32       — Q-mean correction
                 G = L_pad//128 (per_block_mean) or G = 1 (global mean)
    """

    def pad_128(x: torch.Tensor) -> torch.Tensor:
        """Pad seq_len (dim 2) to next multiple of 128, fill with 0."""
        L = x.size(2)
        pad_len = (128 - L % 128) % 128
        if pad_len == 0:
            return x.contiguous()
        return F.pad(x, (0, 0, 0, pad_len), value=0.0).contiguous()

    # Step 1: K smoothing — subtract per-seq mean
    k = k - k.mean(dim=-2, keepdim=True)  # k_mean: [B,H,1,D]

    # Step 2: Pad to multiple of 128
    q, k, v = pad_128(q), pad_128(k), pad_128(v)

    # Step 3: Q smoothing
    if per_block_mean:
        q, q_mean = triton_group_mean(q)
        # q:      [B,H,L_pad,D]
        # q_mean: [B,H,G,D]  where G = L_pad // 128
    else:
        q_mean = q.mean(dim=-2, keepdim=True)  # [B,H,1,D]
        q = q - q_mean

    # Step 4: Delta correction
    # q_mean: [B,H,G,D] or [B,H,1,D]
    # k^T:    [B,H,D,L_pad]
    # delta_s = q_mean @ k^T  →  [B,H,G,L_pad]
    delta_s = torch.matmul(q_mean, k.transpose(-2, -1)).to(torch.float32).contiguous()

    return q, k, v, delta_s


@triton.jit
def group_mean_kernel(
    q_ptr, q_out_ptr, qm_out_ptr,
    B, H, L, D: tl.constexpr,
    stride_qb, stride_qh, stride_ql, stride_qd,
    stride_qmb, stride_qmh, stride_qml, stride_qmd,
    GROUP_SIZE: tl.constexpr,
):
    """
    Triton kernel: per-block mean subtraction for Q.
    Grid: (B, H, num_groups).
    Each program handles one 128-token group and computes:
      q_mean = mean(q[group], dim=0)          →  [D]
      q_out[group] = q[group] - q_mean        →  [128, D]
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_group = tl.program_id(2)

    group_start = pid_group * GROUP_SIZE
    offsets = group_start + tl.arange(0, GROUP_SIZE)

    # Load group [GROUP_SIZE, D]
    q_offsets = (pid_b * stride_qb + pid_h * stride_qh
                 + offsets[:, None] * stride_ql
                 + tl.arange(0, D)[None, :] * stride_qd)
    q_group = tl.load(q_ptr + q_offsets)

    # Compute mean [D] and subtract
    qm_group = tl.sum(q_group, axis=0) / GROUP_SIZE
    q_group = q_group - qm_group
    tl.store(q_out_ptr + q_offsets, q_group)

    # Store mean [D]
    qm_offset = (pid_b * stride_qmb + pid_h * stride_qmh
                  + pid_group * stride_qml
                  + tl.arange(0, D) * stride_qmd)
    tl.store(qm_out_ptr + qm_offset, qm_group)


def triton_group_mean(
    q: torch.Tensor,   # [B, H, L, D]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Per-block (128 tokens) mean subtraction via Triton.

    Returns:
        q_out: [B, H, L, D] — centered within each group
        qm:    [B, H, G, D] — per-group means, G = L // 128
    """
    B, H, L, D = q.shape
    GROUP_SIZE = 128
    num_groups = L // GROUP_SIZE

    q_out = torch.empty_like(q)
    qm = torch.empty(B, H, num_groups, D, device=q.device, dtype=q.dtype)

    grid = (B, H, num_groups)
    group_mean_kernel[grid](
        q, q_out, qm,
        B, H, L, D,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        qm.stride(0), qm.stride(1), qm.stride(2), qm.stride(3),
        GROUP_SIZE=GROUP_SIZE,
    )
    return q_out, qm


# =============================================================================
# FP4 Quantization Kernels (mirrors sageattn3/quantization/fp4_quantization_4d.cu)
# =============================================================================
#
# All three quantization functions share the same per-thread logic:
#   1. Load 16 consecutive fp16/bf16 elements (1 microscaling block)
#   2. Compute max(|elements|) via __habs2 + __hmax2 reduction
#   3. Scale factor: sf = max_abs / 6.0,  then round to FP8 E4M3
#   4. Scale: elements *= 1/sf
#   5. Convert to E2M1 via PTX: cvt.rn.satfinite.e2m1x2.f32
#   6. Pack 2 FP4 values per uint8 byte
#
# Kernel config (all variants):
#   BLOCK_SIZE = 128 (tokens per thread block)
#   CVT_FP4_ELTS_PER_THREAD = 16 (elements per thread = 1 microscaling block)
#   blockDim = (BLOCK_SIZE * D / 16, 1, 1)
#   gridDim  = (ceil(N / BLOCK_SIZE), B, H)
#
# Differences:
#   scale_and_quant_fp4:           standard layout
#   scale_and_quant_fp4_permute:   token reordering within 32-token groups (for K)
#   scale_and_quant_fp4_transpose: shared-memory transpose (for V)


def scale_and_quant_fp4(x: torch.Tensor) -> QuantizedTensor:
    """
    Standard NVFP4 microscaling quantization for Q.

    Per thread (processes 1 microscaling block = 16 consecutive elements):
      1. in_vec = load x[batch, head, token, elem_offset : elem_offset+16]   # 16 × fp16
      2. max_abs = max(|in_vec[0..15]|)                # via __habs2 + __hmax2
      3. sf_fp8  = fp8_e4m3(max_abs / 6.0)             # scale factor → FP8 round-trip
      4. sf_f32  = float(sf_fp8)                        # use quantized value
      5. scaled  = in_vec * (1/sf_f32)                  # scale to FP4 range [-6, 6]
      6. e2m1 = cvt.rn.satfinite.e2m1x2(scaled)        # PTX: 2 FP4 → 1 byte
      7. store packed bytes → output[..., elem_offset//2]
      8. store sf_fp8       → output_sf[..., elem_offset//16]

    Scale factor storage layout (within 64-row × (D//16) block):
      offset = (col_sf // 4) * 256 + (col_sf % 4) + (row%64 // 16)*4 + (row%16)*16

    Args:
        x: [B, H, N, D] fp16/bf16 (N already padded to 128)

    Returns:
        QuantizedTensor:
          .data:  [B, H, N, D//2]  uint8
          .scale: [B, H, N, D//16] fp8_e4m3fn
    """
    assert x.ndim == 4
    B, H, N, D = x.shape
    packed = torch.empty((B, H, N, D // 2), device=x.device, dtype=torch.uint8)
    sf = torch.empty((B, H, N, D // 16), device=x.device, dtype=torch.float8_e4m3fn)
    # fp4quant_cuda.scaled_fp4_quant(x, packed, sf, 1)
    _scaled_fp4_quant_kernel(x, packed, sf, permute=False)
    return QuantizedTensor(packed, sf)


def scale_and_quant_fp4_permute(x: torch.Tensor) -> QuantizedTensor:
    """
    NVFP4 quantization with K-column permutation for FP4MM accumulator layout.

    Same per-thread quantization logic as scale_and_quant_fp4, but the token
    index is permuted within 32-token groups before loading from global memory.

    Permutation formula (within each 32-token group):
      local = token_id % 32
      permuted = (local // 8) * 2 + ((local % 8) // 2) * 8 + (local % 2)

    Resulting order:
      original: [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,
                  16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
      permuted: [ 0, 1, 8, 9,16,17,24,25, 2, 3,10,11,18,19,26,27,
                   4, 5,12,13,20,21,28,29, 6, 7,14,15,22,23,30,31]

    This matches how the SM120 BLOCKSCALED MMA instruction reads its B operand,
    eliminating thread shuffles during QK^T computation.

    Args:
        x: [B, H, N, D] fp16/bf16

    Returns:
        QuantizedTensor:
          .data:  [B, H, N, D//2]  uint8 (permuted token order)
          .scale: [B, H, N, D//16] fp8_e4m3fn
    """
    assert x.ndim == 4
    B, H, N, D = x.shape
    packed = torch.empty((B, H, N, D // 2), device=x.device, dtype=torch.uint8)
    sf = torch.empty((B, H, N, D // 16), device=x.device, dtype=torch.float8_e4m3fn)
    # fp4quant_cuda.scaled_fp4_quant_permute(x, packed, sf, 1)
    _scaled_fp4_quant_kernel(x, packed, sf, permute=True)
    return QuantizedTensor(packed, sf)


def scale_and_quant_fp4_transpose(x: torch.Tensor) -> QuantizedTensor:
    """
    NVFP4 quantization with transpose for V matrix.

    Transposes [N, D] → [D, N] within each (batch, head) using shared memory,
    then quantizes along the new last dimension (original seq_len).

    After transpose, each microscaling block of 16 elements spans 16 consecutive
    tokens at a fixed head_dim position (vs. 16 head_dim elements at a fixed token).
    This is the access pattern needed for PV matmul: P [M, N] @ V^T [D, N]^T.

    Shared memory tile: T shared_input[BLOCK_SIZE * D]
      1. Coalesced load: shared[local_tok * D + col] = x[..., token, col]
      2. __syncthreads()
      3. Transposed read: val = shared[head_dim_idx + local_seq * D]
      4. Quantize 16 transposed elements and write to [D, N//2] output

    Args:
        x: [B, H, N, D] fp16/bf16

    Returns:
        QuantizedTensor:
          .data:  [B, H, D, N//2]  uint8 (transposed + packed)
          .scale: [B, H, D, N//16] fp8_e4m3fn (transposed)
    """
    assert x.ndim == 4
    B, H, N, D = x.shape
    packed = torch.empty((B, H, D, N // 2), device=x.device, dtype=torch.uint8)
    sf = torch.empty((B, H, D, N // 16), device=x.device, dtype=torch.float8_e4m3fn)
    # fp4quant_cuda.scaled_fp4_quant_trans(x, packed, sf, 1)
    _scaled_fp4_quant_transpose_kernel(x, packed, sf)
    return QuantizedTensor(packed, sf)


# =============================================================================
# Attention Kernel (mirrors sageattn3/blackwell/kernel_ws.h + mainloop_tma_ws.h)
# =============================================================================

def blockscaled_fp4_attn(
    q_quant: QuantizedTensor,   # Q: data [B,H,L,D//2], scale [B,H,L,D//16]
    k_quant: QuantizedTensor,   # K: data [B,H,L,D//2], scale [B,H,L,D//16]
    v_quant: QuantizedTensor,   # V: data [B,H,D,L//2], scale [B,H,D,L//16]
    delta_s: torch.Tensor,      # [B,H,G,L] fp32
    unpadded_k_len: int,        # original key seq_len (before padding)
    is_causal: bool = False,
    per_block_mean: bool = True,
    is_bf16: bool = True,
) -> torch.Tensor:
    """
    Warp-specialized persistent FP4MM attention kernel (pseudocode).

    Mirrors: kernel_ws.h :: compute_attn_ws
             mainloop_tma_ws.h :: CollectiveMainloopFwd::mma()
             softmax_fused.h :: SoftmaxFused::online_softmax_with_quant()

    Kernel config (from kernel_traits.h):
      TileShape_MNK = (128, 128, D)     — TILE_M=128, TILE_N=128
      kStages = 3                         — TMA pipeline depth for K/V
      MMA atom: SM120_16x32x64_TN_VS_NVFP4  — Blackwell FP4MM instruction
      Warps: 12 (kBlockM=128) = 1 producer WG + 2 consumer WGs
      Register budget: producer 24, consumer 232

    Softmax uses log2 domain for hardware exp2 instruction:
      softmax_scale = 1/sqrt(D)
      softmax_scale_log2 = log2(softmax_scale)

    Two-level P quantization constants:
      fp8_scale × fp4_scale = 1 / (448 × 6) ≈ 3.72e-4
      fp8xfp4_scale_log2 = log2(1/(448*6)) ≈ -11.39
      fp4_scale_log2 = log2(1/6) ≈ -2.58

    Returns:
        output: [B, H, L_pad, D] fp16/bf16
    """
    import math

    q_packed, q_sf = q_quant
    k_packed, k_sf = k_quant
    v_packed, v_sf = v_quant

    B, H, L, D_half = q_packed.shape
    D = q_sf.size(-1) * 16      # reconstruct head_dim from scale tensor shape
    G = delta_s.size(-2)         # number of Q-mean groups

    # Kernel constants (from kernel_traits.h and softmax_fused.h)
    TILE_M = 128                 # query tile size = kBlockM
    TILE_N = 128                 # key/value tile size = kBlockN
    softmax_scale = D ** (-0.5)
    softmax_scale_log2 = math.log2(softmax_scale)
    fp8xfp4_scale_log2 = math.log2(1.0 / (448.0 * 6.0))  # ≈ -11.39
    fp4_scale_log2 = math.log2(1.0 / 6.0)                  # ≈ -2.58

    num_tile_m = (L + TILE_M - 1) // TILE_M
    num_tile_n = (L + TILE_N - 1) // TILE_N

    output_dtype = torch.bfloat16 if is_bf16 else torch.float16
    output = torch.zeros((B, H, L, D), device=q_packed.device, dtype=output_dtype)

    # ── Persistent tile scheduler (mirrors StaticPersistentTileScheduler) ──
    # In hardware: grid = (num_SMs, 1, 1), each SM loops over tiles.
    # Here we emulate with sequential iteration over (B, H, tile_m).
    for b in range(B):
        for h in range(H):
            for tile_m in range(num_tile_m):

                m_start = tile_m * TILE_M
                m_end   = min(m_start + TILE_M, L)
                m_size  = m_end - m_start

                # Causal: limit n_block_max
                n_block_max = num_tile_n
                if is_causal:
                    n_block_max = min(
                        n_block_max,
                        ((tile_m + 1) * TILE_M + L - L + TILE_N - 1) // TILE_N,
                    )

                # ── Register-resident accumulators (fp32) ──────────────────
                # In kernel: tOrO = partition_fragment_C(tiled_mma_pv, ...)
                O_acc = torch.zeros((m_size, D), dtype=torch.float32,
                                    device=q_packed.device)
                # Online softmax state (per row):
                row_max = torch.full((m_size,), float('-inf'), dtype=torch.float32,
                                     device=q_packed.device)
                row_sum = torch.zeros((m_size,), dtype=torch.float32,
                                      device=q_packed.device)

                # ── TMA load Q tile (smem → rmem, done once per m_tile) ────
                # In kernel: consumer_wait(pipeline_q), copy(smem_tiled_copy_Q)
                q_tile_packed = q_packed[b, h, m_start:m_end]   # [m_size, D//2] uint8
                q_tile_sf     = q_sf[b, h, m_start:m_end]       # [m_size, D//16] fp8

                # ── K/V tile loop (n_block from n_block_max-1 down to 0) ───
                # In kernel: this is the main #pragma unroll 1 loop.
                # TMA pipeline: producer loads K[n], SFK[n], DS[n], V[n], SFV[n]
                # into smem stages (3 stages deep); consumer reads and computes.
                for n_idx in range(n_block_max):
                    n_block = n_block_max - 1 - n_idx   # descending order
                    n_start = n_block * TILE_N
                    n_end   = min(n_start + TILE_N, L)
                    n_size  = n_end - n_start
                    is_first_tile = (n_idx == 0)

                    # ── TMA load K tile + SFK + delta_s from smem ──────────
                    # In kernel: consumer_wait(pipeline_k), copy_k_block(_0{})
                    k_tile_packed = k_packed[b, h, n_start:n_end]  # [n_size, D//2]
                    k_tile_sf     = k_sf[b, h, n_start:n_end]      # [n_size, D//16]

                    # ── Initialize S with delta_s (add_delta_s) ────────────
                    # In kernel: add_delta_s(tSrS) before QK GEMM
                    # delta_s[b, h, group_id, n_start:n_end] → broadcast to [m_size, n_size]
                    group_id = tile_m if per_block_mean else 0
                    ds_tile = delta_s[b, h, group_id, n_start:n_end]  # [n_size] fp32
                    S = ds_tile.unsqueeze(0).expand(m_size, -1).clone()  # [m_size, n_size]

                    # ══════════════════════════════════════════════════════
                    # Phase 1: QK^T via FP4MM (GEMM-I)
                    # ══════════════════════════════════════════════════════
                    # In kernel (per k_block iteration over head_dim chunks):
                    #   cute::gemm(tiled_mma_qk,
                    #     make_zip_tensor(tSrQ(_, _, k_block), tSrSFQ(_, _, k_block)),
                    #     make_zip_tensor(tSrK(_, _, k_block), tSrSFK(_, _, k_block)),
                    #     tSrS);
                    #
                    # FP4MM atom: SM120_16x32x64_TN_VS_NVFP4
                    #   A (Q): [16, 64] E2M1 + SFA [16, 64//16=4] FP8
                    #   B (K): [32, 64] E2M1 + SFB [32, 64//16=4] FP8
                    #   C (S): [16, 32] fp32
                    #
                    # The instruction dequantizes A and B internally:
                    #   a_float = fp4_to_float(a_bits) * float(sfa)
                    #   b_float = fp4_to_float(b_bits) * float(sfb)
                    #   C += a_float * b_float
                    S += fp4mm_gemm(
                        q_tile_packed, q_tile_sf,    # A: Q [m_size, D//2], sf [m_size, D//16]
                        k_tile_packed, k_tile_sf,    # B: K [n_size, D//2], sf [n_size, D//16]
                        m_size, n_size, D,
                    )
                    # S: [m_size, n_size] fp32 — raw attention logits + delta correction

                    # ── Masking (causal + padding) ─────────────────────────
                    # In kernel: iterates over tScS identity tensor
                    for i in range(m_size):
                        qi = m_start + i
                        for j in range(n_size):
                            kj = n_start + j
                            if kj >= unpadded_k_len:
                                S[i, j] = float('-inf')
                            elif is_causal and kj > qi:
                                S[i, j] = float('-inf')

                    # ══════════════════════════════════════════════════════
                    # Phase 2: Fused online softmax + P quantization
                    # ══════════════════════════════════════════════════════
                    # In kernel: softmax_fused.online_softmax_with_quant<Is_first>(
                    #               tSrS, AbsMaxP, softmax_scale_log2)
                    # Then:      quantize(mma_k, tSrS_conversion_view)
                    #
                    # This fuses softmax with FP4 quantization of P,
                    # reusing the max-reduction from softmax as Level 1 scaling.
                    O_acc, row_max, row_sum, P_fp4_packed, P_fp4_sf = (
                        online_softmax_with_quant(
                            S, O_acc, row_max, row_sum,
                            softmax_scale_log2,
                            fp8xfp4_scale_log2,
                            fp4_scale_log2,
                            is_first_tile,
                        )
                    )
                    # P_fp4_packed: [m_size, n_size//2] uint8
                    # P_fp4_sf:     [m_size, n_size//16] fp8_e4m3fn

                    # ══════════════════════════════════════════════════════
                    # Phase 3: PV matmul via FP4MM (GEMM-II)
                    # ══════════════════════════════════════════════════════
                    # In kernel (per v_block iteration):
                    #   cute::gemm(tiled_mma_pv,
                    #     make_zip_tensor(tOrP(_, _, v_block), tOrSFP(_, _, v_block)),
                    #     make_zip_tensor(tOrVt(_, _, v_block), tOrSFVt(_, _, v_block)),
                    #     tOrO);
                    #
                    # P  [M, N] in FP4 + SFP,  V^T [D, N] in FP4 + SFVt
                    # → O_partial [M, D] fp32

                    # V is transposed: v_packed [B,H,D,L//2] (N is contiguous)
                    v_tile_packed = v_packed[b, h, :, n_start//2 : n_end//2]  # [D, n_size//2]
                    v_tile_sf     = v_sf[b, h, :, n_start//16 : n_end//16]    # [D, n_size//16]

                    O_partial = fp4mm_gemm(
                        P_fp4_packed, P_fp4_sf,      # A: P [m_size, n_size//2], sf
                        v_tile_packed, v_tile_sf,     # B: V^T [D, n_size//2], sf
                        m_size, D, n_size,
                    )
                    # O_partial: [m_size, D] fp32

                    # ── Accumulate with softmax rescaling ──────────────────
                    # In kernel (not first tile):
                    #   softmax_fused.rescale_o(tOrO_store, tOrO)
                    #     → O_store = O_store * scores_scale + O_partial
                    # Note: O_acc was already rescaled inside online_softmax_with_quant
                    O_acc = O_acc + O_partial

                # ── Finalize: normalize by row_sum ─────────────────────────
                # In kernel: softmax_fused.finalize(tOrO_store)
                #   foreach row mi:
                #     row_sum[mi] = warp_reduce_sum(row_sum[mi])
                #     inv_sum = (sum == 0 || isnan) ? 0 : 1/sum
                #     O[mi] *= inv_sum
                inv_sum = torch.where(
                    (row_sum == 0) | (row_sum != row_sum),
                    torch.zeros_like(row_sum),
                    1.0 / row_sum,
                )
                O_final = O_acc * inv_sum.unsqueeze(-1)  # [m_size, D]

                # ── Epilogue: write O to smem → TMA store to global ────────
                # In kernel: collective_epilogue.mma_store() then tma_store()
                output[b, h, m_start:m_end, :] = O_final.to(output_dtype)

    return output


# =============================================================================
# Online Softmax with Fused Two-Level P Quantization
# (mirrors softmax_fused.h :: SoftmaxFused::online_softmax_with_quant)
# =============================================================================

def online_softmax_with_quant(
    S: torch.Tensor,               # [M, N] fp32 — raw QK^T scores (with delta_s)
    O_acc: torch.Tensor,            # [M, D] fp32 — running output accumulator
    row_max: torch.Tensor,          # [M] fp32 — running row max
    row_sum: torch.Tensor,          # [M] fp32 — running row sum of exp
    softmax_scale_log2: float,      # log2(1/sqrt(D))
    fp8xfp4_scale_log2: float,      # log2(1/(448*6)) ≈ -11.39
    fp4_scale_log2: float,           # log2(1/6) ≈ -2.58
    is_first_tile: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Online softmax with fused two-level P quantization to NVFP4.

    This function is the heart of SageAttention3's innovation. It:
    1. Computes streaming softmax (handling tiles incrementally)
    2. Fuses P quantization into the softmax, reusing max-reduction results
    3. Applies two-level scaling to solve Challenge C2 (P values in [0,1])

    Two-Level P Scaling (from paper Section 3.2):
    ─────────────────────────────────────────────
    Problem: P = softmax(S) ∈ [0, 1]. If we directly compute FP8 scale factor
    as sf = max(P_block) / 6.0, then sf ∈ [0, 1/6 ≈ 0.167]. This tiny range
    causes large E4M3 representation error (only a few E4M3 values are < 0.167).

    Solution: Two-level scaling shifts P into a range where FP8 E4M3 has good
    resolution.

    Level 1 (per-row, absorbed into exp2 offset):
      Instead of computing P = exp(S * scale - max(S * scale)),
      compute P̃ = exp2(S * scale_log2 - max_scaled)
      where max_scaled = row_max * scale_log2 + log2(1/(448×6))

      This means P̃ = P * 448 * 6 = P * 2688, shifting values to [0, 2688].

    Level 2 (per-16-element block = microscaling):
      AbsMaxP[block] = max(|P̃[block_start:block_start+16]|)
      sf_p = fp8_e4m3(AbsMaxP / 6.0)   (not / 6.0 here since already in P̃ space)
      Actually, in the kernel the AbsMaxP is computed as:
        AbsMaxP = exp2(max_of_block * scale_log2 - max_scaled + log2(1/6))
      Then elements are divided by AbsMaxP before FP4 quantization.

    Result: FP8 scale factors for P now span the full useful E4M3 range,
    improving CosSim from 93.32% to 99.52% (Table 1b in paper).

    Implementation in kernel (softmax_fused.h):
    ─────────────────────────────────────────────
    For each row mi:
      1. Find per-16-block max: AbsMaxP(mi, ni) = max(|S[mi, block]|)
         using __shfl_xor_sync(1) to share max between pairs of threads
      2. Find row max: row_max(mi) = max(AbsMaxP[mi, :])
         using __shfl_xor_sync(2) across quad (4 threads per row)
      3. Compute max_scaled = row_max * scale_log2 + fp8xfp4_scale_log2
      4. S_exp[mi, ni] = exp2(S[mi,ni] * scale_log2 - max_scaled)
      5. AbsMaxP[mi, si] = exp2(block_max * scale_log2 - max_scaled + fp4_scale_log2)
      6. row_sum[mi] += sum(S_exp[mi, :])
      7. S_exp[mi, block] /= AbsMaxP[mi, block_idx]  (Level 2 normalization)
      8. cvt.e2m1(S_exp)  →  P_fp4
      9. fp8_e4m3(AbsMaxP) → P_sf

    For non-first tiles, also:
      scores_scale = exp2((prev_max - new_max) * scale_log2)
      row_sum *= scores_scale
      O_acc *= scores_scale    (rescale previous output)

    Returns:
        O_acc:     [M, D] fp32 — rescaled if not first tile
        row_max:   [M] fp32 — updated running max
        row_sum:   [M] fp32 — updated running sum
        P_packed:  [M, N//2] uint8 — P in FP4
        P_sf:      [M, N//16] fp8_e4m3fn — Level 2 scale factors
    """
    M, N = S.shape
    num_sf_blocks = (N + 15) // 16

    # ── Step 1: Per-16-element block max + row max ─────────────────────────
    # In kernel: nested reduction with warp shuffles
    abs_max_p = torch.zeros((M, num_sf_blocks), dtype=torch.float32, device=S.device)
    for mi in range(M):
        for si in range(num_sf_blocks):
            s = si * 16
            e = min(s + 16, N)
            abs_max_p[mi, si] = S[mi, s:e].max().item()

    scores_max_prev = row_max.clone()
    for mi in range(M):
        row_max[mi] = max(row_max[mi].item(), abs_max_p[mi].max().item())

    # ── Step 2: Compute exp2 with two-level offset ─────────────────────────
    S_exp = torch.zeros_like(S)

    if is_first_tile:
        for mi in range(M):
            max_scaled = row_max[mi].item() * softmax_scale_log2 + fp8xfp4_scale_log2
            # P̃ = exp2(S * scale_log2 - max_scaled)
            #    = softmax_unnorm(S) * 448 * 6
            for ni in range(N):
                S_exp[mi, ni] = 2.0 ** (S[mi, ni].item() * softmax_scale_log2 - max_scaled)
            row_sum[mi] = S_exp[mi].sum().item()

            # AbsMaxP in exp2 domain (for Level 2 sf computation)
            for si in range(num_sf_blocks):
                abs_max_p[mi, si] = 2.0 ** (
                    abs_max_p[mi, si].item() * softmax_scale_log2
                    - max_scaled + fp4_scale_log2
                )
    else:
        # ── Rescale previous accumulator ───────────────────────────────────
        for mi in range(M):
            scores_scale = 2.0 ** (
                (scores_max_prev[mi].item() - row_max[mi].item()) * softmax_scale_log2
            )
            row_sum[mi] = row_sum[mi].item() * scores_scale
            O_acc[mi] *= scores_scale   # rescale_o

            max_scaled = row_max[mi].item() * softmax_scale_log2 + fp8xfp4_scale_log2
            for ni in range(N):
                S_exp[mi, ni] = 2.0 ** (S[mi, ni].item() * softmax_scale_log2 - max_scaled)
            row_sum[mi] += S_exp[mi].sum().item()

            for si in range(num_sf_blocks):
                abs_max_p[mi, si] = 2.0 ** (
                    abs_max_p[mi, si].item() * softmax_scale_log2
                    - max_scaled + fp4_scale_log2
                )

    # ── Step 3: Level 2 normalization: divide by per-block AbsMaxP ─────────
    # In kernel: acc_conversion_flatten(j, i) /= AbsMaxP(i)
    for mi in range(M):
        for si in range(num_sf_blocks):
            s = si * 16
            e = min(s + 16, N)
            if abs_max_p[mi, si] > 0:
                S_exp[mi, s:e] /= abs_max_p[mi, si]

    # ── Step 4: Quantize P to FP4 (E2M1) ──────────────────────────────────
    # In kernel: packed_float_to_e2m1() — PTX cvt.rn.satfinite.e2m1x2.f32
    P_packed = _quantize_to_fp4_packed(S_exp)     # [M, N//2] uint8

    # ── Step 5: AbsMaxP → FP8 E4M3 scale factors ──────────────────────────
    # In kernel: packed_float_to_ue4m3() — 4 floats → 4 E4M3 packed in uint32
    P_sf = abs_max_p.to(torch.float8_e4m3fn)      # [M, N//16] fp8_e4m3fn

    return O_acc, row_max, row_sum, P_packed, P_sf


# =============================================================================
# FP4MM GEMM Pseudocode (mirrors SM120 BLOCKSCALED MMA instruction)
# =============================================================================

def fp4mm_gemm(
    a_packed: torch.Tensor,   # [M, K//2] uint8 (packed FP4)
    a_sf: torch.Tensor,       # [M, K//16] fp8_e4m3fn
    b_packed: torch.Tensor,   # [N, K//2] uint8 (TN layout)
    b_sf: torch.Tensor,       # [N, K//16] fp8_e4m3fn
    M: int, N: int, K: int,
) -> torch.Tensor:
    """
    Pseudocode for Blackwell FP4MM tensor core GEMM with block-scaled microscaling.

    Hardware instruction: SM120_16x32x64_TN_VS_NVFP4
    ─────────────────────────────────────────────────
      Atom shape: M=16, N=32, K=64
      Layout: TN (A contiguous in K, B contiguous in K)
      Data: E2M1 (FP4) + E4M3 (FP8) scale factors
      Accumulator: fp32
      "VS" = value-scale (SFA/SFB loaded alongside data)

    The MMA instruction fuses dequantization and multiply-accumulate:
      for each (mi, ni, ki) in the MMA atom:
        sf_a = float(SFA[mi, ki // 16])
        sf_b = float(SFB[ni, ki // 16])
        a_val = e2m1_to_float(A_packed[mi, ki]) * sf_a
        b_val = e2m1_to_float(B_packed[ni, ki]) * sf_b
        C[mi, ni] += a_val * b_val

    This is NOT a separate dequant→fp32_gemm; dequantization is integral
    to the tensor core instruction pipeline.

    Args:
        a_packed: [M, K//2] uint8  — packed E2M1 values
        a_sf:     [M, K//16] fp8   — scale factors for A
        b_packed: [N, K//2] uint8  — packed E2M1 values (TN = B^T row-major)
        b_sf:     [N, K//16] fp8   — scale factors for B

    Returns:
        C: [M, N] fp32
    """
    C = torch.zeros((M, N), dtype=torch.float32, device=a_packed.device)

    # NVFP4 E2M1 decode table: 4-bit index → float
    # Bit 3 = sign, bits 2-0 encode magnitude
    # Magnitudes: {0, 0.5, 1, 2, 3, 4, 6, 6(sat)}
    fp4_lut = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 6.0,      # positive (0-7)
               0.0, -0.5, -1.0, -2.0, -3.0, -4.0, -6.0, -6.0] # negative (8-15)

    def _unpack_and_dequant(packed, sf, row, k_start, K_total):
        """Dequant one 16-element microscaling block from packed FP4."""
        sf_idx = k_start // 16
        sf_val = float(sf[row, sf_idx]) if sf_idx < sf.size(-1) else 1.0
        vals = []
        for e in range(16):
            k_idx = k_start + e
            if k_idx >= K_total:
                vals.append(0.0)
                continue
            byte_idx = k_idx // 2
            if byte_idx >= packed.size(-1):
                vals.append(0.0)
                continue
            byte_val = int(packed[row, byte_idx])
            bits = (byte_val & 0x0F) if (k_idx % 2 == 0) else ((byte_val >> 4) & 0x0F)
            vals.append(fp4_lut[bits] * sf_val)
        return vals

    # Emulate tiled MMA over K-dimension in 16-element microscaling blocks
    for mi in range(M):
        for ni in range(N):
            acc = 0.0
            for k_block in range(0, K, 16):
                a_vals = _unpack_and_dequant(a_packed, a_sf, mi, k_block, K)
                b_vals = _unpack_and_dequant(b_packed, b_sf, ni, k_block, K)
                for e in range(min(16, K - k_block)):
                    acc += a_vals[e] * b_vals[e]
            C[mi, ni] = acc

    return C


# =============================================================================
# Internal Helpers — FP4 Quantization Logic
# =============================================================================

def _scaled_fp4_quant_kernel(
    x: torch.Tensor,           # [B, H, N, D] fp16/bf16
    out: torch.Tensor,         # [B, H, N, D//2] uint8
    out_sf: torch.Tensor,      # [B, H, N, D//16] fp8_e4m3fn
    permute: bool = False,
):
    """
    Pseudocode for scaled_fp4_quant_kernel / scaled_fp4_quant_kernel<permute=true>.

    Mirrors fp4_quantization_4d.cu :: scaled_fp4_quant_kernel<head_dim, BLOCK_SIZE, permute, T>

    Thread mapping:
      gridDim  = (ceil(N/128), B, H)
      blockDim = (128 * D / 16,)   — 128 tokens × (D/16) blocks per token
      Each thread: 1 microscaling block = 16 consecutive elements

    K permutation (permute=True):
      Within each 32-token group, tokens are reordered to match
      FP4MM B-operand memory layout:
        local_residue = local_token_id % 32
        permuted = (local_residue // 8) * 2
                 + ((local_residue % 8) // 2) * 8
                 + (local_residue % 2)
    """
    B, H, N, D = x.shape
    CVT = 16  # CVT_FP4_ELTS_PER_THREAD

    for b_idx in range(B):
        for h_idx in range(H):
            for token_block in range((N + 127) // 128):
                for local_tid in range(128):
                    token_id = token_block * 128 + local_tid

                    if permute:
                        r = local_tid % 32
                        load_tid = (token_block * 128
                                    + (local_tid // 32) * 32
                                    + (r // 8) * 2
                                    + ((r % 8) // 2) * 8
                                    + (r % 2))
                    else:
                        load_tid = token_id

                    if load_tid >= N:
                        continue

                    for col_block in range(0, D, CVT):
                        # Load PackedVec (16 × fp16 = 32 bytes)
                        vals = x[b_idx, h_idx, load_tid, col_block:col_block+CVT].float()

                        # Max abs via __habs2 + __hmax2
                        max_abs = vals.abs().max().item()

                        # Scale factor: max_abs / 6.0 → FP8 E4M3
                        sf_f32 = max_abs / 6.0
                        sf_fp8 = torch.tensor(sf_f32, dtype=torch.float8_e4m3fn)
                        sf_f32 = float(sf_fp8)  # round-trip for consistency

                        # Scale and convert
                        sf_inv = (1.0 / sf_f32) if sf_f32 > 0 else 0.0
                        scaled = vals * sf_inv

                        # PTX: cvt.rn.satfinite.e2m1x2.f32 → pack 2 per byte
                        for i in range(0, CVT, 2):
                            lo = _float_to_e2m1(scaled[i].item())
                            hi = _float_to_e2m1(scaled[i+1].item())
                            out[b_idx, h_idx, token_id, (col_block + i) // 2] = lo | (hi << 4)

                        out_sf[b_idx, h_idx, token_id, col_block // CVT] = sf_fp8


def _scaled_fp4_quant_transpose_kernel(
    x: torch.Tensor,           # [B, H, N, D] fp16/bf16
    out: torch.Tensor,         # [B, H, D, N//2] uint8
    out_sf: torch.Tensor,      # [B, H, D, N//16] fp8_e4m3fn
):
    """
    Pseudocode for scaled_fp4_quant_trans_kernel.

    Mirrors fp4_quantization_4d.cu :: scaled_fp4_quant_trans_kernel

    Key difference from non-transpose version: uses __shared__ memory for
    in-tile transposition.

    Thread mapping:
      NUM_THREADS_PER_TOKEN = D / 16        (threads for one token's head_dim)
      NUM_THREADS_PER_SEQ   = 128 / 16 = 8  (threads for one head_dim position across seq)

    Shared memory: T shared_input[BLOCK_SIZE * D]
      Write (coalesced on D): shared[local_token * D + col] = x[token, col]
      Read  (transposed):     val = shared[d_pos + local_seq * D]

    Output layout: [B, H, D, N//2] — head_dim is the "row" dimension,
    with seq_len packed along the last dimension.
    """
    B, H, N, D = x.shape
    CVT = 16
    BLOCK_SIZE = 128

    for b_idx in range(B):
        for h_idx in range(H):
            for token_block in range((N + BLOCK_SIZE - 1) // BLOCK_SIZE):
                # ── Shared memory: load tile [BLOCK_SIZE, D] coalesced ─
                shared = torch.zeros((BLOCK_SIZE, D), dtype=x.dtype, device=x.device)
                t_start = token_block * BLOCK_SIZE
                t_end = min(t_start + BLOCK_SIZE, N)
                shared[:t_end - t_start, :] = x[b_idx, h_idx, t_start:t_end, :]

                # ── __syncthreads() ──

                # ── Transposed access: for each head_dim position, read seq ─
                for d_idx in range(D):
                    for seq_block in range(0, BLOCK_SIZE, CVT):
                        # Read 16 elements along seq dim at fixed d_idx
                        vals = shared[seq_block:seq_block + CVT, d_idx].float()

                        # Same quantization as non-transpose version
                        max_abs = vals.abs().max().item()
                        sf_f32 = max_abs / 6.0
                        sf_fp8 = torch.tensor(sf_f32, dtype=torch.float8_e4m3fn)
                        sf_f32 = float(sf_fp8)
                        sf_inv = (1.0 / sf_f32) if sf_f32 > 0 else 0.0
                        scaled = vals * sf_inv

                        global_seq = t_start + seq_block
                        for i in range(0, CVT, 2):
                            lo = _float_to_e2m1(scaled[i].item())
                            hi = _float_to_e2m1(scaled[i+1].item())
                            out[b_idx, h_idx, d_idx, (global_seq + i) // 2] = lo | (hi << 4)

                        out_sf[b_idx, h_idx, d_idx, global_seq // CVT] = sf_fp8


def _float_to_e2m1(val: float) -> int:
    """
    Convert a pre-scaled float to NVFP4 E2M1 4-bit representation.

    Emulates PTX: cvt.rn.satfinite.e2m1x2.f32
    Used in fp32_vec_to_e2m1() assembly in fp4_quantization_4d.cu

    E2M1 encoding (4 bits):
      Bit 3: sign
      Bits 2-0: magnitude code
        000 → 0.0
        001 → 0.5
        010 → 1.0
        011 → 2.0
        100 → 3.0
        101 → 4.0
        110 → 6.0
        111 → ∞ (saturated to 6.0 by satfinite)

    Rounding: to nearest, ties to even.
    Saturation: ∞ → 6.0 (satfinite mode).
    """
    sign = 0 if val >= 0 else 1
    mag = abs(val)

    # Round to nearest representable E2M1 magnitude
    if mag < 0.25:
        code = 0b000     # 0.0
    elif mag < 0.75:
        code = 0b001     # 0.5
    elif mag < 1.5:
        code = 0b010     # 1.0
    elif mag < 2.5:
        code = 0b011     # 2.0
    elif mag < 3.5:
        code = 0b100     # 3.0
    elif mag < 5.0:
        code = 0b101     # 4.0
    else:
        code = 0b110     # 6.0 (satfinite)

    return (sign << 3) | code


def _quantize_to_fp4_packed(x: torch.Tensor) -> torch.Tensor:
    """Pack a float tensor [M, N] into FP4 [M, N//2] uint8.

    Packing: lower nibble = even index, upper nibble = odd index.
    """
    M, N = x.shape
    assert N % 2 == 0
    out = torch.zeros((M, N // 2), dtype=torch.uint8, device=x.device)
    for i in range(M):
        for j in range(0, N, 2):
            lo = _float_to_e2m1(x[i, j].item())
            hi = _float_to_e2m1(x[i, j + 1].item())
            out[i, j // 2] = lo | (hi << 4)
    return out


# =============================================================================
# Scale Factor Layout Detail (from blockscaled_layout.h)
# =============================================================================
#
# NVFP4 microscaling stores scale factors in a tiled layout optimized for
# TMA bulk copies and MMA instruction operand access patterns.
#
# BlockScaledConfig<SFVectorSize=16>:
#   - SfAtom layout: maps logical (row, col_sf) to physical offset within
#     a 64×4 tile (64 rows × 4 scale factor columns).
#   - Physical offset formula:
#       offset = (col_sf // 4) * 256
#              + (col_sf % 4)
#              + (row % 64 // 16) * 4
#              + (row % 16) * 16
#   - This interleaving ensures each TMA load brings in the exact scale
#     factors needed by the threads in one MMA instruction atom.
#
# For Q/K scale factors [seqlen, D//16]:
#   tile_atom_to_shape_SFQKV() tiles the SfAtom across rows and columns.
#
# For V^T scale factors [D, seqlen//16]:
#   tile_atom_to_shape_SFVt() handles the transposed layout.


# =============================================================================
# Example Usage
# =============================================================================

def example_usage():
    """Demonstrate SageAttention3 data flow with concrete tensor shapes."""

    B, H, L, D = 1, 8, 256, 128
    device = torch.device("cuda")
    dtype = torch.bfloat16

    q = torch.randn(B, H, L, D, device=device, dtype=dtype)
    k = torch.randn(B, H, L, D, device=device, dtype=dtype)
    v = torch.randn(B, H, L, D, device=device, dtype=dtype)

    print(f"Input:  q={list(q.shape)}, k={list(k.shape)}, v={list(v.shape)}")

    # Preprocessing
    q_p, k_p, v_p, ds = preprocess_qkv(q, k, v, per_block_mean=True)
    print(f"After preprocess:")
    print(f"  q={list(q_p.shape)}, k={list(k_p.shape)}, v={list(v_p.shape)}")
    print(f"  delta_s={list(ds.shape)} (G={ds.size(2)} groups)")

    # Quantization
    q_q = scale_and_quant_fp4(q_p)
    k_q = scale_and_quant_fp4_permute(k_p)
    v_q = scale_and_quant_fp4_transpose(v_p)
    print(f"After quantization:")
    print(f"  Q: data={list(q_q.data.shape)} {q_q.data.dtype}, "
          f"sf={list(q_q.scale.shape)} {q_q.scale.dtype}")
    print(f"  K: data={list(k_q.data.shape)} {k_q.data.dtype}, "
          f"sf={list(k_q.scale.shape)} {k_q.scale.dtype}")
    print(f"  V: data={list(v_q.data.shape)} {v_q.data.dtype}, "
          f"sf={list(v_q.scale.shape)} {v_q.scale.dtype}")

    # Full pipeline
    out = sageattn3_blackwell(q, k, v, is_causal=True)
    print(f"\nOutput: {list(out.shape)} {out.dtype}")


if __name__ == "__main__":
    print("SageAttention3 Pseudocode — Kernel-Aligned Implementation")
    print("=" * 60)
    print()
    print("Key design principles:")
    print("  1. FP4MM GEMM used directly — no dequant before matmul")
    print("  2. Two-level P quantization fused into online softmax")
    print("  3. Function signatures match actual CUDA/CUTLASS code")
    print("  4. Tensor shapes annotated at every step")
    print("  5. Scale factor layout matches hardware TMA/MMA requirements")
    print("  6. Data flow mirrors warp-specialized persistent kernel")
    print()
    # example_usage()  # uncomment on Blackwell GPU with triton
