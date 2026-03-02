"""
SageAttention3 Pseudocode — NVFP4 Microscaling Attention for Blackwell GPUs
============================================================================

Based on: "SageAttention3: Microscaling FP4 Attention for Inference" (NeurIPS 2025)
Codebase: SageAttention-Fork/sageattention3_blackwell/

This pseudocode mirrors the actual kernel data flow in the SageAttention3 codebase,
covering three stages:
  1. Preprocessing  — smooth Q/K, pad, compute delta correction
  2. FP4 Quantization — three variants for Q, K (permuted), V (transposed)
  3. Attention Kernel  — tiled FP4MM with online softmax + two-level P quantisation

Key design choices (from paper § 2):
  • NVFP4 E2M1 with 1×16 block size, E4M3 FP8 scale factors
  • Two-level scaling for P matrix  (s_P1: per-row fp32, s_P2: per-16-element fp8)
  • Q+K smoothing inherited from SageAttention2
  • K column permutation to match FP4MM accumulator layout
  • Fused softmax-max and quantization to reduce shuffles

Tensor Shape Conventions (all tensors are [B, H, L, D] unless noted):
  B = batch_size    H = num_heads
  L = seq_len       D = head_dim (64 or 128, must be < 256)
  G = L_padded // 128   (number of 128-token blocks for per-block-mean)
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional


# ═══════════════════════════════════════════════════════════════════════════════
# 0. Top-Level API  (mirrors sageattn3/api.py :: sageattn3_blackwell)
# ═══════════════════════════════════════════════════════════════════════════════

def sageattn3_blackwell(
    q: torch.Tensor,                    # [B, H, L, D]  fp16/bf16
    k: torch.Tensor,                    # [B, H, L, D]  fp16/bf16
    v: torch.Tensor,                    # [B, H, L, D]  fp16/bf16
    attn_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    per_block_mean: bool = True,
    **kwargs,
) -> torch.Tensor:                      # [B, H, L, D]  fp16/bf16
    """
    Main entry point.  Data flow:

        q,k,v  [B,H,L,D] fp16/bf16
            │  preprocess_qkv()
            ▼
        q_smooth  [B,H,L_pad,D]     – Q with per-block mean subtracted
        k_smooth  [B,H,L_pad,D]     – K with global mean subtracted
        v_padded  [B,H,L_pad,D]     – V zero-padded only
        delta_s   [B,H,G,L_pad] fp32 – correction for Q smoothing
            │
            │  scale_and_quant_fp4*()   (3 separate quantise kernels)
            ▼
        q_fp4  (packed_uint8 [B,H,L_pad,D//2],  scale_fp8 [B,H,L_pad,D//16])
        k_fp4  (packed_uint8 [B,H,L_pad,D//2],  scale_fp8 [B,H,L_pad,D//16])  – column-permuted
        v_fp4  (packed_uint8 [B,H,D,L_pad//2],  scale_fp8 [B,H,D,L_pad//16])  – transposed
            │
            │  blockscaled_fp4_attn()   (CUTLASS kernel on Blackwell)
            ▼
        output [B,H,L,D] fp16/bf16
    """
    assert q.size(-1) < 256, f"Unsupported head_dim={q.size(-1)}, max 256"

    QL = q.size(2)  # original query seq len
    KL = k.size(2)  # original key seq len (before pad)
    is_bf16 = q.dtype == torch.bfloat16

    # ── Stage 1: Preprocessing ────────────────────────────────────────────
    q, k, v, delta_s = preprocess_qkv(q, k, v, per_block_mean)
    # shapes after pad:  q,k,v [B,H,L_pad,D],  delta_s [B,H,G,L_pad]

    # ── Stage 2: FP4 Quantisation (three separate CUDA kernels) ───────────
    qlist = scale_and_quant_fp4(q)              # standard
    klist = scale_and_quant_fp4_permute(k)      # column-permuted for FP4MM accum layout
    vlist = scale_and_quant_fp4_transpose(v)    # transposed for PV matmul

    # ── Stage 3: Attention kernel ─────────────────────────────────────────
    output = blockscaled_fp4_attn(
        qlist, klist, vlist,
        delta_s, KL,
        is_causal=is_causal,
        per_block_mean=per_block_mean,
        is_bf16=is_bf16,
    )

    # ── Crop padding ──────────────────────────────────────────────────────
    return output[:, :, :QL, :].contiguous()


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Preprocessing  (mirrors sageattn3/api.py :: preprocess_qkv)
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess_qkv(
    q: torch.Tensor,  # [B, H, L, D]
    k: torch.Tensor,  # [B, H, L, D]
    v: torch.Tensor,  # [B, H, L, D]
    per_block_mean: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Three things happen here:
      (a) Smooth K  – subtract global mean along seq dim (lossless, softmax shift-invariant)
      (b) Pad to multiple of 128
      (c) Smooth Q  – subtract per-block (128-token) or global mean, produce delta correction

    Returns:
        q   [B,H,L_pad,D]    – smoothed Q
        k   [B,H,L_pad,D]    – smoothed K
        v   [B,H,L_pad,D]    – zero-padded V
        delta_s [B,H,G,L_pad] fp32  – correction term  (G = L_pad//128 if per_block, else 1)
    """

    def pad_128(x: torch.Tensor) -> torch.Tensor:
        """Pad seq dim to next multiple of 128 (required by FP4MM tile size)."""
        L = x.size(2)
        pad = (128 - L % 128) % 128
        if pad == 0:
            return x.contiguous()
        return F.pad(x, (0, 0, 0, pad), value=0.0).contiguous()

    # (a) K smoothing  ── γ(K) = K − mean(K)
    # Softmax is shift-invariant: softmax(Q·Kᵀ − Q·mean(K)ᵀ) = softmax(Q·Kᵀ)
    # This removes channel-wise outliers, improving quantisation accuracy (<0.2% overhead)
    k = k - k.mean(dim=-2, keepdim=True)   # k_mean [B,H,1,D]; broadcast subtract

    # (b) Pad  ── L_pad = ceil(L/128) * 128
    q, k, v = map(pad_128, [q, k, v])      # [B,H,L_pad,D] each

    # (c) Q smoothing + delta correction
    if per_block_mean:
        # Per-block (128 tokens) mean subtraction via Triton kernel
        q, qm = triton_group_mean(q, group_size=128)
        # q  [B,H,L_pad,D]   – centred within each block
        # qm [B,H,G,D]       – per-block means, G = L_pad // 128
    else:
        qm = q.mean(dim=-2, keepdim=True)   # [B,H,1,D]
        q  = q - qm                         # [B,H,L_pad,D]

    # Delta correction:  delta_s = qm @ K^T
    #   Math: softmax((Q − qm)·Kᵀ + qm·Kᵀ)·V  ≈  softmax(Q_smooth·Kᵀ + delta_s)·V
    # per_block_mean:  qm [B,H,G,D] @ k^T [B,H,D,L_pad] → [B,H,G,L_pad]
    # global mean:     qm [B,H,1,D] @ k^T [B,H,D,L_pad] → [B,H,1,L_pad]
    delta_s = torch.matmul(qm, k.transpose(-2, -1))  # [B,H,G,L_pad] or [B,H,1,L_pad]
    delta_s = delta_s.to(torch.float32).contiguous()

    return q, k, v, delta_s


def triton_group_mean(
    q: torch.Tensor,        # [B, H, L, D]
    group_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Triton kernel that computes per-group mean and subtracts it in one pass.

    Grid: (B, H, num_groups)
    Each program handles one 128-token block, computing the mean over dim D
    and writing both the centred output and the mean.

    Returns:
        q_out  [B,H,L,D]            – Q with per-group mean subtracted
        q_mean [B,H,num_groups,D]    – the means (needed for delta_s)
    """
    B, H, L, D = q.shape
    assert L % group_size == 0
    G = L // group_size

    q_out  = torch.empty_like(q)
    q_mean = torch.empty(B, H, G, D, device=q.device, dtype=q.dtype)

    # --- Triton kernel pseudocode (JIT compiled in practice) ---
    # @triton.jit
    # def group_mean_kernel(q_ptr, q_out_ptr, qm_ptr, ...):
    #     pid_b, pid_h, pid_g = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    #     offs = pid_g * GROUP_SIZE + tl.arange(0, GROUP_SIZE)        # [128]
    #     block = tl.load(q_ptr + ...)                                 # [128, D]
    #     mean  = tl.sum(block, axis=0) / GROUP_SIZE                   # [D]
    #     tl.store(q_out_ptr + ..., block - mean)
    #     tl.store(qm_ptr + ..., mean)

    # Python-level emulation:
    q_blocks = q.reshape(B, H, G, group_size, D)         # [B,H,G,128,D]
    q_mean   = q_blocks.mean(dim=3)                       # [B,H,G,D]
    q_out    = (q_blocks - q_mean.unsqueeze(3)).reshape(B, H, L, D)

    return q_out, q_mean


# ═══════════════════════════════════════════════════════════════════════════════
# 2. FP4 Quantisation Kernels
# ═══════════════════════════════════════════════════════════════════════════════
#
# All three kernels share the same core logic:
#   For each 1×16 microscaling block:
#     (1) find max_abs over 16 elements
#     (2) compute fp8 scale: sf = max_abs / 6.0,  then round to E4M3
#     (3) scale_inv = 1/sf  (or 0 if sf==0)
#     (4) quantise each element: fp4_val = cvt.rn.satfinite.e2m1(x * scale_inv)
#     (5) pack 2 fp4 values per uint8 byte
#
# The three variants differ only in memory layout transformations:
#
# ┌────────────────────────────┬──────────────────────────────────────────────┐
# │ Kernel                     │ Difference                                   │
# ├────────────────────────────┼──────────────────────────────────────────────┤
# │ scale_and_quant_fp4        │ Standard row-major layout                    │
# │ scale_and_quant_fp4_permute│ K column permute to match FP4MM accum layout │
# │ scale_and_quant_fp4_trans  │ V transposed (seq ↔ head_dim) for PV matmul  │
# └────────────────────────────┴──────────────────────────────────────────────┘

# ----- NVFP4 E2M1 value table -----
#  4-bit encoding → float magnitude (sign handled separately)
#  bits [3]=sign  [2:1]=exponent  [0]=mantissa
#
#  Positive values (sign=0):
#    0b0000 → +0.0     0b0001 → +0.5
#    0b0010 → +1.0     0b0011 → +1.5
#    0b0100 → +2.0     0b0101 → +3.0
#    0b0110 → +4.0     0b0111 → +6.0
#  Negative mirror with sign bit set.


def scale_and_quant_fp4(
    x: torch.Tensor,    # [B, H, N, D] fp16/bf16
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Standard NVFP4 quantisation for Q.

    CUDA kernel config (from fp4_quantization_4d.cu):
        block = (BLOCK_SIZE * HEAD_DIM / 16, 1, 1)   e.g. (128*128/16 = 1024)
        grid  = (ceil(N/BLOCK_SIZE), B, H)
        Each thread processes 16 contiguous elements = one microscaling block.

    Returns:
        packed_fp4 [B,H,N,D//2]   uint8   – 2 fp4 values per byte
        fp8_scale  [B,H,N,D//16]  fp8_e4m3 – one scale per 16-element block
    """
    B, H, N, D = x.shape
    assert D % 16 == 0

    packed_fp4 = torch.empty((B, H, N, D // 2), device=x.device, dtype=torch.uint8)
    fp8_scale  = torch.empty((B, H, N, D // 16), device=x.device, dtype=torch.float8_e4m3fn)

    # -- CUDA kernel pseudocode --
    # for each (batch, head, token, element_group_of_16):
    #     vec[16] = load 16 contiguous fp16 elements
    #     max_abs = hmax2 reduction over vec
    #     sf      = max_abs / 6.0
    #     sf_fp8  = (fp8_e4m3)sf              # cast to E4M3
    #     sf      = (float)sf_fp8             # use quantised scale for consistency
    #     inv     = 1.0 / sf  if sf > 0 else 0
    #     for i in 0..15:
    #         val_scaled = vec[i] * inv
    #         fp4_val    = cvt.rn.satfinite.e2m1(val_scaled)   # PTX SM≥100
    #     pack 16 fp4 values → 8 bytes (2 per byte)
    #     store packed + sf_fp8

    _quant_fp4_core(x, packed_fp4, fp8_scale)
    return packed_fp4, fp8_scale


def scale_and_quant_fp4_permute(
    x: torch.Tensor,    # [B, H, N, D] fp16/bf16   (K matrix)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    FP4 quantisation with column permutation for K.

    The permutation reorders tokens within each 32-element group to match
    the FP4MM accumulator register layout on Blackwell, avoiding costly
    thread shuffles during the QK^T matmul.

    Permutation pattern (per group of 32 consecutive tokens):
        original: [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,
                   16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
        permuted: [ 0, 1, 8, 9,16,17,24,25, 2, 3,10,11,18,19,26,27,
                    4, 5,12,13,20,21,28,29, 6, 7,14,15,22,23,30,31]

    Formula (from CUDA kernel):
        local_residue = local_token_id % 32
        permuted_id = (local_token_id // 32) * 32
                    + (local_residue // 8) * 2
                    + ((local_residue % 8) // 2) * 8
                    + (local_residue % 8) % 2

    Returns:  same shapes as scale_and_quant_fp4
        packed_fp4 [B,H,N,D//2]   uint8
        fp8_scale  [B,H,N,D//16]  fp8_e4m3
    """
    B, H, N, D = x.shape
    assert D % 16 == 0

    packed_fp4 = torch.empty((B, H, N, D // 2), device=x.device, dtype=torch.uint8)
    fp8_scale  = torch.empty((B, H, N, D // 16), device=x.device, dtype=torch.float8_e4m3fn)

    # Same quantisation logic as scale_and_quant_fp4, but each thread
    # loads from a permuted token index (load_token_id) while storing
    # to the canonical token_id position.
    _quant_fp4_core(x, packed_fp4, fp8_scale, permute=True)
    return packed_fp4, fp8_scale


def scale_and_quant_fp4_transpose(
    x: torch.Tensor,    # [B, H, N, D] fp16/bf16   (V matrix)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    FP4 quantisation with transposition for V.

    Output tensors have seq and head_dim swapped so that PV matmul
    (P [M, N] @ V^T [N, D]  ≡  P [M, N] @ V_transposed [D, N])
    reads V contiguously along the N (sequence) dimension.

    CUDA kernel uses shared memory for the transpose:
        1. Coalesced load [BLOCK_SIZE, D] into shared memory
        2. __syncthreads()
        3. Read in transposed order [D, BLOCK_SIZE]
        4. Quantise to FP4 and store

    Returns:
        packed_fp4 [B,H,D,N//2]   uint8      – note axes swapped
        fp8_scale  [B,H,D,N//16]  fp8_e4m3   – note axes swapped
    """
    B, H, N, D = x.shape
    assert D % 16 == 0 and N % 16 == 0

    packed_fp4 = torch.empty((B, H, D, N // 2), device=x.device, dtype=torch.uint8)
    fp8_scale  = torch.empty((B, H, D, N // 16), device=x.device, dtype=torch.float8_e4m3fn)

    _quant_fp4_core(x, packed_fp4, fp8_scale, transpose=True)
    return packed_fp4, fp8_scale


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Attention Kernel  (mirrors sageattn3/blackwell/ CUTLASS kernel)
# ═══════════════════════════════════════════════════════════════════════════════

def blockscaled_fp4_attn(
    qlist: Tuple[torch.Tensor, torch.Tensor],   # (q_packed [B,H,L,D//2], q_sf [B,H,L,D//16])
    klist: Tuple[torch.Tensor, torch.Tensor],   # (k_packed [B,H,L,D//2], k_sf [B,H,L,D//16])
    vlist: Tuple[torch.Tensor, torch.Tensor],   # (v_packed [B,H,D,L//2], v_sf [B,H,D,L//16])
    delta_s: torch.Tensor,                      # [B,H,G,L] fp32
    KL: int,                                    # unpadded key seq len (for masking)
    is_causal: bool = False,
    per_block_mean: bool = True,
    is_bf16: bool = True,
) -> torch.Tensor:                              # [B,H,L,D] fp16/bf16
    """
    CUTLASS warp-specialised attention kernel with FP4MM tensor cores.

    Tile sizes (from kernel_traits.h):
        kBlockM = 128   (query tile along seq)
        kBlockN = 128   (key/value tile along seq)
        kBlockK = D     (head dim, no tiling)

    Warp organisation (3 warp groups):
        WarpGroup 0  (Producer): 1 warp loads Q/K/V via TMA, 1 warp stores O via TMA
        WarpGroup 1  (Consumer0): FP4MM matmuls + softmax
        WarpGroup 2  (Consumer1): same as Consumer0 (unused in single-CTA mode)

    Pipeline: async TMA loads with kStages deep buffering (default 2).

    Algorithm (per query tile m_block, iterating over key tiles n_block):
        1. Load Q tile + Q scales from global → shared → registers (once per m_block)
        2. For each n_block (from last to first if causal):
           a. Load K tile + K scales + delta_s tile (via TMA pipeline)
           b. GEMM-I:  S = FP4MM(Q_fp4, sf_Q, K_fp4, sf_K)   [M×N] fp32 accum
           c. Add delta_s correction to S
           d. Apply causal mask (if enabled)
           e. Online softmax with fused two-level P quantisation:
              • Compute per-16-element max (reused from softmax row-max)
              • exp(S - max) → unnormalised P̃ in fp32
              • Level-1 scale:  s_P1 = row_max(P̃) / (448 × 6)    per-row fp32
              • Level-2 scale:  P̃ / s_P1 → standard NVFP4 microscaling
              • Quantise to E2M1 fp4 with E4M3 fp8 scale factors
           f. Load V tile + V scales (transposed, via TMA pipeline)
           g. GEMM-II:  O_tile = FP4MM(P_fp4, sf_P2, V_fp4, sf_V) × s_P1
           h. Accumulate with online softmax rescaling:
              O_accum = diag(exp(m_old - m_new)) · O_accum + O_tile
        3. Final normalisation:   O = O_accum / row_sum
        4. Store O tile via TMA

    Returns:  output [B,H,L_pad,D]
    """
    q_packed, q_sf = qlist
    k_packed, k_sf = klist
    v_packed, v_sf = vlist

    # Reconstruct dimensions
    B, H, L, D_half = q_packed.shape
    D = q_sf.size(-1) * 16                        # head dim
    softmax_scale = D ** (-0.5)

    TILE_M = 128
    TILE_N = 128
    num_m_tiles = (L + TILE_M - 1) // TILE_M
    num_n_tiles = (L + TILE_N - 1) // TILE_N
    G = delta_s.size(-2)                          # number of Q-mean groups

    out_dtype = torch.bfloat16 if is_bf16 else torch.float16
    output = torch.empty((B, H, L, D), device=q_packed.device, dtype=out_dtype)

    # ── Per query-tile loop (in CUDA: grid dim over m_block, B, H) ────────
    for b in range(B):
        for h in range(H):
            for m in range(num_m_tiles):

                m_start = m * TILE_M
                m_end   = min(m_start + TILE_M, L)
                m_size  = m_end - m_start

                # Determine how many key tiles to process
                if is_causal:
                    n_block_max = min(num_n_tiles,
                                     ((m + 1) * TILE_M + L - L) + TILE_N - 1) // TILE_N
                else:
                    n_block_max = num_n_tiles

                # ── Load Q tile once per m_block ──────────────────────────
                q_tile = _dequant_fp4(
                    q_packed[b, h, m_start:m_end],  # [m_size, D//2]
                    q_sf[b, h, m_start:m_end],      # [m_size, D//16]
                )   # → [m_size, D] fp32

                # Online softmax accumulators (per-row fp32)
                row_max = torch.full((m_size,), float('-inf'), dtype=torch.float32,
                                     device=q_packed.device)
                row_sum = torch.zeros(m_size, dtype=torch.float32,
                                      device=q_packed.device)
                o_acc   = torch.zeros(m_size, D, dtype=torch.float32,
                                      device=q_packed.device)

                # ── Iterate over key tiles (last → first for causal) ─────
                for n in reversed(range(n_block_max)):

                    n_start = n * TILE_N
                    n_end   = min(n_start + TILE_N, L)
                    n_size  = n_end - n_start

                    # ── GEMM-I: S = Q_fp4 × K_fp4^T ──────────────────────
                    k_tile = _dequant_fp4(
                        k_packed[b, h, n_start:n_end],  # [n_size, D//2]
                        k_sf[b, h, n_start:n_end],      # [n_size, D//16]
                    )  # → [n_size, D] fp32

                    # FP4MM emulation: [m_size, D] @ [n_size, D]^T → [m_size, n_size]
                    S = torch.matmul(q_tile, k_tile.transpose(0, 1))  # [m_size, n_size]
                    S = S * softmax_scale                              # scale by 1/√D

                    # ── Add delta correction for Q smoothing ──────────────
                    if per_block_mean:
                        group_id = m_start // 128
                        ds = delta_s[b, h, group_id, n_start:n_end]   # [n_size]
                    else:
                        ds = delta_s[b, h, 0, n_start:n_end]          # [n_size]
                    S = S + ds.unsqueeze(0) * softmax_scale            # broadcast [1, n_size]

                    # ── Causal mask ───────────────────────────────────────
                    if is_causal:
                        for i in range(m_size):
                            q_pos = m_start + i
                            for j in range(n_size):
                                k_pos = n_start + j
                                if k_pos > q_pos:
                                    S[i, j] = float('-inf')

                    # ── Padding mask (mask out padded key positions) ──────
                    for j in range(n_size):
                        if n_start + j >= KL:
                            S[:, j] = float('-inf')

                    # ── Online softmax + two-level P quantisation ─────────
                    # (fused in real kernel:  softmax_fused.h  online_softmax_with_quant)

                    old_max = row_max.clone()

                    # Step 1: row-wise max over this tile (reused for quantisation)
                    tile_max = S.max(dim=-1).values                # [m_size]
                    row_max  = torch.maximum(row_max, tile_max)    # update running max

                    # Step 2: exponentiate with numerical stability
                    #   In real kernel: ptx_exp2(S * scale_log2 - max_scaled)
                    #   where max_scaled = row_max * scale_log2 + log2(1/(448*6))
                    P_unnorm = torch.exp(S - row_max.unsqueeze(-1))  # [m_size, n_size]

                    # Step 3: rescale previous accumulators for new max
                    correction = torch.exp(old_max - row_max)          # [m_size]
                    row_sum = row_sum * correction
                    o_acc   = o_acc * correction.unsqueeze(-1)

                    # Step 4: accumulate row sums
                    row_sum = row_sum + P_unnorm.sum(dim=-1)           # [m_size]

                    # ── Two-level P quantisation (paper § 2.2) ────────────
                    # Level 1:  per-row fp32 scale to expand P̃ into [0, 448×6]
                    #   s_P1 = row_max(P̃) / (448 × 6)
                    # Level 2:  standard NVFP4 microscaling (per-16-element fp8 scale)
                    #   P̃_scaled = P̃ / s_P1
                    #   s_P2, P̂ = φ(P̃_scaled)     # NVFP4 quantise

                    p_row_max = P_unnorm.max(dim=-1, keepdim=True).values  # [m_size, 1]
                    s_P1 = p_row_max / (448.0 * 6.0)                       # [m_size, 1]
                    s_P1 = torch.where(s_P1 > 0, s_P1, torch.ones_like(s_P1))

                    P_level2 = P_unnorm / s_P1                             # [m_size, n_size]
                    # In real kernel:  P_level2 is quantised to fp4 and
                    # s_P2 (fp8 per 16 elements) is computed.
                    # We emulate as float for pseudocode clarity.

                    # ── GEMM-II: O_tile = P_fp4 × V_fp4 ──────────────────
                    v_tile = _dequant_fp4_transposed(
                        v_packed[b, h, :, n_start // 2 : n_end // 2],  # [D, n_size//2]
                        v_sf[b, h, :, n_start // 16 : n_end // 16],    # [D, n_size//16]
                        n_size,
                    )  # → [n_size, D] fp32   (un-transposed for matmul)

                    # FP4MM emulation: [m_size, n_size] @ [n_size, D] → [m_size, D]
                    # In real kernel the Level-2 quantised P_fp4 + s_P2 is used
                    o_tile = torch.matmul(P_level2, v_tile)            # [m_size, D]
                    o_tile = o_tile * s_P1                              # apply Level-1 scale back

                    # ── Accumulate output ─────────────────────────────────
                    o_acc = o_acc + o_tile

                # ── Final normalisation  ──────────────────────────────────
                o_final = o_acc / row_sum.unsqueeze(-1)                # [m_size, D]
                output[b, h, m_start:m_end, :] = o_final.to(out_dtype)

    return output


# ═══════════════════════════════════════════════════════════════════════════════
# 4. FP4 Quantisation Core  (mirrors fp4_quantization_4d.cu kernels)
# ═══════════════════════════════════════════════════════════════════════════════

def _quant_fp4_core(
    x: torch.Tensor,           # [B, H, N, D]
    packed_out: torch.Tensor,  # [B, H, ?, ?]  pre-allocated
    scale_out: torch.Tensor,   # [B, H, ?, ?]  pre-allocated
    permute: bool = False,
    transpose: bool = False,
) -> None:
    """
    Core quantisation logic shared by all three FP4 kernels.

    CUDA kernel structure (from fp4_quantization_4d.cu):
        Thread block: (BLOCK_SIZE * HEAD_DIM / CVT_FP4_ELTS_PER_THREAD, 1, 1)
        Grid:         (ceil(num_tokens / BLOCK_SIZE), batch_size, num_heads)
        CVT_FP4_ELTS_PER_THREAD = 16
        BLOCK_SIZE = 128

    Each thread:
        1. Computes its (token_id, element_offset) from threadIdx.x
        2. Loads 16 fp16/bf16 elements as a PackedVec (32 bytes)
        3. hmax2 reduction → max_abs (for CVT_FP4_ELTS_PER_THREAD=8, shuffle with neighbor)
        4. sf = max_abs / 6.0;  sf_fp8 = (e4m3)sf;  sf = (float)sf_fp8
        5. scale_inv = 1/sf  (or 0)
        6. Multiply each element by scale_inv, convert to float2
        7. PTX cvt.rn.satfinite.e2m1x2.f32 → pack 8 fp4 values per uint32
        8. Store packed fp4 (8 bytes for 16 elements) and sf_fp8 (1 byte)

    Scale factor storage layout (interleaved for TMA):
        Within each 64-row block, sf is stored at:
            offset = (col_id_local / 4) * 256
                   + (col_id_local % 4)
                   + (row_id_local / 16) * 4
                   + (row_id_local % 16) * 16
    """
    B, H, N, D = x.shape
    CVT_FP4_ELTS = 16

    if transpose:
        # Transpose kernel: uses shared memory to reorder [N, D] → [D, N]
        # then quantises along the N dimension (which is now the fast axis)
        x_t = x.permute(0, 1, 3, 2).contiguous()   # [B,H,D,N]
        for b in range(B):
            for h in range(H):
                for d_idx in range(D):
                    for blk_start in range(0, N, CVT_FP4_ELTS):
                        blk_end = min(blk_start + CVT_FP4_ELTS, N)
                        blk = x_t[b, h, d_idx, blk_start:blk_end].float()
                        _quantise_block_fp4(
                            blk, packed_out, scale_out,
                            b, h, d_idx, blk_start, CVT_FP4_ELTS,
                            row_dim_is_D=True,
                        )
    else:
        for b in range(B):
            for h in range(H):
                for tok in range(N):
                    load_tok = tok
                    if permute:
                        # Apply K column permutation
                        local = tok % 32
                        load_tok = (tok // 32) * 32 \
                                   + (local // 8) * 2 \
                                   + ((local % 8) // 2) * 8 \
                                   + (local % 8) % 2
                        if load_tok >= N:
                            load_tok = tok
                    for blk_start in range(0, D, CVT_FP4_ELTS):
                        blk = x[b, h, load_tok, blk_start:blk_start + CVT_FP4_ELTS].float()
                        _quantise_block_fp4(
                            blk, packed_out, scale_out,
                            b, h, tok, blk_start, CVT_FP4_ELTS,
                        )


def _quantise_block_fp4(
    block: torch.Tensor,       # [16] fp32 — one microscaling block
    packed_out: torch.Tensor,
    scale_out: torch.Tensor,
    b: int, h: int,
    row: int,
    col_start: int,
    block_size: int = 16,
    row_dim_is_D: bool = False,
) -> None:
    """
    Quantise a single 16-element block to NVFP4 E2M1.

    Steps (matching CUDA kernel logic):
        1. max_abs = max(|block|)                              # hmax2 in CUDA
        2. sf = max_abs / 6.0                                   # FP4 E2M1 max magnitude = 6
        3. sf_fp8 = round_to_e4m3(sf)                           # fit into fp8 scale
        4. sf = dequant(sf_fp8)                                  # use quantised value for consistency
        5. for each element: fp4 = round_to_e2m1(x / sf)        # PTX cvt instruction
        6. pack pairs of fp4 into uint8 bytes
    """
    max_abs = block.abs().max().item()
    sf = max_abs / 6.0

    # Quantise sf to E4M3 and read back (matching CUDA: cast to fp8 then back)
    if sf > 0:
        sf_fp8 = _float_to_e4m3(sf)
        sf_q   = _e4m3_to_float(sf_fp8)
        inv    = 1.0 / sf_q
    else:
        sf_fp8 = 0
        sf_q   = 0.0
        inv    = 0.0

    # Quantise each element to E2M1 and pack
    scaled = block * inv if inv > 0 else block * 0
    fp4_vals = [_float_to_e2m1(v.item()) for v in scaled]

    # Pack 2 fp4 values per byte  (lo nibble = even index, hi nibble = odd)
    for i in range(0, len(fp4_vals), 2):
        lo = fp4_vals[i] & 0xF
        hi = fp4_vals[i + 1] & 0xF if i + 1 < len(fp4_vals) else 0
        byte_val = lo | (hi << 4)

        if row_dim_is_D:
            packed_out[b, h, row, col_start // 2 + i // 2] = byte_val
        else:
            packed_out[b, h, row, col_start // 2 + i // 2] = byte_val

    # Store scale factor
    scale_idx = col_start // 16
    if row_dim_is_D:
        scale_out[b, h, row, scale_idx] = sf_fp8
    else:
        scale_out[b, h, row, scale_idx] = sf_fp8


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Dequantisation helpers (for pseudocode emulation of FP4MM)
# ═══════════════════════════════════════════════════════════════════════════════

def _dequant_fp4(
    packed: torch.Tensor,   # [N, D//2]  uint8
    scales: torch.Tensor,   # [N, D//16] fp8_e4m3
) -> torch.Tensor:
    """Dequantise FP4 packed tensor → fp32 [N, D]."""
    N, D_half = packed.shape
    D = D_half * 2
    out = torch.zeros(N, D, dtype=torch.float32, device=packed.device)

    for i in range(N):
        for blk in range(0, D, 16):
            sf = float(scales[i, blk // 16])
            for k in range(16):
                j = blk + k
                byte_val = int(packed[i, j // 2])
                fp4_bits = byte_val & 0xF if j % 2 == 0 else (byte_val >> 4) & 0xF
                out[i, j] = _e2m1_to_float(fp4_bits) * sf
    return out


def _dequant_fp4_transposed(
    packed: torch.Tensor,      # [D, n_size//2] uint8
    scales: torch.Tensor,      # [D, n_size//16] fp8_e4m3
    n_size: int,
) -> torch.Tensor:
    """Dequantise transposed V back to [n_size, D] fp32."""
    D = packed.size(0)
    out = torch.zeros(n_size, D, dtype=torch.float32, device=packed.device)

    for d in range(D):
        for blk in range(0, n_size, 16):
            sf = float(scales[d, blk // 16])
            for k in range(16):
                j = blk + k
                if j >= n_size:
                    break
                byte_val = int(packed[d, j // 2])
                fp4_bits = byte_val & 0xF if j % 2 == 0 else (byte_val >> 4) & 0xF
                out[j, d] = _e2m1_to_float(fp4_bits) * sf
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 6. NVFP4 / FP8 numeric conversion helpers
# ═══════════════════════════════════════════════════════════════════════════════

# NVFP4 E2M1 format:
#   bit 3: sign,   bits 2-1: exponent (2 bits),   bit 0: mantissa (1 bit)
#
# Encoding table (positive values; negative values mirror with sign bit):
#   0b0000 → 0.0      0b0001 → 0.5
#   0b0010 → 1.0      0b0011 → 1.5
#   0b0100 → 2.0      0b0101 → 3.0
#   0b0110 → 4.0      0b0111 → 6.0

_E2M1_TABLE = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def _e2m1_to_float(bits: int) -> float:
    """NVFP4 E2M1 (4 bits) → float32."""
    sign = -1.0 if (bits & 0x8) else 1.0
    mag  = _E2M1_TABLE[bits & 0x7]
    return sign * mag


def _float_to_e2m1(val: float) -> int:
    """
    float32 → NVFP4 E2M1 (4 bits).

    Matches PTX: cvt.rn.satfinite.e2m1x2.f32
    Round-to-nearest with saturation to finite (no NaN/Inf in output).
    """
    sign_bit = 0x8 if val < 0 else 0
    a = abs(val)
    # Round to nearest representable magnitude
    # Boundaries: 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0
    if a < 0.25:
        mag = 0      # 0.0
    elif a < 0.75:
        mag = 1      # 0.5
    elif a < 1.25:
        mag = 2      # 1.0
    elif a < 1.75:
        mag = 3      # 1.5
    elif a < 2.5:
        mag = 4      # 2.0
    elif a < 3.5:
        mag = 5      # 3.0
    elif a < 5.0:
        mag = 6      # 4.0
    else:
        mag = 7      # 6.0  (satfinite — saturates here, no inf)
    return sign_bit | mag


def _float_to_e4m3(val: float) -> int:
    """
    float32 → FP8 E4M3 (8 bits).
    Simplified — in real kernel this is a hardware cast:
        reinterpret_cast<__nv_fp8_e4m3&>(out) = __nv_fp8_e4m3(val);
    """
    # E4M3 range: ~±448, minimal subnormal ~2^-9 ≈ 0.00195
    # This is a simplified emulation; the real hardware handles rounding,
    # NaN, overflow/underflow etc.
    return int(val * 1e6) & 0xFF  # placeholder — real impl uses hardware cast


def _e4m3_to_float(bits: int) -> float:
    """FP8 E4M3 → float32.  Placeholder for hardware dequant."""
    # In practice this is just reading back the fp8 value
    return bits / 1e6  # placeholder


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Online Softmax with Fused Quantisation
#    (mirrors softmax_fused.h :: SoftmaxFused::online_softmax_with_quant)
# ═══════════════════════════════════════════════════════════════════════════════

def online_softmax_with_quant_tile(
    S: torch.Tensor,                 # [m_size, n_size] fp32 — raw QK^T scores
    row_max: torch.Tensor,           # [m_size] fp32 — running max
    row_sum: torch.Tensor,           # [m_size] fp32 — running sum
    o_acc: torch.Tensor,             # [m_size, D] fp32 — accumulated output
    softmax_scale_log2: float,       # log2(1/√D)
    is_first_tile: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused online softmax + two-level P quantisation for one KV tile.

    In the real kernel (softmax_fused.h), this function simultaneously:
      1. Computes per-16-element absolute max (AbsMaxP) for P quantisation
      2. Performs warp-level reduction for row_max (reusing AbsMaxP data)
      3. Exponentiates and accumulates for online softmax
      4. Divides by s_P1 and prepares for NVFP4 microscaling

    Key constant:
        fp8_scale × fp4_scale = 1 / (448 × 6)
        log2(fp8_scale × fp4_scale) = -11.392317422778762

    The fused approach avoids redundant shuffle operations by reusing
    the max reduction computed for per-16-element quantisation scaling
    as part of the row-wise softmax max computation (~10% speedup).

    Returns:
        P_unnorm    [m_size, n_size] — unnormalised attention weights
        s_P1        [m_size, 1]      — Level-1 per-row scale (fp32)
        row_max     [m_size]         — updated running max
        row_sum     [m_size]         — updated running sum
        correction  [m_size]         — exp(old_max - new_max) for O rescaling
    """
    m_size, n_size = S.shape

    old_max = row_max.clone()

    # --- Per-16-element max (AbsMaxP) ---
    # In real kernel: computed per warp across 16 elements, then shuffled
    # via __shfl_xor_sync for row-level reduction
    abs_max_p = torch.zeros(m_size, (n_size + 15) // 16, dtype=torch.float32,
                             device=S.device)
    for i in range(m_size):
        for j in range(0, n_size, 16):
            blk_end = min(j + 16, n_size)
            abs_max_p[i, j // 16] = S[i, j:blk_end].max().item()

    # Row-level max (reuses per-16-element max — fused shuffle)
    tile_max = abs_max_p.max(dim=-1).values    # [m_size]
    row_max  = torch.maximum(row_max, tile_max)

    # --- Exponentiate with base-2 (hardware ptx_exp2) ---
    # Real kernel:  acc = ptx_exp2(acc * scale_log2 - max_scaled)
    #   where max_scaled = row_max * scale_log2 + log2(1/(448*6))
    # This simultaneously applies the softmax scale and Level-1 P scale.
    FP8xFP4_SCALE_LOG2 = -11.392317422778762
    max_scaled = row_max * softmax_scale_log2 + FP8xFP4_SCALE_LOG2

    P_unnorm = torch.exp2(S * softmax_scale_log2 - max_scaled.unsqueeze(-1))

    # --- Also exponentiate AbsMaxP for Level-2 scale factors ---
    FP4_SCALE_LOG2 = -2.584962500721156  # log2(1/6)
    abs_max_p_exp = torch.exp2(
        abs_max_p * softmax_scale_log2 - max_scaled.unsqueeze(-1) + FP4_SCALE_LOG2
    )
    # abs_max_p_exp now contains the Level-2 FP8 scale factors (per 16 elements)

    # --- Rescale previous accumulators ---
    if not is_first_tile:
        correction = torch.exp2((old_max - row_max) * softmax_scale_log2)
    else:
        correction = torch.ones(m_size, dtype=torch.float32, device=S.device)
    row_sum = row_sum * correction

    # --- Accumulate row sums ---
    row_sum = row_sum + P_unnorm.sum(dim=-1)

    # --- Quantise P to FP4 using Level-2 scales ---
    # In real kernel: each thread converts its 8 float values to e2m1
    # using packed_float_to_e2m1, and packs 4 FP8 scales using packed_float_to_ue4m3
    # For pseudocode we keep P_unnorm in fp32 and just track the scale.

    # Level-1 scale:  s_P1 is implicitly  448 * 6 * exp2(max_scaled)
    # (baked into the exponentiation above)
    # The final O_tile = FP4MM(P_fp4, s_P2, V_fp4, s_V) already includes s_P1
    # because P_unnorm was scaled by 1/(448*6) during exponentiation.

    s_P1 = None  # implicit in the max_scaled formulation

    return P_unnorm, abs_max_p_exp, row_max, row_sum, correction


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Summary of kernel data flow
# ═══════════════════════════════════════════════════════════════════════════════
#
#  ┌─────────────────────────────────────────────────────────────────────────┐
#  │                        HOST (Python / API)                             │
#  │                                                                        │
#  │  q, k, v  [B,H,L,D] fp16/bf16                                         │
#  │     │                                                                  │
#  │     ├── k -= mean(k, dim=seq)           # smooth K                     │
#  │     ├── pad to multiple of 128          # required for FP4MM tiles     │
#  │     ├── q, qm = group_mean(q, 128)     # smooth Q per block           │
#  │     └── delta_s = qm @ k^T             # correction [B,H,G,L_pad]     │
#  │                                                                        │
#  │  ┌── scale_and_quant_fp4(q)         → (q_packed, q_sf)                 │
#  │  ├── scale_and_quant_fp4_permute(k) → (k_packed, k_sf)   # K permuted │
#  │  └── scale_and_quant_fp4_trans(v)   → (v_packed, v_sf)   # V transposed│
#  │                                                                        │
#  │  fp4attn_cuda.fwd(q_packed, k_packed, v_packed,                        │
#  │                   q_sf, k_sf, v_sf, delta_s, ...)                      │
#  └────────────────────────────────────────┬────────────────────────────────┘
#                                           │
#  ┌────────────────────────────────────────▼────────────────────────────────┐
#  │                    DEVICE (CUTLASS kernel)                              │
#  │                                                                        │
#  │  Grid: (num_m_tiles, B, H)   or persistent tile scheduler              │
#  │  Warp groups: Producer (TMA load/store), Consumer (MMA + softmax)      │
#  │                                                                        │
#  │  For each m_block:                                                     │
#  │    TMA load Q tile [128, D] + SF_Q                                     │
#  │    Copy Q from SMEM → registers                                        │
#  │                                                                        │
#  │    For each n_block (KV tiles):                                        │
#  │      TMA load K tile [128, D] + SF_K + delta_s [128, 128]              │
#  │      TMA load V tile [D, 128] + SF_V                                   │
#  │                                                                        │
#  │      ┌─ GEMM-I: S = FP4MM(Q, SF_Q, K, SF_K)  [128×128] ──────────┐   │
#  │      │  Add delta_s correction                                     │   │
#  │      │  Apply causal/padding mask                                  │   │
#  │      └─────────────────────────────────────────────────────────────┘   │
#  │                                                                        │
#  │      ┌─ Fused softmax + P quantisation ────────────────────────────┐   │
#  │      │  Per-16-elem max → row max (fused shuffle reuse)            │   │
#  │      │  P̃ = exp2(S * scale_log2 - max_scaled)                     │   │
#  │      │  Level-1: s_P1 implicit in max_scaled (×448×6)              │   │
#  │      │  Level-2: s_P2 = per-16-elem max → fp8 E4M3                │   │
#  │      │  P̂ = cvt.e2m1(P̃ / s_P2_per_block)                         │   │
#  │      └─────────────────────────────────────────────────────────────┘   │
#  │                                                                        │
#  │      ┌─ GEMM-II: O = FP4MM(P̂, SF_P2, V, SF_V)  [128×D] ─────────┐   │
#  │      │  Online softmax rescale: O_acc = O_acc * correction + O      │   │
#  │      └─────────────────────────────────────────────────────────────┘   │
#  │                                                                        │
#  │    Final: O = O_acc / row_sum                                          │
#  │    TMA store O tile [128, D] → global memory                           │
#  └────────────────────────────────────────────────────────────────────────┘


if __name__ == "__main__":
    print("SageAttention3 Pseudocode v2")
    print("=" * 60)
    print()
    print("This file documents the algorithmic flow of SageAttention3")
    print("matching the actual CUDA/CUTLASS kernel implementation.")
    print()
    print("Key files in the codebase this pseudocode corresponds to:")
    print("  sageattn3/api.py                – Python API & preprocessing")
    print("  sageattn3/quantization/*.cu     – FP4 quantisation kernels")
    print("  sageattn3/blackwell/kernel_ws.h – Top-level kernel launch")
    print("  sageattn3/blackwell/mainloop_tma_ws.h – Main MMA loop")
    print("  sageattn3/blackwell/softmax_fused.h   – Fused softmax + quant")
    print("  sageattn3/blackwell/kernel_traits.h   – Tile sizes & MMA config")
