"""
Tiled online attention Triton kernel with zero-dispatch P-quantization.

The kernel accepts P_QUANT_FN as a tl.constexpr parameter — Triton inlines
the function at compile time. Each unique P-quant function produces a
separately compiled kernel with no dispatch overhead.
"""

import torch
import triton
import triton.language as tl
from typing import Optional


@triton.jit
def tiled_online_attention_kernel(
    # Input pointers
    Q_ptr, K_ptr, V_ptr,
    Delta_s_ptr,  # Optional delta_s correction
    Out_ptr,

    # Tensor strides
    stride_q_b, stride_q_h, stride_q_n, stride_q_d,
    stride_k_b, stride_k_h, stride_k_n, stride_k_d,
    stride_v_b, stride_v_h, stride_v_n, stride_v_d,
    stride_delta_b, stride_delta_h, stride_delta_g, stride_delta_n,
    stride_o_b, stride_o_h, stride_o_n, stride_o_d,

    # Dimensions
    B, H, N, D, num_groups,

    # Parameters
    sm_scale,
    is_causal: tl.constexpr,
    has_delta_s: tl.constexpr,

    # Zero-dispatch P-quantization function
    P_QUANT_FN: tl.constexpr,

    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """
    Tiled online attention kernel with all SageAttention3 features.

    Processes one query tile at a time, iterating through all key/value tiles
    using the online softmax algorithm. P-quantization is performed by the
    P_QUANT_FN constexpr function, inlined at compile time.
    """
    # Program IDs
    pid_b = tl.program_id(0)  # Batch
    pid_h = tl.program_id(1)  # Head
    pid_m = tl.program_id(2)  # Query tile

    # Calculate query tile boundaries
    q_start = pid_m * BLOCK_M
    q_end = tl.minimum(q_start + BLOCK_M, N)
    actual_block_m = q_end - q_start

    # Early exit if out of bounds
    if pid_b >= B or pid_h >= H or q_start >= N:
        return

    # Query tile indices
    offs_m = q_start + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    # Load query tile
    q_ptrs = (Q_ptr +
              pid_b * stride_q_b +
              pid_h * stride_q_h +
              offs_m[:, None] * stride_q_n +
              offs_d[None, :] * stride_q_d)

    q_mask = (offs_m[:, None] < N) & (offs_d[None, :] < D)
    q_tile = tl.load(q_ptrs, mask=q_mask, other=0.0)

    q_tile = q_tile.to(tl.float32)

    # Initialize running statistics for online softmax
    running_max = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    running_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
    output_tile = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Number of K/V tiles
    num_k_tiles = tl.cdiv(N, BLOCK_N)

    # Iterate through K/V tiles
    for k_idx in range(num_k_tiles):
        k_start = k_idx * BLOCK_N
        k_end = tl.minimum(k_start + BLOCK_N, N)

        # Process tile only if not empty
        if k_start < N:
            # Key/Value tile indices
            offs_n = k_start + tl.arange(0, BLOCK_N)

            # Load key tile
            k_ptrs = (K_ptr +
                      pid_b * stride_k_b +
                      pid_h * stride_k_h +
                      offs_n[None, :] * stride_k_n +
                      offs_d[:, None] * stride_k_d)

            k_mask = (offs_n[None, :] < N) & (offs_d[:, None] < D)
            k_tile = tl.load(k_ptrs, mask=k_mask, other=0.0).to(tl.float32)

            # Load value tile
            v_ptrs = (V_ptr +
                      pid_b * stride_v_b +
                      pid_h * stride_v_h +
                      offs_n[:, None] * stride_v_n +
                      offs_d[None, :] * stride_v_d)

            v_mask = (offs_n[:, None] < N) & (offs_d[None, :] < D)
            v_tile = tl.load(v_ptrs, mask=v_mask, other=0.0).to(tl.float32)

            # Compute QK^T (do NOT apply sm_scale yet — delta_s must be added first)
            qk_tile = tl.dot(q_tile, k_tile, out_dtype=tl.float32)

            # Add delta_s correction if provided
            if has_delta_s:
                # GROUP_SIZE must match BLOCK_M (tile_size_q) — both are 128.
                # group_id = q_tile_index = pid_m
                GROUP_SIZE: tl.constexpr = 128
                group_id = (q_start) // GROUP_SIZE if num_groups > 1 else 0
                group_id = tl.minimum(group_id, num_groups - 1)

                ds_ptrs = (Delta_s_ptr +
                           pid_b * stride_delta_b +
                           pid_h * stride_delta_h +
                           group_id * stride_delta_g +
                           offs_n * stride_delta_n)

                ds_mask = offs_n < N
                ds_tile = tl.load(ds_ptrs, mask=ds_mask, other=0.0).to(tl.float32)

                # Add correction BEFORE scaling
                ds_broadcasted = ds_tile[None, :]
                qk_tile = qk_tile + ds_broadcasted

            # Apply sm_scale AFTER adding delta_s
            qk_tile = qk_tile * sm_scale

            # Apply causal mask
            if is_causal:
                causal_mask = offs_m[:, None] >= offs_n[None, :]
                qk_tile = tl.where(causal_mask, qk_tile, float('-inf'))

            # Apply bounds mask
            bounds_mask = (offs_m[:, None] < N) & (offs_n[None, :] < N)
            qk_tile = tl.where(bounds_mask, qk_tile, float('-inf'))

            # Online softmax update
            tile_max = tl.max(qk_tile, axis=1)
            old_max = running_max
            new_max = tl.maximum(running_max, tile_max)

            # Renormalization factor
            alpha = tl.exp(old_max - new_max)

            # Update output with renormalization
            output_tile = output_tile * alpha[:, None]

            # Compute probabilities
            qk_shifted = qk_tile - new_max[:, None]
            p_tile = tl.exp(qk_shifted)

            # ★ Zero-dispatch P quantization — inlined at compile time ★
            p_quantized = P_QUANT_FN(p_tile, BLOCK_N)

            # PV computation
            pv_tile = tl.dot(p_quantized, v_tile, out_dtype=tl.float32)

            # Update running statistics.
            #
            # Denominator uses the pre-P-quant softmax tile to match the CUTE/
            # Blackwell kernel semantics. The numerator path (PV) still uses the
            # quantized-and-reconstructed probabilities.
            tile_sum = tl.sum(p_tile, axis=1)
            running_sum = running_sum * alpha + tile_sum
            running_max = new_max

            # Accumulate output
            output_tile = output_tile + pv_tile

    # Final normalization
    output_tile = output_tile / (running_sum[:, None])

    # Store output
    out_ptrs = (Out_ptr +
                pid_b * stride_o_b +
                pid_h * stride_o_h +
                offs_m[:, None] * stride_o_n +
                offs_d[None, :] * stride_o_d)

    out_mask = (offs_m[:, None] < N) & (offs_d[None, :] < D)
    # tl.store auto-casts float32 output_tile to the output tensor's dtype (bf16/fp16)
    tl.store(out_ptrs, output_tile, mask=out_mask)


# ============================================================================
# Kernel launcher
# ============================================================================

def launch_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    delta_s: Optional[torch.Tensor],
    sm_scale: float,
    is_causal: bool,
    p_quant_fn: triton.JITFunction,
    tile_size_q: int = 128,
    tile_size_k: int = 128,
) -> torch.Tensor:
    """
    Launch the tiled online attention kernel.

    Args:
        q: Query tensor [B, H, N, D]
        k: Key tensor [B, H, N, D]
        v: Value tensor [B, H, N, D]
        delta_s: QK correction [B, H, num_groups, N] (optional)
        sm_scale: Softmax scaling factor
        is_causal: Apply causal masking
        p_quant_fn: @triton.jit P-quant function (passed as constexpr)
        tile_size_q: Query tile size (BLOCK_M)
        tile_size_k: Key/Value tile size (BLOCK_N)

    Returns:
        output: Attention output [B, H, N, D]
    """
    B, H, N, D = q.shape

    # P-quant kernels hardcode block boundaries for BLOCK_N=128 (e.g., cols 0-16,
    # 16-32, ..., 112-128). Non-128 tile sizes produce silently wrong results.
    assert tile_size_k == 128, (
        f"P-quant kernels require tile_size_k=128, got {tile_size_k}. "
        "Block boundaries are hardcoded in P-quant functions."
    )

    output = torch.zeros_like(q)

    if delta_s is not None:
        num_groups = delta_s.shape[2]
        has_delta_s = True
    else:
        num_groups = 1
        has_delta_s = False
        delta_s = torch.zeros(B, H, 1, N, device=q.device, dtype=q.dtype)

    num_q_tiles = (N + tile_size_q - 1) // tile_size_q
    grid = (B, H, num_q_tiles)

    tiled_online_attention_kernel[grid](
        q, k, v, delta_s, output,
        # Q strides
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        # K strides
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        # V strides
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        # Delta_s strides
        delta_s.stride(0), delta_s.stride(1), delta_s.stride(2), delta_s.stride(3),
        # Output strides
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        # Dimensions
        B, H, N, D, num_groups,
        # Parameters
        sm_scale,
        is_causal,
        has_delta_s,
        # Zero-dispatch P-quant function
        P_QUANT_FN=p_quant_fn,
        # Block sizes
        BLOCK_M=tile_size_q,
        BLOCK_N=tile_size_k,
        HEAD_DIM=triton.next_power_of_2(D) if D <= 256 else D,
    )

    return output
