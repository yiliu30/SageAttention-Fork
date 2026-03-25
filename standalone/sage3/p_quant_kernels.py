"""
P-quantization Triton kernels for in-kernel attention probability quantization.

Each kernel self-registers via @register_p_quant and composes shared primitives
from quant_primitives.py, reducing the per-scheme boilerplate to the unique logic:
scale rounding method, data quantization, and single-vs-two-level factorization.

All functions have the signature: (p_tile, BLOCK_N: tl.constexpr) -> p_quantized
"""

import triton.language as tl

from .p_quant_registry import register_p_quant
from .quant_primitives import (
    apply_e2m1_quantization_triton,
    apply_e4m3_quantization_triton,
    round_to_e4m3_triton,
    round_to_e8m0_triton,
    compute_block_max,
    build_microscale_tensor_4,
    build_microscale_tensor_8,
)


# ============================================================================
# NVFP4: Two-level, E4M3 scales, block_size=16
# ============================================================================

@register_p_quant("nvfp4")
def p_quant_nvfp4(p_tile, BLOCK_N: tl.constexpr):
    """
    Two-level P quantization with per-row global + per-16-col-block microscaling.

    Level 1: Per-row global scale (row_max / COMBINED_MAX), not format-rounded
    Level 2: Per-row, per-16-col-block microscale with E4M3 rounding

    For 128x128 tiles: 8 blocks of 16 columns each.
    """
    FP4_MAX: tl.constexpr = 6.0
    FP8_MAX: tl.constexpr = 448.0
    MICROSCALE_BLOCK_SIZE: tl.constexpr = 16
    COMBINED_MAX = FP8_MAX * FP4_MAX  # 2688

    # Level 1: Global per-row FP32 scaling
    row_max = tl.max(tl.abs(p_tile), axis=1)
    global_scales = tl.maximum(row_max / COMBINED_MAX, 1e-8)
    p_level1 = p_tile / global_scales[:, None]

    col_indices = tl.arange(0, BLOCK_N)
    block_ids = col_indices // MICROSCALE_BLOCK_SIZE

    # Level 2: Per-row, per-block microscaling (8 blocks of 16)
    b0 = compute_block_max(p_level1, col_indices, 0, 16)
    b1 = compute_block_max(p_level1, col_indices, 16, 32)
    b2 = compute_block_max(p_level1, col_indices, 32, 48)
    b3 = compute_block_max(p_level1, col_indices, 48, 64)
    b4 = compute_block_max(p_level1, col_indices, 64, 80)
    b5 = compute_block_max(p_level1, col_indices, 80, 96)
    b6 = compute_block_max(p_level1, col_indices, 96, 112)
    b7 = compute_block_max(p_level1, col_indices, 112, 128)

    # E4M3-rounded microscales
    s0 = round_to_e4m3_triton(tl.maximum(b0 / FP4_MAX, 1e-8))
    s1 = round_to_e4m3_triton(tl.maximum(b1 / FP4_MAX, 1e-8))
    s2 = round_to_e4m3_triton(tl.maximum(b2 / FP4_MAX, 1e-8))
    s3 = round_to_e4m3_triton(tl.maximum(b3 / FP4_MAX, 1e-8))
    s4 = round_to_e4m3_triton(tl.maximum(b4 / FP4_MAX, 1e-8))
    s5 = round_to_e4m3_triton(tl.maximum(b5 / FP4_MAX, 1e-8))
    s6 = round_to_e4m3_triton(tl.maximum(b6 / FP4_MAX, 1e-8))
    s7 = round_to_e4m3_triton(tl.maximum(b7 / FP4_MAX, 1e-8))

    microscale_final = build_microscale_tensor_8(s0, s1, s2, s3, s4, s5, s6, s7, block_ids)

    # Quantize and reconstruct with both scale levels
    p_microscaled = p_level1 / microscale_final
    p_quantized = apply_e2m1_quantization_triton(p_microscaled)
    return p_quantized * microscale_final * global_scales[:, None]


# ============================================================================
# MXFP4: Two-level, E8M0 scales, block_size=32
# ============================================================================

@register_p_quant("mxfp4")
def p_quant_mxfp4(p_tile, BLOCK_N: tl.constexpr):
    """
    Two-level P quantization with MXFP4 microscaling (block_size=32, E8M0 scales).

    Level 1: Per-row global scale (row_max / COMBINED_MAX), not format-rounded
    Level 2: Per-row, per-32-col-block microscale with E8M0 rounding

    For 128x128 tiles: 4 blocks of 32 columns each.
    """
    FP4_MAX: tl.constexpr = 6.0
    FP8_MAX: tl.constexpr = 448.0
    MICROSCALE_BLOCK_SIZE: tl.constexpr = 32
    COMBINED_MAX = FP8_MAX * FP4_MAX  # 2688

    # Level 1: Global per-row FP32 scaling
    row_max = tl.max(tl.abs(p_tile), axis=1)
    global_scales = tl.maximum(row_max / COMBINED_MAX, 1e-8)
    p_level1 = p_tile / global_scales[:, None]

    col_indices = tl.arange(0, BLOCK_N)
    block_ids = col_indices // MICROSCALE_BLOCK_SIZE

    # Level 2: Per-row, per-block microscaling (4 blocks of 32)
    b0 = compute_block_max(p_level1, col_indices, 0, 32)
    b1 = compute_block_max(p_level1, col_indices, 32, 64)
    b2 = compute_block_max(p_level1, col_indices, 64, 96)
    b3 = compute_block_max(p_level1, col_indices, 96, 128)

    # E8M0-rounded microscales
    s0 = round_to_e8m0_triton(tl.maximum(b0 / FP4_MAX, 1e-8))
    s1 = round_to_e8m0_triton(tl.maximum(b1 / FP4_MAX, 1e-8))
    s2 = round_to_e8m0_triton(tl.maximum(b2 / FP4_MAX, 1e-8))
    s3 = round_to_e8m0_triton(tl.maximum(b3 / FP4_MAX, 1e-8))

    microscale_final = build_microscale_tensor_4(s0, s1, s2, s3, block_ids)

    # Quantize and reconstruct with both scale levels
    p_microscaled = p_level1 / microscale_final
    p_quantized = apply_e2m1_quantization_triton(p_microscaled)
    return p_quantized * microscale_final * global_scales[:, None]


# ============================================================================
# MXFP4_S1: Single-level, E8M0 scales, block_size=32
# ============================================================================

@register_p_quant("mxfp4_s1")
def p_quant_mxfp4_s1(p_tile, BLOCK_N: tl.constexpr):
    """
    Single-level P quantization with per-block E8M0 ceil microscaling (MXFP4).

    No global FP32 per-row scale — only per-block E8M0 microscaling.
    This avoids the information loss from two-level global→local factorization.

    For 128x128 tiles: 4 blocks of 32 columns each.
    """
    FP4_MAX: tl.constexpr = 6.0
    MICROSCALE_BLOCK_SIZE: tl.constexpr = 32

    col_indices = tl.arange(0, BLOCK_N)
    block_ids = col_indices // MICROSCALE_BLOCK_SIZE

    # Single-level: per-block microscaling directly on p_tile
    b0 = compute_block_max(p_tile, col_indices, 0, 32)
    b1 = compute_block_max(p_tile, col_indices, 32, 64)
    b2 = compute_block_max(p_tile, col_indices, 64, 96)
    b3 = compute_block_max(p_tile, col_indices, 96, 128)

    # E8M0-rounded microscales
    s0 = round_to_e8m0_triton(tl.maximum(b0 / FP4_MAX, 1e-8))
    s1 = round_to_e8m0_triton(tl.maximum(b1 / FP4_MAX, 1e-8))
    s2 = round_to_e8m0_triton(tl.maximum(b2 / FP4_MAX, 1e-8))
    s3 = round_to_e8m0_triton(tl.maximum(b3 / FP4_MAX, 1e-8))

    microscale_final = build_microscale_tensor_4(s0, s1, s2, s3, block_ids)

    # Quantize and reconstruct (single level — no global_scales)
    p_microscaled = p_tile / microscale_final
    p_quantized = apply_e2m1_quantization_triton(p_microscaled)
    return p_quantized * microscale_final


# ============================================================================
# MXFP8_S1: Single-level, E8M0 scales, FP8 data, block_size=32
# ============================================================================

@register_p_quant("mxfp8_s1")
def p_quant_mxfp8_s1(p_tile, BLOCK_N: tl.constexpr):
    """
    Single-level P quantization with per-block E8M0 ceil microscaling (MXFP8).

    Like MXFP4_S1 but uses E4M3 data quantization (fp_max=448) instead of
    E2M1 (fp_max=6). No global FP32 per-row scale.

    For 128x128 tiles: 4 blocks of 32 columns each.
    """
    FP8_MAX_VAL: tl.constexpr = 448.0
    MICROSCALE_BLOCK_SIZE: tl.constexpr = 32

    col_indices = tl.arange(0, BLOCK_N)
    block_ids = col_indices // MICROSCALE_BLOCK_SIZE

    # Single-level: per-block microscaling directly on p_tile
    b0 = compute_block_max(p_tile, col_indices, 0, 32)
    b1 = compute_block_max(p_tile, col_indices, 32, 64)
    b2 = compute_block_max(p_tile, col_indices, 64, 96)
    b3 = compute_block_max(p_tile, col_indices, 96, 128)

    # E8M0-rounded microscales (using FP8 max instead of FP4 max)
    s0 = round_to_e8m0_triton(tl.maximum(b0 / FP8_MAX_VAL, 1e-8))
    s1 = round_to_e8m0_triton(tl.maximum(b1 / FP8_MAX_VAL, 1e-8))
    s2 = round_to_e8m0_triton(tl.maximum(b2 / FP8_MAX_VAL, 1e-8))
    s3 = round_to_e8m0_triton(tl.maximum(b3 / FP8_MAX_VAL, 1e-8))

    microscale_final = build_microscale_tensor_4(s0, s1, s2, s3, block_ids)

    # E4M3 data quantization (not E2M1) and reconstruct
    p_microscaled = p_tile / microscale_final
    p_quantized = apply_e4m3_quantization_triton(p_microscaled)
    return p_quantized * microscale_final
