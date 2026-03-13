#!/usr/bin/env python3
"""
Debug E2E Demo for single_level_p_quantization_mxfp8_triton
============================================================

This script isolates the MXFP8 single-level P quantization kernel and tests it
independently — both as a standalone Triton kernel (via a thin wrapper) and as
a pure-PyTorch reference, so you can compare outputs element-by-element.

Usage:
    python debug_mxfp8_p_quant.py                # default
    python debug_mxfp8_p_quant.py --seed 123      # custom seed
    python debug_mxfp8_p_quant.py --verbose        # print full tensors
    python debug_mxfp8_p_quant.py --softmax        # use realistic softmax-like input
"""

import argparse
import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Import the kernel-internal helpers from the standalone module
# ---------------------------------------------------------------------------
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sageattention3_standalone import (
    round_to_e8m0_torch,
    apply_e4m3_quantization_torch,
    single_level_p_quantization_mxfp8_triton,
    apply_e4m3_quantization_triton,
    round_to_e8m0_triton,
)


# ============================================================================
# 1) Thin Triton wrapper: run the JIT kernel on a [128, 128] tile
# ============================================================================

@triton.jit
def _p_quant_mxfp8_wrapper_kernel(
    P_ptr,          # [128, 128] input
    Out_ptr,        # [128, 128] output
    stride_p_row, stride_p_col,
    stride_o_row, stride_o_col,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Launch single_level_p_quantization_mxfp8_triton on one tile."""
    row_offs = tl.arange(0, BLOCK_M)
    col_offs = tl.arange(0, BLOCK_N)

    p_ptrs = P_ptr + row_offs[:, None] * stride_p_row + col_offs[None, :] * stride_p_col
    p_tile = tl.load(p_ptrs).to(tl.float32)

    p_out = single_level_p_quantization_mxfp8_triton(p_tile, BLOCK_N)

    o_ptrs = Out_ptr + row_offs[:, None] * stride_o_row + col_offs[None, :] * stride_o_col
    tl.store(o_ptrs, p_out.to(tl.float32))


def run_triton_p_quant(p_tile: torch.Tensor) -> torch.Tensor:
    """Run the Triton MXFP8 P-quant kernel on a [128, 128] FP32 tile."""
    assert p_tile.shape == (128, 128), f"Expected [128, 128], got {p_tile.shape}"
    p_tile = p_tile.contiguous().cuda().float()
    out = torch.empty_like(p_tile)
    _p_quant_mxfp8_wrapper_kernel[(1,)](
        p_tile, out,
        p_tile.stride(0), p_tile.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=128, BLOCK_N=128,
    )
    return out


# ============================================================================
# 2) Pure-PyTorch reference implementation (for comparison)
# ============================================================================

def single_level_p_quantization_mxfp8_pytorch(p_tile: torch.Tensor) -> torch.Tensor:
    """
    Pure-PyTorch reference matching single_level_p_quantization_mxfp8_triton.

    p_tile: [128, 128] float32 tensor (attention probabilities for one tile).
    Returns: [128, 128] quantized+reconstructed tensor.
    """
    FP8_MAX_VAL = 448.0
    BLOCK_SIZE = 32
    M, N = p_tile.shape
    assert N == 128

    # Reshape into 4 column blocks of 32
    # [128, 128] -> [128, 4, 32]
    p_blocks = p_tile.view(M, N // BLOCK_SIZE, BLOCK_SIZE)

    # Per-row, per-block max: [128, 4]
    block_max = p_blocks.abs().max(dim=-1)[0]

    # Microscale = ceil_e8m0(block_max / FP8_MAX_VAL)
    microscales = torch.clamp(block_max / FP8_MAX_VAL, min=1e-8)
    microscales_e8m0 = round_to_e8m0_torch(microscales)  # [128, 4]

    # Normalize, quantize E4M3, reconstruct
    p_normalized = p_blocks / microscales_e8m0.unsqueeze(-1)  # [128, 4, 32]
    p_quantized = apply_e4m3_quantization_torch(p_normalized)
    p_reconstructed = p_quantized * microscales_e8m0.unsqueeze(-1)

    return p_reconstructed.view(M, N)


# ============================================================================
# 3) E2E attention test: full MXFP8_S1 attention vs SDPA
# ============================================================================

def run_e2e_attention_test(seed: int = 42, verbose: bool = False):
    """Run full attention with mxfp8_s1 and compare to SDPA."""
    from sageattention3_standalone import scaled_dot_product_attention
    import torch.nn.functional as F

    torch.manual_seed(seed)
    B, H, N, D = 1, 4, 128, 64
    q = torch.randn(B, H, N, D, dtype=torch.float16, device='cuda')
    k = torch.randn(B, H, N, D, dtype=torch.float16, device='cuda')
    v = torch.randn(B, H, N, D, dtype=torch.float16, device='cuda')

    with torch.no_grad():
        ref = F.scaled_dot_product_attention(q, k, v)
        out = scaled_dot_product_attention(q, k, v, quant_format="mxfp8_s1")

    ref_flat = ref.flatten().float()
    out_flat = out.flatten().float()
    cos_sim = F.cosine_similarity(ref_flat.unsqueeze(0), out_flat.unsqueeze(0)).item()
    l2_err = (torch.norm(ref_flat - out_flat) / torch.norm(ref_flat)).item()
    max_err = torch.max(torch.abs(ref_flat - out_flat)).item()

    print(f"\n{'='*60}")
    print(f"E2E Attention: MXFP8_S1 vs SDPA  (B={B}, H={H}, N={N}, D={D})")
    print(f"{'='*60}")
    print(f"  Cosine similarity : {cos_sim:.6f}")
    print(f"  L2 relative error : {l2_err:.6f}")
    print(f"  Max absolute error: {max_err:.6f}")
    print(f"  Result: {'✅ PASS' if cos_sim > 0.85 else '❌ FAIL'}")

    if verbose:
        print(f"\n  SDPA output [0,0,:4,:4]:\n{ref[0,0,:4,:4]}")
        print(f"  MXFP8 output [0,0,:4,:4]:\n{out[0,0,:4,:4]}")

    return cos_sim, l2_err


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Debug MXFP8 P-quantization kernel")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true", help="Print full tensor slices")
    parser.add_argument("--softmax", action="store_true",
                        help="Use softmax-like input (non-negative, row-sums-to-1)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda"

    # ---- Generate input tile ----
    if args.softmax:
        # Realistic: softmax output (non-negative, rows sum to 1)
        logits = torch.randn(128, 128, device=device, dtype=torch.float32)
        p_tile = torch.softmax(logits, dim=-1)
        print("Input: softmax probabilities (non-negative, rows sum to ~1)")
    else:
        # Raw randn (tests full signed range)
        p_tile = torch.randn(128, 128, device=device, dtype=torch.float32)
        print("Input: random Gaussian (signed values)")

    # ---- Run both implementations ----
    triton_out = run_triton_p_quant(p_tile)
    pytorch_out = single_level_p_quantization_mxfp8_pytorch(p_tile)

    # ---- Compare ----
    diff = (triton_out - pytorch_out).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    flat_t = triton_out.flatten()
    flat_p = pytorch_out.flatten()
    cos_sim = torch.nn.functional.cosine_similarity(
        flat_t.unsqueeze(0), flat_p.unsqueeze(0)
    ).item()

    print(f"\n{'='*60}")
    print(f"P-Quantization Kernel: Triton vs PyTorch Reference")
    print(f"{'='*60}")
    print(f"  Input range      : [{p_tile.min().item():.4f}, {p_tile.max().item():.4f}]")
    print(f"  Triton out range : [{triton_out.min().item():.4f}, {triton_out.max().item():.4f}]")
    print(f"  PyTorch out range: [{pytorch_out.min().item():.4f}, {pytorch_out.max().item():.4f}]")
    print(f"  Max element diff : {max_diff:.2e}")
    print(f"  Mean element diff: {mean_diff:.2e}")
    print(f"  Cosine similarity: {cos_sim:.8f}")

    # ---- Per-block breakdown ----
    print(f"\n  Per-block (32-col) breakdown:")
    for b in range(4):
        col_start, col_end = b * 32, (b + 1) * 32
        block_diff = diff[:, col_start:col_end]
        block_t = triton_out[:, col_start:col_end]
        block_p = pytorch_out[:, col_start:col_end]
        print(f"    Block {b} (cols {col_start}-{col_end-1}): "
              f"max_diff={block_diff.max().item():.2e}, "
              f"mean_diff={block_diff.mean().item():.2e}, "
              f"triton_range=[{block_t.min().item():.4f}, {block_t.max().item():.4f}], "
              f"pytorch_range=[{block_p.min().item():.4f}, {block_p.max().item():.4f}]")

    # ---- Quantization error vs original ----
    quant_err_triton = (triton_out - p_tile).abs()
    quant_err_pytorch = (pytorch_out - p_tile).abs()
    print(f"\n  Quantization error (vs original input):")
    print(f"    Triton  — max: {quant_err_triton.max().item():.4e}, "
          f"mean: {quant_err_triton.mean().item():.4e}")
    print(f"    PyTorch — max: {quant_err_pytorch.max().item():.4e}, "
          f"mean: {quant_err_pytorch.mean().item():.4e}")

    match = max_diff < 1e-5
    print(f"\n  Triton == PyTorch: {'✅ MATCH' if match else '❌ MISMATCH'}")

    if args.verbose or not match:
        # Show first few rows where differences are largest
        row_max_diff = diff.max(dim=1)[0]
        worst_rows = row_max_diff.topk(min(5, 128)).indices
        print(f"\n  Worst rows (by max diff): {worst_rows.tolist()}")
        for row_idx in worst_rows:
            r = row_idx.item()
            print(f"\n    Row {r}:")
            print(f"      Input  : {p_tile[r, :8].tolist()}")
            print(f"      Triton : {triton_out[r, :8].tolist()}")
            print(f"      PyTorch: {pytorch_out[r, :8].tolist()}")
            print(f"      Diff   : {diff[r, :8].tolist()}")

    # ---- Full E2E attention test ----
    run_e2e_attention_test(seed=args.seed, verbose=args.verbose)

    print()


if __name__ == "__main__":
    main()
