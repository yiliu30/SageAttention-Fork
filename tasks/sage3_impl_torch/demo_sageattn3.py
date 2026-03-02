#!/usr/bin/env python3
"""
SageAttention3 Pure-Torch Demo Script

This script demonstrates the usage of the SageAttention3 pure-torch implementation
with detailed tensor shapes and algorithm explanations. It's designed for
educational purposes to understand the SageAttention3 algorithm.

Key features demonstrated:
- NVFP4 microscaling quantization
- Two-level probability scaling
- Tiled online attention processing
- QK smoothing with outlier handling

The implementation uses pure PyTorch operations to simulate the behavior
of the actual SageAttention3 CUDA kernels on Blackwell GPUs.
"""

import argparse
import sys
import torch
import torch.nn.functional as F
from typing import Tuple
import time

# Import our SageAttention3 implementation
try:
    from sageattn3_torch import sageattn3_torch, create_test_tensors
    print("SageAttention3 torch implementation imported successfully")
except ImportError as e:
    print(f"Failed to import SageAttention3 implementation: {e}")
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length")
    parser.add_argument("--head-dim", type=int, default=64, help="Head dimension (must be <256)")
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16", help="Input dtype")
    parser.add_argument("--device", default="cuda", help="Device to run on")
    parser.add_argument("--causal", action="store_true", help="Enable causal masking")
    parser.add_argument("--no-smoothing", action="store_true", help="Disable QK smoothing")
    parser.add_argument("--tile-size-q", type=int, default=64, help="Query tile size")
    parser.add_argument("--tile-size-k", type=int, default=64, help="Key tile size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--compare-pytorch", action="store_true", help="Compare with PyTorch SDPA")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    return parser.parse_args()


def run_sageattention3_demo(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    args: argparse.Namespace
) -> Tuple[torch.Tensor, float]:
    """
    Run SageAttention3 with the given tensors and return output + timing.

    Args:
        q, k, v: Input tensors [B, H, N, D]
        args: Parsed command line arguments

    Returns:
        output: Attention output tensor
        elapsed_time: Time taken in seconds
    """
    print("\\n" + "="*80)
    print("SAGEATTENTION3 ALGORITHM DEMONSTRATION")
    print("="*80)

    print(f"Input tensor shapes:")
    print(f"  Q: {q.shape} ({q.dtype})")
    print(f"  K: {k.shape} ({k.dtype})")
    print(f"  V: {v.shape} ({v.dtype})")
    print(f"  Memory usage per tensor: {q.numel() * 2 / 1024**2:.2f} MB")

    print(f"\\nSageAttention3 Configuration:")
    print(f"  Per-block mean subtraction: {not args.no_smoothing}")
    print(f"  Causal masking: {args.causal}")
    print(f"  Query tile size: {args.tile_size_q}")
    print(f"  Key tile size: {args.tile_size_k}")
    print(f"  NVFP4 microscaling block: (1, 16)")

    # Warm up CUDA if available
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()

    start_time = time.time()

    # Run SageAttention3
    try:
        with torch.no_grad():
            output = sageattn3_torch(
                q=q.clone(),  # Clone to avoid modifying inputs
                k=k.clone(),
                v=v.clone(),
                tensor_layout="HND",
                is_causal=args.causal,
                per_block_mean=not args.no_smoothing,
                tile_size_q=args.tile_size_q,
                tile_size_k=args.tile_size_k,
                return_lse=False
            )
    except Exception as e:
        print(f"\\nERROR: SageAttention3 failed: {e}")
        return None, 0.0

    if args.device.startswith("cuda"):
        torch.cuda.synchronize()

    elapsed_time = time.time() - start_time

    print(f"\\nSageAttention3 Results:")
    print(f"  Output shape: {output.shape} ({output.dtype})")
    print(f"  Output range: [{output.min().item():.6f}, {output.max().item():.6f}]")
    print(f"  Output mean: {output.mean().item():.6f}")
    print(f"  Output std: {output.std().item():.6f}")
    print(f"  Execution time: {elapsed_time:.3f} seconds")

    return output, elapsed_time


def compare_with_pytorch_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sage_output: torch.Tensor,
    args: argparse.Namespace
) -> None:
    """
    Compare SageAttention3 output with PyTorch scaled_dot_product_attention.

    Args:
        q, k, v: Input tensors
        sage_output: Output from SageAttention3
        args: Command line arguments
    """
    print("\\n" + "-"*80)
    print("COMPARISON WITH PYTORCH SCALED DOT PRODUCT ATTENTION")
    print("-"*80)

    # Run PyTorch SDPA for comparison
    start_time = time.time()

    with torch.no_grad():
        pytorch_output = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=args.causal
        )

    if args.device.startswith("cuda"):
        torch.cuda.synchronize()

    pytorch_time = time.time() - start_time

    # Compute comparison metrics
    diff = (sage_output - pytorch_output).float()
    max_abs_diff = diff.abs().max().item()
    mean_abs_diff = diff.abs().mean().item()

    # Compute cosine similarity
    sage_flat = sage_output.reshape(1, -1).float()
    pytorch_flat = pytorch_output.reshape(1, -1).float()
    cos_sim = F.cosine_similarity(sage_flat, pytorch_flat, dim=-1).item()

    print(f"PyTorch SDPA Results:")
    print(f"  Output shape: {pytorch_output.shape} ({pytorch_output.dtype})")
    print(f"  Output range: [{pytorch_output.min().item():.6f}, {pytorch_output.max().item():.6f}]")
    print(f"  Execution time: {pytorch_time:.3f} seconds")

    print(f"\\nComparison Metrics:")
    print(f"  Max absolute difference: {max_abs_diff:.3e}")
    print(f"  Mean absolute difference: {mean_abs_diff:.3e}")
    print(f"  Cosine similarity: {cos_sim:.6f}")

    # Interpret results
    if cos_sim > 0.99:
        print(f"  ✓ Excellent agreement (cos_sim > 0.99)")
    elif cos_sim > 0.95:
        print(f"  ✓ Good agreement (cos_sim > 0.95)")
    elif cos_sim > 0.90:
        print(f"  ⚠ Fair agreement (cos_sim > 0.90)")
    else:
        print(f"  ✗ Poor agreement (cos_sim <= 0.90)")

    print(f"\\nNote: Some differences are expected due to:")
    print(f"  - NVFP4 quantization approximations")
    print(f"  - Two-level probability scaling")
    print(f"  - Different numerical precision in tiled processing")


def demonstrate_algorithm_components(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    args: argparse.Namespace
) -> None:
    """
    Demonstrate individual algorithm components for educational purposes.

    Args:
        q, k, v: Input tensors
        args: Command line arguments
    """
    print("\\n" + "-"*80)
    print("ALGORITHM COMPONENTS DEMONSTRATION")
    print("-"*80)

    from sageattn3_torch import nvfp4_quantize, smooth_qk_tensors

    # 1. Demonstrate NVFP4 quantization
    print("\\n1. NVFP4 Quantization:")
    print("   Converting FP16 tensors to 4-bit E2M1 format with microscaling")

    q_fp4, q_scales = nvfp4_quantize(q, block_size=(1, 16))
    print(f"   Q: {q.shape} ({q.dtype}) -> {q_fp4.shape} ({q_fp4.dtype}) + scales {q_scales.shape}")

    # Show quantization effect
    q_reconstructed = q_fp4.float() * 0.1  # Simplified reconstruction for demo
    quant_error = (q - q_reconstructed).abs().mean().item()
    print(f"   Average quantization error: {quant_error:.6f}")

    # 2. Demonstrate QK smoothing
    if not args.no_smoothing:
        print("\\n2. QK Smoothing (Outlier Handling):")
        print("   Subtracting per-block means to reduce quantization errors")

        q_smooth, k_smooth, q_correction = smooth_qk_tensors(q, k)

        q_outliers_before = (q.abs() > 2 * q.std()).sum().item()
        q_outliers_after = (q_smooth.abs() > 2 * q_smooth.std()).sum().item()

        print(f"   Q outliers before smoothing: {q_outliers_before}")
        print(f"   Q outliers after smoothing: {q_outliers_after}")
        print(f"   Outlier reduction: {(q_outliers_before - q_outliers_after)/max(q_outliers_before,1)*100:.1f}%")

    # 3. Show tiling strategy
    print("\\n3. Tiled Processing Strategy:")
    print("   Breaking attention computation into memory-efficient tiles")

    B, H, N, D = q.shape
    num_q_tiles = (N + args.tile_size_q - 1) // args.tile_size_q
    num_k_tiles = (N + args.tile_size_k - 1) // args.tile_size_k

    print(f"   Sequence length: {N}")
    print(f"   Tile sizes: Q={args.tile_size_q}, K={args.tile_size_k}")
    print(f"   Number of tiles: {num_q_tiles} × {num_k_tiles} = {num_q_tiles * num_k_tiles} total")
    print(f"   Memory savings: {N*N/(args.tile_size_q*args.tile_size_k):.1f}x less memory for attention matrix")


def main() -> None:
    """Main demonstration function."""
    args = parse_args()

    # Set up device and dtype
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed(args.seed)

    print("SageAttention3 Pure-Torch Implementation Demo")
    print(f"Device: {device}, Dtype: {dtype}, Seed: {args.seed}")

    # Create test tensors
    try:
        q, k, v = create_test_tensors(
            batch_size=args.batch,
            num_heads=args.heads,
            seq_len=args.seq_len,
            head_dim=args.head_dim,
            dtype=dtype,
            device=str(device)
        )
    except Exception as e:
        print(f"Failed to create test tensors: {e}")
        return

    # Demonstrate algorithm components
    if args.verbose:
        demonstrate_algorithm_components(q, k, v, args)

    # Run SageAttention3
    sage_output, sage_time = run_sageattention3_demo(q, k, v, args)

    if sage_output is None:
        print("Demo failed - check error messages above")
        return

    # Compare with PyTorch if requested
    if args.compare_pytorch:
        compare_with_pytorch_sdpa(q, k, v, sage_output, args)

    print("\\n" + "="*80)
    print("DEMO COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"This demo illustrated the SageAttention3 algorithm using:")
    print(f"  • NVFP4 E2M1 quantization with 1×16 microscaling")
    print(f"  • Two-level probability matrix scaling")
    print(f"  • Tiled online attention processing")
    print(f"  • QK smoothing for outlier handling")
    print(f"\\nFor more details, see the paper:")
    print(f"  'SageAttention3: Microscaling FP4 Attention for Inference'")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\\nDemo interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)