#!/usr/bin/env python3
"""
SageAttention3 Kernel vs Pure-Torch Implementation Comparison

This script compares the output of our pure-torch implementation against
the real SageAttention3 Blackwell kernel to verify correctness.
"""

import sys
import argparse
import torch
import torch.nn.functional as F
from typing import Tuple, Optional
import time

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length")
    parser.add_argument("--head-dim", type=int, default=64, help="Head dimension (<256)")
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16", help="Data type")
    parser.add_argument("--device", default="cuda", help="Device")
    parser.add_argument("--causal", action="store_true", help="Causal attention")
    parser.add_argument("--no-per-block-mean", action="store_true", help="Disable smoothing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--tile-size-q", type=int, default=64, help="Query tile size")
    parser.add_argument("--tile-size-k", type=int, default=64, help="Key tile size")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    return parser.parse_args()

def create_test_inputs(args) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create reproducible test inputs matching the real kernel demo."""
    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    if args.head_dim >= 256:
        raise ValueError("Head dimension must be < 256 for SageAttention3")

    # Use same random generation as real demo for reproducibility
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    shape = (args.batch, args.heads, args.seq_len, args.head_dim)
    q = torch.randn(shape, dtype=dtype, device=device, generator=generator)
    k = torch.randn(shape, dtype=dtype, device=device, generator=generator)
    v = torch.randn(shape, dtype=dtype, device=device, generator=generator)

    return q, k, v

def run_real_kernel(q, k, v, args) -> Tuple[Optional[torch.Tensor], float, str]:
    """Run the real SageAttention3 Blackwell kernel."""
    try:
        # Import the real kernel
        sys.path.insert(0, '/mnt/disk1/yiliu7/SageAttention-Fork/sageattention3_blackwell')
        from sageattn3 import sageattn3_blackwell

        print("✅ Successfully imported real SageAttention3 kernel")

        # Run the kernel
        start_time = time.time()
        with torch.no_grad():
            # Clone inputs since kernel modifies them in-place
            output = sageattn3_blackwell(
                q.clone(),
                k.clone(),
                v.clone(),
                is_causal=args.causal,
                per_block_mean=not args.no_per_block_mean
            )

        if args.device.startswith("cuda"):
            torch.cuda.synchronize()

        elapsed_time = time.time() - start_time
        return output, elapsed_time, "success"

    except ImportError as e:
        return None, 0.0, f"import_error: {e}"
    except Exception as e:
        return None, 0.0, f"runtime_error: {e}"

def run_pure_torch_impl(q, k, v, args) -> Tuple[Optional[torch.Tensor], float, str]:
    """Run our pure-torch implementation."""
    try:
        # Import our implementation
        sys.path.insert(0, '/mnt/disk1/yiliu7/SageAttention-Fork/tasks/sage3_impl_torch')
        from sageattn3_torch import sageattn3_torch

        print("✅ Successfully imported pure-torch FINAL implementation")

        # Run our implementation
        start_time = time.time()
        with torch.no_grad():
            output = sageattn3_torch(
                q=q.clone(),
                k=k.clone(),
                v=v.clone(),
                tensor_layout="HND",
                is_causal=args.causal,
                per_block_mean=not args.no_per_block_mean,
                tile_size_q=args.tile_size_q,
                tile_size_k=args.tile_size_k,
                return_lse=False
            )

        if args.device.startswith("cuda"):
            torch.cuda.synchronize()

        elapsed_time = time.time() - start_time
        return output, elapsed_time, "success"

    except Exception as e:
        return None, 0.0, f"error: {e}"

def run_pytorch_reference(q, k, v, args) -> Tuple[torch.Tensor, float]:
    """Run PyTorch scaled_dot_product_attention as reference."""
    start_time = time.time()
    with torch.no_grad():
        output = F.scaled_dot_product_attention(q, k, v, is_causal=args.causal)

    if args.device.startswith("cuda"):
        torch.cuda.synchronize()

    elapsed_time = time.time() - start_time
    return output, elapsed_time

def compare_outputs(output1, output2, name1, name2):
    """Compare two attention outputs and print metrics."""
    if output1 is None or output2 is None:
        print(f"❌ Cannot compare - one output is None")
        return None, None, None

    # Compute differences
    diff = (output1 - output2).float()
    max_abs_diff = diff.abs().max().item()
    mean_abs_diff = diff.abs().mean().item()

    # Compute cosine similarity
    flat1 = output1.reshape(1, -1).float()
    flat2 = output2.reshape(1, -1).float()
    cos_sim = F.cosine_similarity(flat1, flat2, dim=-1).item()

    print(f"\\n📊 Comparison: {name1} vs {name2}")
    print(f"  Max absolute difference:  {max_abs_diff:.3e}")
    print(f"  Mean absolute difference: {mean_abs_diff:.3e}")
    print(f"  Cosine similarity:        {cos_sim:.6f}")

    # Interpret results
    if cos_sim > 0.99:
        print(f"  ✅ Excellent agreement (cos_sim > 0.99)")
    elif cos_sim > 0.95:
        print(f"  ✅ Good agreement (cos_sim > 0.95)")
    elif cos_sim > 0.90:
        print(f"  ⚠️  Fair agreement (cos_sim > 0.90)")
    else:
        print(f"  ❌ Poor agreement (cos_sim <= 0.90)")

    return max_abs_diff, mean_abs_diff, cos_sim

def main():
    args = parse_args()

    print("🔍 SageAttention3 Kernel vs Pure-Torch Implementation Comparison")
    print("=" * 80)

    # Check device availability
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("❌ CUDA not available, switching to CPU")
        args.device = "cpu"

    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    print(f"Configuration:")
    print(f"  Shape: [{args.batch}, {args.heads}, {args.seq_len}, {args.head_dim}]")
    print(f"  Device: {device}, Dtype: {dtype}")
    print(f"  Causal: {args.causal}, Smoothing: {not args.no_per_block_mean}")
    print(f"  Tile sizes: Q={args.tile_size_q}, K={args.tile_size_k}")

    # Create test inputs
    try:
        q, k, v = create_test_inputs(args)
        print(f"✅ Created test tensors with seed {args.seed}")
    except Exception as e:
        print(f"❌ Failed to create test tensors: {e}")
        return 1

    # Run PyTorch reference
    print(f"\\n🔧 Running PyTorch SDPA reference...")
    pytorch_output, pytorch_time = run_pytorch_reference(q, k, v, args)
    print(f"✅ PyTorch SDPA completed in {pytorch_time:.3f}s")
    print(f"   Output shape: {pytorch_output.shape}, dtype: {pytorch_output.dtype}")
    print(f"   Output range: [{pytorch_output.min().item():.6f}, {pytorch_output.max().item():.6f}]")

    # Run real kernel
    print(f"\\n🚀 Running real SageAttention3 Blackwell kernel...")
    kernel_output, kernel_time, kernel_status = run_real_kernel(q, k, v, args)

    if kernel_status == "success":
        print(f"✅ Real kernel completed in {kernel_time:.3f}s")
        print(f"   Output shape: {kernel_output.shape}, dtype: {kernel_output.dtype}")
        print(f"   Output range: [{kernel_output.min().item():.6f}, {kernel_output.max().item():.6f}]")
    else:
        print(f"❌ Real kernel failed: {kernel_status}")

    # Run our implementation
    print(f"\\n🧠 Running pure-torch implementation...")
    torch_output, torch_time, torch_status = run_pure_torch_impl(q, k, v, args)

    if torch_status == "success":
        print(f"✅ Pure-torch implementation completed in {torch_time:.3f}s")
        print(f"   Output shape: {torch_output.shape}, dtype: {torch_output.dtype}")
        print(f"   Output range: [{torch_output.min().item():.6f}, {torch_output.max().item():.6f}]")
    else:
        print(f"❌ Pure-torch implementation failed: {torch_status}")

    # Perform comparisons
    print(f"\\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)

    results = {}

    # Compare real kernel vs PyTorch reference
    if kernel_status == "success":
        max_diff, mean_diff, cos_sim = compare_outputs(
            kernel_output, pytorch_output, "Real Kernel", "PyTorch SDPA"
        )
        results["kernel_vs_pytorch"] = (max_diff, mean_diff, cos_sim)

    # Compare our implementation vs PyTorch reference
    if torch_status == "success":
        max_diff, mean_diff, cos_sim = compare_outputs(
            torch_output, pytorch_output, "Pure-Torch Impl", "PyTorch SDPA"
        )
        results["torch_vs_pytorch"] = (max_diff, mean_diff, cos_sim)

    # Compare our implementation vs real kernel (most important!)
    if kernel_status == "success" and torch_status == "success":
        max_diff, mean_diff, cos_sim = compare_outputs(
            torch_output, kernel_output, "Pure-Torch Impl", "Real Kernel"
        )
        results["torch_vs_kernel"] = (max_diff, mean_diff, cos_sim)
        print(f"\\n🎯 KEY COMPARISON: Pure-Torch vs Real Kernel")
        print(f"   This measures how well our implementation matches the real kernel")

    # Performance summary
    if kernel_status == "success" and torch_status == "success":
        speedup = torch_time / kernel_time if kernel_time > 0 else float('inf')
        print(f"\\n⚡ Performance Summary:")
        print(f"   PyTorch SDPA:     {pytorch_time:.3f}s")
        print(f"   Real Kernel:      {kernel_time:.3f}s ({pytorch_time/kernel_time:.1f}x faster than PyTorch)")
        print(f"   Pure-Torch Impl:  {torch_time:.3f}s ({speedup:.1f}x slower than real kernel)")
        print(f"   Note: Pure-torch is educational - real kernel optimized for speed")

    # Final verdict
    print(f"\\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)

    if "torch_vs_kernel" in results:
        _, _, cos_sim = results["torch_vs_kernel"]
        if cos_sim > 0.95:
            print(f"🎉 SUCCESS: Pure-torch implementation correctly matches real kernel!")
            print(f"   Cosine similarity: {cos_sim:.6f} (> 0.95 threshold)")
            return 0
        else:
            print(f"❌ NEEDS IMPROVEMENT: Pure-torch implementation differs from real kernel")
            print(f"   Cosine similarity: {cos_sim:.6f} (< 0.95 threshold)")
            return 1
    else:
        print(f"❌ CANNOT VERIFY: Unable to run both implementations for comparison")
        return 1

if __name__ == "__main__":
    sys.exit(main())