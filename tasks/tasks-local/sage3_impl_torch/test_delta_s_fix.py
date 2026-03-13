#!/usr/bin/env python3
"""
Test script to verify Delta_S indexing fix in Triton kernel.

This script tests the specific fix for Issue #6: Delta_S Indexing and Timing
by comparing the corrected Triton implementation against the PyTorch reference.
"""

import torch
import sys
import os

# Add sage3 implementation to path
sys.path.insert(0, '/mnt/disk1/yiliu7/SageAttention-Fork/tasks/sage3_impl_torch')

from sageattn3_torch import sageattn3_torch
from sageattn3_torch_triton import sageattn3_torch_triton

def test_delta_s_indexing_fix():
    """Test that Delta_S indexing now matches PyTorch reference."""
    print("=" * 80)
    print("TESTING DELTA_S INDEXING FIX (Issue #6)")
    print("=" * 80)

    device = torch.device('cuda')
    dtype = torch.float16

    # Test configuration that exercises delta_s indexing
    torch.manual_seed(42)
    B, H, N, D = 1, 4, 256, 128  # 2 groups (256//128=2), 2 tiles (256//128=2)

    print(f"Test config: B={B}, H={H}, N={N}, D={D}")
    print(f"Groups: {N//128}, Tiles: {N//128}")

    # Create test tensors
    q = torch.randn(B, H, N, D, dtype=dtype, device=device)
    k = torch.randn(B, H, N, D, dtype=dtype, device=device)
    v = torch.randn(B, H, N, D, dtype=dtype, device=device)

    # Test with per_block_mean=True (uses delta_s)
    print("\n--- Testing with per_block_mean=True ---")

    # PyTorch reference
    print("Running PyTorch reference...")
    ref_output = sageattn3_torch(
        q, k, v,
        tensor_layout="HND",
        is_causal=False,
        per_block_mean=True,
        tile_size_q=128,  # Match GROUP_SIZE
        tile_size_k=128
    )

    # Fixed Triton implementation
    print("Running fixed Triton implementation...")
    triton_output = sageattn3_torch_triton(
        q, k, v,
        per_block_mean=True,
        is_causal=False,
        tile_size_q=128,
        tile_size_k=128
    )

    # Compare results
    cosine_sim = torch.nn.functional.cosine_similarity(
        ref_output.flatten(), triton_output.flatten(), dim=0
    ).item()

    max_diff = torch.max(torch.abs(ref_output - triton_output)).item()
    mean_diff = torch.mean(torch.abs(ref_output - triton_output)).item()

    print(f"\n--- RESULTS ---")
    print(f"Cosine similarity: {cosine_sim:.6f}")
    print(f"Max absolute diff: {max_diff:.6e}")
    print(f"Mean absolute diff: {mean_diff:.6e}")

    # Check if improvement
    if cosine_sim > 0.995:
        print("✅ EXCELLENT: Cosine similarity > 99.5%")
        return True
    elif cosine_sim > 0.99:
        print("✅ GOOD: Cosine similarity > 99%")
        return True
    elif cosine_sim > 0.97:
        print("⚠️  IMPROVED: Cosine similarity > 97% (was ~77% before)")
        return True
    else:
        print("❌ STILL NEEDS WORK: Cosine similarity < 97%")
        return False

def test_without_per_block_mean():
    """Test without per_block_mean for baseline comparison."""
    print("\n" + "=" * 80)
    print("BASELINE TEST: without per_block_mean (should be ~97%)")
    print("=" * 80)

    device = torch.device('cuda')
    dtype = torch.float16

    torch.manual_seed(42)
    B, H, N, D = 1, 4, 256, 128

    q = torch.randn(B, H, N, D, dtype=dtype, device=device)
    k = torch.randn(B, H, N, D, dtype=dtype, device=device)
    v = torch.randn(B, H, N, D, dtype=dtype, device=device)

    # Test with per_block_mean=False (no delta_s)
    ref_output = sageattn3_torch(
        q, k, v,
        tensor_layout="HND",
        is_causal=False,
        per_block_mean=False,
        tile_size_q=128,
        tile_size_k=128
    )

    triton_output = sageattn3_torch_triton(
        q, k, v,
        per_block_mean=False,
        is_causal=False,
        tile_size_q=128,
        tile_size_k=128
    )

    cosine_sim = torch.nn.functional.cosine_similarity(
        ref_output.flatten(), triton_output.flatten(), dim=0
    ).item()

    print(f"Baseline cosine similarity (no delta_s): {cosine_sim:.6f}")
    return cosine_sim

if __name__ == "__main__":
    print("Testing Delta_S indexing fix...")

    # Test baseline first
    baseline_sim = test_without_per_block_mean()

    # Test the fix
    fix_works = test_delta_s_indexing_fix()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Baseline accuracy (no delta_s): {baseline_sim:.4f}")

    if fix_works:
        print("✅ Delta_S indexing fix: SUCCESS")
        print("   per_block_mean accuracy improved significantly")
    else:
        print("❌ Delta_S indexing fix: NEEDS MORE WORK")
        print("   Additional debugging required")

    print("\nNext steps:")
    print("1. If fixed: Test other accuracy issues (softmax scale, quantization)")
    print("2. If not fixed: Debug delta_s computation and application in detail")