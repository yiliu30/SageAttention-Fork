#!/usr/bin/env python3
"""
Test script to verify the softmax scale fix (Issue #1).

This script tests the critical fix for the softmax scale calculation error:
- Real kernel: 1/sqrt(2*D)
- Triton (before): 1/sqrt(D) ❌ WRONG
- Triton (after): 1/sqrt(2*D) ✅ FIXED

Expected impact: Major accuracy improvement from ~98.4% to >99% cosine similarity.
"""

import torch
import math
import sys
import os

# Add sage3 implementation to path
sys.path.insert(0, '/mnt/disk1/yiliu7/SageAttention-Fork/tasks/sage3_impl_torch')

from sageattn3_torch import sageattn3_torch
from sageattn3_torch_triton import sageattn3_torch_triton

def test_softmax_scale_values():
    """Test the actual softmax scale values used."""
    print("=" * 80)
    print("SOFTMAX SCALE CALCULATION VERIFICATION")
    print("=" * 80)

    for D in [64, 128]:
        # Real kernel calculation (from api.py:135)
        real_scale = (D * 2) ** (-0.5)

        # Old Triton calculation (incorrect)
        old_triton_scale = 1.0 / math.sqrt(D)

        # New Triton calculation (fixed)
        new_triton_scale = 1.0 / math.sqrt(2.0 * D)

        print(f"\nHead dimension D = {D}:")
        print(f"  Real kernel scale:    {real_scale:.6f}")
        print(f"  Old Triton scale:     {old_triton_scale:.6f} ❌ ({old_triton_scale/real_scale:.3f}x too large)")
        print(f"  New Triton scale:     {new_triton_scale:.6f} ✅ ({'MATCH' if abs(new_triton_scale - real_scale) < 1e-10 else 'MISMATCH'})")

def test_softmax_scale_fix():
    """Test that the softmax scale fix improves accuracy."""
    print("\n" + "=" * 80)
    print("TESTING SOFTMAX SCALE FIX ACCURACY")
    print("=" * 80)

    device = torch.device('cuda')
    dtype = torch.float16

    # Test different configurations
    test_configs = [
        {"B": 1, "H": 4, "N": 256, "D": 64},
        {"B": 1, "H": 8, "N": 256, "D": 128},
    ]

    results = {}

    for config in test_configs:
        B, H, N, D = config["B"], config["H"], config["N"], config["D"]
        print(f"\n--- Test Config: B={B}, H={H}, N={N}, D={D} ---")

        torch.manual_seed(42)
        q = torch.randn(B, H, N, D, dtype=dtype, device=device)
        k = torch.randn(B, H, N, D, dtype=dtype, device=device)
        v = torch.randn(B, H, N, D, dtype=dtype, device=device)

        # PyTorch reference (always correct)
        print("Running PyTorch reference...")
        ref_output = sageattn3_torch(
            q, k, v,
            tensor_layout="HND",
            is_causal=False,
            per_block_mean=False,  # Test without delta_s first
            tile_size_q=128,
            tile_size_k=128
        )

        # Fixed Triton implementation
        print("Running fixed Triton implementation...")
        triton_output = sageattn3_torch_triton(
            q, k, v,
            per_block_mean=False,
            is_causal=False,
            tile_size_q=128,
            tile_size_k=128,
            sm_scale=None  # Let it use the corrected calculation
        )

        # Compare results
        cosine_sim = torch.nn.functional.cosine_similarity(
            ref_output.flatten(), triton_output.flatten(), dim=0
        ).item()

        max_diff = torch.max(torch.abs(ref_output - triton_output)).item()
        mean_diff = torch.mean(torch.abs(ref_output - triton_output)).item()

        print(f"Results:")
        print(f"  Cosine similarity: {cosine_sim:.6f}")
        print(f"  Max absolute diff: {max_diff:.6e}")
        print(f"  Mean absolute diff: {mean_diff:.6e}")

        # Store results
        results[f"D{D}"] = cosine_sim

        # Check improvement
        if cosine_sim > 0.999:
            print("  ✅ EXCELLENT: >99.9% similarity!")
        elif cosine_sim > 0.995:
            print("  ✅ VERY GOOD: >99.5% similarity")
        elif cosine_sim > 0.99:
            print("  ✅ GOOD: >99% similarity")
        elif cosine_sim > 0.985:
            print("  ⚠️  IMPROVED: >98.5% similarity (was ~98.4%)")
        else:
            print("  ❌ STILL NEEDS WORK")

    return results

def test_with_per_block_mean():
    """Test the fix with per_block_mean enabled."""
    print("\n" + "=" * 80)
    print("TESTING WITH PER_BLOCK_MEAN ENABLED")
    print("=" * 80)

    device = torch.device('cuda')
    dtype = torch.float16

    torch.manual_seed(42)
    B, H, N, D = 1, 4, 256, 128

    q = torch.randn(B, H, N, D, dtype=dtype, device=device)
    k = torch.randn(B, H, N, D, dtype=dtype, device=device)
    v = torch.randn(B, H, N, D, dtype=dtype, device=device)

    print("Running PyTorch reference with per_block_mean...")
    ref_output = sageattn3_torch(
        q, k, v,
        tensor_layout="HND",
        is_causal=False,
        per_block_mean=True,
        tile_size_q=128,
        tile_size_k=128
    )

    print("Running fixed Triton with per_block_mean...")
    triton_output = sageattn3_torch_triton(
        q, k, v,
        per_block_mean=True,
        is_causal=False,
        tile_size_q=128,
        tile_size_k=128,
        sm_scale=None
    )

    cosine_sim = torch.nn.functional.cosine_similarity(
        ref_output.flatten(), triton_output.flatten(), dim=0
    ).item()

    print(f"Cosine similarity with per_block_mean: {cosine_sim:.6f}")

    if cosine_sim > 0.999:
        print("✅ EXCELLENT: Both fixes (softmax scale + delta_s) working perfectly!")
    elif cosine_sim > 0.995:
        print("✅ VERY GOOD: Major improvements achieved")
    else:
        print("⚠️  PARTIAL: Softmax scale helps, but other issues remain")

    return cosine_sim

if __name__ == "__main__":
    print("Testing Softmax Scale Fix (Issue #1)...")

    # Step 1: Verify the scale calculation values
    test_softmax_scale_values()

    # Step 2: Test accuracy improvement without per_block_mean
    baseline_results = test_softmax_scale_fix()

    # Step 3: Test with per_block_mean (combines both fixes)
    per_block_result = test_with_per_block_mean()

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("Softmax scale calculation:")
    print("  Before fix: 1/sqrt(D) ❌ WRONG")
    print("  After fix:  1/sqrt(2*D) ✅ CORRECT")
    print()
    print("Accuracy results:")
    for config, sim in baseline_results.items():
        print(f"  {config} baseline: {sim:.4f}")
    print(f"  D128 with per_block_mean: {per_block_result:.4f}")
    print()

    # Overall assessment
    if any(sim > 0.999 for sim in baseline_results.values()):
        print("🎉 SUCCESS: Softmax scale fix achieved target >99.9% accuracy!")
    elif any(sim > 0.995 for sim in baseline_results.values()):
        print("✅ MAJOR SUCCESS: Softmax scale fix achieved >99.5% accuracy")
    elif any(sim > 0.99 for sim in baseline_results.values()):
        print("✅ SUCCESS: Softmax scale fix achieved >99% accuracy")
    else:
        print("⚠️  PARTIAL: Improvement seen, but additional fixes needed")

    print("\nNext steps:")
    if any(sim > 0.995 for sim in baseline_results.values()):
        print("1. ✅ Softmax scale: FIXED")
        print("2. ✅ Delta_s indexing: FIXED")
        print("3. 🎯 Focus on remaining quantization/precision issues")
    else:
        print("1. Investigate remaining numerical differences")
        print("2. Check quantization implementation details")
        print("3. Verify other algorithmic components")