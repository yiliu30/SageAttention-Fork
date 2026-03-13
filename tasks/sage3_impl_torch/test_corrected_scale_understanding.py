#!/usr/bin/env python3
"""
Test script to verify the corrected understanding of softmax scale calculation.

The real kernel's calculation is:
  softmax_scale = (qlist[0].shape[-1] * 2) ** (-0.5)

Where qlist[0].shape[-1] = D//2 (due to FP4 packing)
So: softmax_scale = ((D//2) * 2) ** (-0.5) = D ** (-0.5) = 1/sqrt(D)

This means the original Triton implementation was CORRECT!
"""

import torch
import math
import sys

# Add sage3 implementation to path
sys.path.insert(0, '/mnt/disk1/yiliu7/SageAttention-Fork/tasks/sage3_impl_torch')

from sageattn3_torch import sageattn3_torch
from sageattn3_torch_triton import sageattn3_torch_triton

def test_softmax_scale_understanding():
    """Test our understanding of the real kernel's scale calculation."""
    print("=" * 80)
    print("SOFTMAX SCALE CALCULATION - CORRECTED UNDERSTANDING")
    print("=" * 80)

    for D in [64, 128]:
        # Simulate real kernel calculation
        # qlist[0] would have shape [..., D//2] due to FP4 packing
        packed_last_dim = D // 2
        real_kernel_scale = (packed_last_dim * 2) ** (-0.5)

        # Standard attention scale
        standard_scale = 1.0 / math.sqrt(D)

        print(f"\nHead dimension D = {D}:")
        print(f"  Packed last dim (D//2):     {packed_last_dim}")
        print(f"  Real kernel calculation:    ({packed_last_dim} * 2) ** (-0.5) = {real_kernel_scale:.6f}")
        print(f"  Standard 1/sqrt(D):        {standard_scale:.6f}")
        print(f"  Match: {'✅ YES' if abs(real_kernel_scale - standard_scale) < 1e-10 else '❌ NO'}")

def test_reverted_accuracy():
    """Test accuracy with the reverted (correct) softmax scale."""
    print("\n" + "=" * 80)
    print("TESTING ACCURACY WITH REVERTED SOFTMAX SCALE")
    print("=" * 80)

    device = torch.device('cuda')
    dtype = torch.float16

    torch.manual_seed(42)
    B, H, N, D = 1, 4, 256, 128

    q = torch.randn(B, H, N, D, dtype=dtype, device=device)
    k = torch.randn(B, H, N, D, dtype=dtype, device=device)
    v = torch.randn(B, H, N, D, dtype=dtype, device=device)

    # Test without per_block_mean first
    print("--- Testing without per_block_mean ---")
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
        tile_size_k=128,
        sm_scale=None  # Use default 1/sqrt(D)
    )

    cosine_sim_no_pbm = torch.nn.functional.cosine_similarity(
        ref_output.flatten(), triton_output.flatten(), dim=0
    ).item()

    print(f"Cosine similarity (no per_block_mean): {cosine_sim_no_pbm:.6f}")

    # Test with per_block_mean
    print("\n--- Testing with per_block_mean ---")
    ref_output_pbm = sageattn3_torch(
        q, k, v,
        tensor_layout="HND",
        is_causal=False,
        per_block_mean=True,
        tile_size_q=128,
        tile_size_k=128
    )

    triton_output_pbm = sageattn3_torch_triton(
        q, k, v,
        per_block_mean=True,
        is_causal=False,
        tile_size_q=128,
        tile_size_k=128,
        sm_scale=None
    )

    cosine_sim_pbm = torch.nn.functional.cosine_similarity(
        ref_output_pbm.flatten(), triton_output_pbm.flatten(), dim=0
    ).item()

    print(f"Cosine similarity (with per_block_mean): {cosine_sim_pbm:.6f}")

    return cosine_sim_no_pbm, cosine_sim_pbm

if __name__ == "__main__":
    print("Testing corrected softmax scale understanding...")

    # Verify our understanding
    test_softmax_scale_understanding()

    # Test accuracy
    no_pbm_sim, pbm_sim = test_reverted_accuracy()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("Softmax scale analysis:")
    print("  Real kernel: (D//2 * 2) ** (-0.5) = 1/sqrt(D)")
    print("  Triton:      1/sqrt(D)")
    print("  Conclusion:  ✅ TRITON WAS CORRECT ORIGINALLY")
    print()
    print("This means Issue #1 was a MISDIAGNOSIS!")
    print("The real issues are likely:")
    print("  - Quantization differences (software vs hardware)")
    print("  - Numerical precision in online softmax")
    print("  - Other algorithmic details")
    print()
    print("Accuracy results:")
    print(f"  Without per_block_mean: {no_pbm_sim:.6f}")
    print(f"  With per_block_mean:    {pbm_sim:.6f}")
    print()

    if pbm_sim > 0.99:
        print("✅ EXCELLENT: >99% accuracy achieved!")
    elif pbm_sim > 0.985:
        print("✅ GOOD: Previous delta_s fix was the main issue")
    else:
        print("⚠️  The accuracy gap remains - need to investigate other factors")