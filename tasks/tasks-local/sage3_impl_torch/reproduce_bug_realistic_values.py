#!/usr/bin/env python3
"""
Reproduce the per_block_mean bug with realistic value ranges
"""

import torch
import sys
import os
sys.path.insert(0, '/mnt/disk1/yiliu7/SageAttention-Fork/tasks/sage3_impl_torch')

from sageattn3_torch import sageattn3_torch
from sageattn3_torch_triton import sageattn3_torch_triton

def test_with_normal_values():
    """Test with normal-sized random values like the original debug script."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float16

    # Use the same seed and values as debug_triton_delta_indexing.py
    torch.manual_seed(42)
    B, H, N, D = 1, 1, 256, 64

    # Normal randn values (not scaled down)
    q = torch.randn(B, H, N, D, dtype=dtype, device=device)
    k = torch.randn(B, H, N, D, dtype=dtype, device=device)
    v = torch.randn(B, H, N, D, dtype=dtype, device=device)

    print(f"Input value ranges:")
    print(f"  Q: min={q.min():.4f}, max={q.max():.4f}, std={q.std():.4f}")
    print(f"  K: min={k.min():.4f}, max={k.max():.4f}, std={k.std():.4f}")
    print(f"  V: min={v.min():.4f}, max={v.max():.4f}, std={v.std():.4f}")

    # Test PyTorch reference
    output_ref = sageattn3_torch(
        q=q, k=k, v=v,
        tensor_layout="HND",
        is_causal=False,
        per_block_mean=True,
        tile_size_q=128,
        tile_size_k=128
    )

    # Test Triton implementation
    output_triton = sageattn3_torch_triton(
        q=q, k=k, v=v,
        tensor_layout="HND",
        is_causal=False,
        per_block_mean=True,
        tile_size_q=128,
        tile_size_k=128,
        debug=True
    )

    # Compare outputs
    diff = (output_ref - output_triton).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    cosine_sim = torch.nn.functional.cosine_similarity(
        output_ref.flatten(), output_triton.flatten(), dim=0
    ).item()

    print(f"\nOutput comparison:")
    print(f"  Max difference: {max_diff:.6f}")
    print(f"  Mean difference: {mean_diff:.6f}")
    print(f"  Cosine similarity: {cosine_sim:.6f}")

    if cosine_sim < 0.99:
        print("⚠️  LOW COSINE SIMILARITY - BUG REPRODUCED!")
        return True
    else:
        print("✅ High cosine similarity - no bug detected")
        return False

def test_cogvideox_case():
    """Test the problematic CogVideoX case with normal values."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float16

    torch.manual_seed(42)
    B, H, N, D = 2, 30, 1024, 64  # CogVideoX configuration

    q = torch.randn(B, H, N, D, dtype=dtype, device=device)
    k = torch.randn(B, H, N, D, dtype=dtype, device=device)
    v = torch.randn(B, H, N, D, dtype=dtype, device=device)

    print(f"\nCogVideoX test: B={B}, H={H}, N={N}, D={D}")
    print(f"Input value ranges:")
    print(f"  Q: min={q.min():.4f}, max={q.max():.4f}")
    print(f"  K: min={k.min():.4f}, max={k.max():.4f}")
    print(f"  V: min={v.min():.4f}, max={v.max():.4f}")

    # Test with per_block_mean=False (should work)
    output_no_pbm = sageattn3_torch_triton(
        q=q, k=k, v=v,
        tensor_layout="HND",
        is_causal=False,
        per_block_mean=False,
        tile_size_q=128,
        tile_size_k=128,
        debug=False
    )

    # Test with per_block_mean=True (may have issues)
    output_pbm = sageattn3_torch_triton(
        q=q, k=k, v=v,
        tensor_layout="HND",
        is_causal=False,
        per_block_mean=True,
        tile_size_q=128,
        tile_size_k=128,
        debug=True
    )

    # Compare the two outputs
    diff = (output_no_pbm - output_pbm).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"Difference between per_block_mean=False and True:")
    print(f"  Max difference: {max_diff:.6f}")
    print(f"  Mean difference: {mean_diff:.6f}")

    if max_diff > 0.1:
        print("⚠️  LARGE DIFFERENCE - BUG REPRODUCED!")
        return True
    else:
        print("✅ Small difference")
        return False

if __name__ == "__main__":
    print("Reproducing per_block_mean bug with realistic values")
    print("=" * 60)

    bug_reproduced = False

    print("Test 1: Simple case with normal random values")
    bug_reproduced |= test_with_normal_values()

    print("\nTest 2: CogVideoX case comparison")
    bug_reproduced |= test_cogvideox_case()

    if bug_reproduced:
        print("\n❌ BUG REPRODUCED! The issue occurs with normal-sized input values.")
        print("Next steps: Examine the Triton kernel's delta_s handling more carefully.")
    else:
        print("\n✅ No bug detected. The issue may be even more specific.")