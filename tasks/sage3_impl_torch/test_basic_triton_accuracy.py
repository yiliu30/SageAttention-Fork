#!/usr/bin/env python3
"""
Test basic Triton attention accuracy without SageAttention3 features
"""

import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '/mnt/disk1/yiliu7/SageAttention-Fork/tasks/sage3_impl_torch')

from sageattn3_torch_triton import sageattn3_torch_triton

def test_basic_attention():
    """Test if basic attention computation is accurate."""
    device = torch.device('cuda')
    dtype = torch.float16

    torch.manual_seed(42)
    B, H, N, D = 1, 1, 256, 64

    q = torch.randn(B, H, N, D, dtype=dtype, device=device)
    k = torch.randn(B, H, N, D, dtype=dtype, device=device)
    v = torch.randn(B, H, N, D, dtype=dtype, device=device)

    print(f"Testing basic attention: B={B}, H={H}, N={N}, D={D}")

    # Ground truth: PyTorch SDPA
    output_sdpa = F.scaled_dot_product_attention(q, k, v)

    # Triton without any SageAttention3 features
    output_triton = sageattn3_torch_triton(
        q=q, k=k, v=v,
        tensor_layout="HND",
        is_causal=False,
        per_block_mean=False,  # No QK smoothing
        tile_size_q=128,
        tile_size_k=128
    )

    # Compare
    diff = (output_sdpa - output_triton).abs()
    cosine_sim = torch.nn.functional.cosine_similarity(
        output_sdpa.flatten(), output_triton.flatten(), dim=0
    ).item()

    print(f"PyTorch SDPA vs Triton (no SageAttention3 features):")
    print(f"  Max diff: {diff.max():.6f}")
    print(f"  Mean diff: {diff.mean():.6f}")
    print(f"  Cosine similarity: {cosine_sim:.6f}")

    if cosine_sim > 0.999:
        print("✅ Basic Triton attention is accurate")
        print("   The issue is specifically with SageAttention3 features")
    else:
        print("❌ Basic Triton attention has issues")
        print("   The problem is more fundamental than just per_block_mean")

    return cosine_sim > 0.999

def test_incremental_features():
    """Test SageAttention3 features incrementally."""
    device = torch.device('cuda')
    dtype = torch.float16

    torch.manual_seed(42)
    B, H, N, D = 1, 1, 256, 64

    q = torch.randn(B, H, N, D, dtype=dtype, device=device)
    k = torch.randn(B, H, N, D, dtype=dtype, device=device)
    v = torch.randn(B, H, N, D, dtype=dtype, device=device)

    # Reference: PyTorch SDPA
    output_ref = F.scaled_dot_product_attention(q, k, v)

    print(f"\nIncremental feature testing:")

    # Test 1: Basic Triton (already tested above)
    print("1. Basic Triton attention: (tested above)")

    # Test 2: With quantization but no per_block_mean
    print("2. Triton with quantization, no per_block_mean:")
    try:
        output_quant = sageattn3_torch_triton(
            q=q, k=k, v=v,
            tensor_layout="HND",
            is_causal=False,
            per_block_mean=False
        )

        cosine_quant = torch.nn.functional.cosine_similarity(
            output_ref.flatten(), output_quant.flatten(), dim=0
        ).item()
        print(f"   Cosine similarity: {cosine_quant:.6f}")

    except Exception as e:
        print(f"   ❌ Failed: {e}")

    # Test 3: With per_block_mean
    print("3. Triton with per_block_mean:")
    try:
        output_pbm = sageattn3_torch_triton(
            q=q, k=k, v=v,
            tensor_layout="HND",
            is_causal=False,
            per_block_mean=True
        )

        cosine_pbm = torch.nn.functional.cosine_similarity(
            output_ref.flatten(), output_pbm.flatten(), dim=0
        ).item()
        print(f"   Cosine similarity: {cosine_pbm:.6f}")

    except Exception as e:
        print(f"   ❌ Failed: {e}")

if __name__ == "__main__":
    print("Basic Triton Attention Accuracy Test")
    print("=" * 50)

    basic_ok = test_basic_attention()
    test_incremental_features()

    if not basic_ok:
        print("\n❌ CONCLUSION: The issue is in the basic Triton kernel, not just per_block_mean")
        print("   Need to debug the fundamental attention computation")
    else:
        print("\n✅ CONCLUSION: Basic Triton kernel is OK, issue is in SageAttention3 features")
        print("   Focus debugging on quantization and per_block_mean implementation")