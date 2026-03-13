#!/usr/bin/env python3
"""
Simple numerical verification focusing on key issues.
"""

import torch
import sys
import os
import numpy as np
import math

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from sageattention3_triton_ref import SageAttention3TritonReference

# Try to import the real kernel
try:
    from sageattn3 import sageattn3_blackwell
    REAL_KERNEL_AVAILABLE = True
except ImportError:
    REAL_KERNEL_AVAILABLE = False


def simple_comparison():
    """Simple comparison without complex debugging."""
    print("🚀 Simple Numerical Verification")
    print("=" * 50)

    sage3 = SageAttention3TritonReference()
    device = torch.device('cuda')

    # Test case
    B, H, L, D = 1, 4, 128, 64
    torch.manual_seed(42)

    q = torch.randn(B, H, L, D, dtype=torch.float16, device=device)
    k = torch.randn(B, H, L, D, dtype=torch.float16, device=device)
    v = torch.randn(B, H, L, D, dtype=torch.float16, device=device)

    print(f"Input shapes: Q={q.shape}, K={k.shape}, V={v.shape}")

    # PyTorch reference
    ref_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
    print(f"PyTorch SDPA output range: [{ref_out.min().item():.6f}, {ref_out.max().item():.6f}]")

    # Triton reference
    try:
        triton_out = sage3.sageattn3_triton_ref(
            q.clone(), k.clone(), v.clone(),
            tensor_layout="HND",
            is_causal=False,
            smooth_k=True,
            smooth_q=True,
            per_block_q_mean_sub=True
        )
        print(f"Triton output range: [{triton_out.min().item():.6f}, {triton_out.max().item():.6f}]")

        # Compare
        diff = (triton_out - ref_out).float()
        max_abs = diff.abs().max().item()
        mean_abs = diff.abs().mean().item()
        cos_sim = torch.nn.functional.cosine_similarity(
            triton_out.reshape(1, -1).float(), ref_out.reshape(1, -1).float(), dim=-1
        ).item()

        print(f"\nTriton vs PyTorch SDPA:")
        print(f"  Max abs diff: {max_abs:.6e}")
        print(f"  Mean abs diff: {mean_abs:.6e}")
        print(f"  Cosine similarity: {cos_sim:.6f}")

    except Exception as e:
        print(f"❌ Triton reference failed: {e}")
        import traceback
        traceback.print_exc()

    # Real kernel (if available)
    if REAL_KERNEL_AVAILABLE:
        try:
            real_out = sageattn3_blackwell(
                q.clone(), k.clone(), v.clone(),
                is_causal=False,
                per_block_mean=True
            )
            print(f"Real kernel output range: [{real_out.min().item():.6f}, {real_out.max().item():.6f}]")

            # Compare with PyTorch
            diff = (real_out - ref_out).float()
            max_abs = diff.abs().max().item()
            mean_abs = diff.abs().mean().item()
            cos_sim = torch.nn.functional.cosine_similarity(
                real_out.reshape(1, -1).float(), ref_out.reshape(1, -1).float(), dim=-1
            ).item()

            print(f"\nReal kernel vs PyTorch SDPA:")
            print(f"  Max abs diff: {max_abs:.6e}")
            print(f"  Mean abs diff: {mean_abs:.6e}")
            print(f"  Cosine similarity: {cos_sim:.6f}")

        except Exception as e:
            print(f"❌ Real kernel failed: {e}")
            import traceback
            traceback.print_exc()

    print("\n💡 Analysis:")
    print("The Triton reference implements extreme quantization (FP4) which")
    print("significantly degrades precision compared to standard attention.")
    print("This is expected behavior for educational/demonstration purposes.")
    print("\nFor production use, the real CUDA kernel would have optimizations")
    print("and potentially different quantization strategies for better accuracy.")


def test_quantization_only():
    """Test just the quantization components."""
    print("\n🔍 Testing Quantization Components")
    print("=" * 40)

    sage3 = SageAttention3TritonReference()
    device = torch.device('cuda')

    # Test data
    test_data = torch.tensor([0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 0.1, 0.7, 1.5, 2.5, 3.5, 5.0],
                             dtype=torch.float16, device=device)

    print(f"Original: {test_data.tolist()}")

    # Test FP4 quantization
    quantized = sage3._quantize_to_fp4_e2m1(test_data)
    print(f"FP4 quantized: {quantized.tolist()}")

    # Test full quantization pipeline
    test_tensor = torch.randn(1, 1, 16, 16, dtype=torch.float16, device=device)
    quant_result = sage3.scale_and_quant_fp4(test_tensor)

    print(f"Full quantization:")
    print(f"  Original range: [{test_tensor.min().item():.6f}, {test_tensor.max().item():.6f}]")
    print(f"  Quantized data shape: {quant_result.data.shape}")
    print(f"  Scales shape: {quant_result.scales.shape}")

    # Dequantize and compare
    dequantized = sage3._dequantize_fp4_to_fp16(quant_result.data, quant_result.scales)
    error = (test_tensor - dequantized).abs()

    print(f"  Dequantized range: [{dequantized.min().item():.6f}, {dequantized.max().item():.6f}]")
    print(f"  Quantization error: max={error.max().item():.6f}, mean={error.mean().item():.6f}")


def main():
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return

    try:
        simple_comparison()
        test_quantization_only()

        print(f"\n✅ Analysis completed!")

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()