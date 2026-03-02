#!/usr/bin/env python3
"""
Detailed SageAttention3 Debug Analysis
======================================

Debug the numerical differences found in the Triton reference implementation.
This script will help identify which specific components are causing the accuracy issues.
"""

import torch
import sys
import os
import numpy as np
from typing import Dict, Any, Tuple

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from sageattention3_triton_ref import SageAttention3TritonReference

# Try to import the real kernel
try:
    from sageattn3 import sageattn3_blackwell
    REAL_KERNEL_AVAILABLE = True
except ImportError:
    REAL_KERNEL_AVAILABLE = False


def analyze_quantization_accuracy():
    """Analyze the accuracy of the FP4 quantization implementation."""
    print("\n🔍 ANALYZING FP4 QUANTIZATION ACCURACY")
    print("=" * 50)

    sage3 = SageAttention3TritonReference()
    device = torch.device('cuda')

    # Test FP4 quantization with known values
    test_values = torch.tensor([0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 0.2, 0.7, 1.5, 2.8, 3.9, 5.5],
                              dtype=torch.float16, device=device).unsqueeze(0).unsqueeze(0)

    print(f"Original values: {test_values.flatten().tolist()}")

    # Quantize to FP4
    quantized_fp4 = sage3._quantize_to_fp4_e2m1(test_values)
    print(f"Quantized FP4: {quantized_fp4.flatten().tolist()}")

    # Check representable values
    fp4_values = sage3.fp4_values[:7]  # Exclude inf
    print(f"Expected FP4 values: {fp4_values.tolist()}")

    # Test microscaling quantization
    test_tensor = torch.randn(1, 1, 32, 16, dtype=torch.float16, device=device) * 2  # Scale up for better test
    quantized = sage3.scale_and_quant_fp4(test_tensor)
    dequantized = sage3._dequantize_fp4_to_fp16(quantized.data, quantized.scales)

    diff = (test_tensor - dequantized).abs()
    print(f"\nMicroscaling quantization error:")
    print(f"  Max abs diff: {diff.max().item():.6f}")
    print(f"  Mean abs diff: {diff.mean().item():.6f}")
    print(f"  Relative error: {(diff / (test_tensor.abs() + 1e-8)).mean().item():.6f}")


def analyze_attention_components():
    """Analyze individual components of the attention computation."""
    print("\n🔍 ANALYZING ATTENTION COMPONENTS")
    print("=" * 50)

    sage3 = SageAttention3TritonReference()
    device = torch.device('cuda')

    # Small test case for detailed analysis
    B, H, L, D = 1, 2, 64, 64
    q = torch.randn(B, H, L, D, dtype=torch.float16, device=device) * 0.1  # Small values
    k = torch.randn(B, H, L, D, dtype=torch.float16, device=device) * 0.1
    v = torch.randn(B, H, L, D, dtype=torch.float16, device=device) * 0.1

    # Step 1: Test QK^T computation
    print("\n1. Testing QK^T computation:")

    # Standard computation
    scale = 1.0 / math.sqrt(D)
    qk_standard = torch.matmul(q, k.transpose(-2, -1)) * scale
    print(f"  Standard QK^T range: [{qk_standard.min().item():.6f}, {qk_standard.max().item():.6f}]")

    # With preprocessing
    q_processed, k_processed, v_processed = sage3.preprocess_qkv(
        q.clone(), k.clone(), v.clone(), smooth_k=True, smooth_q=True, per_block_q_mean_sub=True
    )
    qk_processed = torch.matmul(q_processed, k_processed.transpose(-2, -1)) * scale
    print(f"  Processed QK^T range: [{qk_processed.min().item():.6f}, {qk_processed.max().item():.6f}]")

    # Step 2: Test quantization effects
    print("\n2. Testing quantization effects:")

    q_quant = sage3.scale_and_quant_fp4(q_processed)
    k_quant = sage3.scale_and_quant_fp4_permute(k_processed)

    q_dequant = sage3._dequantize_fp4_to_fp16(q_quant.data, q_quant.scales)
    k_dequant = sage3._dequantize_fp4_to_fp16(k_quant.data, k_quant.scales)

    # Undo permutation for comparison
    perm_pattern = sage3._get_k_permutation_pattern(D)
    inverse_perm = torch.argsort(perm_pattern)
    k_dequant_unperm = k_dequant[..., inverse_perm]

    qk_dequant = torch.matmul(q_dequant, k_dequant_unperm.transpose(-2, -1)) * scale
    print(f"  Dequantized QK^T range: [{qk_dequant.min().item():.6f}, {qk_dequant.max().item():.6f}]")

    qk_diff = (qk_processed - qk_dequant).abs()
    print(f"  QK^T quantization error: max={qk_diff.max().item():.6f}, mean={qk_diff.mean().item():.6f}")

    # Step 3: Test softmax
    print("\n3. Testing softmax:")

    attn_standard = torch.softmax(qk_standard, dim=-1)
    attn_processed = torch.softmax(qk_processed, dim=-1)
    attn_dequant = torch.softmax(qk_dequant, dim=-1)

    print(f"  Standard softmax sum: {attn_standard.sum(dim=-1).mean().item():.6f}")
    print(f"  Processed softmax sum: {attn_processed.sum(dim=-1).mean().item():.6f}")
    print(f"  Dequantized softmax sum: {attn_dequant.sum(dim=-1).mean().item():.6f}")

    softmax_diff = (attn_processed - attn_dequant).abs()
    print(f"  Softmax error: max={softmax_diff.max().item():.6f}, mean={softmax_diff.mean().item():.6f}")


def compare_step_by_step():
    """Step-by-step comparison with PyTorch reference."""
    print("\n🔍 STEP-BY-STEP COMPARISON")
    print("=" * 50)

    sage3 = SageAttention3TritonReference()
    device = torch.device('cuda')

    # Simple test case
    B, H, L, D = 1, 4, 128, 64
    torch.manual_seed(42)

    q = torch.randn(B, H, L, D, dtype=torch.float16, device=device)
    k = torch.randn(B, H, L, D, dtype=torch.float16, device=device)
    v = torch.randn(B, H, L, D, dtype=torch.float16, device=device)

    print(f"Input shapes: Q={q.shape}, K={k.shape}, V={v.shape}")

    # PyTorch reference
    import math
    scale = 1.0 / math.sqrt(D)
    qk = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = torch.softmax(qk, dim=-1)
    output_ref = torch.matmul(attn, v)

    print(f"PyTorch reference output range: [{output_ref.min().item():.6f}, {output_ref.max().item():.6f}]")

    # Triton reference (without quantization for comparison)
    print("\nTesting without quantization:")

    # Test preprocessing only
    q_proc, k_proc, v_proc = sage3._preprocess_qkv_tensors(
        q.clone(), k.clone(), v.clone(),
        smooth_k=False, smooth_q=False, per_block_q_mean_sub=False
    )

    qk_proc = torch.matmul(q_proc, k_proc.transpose(-2, -1)) * scale
    attn_proc = torch.softmax(qk_proc, dim=-1)
    output_proc = torch.matmul(attn_proc, v_proc)

    diff = (output_ref - output_proc).abs()
    print(f"No preprocessing - error: max={diff.max().item():.6f}, mean={diff.mean().item():.6f}")

    # Test with smoothing
    q_smooth, k_smooth, v_smooth = sage3._preprocess_qkv_tensors(
        q.clone(), k.clone(), v.clone(),
        smooth_k=True, smooth_q=True, per_block_q_mean_sub=False
    )

    qk_smooth = torch.matmul(q_smooth, k_smooth.transpose(-2, -1)) * scale
    attn_smooth = torch.softmax(qk_smooth, dim=-1)
    output_smooth = torch.matmul(attn_smooth, v_smooth)

    diff = (output_ref - output_smooth).abs()
    print(f"With smoothing - error: max={diff.max().item():.6f}, mean={diff.mean().item():.6f}")

    # Test with all preprocessing
    q_full, k_full, v_full = sage3._preprocess_qkv_tensors(
        q.clone(), k.clone(), v.clone(),
        smooth_k=True, smooth_q=True, per_block_q_mean_sub=True
    )

    qk_full = torch.matmul(q_full, k_full.transpose(-2, -1)) * scale
    attn_full = torch.softmax(qk_full, dim=-1)
    output_full = torch.matmul(attn_full, v_full)

    diff = (output_ref - output_full).abs()
    print(f"Full preprocessing - error: max={diff.max().item():.6f}, mean={diff.mean().item():.6f}")


def main():
    print("🚀 SageAttention3 Detailed Debug Analysis")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return

    try:
        analyze_quantization_accuracy()
        analyze_attention_components()
        compare_step_by_step()

        print(f"\n✅ Debug analysis completed!")
        print("\n💡 Recommendations:")
        print("1. Check FP4 quantization implementation for representable values")
        print("2. Verify scale factor computation in microscaling")
        print("3. Review preprocessing effects (smoothing, mean subtraction)")
        print("4. Test with reduced quantization for better accuracy")

    except Exception as e:
        print(f"❌ Debug analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import math
    main()