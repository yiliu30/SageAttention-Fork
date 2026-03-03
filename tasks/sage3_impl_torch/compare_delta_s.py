#!/usr/bin/env python3
"""
Delta_s Comparison Between Triton and Real Kernel
================================================

This script specifically compares the delta_s QK smoothing correction values
between the Triton implementation and the real kernel to determine if they
use identical QK smoothing algorithms.

Delta_s is computed as: delta_s = qm @ k^T
Where qm is the query mean (per-block or global) and k is the key tensor.

This is a critical component of SageAttention3 that corrects for the bias
introduced by QK smoothing.

Usage:
    python compare_delta_s.py
"""

import torch
import torch.nn.functional as F
import sys
import os
from typing import Dict, Any, Tuple, Optional
import numpy as np

def setup_paths():
    """Add necessary paths for imports."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    blackwell_path = '/mnt/disk1/yiliu7/SageAttention-Fork/sageattention3_blackwell'
    sys.path.insert(0, blackwell_path)

def capture_triton_delta_s(q, k, v):
    """Capture delta_s from Triton implementation."""
    print("🔍 Capturing Triton delta_s...")

    # Import the required functions
    import sageattn3_torch
    from sageattn3_torch import apply_qk_smoothing

    # Hook into QK smoothing to capture delta_s
    captured_delta_s = {}

    original_smoothing = sageattn3_torch.apply_qk_smoothing
    def capture_smoothing(*args, **kwargs):
        result = original_smoothing(*args, **kwargs)
        captured_delta_s['triton'] = {
            'q_smoothed': result[0].clone(),
            'k_smoothed': result[1].clone(),
            'delta_s': result[2].clone() if result[2] is not None else None
        }
        return result

    # Apply patch temporarily
    sageattn3_torch.apply_qk_smoothing = capture_smoothing

    try:
        # Run Triton implementation
        from sageattn3_torch_triton import sageattn3_torch_triton
        triton_output = sageattn3_torch_triton(
            q=q.clone(), k=k.clone(), v=v.clone(),
            tensor_layout="HND",
            is_causal=False,
            per_block_mean=True,
            tile_size_q=128,
            tile_size_k=128,
            debug=False
        )

        # Extract captured delta_s
        triton_delta_s = captured_delta_s['triton']['delta_s']

        if triton_delta_s is not None:
            print(f"   ✅ Triton delta_s: {triton_delta_s.shape} {triton_delta_s.dtype}")
            print(f"      Range: [{triton_delta_s.min():.6f}, {triton_delta_s.max():.6f}]")
            print(f"      Mean: {triton_delta_s.mean():.6f}, Std: {triton_delta_s.std():.6f}")
        else:
            print("   ❌ Triton delta_s is None")

        return triton_delta_s, triton_output

    finally:
        # Restore original function
        sageattn3_torch.apply_qk_smoothing = original_smoothing

def capture_real_kernel_delta_s(q, k, v):
    """Capture delta_s from real kernel using API patching."""
    print("🔍 Capturing Real Kernel delta_s...")

    # Import required modules
    import sageattn3.api as api_module

    # Store captured data
    captured_data = {}

    # Patch preprocess_qkv to capture delta_s
    original_preprocess = api_module.preprocess_qkv
    def capture_preprocess(q, k, v, per_block_mean=True):
        q_processed, k_processed, v_processed, delta_s = original_preprocess(q, k, v, per_block_mean)
        captured_data['real'] = {
            'q_processed': q_processed.clone(),
            'k_processed': k_processed.clone(),
            'v_processed': v_processed.clone(),
            'delta_s': delta_s.clone() if delta_s is not None else None
        }
        return q_processed, k_processed, v_processed, delta_s

    # Apply patch
    api_module.preprocess_qkv = capture_preprocess

    try:
        # Run real kernel
        from sageattn3 import sageattn3_blackwell
        real_output = sageattn3_blackwell(
            q.clone(), k.clone(), v.clone(),
            is_causal=False,
            per_block_mean=True
        )

        # Extract captured delta_s
        real_delta_s = captured_data['real']['delta_s']

        if real_delta_s is not None:
            print(f"   ✅ Real delta_s: {real_delta_s.shape} {real_delta_s.dtype}")
            print(f"      Range: [{real_delta_s.min():.6f}, {real_delta_s.max():.6f}]")
            print(f"      Mean: {real_delta_s.mean():.6f}, Std: {real_delta_s.std():.6f}")
        else:
            print("   ❌ Real delta_s is None")

        return real_delta_s, real_output

    finally:
        # Restore original function
        api_module.preprocess_qkv = original_preprocess

def compare_delta_s_detailed(triton_delta_s, real_delta_s):
    """Perform detailed comparison of delta_s values."""
    print("\n🔬 Detailed Delta_s Comparison")
    print("=" * 60)

    if triton_delta_s is None or real_delta_s is None:
        print("❌ Cannot compare: One or both delta_s values are None")
        return 0.0

    # Basic shape and dtype comparison
    print("📊 Basic Comparison:")
    print(f"   Triton shape: {triton_delta_s.shape}")
    print(f"   Real shape:   {real_delta_s.shape}")
    print(f"   Shapes match: {'✅' if triton_delta_s.shape == real_delta_s.shape else '❌'}")

    print(f"   Triton dtype: {triton_delta_s.dtype}")
    print(f"   Real dtype:   {real_delta_s.dtype}")
    print(f"   Dtypes match: {'✅' if triton_delta_s.dtype == real_delta_s.dtype else '❌'}")

    if triton_delta_s.shape != real_delta_s.shape:
        print("❌ Cannot compare: Shape mismatch")
        return 0.0

    # Convert to same precision for comparison
    triton_float = triton_delta_s.float()
    real_float = real_delta_s.float()

    # Statistical comparison
    print("\n📊 Statistical Comparison:")
    print(f"   Triton - Mean: {triton_float.mean():.8f}, Std: {triton_float.std():.8f}")
    print(f"   Real   - Mean: {real_float.mean():.8f}, Std: {real_float.std():.8f}")

    mean_diff = abs(triton_float.mean() - real_float.mean()).item()
    std_diff = abs(triton_float.std() - real_float.std()).item()
    print(f"   Mean difference: {mean_diff:.8e}")
    print(f"   Std difference:  {std_diff:.8e}")

    # Element-wise comparison
    print("\n📊 Element-wise Comparison:")
    diff = triton_float - real_float
    max_abs_diff = diff.abs().max().item()
    mean_abs_diff = diff.abs().mean().item()
    mse = (diff ** 2).mean().item()

    print(f"   Max absolute difference: {max_abs_diff:.8e}")
    print(f"   Mean absolute difference: {mean_abs_diff:.8e}")
    print(f"   Mean squared error: {mse:.8e}")

    # Relative differences
    denom = (triton_float.abs() + real_float.abs()) / 2.0 + 1e-8
    rel_diff = diff.abs() / denom
    max_rel_diff = rel_diff.max().item()
    mean_rel_diff = rel_diff.mean().item()

    print(f"   Max relative difference: {max_rel_diff:.8e}")
    print(f"   Mean relative difference: {mean_rel_diff:.8e}")

    # Cosine similarity
    cos_sim = F.cosine_similarity(triton_float.flatten(), real_float.flatten(), dim=0).item()
    print(f"   Cosine similarity: {cos_sim:.12f}")

    # Correlation
    triton_flat = triton_float.flatten()
    real_flat = real_float.flatten()
    correlation = torch.corrcoef(torch.stack([triton_flat, real_flat]))[0, 1].item()
    print(f"   Pearson correlation: {correlation:.12f}")

    # Grade the comparison
    print("\n🎯 Comparison Grade:")
    if max_abs_diff < 1e-6:
        grade = "🎉 IDENTICAL (< 1e-6 difference)"
    elif max_abs_diff < 1e-5:
        grade = "🔥 NEARLY IDENTICAL (< 1e-5 difference)"
    elif max_abs_diff < 1e-4:
        grade = "✅ EXCELLENT (< 1e-4 difference)"
    elif max_abs_diff < 1e-3:
        grade = "✅ VERY GOOD (< 1e-3 difference)"
    elif max_abs_diff < 1e-2:
        grade = "⚠️ GOOD (< 1e-2 difference)"
    else:
        grade = "❌ SIGNIFICANT DIFFERENCES"

    print(f"   {grade}")

    return cos_sim

def analyze_delta_s_computation(q, k, v):
    """Analyze how delta_s is computed in both implementations."""
    print("\n🧮 Delta_s Computation Analysis")
    print("=" * 60)

    print("📊 Algorithm Overview:")
    print("   delta_s = qm @ k^T")
    print("   Where qm is query mean (per-block or global)")
    print("   This correction compensates for QK smoothing bias")

    # Manual computation to verify understanding
    print("\n📊 Manual Computation for Verification:")

    # Simulate the preprocessing steps
    B, H, N, D = q.shape
    print(f"   Input shape: {q.shape}")

    # K centering (both implementations do this)
    k_centered = k - k.mean(dim=-2, keepdim=True)
    print(f"   K centered: range [{k_centered.min():.6f}, {k_centered.max():.6f}]")

    # Per-block mean computation (GROUP_SIZE = 128)
    GROUP_SIZE = 128
    num_groups = N // GROUP_SIZE
    if N % GROUP_SIZE != 0:
        # Would need padding in real implementation
        pad_len = GROUP_SIZE - (N % GROUP_SIZE)
        q_padded = F.pad(q, (0, 0, 0, pad_len), value=0)
        k_padded = F.pad(k_centered, (0, 0, 0, pad_len), value=0)
        N_padded = N + pad_len
        num_groups = N_padded // GROUP_SIZE
    else:
        q_padded = q
        k_padded = k_centered
        N_padded = N

    print(f"   Group size: {GROUP_SIZE}, Num groups: {num_groups}")
    print(f"   Padded length: {N_padded}")

    # Compute per-block means
    q_reshaped = q_padded.view(B, H, num_groups, GROUP_SIZE, D)
    qm = q_reshaped.mean(dim=3)  # [B, H, num_groups, D]
    print(f"   Query mean shape: {qm.shape}")
    print(f"   Query mean range: [{qm.min():.6f}, {qm.max():.6f}]")

    # Compute delta_s manually
    k_transposed = k_padded.transpose(-2, -1)  # [B, H, D, N_padded]
    delta_s_manual = torch.matmul(qm, k_transposed)  # [B, H, num_groups, N_padded]
    delta_s_manual = delta_s_manual.to(torch.float32)

    if N_padded > N:
        # Trim padding
        delta_s_manual = delta_s_manual[..., :N]

    print(f"   Manual delta_s shape: {delta_s_manual.shape}")
    print(f"   Manual delta_s range: [{delta_s_manual.min():.6f}, {delta_s_manual.max():.6f}]")

    return delta_s_manual

def test_delta_s_comparison():
    """Main comparison test."""
    print("🎯 Delta_s QK Smoothing Comparison")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False

    device = torch.device('cuda')
    print(f"🖥️  GPU: {torch.cuda.get_device_name(0)}")

    # Test configurations
    test_configs = [
        {"B": 1, "H": 8, "N": 256, "D": 64, "name": "Small"},
        {"B": 2, "H": 30, "N": 1024, "D": 64, "name": "CogVideoX-like"},
    ]

    overall_similarities = []

    for config in test_configs:
        B, H, N, D = config["B"], config["H"], config["N"], config["D"]
        name = config["name"]

        print(f"\n🧪 Testing {name} Configuration: {B}×{H}×{N}×{D}")
        print("-" * 60)

        torch.manual_seed(42)
        dtype = torch.float16

        # Create test tensors
        q = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.01
        k = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.01
        v = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.01

        try:
            # Capture delta_s from both implementations
            triton_delta_s, triton_output = capture_triton_delta_s(q, k, v)
            real_delta_s, real_output = capture_real_kernel_delta_s(q, k, v)

            # Compare delta_s values
            delta_s_similarity = compare_delta_s_detailed(triton_delta_s, real_delta_s)
            overall_similarities.append(delta_s_similarity)

            # Also analyze computation manually
            manual_delta_s = analyze_delta_s_computation(q, k, v)

            # Compare manual vs implementations
            print(f"\n📊 Manual vs Implementation Comparison:")
            if triton_delta_s is not None:
                manual_triton_sim = F.cosine_similarity(
                    manual_delta_s.flatten(), triton_delta_s.flatten(), dim=0
                ).item()
                print(f"   Manual vs Triton: {manual_triton_sim:.8f}")

            if real_delta_s is not None:
                manual_real_sim = F.cosine_similarity(
                    manual_delta_s.flatten(), real_delta_s.flatten(), dim=0
                ).item()
                print(f"   Manual vs Real: {manual_real_sim:.8f}")

        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()

    # Overall summary
    print(f"\n{'='*70}")
    print("📈 DELTA_S COMPARISON SUMMARY")
    print(f"{'='*70}")

    if overall_similarities:
        avg_similarity = sum(overall_similarities) / len(overall_similarities)
        min_sim = min(overall_similarities)
        max_sim = max(overall_similarities)

        print(f"📊 Delta_s Similarity Statistics:")
        print(f"   Average similarity: {avg_similarity:.8f}")
        print(f"   Minimum similarity: {min_sim:.8f}")
        print(f"   Maximum similarity: {max_sim:.8f}")

        if avg_similarity >= 0.9999999:
            conclusion = "🎉 IDENTICAL: Delta_s computation is identical in both implementations!"
        elif avg_similarity >= 0.999999:
            conclusion = "🔥 NEARLY IDENTICAL: Negligible differences (likely floating-point precision)"
        elif avg_similarity >= 0.9999:
            conclusion = "✅ EXCELLENT: Very high agreement in delta_s computation"
        elif avg_similarity >= 0.99:
            conclusion = "✅ GOOD: High agreement with minor differences"
        else:
            conclusion = "⚠️ DIFFERENCES: Significant differences in delta_s computation detected"

        print(f"\n🎯 CONCLUSION: {conclusion}")

        return avg_similarity >= 0.9999
    else:
        print("❌ No successful comparisons")
        return False

def main():
    """Main function."""
    setup_paths()

    print("🎯 SageAttention3 Delta_s QK Smoothing Comparison")
    print("Comparing QK smoothing correction values between implementations")
    print()

    success = test_delta_s_comparison()

    if success:
        print(f"\n✅ DELTA_S ANALYSIS COMPLETE!")
        print("QK smoothing algorithms are well-aligned between implementations.")
    else:
        print(f"\n❌ DELTA_S ANALYSIS SHOWS DIFFERENCES")
        print("QK smoothing implementations may have algorithmic differences.")

    return success

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Analysis interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)