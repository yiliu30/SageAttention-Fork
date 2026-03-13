#!/usr/bin/env python3
"""
Delta_s Computation Validation Script
=====================================

This script validates the delta_s computation used in SageAttention3's QK smoothing
algorithm, specifically testing CogVideoX-like configurations that expose the
per_block_mean=True correctness issue.

The delta_s correction terms are computed by apply_qk_smoothing() and have shape:
[B, H, num_groups, N] where num_groups = seq_len // 128

This script tests:
1. Delta_s shape validation for various sequence lengths
2. Grouping logic (128-token blocks)
3. Padding behavior for sequences not divisible by 128
4. Value distribution and outlier handling
5. CogVideoX-specific configurations that cause issues
"""

import torch
import sys
import os
import numpy as np
from typing import Tuple, Dict, List

# Add the sage3 implementation directory to the path
sys.path.insert(0, '/mnt/disk1/yiliu7/SageAttention-Fork/tasks/sage3_impl_torch')

from sageattn3_torch import apply_qk_smoothing

def test_delta_s_shapes():
    """Test delta_s computation for various sequence lengths."""
    print("=" * 80)
    print("DELTA_S SHAPE VALIDATION")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float16

    # Test configurations: (B, H, N, D, description)
    test_configs = [
        (1, 8, 128, 64, "Simple 128-token sequence (exact group)"),
        (1, 8, 256, 64, "256-token sequence (2 groups)"),
        (1, 8, 384, 64, "384-token sequence (3 groups)"),
        (1, 8, 127, 64, "127-token sequence (< 128, edge case)"),
        (1, 8, 129, 64, "129-token sequence (> 128, padding needed)"),
        (1, 8, 255, 64, "255-token sequence (padding needed)"),
        (2, 30, 1024, 64, "CogVideoX-like: B=2, H=30, N=1024"),
        (2, 30, 2048, 64, "CogVideoX-like: B=2, H=30, N=2048"),
        (1, 1, 1280, 64, "Long sequence: N=1280 (10 groups)"),
    ]

    results = {}

    for B, H, N, D, description in test_configs:
        print(f"\n{description}")
        print(f"Input shape: [B={B}, H={H}, N={N}, D={D}]")

        # Create random Q and K tensors
        q = torch.randn(B, H, N, D, dtype=dtype, device=device)
        k = torch.randn(B, H, N, D, dtype=dtype, device=device)

        try:
            # Compute delta_s using apply_qk_smoothing
            q_smoothed, k_smoothed, delta_s = apply_qk_smoothing(q, k)

            # Expected delta_s shape: [B, H, num_groups, N]
            num_groups = (N + 127) // 128  # Ceiling division
            expected_shape = (B, H, num_groups, N)

            print(f"Expected delta_s shape: {expected_shape}")
            print(f"Actual delta_s shape:   {delta_s.shape}")

            shape_match = delta_s.shape == expected_shape
            print(f"Shape match: {'✅' if shape_match else '❌'}")

            if shape_match:
                # Analyze delta_s values
                delta_min = delta_s.min().item()
                delta_max = delta_s.max().item()
                delta_mean = delta_s.mean().item()
                delta_std = delta_s.std().item()

                print(f"Delta_s statistics:")
                print(f"  Min: {delta_min:.4f}, Max: {delta_max:.4f}")
                print(f"  Mean: {delta_mean:.4f}, Std: {delta_std:.4f}")

                # Check for NaN or inf values
                has_nan = torch.isnan(delta_s).any().item()
                has_inf = torch.isinf(delta_s).any().item()

                if has_nan or has_inf:
                    print(f"⚠️  Contains NaN: {has_nan}, Inf: {has_inf}")
                else:
                    print("✅ No NaN/Inf values")

                results[description] = {
                    'shape_match': True,
                    'delta_s_shape': delta_s.shape,
                    'stats': {'min': delta_min, 'max': delta_max, 'mean': delta_mean, 'std': delta_std},
                    'has_nan': has_nan,
                    'has_inf': has_inf
                }
            else:
                print(f"❌ Shape mismatch! Expected {expected_shape}, got {delta_s.shape}")
                results[description] = {
                    'shape_match': False,
                    'expected_shape': expected_shape,
                    'actual_shape': delta_s.shape
                }

        except Exception as e:
            print(f"❌ Error computing delta_s: {e}")
            results[description] = {'error': str(e)}

    return results

def test_grouping_logic():
    """Test the 128-token grouping logic in detail."""
    print("\n" + "=" * 80)
    print("GROUPING LOGIC VALIDATION")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float16
    GROUP_SIZE = 128

    # Test edge cases around group boundaries
    test_sequences = [127, 128, 129, 255, 256, 257, 383, 384, 385, 1024, 1280]

    for N in test_sequences:
        B, H, D = 1, 4, 64
        print(f"\nTesting N={N} (expected {(N + GROUP_SIZE - 1) // GROUP_SIZE} groups)")

        q = torch.randn(B, H, N, D, dtype=dtype, device=device)
        k = torch.randn(B, H, N, D, dtype=dtype, device=device)

        try:
            delta_s = apply_qk_smoothing(q, k)[2]  # Third element is delta_s
            num_groups_actual = delta_s.shape[2]
            num_groups_expected = (N + GROUP_SIZE - 1) // GROUP_SIZE

            print(f"  Expected groups: {num_groups_expected}")
            print(f"  Actual groups:   {num_groups_actual}")
            print(f"  Match: {'✅' if num_groups_actual == num_groups_expected else '❌'}")

            # Test the grouping boundary behavior
            if num_groups_actual > 1:
                # Check if different groups have different delta_s values
                group_0 = delta_s[0, 0, 0, :min(GROUP_SIZE, N)]  # First group
                if num_groups_actual > 1:
                    group_1_start = min(GROUP_SIZE, N)
                    group_1_end = min(2 * GROUP_SIZE, N)
                    if group_1_start < N:
                        group_1 = delta_s[0, 0, 1, group_1_start:group_1_end]

                        # They should be different (group-specific smoothing)
                        if group_1.numel() > 0:
                            diff = (group_0[:min(len(group_0), len(group_1))] -
                                   group_1[:min(len(group_0), len(group_1))]).abs().mean()
                            print(f"  Inter-group difference: {diff:.6f} {'✅' if diff > 1e-6 else '⚠️'}")

        except Exception as e:
            print(f"  ❌ Error: {e}")

def test_cogvideox_configurations():
    """Test specific CogVideoX configurations that are known to cause issues."""
    print("\n" + "=" * 80)
    print("COGVIDEOX CONFIGURATION TESTING")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float16

    # CogVideoX configurations from the failing case
    cogvideox_configs = [
        (2, 30, 1024, 64, "CogVideoX 2B typical"),
        (2, 30, 2048, 64, "CogVideoX 2B long"),
        (1, 30, 1024, 64, "CogVideoX single batch"),
        (4, 30, 1024, 64, "CogVideoX larger batch"),
    ]

    for B, H, N, D, description in cogvideox_configs:
        print(f"\n{description}: [B={B}, H={H}, N={N}, D={D}]")

        q = torch.randn(B, H, N, D, dtype=dtype, device=device)
        k = torch.randn(B, H, N, D, dtype=dtype, device=device)

        try:
            delta_s = apply_qk_smoothing(q, k)[2]  # Third element is delta_s

            print(f"✅ Delta_s computed successfully: {delta_s.shape}")

            # Deep analysis for CogVideoX configs
            print("Detailed analysis:")

            # Check per-batch statistics
            for b in range(B):
                batch_delta = delta_s[b]  # [H, num_groups, N]
                batch_min = batch_delta.min().item()
                batch_max = batch_delta.max().item()
                batch_mean = batch_delta.mean().item()

                print(f"  Batch {b}: min={batch_min:.4f}, max={batch_max:.4f}, mean={batch_mean:.4f}")

            # Check per-head statistics
            head_stats = []
            for h in range(min(H, 5)):  # Check first 5 heads
                head_delta = delta_s[:, h]  # [B, num_groups, N]
                head_mean = head_delta.mean().item()
                head_std = head_delta.std().item()
                head_stats.append((head_mean, head_std))

                print(f"  Head {h}: mean={head_mean:.4f}, std={head_std:.4f}")

            # Check for outlier heads (might indicate issues)
            if len(head_stats) > 1:
                means = [s[0] for s in head_stats]
                mean_std = np.std(means)
                if mean_std > 0.1:
                    print(f"⚠️  High variance across heads: {mean_std:.4f}")
                else:
                    print(f"✅ Consistent across heads: {mean_std:.4f}")

        except Exception as e:
            print(f"❌ Failed: {e}")
            import traceback
            traceback.print_exc()

def test_edge_cases():
    """Test edge cases that might cause issues."""
    print("\n" + "=" * 80)
    print("EDGE CASE TESTING")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float16

    edge_cases = [
        # (description, lambda: create_tensors)
        ("Very small sequence (N=1)", lambda: (torch.randn(1, 1, 1, 64, dtype=dtype, device=device),
                                               torch.randn(1, 1, 1, 64, dtype=dtype, device=device))),
        ("Exact group boundary (N=128)", lambda: (torch.randn(1, 1, 128, 64, dtype=dtype, device=device),
                                                  torch.randn(1, 1, 128, 64, dtype=dtype, device=device))),
        ("Just over boundary (N=129)", lambda: (torch.randn(1, 1, 129, 64, dtype=dtype, device=device),
                                                torch.randn(1, 1, 129, 64, dtype=dtype, device=device))),
        ("Zero tensors", lambda: (torch.zeros(1, 4, 256, 64, dtype=dtype, device=device),
                                  torch.zeros(1, 4, 256, 64, dtype=dtype, device=device))),
        ("One tensors", lambda: (torch.ones(1, 4, 256, 64, dtype=dtype, device=device),
                                 torch.ones(1, 4, 256, 64, dtype=dtype, device=device))),
        ("Very large values", lambda: (torch.randn(1, 4, 256, 64, dtype=dtype, device=device) * 100,
                                       torch.randn(1, 4, 256, 64, dtype=dtype, device=device) * 100)),
        ("Very small values", lambda: (torch.randn(1, 4, 256, 64, dtype=dtype, device=device) * 0.001,
                                       torch.randn(1, 4, 256, 64, dtype=dtype, device=device) * 0.001)),
    ]

    for description, tensor_fn in edge_cases:
        print(f"\n{description}:")

        try:
            q, k = tensor_fn()
            print(f"  Input shapes: q={q.shape}, k={k.shape}")

            delta_s = apply_qk_smoothing(q, k)[2]  # Third element is delta_s
            print(f"  ✅ Success: delta_s shape = {delta_s.shape}")

            # Check for problematic values
            has_nan = torch.isnan(delta_s).any().item()
            has_inf = torch.isinf(delta_s).any().item()
            min_val = delta_s.min().item()
            max_val = delta_s.max().item()

            print(f"  Values: min={min_val:.6f}, max={max_val:.6f}")
            if has_nan or has_inf:
                print(f"  ❌ Contains NaN: {has_nan}, Inf: {has_inf}")
            else:
                print(f"  ✅ No NaN/Inf values")

        except Exception as e:
            print(f"  ❌ Failed: {e}")

def main():
    """Main test function."""
    print("SageAttention3 Delta_s Computation Validation")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"PyTorch version: {torch.__version__}")

    # Run all test suites
    shape_results = test_delta_s_shapes()
    test_grouping_logic()
    test_cogvideox_configurations()
    test_edge_cases()

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    passed = sum(1 for r in shape_results.values() if r.get('shape_match', False))
    total = len(shape_results)

    print(f"Shape validation tests: {passed}/{total} passed")

    if passed == total:
        print("✅ All delta_s computation tests passed!")
        print("The issue is likely in the Triton kernel's usage of delta_s, not the computation itself.")
    else:
        print("❌ Some delta_s computation tests failed.")
        print("The issue might be in the apply_qk_smoothing function itself.")

    # Save results for further analysis
    torch.save(shape_results, 'debug_delta_s_results.pt')
    print(f"\nResults saved to debug_delta_s_results.pt")

if __name__ == "__main__":
    main()