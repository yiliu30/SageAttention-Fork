#!/usr/bin/env python3
"""
Triton Kernel Delta_s Indexing Debug Script
===========================================

This script specifically debugs the delta_s indexing and memory access patterns
in the SageAttention3 Triton kernel. Since delta_s computation is correct (verified
by debug_delta_s_computation.py), the issue must be in how the Triton kernel
loads and applies these correction values.

Key areas to investigate:
1. group_id calculation: group_id = q_start // GROUP_SIZE
2. Delta_s pointer arithmetic and memory access patterns
3. Bounds checking and memory safety
4. Stride calculations and tensor layout assumptions
5. Tile-to-global coordinate mapping

The script instruments critical sections of the Triton kernel with debug outputs
and compares the loaded delta_s values with expected PyTorch reference values.
"""

import torch
import sys
import os
import numpy as np
from typing import Tuple, Dict

# Add the sage3 implementation directory to the path
sys.path.insert(0, '/mnt/disk1/yiliu7/SageAttention-Fork/tasks/sage3_impl_torch')

from sageattn3_torch import apply_qk_smoothing, sageattn3_torch
from sageattn3_torch_triton import sageattn3_torch_triton

def create_debug_tensors(B: int, H: int, N: int, D: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create reproducible debug tensors for consistent testing."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float16

    # Use fixed seed for reproducible results
    torch.manual_seed(42)

    q = torch.randn(B, H, N, D, dtype=dtype, device=device)
    k = torch.randn(B, H, N, D, dtype=dtype, device=device)
    v = torch.randn(B, H, N, D, dtype=dtype, device=device)

    return q, k, v

def debug_delta_s_loading_simple():
    """Test delta_s loading with simple, debuggable configurations."""
    print("=" * 80)
    print("SIMPLE DELTA_S LOADING DEBUG")
    print("=" * 80)

    # Start with very simple case: 1 batch, 1 head, 256 tokens (2 groups)
    B, H, N, D = 1, 1, 256, 64
    print(f"Testing configuration: B={B}, H={H}, N={N}, D={D}")
    print(f"Expected groups: {(N + 127) // 128}")

    q, k, v = create_debug_tensors(B, H, N, D)

    # Compute reference delta_s from PyTorch
    q_smoothed, k_smoothed, delta_s_ref = apply_qk_smoothing(q, k)
    print(f"Reference delta_s shape: {delta_s_ref.shape}")
    print(f"Reference delta_s stats: min={delta_s_ref.min():.4f}, max={delta_s_ref.max():.4f}, mean={delta_s_ref.mean():.4f}")

    # Let's examine the delta_s structure in detail
    print("\nDetailed delta_s analysis:")
    for group in range(delta_s_ref.shape[2]):  # num_groups
        group_data = delta_s_ref[0, 0, group, :]  # [N]
        print(f"  Group {group}: shape={group_data.shape}, min={group_data.min():.4f}, max={group_data.max():.4f}")

        # Show first few values for debugging
        if N <= 256:  # Only for small sequences
            first_few = group_data[:min(8, N)].cpu().numpy()
            print(f"    First 8 values: {first_few}")

    # Test PyTorch reference implementation
    try:
        print("\nTesting PyTorch reference implementation...")
        output_ref = sageattn3_torch(
            q=q, k=k, v=v,
            tensor_layout="HND",
            is_causal=False,
            per_block_mean=True,
            tile_size_q=128,
            tile_size_k=128
        )
        print(f"✅ PyTorch reference: output shape {output_ref.shape}")
    except Exception as e:
        print(f"❌ PyTorch reference failed: {e}")
        return

    # Test Triton implementation with debug mode
    try:
        print("\nTesting Triton implementation with debug...")
        output_triton = sageattn3_torch_triton(
            q=q, k=k, v=v,
            tensor_layout="HND",
            is_causal=False,
            per_block_mean=True,
            tile_size_q=128,
            tile_size_k=128,
            debug=True  # Enable debug mode
        )
        print(f"✅ Triton implementation: output shape {output_triton.shape}")

        # Compare outputs
        diff = (output_ref - output_triton).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        cosine_sim = torch.nn.functional.cosine_similarity(
            output_ref.flatten(), output_triton.flatten(), dim=0
        ).item()

        print(f"Output comparison:")
        print(f"  Max difference: {max_diff:.6f}")
        print(f"  Mean difference: {mean_diff:.6f}")
        print(f"  Cosine similarity: {cosine_sim:.6f}")

        if cosine_sim < 0.99:
            print("⚠️  Low cosine similarity - potential correctness issue!")
        else:
            print("✅ High cosine similarity")

    except Exception as e:
        print(f"❌ Triton implementation failed: {e}")
        import traceback
        traceback.print_exc()

def debug_delta_s_loading_cogvideox():
    """Test delta_s loading with CogVideoX-like configurations."""
    print("\n" + "=" * 80)
    print("COGVIDEOX DELTA_S LOADING DEBUG")
    print("=" * 80)

    # CogVideoX problematic configuration
    B, H, N, D = 2, 30, 1024, 64  # This is known to fail
    print(f"Testing CogVideoX configuration: B={B}, H={H}, N={N}, D={D}")
    print(f"Expected groups: {(N + 127) // 128}")

    q, k, v = create_debug_tensors(B, H, N, D)

    # Compute reference delta_s
    q_smoothed, k_smoothed, delta_s_ref = apply_qk_smoothing(q, k)
    print(f"Reference delta_s shape: {delta_s_ref.shape}")

    # Analyze the delta_s pattern for potential indexing issues
    print("\nAnalyzing delta_s pattern for indexing issues:")

    # Check for any unusual patterns that might cause Triton issues
    for b in range(min(B, 2)):  # Check first 2 batches
        for h in range(min(H, 3)):  # Check first 3 heads
            head_delta = delta_s_ref[b, h]  # [num_groups, N]
            head_min = head_delta.min().item()
            head_max = head_delta.max().item()
            head_mean = head_delta.mean().item()

            print(f"  Batch {b}, Head {h}: min={head_min:.4f}, max={head_max:.4f}, mean={head_mean:.4f}")

            # Check for extreme values that might cause numerical issues
            if abs(head_min) > 100 or abs(head_max) > 100:
                print(f"    ⚠️  Extreme values detected!")

    print("\nTesting with per_block_mean=False (should work)...")
    try:
        output_no_pbm = sageattn3_torch_triton(
            q=q, k=k, v=v,
            tensor_layout="HND",
            is_causal=False,
            per_block_mean=False,  # This should work
            tile_size_q=128,
            tile_size_k=128,
            debug=True
        )
        print(f"✅ per_block_mean=False: output shape {output_no_pbm.shape}")
    except Exception as e:
        print(f"❌ per_block_mean=False failed: {e}")

    print("\nTesting with per_block_mean=True (known to have issues)...")
    try:
        output_pbm = sageattn3_torch_triton(
            q=q, k=k, v=v,
            tensor_layout="HND",
            is_causal=False,
            per_block_mean=True,  # This may fail or produce wrong results
            tile_size_q=128,
            tile_size_k=128,
            debug=True
        )
        print(f"✅ per_block_mean=True: output shape {output_pbm.shape}")

        # Compare the two outputs to see the difference
        if 'output_no_pbm' in locals():
            diff = (output_no_pbm - output_pbm).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()

            print(f"Difference between per_block_mean=False and True:")
            print(f"  Max difference: {max_diff:.6f}")
            print(f"  Mean difference: {mean_diff:.6f}")

            if max_diff > 0.1:  # Significant difference suggests bug
                print("⚠️  Large difference suggests per_block_mean bug!")
            else:
                print("✅ Small difference suggests per_block_mean working correctly")

    except Exception as e:
        print(f"❌ per_block_mean=True failed: {e}")
        import traceback
        traceback.print_exc()

def debug_tile_indexing():
    """Debug tile-to-global coordinate mapping for delta_s access."""
    print("\n" + "=" * 80)
    print("TILE INDEXING DEBUG")
    print("=" * 80)

    # Test with configurations that span multiple tiles
    test_configs = [
        (1, 4, 256, 64, "2 groups, 2 Q tiles, 2 K tiles"),
        (1, 4, 384, 64, "3 groups, 3 Q tiles, 3 K tiles"),
        (1, 4, 512, 64, "4 groups, 4 Q tiles, 4 K tiles"),
    ]

    for B, H, N, D, description in test_configs:
        print(f"\n{description}: B={B}, H={H}, N={N}, D={D}")

        q, k, v = create_debug_tensors(B, H, N, D)
        q_smoothed, k_smoothed, delta_s_ref = apply_qk_smoothing(q, k)

        num_groups = delta_s_ref.shape[2]
        tile_size = 128
        num_q_tiles = (N + tile_size - 1) // tile_size
        num_k_tiles = (N + tile_size - 1) // tile_size

        print(f"  Groups: {num_groups}, Q tiles: {num_q_tiles}, K tiles: {num_k_tiles}")

        # Simulate the Triton kernel's tile iteration
        print("  Simulating Triton tile iteration:")
        for q_tile_idx in range(num_q_tiles):
            q_start = q_tile_idx * tile_size
            q_end = min(q_start + tile_size, N)

            # This is the critical calculation that might be wrong in Triton
            group_id = q_start // 128  # GROUP_SIZE = 128

            print(f"    Q tile {q_tile_idx}: q_start={q_start}, q_end={q_end}, group_id={group_id}")

            # Validate that group_id is within bounds
            if group_id >= num_groups:
                print(f"      ❌ group_id {group_id} >= num_groups {num_groups} - BOUNDS ERROR!")
            else:
                print(f"      ✅ group_id {group_id} < num_groups {num_groups} - OK")

                # Check the delta_s values that would be accessed
                delta_slice = delta_s_ref[0, 0, group_id, q_start:q_end]
                print(f"      Delta_s slice shape: {delta_slice.shape}, range: [{delta_slice.min():.4f}, {delta_slice.max():.4f}]")

def debug_memory_layout():
    """Debug memory layout assumptions in delta_s tensor."""
    print("\n" + "=" * 80)
    print("MEMORY LAYOUT DEBUG")
    print("=" * 80)

    B, H, N, D = 1, 2, 384, 64  # 3 groups
    q, k, v = create_debug_tensors(B, H, N, D)
    q_smoothed, k_smoothed, delta_s_ref = apply_qk_smoothing(q, k)

    print(f"Delta_s shape: {delta_s_ref.shape}")
    print(f"Delta_s strides: {delta_s_ref.stride()}")
    print(f"Delta_s is contiguous: {delta_s_ref.is_contiguous()}")

    # Check memory layout expectations
    expected_stride_n = 1
    expected_stride_g = N  # num_groups dimension
    expected_stride_h = delta_s_ref.shape[2] * N  # heads dimension
    expected_stride_b = H * delta_s_ref.shape[2] * N  # batch dimension

    actual_strides = delta_s_ref.stride()
    print(f"\nStride analysis:")
    print(f"  Expected: batch={expected_stride_b}, head={expected_stride_h}, group={expected_stride_g}, seq={expected_stride_n}")
    print(f"  Actual:   batch={actual_strides[0]}, head={actual_strides[1]}, group={actual_strides[2]}, seq={actual_strides[3]}")

    strides_match = (
        actual_strides[0] == expected_stride_b and
        actual_strides[1] == expected_stride_h and
        actual_strides[2] == expected_stride_g and
        actual_strides[3] == expected_stride_n
    )

    print(f"  Strides match expectation: {'✅' if strides_match else '❌'}")

    if not strides_match:
        print("  ⚠️  Stride mismatch might cause Triton indexing errors!")

def main():
    """Main debug function."""
    print("SageAttention3 Triton Delta_s Indexing Debug")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"PyTorch version: {torch.__version__}")

    # Run debug tests in order of complexity
    debug_delta_s_loading_simple()
    debug_delta_s_loading_cogvideox()
    debug_tile_indexing()
    debug_memory_layout()

    print("\n" + "=" * 80)
    print("DEBUGGING COMPLETE")
    print("=" * 80)
    print("If per_block_mean=True shows issues above, the problem is in Triton kernel indexing.")
    print("Check the Triton kernel code for:")
    print("1. group_id calculation: group_id = q_start // 128")
    print("2. Delta_s memory access: delta_s_ptr + group_id * delta_s_stride_g + ...")
    print("3. Bounds checking for group_id")
    print("4. Tensor stride assumptions")

if __name__ == "__main__":
    main()