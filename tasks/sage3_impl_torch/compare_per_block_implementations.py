#!/usr/bin/env python3
"""
PyTorch vs Triton Implementation Comparison
===========================================

This script performs detailed side-by-side comparison between PyTorch reference
and Triton implementations to identify the exact source of the per_block_mean bug.

The debug_triton_delta_indexing.py script revealed a significant accuracy difference
when per_block_mean=True, with cosine similarity dropping to 98.68% on simple cases
and large differences (max 1.95) on CogVideoX configurations.

This script will:
1. Run both implementations with identical inputs
2. Extract and compare intermediate values (delta_s, attention scores)
3. Perform per-tile analysis to isolate the problematic tile
4. Generate detailed diagnostics to guide the bug fix
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

def create_test_case(B: int, H: int, N: int, D: int, seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create reproducible test tensors."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float16

    torch.manual_seed(seed)
    q = torch.randn(B, H, N, D, dtype=dtype, device=device) * 0.1  # Smaller values for stability
    k = torch.randn(B, H, N, D, dtype=dtype, device=device) * 0.1
    v = torch.randn(B, H, N, D, dtype=dtype, device=device) * 0.1

    return q, k, v

def compare_delta_s_values():
    """Compare delta_s computation between implementations."""
    print("=" * 80)
    print("DELTA_S COMPUTATION COMPARISON")
    print("=" * 80)

    B, H, N, D = 1, 2, 256, 64  # Simple case that shows the bug
    q, k, v = create_test_case(B, H, N, D)

    print(f"Test configuration: B={B}, H={H}, N={N}, D={D}")

    # Compute delta_s using PyTorch reference
    q_smoothed_ref, k_smoothed_ref, delta_s_ref = apply_qk_smoothing(q, k)
    print(f"Reference delta_s shape: {delta_s_ref.shape}")
    print(f"Reference delta_s stats: min={delta_s_ref.min():.6f}, max={delta_s_ref.max():.6f}")

    # Run Triton with debug to see what delta_s it uses
    print("\nRunning Triton implementation...")
    try:
        output_triton = sageattn3_torch_triton(
            q=q, k=k, v=v,
            tensor_layout="HND",
            is_causal=False,
            per_block_mean=True,
            tile_size_q=128,
            tile_size_k=128,
            debug=True
        )

        # The Triton implementation should use the same delta_s as reference
        # If it doesn't match, that's where the bug is
        print("✅ Triton implementation ran successfully")

    except Exception as e:
        print(f"❌ Triton implementation failed: {e}")
        import traceback
        traceback.print_exc()

def detailed_accuracy_comparison():
    """Perform detailed accuracy comparison with various configurations."""
    print("\n" + "=" * 80)
    print("DETAILED ACCURACY COMPARISON")
    print("=" * 80)

    test_configs = [
        (1, 1, 128, 64, "Single group"),
        (1, 1, 256, 64, "Two groups"),
        (1, 4, 256, 64, "Multiple heads"),
        (2, 4, 256, 64, "Multiple batches"),
        (1, 1, 384, 64, "Three groups"),
        (2, 30, 512, 64, "CogVideoX-like small"),
        (1, 8, 1024, 64, "Long sequence"),
    ]

    results = {}

    for B, H, N, D, description in test_configs:
        print(f"\n{description}: B={B}, H={H}, N={N}, D={D}")

        q, k, v = create_test_case(B, H, N, D)

        try:
            # PyTorch reference
            output_ref = sageattn3_torch(
                q=q, k=k, v=v,
                tensor_layout="HND",
                is_causal=False,
                per_block_mean=True,
                tile_size_q=128,
                tile_size_k=128
            )

            # Triton implementation
            output_triton = sageattn3_torch_triton(
                q=q, k=k, v=v,
                tensor_layout="HND",
                is_causal=False,
                per_block_mean=True,
                tile_size_q=128,
                tile_size_k=128,
                debug=False
            )

            # Compare outputs
            diff = (output_ref - output_triton).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            cosine_sim = torch.nn.functional.cosine_similarity(
                output_ref.flatten(), output_triton.flatten(), dim=0
            ).item()

            results[description] = {
                'max_diff': max_diff,
                'mean_diff': mean_diff,
                'cosine_sim': cosine_sim,
                'shape': output_ref.shape
            }

            print(f"  Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}, Cosine sim: {cosine_sim:.6f}")

            if cosine_sim < 0.99:
                print("  ⚠️  LOW ACCURACY - significant bug!")
            elif cosine_sim < 0.999:
                print("  ⚠️  Moderate accuracy issue")
            else:
                print("  ✅ Good accuracy")

        except Exception as e:
            print(f"  ❌ Failed: {e}")
            results[description] = {'error': str(e)}

    return results

def analyze_per_tile_differences():
    """Analyze differences on a per-tile basis to identify problematic tiles."""
    print("\n" + "=" * 80)
    print("PER-TILE DIFFERENCE ANALYSIS")
    print("=" * 80)

    # Use a configuration known to have issues
    B, H, N, D = 1, 4, 512, 64  # 4 tiles of 128 each
    tile_size = 128
    q, k, v = create_test_case(B, H, N, D)

    print(f"Configuration: B={B}, H={H}, N={N}, D={D}")
    print(f"Tile size: {tile_size}, Number of tiles: {N // tile_size}")

    # Get full outputs
    output_ref = sageattn3_torch(
        q=q, k=k, v=v,
        tensor_layout="HND",
        is_causal=False,
        per_block_mean=True,
        tile_size_q=tile_size,
        tile_size_k=tile_size
    )

    output_triton = sageattn3_torch_triton(
        q=q, k=k, v=v,
        tensor_layout="HND",
        is_causal=False,
        per_block_mean=True,
        tile_size_q=tile_size,
        tile_size_k=tile_size,
        debug=False
    )

    # Analyze differences per tile
    num_tiles = N // tile_size
    print(f"\nPer-tile accuracy analysis:")

    for tile_idx in range(num_tiles):
        tile_start = tile_idx * tile_size
        tile_end = tile_start + tile_size

        # Extract tile outputs
        ref_tile = output_ref[:, :, tile_start:tile_end, :]
        triton_tile = output_triton[:, :, tile_start:tile_end, :]

        # Compute tile-specific metrics
        tile_diff = (ref_tile - triton_tile).abs()
        tile_max_diff = tile_diff.max().item()
        tile_mean_diff = tile_diff.mean().item()

        tile_cosine = torch.nn.functional.cosine_similarity(
            ref_tile.flatten(), triton_tile.flatten(), dim=0
        ).item()

        print(f"  Tile {tile_idx} (pos {tile_start}-{tile_end}): "
              f"max_diff={tile_max_diff:.6f}, mean_diff={tile_mean_diff:.6f}, cosine={tile_cosine:.6f}")

        if tile_cosine < 0.99:
            print(f"    ❌ PROBLEMATIC TILE - major accuracy issue!")
        elif tile_cosine < 0.999:
            print(f"    ⚠️  Moderate issue in this tile")

def test_with_different_group_alignments():
    """Test with sequences that create different group/tile alignment patterns."""
    print("\n" + "=" * 80)
    print("GROUP/TILE ALIGNMENT TESTING")
    print("=" * 80)

    # Test various sequence lengths that create different alignment patterns
    test_lengths = [
        (127, "Just under 1 group"),
        (128, "Exactly 1 group"),
        (129, "Just over 1 group"),
        (255, "Just under 2 groups"),
        (256, "Exactly 2 groups"),
        (257, "Just over 2 groups"),
        (383, "Just under 3 groups"),
        (384, "Exactly 3 groups"),
        (385, "Just over 3 groups"),
        (640, "5 groups (non-tile boundary)"),
        (768, "6 groups (tile boundary)"),
    ]

    print("Testing different sequence lengths for alignment issues:")
    print("Groups are 128 tokens, tiles are 128 tokens")

    for N, description in test_lengths:
        B, H, D = 1, 2, 64
        q, k, v = create_test_case(B, H, N, D)

        try:
            output_ref = sageattn3_torch(q=q, k=k, v=v, tensor_layout="HND", is_causal=False, per_block_mean=True)
            output_triton = sageattn3_torch_triton(q=q, k=k, v=v, tensor_layout="HND", is_causal=False, per_block_mean=True)

            cosine_sim = torch.nn.functional.cosine_similarity(
                output_ref.flatten(), output_triton.flatten(), dim=0
            ).item()

            num_groups = (N + 127) // 128
            num_tiles = (N + 127) // 128  # Same as groups since tile_size = group_size = 128

            print(f"  N={N:3d} ({num_groups} groups, {num_tiles} tiles): {description:25s} cosine={cosine_sim:.6f}")

            if cosine_sim < 0.99:
                print(f"        ❌ ACCURACY ISSUE at N={N}")

        except Exception as e:
            print(f"  N={N:3d}: ❌ Failed - {e}")

def main():
    """Main comparison function."""
    print("SageAttention3 PyTorch vs Triton Detailed Comparison")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"PyTorch version: {torch.__version__}")

    # Run all comparison tests
    compare_delta_s_values()
    accuracy_results = detailed_accuracy_comparison()
    analyze_per_tile_differences()
    test_with_different_group_alignments()

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY AND DEBUGGING GUIDANCE")
    print("=" * 80)

    if accuracy_results:
        low_accuracy_cases = [desc for desc, result in accuracy_results.items()
                             if isinstance(result, dict) and result.get('cosine_sim', 1.0) < 0.99]

        if low_accuracy_cases:
            print("❌ Cases with significant accuracy issues:")
            for case in low_accuracy_cases:
                result = accuracy_results[case]
                print(f"  - {case}: cosine_sim={result['cosine_sim']:.6f}, max_diff={result['max_diff']:.6f}")

            print("\nLikely causes to investigate in Triton kernel:")
            print("1. Incorrect delta_s indexing in the attention computation")
            print("2. Wrong memory stride calculations")
            print("3. Tile boundary handling issues")
            print("4. Group ID calculation errors")
            print("5. Delta_s broadcasting or addition logic")

        else:
            print("✅ All test cases show good accuracy!")
            print("The bug might be more subtle or configuration-specific.")

    # Save detailed results
    torch.save(accuracy_results, 'pytorch_triton_comparison_results.pt')
    print(f"\nDetailed results saved to pytorch_triton_comparison_results.pt")

if __name__ == "__main__":
    main()