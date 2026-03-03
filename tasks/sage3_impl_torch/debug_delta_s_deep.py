#!/usr/bin/env python3
"""
Deep debugging of delta_s application in Triton kernel
"""

import torch
import sys
import os
sys.path.insert(0, '/mnt/disk1/yiliu7/SageAttention-Fork/tasks/sage3_impl_torch')

from sageattn3_torch import apply_qk_smoothing, sageattn3_torch
from sageattn3_torch_triton import sageattn3_torch_triton

def detailed_delta_s_analysis():
    """Compare delta_s usage between PyTorch and Triton implementations."""
    device = torch.device('cuda')
    dtype = torch.float16

    # Use simple case that reproduces the bug
    torch.manual_seed(42)
    B, H, N, D = 1, 1, 256, 64

    q = torch.randn(B, H, N, D, dtype=dtype, device=device)
    k = torch.randn(B, H, N, D, dtype=dtype, device=device)
    v = torch.randn(B, H, N, D, dtype=dtype, device=device)

    print(f"Debugging configuration: B={B}, H={H}, N={N}, D={D}")

    # Get delta_s from PyTorch reference
    q_smoothed_ref, k_smoothed_ref, delta_s_ref = apply_qk_smoothing(q, k)
    print(f"Reference delta_s shape: {delta_s_ref.shape}")
    print(f"Reference delta_s stats: min={delta_s_ref.min():.6f}, max={delta_s_ref.max():.6f}")

    # Let's manually implement what the Triton kernel should be doing
    print("\n=== MANUAL ATTENTION COMPUTATION ===")

    # Simulate the tiled processing
    tile_size = 128
    num_tiles_q = (N + tile_size - 1) // tile_size
    num_tiles_k = (N + tile_size - 1) // tile_size
    num_groups = delta_s_ref.shape[2]

    print(f"Tiles: Q={num_tiles_q}, K={num_tiles_k}, Groups={num_groups}")

    # For debugging, let's compute attention manually using the same approach as Triton
    sm_scale = 1.0 / (D ** 0.5)

    # Process each Q tile
    for q_tile_idx in range(num_tiles_q):
        q_start = q_tile_idx * tile_size
        q_end = min(q_start + tile_size, N)

        group_id = min(q_start // 128, num_groups - 1)  # With bounds check

        print(f"\nQ tile {q_tile_idx}: positions {q_start}-{q_end}, group_id={group_id}")

        # Extract Q tile
        q_tile = q_smoothed_ref[:, :, q_start:q_end, :]  # [B, H, tile_len, D]

        for k_tile_idx in range(num_tiles_k):
            k_start = k_tile_idx * tile_size
            k_end = min(k_start + tile_size, N)

            # Extract K tile
            k_tile = k_smoothed_ref[:, :, k_start:k_end, :]  # [B, H, tile_len, D]

            # Compute QK scores
            qk_scores = torch.matmul(q_tile, k_tile.transpose(-2, -1)) * sm_scale  # [B, H, q_tile_len, k_tile_len]

            # Apply delta_s correction - this is the critical part!
            # delta_s[batch, head, group_id, k_start:k_end] should be added
            delta_s_slice = delta_s_ref[:, :, group_id, k_start:k_end]  # [B, H, k_tile_len]

            print(f"  K tile {k_tile_idx}: positions {k_start}-{k_end}")
            print(f"    QK scores shape: {qk_scores.shape}")
            print(f"    Delta_s slice shape: {delta_s_slice.shape}")
            print(f"    Delta_s slice stats: min={delta_s_slice.min():.6f}, max={delta_s_slice.max():.6f}")

            # This is how the correction should be applied
            # delta_s needs to be broadcast from [B, H, k_tile_len] to [B, H, q_tile_len, k_tile_len]
            delta_s_broadcasted = delta_s_slice.unsqueeze(2)  # [B, H, 1, k_tile_len]
            corrected_qk = qk_scores + delta_s_broadcasted

            correction_magnitude = (corrected_qk - qk_scores).abs().mean().item()
            print(f"    Correction magnitude: {correction_magnitude:.6f}")

            # This is a significant correction if it's large
            if correction_magnitude > 0.01:
                print(f"    ⚠️  LARGE CORRECTION - delta_s has significant impact")

def test_triton_vs_manual():
    """Compare Triton output with manual computation."""
    device = torch.device('cuda')
    dtype = torch.float16

    torch.manual_seed(42)
    B, H, N, D = 1, 1, 256, 64

    q = torch.randn(B, H, N, D, dtype=dtype, device=device)
    k = torch.randn(B, H, N, D, dtype=dtype, device=device)
    v = torch.randn(B, H, N, D, dtype=dtype, device=device)

    # PyTorch reference (ground truth)
    output_ref = sageattn3_torch(
        q=q, k=k, v=v,
        tensor_layout="HND",
        is_causal=False,
        per_block_mean=True
    )

    # Triton implementation (potentially buggy)
    output_triton = sageattn3_torch_triton(
        q=q, k=k, v=v,
        tensor_layout="HND",
        is_causal=False,
        per_block_mean=True
    )

    # Triton without per_block_mean (should be more accurate)
    output_triton_no_pbm = sageattn3_torch_triton(
        q=q, k=k, v=v,
        tensor_layout="HND",
        is_causal=False,
        per_block_mean=False
    )

    print("\n=== OUTPUT COMPARISONS ===")

    # Compare PyTorch vs Triton with per_block_mean
    diff1 = (output_ref - output_triton).abs()
    cosine1 = torch.nn.functional.cosine_similarity(
        output_ref.flatten(), output_triton.flatten(), dim=0
    ).item()

    print(f"PyTorch vs Triton (per_block_mean=True):")
    print(f"  Max diff: {diff1.max():.6f}, Mean diff: {diff1.mean():.6f}, Cosine: {cosine1:.6f}")

    # Compare Triton with vs without per_block_mean
    diff2 = (output_triton_no_pbm - output_triton).abs()
    cosine2 = torch.nn.functional.cosine_similarity(
        output_triton_no_pbm.flatten(), output_triton.flatten(), dim=0
    ).item()

    print(f"Triton per_block_mean=False vs True:")
    print(f"  Max diff: {diff2.max():.6f}, Mean diff: {diff2.mean():.6f}, Cosine: {cosine2:.6f}")

    # Compare PyTorch vs Triton without per_block_mean
    diff3 = (output_ref - output_triton_no_pbm).abs()
    cosine3 = torch.nn.functional.cosine_similarity(
        output_ref.flatten(), output_triton_no_pbm.flatten(), dim=0
    ).item()

    print(f"PyTorch vs Triton (per_block_mean=False):")
    print(f"  Max diff: {diff3.max():.6f}, Mean diff: {diff3.mean():.6f}, Cosine: {cosine3:.6f}")

    if cosine1 < 0.99:
        print("\n❌ PyTorch vs Triton per_block_mean=True: ACCURACY ISSUE")
    if cosine2 < 0.99:
        print("❌ Triton per_block_mean on/off: LARGE DIFFERENCE")
    if cosine3 > 0.999:
        print("✅ PyTorch vs Triton per_block_mean=False: GOOD ACCURACY")
        print("   This confirms the issue is specifically with per_block_mean=True")

if __name__ == "__main__":
    print("Deep Delta_s Debugging")
    print("=" * 50)

    detailed_delta_s_analysis()
    test_triton_vs_manual()