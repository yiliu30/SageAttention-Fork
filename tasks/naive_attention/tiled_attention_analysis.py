"""
Tiled/Block-wise Attention Analysis: Mathematical Equivalence

This module analyzes and implements the tiled approach to attention computation:
1. Divide Q, K, V into blocks/tiles
2. Compute softmax on each QK tile
3. Multiply with corresponding V tile
4. Normalize at the end

Key Question: Is this mathematically equivalent to standard attention?

Mathematical Analysis:
=====================

Standard Attention:
  S = Q @ K^T                    # [N, N] scores matrix
  P = softmax(S, dim=-1)         # [N, N] probabilities
  O = P @ V                      # [N, D] output

Tiled Approach (naive version):
  For each block i,j:
    S_ij = Q_i @ K_j^T           # Block scores
    P_ij = softmax(S_ij, dim=-1) # Block probabilities
    O_ij = P_ij @ V_j            # Block output

  O = sum_j(O_ij) / sum_j(sum(P_ij, dim=-1))  # Normalize

Mathematical Issue:
==================
The naive tiled approach is NOT mathematically equivalent because:

1. softmax(A) + softmax(B) ≠ softmax([A, B])
2. Each tile computes probabilities independently
3. Global normalization is needed but complex

Correct Tiled Approach (FlashAttention style):
=============================================
Must use online softmax to maintain running statistics across tiles:

For each tile (i, j):
  1. Compute S_ij = Q_i @ K_j^T
  2. Update running maximum: m_new = max(m_old, max(S_ij))
  3. Compute correction: alpha = exp(m_old - m_new)
  4. Rescale previous outputs: O *= alpha
  5. Compute tile contribution: P_ij = exp(S_ij - m_new)
  6. Add to output: O += P_ij @ V_j
  7. Update normalization: l = l * alpha + sum(P_ij)

Final: O = O / l

This maintains mathematical equivalence with standard attention.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple
import math


def naive_tiled_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_size: int = 32,
    is_causal: bool = False,
    sm_scale: float = None
) -> torch.Tensor:
    """
    Naive tiled attention (INCORRECT - for demonstration only).

    This shows what happens if you naively apply softmax to each tile
    and try to combine them. This is NOT mathematically equivalent
    to standard attention.

    WARNING: This implementation is intentionally incorrect to demonstrate
    why the naive approach doesn't work.
    """
    batch_size, num_heads, seq_len_q, head_dim = q.shape
    _, _, seq_len_k, _ = k.shape

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    # Initialize output
    output = torch.zeros_like(q)
    total_weights = torch.zeros((batch_size, num_heads, seq_len_q, 1),
                               device=q.device, dtype=q.dtype)

    # Process in tiles
    for q_start in range(0, seq_len_q, block_size):
        q_end = min(q_start + block_size, seq_len_q)
        q_tile = q[:, :, q_start:q_end, :]

        tile_output = torch.zeros_like(q_tile)
        tile_weights = torch.zeros((batch_size, num_heads, q_end - q_start, 1),
                                  device=q.device, dtype=q.dtype)

        for k_start in range(0, seq_len_k, block_size):
            k_end = min(k_start + block_size, seq_len_k)
            k_tile = k[:, :, k_start:k_end, :]
            v_tile = v[:, :, k_start:k_end, :]

            # Compute attention scores for this tile
            scores = torch.matmul(q_tile, k_tile.transpose(-2, -1)) * sm_scale

            # Apply causal masking if needed
            if is_causal:
                # Create causal mask for this tile
                q_indices = torch.arange(q_start, q_end, device=q.device)
                k_indices = torch.arange(k_start, k_end, device=q.device)
                mask = q_indices.unsqueeze(1) >= k_indices.unsqueeze(0)
                scores = scores.masked_fill(~mask, float('-inf'))

            # Apply softmax to tile (THIS IS THE PROBLEM!)
            tile_probs = F.softmax(scores, dim=-1)

            # Compute contribution from this tile
            tile_contrib = torch.matmul(tile_probs, v_tile)
            tile_weight = tile_probs.sum(dim=-1, keepdim=True)

            tile_output += tile_contrib
            tile_weights += tile_weight

        # Normalize tile output (INCORRECT - doesn't account for global context)
        tile_output = tile_output / (tile_weights + 1e-8)
        output[:, :, q_start:q_end, :] = tile_output

    return output


def correct_tiled_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_size: int = 32,
    is_causal: bool = False,
    sm_scale: float = None
) -> torch.Tensor:
    """
    Correct tiled attention using online softmax (FlashAttention approach).

    This maintains mathematical equivalence with standard attention by
    using running statistics across tiles.
    """
    batch_size, num_heads, seq_len_q, head_dim = q.shape
    _, _, seq_len_k, _ = k.shape

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    # Initialize output and running statistics
    output = torch.zeros_like(q)

    # Process each query block
    for q_start in range(0, seq_len_q, block_size):
        q_end = min(q_start + block_size, seq_len_q)
        q_tile = q[:, :, q_start:q_end, :]

        # Initialize running statistics for this query block
        m = torch.full((batch_size, num_heads, q_end - q_start),
                      float('-inf'), device=q.device, dtype=torch.float32)
        l = torch.zeros((batch_size, num_heads, q_end - q_start),
                       device=q.device, dtype=torch.float32)
        block_output = torch.zeros_like(q_tile, dtype=torch.float32)

        # Process each key/value block
        for k_start in range(0, seq_len_k, block_size):
            k_end = min(k_start + block_size, seq_len_k)
            k_tile = k[:, :, k_start:k_end, :]
            v_tile = v[:, :, k_start:k_end, :]

            # Compute attention scores for this tile
            scores = torch.matmul(q_tile, k_tile.transpose(-2, -1)) * sm_scale

            # Apply causal masking if needed
            if is_causal:
                q_indices = torch.arange(q_start, q_end, device=q.device)
                k_indices = torch.arange(k_start, k_end, device=q.device)
                mask = q_indices.unsqueeze(1) >= k_indices.unsqueeze(0)
                scores = scores.masked_fill(~mask, float('-inf'))

            # Online softmax update
            m_old = m.clone()

            # Update running maximum
            tile_max = scores.max(dim=-1)[0]
            m_new = torch.maximum(m, tile_max)

            # Compute correction factors
            alpha = torch.exp(m_old - m_new)
            beta = torch.exp(tile_max - m_new)

            # Rescale previous output
            block_output = block_output * alpha.unsqueeze(-1)

            # Compute tile probabilities
            tile_probs = torch.exp(scores - m_new.unsqueeze(-1))

            # Add tile contribution
            tile_contrib = torch.matmul(tile_probs, v_tile)
            block_output = block_output + tile_contrib

            # Update running sum
            tile_sum = tile_probs.sum(dim=-1)
            l = l * alpha + tile_sum

            # Update running maximum
            m = m_new

        # Final normalization
        block_output = block_output / l.unsqueeze(-1)
        output[:, :, q_start:q_end, :] = block_output

    return output


def demonstrate_tiled_equivalence():
    """Demonstrate the mathematical equivalence of correct tiled attention."""
    print("🧩 TILED ATTENTION MATHEMATICAL EQUIVALENCE ANALYSIS")
    print("=" * 60)

    # Setup test case
    torch.manual_seed(42)
    batch_size, num_heads, seq_len, head_dim = 2, 4, 128, 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

    print(f"Test configuration:")
    print(f"  Shape: [{batch_size}, {num_heads}, {seq_len}, {head_dim}]")
    print(f"  Device: {device}")

    # Test different block sizes
    block_sizes = [16, 32, 64, 128]

    # Get reference output (standard attention)
    reference_output = F.scaled_dot_product_attention(q, k, v)

    print(f"\n{'Block Size':<10} {'Method':<20} {'Max Diff':<12} {'Cosine Sim':<12} {'Status'}")
    print("-" * 65)

    for block_size in block_sizes:
        # Test naive tiled attention (should be incorrect)
        naive_output = naive_tiled_attention(q, k, v, block_size=block_size)
        naive_diff = torch.abs(reference_output - naive_output).max().item()
        naive_cos_sim = F.cosine_similarity(
            reference_output.flatten().unsqueeze(0),
            naive_output.flatten().unsqueeze(0)
        ).item()
        naive_status = "✓ CORRECT" if naive_diff < 1e-4 else "✗ INCORRECT"

        # Test correct tiled attention (should be equivalent)
        correct_output = correct_tiled_attention(q, k, v, block_size=block_size)
        correct_diff = torch.abs(reference_output - correct_output).max().item()
        correct_cos_sim = F.cosine_similarity(
            reference_output.flatten().unsqueeze(0),
            correct_output.flatten().unsqueeze(0)
        ).item()
        correct_status = "✓ CORRECT" if correct_diff < 1e-4 else "✗ INCORRECT"

        print(f"{block_size:<10} {'Naive':<20} {naive_diff:<12.2e} {naive_cos_sim:<12.6f} {naive_status}")
        print(f"{'':<10} {'Correct (Online)':<20} {correct_diff:<12.2e} {correct_cos_sim:<12.6f} {correct_status}")
        print()


def analyze_why_naive_fails():
    """Analyze why the naive tiled approach fails mathematically."""
    print("\n🔍 WHY NAIVE TILED SOFTMAX FAILS")
    print("=" * 40)

    # Simple example to show the problem
    print("Simple Example:")
    print("Consider scores matrix: [[1, 2], [3, 4]]")

    # Full softmax
    scores = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    full_softmax = F.softmax(scores, dim=-1)
    print(f"Full softmax: {full_softmax}")
    print(f"Row sums: {full_softmax.sum(dim=-1)}")

    # Naive tiled approach (tile size = 1)
    print("\nNaive tiled approach (tile size = 1):")
    tile1 = F.softmax(scores[:, :1], dim=-1)  # First column
    tile2 = F.softmax(scores[:, 1:], dim=-1)  # Second column

    print(f"Tile 1 softmax: {tile1}")
    print(f"Tile 2 softmax: {tile2}")

    # Try to combine (this is wrong!)
    naive_combined = torch.cat([tile1, tile2], dim=-1)
    print(f"Naive combination: {naive_combined}")
    print(f"Row sums: {naive_combined.sum(dim=-1)} (should be 1.0!)")

    # Show the difference
    diff = torch.abs(full_softmax - naive_combined).max().item()
    print(f"Max difference: {diff:.6f}")

    print("\n📝 Key Insight:")
    print("   Softmax normalization depends on ALL elements in the row,")
    print("   not just the elements in the current tile!")
    print("   This is why we need online softmax with running statistics.")


def demonstrate_causal_tiling():
    """Demonstrate tiled attention with causal masking."""
    print("\n🎭 CAUSAL TILED ATTENTION")
    print("=" * 30)

    torch.manual_seed(123)
    batch_size, num_heads, seq_len, head_dim = 1, 1, 32, 16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

    # Reference causal attention
    reference_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    # Test tiled causal attention
    block_sizes = [8, 16, 32]

    print(f"{'Block Size':<10} {'Max Diff':<12} {'Cosine Sim':<12} {'Status'}")
    print("-" * 45)

    for block_size in block_sizes:
        tiled_output = correct_tiled_attention(q, k, v, block_size=block_size, is_causal=True)

        diff = torch.abs(reference_output - tiled_output).max().item()
        cos_sim = F.cosine_similarity(
            reference_output.flatten().unsqueeze(0),
            tiled_output.flatten().unsqueeze(0)
        ).item()
        status = "✓ CORRECT" if diff < 1e-4 else "✗ INCORRECT"

        print(f"{block_size:<10} {diff:<12.2e} {cos_sim:<12.6f} {status}")


if __name__ == "__main__":
    print("Tiled Attention Analysis")
    print("Understanding block-wise attention computation")
    print()

    try:
        demonstrate_tiled_equivalence()
        analyze_why_naive_fails()
        demonstrate_causal_tiling()

        print("\n" + "=" * 60)
        print("🎓 EDUCATIONAL SUMMARY")
        print("=" * 60)
        print("1. Naive tiled softmax is NOT mathematically equivalent")
        print("2. Each tile's softmax normalization is independent")
        print("3. Global context is lost, leading to incorrect results")
        print("4. Correct approach uses online softmax across tiles")
        print("5. Running statistics (max, sum) maintain equivalence")
        print("6. This is the foundation of FlashAttention's efficiency")
        print()
        print("✅ Correct tiled attention maintains mathematical equivalence")
        print("❌ Naive tiled approach breaks mathematical correctness")

    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()