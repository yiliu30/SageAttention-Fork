"""
Simple Demonstration: Why Naive Tiled Softmax Fails

This script provides a minimal, clear example showing why you can't simply:
1. Apply softmax to QK tiles independently
2. Multiply with V tiles
3. Normalize at the end

The fundamental issue: softmax normalization requires global context!
"""

import torch
import torch.nn.functional as F


def simple_example():
    """Show the failure with a tiny 2x2 example."""
    print("🚨 WHY NAIVE TILED SOFTMAX FAILS")
    print("=" * 40)
    print()

    # Simple 2x2 attention scores matrix
    scores = torch.tensor([[1.0, 2.0],
                          [3.0, 4.0]])

    print("Original scores matrix:")
    print(f"{scores}")
    print()

    # Method 1: Standard softmax (CORRECT)
    print("Method 1: Standard Softmax (Correct)")
    correct_probs = F.softmax(scores, dim=-1)
    print(f"Result: {correct_probs}")
    print(f"Row sums: {correct_probs.sum(dim=-1)} ← Should be [1.0, 1.0] ✓")
    print()

    # Method 2: Naive tiled softmax (INCORRECT)
    print("Method 2: Naive Tiled Softmax (Incorrect)")
    print("Treating each column as a separate tile:")

    # Tile 1: First column [1, 3]
    tile1 = F.softmax(scores[:, 0:1], dim=-1)
    print(f"Tile 1 (col 0): {tile1.flatten()} ← Normalizes to 1.0")

    # Tile 2: Second column [2, 4]
    tile2 = F.softmax(scores[:, 1:2], dim=-1)
    print(f"Tile 2 (col 1): {tile2.flatten()} ← Normalizes to 1.0")

    # Combine tiles
    naive_probs = torch.cat([tile1, tile2], dim=-1)
    print(f"Combined: {naive_probs}")
    print(f"Row sums: {naive_probs.sum(dim=-1)} ← Should be [1.0, 1.0] but is [2.0, 2.0] ❌")
    print()

    # Show the error
    error = torch.abs(correct_probs - naive_probs).max().item()
    print(f"Maximum error: {error:.6f}")
    print()


def demonstrate_with_attention():
    """Show how this affects actual attention computation."""
    print("🧠 IMPACT ON ATTENTION OUTPUT")
    print("=" * 30)
    print()

    # Setup tiny attention example
    seq_len = 4
    head_dim = 2

    torch.manual_seed(42)
    q = torch.randn(1, 1, seq_len, head_dim)  # [1, 1, 4, 2]
    k = torch.randn(1, 1, seq_len, head_dim)  # [1, 1, 4, 2]
    v = torch.randn(1, 1, seq_len, head_dim)  # [1, 1, 4, 2]

    print(f"Setup: Q, K, V shapes = {list(q.shape)}")

    # Standard attention (correct)
    scores = torch.matmul(q, k.transpose(-2, -1))  # [1, 1, 4, 4]
    correct_probs = F.softmax(scores, dim=-1)
    correct_output = torch.matmul(correct_probs, v)

    print(f"Scores matrix shape: {list(scores.shape)}")
    print("Attention scores:")
    print(f"{scores[0, 0]}")
    print()

    # Naive tiled approach (incorrect) - tile size 2
    print("Naive Tiled Approach (tile_size=2):")
    naive_output = torch.zeros_like(correct_output)

    # Process each query-key tile pair
    for q_start in range(0, seq_len, 2):
        q_end = min(q_start + 2, seq_len)
        q_tile = q[:, :, q_start:q_end, :]

        tile_output = torch.zeros_like(q_tile)

        for k_start in range(0, seq_len, 2):
            k_end = min(k_start + 2, seq_len)
            k_tile = k[:, :, k_start:k_end, :]
            v_tile = v[:, :, k_start:k_end, :]

            # Compute tile scores
            tile_scores = torch.matmul(q_tile, k_tile.transpose(-2, -1))

            # Apply softmax to JUST THIS TILE (the problem!)
            tile_probs = F.softmax(tile_scores, dim=-1)

            print(f"Tile [{q_start}:{q_end}, {k_start}:{k_end}] probs:")
            print(f"{tile_probs[0, 0]}")

            # Accumulate (this is wrong because tiles don't sum to 1 globally)
            tile_contrib = torch.matmul(tile_probs, v_tile)
            tile_output += tile_contrib

        naive_output[:, :, q_start:q_end, :] = tile_output / 2  # Naive normalization

    print()
    print("Results comparison:")
    print(f"Correct output: {correct_output[0, 0]}")
    print(f"Naive tiled:    {naive_output[0, 0]}")

    error = torch.abs(correct_output - naive_output).max().item()
    cos_sim = F.cosine_similarity(
        correct_output.flatten().unsqueeze(0),
        naive_output.flatten().unsqueeze(0)
    ).item()

    print(f"Max error: {error:.6f}")
    print(f"Cosine similarity: {cos_sim:.6f}")
    print("❌ Naive tiled approach produces incorrect results!")
    print()


def show_solution():
    """Show the correct way using online softmax principles."""
    print("✅ THE CORRECT SOLUTION")
    print("=" * 25)
    print()

    print("The correct approach uses online softmax across tiles:")
    print()
    print("```python")
    print("# For each tile:")
    print("# 1. Update global maximum: m_new = max(m_old, max(tile_scores))")
    print("# 2. Compute correction: alpha = exp(m_old - m_new)")
    print("# 3. Rescale previous output: output *= alpha")
    print("# 4. Add tile contribution: output += exp(tile_scores - m_new) @ v_tile")
    print("# 5. Update normalization: norm = norm * alpha + sum(exp(...))")
    print("# 6. Final: output = output / norm")
    print("```")
    print()
    print("Key insight: Maintain running statistics (max, sum) across ALL tiles")
    print("to preserve the global context needed for correct softmax normalization.")
    print()
    print("This is exactly what FlashAttention does! 🎯")


if __name__ == "__main__":
    print("Tiled Attention: Why the Naive Approach Fails")
    print("=" * 50)
    print()

    simple_example()
    demonstrate_with_attention()
    show_solution()

    print()
    print("🎓 KEY TAKEAWAY:")
    print("Softmax is inherently global - you cannot decompose it into")
    print("independent local operations without losing mathematical correctness.")
    print("The solution requires online algorithms that maintain global state!")