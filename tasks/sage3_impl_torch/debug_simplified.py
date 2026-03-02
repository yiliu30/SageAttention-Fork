#!/usr/bin/env python3
"""
Simplified SageAttention3 implementation to debug numerical issues.
This version focuses on correctness over educational annotations.
"""

import torch
import torch.nn.functional as F
import math
from typing import Optional, Tuple


def sageattn3_torch_simplified(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    per_block_mean: bool = True,
    tile_size: int = 64
) -> torch.Tensor:
    """
    Simplified SageAttention3 for debugging.
    Focus on getting the core algorithm right first.
    """
    B, H, N, D = q.shape
    sm_scale = 1.0 / math.sqrt(D)

    # For debugging, skip complex smoothing and quantization
    # Just use the inputs directly to focus on the attention algorithm
    q_use = q
    k_use = k
    v_use = v

    # Step 3: Tiled attention with correct online softmax
    output = torch.zeros_like(q)

    num_tiles = (N + tile_size - 1) // tile_size

    for i in range(num_tiles):
        q_start = i * tile_size
        q_end = min(q_start + tile_size, N)
        q_tile = q_use[:, :, q_start:q_end, :]

        # Running statistics for this Q tile
        running_max = torch.full((B, H, q_end - q_start), -torch.inf, device=q.device, dtype=torch.float32)
        running_sum = torch.zeros((B, H, q_end - q_start), device=q.device, dtype=torch.float32)
        output_tile = torch.zeros((B, H, q_end - q_start, D), device=q.device, dtype=q.dtype)

        for j in range(num_tiles):
            k_start = j * tile_size
            k_end = min(k_start + tile_size, N)
            k_tile = k_use[:, :, k_start:k_end, :]
            v_tile = v_use[:, :, k_start:k_end, :]

            # Compute QK^T
            qk_tile = torch.matmul(q_tile, k_tile.transpose(-2, -1)) * sm_scale

            # Apply causal mask
            if is_causal:
                mask = torch.triu(
                    torch.full((q_end - q_start, k_end - k_start), -torch.inf, device=q.device),
                    diagonal=k_start - q_start + 1
                )
                qk_tile = qk_tile + mask

            # Online softmax
            tile_max = qk_tile.max(dim=-1)[0].float()  # [B, H, q_tile_size]
            old_max = running_max.clone()
            new_max = torch.maximum(running_max, tile_max)

            # Renormalization factors
            alpha = torch.exp(old_max - new_max)
            beta = torch.exp(tile_max - new_max)

            # Update output with renormalization
            output_tile = output_tile * alpha.unsqueeze(-1).to(q.dtype)

            # Compute probabilities for current tile
            qk_shifted = qk_tile - new_max.unsqueeze(-1).to(q.dtype)
            p_tile = torch.exp(qk_shifted)

            # Compute PV (no quantization for debugging)
            pv_tile = torch.matmul(p_tile, v_tile)

            # Update running sum and max
            running_sum = running_sum * alpha + p_tile.sum(dim=-1).float() * beta
            running_max = new_max

            # Accumulate output
            output_tile = output_tile + pv_tile * beta.unsqueeze(-1).to(q.dtype)

        # Final normalization
        output_tile = output_tile / (running_sum.unsqueeze(-1).to(q.dtype) + 1e-8)

        output[:, :, q_start:q_end, :] = output_tile

    return output


if __name__ == "__main__":
    # Test the simplified implementation
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B, H, N, D = 1, 8, 64, 64
    q = torch.randn(B, H, N, D, dtype=torch.float16, device=device) * 0.1
    k = torch.randn(B, H, N, D, dtype=torch.float16, device=device) * 0.1
    v = torch.randn(B, H, N, D, dtype=torch.float16, device=device) * 0.1

    # Test our implementation
    output = sageattn3_torch_simplified(q, k, v)
    print(f"Our output range: [{output.min().item():.6f}, {output.max().item():.6f}]")

    # Compare with PyTorch
    ref_output = F.scaled_dot_product_attention(q, k, v)
    print(f"PyTorch range: [{ref_output.min().item():.6f}, {ref_output.max().item():.6f}]")

    # Compute metrics
    cos_sim = F.cosine_similarity(output.flatten(), ref_output.flatten(), dim=0).item()
    print(f"Cosine similarity: {cos_sim:.6f}")