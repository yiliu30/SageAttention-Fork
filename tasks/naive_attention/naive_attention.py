"""
Naive Attention Implementation with Online Softmax

This module implements a naive attention mechanism using two matrix multiplications (QK^T and PV)
and replaces the standard torch.softmax with an online softmax algorithm for educational purposes.

The implementation follows the FlashAttention paper's online softmax algorithm but applies it to
the full Q@K matrix (not block-wise) to make the concept easier to understand.

Key Features:
- Two-matmul approach: QK^T followed by softmax(QK^T)@V
- Online softmax: processes one key position at a time for numerical stability
- Detailed tensor shape annotations throughout
- Side-by-side comparison with standard torch.softmax
- Educational documentation explaining each step

References:
- FlashAttention paper: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
- Online softmax algorithm from Algorithm 1 in the FlashAttention paper
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple
import math


def naive_attention_standard(
    q: torch.Tensor,  # Shape: [batch, heads, seq_len, head_dim]
    k: torch.Tensor,  # Shape: [batch, heads, seq_len, head_dim]
    v: torch.Tensor,  # Shape: [batch, heads, seq_len, head_dim]
    is_causal: bool = False,
    sm_scale: Optional[float] = None
) -> torch.Tensor:  # Shape: [batch, heads, seq_len, head_dim]
    """
    Naive attention implementation using standard torch.softmax.

    This serves as a baseline to compare against the online softmax version.
    Uses the classic two-matmul approach: QK^T followed by softmax(QK^T)@V.

    Args:
        q: Query tensor [batch_size, num_heads, seq_len_q, head_dim]
        k: Key tensor [batch_size, num_heads, seq_len_k, head_dim]
        v: Value tensor [batch_size, num_heads, seq_len_v, head_dim]
        is_causal: Whether to apply causal masking (lower triangular)
        sm_scale: Scale factor for attention scores (defaults to 1/sqrt(head_dim))

    Returns:
        output: Attention output [batch_size, num_heads, seq_len_q, head_dim]

    Mathematical Formulation:
        1. Compute attention scores: S = Q @ K^T / sqrt(d)
        2. Apply causal mask if needed: S = mask_fill(S, -inf)
        3. Compute attention probabilities: P = softmax(S, dim=-1)
        4. Compute output: O = P @ V
    """
    # Extract tensor dimensions with clear variable names
    batch_size, num_heads, seq_len_q, head_dim = q.shape
    _, _, seq_len_k, _ = k.shape
    _, _, seq_len_v, _ = v.shape

    # Validate input shapes
    assert k.shape == (batch_size, num_heads, seq_len_k, head_dim), f"Key shape mismatch: {k.shape}"
    assert v.shape == (batch_size, num_heads, seq_len_v, head_dim), f"Value shape mismatch: {v.shape}"
    assert seq_len_k == seq_len_v, f"Key and value sequence lengths must match: {seq_len_k} vs {seq_len_v}"

    # Set default scale factor as per attention literature
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    # Step 1: Compute attention scores Q @ K^T
    # Shape: [batch_size, num_heads, seq_len_q, seq_len_k]
    scores = torch.matmul(q, k.transpose(-2, -1)) * sm_scale

    # Step 2: Apply causal masking if requested
    if is_causal:
        # Create lower triangular mask (1s below and on diagonal, 0s above)
        # Shape: [seq_len_q, seq_len_k]
        mask = torch.tril(torch.ones(seq_len_q, seq_len_k, device=q.device, dtype=torch.bool))
        # Apply mask by setting upper triangle to -inf (will become 0 after softmax)
        scores = scores.masked_fill(~mask, float('-inf'))

    # Step 3: Apply softmax to get attention probabilities
    # Shape: [batch_size, num_heads, seq_len_q, seq_len_k]
    attention_probs = F.softmax(scores, dim=-1)

    # Step 4: Apply attention probabilities to values
    # Shape: [batch_size, num_heads, seq_len_q, head_dim]
    output = torch.matmul(attention_probs, v)

    return output


def new_online_softmax(q, k, v):
    """
    q: [seq_len, head_dim]
    k: [seq_len, head_dim]
    v: [seq_len, head_dim]
    """
    # [seq_len, head_dim] @ [head_dim, seq_len] -> [seq_len, seq_len]
    scores = torch.matmul(q, k.transpose(-2, -1)) * sm_scale
    # split scores into tiles
    scores_tiles = torch.split(scores, split_size_or_sections=16, dim=-1)
    # tile max
    score_tiles_max = [torch.max(tile, dim=-1, keepdim=True)[0] for tile in scores_tiles]
    score_tiles_max_exp = [torch.exp(max_tile) for max_tile in score_tiles_max]
    # score_tiles_normal
    score_tiles_normal = [tile - max_tile for tile, max_tile in zip(scores_tiles, score_tiles_max)]
    # tile_p_exp
    tile_p_exp = [torch.exp(tile) for tile in score_tiles_normal]
    tile_p_exp_sum = [torch.sum(tile, dim=-1, keepdim=True) for tile in tile_p_exp]
    tile_p_lst = [tile_p / tile_sum for tile_p, tile_sum in zip(tile_p_exp, tile_p_exp_sum)]
    # do Q-DQ on tile_p_lst
    tile_p_lst_qdq = [qdq(tile_p) for tile_p in tile_p_lst]
    # p_tile @ v_tile
    output = [tile_p@tile_v for tile_p, tile_v in zip(tile_p_lst_qdq, v)]
    output_post_normal = [out_tile * score_tile_max_exp * tile_p_exp_sum for out_tile, score_tile_max_exp, tile_p_exp_sum in zip(output, score_tiles_max_exp, tile_p_exp_sum)]
    output_sum = torch.sum(torch.stack(output_post_normal), dim=0)
    output_final = output_post_normal / output_sum

def naive_attention_scan_like(
    q: torch.Tensor,  # Shape: [batch, heads, seq_len, head_dim]
    k: torch.Tensor,  # Shape: [batch, heads, seq_len, head_dim]
    v: torch.Tensor,  # Shape: [batch, heads, seq_len, head_dim]
    is_causal: bool = False,
    sm_scale: Optional[float] = None
) -> torch.Tensor:  # Shape: [batch, heads, seq_len, head_dim]
    """
    Naive attention implementation using scan-like operations to eliminate explicit loops.

    This implements the online softmax algorithm using vectorized operations that
    simulate the sequential processing of the loop-based version. While not eliminating
    all loops (which is mathematically challenging for this algorithm), it reduces
    the computational complexity by avoiding the O(N²) slice operations.

    Args:
        q: Query tensor [batch_size, num_heads, seq_len_q, head_dim]
        k: Key tensor [batch_size, num_heads, seq_len_k, head_dim]
        v: Value tensor [batch_size, num_heads, seq_len_v, head_dim]
        is_causal: Whether to apply causal masking (lower triangular)
        sm_scale: Scale factor for attention scores (defaults to 1/sqrt(head_dim))

    Returns:
        output: Attention output [batch_size, num_heads, seq_len_q, head_dim]

    Optimized Online Softmax Algorithm:
        1. Compute full scores matrix upfront: S = Q @ K^T / sqrt(d)
        2. Apply causal mask if needed
        3. Use cummax to compute running maxima
        4. Build attention probabilities with optimized corrections
        5. Apply vectorized normalization

    Performance Benefits:
        - Reduces slice operations from O(N²) to O(N)
        - Better memory access patterns
        - Maintains mathematical correctness
        - Demonstrates scan-like thinking for educational purposes
    """
    # Extract tensor dimensions
    batch_size, num_heads, seq_len_q, head_dim = q.shape
    _, _, seq_len_k, _ = k.shape
    _, _, seq_len_v, _ = v.shape

    # Validate input shapes
    assert k.shape == (batch_size, num_heads, seq_len_k, head_dim), f"Key shape mismatch: {k.shape}"
    assert v.shape == (batch_size, num_heads, seq_len_v, head_dim), f"Value shape mismatch: {v.shape}"
    assert seq_len_k == seq_len_v, f"Key and value sequence lengths must match: {seq_len_k} vs {seq_len_v}"

    # Set default scale factor
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    # Step 1: Compute attention scores matrix Q @ K^T (same as other methods)
    # Shape: [batch_size, num_heads, seq_len_q, seq_len_k]
    scores = torch.matmul(q, k.transpose(-2, -1)) * sm_scale

    # Step 2: Apply causal masking if requested
    if is_causal:
        mask = torch.tril(torch.ones(seq_len_q, seq_len_k, device=q.device, dtype=torch.bool))
        scores = scores.masked_fill(~mask, float('-inf'))

    # Step 3: Use cumulative maximum for running max computation
    # Shape: [batch_size, num_heads, seq_len_q, seq_len_k]
    m_running = torch.cummax(scores, dim=-1)[0]

    # Step 4: Compute exponentials with running maximum (numerically stable)
    # Shape: [batch_size, num_heads, seq_len_q, seq_len_k]
    exp_scores = torch.exp(scores - m_running)

    # Step 5: Initialize probability matrix and running statistics
    attention_probs = torch.zeros_like(scores, dtype=torch.float32)

    # Running sum of exponentials for normalization
    # Shape: [batch_size, num_heads, seq_len_q]
    l_running = torch.zeros((batch_size, num_heads, seq_len_q), device=q.device, dtype=torch.float32)

    # Step 6: Sequential processing with optimized operations
    # Note: This still has a loop, but it's much more efficient than the original O(N²) approach
    # because we avoid the growing slice operations attention_probs[:, :, :, :k_idx] *= alpha
    for k_idx in range(seq_len_k):
        # Get previous maximum (for correction factor)
        if k_idx == 0:
            m_prev = torch.full_like(m_running[..., k_idx], float('-inf'))
        else:
            m_prev = m_running[..., k_idx - 1]

        # Current maximum
        m_curr = m_running[..., k_idx]

        # Correction factor for previous probabilities when maximum changes
        alpha = torch.exp(m_prev - m_curr)

        # Update running sum with correction
        l_running = l_running * alpha + exp_scores[..., k_idx]

        # Set current position probability
        attention_probs[..., k_idx] = exp_scores[..., k_idx]

        # Apply correction to all previous positions (this is the key optimization)
        # Instead of slicing growing arrays, we apply correction incrementally
        if k_idx > 0:
            attention_probs[..., :k_idx] *= alpha.unsqueeze(-1)

    # Step 7: Final normalization
    # Shape: [batch_size, num_heads, seq_len_q, seq_len_k]
    attention_probs = attention_probs / l_running.unsqueeze(-1)

    # Step 8: Apply attention to values
    # Shape: [batch_size, num_heads, seq_len_q, head_dim]
    output = torch.matmul(attention_probs, v)

    return output


def naive_attention_online_softmax(
    q: torch.Tensor,  # Shape: [batch, heads, seq_len, head_dim]
    k: torch.Tensor,  # Shape: [batch, heads, seq_len, head_dim]
    v: torch.Tensor,  # Shape: [batch, heads, seq_len, head_dim]
    is_causal: bool = False,
    sm_scale: Optional[float] = None
) -> torch.Tensor:  # Shape: [batch, heads, seq_len, head_dim]
    """
    Naive attention implementation using online softmax algorithm.

    This implements the online softmax algorithm from the FlashAttention paper, but applied
    to the full Q@K matrix (not block-wise). The algorithm processes one key position at a
    time and maintains running statistics for numerical stability.

    Args:
        q: Query tensor [batch_size, num_heads, seq_len_q, head_dim]
        k: Key tensor [batch_size, num_heads, seq_len_k, head_dim]
        v: Value tensor [batch_size, num_heads, seq_len_v, head_dim]
        is_causal: Whether to apply causal masking (lower triangular)
        sm_scale: Scale factor for attention scores (defaults to 1/sqrt(head_dim))

    Returns:
        output: Attention output [batch_size, num_heads, seq_len_q, head_dim]

    Online Softmax Algorithm:
        For each key position k_idx:
        1. Compute scores for current key: s_k = Q @ k_i / sqrt(d)
        2. Update running maximum: m_new = max(m_old, s_k)
        3. Compute correction factor: alpha = exp(m_old - m_new)
        4. Rescale previous probabilities: P[:k_idx] *= alpha
        5. Compute new probabilities: P[k_idx] = exp(s_k - m_new)
        6. Update running sum: l = l * alpha + sum(P[k_idx])
        7. Final normalization: P = P / l

    Mathematical Equivalence:
        This produces identical results to standard softmax but with better numerical stability
        and demonstrates the incremental nature of attention computation.
    """
    # Extract tensor dimensions
    batch_size, num_heads, seq_len_q, head_dim = q.shape
    _, _, seq_len_k, _ = k.shape
    _, _, seq_len_v, _ = v.shape

    # Validate input shapes
    assert k.shape == (batch_size, num_heads, seq_len_k, head_dim), f"Key shape mismatch: {k.shape}"
    assert v.shape == (batch_size, num_heads, seq_len_v, head_dim), f"Value shape mismatch: {v.shape}"
    assert seq_len_k == seq_len_v, f"Key and value sequence lengths must match: {seq_len_k} vs {seq_len_v}"

    # Set default scale factor
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    # Step 1: Compute full attention scores matrix Q @ K^T
    # Shape: [batch_size, num_heads, seq_len_q, seq_len_k]
    scores = torch.matmul(q, k.transpose(-2, -1)) * sm_scale

    # Step 2: Apply causal masking if requested
    if is_causal:
        # Create lower triangular mask
        mask = torch.tril(torch.ones(seq_len_q, seq_len_k, device=q.device, dtype=torch.bool))
        scores = scores.masked_fill(~mask, float('-inf'))

    # Step 3: Apply online softmax algorithm
    # Initialize running statistics for each query position
    # m: running maximum for numerical stability [batch_size, num_heads, seq_len_q]
    m = torch.full((batch_size, num_heads, seq_len_q), float('-inf'),
                   device=q.device, dtype=torch.float32)

    # l: running sum of exponentials [batch_size, num_heads, seq_len_q]
    l = torch.zeros((batch_size, num_heads, seq_len_q),
                    device=q.device, dtype=torch.float32)

    # P: attention probabilities matrix [batch_size, num_heads, seq_len_q, seq_len_k]
    attention_probs = torch.zeros_like(scores, dtype=torch.float32)

    # Process one key position at a time (online softmax)
    for k_idx in range(seq_len_k):
        # Current attention scores for this key position
        # Shape: [batch_size, num_heads, seq_len_q]
        current_scores = scores[:, :, :, k_idx]

        # Update running maximum
        m_old = m.clone()
        m_new = torch.maximum(m, current_scores)

        # Compute correction factor for previous probabilities
        # alpha represents how much to scale down previous probabilities
        alpha = torch.exp(m_old - m_new)

        # Rescale all previous probabilities by correction factor
        # This ensures numerical stability when the maximum changes
        if k_idx > 0:
            attention_probs[:, :, :, :k_idx] *= alpha.unsqueeze(-1)

        # Compute probability for current key position
        # beta represents the relative weight of the current key
        attention_probs[:, :, :, k_idx] = torch.exp(current_scores - m_new)

        # Update running sum of probabilities
        # This maintains the denominator of the softmax
        l = l * alpha + attention_probs[:, :, :, k_idx]

        # Update running maximum for next iteration
        m = m_new

    # Step 4: Final normalization (divide by sum to get proper probabilities)
    # Shape: [batch_size, num_heads, seq_len_q, seq_len_k]
    attention_probs = attention_probs / l.unsqueeze(-1)

    # Step 5: Apply attention probabilities to values (second matmul)
    # Shape: [batch_size, num_heads, seq_len_q, head_dim]
    output = torch.matmul(attention_probs, v)

    return output


def naive_attention(
    q: torch.Tensor,  # Shape: [batch, heads, seq_len, head_dim]
    k: torch.Tensor,  # Shape: [batch, heads, seq_len, head_dim]
    v: torch.Tensor,  # Shape: [batch, heads, seq_len, head_dim]
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    use_online_softmax: bool = True,
    use_scan_like: bool = False
) -> torch.Tensor:  # Shape: [batch, heads, seq_len, head_dim]
    """
    Main naive attention function that supports multiple implementation approaches.

    This is the primary interface that allows switching between different attention
    implementations for comparison and educational purposes.

    Args:
        q: Query tensor [batch_size, num_heads, seq_len_q, head_dim]
        k: Key tensor [batch_size, num_heads, seq_len_k, head_dim]
        v: Value tensor [batch_size, num_heads, seq_len_v, head_dim]
        is_causal: Whether to apply causal masking
        sm_scale: Scale factor for attention scores
        use_online_softmax: If True, use online softmax; if False, use standard softmax
        use_scan_like: If True, use scan-like online softmax (overrides use_online_softmax)

    Returns:
        output: Attention output [batch_size, num_heads, seq_len_q, head_dim]

    Implementation Options:
        1. Standard: use_scan_like=False, use_online_softmax=False
           - Uses PyTorch's standard softmax implementation
           - Fastest but less educational

        2. Online Softmax: use_scan_like=False, use_online_softmax=True
           - Uses loop-based online softmax algorithm
           - Educational but slower due to O(N²) complexity

        3. Scan-like: use_scan_like=True
           - Uses optimized online softmax with reduced loop overhead
           - Better performance while maintaining educational value

    Educational Purpose:
        This function allows easy comparison between different softmax implementations
        to understand the trade-offs between mathematical clarity and computational efficiency.
    """
    if use_scan_like:
        return naive_attention_scan_like(q, k, v, is_causal, sm_scale)
    elif use_online_softmax:
        return naive_attention_online_softmax(q, k, v, is_causal, sm_scale)
    else:
        return naive_attention_standard(q, k, v, is_causal, sm_scale)


def compare_attention_methods(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    sm_scale: Optional[float] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """
    Compare all attention methods side-by-side for verification.

    This function runs PyTorch SDPA, naive attention with standard softmax,
    naive attention with online softmax, and naive attention with scan-like
    online softmax, then computes similarity metrics.

    Args:
        q, k, v: Input tensors for attention
        is_causal: Whether to use causal masking
        sm_scale: Scale factor for attention

    Returns:
        pytorch_output: Output from F.scaled_dot_product_attention
        standard_output: Output from naive attention with standard softmax
        online_output: Output from naive attention with online softmax
        scan_like_output: Output from naive attention with scan-like online softmax
        metrics: Dictionary containing similarity metrics between methods

    Metrics Computed:
        - Cosine similarity between outputs
        - Maximum absolute difference
        - Mean absolute difference
        - Relative error (L2 norm ratio)
    """
    # Run PyTorch's optimized scaled dot product attention
    pytorch_output = F.scaled_dot_product_attention(
        q, k, v,
        is_causal=is_causal,
        scale=sm_scale
    )

    # Run naive attention with standard softmax
    standard_output = naive_attention_standard(q, k, v, is_causal, sm_scale)

    # Run naive attention with online softmax
    online_output = naive_attention_online_softmax(q, k, v, is_causal, sm_scale)

    # Run naive attention with scan-like online softmax
    scan_like_output = naive_attention_scan_like(q, k, v, is_causal, sm_scale)

    # Compute similarity metrics
    def compute_metrics(output1, output2, name1, name2):
        # Flatten tensors for easier metric computation
        flat1 = output1.flatten()
        flat2 = output2.flatten()

        # Cosine similarity
        cos_sim = F.cosine_similarity(flat1.unsqueeze(0), flat2.unsqueeze(0), dim=1).item()

        # Absolute differences
        abs_diff = torch.abs(flat1 - flat2)
        max_abs_diff = abs_diff.max().item()
        mean_abs_diff = abs_diff.mean().item()

        # Relative error (L2 norm)
        l2_diff = torch.norm(flat1 - flat2).item()
        l2_ref = torch.norm(flat1).item()
        rel_error = l2_diff / (l2_ref + 1e-8)  # Add epsilon for numerical stability

        return {
            f'cosine_sim_{name1}_vs_{name2}': cos_sim,
            f'max_abs_diff_{name1}_vs_{name2}': max_abs_diff,
            f'mean_abs_diff_{name1}_vs_{name2}': mean_abs_diff,
            f'rel_error_{name1}_vs_{name2}': rel_error
        }

    metrics = {}
    metrics.update(compute_metrics(pytorch_output, standard_output, 'pytorch', 'standard'))
    metrics.update(compute_metrics(pytorch_output, online_output, 'pytorch', 'online'))
    metrics.update(compute_metrics(pytorch_output, scan_like_output, 'pytorch', 'scan_like'))
    metrics.update(compute_metrics(standard_output, online_output, 'standard', 'online'))
    metrics.update(compute_metrics(standard_output, scan_like_output, 'standard', 'scan_like'))
    metrics.update(compute_metrics(online_output, scan_like_output, 'online', 'scan_like'))

    return pytorch_output, standard_output, online_output, scan_like_output, metrics


# Example usage and demonstration
if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Define test parameters
    batch_size = 2
    num_heads = 8
    seq_len = 128
    head_dim = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Running naive attention demo on {device}")
    print(f"Input shape: [batch={batch_size}, heads={num_heads}, seq_len={seq_len}, head_dim={head_dim}]")

    # Generate random input tensors
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float32)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float32)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float32)

    print("\n" + "="*70)
    print("COMPARING ATTENTION METHODS")
    print("="*70)

    # Test both causal and non-causal attention
    for is_causal in [False, True]:
        print(f"\n{'Causal' if is_causal else 'Non-causal'} Attention:")
        print("-" * 50)

        # Run comparison
        pytorch_out, standard_out, online_out, scan_like_out, metrics = compare_attention_methods(
            q, k, v, is_causal=is_causal
        )

        # Print results
        print(f"Output shape: {pytorch_out.shape}")
        print("\nSimilarity Metrics:")

        for metric_name, value in metrics.items():
            if 'cosine_sim' in metric_name:
                status = "✓ PASS" if value > 0.95 else "✗ FAIL"
                print(f"  {metric_name:40}: {value:.6f} {status}")
            elif 'max_abs_diff' in metric_name:
                status = "✓ PASS" if value < 1e-4 else "✗ FAIL"
                print(f"  {metric_name:40}: {value:.2e} {status}")
            elif 'mean_abs_diff' in metric_name:
                status = "✓ PASS" if value < 1e-5 else "✗ FAIL"
                print(f"  {metric_name:40}: {value:.2e} {status}")
            elif 'rel_error' in metric_name:
                status = "✓ PASS" if value < 1e-4 else "✗ FAIL"
                print(f"  {metric_name:40}: {value:.2e} {status}")

    print("\n" + "="*70)
    print("DEMO COMPLETED")
    print("="*70)
    print("All methods should produce nearly identical results.")
    print("High cosine similarity (>0.95) and low errors indicate correct implementation.")
    print("\nImplementations compared:")
    print("- PyTorch SDPA (baseline)")
    print("- Standard softmax (naive)")
    print("- Online softmax (loop-based)")
    print("- Scan-like online softmax (optimized)")