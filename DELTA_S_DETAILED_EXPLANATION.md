# Delta_s: Complete Q+K Smoothing Analysis

## Overview

The `delta_s` correction in SageAttention3 is a sophisticated two-part smoothing mechanism that addresses quantization outliers in both Q (queries) and K (keys) tensors. This document provides a comprehensive analysis with visual diagrams.

## Part 1: Why Delta_s is Needed

### The Quantization Challenge

NVFP4 (E2M1) has only 4 bits of precision with representable values: `±{0, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6}`. Direct quantization of Q and K tensors loses significant accuracy due to:

1. **High dynamic range** in raw Q/K values
2. **Outlier values** that dominate the quantization scale
3. **Per-token variance** that makes global scaling ineffective

### SageAttention3's Solution: Two-Level Smoothing

```
Original Problem:
Q_raw: [large_vals, outliers, normal_vals] → FP4 quantization → poor accuracy

SageAttention3 Approach:
Q_raw → Q_smoothing → Q_smooth (smaller range) → FP4 quantization → good accuracy
     ↘ correction term (delta_s) → add back during computation → exact result
```

## Part 2: Mathematical Foundation

### The Core Identity

SageAttention3 maintains mathematical equivalence through this identity:

```
Q @ K^T = (Q_smooth + Q_correction) @ (K_smooth + K_correction)^T
        = Q_smooth @ K_smooth^T + correction_terms

Where correction_terms = delta_s (computed efficiently)
```

### Two Smoothing Components

#### Component 1: K Global Centering
```python
# Step 1: Center K globally (reduces outliers across all tokens)
k_mean = k.mean(dim=-2, keepdim=True)  # [B, H, 1, D]
k_centered = k - k_mean                # [B, H, N, D]
```

#### Component 2: Q Per-Block Smoothing
```python
# Step 2: Smooth Q in 128-token blocks (reduces outliers per block)
# Divide sequence into groups of 128 tokens
num_groups = (N + 127) // 128
q_grouped = q.reshape(B, H, num_groups, 128, D)
q_block_means = q_grouped.mean(dim=3, keepdim=True)  # [B, H, num_groups, 1, D]
q_smoothed = q_grouped - q_block_means               # [B, H, num_groups, 128, D]
```

### Delta_s Computation

The correction term comes from the cross-interaction between Q block means and K centering:

```python
# Compute correction for each (Q_block, K_position) pair
q_means = q_block_means.squeeze(3)  # [B, H, num_groups, D]
delta_s = q_means @ k_centered.transpose(-2, -1)  # [B, H, num_groups, N]

# Mathematical meaning:
# delta_s[b, h, group_i, pos_j] = q_block_mean[group_i] • k_centered[pos_j]
```

## Part 3: Visual Diagrams

### Diagram 1: Input Tensor Structure

```
Input Tensors:
┌─────────────────────────────────────────────────────────────┐
│ Q: [B=1, H=8, N=256, D=128]  ┌─ Head 0 ─┐┌─ Head 1 ─┐    │
│    ┌─────────────────────────┐│ Token 0  ││ Token 0  │    │
│    │    Per-token vectors    ││ Token 1  ││ Token 1  │    │
│    │    D=128 dimensions     ││   ...    ││   ...    │    │
│    │                         ││ Token 255││ Token 255│    │
│    └─────────────────────────┘└──────────┘└──────────┘    │
├─────────────────────────────────────────────────────────────┤
│ K: [B=1, H=8, N=256, D=128]  (Same structure as Q)         │
└─────────────────────────────────────────────────────────────┘
```

### Diagram 2: K Global Centering

```
K Centering (Component 1):
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Compute Global Mean                                 │
│ ┌─────────────────────┐                                     │
│ │ K: [B, H, N=256, D] │ ──reduce_mean──► k_mean: [B,H,1,D] │
│ │ ┌─ Token 0 ─┐       │    (dim=-2)                        │
│ │ │ Token 1   │       │         │                          │
│ │ │   ...     │       │         ▼                          │
│ │ │ Token 255 │       │ ┌─────────────────┐                │
│ │ └───────────┘       │ │ k_centered =    │                │
│ └─────────────────────┘ │ k - k_mean      │                │
│                         │ [B, H, N, D]    │                │
│                         └─────────────────┘                │
│ Effect: Reduces global outliers, better FP4 quantization   │
└─────────────────────────────────────────────────────────────┘
```

### Diagram 3: Q Per-Block Smoothing

```
Q Block Smoothing (Component 2):
┌─────────────────────────────────────────────────────────────┐
│ Step 2a: Divide into 128-token blocks                      │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Q: [B, H, N=256, D] → [B, H, 2 groups, 128, D]        │ │
│ │                                                         │ │
│ │ ┌─── Group 0 ────┐  ┌─── Group 1 ────┐               │ │
│ │ │ Token 0-127    │  │ Token 128-255   │               │ │
│ │ │ [B,H,128,D]    │  │ [B,H,128,D]     │               │ │
│ │ └────────────────┘  └─────────────────┘               │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ Step 2b: Per-block mean subtraction                        │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ For each 128-token block:                               │ │
│ │   q_block_mean = mean(q_block, dim=tokens)              │ │
│ │   q_smooth_block = q_block - q_block_mean               │ │
│ │                                                         │ │
│ │ Result: q_smoothed [B, H, N, D] (reshape back)         │ │
│ │         q_means [B, H, num_groups=2, D]                │ │
│ └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Diagram 4: Delta_s Computation and Application

```
Delta_s Computation and Usage:
┌─────────────────────────────────────────────────────────────────────┐
│ Step 3: Compute correction term                                     │
│ ┌─────────────────────┐   ┌─────────────────────┐                   │
│ │ q_means:            │ @ │ k_centered^T:       │ = delta_s         │
│ │ [B,H,2,D=128]      │   │ [B,H,D=128,N=256]  │   [B,H,2,N=256]   │
│ │                     │   │                     │                   │
│ │ ┌─ Group 0 mean ─┐  │   │ ┌─ All K positions ┐ │                   │
│ │ │ Group 1 mean   │  │   │ │   transposed     │ │                   │
│ │ └────────────────┘  │   │ └──────────────────┘ │                   │
│ └─────────────────────┘   └─────────────────────┘                   │
│                                                                     │
│ Delta_s Shape Analysis:                                             │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │ delta_s[b, h, group_i, pos_j] = q_mean[group_i] • k_centered[j] │ │
│ │                                                                 │ │
│ │ Group 0: [correction for tokens 0-127   vs all K positions]    │ │
│ │ Group 1: [correction for tokens 128-255 vs all K positions]    │ │
│ └─────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘

Step 4: Apply during attention computation
┌─────────────────────────────────────────────────────────────────────┐
│ For each attention tile (Q_tile @ K_tile^T):                       │
│                                                                     │
│ Original:    QK_scores = Q_tile @ K_tile^T                         │
│ With delta:  QK_scores = Q_smooth_tile @ K_smooth_tile^T + delta_s  │
│                                                                     │
│ ┌─── Q_tile ───┐   ┌─── K_tile^T ──┐   ┌── delta_s_tile ──┐        │
│ │ [tile_M, D]  │ @ │ [D, tile_N]   │ + │ [tile_M, tile_N] │        │
│ │              │   │               │   │ (broadcast from  │        │
│ │ Q_smoothed   │   │ K_smoothed    │   │  appropriate     │        │
│ │ FP4 quantized│   │ FP4 quantized │   │  delta_s slice)  │        │
│ └──────────────┘   └───────────────┘   └──────────────────┘        │
│                                                                     │
│ Result: Mathematically equivalent to original Q @ K^T               │
│         but with much better FP4 quantization accuracy!             │
└─────────────────────────────────────────────────────────────────────┘
```

## Part 4: Implementation Details

### Shape Transformations

```python
# Input shapes
q: [B, H, N, D] = [1, 8, 256, 128]
k: [B, H, N, D] = [1, 8, 256, 128]

# Step 1: K centering
k_mean: [B, H, 1, D] = [1, 8, 1, 128]
k_centered: [B, H, N, D] = [1, 8, 256, 128]

# Step 2: Q block grouping (N=256, block_size=128)
num_groups = (256 + 127) // 128 = 2
q_grouped: [B, H, num_groups, block_size, D] = [1, 8, 2, 128, 128]
q_means: [B, H, num_groups, D] = [1, 8, 2, 128]
q_smoothed: [B, H, N, D] = [1, 8, 256, 128]  # (reshaped back)

# Step 3: Delta_s computation
delta_s = q_means @ k_centered.transpose(-2, -1)
        = [1, 8, 2, 128] @ [1, 8, 128, 256]
        = [1, 8, 2, 256]  # [B, H, num_groups, N]
```

### Memory Layout in CUDA Kernel

```cpp
// CUDA kernel memory layout
SmemLayoutDS = Layout<Shape<Int<kBlockM>, Int<kBlockN>>,    // (128, 64)
                     Stride<_0, _1>>;                      // stride-0 on M

// Interpretation:
// - 128 rows (M dimension): all rows in a block share same delta_s values
// - 64 cols (N dimension): portion of sequence being processed
// - Stride-0 on M: broadcast same values across all 128 query positions
```

## Part 5: Quantitative Benefits

### Quantization Quality Improvement

```python
# Without smoothing (poor quantization)
q_raw_range = q.max() - q.min()  # Large dynamic range
q_fp4_error = quantize_fp4(q_raw) - q_raw  # High quantization error

# With smoothing (good quantization)
q_smooth_range = q_smooth.max() - q_smooth.min()  # Smaller range per block
q_smooth_fp4_error = quantize_fp4(q_smooth) - q_smooth  # Lower error
```

### Performance Impact

```
Component          | Cost                    | When
-------------------|-------------------------|------------------------
K centering        | O(N*D) per head        | Once, pre-kernel
Q block smoothing  | O(N*D) per head        | Once, pre-kernel
Delta_s GEMV       | O((N/128)*D*N)         | Once, pre-kernel
Delta_s TMA load   | ~256 bytes/tile        | Per tile (pipelined)
Delta_s injection  | Register writes        | Per tile (nearly free)

Total overhead: ~5-10% of attention computation
Accuracy gain: Maintains >99% similarity vs full precision
```

## Part 6: Why Two Components?

### K Global Centering Benefits
1. **Reduces global outliers** across the entire sequence
2. **Improves K quantization** for all tokens uniformly
3. **Numerical stability** in softmax (centered distributions)
4. **Zero-sum property**: `∑k_centered = 0` helps with delta_s computation

### Q Per-Block Smoothing Benefits
1. **Local outlier reduction** within 128-token windows
2. **Adaptive to content**: Each block gets its own centering
3. **Hardware alignment**: 128 tokens = GPU warp/block size
4. **Memory efficiency**: Block-wise processing in CUDA kernels

### Why Not Global Q Centering?

```
Problem with global Q centering:
- q_global_mean affects ALL tokens equally
- Loses fine-grained adaptation to local content patterns
- 128-token blocks better match GPU memory hierarchy
- Per-block means can be computed efficiently in parallel
```

## Part 7: Edge Cases and Robustness

### Short Sequences (N < 128)
```python
if N < 128:
    # Fallback to global smoothing
    q_mean = q.mean(dim=-2, keepdim=True)  # [B, H, 1, D]
    q_smoothed = q - q_mean
    delta_s = q_mean @ k_centered.transpose(-2, -1)  # [B, H, 1, N]
```

### Non-Multiple of 128 Sequences
```python
# Padding strategy
pad_len = (128 - N % 128) % 128
q_padded = F.pad(q, (0, 0, 0, pad_len))  # Pad sequence dimension
# ... process with padding ...
q_smoothed = q_smoothed_padded[..., :N, :]  # Remove padding
```

### Numerical Stability
```python
# Use FP32 for delta_s computation (even when Q/K are FP16)
delta_s = torch.matmul(
    q_means.to(torch.float32),
    k_centered.to(torch.float32).transpose(-2, -1)
).to(torch.float32)  # Maintain precision for correction terms
```

## Conclusion

The `delta_s` correction in SageAttention3 is a sophisticated two-component system:

1. **K global centering**: Reduces outliers across the entire sequence
2. **Q per-block smoothing**: Adapts to local content patterns in 128-token windows

Together, these create smoothed tensors that quantize much better to FP4, while the `delta_s` correction ensures mathematical equivalence with the original computation. This elegant approach achieves the impossible: FP4 quantization with near-lossless accuracy.

The key insight is **separating quantization-friendly computation from mathematical correctness** - letting the hardware do efficient FP4 math on smooth tensors while maintaining exact results through additive corrections.