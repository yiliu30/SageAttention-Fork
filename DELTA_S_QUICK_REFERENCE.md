# Delta_s Quick Reference

## TL;DR: The Two-Part Smoothing System

```
┌─────────────────────────────────────────────────────────────┐
│                    DELTA_S OVERVIEW                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Original Problem: Q, K → FP4 quantization → poor accuracy │
│                                                             │
│  SageAttention3 Solution: Two-level smoothing               │
│                                                             │
│  ┌─ COMPONENT 1: K Global Centering ─────────────────────┐  │
│  │                                                       │  │
│  │  k_centered = k - mean(k, dim=sequence)              │  │
│  │  • Reduces outliers across ALL tokens                │  │
│  │  • Better K quantization globally                    │  │
│  │  • Shape: [B,H,N,D] → [B,H,N,D]                     │  │
│  │                                                       │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌─ COMPONENT 2: Q Per-Block Smoothing ──────────────────┐  │
│  │                                                       │  │
│  │  q_smooth[block] = q[block] - mean(q[block])         │  │
│  │  • Block size = 128 tokens                           │  │
│  │  • Adapts to local content patterns                  │  │
│  │  • Shape: [B,H,N,D] → groups → [B,H,N,D]            │  │
│  │                                                       │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌─ CORRECTION: Delta_s Computation ──────────────────────┐  │
│  │                                                       │  │
│  │  delta_s = q_block_means @ k_centered^T              │  │
│  │  • Restores mathematical equivalence                 │  │
│  │  • Added during QK^T computation                     │  │
│  │  • Shape: [B,H,num_blocks,D] @ [B,H,D,N]            │  │
│  │           = [B,H,num_blocks,N]                       │  │
│  │                                                       │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  Result: Q_smooth @ K_smooth^T + delta_s = Q @ K^T         │
│          ↑ FP4 quantized      ↑ exact    ↑ original        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Shape Flow Diagram

```
Input Tensors:
  Q: [B, H, N, D] ──┐
  K: [B, H, N, D] ──┤
                    │
                    ▼
┌─────────────────────────────────────┐
│         SMOOTHING STAGE             │
├─────────────────────────────────────┤
│                                     │
│ K Processing:                       │
│   k_mean: [B,H,1,D]               │
│   k_centered: [B,H,N,D]            │
│                                     │
│ Q Processing (N=256, blocks=128):   │
│   q_grouped: [B,H,2,128,D]         │
│   q_means: [B,H,2,D]               │
│   q_smoothed: [B,H,N,D]            │
│                                     │
└─────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────┐
│       CORRECTION STAGE              │
├─────────────────────────────────────┤
│                                     │
│ delta_s = q_means @ k_centered^T    │
│         = [B,H,2,D] @ [B,H,D,N]     │
│         = [B,H,2,N]                 │
│                                     │
│ Meaning:                            │
│   delta_s[b,h,0,:] → Block 0 vs All│
│   delta_s[b,h,1,:] → Block 1 vs All│
│                                     │
└─────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────┐
│        ATTENTION STAGE              │
├─────────────────────────────────────┤
│                                     │
│ For each tile (i,j):                │
│   QK = Q_smooth[i] @ K_smooth[j]^T  │
│   QK = QK + delta_s[block_i, cols_j]│
│   P = softmax(QK)                   │
│   Out = P @ V[j]                    │
│                                     │
└─────────────────────────────────────┘
```

## Key Insights

### 1. Why Two Components?
- **K centering**: Global outlier reduction (affects all tokens)
- **Q smoothing**: Local adaptation (per 128-token block)
- **Combined**: Better quantization than either alone

### 2. Block Size = 128
- Matches GPU warp/memory hierarchy
- Good balance: not too local (1 token) or too global (full sequence)
- Enables efficient CUDA implementation

### 3. Delta_s Broadcasting Pattern
```
delta_s shape: [B, H, num_blocks, N]

In CUDA kernel for tile (block_i, cols_j):
- All 128 rows in block_i use same delta_s values
- Columns j:(j+64) get delta_s[block_i, j:(j+64)]
- Stride-0 memory layout enables efficient broadcasting
```

### 4. Mathematical Equivalence
```
Original:     Q @ K^T
Decomposed:   (Q_smooth + Q_means) @ (K_centered + K_mean)^T
Simplified:   Q_smooth @ K_centered^T + Q_means @ K_centered^T + ...
Key insight:  Many cross-terms cancel or are handled by delta_s!
```

### 5. Performance Trade-offs
- **Cost**: ~5-10% overhead for smoothing computation
- **Benefit**: Enables FP4 quantization with >99% accuracy
- **Memory**: delta_s storage scales as O(N^2/128) vs O(N^2) for full precision

This two-component design is what makes SageAttention3's FP4 quantization so effective!