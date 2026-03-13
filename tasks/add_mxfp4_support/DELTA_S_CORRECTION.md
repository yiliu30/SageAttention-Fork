# δS Correction — Detailed Explanation with Shapes

## 1. Why δS Exists

NVFP4 (E2M1) has only 4 bits of precision, so quantizing Q directly loses significant accuracy. SageAttention3 improves this by **smoothing Q** — subtracting the per-block mean — before quantization. Smooth Q has smaller dynamic range and quantizes more faithfully.

But this changes the attention scores: $Q_{\text{smooth}} K^T \neq Q K^T$. The correction term **δS** restores the exact value.

---

## 2. Mathematical Derivation

Assume $Q, K \in \mathbb{R}^{N \times d}$, block size $B_M = 128$.

### Decomposition

For each block $b$ of 128 rows (rows $128b$ to $128(b+1)-1$):

$$\bar{Q}_b = \frac{1}{128} \sum_{i=128b}^{128(b+1)-1} Q_i \quad \in \mathbb{R}^{1 \times d}$$

$$Q^{\text{smooth}}_i = Q_i - \bar{Q}_b \quad \text{for } i \in [128b,\; 128(b+1))$$

### Attention Score Recovery

For row $i$ in block $b$, and any column $j \in [0, N)$:

$$S_{ij} = Q_i \cdot K_j^T = (Q^{\text{smooth}}_i + \bar{Q}_b) \cdot K_j^T$$

$$= \underbrace{Q^{\text{smooth}}_i \cdot K_j^T}_{\text{FP4 GEMM (in-kernel)}} + \underbrace{\bar{Q}_b \cdot K_j^T}_{\delta S_{bj}}$$

The correction term $\delta S_{bj} = \bar{Q}_b \cdot K_j^T$ depends only on **block index $b$** and **column $j$**, not on the specific row $i$ within the block. This is the key insight that enables efficient broadcasting.

---

## 3. Shapes Through the Pipeline

Assume: `B=1, H=8, N=256, d=128, kBlockM=128, kBlockN=64`.

### 3.1 Pre-Kernel (Python — `api.py`)

```python
# Input
q:  [B, H, N, d] = [1, 8, 256, 128]   # Original query
k:  [B, H, N, d] = [1, 8, 256, 128]   # Key (already mean-centered along seq dim)

# Step 1: Group mean (Triton kernel, GROUP_SIZE=128)
q_smooth: [B, H, N, d]       = [1, 8, 256, 128]   # Q with per-block mean subtracted
qm:       [B, H, N/128, d]   = [1, 8, 2, 128]     # Per-block mean vectors

# Step 2: δS computation (PyTorch matmul)
delta_s = torch.matmul(qm, k.transpose(-2, -1)).to(torch.float32)
#         [1, 8, 2, 128] @ [1, 8, 128, 256] → [1, 8, 2, 256]
#         [B, H, N/128, d] @ [B, H, d, N]   → [B, H, N/128, N]
```

**Shape of δS**: `[B, H, N/128, N]` — one value per (block, key-position) pair.

For each M-tile (128 rows), the kernel needs one row of δS with width N. Within a K-tile of width `kBlockN=64`, that's a slice of shape `[1, 64]` → broadcast across 128 rows.

### 3.2 GMEM Layout — `LayoutDS`

```cpp
SmemLayoutAtomDS = Layout<Shape<Int<kBlockM>, Int<kBlockN>>,   // (128, 64)
                          Stride<_0, _1>>;                     // stride-0 on M → broadcast
```

When tiled to the full tensor:

```cpp
LayoutDS layout_ds = tile_to_shape(
    SmemLayoutAtomDS{},                                          // (128, 64), stride (0, 1)
    make_shape(Seqlen_Q, Seqlen_K, HeadNum, Batch),             // (256, 256, 8, 1)
    Step<_2, _1, _3, _4>{}                                      // tiling order
);
// Result shape: (256, 256, 8, 1)
// But with stride-0 on the M-within-block axis → only N/128 distinct rows stored
```

When `BlockMean=true`, local_tile indexes by `m_block`:
```cpp
gDS = local_tile(mDS(_, _, bidh, bidb),
                 select<0, 1>(TileShape_MNK{}),           // tile shape (128, 64)
                 make_coord(m_block, _));                  // m_block selects which block's mean
// gDS shape: (128, 64, n_tiles)
// But stride-0 on dim 0 → all 128 rows read the same 64 floats
```

### 3.3 TMA Load (bundled with K pipeline)

δS is loaded into SMEM alongside K data and SFK scales as part of `pipeline_k`:

```cpp
// In load():
copy(mainloop_params.tma_load_DS.with(
         *pipeline_k.producer_get_barrier(smem_pipe_write_k), mcast_mask_kv),
     tDSgDS(_, n_block),
     tDSsDS(_, smem_pipe_write_k.index()));
```

**Per-tile TMA transfer**: `kBlockM × kBlockN × sizeof(float)` = `128 × 64 × 4` = **32,768 bytes**.

But with stride-0 on M, the TMA hardware only fetches the unique data: effectively `1 × 64 × 4` = **256 bytes**, broadcast across 128 rows in SMEM.

### 3.4 SMEM Layout

```cpp
SmemLayoutDS = tile_to_shape(SmemLayoutAtomDS{},
    make_shape(kBlockM, kBlockN, kStages),
    ...);
// Shape: (128, 64, kStages)
// Stride on dim 0 = 0 → all 128 "rows" alias the same 64 floats
// Effective storage: 64 × kStages floats per stage
```

### 3.5 In-Kernel: `add_delta_s(acc)` — Register Loading

```cpp
auto add_delta_s = [&](auto& acc) {
    // Recast SMEM and accumulator as float4 for efficient 128-bit loads
    auto tSsDS_stage = recast<float4>(sDS(_, _, smem_pipe_read_k.index()));
    auto acc_float4  = recast<float4>(acc);

    int quad_id = (threadIdx.x % 4) * 2;  // 0, 2, 4, 6

    for (int i = 0; i < 4; i++) {
        auto num = quad_id + i * 8;  // Accesses: 0,8,16,24 / 2,10,18,26 / 4,12,20,28 / 6,14,22,30
        float4 delta_s_0 = tSsDS_stage(make_coord(_0{}, _0{}), make_coord(num, _0{}));
        float4 delta_s_1 = tSsDS_stage(make_coord(_0{}, _0{}), make_coord(num + 1, _0{}));

        // Write the same δS values to 4 accumulator positions
        // (duplicated across the M dimension within the MMA tile)
        acc_float4(make_coord(make_coord(_0{}, _0{}), _0{}), _0{}, i) = delta_s_0;
        acc_float4(make_coord(make_coord(_0{}, _0{}), _1{}), _0{}, i) = delta_s_0;  // same!
        acc_float4(make_coord(make_coord(_0{}, _1{}), _0{}), _0{}, i) = delta_s_1;
        acc_float4(make_coord(make_coord(_0{}, _1{}), _1{}), _0{}, i) = delta_s_1;  // same!
    }
};
```

#### Shape Analysis of the Accumulator

The accumulator `tSrS` holds S = Q@K^T for one tile:

```
tSrS = partition_fragment_C(tiled_mma_qk, (kBlockM, kBlockN))
     = per-thread fragment of shape ~ (MMA_M_per_thread, MMA_N_per_thread)
```

After `recast<float4>`, each `float4` covers 4 consecutive columns of S. The indexing pattern:

| Index expression | What it selects |
|---|---|
| `make_coord(_0{}, _0{})` in dim 0 | Row group 0, sub-row 0 |
| `make_coord(_0{}, _1{})` in dim 0 | Row group 0, sub-row 1 (same block → same δS) |
| `make_coord(_1{}, _0{})` in dim 0 | Row group 1, sub-row 0 |
| `make_coord(_1{}, _1{})` in dim 0 | Row group 1, sub-row 1 (same block → same δS) |
| `_0{}` in dim 1 | Always 0 (single MMA-N partition) |
| `i` in dim 2 | MMA-M repeat index (0..3) |

The duplication (`_0{}` and `_1{}` get the same value) reflects the M-broadcast: rows within the same 128-row block share the same δS values.

---

## 4. How δS Feeds Into the GEMM

The accumulator is initialized to δS **instead of zero**:

```cpp
// Normal GEMM:         acc  = 0 + Q_smooth @ K^T
// With δS correction:  acc  = δS + Q_smooth @ K^T  =  Q @ K^T
```

This is done by calling `add_delta_s(tSrS)` **before** the Q@K GEMM loop:

```cpp
consumer_wait(pipeline_k, smem_pipe_read_k);
copy_k_block(_0{});
add_delta_s(tSrS);       // ← acc = δS (not zero!)

for (int k_block = 0; k_block < size<2>(tSrQ); ++k_block) {
    gemm(Q_block, K_block, tSrS);   // acc += Q_smooth_block @ K_block^T
}
// Now tSrS = δS + Q_smooth @ K^T = Q @ K^T  ✓
```

No extra pass over the accumulator is needed — the correction is "free" relative to a zero-initialization.

---

## 5. Cost Summary

| Component | Cost | When |
|-----------|------|------|
| **Pre-kernel GEMV** | `[B,H,N/128,d] @ [B,H,d,N]` in FP32 | Once, before kernel launch |
| **GMEM storage** | `B × H × (N/128) × N × 4` bytes | Allocated before kernel |
| **TMA load per tile** | ~256 bytes (stride-0 broadcast) | Pipelined with K load |
| **Register writes** | ~32 `float4` stores per thread | Before each Q@K GEMM |

### Concrete Numbers (B=1, H=8, N=4096, d=128)

| Component | Value |
|-----------|-------|
| GEMV shape | `[1,8,32,128] @ [1,8,128,4096]` |
| GEMV FLOPs | `8 × 32 × 128 × 4096 × 2` ≈ 268M FLOPs |
| GMEM for δS | `8 × 32 × 4096 × 4` = 4 MB |
| TMA per tile | 256 bytes (negligible) |

The GEMV cost is small relative to the attention kernel itself (which is $O(N^2 d)$), and it runs as a standard PyTorch `matmul` overlapping with quantization kernels.

---

## 6. K Mean-Centering Interaction

Before δS computation, K is also mean-centered:

```python
k -= k.mean(dim=-2, keepdim=True)   # [B, H, N, d] - [B, H, 1, d]
```

This ensures $\sum_j K_j = 0$, which helps with:
1. **Better FP4 quantization of K** — centering reduces outliers
2. **Numeric stability** — the δS values are smaller since $\bar{Q}_b \cdot \bar{K} = 0$

The combination of Q-smoothing + K-centering means:
- Q is quantized as `Q_smooth = Q - block_mean(Q)` → reduced dynamic range per block
- K is quantized as `K_centered = K - global_mean(K)` → reduced global range
- δS corrects for Q's block mean; K's global mean doesn't affect relative attention scores (it cancels in softmax)

---

## 7. Diagram

```
Pre-kernel:
                      ┌──────────────────────┐
  Q [B,H,N,d] ──────►│ group_mean(Q, 128)   │──► Q_smooth [B,H,N,d]     → quantize to FP4
                      │                      │──► qm [B,H,N/128,d]
                      └──────────────────────┘         │
                                                       │  matmul
  K [B,H,N,d] ────────────────────────────────────────►│
  (mean-centered)                                      ▼
                                                δS [B,H,N/128,N]  (FP32)
                                                       │
                                                       ▼ store to GMEM

In-kernel (per M-tile, per K-tile):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  TMA loads δS slice [1, kBlockN] into SMEM (broadcast across M)   │
  │                                                                     │
  │  add_delta_s(acc):  acc[all_rows, :] = δS[b, n_block*64 : (n+1)*64]│
  │                                                                     │
  │  GEMM loop:         acc += Q_smooth_fp4 @ K_fp4^T                  │
  │                                                                     │
  │  Result:            acc = δS + Q_smooth@K^T = Q@K^T                │
  │                              ↓                                      │
  │                         softmax(acc) → P → quantize → P@V          │
  └─────────────────────────────────────────────────────────────────────┘
```
