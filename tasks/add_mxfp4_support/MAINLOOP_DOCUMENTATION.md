# SageAttention3 Blackwell Mainloop Kernel Documentation

## Overview

`mainloop_tma_ws.h` implements the core fused attention kernel for SageAttention3 on Blackwell (SM120) GPUs. It performs **FlashAttention-style tiled attention** entirely in NVFP4 (E2M1) with block-scaled MMA, achieving ~5× speedup over FlashAttention2.

The kernel computes:

$$O = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d}}\right) \cdot V$$

with all GEMMs executed in NVFP4 via `SM120::BLOCKSCALED::SM120_16x32x64_TN_VS_NVFP4`.

---

## Architecture: Producer–Consumer Pipeline

The kernel uses a **warp-specialized** design with two roles:

| Role | Warps | Responsibility |
|------|-------|----------------|
| **Producer** | 1 warp (TMA warp) | Async TMA loads from GMEM → SMEM |
| **Consumer** | 7–11 warps (MMA warps) | GEMM, softmax, quantization |

Three **independent TMA pipelines** (each with `kStages` buffers) overlap data loading with compute:

| Pipeline | Data loaded | Pipelined with |
|----------|-------------|----------------|
| `pipeline_q` | Q data + SFQ scales | Loaded once, consumed by all tiles |
| `pipeline_k` | K data + SFK scales + δS correction | Each k-tile loaded ahead of consumption |
| `pipeline_v` | V^T data + SFV scales | Each v-tile loaded ahead of consumption |

---

## Key Data Types and Shapes

| Item | Type | Shape (typical: M=128, N=64, d=128) |
|------|------|------|
| Q, K, V data | `float_e2m1_t` (E2M1, FP4) | Quantized before kernel |
| Scale factors (SF) | `float_ue4m3_t` (E4M3, FP8) | 1 per 16 elements |
| P (attention weights) | `float_e2m1_t` | Quantized in-kernel |
| SFP (P's scale factors) | `float_ue4m3_t` | Computed in-kernel |
| Accumulators | `float` (FP32) | All GEMMs accumulate in FP32 |
| δS correction | `float` (FP32) | Pre-computed, loaded via TMA |

---

## `CollectiveMainloopFwd` Structure

```
template <typename Ktraits, bool Is_causal>
struct CollectiveMainloopFwd
```

### Template Parameters
- `Ktraits` — `Flash_fwd_kernel_traits<kHeadDim, kBlockM, kBlockN, kStages, ...>` from `kernel_traits.h`
- `Is_causal` — enables causal masking (upper-triangular mask on S)

### Key Type Aliases

| Alias | Meaning |
|-------|---------|
| `TileShape_MNK` | `Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>`, e.g. `(128, 64, 128)` |
| `TiledMmaQK` | Block-scaled MMA for Q@K^T GEMM |
| `TiledMmaPV` | Block-scaled MMA for P@V GEMM (same MMA atom, different permutation tile) |
| `LayoutP` | Register layout for P: `((8,2,2), 1, kBlockN/64)` with stride `((1,8,16), 0, 32)` |
| `LayoutSFP` | Register layout for P's scale factors: `((16,4), 1, kBlockN/64)` with stride `((0,1), 0, 4)` |
| `SmemLayoutAtomDS` | `Layout<Shape<kBlockM, kBlockN>, Stride<0, 1>>` — broadcast across M (row-invariant) |

---

## `load()` — Producer Thread (TMA Loads)

```cpp
void load(Params const& mainloop_params, ..., SharedStorage& shared_storage, ...)
```

### Execution Flow

1. **Load Q once**: Q data + SFQ scales → SMEM via `pipeline_q` (single-stage, loaded once per tile-row)
2. **Load first K+δS tile**: K data + SFK scales + δS correction → SMEM via `pipeline_k`
3. **Load first V tile**: V^T data + SFV^T scales → SMEM via `pipeline_v`
4. **Loop over remaining K/V tiles** (n_block from `n_block_max-2` down to 0):
   - Acquire `pipeline_k`, issue TMA for K + SFK + δS
   - Acquire `pipeline_v`, issue TMA for V^T + SFV^T

### TMA Transaction Sizes

```
TmaTransactionBytesQ = sizeof(Q_tile) + sizeof(SFQ)
TmaTransactionBytesK = sizeof(K_tile) + sizeof(SFK) + sizeof(DS_tile)
TmaTransactionBytesV = sizeof(Vt_tile) + sizeof(SFVt)
```

δS is bundled into the K pipeline — they share the same pipeline barrier, ensuring δS arrives with K.

---

## `mma()` — Consumer Threads (Compute)

```cpp
void mma(Params const& mainloop_params, ..., SharedStorage& shared_storage)
```

This is the core compute function. It processes tiles from right to left (`n_block` from `n_block_max-1` down to 0) in three phases.

### Phase 1: First Tile (with boundary masking)

```
consumer_wait(pipeline_q)
copy Q + SFQ from SMEM → registers
release pipeline_q

consumer_wait(pipeline_k)
copy_k_block(0)
add_delta_s(tSrS)                    // Initialize accumulator to δS
for k_block in 0..size<2>(tSrQ):     // GEMM Q@K^T
    gemm(Q_block, K_block → tSrS)    // Accumulates onto δS
    copy_k_block(next) or release K

apply causal + boundary masking to tSrS
online_softmax_with_quant<FirstTile>(tSrS, AbsMaxP)

consumer_wait(pipeline_v)
copy_v_block(0)
quantize(0, tSrS)                    // P = quantize(softmax(S))
for v_block in 0..size<2>(tOrP):     // GEMM P@V
    gemm(P_block, V_block → tOrO_store)
    copy_v_block(next) + quantize(next) or release V
```

### Phase 2: Causal Masking Tiles (fully unrolled)

Only active when `Is_causal=true`. Up to `ceil(kBlockM/kBlockN)+1` tiles near the diagonal need per-element causal masking.

Same structure as Phase 1 but:
- Uses `Is_first=false` for softmax (tracks running max/sum)
- Creates a fresh `tOrO` accumulator, then calls `rescale_o(tOrO_store, tOrO)` to merge with rescaled previous output

### Phase 3: Main Loop (`#pragma unroll 1`)

All remaining tiles — no masking needed (all elements are valid).

```
for n_block >= 0:
    // Same sequence: load K → add_delta_s → GEMM QK → softmax → load V → quantize → GEMM PV → rescale_o
```

### Final Step

```cpp
softmax_fused.finalize(tOrO_store);  // Divide by row_sum
```

---

## Key Lambdas in `mma()`

### `copy_k_block(block_id)` — SMEM → Register for K

```cpp
auto copy_k_block = [&](auto block_id) {
    copy(smem_tiled_copy_K, tSsK_stage(_, _, block_id),  tSrK_copy_view(_, _, block_id));
    copy(smem_tiled_copy_SFK, tSsSFK_stage(_, _, block_id), tSrSFK_copy_view(_, _, block_id));
};
```

Copies both K data and K's scale factors from the current pipeline stage in SMEM to registers, for a specific MMA-K block (0-indexed within the tile).

### `copy_v_block(block_id)` — SMEM → Register for V

```cpp
auto copy_v_block = [&](auto block_id) {
    copy(smem_tiled_copy_V, tOsVt_stage(_, _, block_id),  tOrVt_copy_view(_, _, block_id));
    copy(smem_tiled_copy_SFV, tOsSFVt_stage(_, _, block_id), tOrSFVt_copy_view(_, _, block_id));
};
```

Same pattern as `copy_k_block` but for V^T data and V^T's scale factors.

### `add_delta_s(acc)` — δS Correction Initialization

```cpp
auto add_delta_s = [&](auto& acc) {
    auto tSsDS_stage = recast<float4>(sDS(_, _, smem_pipe_read_k.index()));
    auto acc_float4 = recast<float4>(acc);
    int quad_id = (threadIdx.x % 4) * 2;
    for (int i = 0; i < 4; i++) {
        auto num = quad_id + i * 8;
        float4 delta_s_0 = tSsDS_stage(make_coord(_0{}, _0{}), make_coord(num, _0{}));
        float4 delta_s_1 = tSsDS_stage(make_coord(_0{}, _0{}), make_coord(num + 1, _0{}));
        // Duplicate to 4 accumulator positions (rows sharing same δS values)
        acc_float4(...) = delta_s_0;  // broadcast across M dimension
        acc_float4(...) = delta_s_1;
    }
};
```

**Purpose**: Implements the Q-smoothing correction. Instead of `tSrS = 0 + Q@K^T`, the accumulator is initialized to `δS` so after GEMM it holds `δS + Q_smooth@K^T = Q@K^T`.

**Math**: Given Q is decomposed as `Q = Q_smooth + Q_mean` (per 128-row block mean subtracted):

$$S_{ij} = Q_i \cdot K_j = Q^{\text{smooth}}_i \cdot K_j + \bar{Q} \cdot K_j$$

where $\delta S_j = \bar{Q} \cdot K_j$ is the same for all rows in the block → `Stride<0, 1>` (broadcast across M).

**Pre-kernel computation** (in `api.py`):
```python
# qm: [B, H, N/128, d] — per-block mean of Q
# k:  [B, H, N, d]
delta_s = torch.matmul(qm, k.transpose(-2, -1)).to(torch.float32)  # [B, H, N/128, N]
```

**Cost**: One matrix multiply before the kernel + 256 bytes TMA per tile (pipelined with K load) + ~32 float4 register writes per thread.

### `quantize(mma_k, acc_conversion_view)` — In-Kernel P Quantization

```cpp
auto quantize = [&](auto mma_k, auto acc_conversion_view) { ... };
```

Converts the FP32 softmax output P into NVFP4 (E2M1) format with FP8 (E4M3) scale factors, readying it for the P@V GEMM.

**Three sub-steps**:

1. **Scale factors → FP8**: Convert per-16-element `AbsMaxP` values to `float_ue4m3_t` using `packed_float_to_ue4m3()`

2. **Data → E2M1**: Convert 8 FP32 values at a time to packed E2M1 using `packed_float_to_e2m1()`, storing into `tOrP`

3. **Scale factor shuffle**: Use `__shfl_xor_sync(..., 2)` to exchange scale factors between paired threads (quad_id & 1), then merge with bitmasking into the layout expected by the MMA atom

**Two-level scaling trick** (done in `online_softmax_with_quant`): Before quantize is called, softmax already divides P by AbsMaxP. So the values passed to `packed_float_to_e2m1` are pre-normalized to [-1, 1] range, maximizing E2M1 precision.

---

## Software Pipelining Pattern

Throughout the kernel, data copies and GEMM are interleaved:

```
copy_k_block(0)           ← first K block ready in registers
GEMM Q@K[0] (async)       ← tensor cores busy
  copy_k_block(1)         ← while GEMM runs, load next K block
GEMM Q@K[1] (async)
  release K pipeline      ← done with all K blocks for this tile
```

Same pattern for P@V:

```
copy_v_block(0)
quantize(0)               ← quantize P chunk 0 to FP4
GEMM P@V[0] (async)
  copy_v_block(1)         ← load next V block
  quantize(1)             ← quantize next P chunk
GEMM P@V[1] (async)
  release V pipeline
```

**Note**: With `kBlockN=64` and MMA-K=64, `size<2>(tOrP) = kBlockN/64 = 1`, so the V loop body executes once. The software pipelining becomes more significant when kBlockN > 64.

---

## Online Softmax with Two-Level P Scaling

Located in `softmax_fused.h`, `SoftmaxFused::online_softmax_with_quant()` performs online softmax across tiles while simultaneously preparing P for FP4 quantization.

### Constants

```cpp
fp8_scalexfp4_scale = 1 / (448 × 6)  // Combined FP8 × FP4 max range
fp4_scale = 1 / 6                     // FP4 (E2M1) max representable value
```

### Algorithm (for each row `mi`):

1. **Local max over 16-element groups** → `AbsMaxP(mi, ni)` — this serves double duty as the FP4 scale factor
2. **Warp shuffle** (`shfl_xor 1`) to get max across paired threads (8+8 = 16 elements)
3. **Row max** across all groups → `row_max(mi)`, shuffle with `shfl_xor 2` for full row
4. **Exponentiation**: `P_{mi,ni} = exp2(S_{mi,ni} × scale_log2 - max_scaled)` where `max_scaled = row_max × scale_log2 + log2(fp8_scale × fp4_scale)`
5. **AbsMaxP rescale**: `AbsMaxP(mi,sfi) = exp2(AbsMaxP × scale_log2 - max_scaled + log2(fp4_scale))`
6. **Pre-divide P by AbsMaxP**: `P_{j} /= AbsMaxP(group)` → normalizes P to [-1,1] before E2M1 quantization

### Tile Rescaling (non-first tiles)

For subsequent tiles, the online softmax also:
- Computes `scores_scale = exp2((prev_max - curr_max) × scale_log2)`
- Rescales `row_sum *= scores_scale`
- `rescale_o()` applies: `O_store = O_store × scores_scale + O_new`

### Finalization

```cpp
softmax_fused.finalize(tOrO_store);  // O /= row_sum (with warp reduction)
```

---

## TMA Descriptor Setup (`to_underlying_arguments`)

Converts host-side `Arguments` to device-side `Params` by creating TMA descriptors:

| TMA Descriptor | Source | SMEM Layout | Tile Shape |
|---------------|--------|-------------|------------|
| `tma_load_Q` | Q GMEM | `SmemLayoutQ` | `(kBlockM, kHeadDim)` |
| `tma_load_K` | K GMEM | `SmemLayoutK(:,:,0)` | `(kBlockN, kHeadDim)` |
| `tma_load_Vt` | V^T GMEM | `SmemLayoutVt(:,:,0)` | `(kHeadDim, kBlockN)` |
| `tma_load_DS` | δS GMEM | `SmemLayoutDS(:,:,0)` | `(kBlockM, kBlockN)` |
| `tma_load_SFQ` | SFQ GMEM | `SmemLayoutSFQ` | `(kBlockM, kHeadDim)` |
| `tma_load_SFK` | SFK GMEM | `SmemLayoutSFK(:,:,0)` | `(kBlockN, kHeadDim)` |
| `tma_load_SFVt` | SFVt GMEM | `SmemLayoutSFVt(:,:,0)` | `(kHeadDim, kBlockN)` |

### δS Layout Construction

```cpp
LayoutDS layout_ds = tile_to_shape(SmemLayoutAtomDS{}, make_shape(N_Q, N_K, H, B), Step<_2,_1,_3,_4>{});
```

`SmemLayoutAtomDS = Layout<Shape<kBlockM, kBlockN>, Stride<0, 1>>`:
- Stride-0 on the M dimension → broadcast (same values for all rows in a block)
- When `BlockMean=true`, the M-axis indexes block groups (`N/kBlockM` blocks); otherwise it's a single row

---

## Scale Factor Partitioning Helpers

### `thrfrg_SFA` / `thrfrg_SFB`

These functions partition scale factor tensors according to the MMA atom's thread layout for the A-operand (Q or P) and B-operand (K or V) respectively.

Steps:
1. **Permute** the tensor modes according to `TiledPerm`
2. **Tile** for the atom shape (e.g., 16×32×64)
3. **Compose** with the atom's SFA/SFB thread-value layout
4. **Tile** for the thread layout (distribute across warps)

### `partition_fragment_SFA` / `partition_fragment_SFB`

Create register fragments for scale factors, matching the thread's partition of the MMA operand.

### `get_layoutSFA_TV` / `get_layoutSFB_TV`

Compute the `(thread_idx, val) → (M,K)` or `(thread_idx, val) → (N,K)` mapping for the scale factors, used to construct `SmemCopyAtomSF` tiled copies.

---

## Causal Masking

### Boundary Computation

```cpp
auto col_limit_causal = [&](int row, int n_block) {
    return row + 1 + seqlen_k - n_block * kBlockN - seqlen_q + m_block * kBlockM;
};
```

For element `(row, col)` in the S tile, it's masked (set to `-INFINITY`) if:
- **Non-causal**: `col >= unpadded_seqlen_k - n_block * kBlockN` (padding mask only)
- **Causal**: `col >= min(seqlen_k - n_block * kBlockN, col_limit_causal(row, n_block))`

### Three-Phase Loop Structure

1. **First tile** (`n_block = n_block_max - 1`): May need both causal + boundary masking
2. **Causal tiles** (up to `ceil(kBlockM/kBlockN) + 1` tiles): Fully unrolled, causal masking applied
3. **Main loop** (`#pragma unroll 1`): No masking, all elements valid — this is where most tiles are processed

---

## Shared Memory Layout

Defined in `kernel_traits.h` via `SharedStorageQKVOwithSF`:

```
┌─────────────────────────────────────┐
│  smem_q     (aligned 1024)          │  Q data (1 stage)
│  smem_k     (aligned 1024)          │  K data (kStages stages)
│  smem_SFQ                           │  Q scale factors
│  smem_SFK                           │  K scale factors (kStages)
│  smem_SFV                           │  V scale factors (kStages)
│  smem_ds    (aligned 1024)          │  δS correction (kStages)
│  smem_v     (aligned 1024)          │  V data (kStages stages)
│  smem_o     (aligned 1024)          │  Output buffer
│  pipeline barriers                   │
│  tile_count_semaphore                │
└─────────────────────────────────────┘
```

---

## Summary of Execution Flow

```
Producer (1 warp)                    Consumer (N warps)
═══════════════════                  ═══════════════════
Load Q + SFQ ──────────────────────► Wait Q → copy to registers → release Q
Load K₀ + SFK₀ + δS₀ ────────────► Wait K₀
Load V₀ + SFV₀                        copy K₀ → regs
                                       add_delta_s(acc = δS₀)
                                       GEMM Q@K₀ → tSrS (acc now = Q@K⁰ᵀ)
                                       release K₀
                                       mask + softmax → P
Load K₁ + SFK₁ + δS₁               ► Wait V₀
Load V₁ + SFV₁                        copy V₀ → regs
                                       quantize(P → FP4)
                                       GEMM P@V₀ → tOrO
                                       release V₀
                                    ► Wait K₁ → repeat...
        ...                                ...
                                    finalize: O /= row_sum
```
