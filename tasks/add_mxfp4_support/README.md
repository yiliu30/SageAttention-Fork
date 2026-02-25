Task Description

Goal:
Add MXFP4 (block size 32, E8M0 scales) support to SageAttention, starting with accuracy verification and progressing to a Triton implementation.

## Development Plan

### Step 1: NVFP4 Accuracy Baseline — Level 1 (PyTorch, no P quant)

Quant-dequant simulation ignoring P quantization. Validates against SageAttention3's paper numbers.

- Implement `nvfp4_quant_dequant(x, block=16, scale='e4m3')` in PyTorch
- Flow: Smooth Q/K → quant-dequant Q, K, V → `sdpa(Q_deq, K_deq, V_deq)` + δS correction
- Evaluate CosSim/L1/RMSE on real CogVideoX Q/K/V tensors across all layers
- Target: reproduce paper's ~99.5% worst-case CosSim

### Step 2: NVFP4 Accuracy — Level 2 (PyTorch, with tile-level P quant)

Add tile-level P quantization with two-level scaling to capture full accuracy picture.

- Implement tiled attention loop (outer loop over Q-tiles, vectorized P quant across K-tiles)
- P flow per tile: softmax → sP1 = rowmax/(448×6) → rescale → quant-dequant → matmul with V_deq × sP1
- Compare Level 2 vs Level 1 to quantify P quantization's accuracy contribution
- Tile sizes: B_q=128, B_kv=64 (match real kernel)

### Step 3: NVFP4 Attention in Triton

Reimplement the SageAttention3 attention kernel in Triton with NVFP4 quant-dequant.

- Triton kernel with FlashAttention-style tiling and online softmax
- In-kernel NVFP4 quant-dequant for Q/K (pre-tile) and P (tile-level, two-level scaling)
- Validate accuracy matches Step 2 (same data flow, different implementation)
- This becomes the base for MXFP4 extension

### Step 4: MXFP4 in Triton

Update Step 3's Triton kernel to support MXFP4.

- Change block size 16 → 32, scale type E4M3 → E8M0 (power-of-2 only)
- Adapt two-level P scaling for E8M0 constraints
- Compare MXFP4 vs NVFP4 accuracy (CosSim/L1/RMSE) across all CogVideoX layers
- Ablation: MXFP4 for Q/K only vs MXFP4 for P/V only vs full MXFP4

Notes:
- Steps 1–2 are pure PyTorch, no GPU kernel writing required
- Steps 3–4 target accuracy verification, not performance optimization
- End-to-end video generation quality (VQA) can be tested at any step using CogVideoX


Source:
- Paper:
    - /mnt/disk1/yiliu7/SageAttention-Fork/papers/sageattention/SageAttention3-2505.11594.pdf
    - /mnt/disk1/yiliu7/SageAttention-Fork/papers/sageattention/sage_v3
    - /mnt/disk1/yiliu7/SageAttention-Fork/papers/sageattention/paper_summaries.md
- Codebase:
    - All Sage (v1, v2, v3): /mnt/disk1/yiliu7/SageAttention-Fork
    - Sage v3: /mnt/disk1/yiliu7/SageAttention-Fork/sageattention3_blackwell

Notes:
1. No coding required for now.

---

## Step 1: SageAttention3 Summary & Data Flow

### High-Level Summary

SageAttention3 applies **NVFP4 microscaling quantization** (E2M1 data, block size 1×16, E4M3 FP8 scale factors) to **both** GEMMs in attention (Q@K^T and P@V) on Blackwell GPUs (SM120), achieving ~5x speedup over FlashAttention2. Three key techniques:

1. **Smoothing Q+K** (inherited from v1/v2): subtract channel-wise mean from K (lossless via softmax shift-invariance), subtract per-block mean from Q (corrected via GEMV δS = q̄@K^T).
2. **Two-level scaling for P**: softmax output P lies in [0,1], so its FP8 scale factors cluster near 0 → poor E4M3 utilization. Fix: first scale each row by sP1 = rowmax(P)/(448×6) to expand range, then apply standard NVFP4 microscaling with FP8 sP2. Restores CosSim from 93.32% to 99.52%.
3. **Hardware optimizations**: K column permutation (match FP4MM accumulator layout), reuse softmax max-reduction for quantization (~10% speedup), producer-warp epilogue (ping-pong between producer warps for overlapping compute + stores).

### Data Flow (see Mermaid diagram above)

```
Input: Q, K, V in FP16 [B, H, N, d]
                  │
  ┌───────────────┼───────────────────────┐
  │  PREPROCESSING (Python/Triton)        │
  │  1. K -= mean(K)           (smooth K) │
  │  2. Q_blk -= mean(Q_blk)  (smooth Q) │
  │  3. δS = q̄ @ K^T          (GEMV)     │
  └───────────────┼───────────────────────┘
                  │
  ┌───────────────┼───────────────────────┐
  │  QUANTIZATION (CUDA kernels)          │
  │  4. Q → NVFP4 (E2M1 + E4M3 scales)   │
  │  5. K → NVFP4 + column permute       │
  │  6. V → NVFP4 + transpose (fused)     │
  └───────────────┼───────────────────────┘
                  │
  ┌───────────────┼──────────────────────────────┐
  │  FUSED ATTENTION KERNEL (CUTLASS/SM120)       │
  │  For each Q-tile i, K/V-tile j:               │
  │    7. S = FP4MM(Q̂,sQ, K̂,sK) + δS   ← GEMM1  │
  │    8. Online softmax: P̃ = exp(S - max)        │
  │    9. Two-level quant P̃ → P̂, sP1, sP2        │
  │   10. O += FP4MM(P̂,sP2, V̂,sV) × sP1 ← GEMM2 │
  │   11. Rescale O (online softmax)               │
  │  Final: O = diag(l)⁻¹ O → FP16/BF16          │
  └────────────────────────────────────────────────┘
```

---

## Step 2: Core Code Map — GEMM & Quantization

### Pre-kernel Quantization (host-side, before kernel launch)

| Function | File | What it does |
|----------|------|--------------|
| `preprocess_qkv()` | `sageattn3/api.py` L79-102 | Smooth K (mean subtract), smooth Q (per-128-block mean), compute δS = q̄@K^T |
| `scale_and_quant_fp4()` | `sageattn3/api.py` L104-110 | Quantize Q to NVFP4. Calls `fp4quant_cuda.scaled_fp4_quant` |
| `scale_and_quant_fp4_permute()` | `sageattn3/api.py` L112-118 | Quantize K to NVFP4 **with column permutation**. Calls `fp4quant_cuda.scaled_fp4_quant_permute` |
| `scale_and_quant_fp4_transpose()` | `sageattn3/api.py` L120-126 | Quantize V to NVFP4 **with transpose fused**. Calls `fp4quant_cuda.scaled_fp4_quant_trans` |
| `scaled_fp4_quant_kernel<>` | `sageattn3/quantization/fp4_quantization_4d.cu` L133-246 | CUDA kernel: per-16-element block, compute max→E4M3 scale, scale input, convert to E2M1 via PTX `cvt.rn.satfinite.e2m1x2.f32` |
| `scaled_fp4_quant_trans_kernel<>` | `sageattn3/quantization/fp4_quantization_4d.cu` L264-392 | Same as above but with shared-memory transpose for V |

### In-Kernel GEMM (inside the fused attention kernel)

| Code location | File | What it does |
|--------------|------|--------------|
| **GEMM1 (Q@K^T)** | `sageattn3/blackwell/mainloop_tma_ws.h` L737-746 | `cute::gemm(tiled_mma_qk, make_zip_tensor(tSrQ, tSrSFQ), make_zip_tensor(tSrK, tSrSFK), tSrS)` — FP4MM with zip'd data+scale tensors |
| **add δS** | `sageattn3/blackwell/mainloop_tma_ws.h` L695-710 | Loads δS from smem, adds to accumulator before GEMM1 starts |
| **GEMM2 (P@V)** | `sageattn3/blackwell/mainloop_tma_ws.h` L775-782 | `cute::gemm(tiled_mma_pv, make_zip_tensor(tOrP, tOrSFP), make_zip_tensor(tOrVt, tOrSFVt), tOrO)` |
| **In-kernel P quantization** | `sageattn3/blackwell/mainloop_tma_ws.h` L754-803 | `quantize()` lambda: converts softmax output (FP32 accumulators) → E2M1 data + E4M3 scale via `packed_float_to_e2m1()` and `packed_float_to_ue4m3()` |
| **Online softmax + two-level scaling** | `sageattn3/blackwell/softmax_fused.h` L42-155 | `online_softmax_with_quant()`: computes row max, exp, row sum, while simultaneously computing AbsMaxP for P's quantization scales. Divides acc by AbsMaxP to prescale before E2M1 conversion |

### MMA Instructions

| MMA atom | File | Instruction |
|----------|------|------------|
| `TiledMmaQK` | `sageattn3/blackwell/kernel_traits.h` L107-111 | `SM120::BLOCKSCALED::SM120_16x32x64_TN_VS_NVFP4` — NVFP4 block-scaled MMA |
| `TiledMmaPV` | `sageattn3/blackwell/kernel_traits.h` L113-117 | Same MMA atom `SM120_16x32x64_TN_VS_NVFP4` |

Both GEMMs use the **same** NVFP4 MMA instruction (`SM120_16x32x64_TN_VS_NVFP4`), which takes `zip_tensor(data_E2M1, scale_E4M3)` operands.

---

## Step 3: Scaling Factor Layout & Block Size

### NVFP4 Quantization Parameters

| Parameter | Value | Where defined |
|-----------|-------|---------------|
| **Data type** | E2M1 (FP4) | `kernel_traits.h` L96: `using Element = cutlass::float_e2m1_t` |
| **Scale factor type** | E4M3 (FP8) | `kernel_traits.h` L95: `using ElementSF = cutlass::float_ue4m3_t` |
| **Block size** | 1×16 (SFVectorSize) | `kernel_traits.h` L98: `static constexpr auto SFVectorSize = 16` |
| **Num SF per head_dim** | head_dim / 16 | `kernel_traits.h` L93: `static constexpr int NumSFQK = kHeadDim / 16` |
| **Num SF per block_N** | block_N / 16 | `kernel_traits.h` L94: `static constexpr int NumSFPV = kBlockN / 16` |

### Scale Factor Memory Layout (blockscaled_layout.h)

The `BlockScaledConfig<16>` struct defines the NVFP4 scale factor layout for SM120's block-scaled MMA:

- **SfAtom**: a 2D layout mapping `(MN, K)` where:
  - MN dimension: 64 rows, grouped as `(16, 4)` with stride `(16, 4)` — 4 scale factors per 64-row group forming a 32-bit word
  - K dimension: `(16, 4)` with stride `(0, 1)` — each scale is shared across SFVectorSize=16 elements, 4 scales per MMA-K step
  
- **Global memory layout**: `tile_atom_to_shape_SFQKV(shape)` tiles the SfAtom across `(Seqlen, Dim, HeadNum, Batch)`
  - Q scales: shape `[B, H, N, d/16]` stored as `float8_e4m3fn`
  - K scales: shape `[B, H, N, d/16]`
  - V scales (transposed): shape `[B, H, d, N/16]`

- **Shared memory layout**: `deduce_smem_layoutSFQ/SFK/SFVt()` functions produce layouts that match the MMA instruction's expected scale factor access pattern. The layout uses the SfAtom as a building block and tiles it to cover the full `(M/N, K)` tile.

### In-Kernel P Quantization Scale Layout

P is quantized **in-register** during the kernel, using two custom layouts:
- `LayoutSFP`: `((16,4), 1, kBlockN/64)` with stride `((0,1), 0, 4)` — 4 E4M3 scales packed per 64-column chunk
- `LayoutP`: `((8,2,2), 1, kBlockN/64)` with stride `((1,8,16), 0, 32)` — E2M1 data layout matching the FP4MM accumulator's permuted output

### Pre-kernel Quantization Scale Storage

In `fp4_quantization_4d.cu`, the scale factor is stored in a specific interleaved pattern within 64-row groups:
```
offset = (col_id/4)*256 + (col_id%4) + (row_id/16)*4 + (row_id%16)*16
```
This matches the SfAtom layout expected by TMA loads.

---

## Step 4: MXFP4 Feasibility Analysis

### NVFP4 vs MXFP4 Comparison

| Property | NVFP4 (current) | MXFP4 (target) |
|----------|-----------------|-----------------|
| Data type | E2M1 | E2M1 (same) |
| Block size | 1×16 | 1×32 |
| Scale type | E4M3 (FP8, 8-bit with mantissa) | E8M0 (8-bit, powers-of-2 only) |
| Scale precision | Fine-grained (mantissa bits) | Coarse (integer exponents only) |
| CosSim (worst) | 99.52% | 98.37% |
| HW instruction | `SM120_16x32x64_TN_VS_NVFP4` | Needs different MMA atom |
| CUTLASS support | Yes, used in SageAttn3 | Yes, exists in CUTLASS (e.g., `ue8m0xf4` configs) |

### What Needs to Change

**A) Pre-kernel quantization (`fp4_quantization_4d.cu`)**:
1. **Block size 16 → 32**: The `CVT_FP4_ELTS_PER_THREAD=16` naturally processes 16 elements per thread. For MXFP4, the max-reduction must span 32 consecutive elements instead of 16. This means either:
   - Processing 32 elements per thread (double the register pressure), or
   - Cross-thread shuffle to find max across 2 threads' worth of 16 elements
2. **Scale type E4M3 → E8M0**: Replace `__nv_fp8_e4m3(SFValue)` with E8M0 conversion (round to nearest power of 2). CUTLASS has `cutlass::float_ue8m0_t` type.
3. **Scale factor storage layout**: The 64-row interleaved layout must change to match MXFP4's MMA instruction expectations. Fewer scale factors (half as many: d/32 instead of d/16).

**B) Kernel traits (`kernel_traits.h`)**:
1. **MMA atom**: Replace `SM120_16x32x64_TN_VS_NVFP4` with an MXFP4 variant. CUTLASS already has MXFP4 atoms — search for `ue8m0` entries in the CUTLASS MMA traits. The MMA-K dimension may change (e.g., from 64 to 128 if block=32 doubles the K-step).
2. **`SFVectorSize`**: Change from 16 to 32.
3. **`ElementSF`**: Change from `cutlass::float_ue4m3_t` to `cutlass::float_ue8m0_t`.
4. **`NumSFQK`**: Changes from `kHeadDim/16` to `kHeadDim/32` — half as many scale factors.
5. **Shared memory scale layouts**: `BlockScaledConfig<32>` would produce different SfAtom shapes.

**C) Block-scaled layout (`blockscaled_layout.h`)**:
1. **SfAtom**: The `kBasicBlockShape` would change from `(16, 4)` to `(32, X)` — need to check what MXFP4's hardware SfAtom looks like.
2. **All smem layout deduction functions** (`deduce_smem_layoutSFQ/SFK/SFVt`) would produce different shapes due to changed SFVectorSize.

**D) In-kernel P quantization (`mainloop_tma_ws.h` quantize lambda + `softmax_fused.h`)**:
1. **Two-level scaling**: The P two-level scaling technique was specifically designed for E4M3 scale factors. With MXFP4's E8M0 scales (powers of 2 only), the range problem is **even worse** — E8M0 can't represent the fine-grained scale values at all. The two-level trick may need redesign, or a different compensation approach.
2. **AbsMaxP computation**: Currently reduces over 16-element chunks (4 threads × 4 elements, then shuffle). For block=32, would need reduction over 32-element chunks.
3. **`LayoutSFP` and `LayoutP`**: Must be redesigned for the MXFP4 MMA's accumulator layout.

**E) TMA descriptors and data loading (`mainloop_tma_ws.h`)**:
1. Scale factor tensor shapes change (half as many scale factors).
2. TMA copy descriptors and transaction byte counts change accordingly.

### Key Risks / Open Questions

1. **Accuracy of two-level scaling with E8M0**: The current two-level scheme maps P's scales into E4M3's fine-grained range. With E8M0 (only powers of 2), the scale quantization error for P could be much larger. This is the **biggest accuracy risk** — the 98.37% CosSim number in the paper was measured **without** two-level scaling for MXFP4; with E8M0 scales, two-level may not help as much.

2. **Does Blackwell SM120 have a native MXFP4 MMA?**: The `SM120_16x32x64_TN_VS_NVFP4` is specifically for NVFP4. CUTLASS has `ue8m0xf4` types suggesting MXFP4 support, but it's unclear if this maps to a distinct hardware instruction or is SW-emulated. Need to verify in PTX/SASS docs.

3. **Performance**: MXFP4 with block=32 means half the scale factors → potentially less overhead. But if the MMA K-step is different, the tile loop structure may change.

### Conclusion

Extending to MXFP4 is **technically feasible** — the data format (E2M1) is identical, CUTLASS has E8M0 type support, and the changes are primarily:
- Changing `SFVectorSize` from 16→32, `ElementSF` from E4M3→E8M0
- Updating scale factor layouts and quantization kernels
- Finding/using the MXFP4 MMA atom in CUTLASS

However, the **accuracy concern is significant**. The E8M0 scale factor (powers-of-2 only) combined with larger block size (32) will make the P two-level scaling less effective. The ablation showing 98.37% CosSim is GEMM-level only; end-to-end video quality tests should be done first to assess whether the accuracy gap is acceptable before investing in full kernel implementation.

1. NVFP4 vs MXFP4 ablation — GEMM-level, not end-to-end
---