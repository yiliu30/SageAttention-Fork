# Bug Findings: Triton/PyTorch Educational Impl vs Real Kernel

> **Found by**: Claude Opus 4.6 via code review of real kernel (`sageattention3_blackwell/sageattn3/blackwell/`) vs educational implementations (`tasks/sage3_impl_torch/`)
> **Date**: 2026-03-03
> **Files fixed**: `sageattn3_torch_triton.py`, `sageattn3_torch.py`

---

## Bug 1 (CRITICAL — root cause of `per_block_mean` failure): `delta_s` not scaled by `sm_scale`

**Symptom**: Enabling `per_block_mean=True` produces incorrect output.

### Real kernel behavior (`mainloop_tma_ws.h`)

```cpp
// 1. Accumulator tSrS is zero-initialized
Tensor tSrS = partition_fragment_C(tiled_mma_qk, select<0, 1>(TileShape_MNK{}));

// 2. add_delta_s WRITES delta_s into the zero accumulator (acc = delta_s)
add_delta_s(tSrS);

// 3. GEMM ACCUMULATES QK^T onto the accumulator (acc = delta_s + QK^T)
for (int k_block = 0; k_block < size<2>(tSrQ); ++k_block) {
    cute::gemm(..., tSrS);  // tSrS += Q_block @ K_block
}

// 4. Softmax applies scale to the COMBINED result
//    exp2((delta_s + QK^T) * softmax_scale_log2 - max_scaled)
softmax_fused.online_softmax_with_quant(tSrS, AbsMaxP, softmax_scale_log2);
```

So the real kernel computes: **`softmax((QK^T + delta_s) × sm_scale)`**

### Bug in educational implementations

```python
# WRONG (before fix):
qk_tile = matmul(Q, K^T) * sm_scale   # scale applied to QK^T only
qk_tile = qk_tile + delta_s           # delta_s NOT scaled
# Result: softmax(QK^T * sm_scale + delta_s)  ← delta_s has wrong magnitude

# CORRECT (after fix):
qk_tile = matmul(Q, K^T)              # raw QK^T
qk_tile = qk_tile + delta_s           # add delta_s first
qk_tile = qk_tile * sm_scale          # scale the COMBINED result
# Result: softmax((QK^T + delta_s) * sm_scale)  ← matches real kernel
```

This caused the `delta_s` contribution to be off by a factor of `sm_scale` = `1/sqrt(D)` (≈ 0.088 for D=128), completely breaking the QK smoothing correction.

---

## Bug 2: Incorrect `beta` factor in online softmax accumulation

### Real kernel behavior (`softmax_fused.h`)

```cpp
// Non-first tile: compute exp2(acc * scale - max_scaled)
// The probabilities are already relative to new_max via max_scaled
for (int ni = 0; ni < size<1>(acc_reduction_view); ni++) {
    acc_reduction_view(mi, ni) = flash::ptx_exp2(
        acc_reduction_view(mi, ni) * softmax_scale_log2 - max_scaled
    );
    row_sum(mi) += acc_reduction_view(mi, ni);  // Direct accumulation, no beta
}
```

The real kernel computes `p = exp(score - new_max)` and directly adds to `row_sum`. No `beta = exp(tile_max - new_max)` factor.

### Bug in educational implementations

```python
# WRONG (before fix):
p_tile = exp(qk_tile - new_max)                       # already relative to new_max
running_sum = running_sum * alpha + tile_sum * beta    # beta double-counts
output_tile = output_tile + pv_tile * beta             # beta double-counts

# CORRECT (after fix):
p_tile = exp(qk_tile - new_max)                        # already relative to new_max
running_sum = running_sum * alpha + tile_sum            # direct accumulation
output_tile = output_tile + pv_tile                     # direct accumulation
```

Since `p_tile = exp(qk - new_max)` is already computed relative to `new_max`, multiplying by `beta = exp(tile_max - new_max)` would incorrectly shift the probabilities.

---

## Verification Results (after fix)

Test config: B=1, H=8, N=240, D=64 with `per_block_mean=True`

| Comparison | Cosine Similarity | Grade |
|---|---|---|
| PyTorch vs Triton | 99.53% | EXCELLENT |
| PyTorch vs Real kernel | 97.18% | VERY GOOD |
| Triton vs Real kernel | 96.52% | VERY GOOD |

The remaining ~3-4% gap between educational impls and real kernel is expected from the educational quantization emulation (nearest-neighbor FP4 lookup in Python) vs real hardware FP4 tensor core execution.

---

## Quantitative Impact of Each Bug

Measured by isolating each bug independently (no FP4 quantization, pure attention logic).
Cosine similarity to the correct (both-fixed) implementation:

| Config | Bug1 only | Bug2 only | Both bugs |
|---|---|---|---|
| small rand (scale=0.1, N=256) | 99.998% | 99.999% | 99.998% |
| **unit rand (scale=1.0, N=256)** | **83.9%** | **98.3%** | **80.3%** |
| **large rand (scale=3.0, N=256)** | **46.0%** | **99.7%** | **45.4%** |
| **long seq (scale=1.0, N=1024)** | **83.5%** | **94.1%** | **72.4%** |
| **long seq (scale=1.0, N=4096)** | **82.6%** | **89.4%** | **63.7%** |
| **D=128 (scale=1.0, N=256)** | **71.6%** | **98.5%** | **67.9%** |

### Key takeaways

1. **Bug 1 (delta_s unscaled) is the dominant bug.** With unit-scale inputs it drops similarity to ~84%, and with larger inputs or D=128 it drops to 46-72%. This makes sense: the delta_s correction is amplified by `1/sm_scale` = `sqrt(D)` (8x for D=64, 11.3x for D=128), completely distorting the attention distribution.

2. **Bug 2 (beta double-counting) has moderate impact.** It worsens with more tiles (longer sequences): 98.3% at N=256 → 89.4% at N=4096. This is because the bug under-weights contributions from non-dominant tiles, and more tiles means more under-weighting. For short sequences with 1-2 tiles, the impact is small.

3. **Both bugs together are worse than either alone**, especially at long sequences (63.7% at N=4096).

4. **With very small inputs (scale=0.1)** as used in the previous unit tests, both bugs are essentially invisible (~99.99%). This explains why the bugs weren't caught earlier — the test used `* 0.1` scaling which made delta_s negligibly small.
