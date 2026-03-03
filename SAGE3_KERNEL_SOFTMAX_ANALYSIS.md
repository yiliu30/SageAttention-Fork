# SageAttention3 Real Kernel: Online Softmax & Quantization Analysis

## Overview

This document provides a line-by-line analysis of the online softmax and quantization implementation in the real SageAttention3 Blackwell kernel (`softmax_fused.h` and `mainloop_tma_ws.h`).

## Key Constants and Setup

### Critical Scale Factors (Lines 32-35, softmax_fused.h)

```cpp
static constexpr float fp8_scalexfp4_scale = 1.f / (448 * 6);           // Line 32
static constexpr float fp8_scalexfp4_scale_log2 = -11.392317422778762f; // Line 33
static constexpr float fp4_scale_log2 = -2.584962500721156f;            // Line 34
static constexpr int RowReductionThr = 4;                               // Line 35
```

**Analysis:**
- `fp8_scalexfp4_scale = 1/(448*6)`: Combined scaling for two-level quantization
  - `448`: Maximum FP8 E4M3 representable value (2^7 * (1 + 7/8))
  - `6`: Maximum FP4 E2M1 representable value
  - `Combined scale = 2688`: Total dynamic range for P quantization
- `fp8_scalexfp4_scale_log2`: Pre-computed log2 for efficiency (`log2(1/2688) ≈ -11.39`)
- `fp4_scale_log2`: Pre-computed log2(1/6) for FP4 microscaling (`≈ -2.58`)
- `RowReductionThr = 4`: Warp-level reduction threshold (32 threads / 8 = 4)

## Online Softmax Entry Point (Lines 40-44)

```cpp
template<bool FirstTile, bool InfCheck = false, typename TensorAcc, typename TensorMax>
CUTLASS_DEVICE auto online_softmax_with_quant(
    TensorAcc& acc,           // QK^T attention scores (input/output)
    TensorMax& AbsMaxP,       // Per-block absolute maximum for FP8 scaling
    const float softmax_scale_log2  // Pre-computed log2(softmax_scale)
) {
```

**Analysis:**
- `FirstTile`: Template parameter - different logic for first vs subsequent tiles
- `InfCheck`: Template parameter - enables infinity checking for numerical safety
- `acc`: MMA accumulator containing QK^T scores (gets overwritten with probabilities)
- `AbsMaxP`: Tensor storing absolute maximum values for FP8 E4M3 global scaling
- `softmax_scale_log2`: Pre-computed `log2(1/√D)` for efficiency

## Tensor Layout Setup (Lines 45-48)

```cpp
Tensor acc_reduction_view = make_tensor(acc.data(), flash::convert_to_reduction_layout(acc.layout()));
Tensor acc_conversion_view = make_tensor(acc.data(), flash::convert_to_conversion_layout(acc.layout()));
Tensor acc_conversion_flatten = group_modes<1, 5>(group_modes<0, 2>(flatten(acc_conversion_view)));
```

**Analysis:**
- `acc_reduction_view`: Reshapes accumulator for warp-level reductions (row-wise operations)
- `acc_conversion_view`: Reshapes accumulator for type conversions and quantization
- `acc_conversion_flatten`: Flattened view for efficient quantization loops
- Multiple tensor views of same data enable different access patterns without copies

## First Tile Processing (Lines 49-89)

### Initialization (Lines 50-53)

```cpp
if constexpr (FirstTile) {
    fill(row_max, -INFINITY);    // Initialize row maximums to -∞
    clear(row_sum);              // Initialize row sums to 0
    fill(scores_scale, 1.f);     // Initialize scaling factors to 1
```

**Analysis:**
- First tile initializes running statistics from scratch
- `row_max`: Tracks maximum attention score per row (for numerical stability)
- `row_sum`: Tracks sum of exponentials per row (for normalization)
- `scores_scale`: Tracks rescaling factors for online updates

### FP8 Scale Factor Computation (Lines 55-65)

```cpp
CUTLASS_PRAGMA_UNROLL
for (int mi = 0; mi < size<0>(acc_reduction_view); mi++) {           // For each row
    CUTLASS_PRAGMA_UNROLL
    for (int ni = 0; ni < size<1, 1>(acc_reduction_view); ni++) {    // For each block within row
        CUTLASS_PRAGMA_UNROLL
        for (int ei = 0; ei < size<1, 0>(acc_reduction_view); ei++) { // For each element within block
            AbsMaxP(mi, ni) = fmaxf(AbsMaxP(mi, ni), acc_reduction_view(mi, make_coord(ei, ni)));
        }
        float max_recv = __shfl_xor_sync(int32_t(-1), AbsMaxP(mi, ni), 1);
        AbsMaxP(mi, ni) = fmaxf(AbsMaxP(mi, ni), max_recv);          // Warp shuffle for max reduction
        row_max(mi) = fmaxf(row_max(mi), AbsMaxP(mi, ni));           // Global row maximum
    }
```

**Analysis:**
- **Triple nested loop**: Processes MMA accumulator in hierarchical manner
  - `mi`: Row index within the tile
  - `ni`: Block index within row (for 16-element FP4 microscaling blocks)
  - `ei`: Element index within block
- **Line 60**: Find local maximum within each 16-element block for FP4 microscaling
- **Line 62**: `__shfl_xor_sync(..., 1)`: Warp shuffle exchanges values between adjacent threads
- **Line 63**: Combine local max with neighbor's max (warp-level reduction)
- **Line 64**: Update global row maximum (needed for softmax numerical stability)

### Warp-Level Max Reduction (Lines 67-68)

```cpp
float max_recv = __shfl_xor_sync(int32_t(-1), row_max(mi), 2); // exchange max in a quad in a row
row_max(mi) = fmaxf(row_max(mi), max_recv);
```

**Analysis:**
- **Shuffle with offset 2**: Exchanges values across 4-thread groups (quads)
- **Two-stage reduction**: First shuffle offset 1 (pairs), then offset 2 (quads)
- **Result**: Each thread in a warp gets the maximum across all 4 threads processing same row

### Combined Scale Computation (Lines 70-72)

```cpp
const float max_scaled = InfCheck
                        ? (row_max(mi) == -INFINITY ? 0.f : (row_max(mi) * softmax_scale_log2 + fp8_scalexfp4_scale_log2))
                        : (row_max(mi) * softmax_scale_log2 + fp8_scalexfp4_scale_log2);
```

**Analysis:**
- **Combined scaling**: `row_max * softmax_scale + log2(1/(448*6))`
- **Purpose**: Pre-compute the combined scale factor for both softmax and quantization
- **InfCheck branch**: Handles edge case where all attention scores are -∞
- **Mathematical meaning**: `max_scaled = log2(exp(row_max * sm_scale) / 2688)`

### Probability and FP8 Scale Computation (Lines 74-80)

```cpp
CUTLASS_PRAGMA_UNROLL
for (int ni = 0; ni < size<1>(acc_reduction_view); ni++) {
    acc_reduction_view(mi, ni) = flash::ptx_exp2(acc_reduction_view(mi, ni) * softmax_scale_log2 - max_scaled);
}
CUTLASS_PRAGMA_UNROLL
for (int sfi = 0; sfi < size<1>(AbsMaxP); sfi++) {
    AbsMaxP(mi, sfi) = flash::ptx_exp2(AbsMaxP(mi, sfi) * softmax_scale_log2 - max_scaled + fp4_scale_log2);
}
```

**Analysis:**
- **Line 75**: Compute probabilities using hardware exp2 instruction
  - `ptx_exp2(score * sm_scale - max_scaled)` = `exp2(score * sm_scale - max - log2(2688))`
  - Result: probabilities scaled by combined quantization factor
- **Line 79**: Compute FP8 global scale factors for each 16-element block
  - `ptx_exp2(block_max * sm_scale - max_scaled + fp4_scale_log2)`
  - `fp4_scale_log2` adjustment accounts for FP4 microscaling level

### Row Sum Accumulation (Lines 82-88)

```cpp
CUTLASS_PRAGMA_UNROLL
for (int mi = 0; mi < size<0>(acc_reduction_view); mi++) {
    CUTLASS_PRAGMA_UNROLL
    for (int ni = 0; ni < size<1>(acc_reduction_view); ni++) {
        row_sum(mi) += acc_reduction_view(mi, ni);
    }
}
```

**Analysis:**
- **Simple accumulation**: Sum all probabilities in each row
- **No beta factor**: Direct accumulation since probabilities already computed relative to current max
- **This resolves Bug 2**: No redundant beta scaling that caused issues in educational implementations

## Subsequent Tile Processing (Lines 90-130)

### Running Statistics Update (Lines 91-114)

```cpp
else {  // Not FirstTile
    Tensor scores_max_prev = make_fragment_like(row_max);
    cute::copy(row_max, scores_max_prev);                    // Save previous maximum
    // ... find new maximum in current tile ...
    scores_scale(mi) = flash::ptx_exp2((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
```

**Analysis:**
- **Line 92**: Save previous row maximum before updating
- **Line 113**: Compute rescaling factor for previous accumulations
  - `exp2((old_max - new_max) * softmax_scale_log2)`
  - This is the "alpha" factor in online softmax literature

### Running Sum Update (Lines 118-123)

```cpp
row_sum(mi) = row_sum(mi) * scores_scale(mi);              // Rescale previous sum
CUTLASS_PRAGMA_UNROLL
for (int ni = 0; ni < size<1>(acc_reduction_view); ni++) {
    acc_reduction_view(mi, ni) = flash::ptx_exp2(acc_reduction_view(mi, ni) * softmax_scale_log2 - max_scaled);
    row_sum(mi) += acc_reduction_view(mi, ni);             // Direct accumulation - no beta!
}
```

**Analysis:**
- **Line 118**: Rescale previous running sum by alpha factor
- **Line 121**: Compute current tile probabilities relative to new global maximum
- **Line 122**: **Direct accumulation** - no beta factor multiplication
- **Critical insight**: This confirms the fix for Bug 2 - no beta double-scaling

## Delta_s Injection (Lines 691-702, mainloop_tma_ws.h)

```cpp
auto add_delta_s = [&](auto& acc) {
    auto tSsDS_stage = recast<float4>(sDS(_, _, smem_pipe_read_k.index()));
    auto acc_float4 = recast<float4>(acc);
    int quad_id = (threadIdx.x % 4) * 2;                    // 0, 2, 4, 6
    for (int i = 0; i < 4; i++) {
        auto num = quad_id + i * 8;                         // Access pattern: 0,8,16,24 / 2,10,18,26 / ...
        float4 delta_s_0 = tSsDS_stage(make_coord(_0{}, _0{}), make_coord(num, _0{}));
        float4 delta_s_1 = tSsDS_stage(make_coord(_0{}, _0{}), make_coord(num + 1, _0{}));
        acc_float4(make_coord(make_coord(_0{}, _0{}), _0{}), _0{}, i) = delta_s_0;  // Write to multiple positions
        acc_float4(make_coord(make_coord(_0{}, _0{}), _1{}), _0{}, i) = delta_s_0;  // Broadcasting across M
        acc_float4(make_coord(make_coord(_0{}, _1{}), _0{}), _0{}, i) = delta_s_1;
        acc_float4(make_coord(make_coord(_0{}, _1{}), _1{}), _0{}, i) = delta_s_1;  // Broadcasting across M
    }
};
```

**Analysis:**
- **Line 692**: Recast shared memory as float4 for 128-bit vectorized loads
- **Line 693**: Recast accumulator as float4 for 128-bit vectorized stores
- **Line 694**: `quad_id` determines which elements this thread processes
- **Line 696**: Strided access pattern for coalesced memory access
- **Lines 699-702**: **Broadcasting pattern** - same delta_s values written to multiple accumulator positions
  - This implements the M-dimension broadcasting (all queries in a block share same delta_s)
  - Stride-0 memory layout enables efficient broadcasting

## Kernel Integration (Lines 718, 799, 826, etc.)

```cpp
add_delta_s(tSrS);                                          // Initialize accumulator with delta_s
// ... QK GEMM loop ...
softmax_fused.template online_softmax_with_quant</*Is_first=*/true>(tSrS, AbsMaxP, mainloop_params.softmax_scale_log2);
```

**Analysis:**
- **Line 718**: `add_delta_s` called **before** GEMM loop - initializes accumulator to delta_s instead of zero
- **Mathematical effect**: `final_scores = delta_s + Q_smooth @ K_smooth^T = Q @ K^T` (exact equivalence)
- **Line 799**: First tile uses `Is_first=true` template parameter
- **Subsequent calls**: Use `Is_first=false` for online updates

## Two-Level Quantization (Lines 131-137)

```cpp
CUTLASS_PRAGMA_UNROLL
for (int i = 0; i < size(AbsMaxP); ++i) {
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < size<0>(acc_conversion_flatten); ++j)
        acc_conversion_flatten(j, i) /= AbsMaxP(i);         // Divide by FP8 global scale
}
```

**Analysis:**
- **Final quantization step**: Divide probabilities by FP8 global scale factors
- **Two-level scheme**:
  1. **Level 1 (FP8)**: Global scale per row/block (stored in `AbsMaxP`)
  2. **Level 2 (FP4)**: Microscaling per 16-element block (applied during hardware quantization)
- **Result**: Probabilities ready for FP4 microscaling quantization in subsequent operations

## Key Insights

### 1. **No Beta Factor in Real Kernel**
- Lines 122, 86: Direct accumulation `row_sum(mi) += acc_reduction_view(mi, ni)`
- No beta multiplication confirms Bug 2 fix in educational implementations

### 2. **Combined Scale Optimization**
- Line 70-72: Pre-compute combined softmax + quantization scaling
- Reduces arithmetic operations in tight loops

### 3. **Warp-Level Parallelism**
- Lines 62, 67: Explicit warp shuffle operations for efficient max reductions
- Leverages GPU's warp-synchronous execution model

### 4. **Vectorized Memory Access**
- Lines 692-693: float4 recasting for 128-bit memory operations
- Critical for memory bandwidth utilization

### 5. **Template Specialization**
- `FirstTile` template parameter eliminates branching overhead
- Compiler generates optimized code paths for each case

This analysis reveals how SageAttention3 achieves both mathematical correctness and extreme hardware efficiency through careful algorithm design and low-level GPU optimization techniques.