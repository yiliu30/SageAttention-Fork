# Phase 3: SageAttention3 C++ Kernel — Architecture Overview

## Reading Order

Read the files in this order for maximum understanding:

```
1. api.cu          → Entry point: what Python calls
2. params.h        → Data structure: all kernel arguments
3. static_switch.h → Compile-time branching macro
4. launch.h        → Template dispatch + kernel launch
5. kernel_traits.h → Type definitions: tile sizes, MMA atoms, layouts
6. kernel_ws.h     → Warp specialization: the top-level kernel function
7. tile_scheduler.h → How tiles are assigned to SMs
8. named_barrier.h → Producer-consumer synchronization
```

---

## 3.1 `api.cu` — The Entry Point (364 lines)

### What It Does

This is where PyTorch meets CUDA. The `mha_fwd()` function:
1. Validates inputs (shapes, dtypes, device)
2. Creates the output tensor
3. Fills the `Flash_fwd_params` struct
4. Calls `run_mha_fwd()` to launch the kernel

### Key Details

**Input format** (line 204-210):
```cpp
at::Tensor &q,         // [B, H, S_q, D/2] — uint8 (packed FP4 pairs)
const at::Tensor &k,   // [B, H_k, S_k, D/2]
const at::Tensor &v,   // [B, H_k, D, S_k/2] — NOTE: transposed!
const at::Tensor &sfq, // [B, H, S_q, D/16] — E4M3 scale factors
const at::Tensor &sfk, // [B, H_k, S_k, D/16]
const at::Tensor &sfv, // [B, H_k, D/16, S_k]
const at::Tensor &delta_s, // Smooth-Q correction
```

**Critical observation**: Q, K are stored as `uint8` with shape `[B, H, S, D/2]`. Each byte packs **two FP4 values**. Hence `head_size_og * 2 = unpacked_head_size`. All strides are multiplied by 2 (line 80-85) to account for the 4-bit packing.

**V is transposed**: `v` has shape `[B, H_k, D, S_k/2]` — this is V^T, not V. This matches the GEMM-II layout: `P[M×N] × V^T[N×D] = O[M×D]` becomes `P[M×N] × V^T[D×N]^T` with TN layout.

**Softmax scale** (line 142-145):
```cpp
params.scale_softmax = softmax_scale;
params.scale_softmax_log2 = softmax_scale * M_LOG2E;
// Also precomputed as half2 for potential FP16 paths
```

**Dispatch** (line 188-201):
```cpp
// BF16 vs FP16 output
template<bool IsBF16>
void run_mha_fwd_dispatch_dtype(Flash_fwd_params &params, cudaStream_t stream) {
    using OType = std::conditional_t<IsBF16, cutlass::bfloat16_t, cutlass::half_t>;
    if (params.d == 64)  run_mha_fwd_<nv_float4_t<float_e2m1_t>, 64, OType>(...);
    if (params.d == 128) run_mha_fwd_<nv_float4_t<float_e2m1_t>, 128, OType>(...);
}
```

Only `d=64` and `d=128` are supported. The element type is always `nv_float4_t<float_e2m1_t>` — NVFP4 with E2M1 encoding.

**Grid dimension** (line 334): Uses `132` as SM count for semaphore (should match actual GPU SM count).

---

## 3.2 `params.h` — The Params Struct (178 lines)

### `Qkv_params` (base class, line 34-74)

Contains pointers and strides for Q, K, V and their scale factors. Uses `int64_t` as index type for large sequences.

Key fields:
```cpp
void *q_ptr, *k_ptr, *v_ptr;        // FP4 data
void *sfq_ptr, *sfk_ptr, *sfv_ptr;  // E4M3 scale factors
void *delta_s_ptr;                    // Smooth-Q correction (float)
index_t q_row_stride, k_row_stride, v_row_stride;  // Strides (in elements)
int h, h_k, h_h_k_ratio;            // Head counts
```

### `Flash_fwd_params` (derived, line 78-176)

Adds:
```cpp
void *o_ptr;                          // Output (BF16/FP16)
void *softmax_lse_ptr;                // Log-sum-exp for each row
int b, seqlen_q, seqlen_k, d;        // Dimensions
float scale_softmax, scale_softmax_log2; // Scaling factors
cutlass::FastDivmod head_divmod, m_block_divmod; // For tile scheduler
int total_blocks;                     // Total tiles to process
int *tile_count_semaphore;            // For dynamic scheduling
bool is_causal, is_bf16, per_block_mean;
int seqlen_s;                         // DS dimension: seqlen_q if per_block, else 128
```

### Important: `per_block_mean` vs `seqlen_s`

When `per_block_mean = true`, the delta_s correction has shape `[B, H, S_q, S_k]` (one correction per query token).
When `per_block_mean = false`, it has shape `[B, H, 128, S_k]` (one correction per tile of 128 tokens, broadcast across the tile).

This controls line 476-482 in `mainloop_tma_ws.h`:
```cpp
if constexpr (BlockMean) {
    return local_tile(mDS, ..., make_coord(m_block, _));  // select m_block's row
} else {
    return local_tile(mDS, ..., make_coord(_0{}, _));     // always use row 0
}
```

---

## 3.3 `launch.h` — Template Dispatch (118 lines)

### `run_flash_fwd<Kernel_traits, Is_causal>()` (line 33-101)

This function:
1. Creates `CollectiveMainloop::Params` via `to_underlying_arguments()` — fills in TMA descriptors
2. Creates `CollectiveEpilogue::Params` similarly
3. Sets up the tile scheduler
4. Computes grid dimensions
5. Launches the kernel via `cutlass::launch_kernel_on_cluster()`

**Key parameters passed to the kernel:**
```cpp
cutlass::launch_kernel_on_cluster(launch_params, kernel,
    params,           // Flash_fwd_params (raw tensor pointers + dims)
    mainloop_params,  // TMA descriptors + layouts (created from params)
    epilogue_params,  // TMA store descriptor for output
    scheduler_params  // Tile assignment logic
);
```

### `run_mha_fwd_<T, Headdim, O>()` (line 104-118)

The innermost dispatch that instantiates the kernel traits:
```cpp
run_flash_fwd<
    Flash_fwd_kernel_traits<Headdim, 128, 128, 3, 1, per_block, T, O>,
    Is_causal
>(params, stream);
```

Concrete values:
- **Headdim**: 64 or 128
- **kBlockM**: 128
- **kBlockN**: 128
- **kStages**: 3 (triple-buffered pipeline for K, V)
- **kClusterM**: 1 (no clusters)
- **per_block**: compile-time bool for delta_s mode
- **T**: `nv_float4_t<float_e2m1_t>` (FP4)
- **O**: `bfloat16_t` or `half_t`

---

## 3.4 `kernel_traits.h` — Type Definitions (202 lines)

### The Central Configuration Hub

`Flash_fwd_kernel_traits` is a compile-time configuration struct. Everything else reads from it.

### Key Constants

```cpp
kBlockM = 128;      // Rows of Q processed per tile
kBlockN = 128;      // Columns of K processed per tile
kHeadDim = 128;     // Head dimension (or 64)
kStages = 3;        // Pipeline stages for K, V
kNWarps = 12;       // 12 warps per CTA (for kBlockM=128; 8 for kBlockM=64)
kNThreads = 384;    // 12 × 32
NumSFQK = kHeadDim / 16;  // Number of scale factors along head dim
NumSFPV = kBlockN / 16;   // Number of scale factors along N dim
SFVectorSize = 16;  // FP4 microscaling block size
```

### Warp Distribution

With `kNWarps = 12`:
- **WG0** (Warp 0-3): Producer — 4 warps, 128 threads, 1 warp group
- **WG1** (Warp 4-7): Consumer0 — 4 warps, 128 threads, 1 warp group
- **WG2** (Warp 8-11): Consumer1 — 4 warps, 128 threads, 1 warp group

Total: 3 warp groups × 128 threads = 384 threads.

### MMA Definitions (lines 112-122)

```cpp
// GEMM-I: Q × K^T → S (attention scores)
using TiledMmaQK = make_tiled_mma(
    SM120_16x32x64_TN_VS_NVFP4{},     // Atom: 16×32×64, TN layout, NVFP4
    Layout<Shape<_8, _1, _1>>{},        // 8 atoms tiled along M
    Tile<_128, _32, Int<kHeadDim>>{}    // Full tile: 128 × 32 × HeadDim
);

// GEMM-II: P × V^T → O (attention output)
using TiledMmaPV = make_tiled_mma(
    SM120_16x32x64_TN_VS_NVFP4{},     // Same atom
    Layout<Shape<_8, _1, _1>>{},        // Same 8× tiling along M
    Tile<_128, _32, Int<kHeadDim>>{}    // Same tile shape
);
```

Both GEMMs use the same MMA atom and tiling. The difference is in what operands they receive:
- GEMM-I: A=Q, B=K, C=S
- GEMM-II: A=P(quantized), B=V^T, C=O

### Shared Memory Layout

```cpp
// Data tensors
using SmemLayoutQ = tile_to_shape(SmemLayoutAtomQ{}, Shape<128, HeadDim>{});
using SmemLayoutK = tile_to_shape(SmemLayoutAtomK{}, Shape<128, HeadDim, 3>{});  // 3 stages
using SmemLayoutV = tile_to_shape(SmemLayoutAtomV{}, Shape<128, HeadDim, 3>{});  // 3 stages

// Scale factor tensors
using SmemLayoutSFQ = ...;   // 1 stage (loaded once)
using SmemLayoutSFK = ...;   // 3 stages
using SmemLayoutSFV = ...;   // 3 stages

// Output
using SmemLayoutO = tile_to_shape(..., Shape<128, HeadDim>{});  // For epilogue
```

The `SmemLayoutAtom` types are selected by CUTLASS's `sm120_rr_smem_selector` for optimal bank-conflict-free access patterns with swizzling.

### SharedStorage Layout (lines 48-66, in kernel_traits.h)

```cpp
struct SharedStorageQKVOwithSF {
    alignas(1024) ArrayEngine<Element, cosize(SmemLayoutQ)>   smem_q;     // Q: once
    alignas(1024) ArrayEngine<Element, cosize(SmemLayoutK)>   smem_k;     // K: 3 stages
    ArrayEngine<ElementSF, cosize(SmemLayoutSFQ)>             smem_SFQ;   // SFQ: once
    ArrayEngine<ElementSF, cosize(SmemLayoutSFK)>             smem_SFK;   // SFK: 3 stages
    ArrayEngine<ElementSF, cosize(SmemLayoutSFV)>             smem_SFV;   // SFV: 3 stages
    alignas(1024) ArrayEngine<float, cosize(SmemLayoutDS)>    smem_ds;    // DS: 3 stages
    alignas(1024) ArrayEngine<Element, cosize(SmemLayoutV)>   smem_v;     // V: 3 stages
    alignas(1024) ArrayEngine<OutputType, cosize(SmemLayoutO)> smem_o;    // O: epilogue

    // Pipeline barriers
    PipelineTmaAsync<1>::SharedStorage   pipeline_q;    // 1 stage
    PipelineTmaAsync<3>::SharedStorage   pipeline_k;    // 3 stages
    PipelineTmaAsync<3>::SharedStorage   pipeline_v;    // 3 stages
    OrderedSequenceBarrier::SharedStorage barrier_o;     // Epilogue sync
    int tile_count_semaphore;                            // Dynamic scheduling
};
```

### Custom Layouts for P and SFP (lines 160-171)

These are register-only layouts — P never goes through shared memory:

```cpp
// LayoutSFP: how FP4 scale factors are arranged in registers after quantize()
using LayoutSFP = Layout<
    Shape<Shape<_16, _4>, _1, Int<kBlockN/64>>,
    Stride<Stride<_0, _1>, _0, _4>
>;

// LayoutP: how FP4 P values are arranged in registers after quantize()
using LayoutP = Layout<
    Shape<Shape<_8, _2, _2>, _1, Int<kBlockN/64>>,
    Stride<Stride<_1, _8, _16>, _0, _32>
>;
```

These are carefully designed to match the MMA instruction's expected input layout.

---

## 3.5 `kernel_ws.h` — Warp Specialization (202 lines)

### The Top-Level Kernel Function

```cpp
template <typename Ktraits, bool Is_causal, typename TileScheduler>
__global__ void compute_attn_ws(
    Flash_fwd_params const params,
    MainloopParams const mainloop_params,
    EpilogueParams const epilogue_params,
    SchedulerParams const scheduler_params
);
```

### Warp Group Roles (lines 70-91)

```cpp
enum class WarpGroupRole {
    Producer = 0,   // WG0: TMA loads + TMA stores
    Consumer0 = 1,  // WG1: GEMM + softmax
    Consumer1 = 2   // WG2: GEMM + softmax (same work, different tile)
};

enum class ProducerWarpRole {
    Mainloop = 0,   // Warp 0: loads Q, K, V from gmem → smem
    Epilogue = 1,   // Warp 1: stores O from smem → gmem via TMA
    Warp2 = 2,      // Warp 2: idle (could be used for masking etc)
    Warp3 = 3       // Warp 3: idle
};
```

### Register Redistribution (lines 141, 169)

```cpp
if (warp_group_role == WarpGroupRole::Producer) {
    cutlass::arch::warpgroup_reg_dealloc<24>();   // Only 24 registers per thread
    // Producer barely does any compute — just TMA commands
} else {
    cutlass::arch::warpgroup_reg_alloc<232>();     // 232 registers per thread
    // Consumers do all the MMA + softmax + quantization
}
```

**Why 24 vs 232?** Total register file per SM is limited. By deallocating registers from the producer warp group (which only issues TMA commands), those registers become available to the consumer warp groups that need them for the large MMA accumulators (tSrS has 128×128 / 256_threads × elements = many registers).

### Producer Flow (lines 140-167)

```
Producer WG0:
├── Warp 0 (Mainloop): Load Q, K, V, SFQ, SFK, SFV, DS via TMA
│   └── Calls collective_mainloop.load(...)
├── Warp 1 (Epilogue): Store O via TMA
│   └── Calls collective_epilogue.tma_store(...)
├── Warp 2: Idle
└── Warp 3: Idle
```

The epilogue warp waits on `barrier_o` before each store — it must wait until the consumer has finished writing O to smem.

### Consumer Flow (lines 168-199)

```
Consumer WG1/WG2:
for each tile:
    1. Allocate O accumulator (zeroed)
    2. Create SoftmaxFused state
    3. Get block coordinates (m_block, head, batch)
    4. Compute n_block_max
    5. If causal and n_block_max <= 0: store zeros, continue
    6. Call collective_mainloop.mma(...)  ← The main computation
    7. Wait on barrier_o
    8. Call collective_epilogue.mma_store(...)  ← Write O to smem
    9. Signal barrier_o  ← Let producer's epilogue warp do TMA store
```

### Pipeline Initialization (lines 101-128)

Three pipelines are created:
- `pipeline_q`: single-stage (Q loaded once per m_block tile)
- `pipeline_k`: 3-stage (triple-buffered K tiles)
- `pipeline_v`: 3-stage (triple-buffered V tiles)

```cpp
PipelineParams pipeline_params_k;
pipeline_params_k.transaction_bytes = TmaTransactionBytesK;  // Size of one K+SFK+DS TMA load
pipeline_params_k.role = producer ? Producer : Consumer;
pipeline_params_k.num_consumers = NumMmaThreads;  // 256 consumer threads
```

### Epilogue Barrier (lines 130-134)

```cpp
// Two groups: Producer (32 threads = 1 warp) and Consumer (256 threads)
uint32_t epilogue_barrier_group_size_list[2] = {NumThreadsPerWarp, NumMmaThreads};
```

This `OrderedSequenceBarrierVarGroupSize` ensures:
1. Consumer finishes writing O to smem
2. Consumer signals (arrives)
3. Producer epilogue warp waits
4. Producer does TMA store
5. Producer signals (arrives)
6. Consumer can overwrite smem O for next tile

---

## 3.6 `tile_scheduler.h` — Tile Assignment (304 lines)

### Three Schedulers (only StaticPersistent is used)

| Scheduler | Grid Size | Tile Assignment |
|---|---|---|
| `SingleTileScheduler` | (num_blocks_m, num_head, num_batch) | 1:1 mapping |
| `StaticPersistentTileScheduler` | (num_sm = 170) | Round-robin loop |
| `DynamicPersistentTileScheduler` | (num_sm) | Atomic fetch-and-add |

### StaticPersistentTileScheduler (used in SA3)

```cpp
// Grid: just 170 CTAs (one per SM)
static dim3 get_grid_dim(Arguments const& args, int num_sm) {
    return {uint32_t(num_sm)};  // 170 for B200
}

// Each CTA loops over tiles:
WorkTileInfo get_initial_work() const {
    return {int(blockIdx.x)};  // Start at SM index
}

WorkTileInfo get_next_work(Params const& params, WorkTileInfo const& current_work) const {
    return {current_work.tile_idx + int(gridDim.x)};  // Stride by num_SMs
}

// Map tile index to (m_block, head, batch):
auto [m_block, bidh, bidb] = get_block_coord(params);
// tile_idx → (m_block, head, batch) via FastDivmod
```

**Why persistent?** Launching one CTA per SM avoids kernel launch overhead. Each CTA processes multiple tiles in a loop, amortizing the pipeline setup cost.

**Tile ordering**: `tile_idx = m_block + head * num_blocks_m + batch * num_blocks_m * num_heads`. This means tiles for the same head/batch are contiguous, which is good for data locality.

---

## 3.7 `named_barrier.h` — Ordered Barriers (118 lines)

### `OrderedSequenceBarrierVarGroupSize<SequenceDepth, SequenceLength>`

A synchronization primitive that enforces ordering between two groups of threads.

In SA3: `<EpiStages=1, 2 groups>`:
- Group 0: Producer epilogue warp (32 threads)
- Group 1: Consumer warp groups (256 threads)

**Protocol:**
```
Step 1: Consumer writes O to smem
Step 2: Consumer calls barrier_o.arrive()  ← signals group 0
Step 3: Producer calls barrier_o.wait()    ← waits for group 1's signal
Step 4: Producer does TMA store of O
Step 5: Producer calls barrier_o.arrive()  ← signals group 1
Step 6: Consumer calls barrier_o.wait()    ← waits for group 0's signal
Step 7: Consumer can now overwrite smem O for next tile
```

---

## 3.8 Summary: Architecture at a Glance

```
                    ┌─────────────────────────────────────┐
                    │         Persistent Kernel            │
                    │  Grid: 170 CTAs, Block: 384 threads  │
                    └────────────────┬────────────────────┘
                                     │
                    ┌────────────────┴────────────────────┐
                    │                                      │
          ┌────────┴────────┐                   ┌─────────┴─────────┐
          │   WG0: Producer  │                   │  WG1/WG2: Consumer │
          │   (24 registers) │                   │  (232 registers)   │
          └────────┬────────┘                   └─────────┬─────────┘
                   │                                       │
    ┌──────────────┼──────────────┐              ┌────────┴────────┐
    │              │              │              │                  │
  Warp 0       Warp 1        Warp 2/3          GEMM-I            GEMM-II
  Mainloop     Epilogue      (idle)          Q × K → S          P × V → O
  Load Q,K,V   Store O                      + softmax           + rescale
  via TMA      via TMA                      + FP4 quant         + finalize
    │              │                              │                  │
    └──────────────┴──────────────────────────────┴──────────────────┘
                         Pipeline Barriers
                    (Q: 1 stage, K/V: 3 stages)
                    Epilogue Barrier (O: ordered)
```

## What to Read Next

- Phase 4: The Mainloop — Line by Line (mainloop_tma_ws.h)
