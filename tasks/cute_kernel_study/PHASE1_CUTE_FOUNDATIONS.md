# Phase 1: CuTe Foundations — From Zero to Layout Algebra

## Prerequisites

- Familiarity with CUDA programming model (grids, blocks, threads, warps, shared memory)
- Understanding of PTX instructions (you've written PTX before)
- Basic understanding of matrix multiplication on GPU (GEMM tiling)

## 1.1 Setup CuTeDSL

```bash
source /mnt/disk1/yiliu7/sage/bin/activate
cd sageattention3_blackwell/csrc/cutlass
pip install -e python/ --no-build-isolation
# Verify:
python -c "import cutlass; import cutlass.cute as cute; print('CuTeDSL OK')"
```

If `cutlass` module isn't available, the CUTLASS headers are still in the repo at:
```
sageattention3_blackwell/csrc/cutlass/
```

For C++ CuTe exploration, the include path is:
```
sageattention3_blackwell/csrc/cutlass/include/cute/
```

---

## 1.2 Layout Algebra — The Most Important CuTe Concept

### What is a Layout?

A `Layout` is the foundational abstraction in CuTe. It maps **logical coordinates** to **physical offsets** (indices into memory). Every tensor, every partition, every MMA instruction uses layouts.

```
Layout = (Shape, Stride)
```

**Shape** defines the logical structure (dimensions).
**Stride** defines how to compute the physical offset from logical coordinates.

### Simple Example

```
Layout<Shape<_4, _8>, Stride<_8, _1>>
```

This represents a 4×8 matrix stored in **row-major** order:
- Shape = (4, 8) — 4 rows, 8 columns
- Stride = (8, 1) — moving one row jumps 8 elements, moving one column jumps 1

The offset formula: `offset(i, j) = i * 8 + j * 1`

### Column-major variant

```
Layout<Shape<_4, _8>, Stride<_1, _4>>
```

Same logical shape (4×8), but stored column-major: `offset(i, j) = i * 1 + j * 4`

### Hierarchical (Nested) Shapes

CuTe's power comes from **hierarchical shapes**. Instead of flat dimensions, you can nest them:

```
Shape<Shape<_2, _4>, _8>
```

This represents a logically 2D structure where mode-0 is itself composed of two sub-modes. It's equivalent to 2×4×8 = 64 elements total.

**Why hierarchical?** Because hardware has natural hierarchies:
- Warp = 32 threads = 4 rows × 8 columns (in the MMA instruction)
- Block = multiple warps
- Grid = multiple blocks

Hierarchical shapes let you express these without flattening.

### Key Layout Operations

#### `make_layout(shape, stride)`
Constructs a layout from shape and stride tuples.

#### `coalesce(layout)`
Simplifies a layout by merging modes with compatible strides. Like "defragmenting" the layout.

```
Layout<(_4, _2), (_2, _1)> → coalesce → Layout<_8, _1>
```
Because 4×stride_2 and 2×stride_1 form a contiguous range.

#### `complement(layout, bound)`
Creates a layout that covers the "gaps" in the original layout's range. Critical for understanding how threads and values are distributed across a tile.

#### `logical_divide(layout, tile)`
Divides a layout's modes by a tile layout, creating a two-level hierarchy:
```
logical_divide(Layout<_16, _1>, Layout<_4, _1>) → Layout<(_4, _4), (_1, _4)>
```
The first mode is "within-tile" indices, the second is "across-tiles" indices.

#### `zipped_divide(layout, tile)`
Like `logical_divide` but zips the inner and outer parts together:
```
zipped_divide(tensor, tile) → ((tile_inner_modes...), (tile_outer_modes...))
```
This is the **most confusing but most important** operation. It's how CuTe assigns data to threads and MMA atoms.

**Mental model for zipped_divide:**
Think of it as "fold the layout into tiles, then give me two views: the elements within each tile, and the tiles themselves."

### Practice Exercise 1: Layout Arithmetic

Using CuTeDSL or pen-and-paper:

1. Create `Layout<_8, _1>` (8 contiguous elements). What's the offset of index 5? → Answer: 5
2. Create `Layout<(_4, _2), (_1, _4)>` (4×2, strided). What's the offset of coord (3, 1)? → 3*1 + 1*4 = 7
3. Apply `logical_divide` to `Layout<_16, _1>` with tile `Layout<_4, _1>`. Result?
4. What does `complement(Layout<_4, _2>, 8)` give? → `Layout<_2, _1>` (fills the gaps at offsets 0,1 between stride-2 elements)

---

## 1.3 Tensor = Pointer + Layout

A `Tensor` in CuTe is simply:

```
Tensor = (data_pointer, Layout)
```

It's a **non-owning view**. The layout defines how logical coordinates map to physical addresses relative to the pointer.

### Memory Spaces

CuTe tensors can live in different memory spaces:

| Constructor | Memory | Used for |
|---|---|---|
| `make_gmem_ptr(ptr)` | Global memory (HBM) | Input/output tensors |
| `make_smem_ptr(ptr)` | Shared memory (SRAM) | On-chip buffers |
| Register fragments | Register file | Per-thread compute |

### Creating Tensors

```cpp
// Global memory tensor (from PyTorch tensor pointer)
auto mQ = make_tensor(make_gmem_ptr(ptr_Q), shape, stride);

// Shared memory tensor (from __shared__ allocation)
auto sQ = make_tensor(make_smem_ptr(smem_q_ptr), SmemLayoutQ{});

// Register fragment (allocated on the fly, owned)
auto fragment = make_fragment_like<float>(some_tensor);
```

### The SA3 Pattern

In SageAttention3, you'll see this pattern everywhere:

```cpp
// 1. Create smem tensor from SharedStorage
Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.begin()), SmemLayoutQ{});

// 2. Create gmem tensor from TMA descriptor
Tensor mQ = mainloop_params.tma_load_Q.get_tma_tensor(mainloop_params.shape_Q);

// 3. Tile gmem tensor for the current block
Tensor gQ = local_tile(mQ(_, _, bidh, bidb), select<0, 2>(TileShape_MNK{}), make_coord(m_block, _0{}));

// 4. Create register fragment for compute
Tensor tSrQ = thread_mma_qk.partition_fragment_A(sQ);
```

### Key Tensor Operations

#### `local_tile(tensor, tile_shape, coord)`
Extracts a tile from a tensor at the given coordinate. Returns a view (no copy).

Example from SA3:
```cpp
// From the full K matrix (seqlen_k × head_dim), extract a (kBlockN × kHeadDim) tile at position n_block
Tensor gK = local_tile(mK(_, _, bidh, bidb), select<1, 2>(TileShape_MNK{}), make_coord(n_block, _0{}));
```

#### `partition_fragment_A/B/C(tensor)`
Part of TiledMMA. Distributes elements of a tensor across threads according to the MMA instruction's data layout. Returns a register-resident fragment.

#### `as_position_independent_swizzle_tensor(tensor)`
Removes the absolute pointer from swizzled smem tensors, making them position-independent for shared memory copy operations.

### Practice Exercise 2: Tensor Views

Trace through this SA3 code mentally:

```cpp
// shared_storage.smem_q is an ArrayEngine with capacity = cosize(SmemLayoutQ)
Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.begin()), SmemLayoutQ{});
// sQ is now a (kBlockM × kHeadDim) view of shared memory
```

Q: What is the logical shape of `sQ`?
A: For SA3 with kBlockM=128, kHeadDim=128: sQ has shape (128, 128), but with a swizzled layout for bank-conflict-free access.

---

## 1.4 Tiling and Partitioning

This is where CuTe gets confusing but powerful. The key pattern:

```
Full tensor → local_tile (select a block) → partition (distribute across threads) → fragments (per-thread data)
```

### `local_tile`

Slices a large tensor into block-sized tiles:

```cpp
// mQ has shape (seqlen_q, d, h, b)
// select<0, 2>(TileShape_MNK{}) = (kBlockM, kHeadDim)
// make_coord(m_block, _0{}) = which tile
Tensor gQ = local_tile(mQ(_, _, bidh, bidb), select<0, 2>(TileShape_MNK{}), make_coord(m_block, _0{}));
// gQ now has shape (kBlockM, kHeadDim) — a single tile of Q
```

### `partition_S` / `partition_D` (TMA partitioning)

For TMA (Tensor Memory Accelerator) operations, data is partitioned into TMA-sized chunks:

```cpp
auto block_tma_q = mainloop_params.tma_load_Q.get_slice(_0{});
Tensor tQgQ = block_tma_q.partition_S(gQ);  // Source (gmem) partition
Tensor tQsQ = block_tma_q.partition_D(sQ);  // Destination (smem) partition
// Then: copy(tma_load_Q.with(barrier, ...), tQgQ, tQsQ);
```

### `partition_fragment_A/B/C` (MMA partitioning)

For MMA (Matrix Multiply-Accumulate), data is partitioned according to the MMA instruction's thread-value mapping:

```cpp
TiledMmaQK tiled_mma_qk;
auto thread_mma_qk = tiled_mma_qk.get_thread_slice(thread_idx);

// A-operand (Q): partitioned for the "A" input of the MMA
Tensor tSrQ = thread_mma_qk.partition_fragment_A(sQ);
// B-operand (K): partitioned for the "B" input
Tensor tSrK = thread_mma_qk.partition_fragment_B(sK(_,_,Int<0>{}));
// C-operand (S = Q×K): partitioned for the accumulator
Tensor tSrS = partition_fragment_C(tiled_mma_qk, select<0, 1>(TileShape_MNK{}));
```

### `retile_S` / `retile_D` (Copy retiling)

Adapts a tensor's layout between copy-compatible and MMA-compatible forms:

```cpp
auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(thread_idx);
Tensor tSsQ = smem_thr_copy_Q.partition_S(as_position_independent_swizzle_tensor(sQ));
Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
// Now we can copy: copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
```

### The Full Data Movement Pipeline in SA3

```
                   TMA                   LDSM                    MMA
gmem ──────────► smem ──────────────► registers ──────────► accumulator
     tma_load_Q         smem_tiled_copy_Q         tiled_mma_qk.gemm()
     partition_S/D      partition_S, retile_D      partition_fragment_A/B
```

### Practice Exercise 3: Trace the Partition Chain

For Q in SA3 mainloop:
1. `sQ` = smem tensor, shape (128, 128) [from SmemLayoutQ]
2. `tSrQ` = `thread_mma_qk.partition_fragment_A(sQ)` — what's this shape?
   - For SM120 NVFP4 MMA with 128 threads: each thread gets a fragment of shape `(vals_per_thread, MMA_M, MMA_K)`
   - With kBlockM=128, kHeadDim=128, AtomShape 16×32×64: `(32, 8, 2)` register elements per thread
3. `tSsQ` = copy partition of smem Q for this thread
4. `tSrQ_copy_view` = retiled view of the register fragment, compatible with the smem→reg copy

---

## 1.5 Copy Atoms and TMA

### Copy Atoms

A `Copy_Atom` defines how to move data between memory levels. Different atoms for different hardware:

| Atom | What it does | Used for |
|---|---|---|
| `SM90_TMA_LOAD` | Tensor Memory Accelerator load | gmem → smem (async, DMA) |
| `SM90_TMA_STORE` | TMA store | smem → gmem (async) |
| `SM75_U32x4_LDSM_N` | Load Shared Memory (LDSM) | smem → registers |
| `UniversalCopy<T>` | Generic copy | Scale factors smem → reg |

### TMA (Tensor Memory Accelerator)

TMA is Hopper/Blackwell's hardware DMA engine. It loads/stores multi-dimensional tiles from global memory to shared memory **without using any threads** — the hardware does it all.

Key components:
1. **TMA Descriptor**: Created on the host, describes the tensor's layout in global memory
2. **Pipeline barriers**: Synchronize between TMA loads and consumer warps

### TMA in SA3 — The Pattern

```cpp
// Host side (in to_underlying_arguments):
Tensor mQ = make_tensor(make_gmem_ptr(args.ptr_Q), args.shape_Q, args.stride_Q);
TMA_Q tma_load_Q = make_tma_copy(
    GmemTiledCopy{},    // SM90_TMA_LOAD
    mQ,                  // The global memory tensor
    SmemLayoutQ{},       // Target smem layout
    select<0, 2>(TileShape_MNK{}),  // Tile shape to load
    _1{});               // No multicast

// Device side (in load function):
// 1. Get the TMA tensor (a logical view of the full gmem tensor)
Tensor mQ = mainloop_params.tma_load_Q.get_tma_tensor(mainloop_params.shape_Q);

// 2. Select the tile to load
Tensor gQ = local_tile(mQ(_, _, bidh, bidb), select<0, 2>(TileShape_MNK{}), make_coord(m_block, _0{}));

// 3. Partition for TMA
auto block_tma_q = mainloop_params.tma_load_Q.get_slice(_0{});
Tensor tQgQ = block_tma_q.partition_S(gQ);   // Source partitions
Tensor tQsQ = block_tma_q.partition_D(sQ);   // Destination partitions

// 4. Acquire pipeline stage and issue TMA
pipeline_q.producer_acquire(smem_pipe_write_q);
copy(mainloop_params.tma_load_Q.with(*pipeline_q.producer_get_barrier(smem_pipe_write_q), 0),
     tQgQ, tQsQ);
++smem_pipe_write_q;
```

### Pipeline Pattern (Producer-Consumer)

```
Producer (WG0):                    Consumer (WG1/WG2):
  producer_acquire(stage)            barrier_token = consumer_try_wait(stage)
  copy(TMA, src, dst)                consumer_wait(stage, token)
  producer_release(stage)            [use the data]
  advance to next stage              consumer_release(stage)
                                     advance to next stage
```

SA3 uses 3 pipeline stages (`kStages = 3`) for K and V, meaning:
- While consumer processes stage 0, producer loads stage 1 and 2
- Triple-buffering hides memory latency

### Practice Exercise 4: TMA Pipeline Trace

Walk through the SA3 `load()` function (mainloop_tma_ws.h:432-554):

1. Q is loaded **once** via `pipeline_q` (single-stage pipeline) — why? Because Q stays the same across all K,V tiles.
2. The first K,V tile is loaded: pipeline_k stage 0, pipeline_v stage 0
3. Then remaining tiles are loaded in a loop: each K uses one pipeline_k stage, each V uses one pipeline_v stage
4. The loop `#pragma unroll 2` hints the compiler to partially unroll

Q: Why is DS (delta_s) loaded through the K pipeline rather than having its own pipeline?
A: Because DS is always needed at the same time as K (for the smooth-Q correction), so they share a pipeline barrier — a single TMA transaction includes both K, SFK, and DS.

---

## 1.6 MMA Atoms — The Hardware Multiply-Accumulate

### What is a TiledMMA?

A `TiledMMA` wraps a hardware MMA instruction and tiles it across the full block dimensions.

In SA3:
```cpp
using TiledMmaQK = decltype(cute::make_tiled_mma(
    cute::SM120::BLOCKSCALED::SM120_16x32x64_TN_VS_NVFP4{},  // The atom
    AtomLayoutMNK{},  // Layout<Shape<_8, _1, _1>> for kBlockM=128
    Tile<PermTileM, PermTileN, PermTileK>{}  // Tile<_128, _32, _128>
));
```

This means:
- **Atom**: One `mma.sync` instruction computes m16×n32×k64 (via 4 sub-instructions of m16×n8×k64)
- **AtomLayout**: _8 atoms tiled along M direction → covers 8×16 = 128 rows
- **PermTile**: Full tile coverage is 128×32×128 (but kBlockN=128, so we loop over N in chunks of 32... actually the N dimension needs size<2>(tSrK) iterations)

Wait, let me correct this. The tile shape is:
```
TileShape_MNK = Shape<Int<128>, Int<128>, Int<128>>  // for kHeadDim=128
```

But the MMA atom processes 16×32×64 at a time. With 8 atoms along M: covers 128×32×64. So:
- Along K (head dimension 128): 2 iterations (128/64 = 2)
- Along N (sequence dimension 128): 4 iterations (128/32 = 4)

### The SM120 NVFP4 MMA Atom

From `cute_extension.h`:

```cpp
struct SM120_16x32x64_TN_VS_NVFP4 {
    // D = A × B + C, with A scale factors (SFA) and B scale factors (SFB)
    using DRegisters = float[16];    // 16 output values per thread
    using ARegisters = uint32_t[4];  // 4 × 32-bit = 128 bits = 32 FP4 elements
    using BRegisters = uint32_t[8];  // 8 × 32-bit = 256 bits = 64 FP4 elements
    using CRegisters = float[16];    // Accumulator
    using SFARegisters = RegTypeSF[1]; // 1 × 32-bit = 4 E4M3 scale factors
    using SFBRegisters = RegTypeSF[1]; // 1 × 32-bit = 4 E4M3 scale factors
};
```

The PTX instruction:
```
mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3
```

Breakdown:
- `mma.sync.aligned` — synchronous warp-level MMA
- `kind::mxf4nvf4` — microscaling FP4 × NVFP4 format
- `block_scale` — uses block-level scale factors
- `scale_vec::4X` — 4 scale factors per vector (SF covers 16 elements each)
- `m16n8k64` — matrix dimensions per instruction
- `row.col` — A is row-major, B is column-major
- `f32.e2m1.e2m1.f32.ue4m3` — output=FP32, A=E2M1(FP4), B=E2M1(FP4), accum=FP32, SF=UE4M3(unsigned FP8)

### MMA Traits: Thread-Value Layouts

From `MMA_Traits<SM120_16x32x64_TN_VS_NVFP4>`:

```cpp
// How 32 threads map to the 16×64 A matrix elements:
using ALayout = Layout<Shape <Shape <  _4,_8>,Shape < _8,_2,  _2>>,
                       Stride<Stride<_128,_1>,Stride<_16,_8,_512>>>;
// (T32, V32) -> (M16, K64)
// 32 threads × 32 values = 1024 elements = 16 × 64 ✓

// How 32 threads map to the 32×64 B matrix elements:
using BLayout = Layout<Shape <Shape < _4,_8>,Shape <_8,  _2, _4>>,
                       Stride<Stride<_256,_1>,Stride<_32,_1024, _8>>>;
// (T32, V64) -> (N32, K64)
// 32 threads × 64 values = 2048 elements = 32 × 64 ✓

// How 32 threads map to the 16×32 C/D matrix elements:
using CLayout = Layout<Shape <Shape < _4,_8>,Shape < Shape<_2, _4>,_2>>,
                       Stride<Stride<_32,_1>,Stride<Stride<_16, _128>,_8>>>;
// (T32, V16) -> (M16, N32)
// 32 threads × 16 values = 512 elements = 16 × 32 ✓
```

### Scale Factor Layouts

For block-scaled MMA, scale factors (SF) have their own layouts:

```cpp
// SFALayout: (T32, V64) -> (M16, K64)
using SFALayout = Layout<Shape <Shape<_2,_2,_8>,_64>,
                         Stride<Stride<_8,_0,_1>,_16>>;

// SFBLayout: (T32, V64) -> (N32, K64)
using SFBLayout = Layout<Shape <Shape<_4,_8>,_64>,
                         Stride<Stride<_8,_1>, _32>>;
```

Each SF is an E4M3 value that scales a block of 16 FP4 elements. With K=64, there are 64/16 = 4 scale factor blocks along K.

### How MMA is Called in SA3

```cpp
// GEMM-I: Q × K → S (attention scores)
cute::gemm(tiled_mma_qk,
    make_zip_tensor(tSrQ(_, _, k_block), tSrSFQ(_, _, k_block)),   // A + SFA
    make_zip_tensor(tSrK(_, _, k_block), tSrSFK(_, _, k_block)),   // B + SFB
    tSrS);                                                          // C (accumulator)

// GEMM-II: P × V → O (attention output)
cute::gemm(tiled_mma_pv,
    make_zip_tensor(tOrP(_, _, v_block), tOrSFP(_, _, v_block)),
    make_zip_tensor(tOrVt(_, _, v_block), tOrSFVt(_, _, v_block)),
    tOrO_store);
```

The `make_zip_tensor` zips the data tensor with its scale factor tensor — this is how block-scaled MMA receives both data and scales.

### Practice Exercise 5: MMA Dimensions

For SA3 with kBlockM=128, kBlockN=128, kHeadDim=128:

1. How many MMA atoms tile along M? → 8 (AtomLayoutMNK = Shape<_8,_1,_1>)
2. Each atom is 16×32×64. What's the tiled dimension? → 128×32×64
3. How many k_block iterations for GEMM-I? → kHeadDim/64 = 128/64 = 2
4. How many v_block iterations for GEMM-II? → kBlockN/64 = 128/64 = 2 (but this is over the shared K dim of the P×V matmul)
5. What data type is the accumulator tSrS? → float (FP32)
6. How many values does each thread hold in tSrS? → 16 per atom × 8 atoms × 1 (MmaN) = ... need to compute from CLayout

---

## 1.7 Putting It All Together: The CuTe Data Flow

```
┌─────────────────────────────────────────────────────────┐
│  Global Memory (HBM)                                     │
│  Q[B,H,S_q,D]  K[B,H,S_k,D]  V[B,H,D,S_k]             │
│  SFQ[B,H,S_q,D/16]  SFK[B,H,S_k,D/16]  SFV[B,H,D/16,S_k] │
│  DS[B,H,S_q or 128, S_k]                                │
└────────────┬────────────────────────────────────────────┘
             │ TMA Load (SM90_TMA_LOAD)
             │ Async DMA, no thread involvement
             │ Pipeline barriers for synchronization
             ▼
┌─────────────────────────────────────────────────────────┐
│  Shared Memory (SRAM, ~100KB+)                          │
│  sQ[128×128]  sK[128×128×3]  sV[128×128×3]             │
│  sSFQ, sSFK[×3], sSFV[×3]                              │
│  sDS[128×128×3]                                         │
│  sO[128×128]  (for output epilogue)                     │
│                                                          │
│  Layouts: Swizzled for bank-conflict-free access         │
│  Pipeline: Triple-buffered (kStages=3) for K,V           │
└────────────┬────────────────────────────────────────────┘
             │ LDSM (SM75_U32x4_LDSM_N) / UniversalCopy
             │ smem → register copy
             │ make_tiled_copy_A/B + partition + retile
             ▼
┌─────────────────────────────────────────────────────────┐
│  Register File                                           │
│  tSrQ[32, 8, 2]  — Q fragment per thread                │
│  tSrK[64, 1, 2]  — K fragment (one stage)               │
│  tSrSFQ, tSrSFK  — Scale factor fragments               │
│  tSrS[16, 8, 4]  — QK accumulator (S = scores)          │
│  tOrP[FP4]        — Quantized softmax output             │
│  tOrSFP[E4M3]     — P scale factors (computed, not loaded)│
│  tOrVt, tOrSFVt   — V fragment + scales                 │
│  tOrO[16, 8, ...]  — Output accumulator                  │
└────────────┬────────────────────────────────────────────┘
             │ cute::gemm (block-scaled MMA)
             │ mma.sync.aligned.kind::mxf4nvf4
             ▼
┌─────────────────────────────────────────────────────────┐
│  Result: tSrS (QK scores) or tOrO (output)              │
│  + online softmax + FP4 quantization (fused)            │
└─────────────────────────────────────────────────────────┘
```

---

## Summary of Phase 1 Key Takeaways

1. **Layout = (Shape, Stride)** — the universal abstraction. Master this and everything else follows.
2. **Tensor = pointer + Layout** — a non-owning view. No data copying.
3. **Tiling = local_tile** — extract a block-sized piece from a larger tensor.
4. **Partitioning = partition_fragment_A/B/C** — distribute data across threads according to hardware MMA layout.
5. **Copy = TMA + pipeline** — asynchronous DMA from gmem to smem, synchronized via barriers.
6. **MMA = cute::gemm** — hardware matrix multiply, driven by MMA_Traits and TiledMMA.
7. **Block-scaled MMA** = data + scale factors zipped together, hardware handles dequantization.

## What to Read Next

- Phase 2: FMHA in CuTeDSL — understand the attention algorithm
- CuTe docs: `csrc/cutlass/media/docs/cute/01_layout.md` (if available)
- CuTeDSL examples: Start with `blackwell/tutorial_gemm/nvfp4_gemm_0.py`
