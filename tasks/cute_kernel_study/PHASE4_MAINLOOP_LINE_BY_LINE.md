# Phase 4: The Mainloop — Line by Line

## File: `mainloop_tma_ws.h` (908 lines)

This is the heart of SageAttention3. Every attention computation happens here.

---

## 4.1 CollectiveMainloopFwd Type Definitions (lines 35-154)

### TMA Descriptors (lines 70-125)

Seven TMA descriptors are needed (one per tensor to load):

```cpp
TMA_Q       // Q data: (kBlockM, kHeadDim) tile, loaded once per m_block
TMA_KV      // K data: (kBlockN, kHeadDim) tile, loaded per n_block
TMA_Vt      // V^T data: (kHeadDim, kBlockN) tile, loaded per n_block
TMA_DS      // DeltaS: (kBlockM, kBlockN) tile, loaded per n_block
TMA_SFQ     // Q scale factors, loaded once per m_block
TMA_SFKV    // K scale factors, loaded per n_block
TMA_SFVt    // V scale factors, loaded per n_block
```

Each TMA descriptor encodes:
- Source tensor layout in global memory
- Destination layout in shared memory
- Tile shape to transfer

**Why separate TMA for K and V?** K has layout `[S_k, D]` while V^T has layout `[D, S_k]`. Different memory layouts require different TMA descriptors.

### Transaction Bytes (lines 142-153)

```cpp
// Q transaction: Q data + SFQ scale factors
TmaTransactionBytesQ = bits_to_bytes(cosize(SmemLayoutSFQ) * 8) +
                       bits_to_bytes(size(SmemLayoutQ) * 4);    // 4 bits per FP4

// K transaction: K data + SFK + DeltaS
TmaTransactionBytesK = bits_to_bytes(cosize(SmemLayoutSFK[one_stage]) * 8) +
                       bits_to_bytes(cosize(SmemLayoutDS[one_stage]) * 32) +  // float = 32 bits
                       bits_to_bytes(size(SmemLayoutK[one_stage]) * 4);

// V transaction: V data + SFV
TmaTransactionBytesV = bits_to_bytes(cosize(SmemLayoutSFVt[one_stage]) * 8) +
                       bits_to_bytes(size(SmemLayoutVt[one_stage]) * 4);
```

The pipeline barrier must know the expected byte count to wait for all TMA operations to complete.

---

## 4.2 `to_underlying_arguments()` — TMA Setup (lines 200-265)

This runs on the **host** before kernel launch. It creates all 7 TMA descriptors.

### Step-by-step:

```cpp
// 1. Create gmem tensors from raw pointers
Tensor mQ = make_tensor(make_gmem_ptr(args.ptr_Q), args.shape_Q, args.stride_Q);
Tensor mK = make_tensor(make_gmem_ptr(args.ptr_K), args.shape_K, args.stride_K);
Tensor mVt = make_tensor(make_gmem_ptr(args.ptr_Vt), args.shape_Vt, args.stride_Vt);

// 2. Create TMA copy operations
TMA_Q tma_load_Q = make_tma_copy(
    GmemTiledCopy{},   // SM90_TMA_LOAD
    mQ,                 // The full Q tensor in gmem
    SmemLayoutQ{},      // How to lay it out in smem
    select<0,2>(TileShape_MNK{}),  // Tile shape: (kBlockM, kHeadDim) = (128, 128)
    _1{});              // No multicast

// 3. For scale factors, create special layouts
LayoutSF layout_sfq = BlkScaledConfig::tile_atom_to_shape_SFQKV(args.shape_SFQ);
Tensor mSFQ = make_tensor(make_gmem_ptr(args.ptr_SFQ), layout_sfq);
TMA_SFQ tma_load_sfq = make_tma_copy<uint16_t>(  // Note: uint16 for 8-bit SFs
    GmemTiledCopySF{},
    mSFQ,
    SmemLayoutSFQ{},
    make_shape(kBlockM, kHeadDim),
    _1{});

// 4. DeltaS uses a special broadcasted layout
LayoutDS layout_ds = tile_to_shape(SmemLayoutAtomDS{}, ...);
// SmemLayoutAtomDS = Layout<Shape<128, 128>, Stride<_0, _1>>
// Stride<_0, _1> means: all rows share the same data! Broadcasting along M.
```

**The DeltaS broadcasting trick**: `SmemLayoutAtomDS` has stride `(_0, _1)` for the M dimension, meaning every row reads the same correction vector. This is because `delta_s` = mean(Q) × K^T, and when `per_block_mean = false`, the mean is computed over the whole tile of 128 Q tokens.

---

## 4.3 `load()` — The Producer TMA Loop (lines 432-554)

### Function Signature

```cpp
void load(
    Params const& mainloop_params,
    SchedulerParams const& scheduler_params,
    MainloopPipelineQ pipeline_q,      // 1-stage for Q
    MainloopPipeline pipeline_k,       // 3-stage for K
    MainloopPipeline pipeline_v,       // 3-stage for V
    PipelineStateQ& smem_pipe_write_q, // Current write stage for Q
    PipelineState& smem_pipe_write_k,  // Current write stage for K
    PipelineState& smem_pipe_write_v,  // Current write stage for V
    SharedStorage &shared_storage,
    WorkTileInfo work_tile_info,
    int& work_idx,
    int& tile_count_semaphore
);
```

### Step-by-step Walkthrough

**Lines 448-461: Create smem tensors**
```cpp
Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.begin()), SmemLayoutQ{});
Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.begin()), SmemLayoutK{});
Tensor sVt = make_tensor(make_smem_ptr(shared_storage.smem_v.begin()), SmemLayoutVt{});
Tensor sSFQ = make_tensor(make_smem_ptr(shared_storage.smem_SFQ.begin()), SmemLayoutSFQ{});
Tensor sSFK = make_tensor(make_smem_ptr(shared_storage.smem_SFK.begin()), SmemLayoutSFK{});
Tensor sSFVt = make_tensor(make_smem_ptr(shared_storage.smem_SFV.begin()), SmemLayoutSFVt{});
Tensor sDS = make_tensor(make_smem_ptr(shared_storage.smem_ds.begin()), SmemLayoutDS{});
```

**Lines 463-469: Create gmem TMA tensors**
```cpp
Tensor mQ = mainloop_params.tma_load_Q.get_tma_tensor(mainloop_params.shape_Q);
// mQ is NOT a pointer — it's a logical view that TMA uses to compute addresses
```

**Lines 473-485: Tile gmem tensors for current block**
```cpp
Tensor gQ = local_tile(mQ(_, _, bidh, bidb), select<0,2>(TileShape_MNK{}),
                        make_coord(m_block, _0{}));
// gQ = (kBlockM, kHeadDim) at position m_block

Tensor gK = local_tile(mK(_, _, bidh, bidb), select<1,2>(TileShape_MNK{}),
                        make_coord(_, _0{}));
// gK = (kBlockN, kHeadDim, num_n_blocks) — ALL K tiles, indexed by n_block later
```

**Lines 486-506: Partition for TMA**
```cpp
auto block_tma_q = mainloop_params.tma_load_Q.get_slice(_0{});
Tensor tQgQ = block_tma_q.partition_S(gQ);    // Source partitions in gmem
Tensor tQsQ = block_tma_q.partition_D(sQ);    // Destination partitions in smem

auto block_tma_k = mainloop_params.tma_load_K.get_slice(cluster_local_block_id.x);
Tensor tKgK = group_modes<0,3>(block_tma_k.partition_S(gK));  // Grouped by n_block
Tensor tKsK = group_modes<0,3>(block_tma_k.partition_D(sK));  // Grouped by stage
```

**Lines 509-530: First tile load (Q + first K/V)**
```cpp
n_block = n_block_max - 1;  // Start from the LAST K,V block
int lane_predicate = cute::elect_one_sync();  // Only 1 thread issues TMA

if (lane_predicate) {
    // Load Q (once, single-stage pipeline)
    pipeline_q.producer_acquire(smem_pipe_write_q);
    copy(tma_load_Q.with(*pipeline_q.producer_get_barrier(...), 0), tQgQ, tQsQ);
    copy(tma_load_SFQ.with(*pipeline_q.producer_get_barrier(...), 0), tQgSFQ, tQsSFQ);
    ++smem_pipe_write_q;

    // Load first K + SFK + DS
    pipeline_k.producer_acquire(smem_pipe_write_k);
    copy(tma_load_K.with(...), tKgK(_, n_block), tKsK(_, smem_pipe_write_k.index()));
    copy(tma_load_SFK.with(...), tKgSFK(_, n_block), tKsSFK(_, smem_pipe_write_k.index()));
    copy(tma_load_DS.with(...), tDSgDS(_, n_block), tDSsDS(_, smem_pipe_write_k.index()));
    ++smem_pipe_write_k;

    // Load first V + SFV
    pipeline_v.producer_acquire(smem_pipe_write_v);
    copy(tma_load_Vt.with(...), tVgVt(_, n_block), tVsVt(_, smem_pipe_write_v.index()));
    copy(tma_load_SFVt.with(...), tVgSFVt(_, n_block), tVsSFVt(_, smem_pipe_write_v.index()));
    ++smem_pipe_write_v;
}
```

**Key insight**: Q is loaded through `pipeline_q` (1 stage) while K,V use `pipeline_k`/`pipeline_v` (3 stages). This is because Q stays the same for all n_blocks, while K,V change each iteration.

**Lines 532-552: Remaining tiles**
```cpp
n_block--;
if (lane_predicate) {
    #pragma unroll 2
    for (; n_block >= 0; --n_block) {
        // Load K[n_block] + SFK[n_block] + DS[n_block]
        pipeline_k.producer_acquire(smem_pipe_write_k);
        copy(tma_load_K.with(...), tKgK(_, n_block), tKsK(_, smem_pipe_write_k.index()));
        copy(tma_load_SFK.with(...), tKgSFK(_, n_block), tKsSFK(_, smem_pipe_write_k.index()));
        copy(tma_load_DS.with(...), tDSgDS(_, n_block), tDSsDS(_, smem_pipe_write_k.index()));
        ++smem_pipe_write_k;

        // Load V[n_block] + SFV[n_block]
        pipeline_v.producer_acquire(smem_pipe_write_v);
        copy(tma_load_Vt.with(...), tVgVt(_, n_block), tVsVt(_, smem_pipe_write_v.index()));
        copy(tma_load_SFVt.with(...), tVgSFVt(_, n_block), tVsSFVt(_, smem_pipe_write_v.index()));
        ++smem_pipe_write_v;
    }
}
```

---

## 4.4 `mma()` — The Consumer Main Loop (lines 573-903)

This is the most complex function. Let's break it down into stages.

### Stage A: Setup (lines 596-670)

**Lines 596-602: Create smem tensors and identity tensors**
```cpp
Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.begin()), SmemLayoutQ{});
Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.begin()), SmemLayoutK{});
Tensor sVt = make_tensor(make_smem_ptr(shared_storage.smem_v.begin()), SmemLayoutVt{});
Tensor sDS = make_tensor(make_smem_ptr(shared_storage.smem_ds.begin()), SmemLayoutDS{});
Tensor sSFQ = make_tensor(make_smem_ptr(shared_storage.smem_SFQ.begin()), SmemLayoutSFQ{});
Tensor sSFK = make_tensor(make_smem_ptr(shared_storage.smem_SFK.begin()), SmemLayoutSFK{});
Tensor sSFVt = make_tensor(make_smem_ptr(shared_storage.smem_SFV.begin()), SmemLayoutSFVt{});
```

**Lines 604-609: Get per-thread MMA slices**
```cpp
TiledMmaQK tiled_mma_qk;
TiledMmaPV tiled_mma_pv;
auto thread_mma_qk = tiled_mma_qk.get_thread_slice(thread_idx);
auto thread_mma_pv = tiled_mma_pv.get_thread_slice(thread_idx);
```

**Lines 611-620: Create register fragments**
```cpp
Tensor tSrQ = thread_mma_qk.partition_fragment_A(sQ);         // Q for GEMM-I
Tensor tSrK = thread_mma_qk.partition_fragment_B(sK(_,_,0));  // K for GEMM-I (one stage)
Tensor tOrVt = thread_mma_pv.partition_fragment_B(sVt(_,_,0)); // V for GEMM-II
Tensor tOrP = make_tensor_like<Element>(LayoutP{});             // P in FP4 (register-computed)
Tensor tSrSFQ = partition_fragment_SFA(sSFQ, thread_mma_qk);   // Q scale factors
Tensor tSrSFK = partition_fragment_SFB(sSFK(_,_,0), thread_mma_qk);  // K scale factors
Tensor tOrSFVt = partition_fragment_SFB(sSFVt(_,_,0), thread_mma_pv); // V scale factors
Tensor tOrSFP = make_tensor<ElementSF>(LayoutSFP{});            // P scale factors (computed)
Tensor tSrDS = make_tensor<float>(Shape<_8, _4>{}, Stride<_1, _8>{}); // DS fragment
```

**Lines 622-660: Set up smem → register copy operations**

For each tensor that needs to be copied from smem to registers, we create a tiled copy:

```cpp
// Q copy (smem → reg)
auto smem_tiled_copy_Q = make_tiled_copy_A(SmemCopyAtomQ{}, tiled_mma_qk);
auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(thread_idx);
Tensor tSsQ = smem_thr_copy_Q.partition_S(as_position_independent_swizzle_tensor(sQ));
Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);

// K copy (smem → reg)
auto smem_tiled_copy_K = make_tiled_copy_B(SmemCopyAtomKV{}, tiled_mma_qk);
auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(thread_idx);
Tensor tSsK = smem_thr_copy_K.partition_S(as_position_independent_swizzle_tensor(sK));
Tensor tSrK_copy_view = smem_thr_copy_K.retile_D(tSrK);

// V copy (smem → reg)
auto smem_tiled_copy_V = make_tiled_copy_B(SmemCopyAtomKV{}, tiled_mma_pv);
// ... similar pattern

// Scale factor copies use custom TV layouts from get_layoutSFA_TV / get_layoutSFB_TV
auto smem_tiled_copy_SFQ = make_tiled_copy_impl(SmemCopyAtomSF{},
    get_layoutSFA_TV(tiled_mma_qk), ...);
```

**Lines 662-665: Consumer wait helper**
```cpp
auto consumer_wait = [](auto& pipeline, auto& smem_pipe_read) {
    auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
    pipeline.consumer_wait(smem_pipe_read, barrier_token);
};
```

**Lines 672-704: Helper lambdas**

```cpp
// Copy one k_block from smem to registers
auto copy_k_block = [&](auto block_id) {
    auto tSsK_stage = tSsK(_, _, _, smem_pipe_read_k.index());  // Current pipeline stage
    copy(smem_tiled_copy_K, tSsK_stage(_, _, block_id), tSrK_copy_view(_, _, block_id));
    copy(smem_tiled_copy_SFK, tSsSFK_stage(_, _, block_id), tSrSFK_copy_view(_, _, block_id));
};

// Copy one v_block from smem to registers
auto copy_v_block = [&](auto block_id) {
    // Similar pattern for V and SFV
};

// Add delta_s correction to the S accumulator
auto add_delta_s = [&](auto& acc) {
    auto tSsDS_stage = recast<float4>(sDS(_, _, smem_pipe_read_k.index()));
    auto acc_float4 = recast<float4>(acc);
    int quad_id = (threadIdx.x % 4) * 2;
    for (int i = 0; i < 4; i++) {
        auto num = quad_id + i * 8;
        float4 delta_s_0 = tSsDS_stage(make_coord(_0{}, _0{}), make_coord(num, _0{}));
        float4 delta_s_1 = tSsDS_stage(make_coord(_0{}, _0{}), make_coord(num + 1, _0{}));
        // Broadcast delta_s to the accumulator locations
        acc_float4(..., _0{}, i) = delta_s_0;
        acc_float4(..., _1{}, i) = delta_s_0;  // Same value — broadcast!
        acc_float4(..., _0{}, i) = delta_s_1;
        acc_float4(..., _1{}, i) = delta_s_1;
    }
};
```

**The add_delta_s trick**: Instead of adding delta_s *after* the GEMM (which would require a separate addition pass), SA3 **initializes the accumulator with delta_s** before the GEMM. Since GEMM computes `C = A × B + C`, setting `C = delta_s` before the first GEMM makes the result `S = Q × K + delta_s`. This is the smooth-Q correction: `delta_s = mean(Q_block) × K^T`.

### Stage B: First Q×K GEMM (lines 705-729)

```cpp
// Wait for Q to arrive in smem
consumer_wait(pipeline_q, smem_pipe_read_q);

// Copy Q + SFQ from smem to registers (one-time)
copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
copy(smem_tiled_copy_SFQ, tSsSFQ, tSrSFQ_copy_view);
pipeline_q.consumer_release(smem_pipe_read_q);
++smem_pipe_read_q;

// Allocate S accumulator
Tensor tSrS = partition_fragment_C(tiled_mma_qk, select<0,1>(TileShape_MNK{}));
// tSrS_converion_view: reinterpret tSrS with a layout suitable for FP4 conversion
Tensor tSrS_converion_view = make_tensor(tSrS.data(), convert_to_conversion_layout(tSrS.layout()));
// AbsMaxP: per-16-element block maxima for microscaling
Tensor AbsMaxP = make_tensor_like<float>(...);

// Wait for first K tile
consumer_wait(pipeline_k, smem_pipe_read_k);
copy_k_block(_0{});  // Copy first k_block of K to registers

// Initialize S with delta_s correction
add_delta_s(tSrS);

// GEMM-I: S = Q × K + delta_s (accumulated over k_blocks)
for (int k_block = 0; k_block < size<2>(tSrQ); ++k_block) {
    cute::gemm(tiled_mma_qk,
        make_zip_tensor(tSrQ(_, _, k_block), tSrSFQ(_, _, k_block)),
        make_zip_tensor(tSrK(_, _, k_block), tSrSFK(_, _, k_block)),
        tSrS);
    if (k_block < size<2>(tSrQ) - 1) {
        copy_k_block(k_block + 1);  // Prefetch next k_block
    } else {
        pipeline_k.consumer_release(smem_pipe_read_k);  // Release K smem
        ++smem_pipe_read_k;
    }
}
```

**Number of k_block iterations**: For kHeadDim=128 and atom K=64: `size<2>(tSrQ) = 128/64 = 2` iterations.

**Interleaving pattern**: While the MMA computes k_block 0, we prefetch k_block 1. On the last iteration, we release the pipeline stage.

### Stage C: Masking (lines 731-749)

```cpp
auto col_limit_causal = [&](int row, int n_block) {
    return row + 1 + seqlen_k - n_block * kBlockN - seqlen_q + m_block * kBlockM;
};

{
    Tensor cS = cute::make_identity_tensor(select<0,1>(TileShape_MNK{}));
    Tensor tScS = thread_mma_qk.partition_C(cS);  // Map MMA fragment → (row, col)

    for (int i = 0; i < size(tSrS); ++i) {
        if constexpr (!Is_causal) {
            // Only mask padding: K beyond unpadded_seqlen_k
            if (int(get<1>(tScS(i))) >= int(unpadded_seqlen_k - n_block * kBlockN))
                tSrS(i) = -INFINITY;
        } else {
            // Causal mask: col >= min(seqlen_k, col_limit_causal(row))
            if (int(get<1>(tScS(i))) >= std::min(seqlen_k - n_block * kBlockN,
                                                  col_limit_causal(int(get<0>(tScS(i))), n_block)))
                tSrS(i) = -INFINITY;
        }
    }
}
```

**The identity tensor trick**: `make_identity_tensor(Shape<128,128>)` creates a tensor where `tensor(i,j) = (i, j)`. By partitioning this with the same MMA, we get the (row, col) coordinate that each register element corresponds to. This lets us apply position-dependent masking.

### Stage D: Fused Softmax + FP4 Quantization (lines 750-799)

```cpp
// The quantize lambda — converts softmax output to FP4 + E4M3 scale factors
auto quantize = [&](auto mma_k, auto acc_conversion_view) {
    Tensor AbsMaxP_stagek = AbsMaxP(_, make_coord(_, _, mma_k));
    Tensor acc_conversion_stagek = acc_conversion_view(_, _, mma_k);
    Tensor SFP = make_tensor_like<float_ue4m3_t>(AbsMaxP_stagek.layout());
    Tensor SFP_uint32_view = recast<uint32_t>(SFP);

    // Step 1: Convert 4 float AbsMax values to 4 E4M3 scale factors (packed in uint32)
    for (int i = 0; i < size(AbsMaxP_stagek); i += 4) {
        packed_float_to_ue4m3(AbsMaxP_stagek(i..i+3), SFP_uint32_view(i/4));
    }

    // Step 2: Convert 8 float P values to 8 FP4 values (packed in uint32)
    int const quad_id = threadIdx.x & 3;
    uint32_t MASK = (0xFF00FF) << ((quad_id & 1) * 8);

    for (int mma_m = 0; mma_m < size<1>(tOrP); ++mma_m) {
        for (int i = 0; i < 4; ++i) {
            packed_float_to_e2m1(
                acc_conversion_stagek(0..7, i, mma_m),  // 8 float values
                tOrP_uint32_view(i, mma_m)               // → 1 uint32 of 8 FP4 values
            );
        }

        // Step 3: Exchange scale factors between thread pairs via warp shuffle
        uint32_t local_sfp = SFP_uint32_view(0, 0, mma_m);
        uint32_t peer_sfp  = __shfl_xor_sync(-1, local_sfp, 2);  // Thread pair exchange

        if ((quad_id & 1) == 0) {
            tOrSFP_uint32_view(0, mma_m) = (local_sfp & MASK) | ((peer_sfp & MASK) << 8);
        } else {
            tOrSFP_uint32_view(0, mma_m) = (peer_sfp & MASK) | ((local_sfp & MASK) >> 8);
        }
    }
};

// Online softmax with quantization (first tile)
softmax_fused.online_softmax_with_quant</*FirstTile=*/true>(
    tSrS, AbsMaxP, mainloop_params.softmax_scale_log2);
```

**The shfl_xor trick for scale factors**: Each thread computes AbsMax for 8 consecutive elements (half of the 16-element SF block). Threads at `quad_id & 1 == 0` hold the first 8 elements; threads at `quad_id & 1 == 1` hold the next 8. `shfl_xor(_, 2)` exchanges between these thread pairs, allowing each thread to construct the full 16-element block's scale factor.

The MASK and shift operations interleave the E4M3 scale factor bytes from two threads into the layout expected by the MMA instruction.

### Stage E: First P×V GEMM (lines 801-815)

```cpp
// Wait for V to arrive in smem
consumer_wait(pipeline_v, smem_pipe_read_v);
copy_v_block(_0{});     // Copy first v_block to registers
quantize(_0{}, tSrS_converion_view);  // Quantize first block of P

for (int v_block = 0; v_block < size<2>(tOrP); ++v_block) {
    cute::gemm(tiled_mma_pv,
        make_zip_tensor(tOrP(_, _, v_block), tOrSFP(_, _, v_block)),      // P (FP4) + SFP
        make_zip_tensor(tOrVt(_, _, v_block), tOrSFVt(_, _, v_block)),    // V (FP4) + SFV
        tOrO_store);
    if (v_block < size<2>(tOrP) - 1) {
        copy_v_block(v_block + 1);
        quantize(v_block + 1, tSrS_converion_view);
    } else {
        pipeline_v.consumer_release(smem_pipe_read_v);
        ++smem_pipe_read_v;
    }
}
```

Same interleaving pattern as GEMM-I: while computing one v_block, prefetch/quantize the next.

### Stage F: Remaining Tiles — Causal Masking Loop (lines 817-862)

```cpp
n_block--;
constexpr int n_masking_steps = !Is_causal ? 1 : ceil_div(kBlockM, kBlockN) + 1;

// Causal masking loop (unrolled — these tiles need per-element masking)
for (int masking_step = 0; masking_step < n_masking_steps - 1 && n_block >= 0;
     ++masking_step, --n_block) {

    // Same pattern: GEMM-I → mask → softmax → quantize → GEMM-II → rescale
    Tensor tSrS = partition_fragment_C(...);
    consumer_wait(pipeline_k, smem_pipe_read_k);
    copy_k_block(_0{});
    add_delta_s(tSrS);

    for (k_block) { gemm(QK); copy_k_next; }
    pipeline_k.consumer_release();

    // Apply causal mask
    for (i) { if (col >= col_limit_causal(row)) tSrS(i) = -INFINITY; }

    // Online softmax (not first tile)
    softmax_fused.online_softmax_with_quant<false>(tSrS, AbsMaxP, ...);

    // GEMM-II
    Tensor tOrO = make_fragment_like(tOrO_store);  // New O accumulator
    consumer_wait(pipeline_v, smem_pipe_read_v);
    for (v_block) { gemm(PV); }

    // Rescale: tOrO_store = tOrO_store * scale + tOrO
    if (masking_step > 0) { softmax_fused.rescale_o(tOrO_store, tOrO); }
}
```

**Why `n_masking_steps = ceil_div(kBlockM, kBlockN) + 1`?** For causal attention, only the first few n_blocks (near the diagonal) have partial masking. With kBlockM=kBlockN=128, at most 2 tiles need masking. The `+1` accounts for the first tile processed before this loop.

### Stage G: Full Tiles Loop (lines 864-900)

```cpp
#pragma unroll 1  // Don't unroll — unknown iteration count
for (; n_block >= 0; --n_block) {
    // Same as causal loop but WITHOUT masking
    Tensor tSrS = partition_fragment_C(...);
    consumer_wait(pipeline_k, smem_pipe_read_k);
    copy_k_block(_0{});
    add_delta_s(tSrS);
    for (k_block) { gemm(QK); }

    // No masking needed — these tiles are fully valid
    softmax_fused.online_softmax_with_quant<false>(tSrS, AbsMaxP, ...);

    Tensor tOrO = make_fragment_like(tOrO_store);
    consumer_wait(pipeline_v, smem_pipe_read_v);
    for (v_block) { gemm(PV); }

    softmax_fused.rescale_o(tOrO_store, tOrO);
}
```

### Stage H: Finalize (line 901)

```cpp
softmax_fused.finalize(tOrO_store);
```

This divides each row of O by its row_sum:
```cpp
// In softmax_fused.h:
void finalize(TensorAcc& o_store) {
    // Reduce row_sum across 4 threads in a quad
    for (i = 1; i < 4; i <<= 1) {
        float sum_recv = __shfl_xor_sync(-1, row_sum(mi), i);
        row_sum(mi) += sum_recv;
    }
    float inv_sum = (sum == 0 || sum != sum) ? 0 : 1 / sum;
    for (ni) o_store(mi, ni) *= inv_sum;
}
```

**NaN check**: `sum != sum` catches NaN from all-masked rows.

---

## 4.5 `softmax_fused.h` — Fused Online Softmax + Quantization (173 lines)

### Constants

```cpp
static constexpr float fp8_scalexfp4_scale = 1.f / (448 * 6);
static constexpr float fp8_scalexfp4_scale_log2 = -11.392317422778762f;
static constexpr float fp4_scale_log2 = -2.584962500721156f;
static constexpr int RowReductionThr = 4;
```

- `448` = max representable value in E4M3
- `6` = max representable value in FP4 E2M1
- `fp8_scalexfp4_scale = 1/(448×6)` — normalizes so softmax output × scale lands in E4M3 range
- `fp4_scale_log2 = log2(1/6)` — for converting AbsMaxP to E4M3 SF values

### `online_softmax_with_quant<FirstTile>()` (lines 39-137)

**For FirstTile=true** (lines 49-88):

```cpp
fill(row_max, -INFINITY);
clear(row_sum);
fill(scores_scale, 1.f);

for (mi = 0; mi < num_rows; mi++) {
    // 1. Compute per-16-element-block max (for microscaling)
    for (ni = 0; ni < num_blocks; ni++) {
        for (ei = 0; ei < 8; ei++) {  // 8 elements per thread
            AbsMaxP(mi, ni) = fmax(AbsMaxP(mi, ni), acc(mi, (ei, ni)));
        }
        // Exchange with neighbor thread to complete 16-element block max
        float max_recv = __shfl_xor_sync(-1, AbsMaxP(mi, ni), 1);
        AbsMaxP(mi, ni) = fmax(AbsMaxP(mi, ni), max_recv);
        row_max(mi) = fmax(row_max(mi), AbsMaxP(mi, ni));
    }

    // 2. Complete row max across quad
    float max_recv = __shfl_xor_sync(-1, row_max(mi), 2);
    row_max(mi) = fmax(row_max(mi), max_recv);

    // 3. Apply softmax: exp2(S * scale - max_scaled)
    const float max_scaled = row_max(mi) * softmax_scale_log2 + fp8_scalexfp4_scale_log2;
    for (ni = 0; ni < total_elements; ni++) {
        acc(mi, ni) = ptx_exp2(acc(mi, ni) * softmax_scale_log2 - max_scaled);
    }

    // 4. Compute AbsMaxP scale factors: exp2(AbsMaxP * scale - max_scaled + fp4_scale)
    for (sfi = 0; sfi < num_sf_blocks; sfi++) {
        AbsMaxP(mi, sfi) = ptx_exp2(AbsMaxP(mi, sfi) * softmax_scale_log2
                                     - max_scaled + fp4_scale_log2);
    }
}

// 5. Compute row sums
for (mi) for (ni) row_sum(mi) += acc(mi, ni);
```

**For FirstTile=false** (lines 90-130):

Same structure but with rescaling:
```cpp
scores_max_prev = copy(row_max);  // Save previous max

// Steps 1-2: Same max computation
// ...

// 3. Rescale factor
scores_scale(mi) = ptx_exp2((scores_max_prev(mi) - row_max(mi)) * softmax_scale_log2);
row_sum(mi) = row_sum(mi) * scores_scale(mi);

// Steps 4-5: Same softmax + sum, but add to existing sum
for (ni) {
    acc(mi, ni) = ptx_exp2(acc(mi, ni) * softmax_scale_log2 - max_scaled);
    row_sum(mi) += acc(mi, ni);
}
```

**After softmax** (lines 131-136):

```cpp
// Normalize P by AbsMaxP for FP4 conversion
for (i = 0; i < size(AbsMaxP); ++i) {
    for (j = 0; j < num_elements_per_block; ++j) {
        acc_conversion_flatten(j, i) /= AbsMaxP(i);
    }
}
```

This divides each element by its block's AbsMaxP, putting values in [-1, 1] range for FP4 conversion.

### `rescale_o()` (lines 158-170)

```cpp
void rescale_o(TensorAcc& o_store, TensorAcc const& o_tmp) {
    for (mi = 0; mi < num_rows; mi++) {
        for (ni = 0; ni < num_cols; ni++) {
            o_store(mi, ni) = o_store(mi, ni) * scores_scale(mi) + o_tmp(mi, ni);
        }
    }
}
```

This implements the online softmax correction: when the max changes, previous O values must be rescaled by `exp2((old_max - new_max) * scale)`.

---

## 4.6 `cute_extension.h` — Custom SM120 MMA Atom (326 lines)

### The MMA Instruction

```cpp
struct SM120_16x32x64_TN_VS_NVFP4 {
    // Issues 4 separate mma.sync instructions
    // Each: m16n8k64 with block-scaled NVFP4
    // Together: m16n32k64 (32 = 4 × 8)

    void fma(...) {
        // Instruction 1: columns 0-7 of N
        asm("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
            "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 ..., tidB0");
        // Instruction 2: columns 8-15
        asm("... tidB1");
        // Instruction 3: columns 16-23
        asm("... tidB2");
        // Instruction 4: columns 24-31
        asm("... tidB3");
    }
};
```

The `tidB0..tidB3` selects which 8-column chunk of B to use. Each instruction produces 4 output values (d0,d1,d8,d9 etc.), and after all 4 instructions, each thread has 16 output values covering a 16×32 tile.

### Thread-Value Layouts (MMA_Traits)

The layouts encode how the 32 threads of a warp distribute the matrix elements:

```
ALayout (M=16, K=64):
  32 threads × 32 values = 1024 = 16×64
  Thread (t): rows = {t%4 * 4 + ...}, cols vary within thread

BLayout (N=32, K=64):
  32 threads × 64 values = 2048 = 32×64
  Each thread holds 64 values spanning the full K dimension

CLayout (M=16, N=32):
  32 threads × 16 values = 512 = 16×32
  Thread (t): holds 2×2 blocks at various (row, col) positions
```

### Scale Factor Partitioning Functions

```cpp
// thrfrg_SFA: Partition an SF tensor for A-operand (Q) scale factors
auto thrfrg_SFA(SFATensor, TiledMMA) {
    // 1. Reorder for TiledAtom permutation
    // 2. Divide into atom-sized tiles
    // 3. Map from (M,K) to (Thread, Value) using SFALayout
    // 4. Divide by thread tiling
}

// partition_fragment_SFA: Create a register fragment for A scale factors
auto partition_fragment_SFA(SFATensor, thread_mma) {
    return make_fragment_like<ValTypeSF>(partition_SFA(...));
}
```

These functions handle the complex mapping between scale factor memory layout and the MMA instruction's expected SF layout.

---

## 4.7 `utils.h` — PTX Building Blocks (408 lines)

### Reduction Operations

```cpp
// Warp-level allreduce (for 4, 8, 16, or 32 threads)
template<int THREADS>
struct Allreduce {
    template<typename T, typename Operator>
    static T run(T x, Operator &op) {
        x = op(x, __shfl_xor_sync(-1, x, THREADS/2));
        return Allreduce<THREADS/2>::run(x, op);
    }
};
```

### PTX Intrinsics

```cpp
// Fast exp2 approximation
float ptx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

// Pack 4 floats → 4 E4M3 values in one uint32
void packed_float_to_ue4m3(float f0, f1, f2, f3, uint32_t &out) {
    asm("cvt.rn.satfinite.e4m3x2.f32 lo, %2, %1;"
        "cvt.rn.satfinite.e4m3x2.f32 hi, %4, %3;"
        "mov.b32 %0, {lo, hi};");
}

// Pack 8 floats → 8 E2M1 (FP4) values in one uint32
void packed_float_to_e2m1(float f0..f7, uint32_t &out) {
    asm("cvt.rn.satfinite.e2m1x2.f32 byte0, %2, %1;"
        "cvt.rn.satfinite.e2m1x2.f32 byte1, %4, %3;"
        "cvt.rn.satfinite.e2m1x2.f32 byte2, %6, %5;"
        "cvt.rn.satfinite.e2m1x2.f32 byte3, %8, %7;"
        "mov.b32 %0, {byte0, byte1, byte2, byte3};");
}
```

### Layout Transform Functions

```cpp
// convert_to_reduction_layout: Reorder MMA fragment for row-wise operations
// Input layout: (MmaAtom, MmaM, MmaN) where MmaAtom = (AtomN, AtomM)
// Output layout: ((AtomM, MmaM), (AtomN, MmaN))
// This groups all elements in the same row together for efficient reduction

// convert_to_conversion_layout: Reorder for FP4 packing
// Groups elements into 8-element blocks aligned with the E2M1 conversion
// Input: (MmaAtom, MmaM, MmaN)
// Output: ((AtomN, (AtomM, MmaN_halved)), MmaM, MmaN_rest)
```

---

## 4.8 `blockscaled_layout.h` — Scale Factor Memory Layout (149 lines)

### The Fundamental Unit: SfAtom

```cpp
// SfAtom: how one block of scale factors is laid out
using SfAtom = Layout<
    Shape<Shape<_16, _4>, Shape<Int<SFVectorSize>, _4>>,
    Stride<Stride<_16, _4>, Stride<_0, _1>>
>;
```

This represents a **64-row × 4-SF chunk**:
- First mode: 64 rows (16 sub-blocks × 4 rows each)
- Second mode: `SFVectorSize×4` scale factor entries, but stride `_0` means they're **broadcast** — each of the 4 rows in a sub-block shares the same SF

This layout is dictated by the TMEM (Tensor Memory) word size on Blackwell — consecutive 32-bit words must have scale factors for only a single row to enable efficient hardware dequantization.

### Layout Deduction Functions

```cpp
// For Q/K scale factors: (M_or_N, K) → smem layout
static auto deduce_smem_layoutSFQ(TiledMma, TileShape_MNK) {
    // Shapes based on:
    //   SFVectorSize = 16 (elements per SF)
    //   MMA_NSF = 4 (SF blocks per MMA atom)
    //   Blk_MN = 64 (rows per chunk)
    //   Blk_SF = 4 (SFs per chunk)
}
```

These functions compute the shared memory layout for scale factors based on the tile shape and MMA configuration, ensuring alignment with the SfAtom and hardware requirements.

---

## Phase 4 Summary

After working through this phase, you should be able to:

1. **Trace a single attention tile** through the complete pipeline:
   ```
   TMA load → smem → LDSM → registers → GEMM-I → softmax+quantize → GEMM-II → rescale → finalize → smem → TMA store
   ```

2. **Explain every register fragment**: What shape, what data type, what it holds

3. **Understand the pipelining**: How TMA loads overlap with MMA compute via 3-stage pipeline barriers

4. **Explain the quantization**: How FP32 softmax output becomes FP4 data + E4M3 scale factors, including the warp shuffle trick

5. **Understand the loop structure**: First tile (no rescaling) → causal masking tiles → full tiles → finalize
