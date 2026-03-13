# Phase 2: FMHA in CuTeDSL — Understanding Flash Attention Before Reading C++

## Goal

Understand the **Flash Attention algorithm** and how it maps to CuTe abstractions by studying CuTeDSL reference implementations. This bridges Phase 1 (CuTe concepts) and Phase 3 (SA3 C++ code).

---

## 2.1 The Flash Attention Algorithm (Refresher)

### The Problem

Standard attention: `O = softmax(QK^T / sqrt(d)) × V`

This requires materializing the full N×N attention matrix — O(N²) memory, prohibitive for long sequences.

### Flash Attention's Solution: Online Softmax + Tiling

Process attention **one tile of K,V at a time**, maintaining running statistics (max and sum) to compute the correct softmax without ever materializing the full attention matrix.

### The Online Softmax Algorithm

For each row i of the output:

```python
# Initialize
m_i = -inf          # running max
l_i = 0             # running sum
O_i = 0             # running output accumulator

for j in range(0, N, BLOCK_N):      # iterate over K,V blocks
    # GEMM-I: Compute attention scores for this block
    S_ij = Q_i @ K_j^T * scale       # shape: (BLOCK_M, BLOCK_N)

    # Update max
    m_new = max(m_i, rowmax(S_ij))

    # Correction factor for previous blocks
    alpha = exp(m_i - m_new)

    # Softmax of current block
    P_ij = exp(S_ij - m_new)

    # Update sum
    l_new = alpha * l_i + rowsum(P_ij)

    # Rescale previous output and add new contribution
    O_i = alpha * O_i + P_ij @ V_j    # GEMM-II

    m_i = m_new
    l_i = l_new

# Final normalization
O_i = O_i / l_i
```

### Key Insight for SA3

SageAttention3 adds a twist: **P is quantized to FP4 before the PV GEMM**. The softmax and quantization are **fused** — both happen in the same pass over the S matrix, reusing the row-max computation for both softmax scaling and FP4 microscaling.

---

## 2.2 Study Order for CuTeDSL Examples

### Step 1: Ampere Flash Attention v2 (Simpler, Non-Blackwell)

**File:** `csrc/cutlass/examples/python/CuTeDSL/ampere/flash_attention_v2.py`

This is a **pure Python** implementation of FA2 using CuTeDSL for Ampere GPUs. It's simpler because:
- No warp specialization
- No TMA (uses cp.async instead)
- FP16/BF16 only (no block-scaled quantization)
- Still shows the full online softmax algorithm

**What to look for:**
1. How Q, K, V tensors are created and tiled
2. The main loop structure: load K → GEMM-I → softmax → load V → GEMM-II → rescale
3. How online softmax statistics (m, l) are maintained
4. How output O is rescaled when max changes

### Step 2: Blackwell FMHA (Full Complexity)

**File:** `csrc/cutlass/examples/python/CuTeDSL/blackwell/fmha.py` (3246 lines)

This is the closest CuTeDSL reference to SA3. It includes:
- Warp specialization (producer/consumer warps)
- TMA loads for Q, K, V
- Persistent tile scheduling
- Causal masking
- Online softmax with rescaling

**What to look for:**
1. **Warp group roles** — producer warps (TMA loads + TMA stores), consumer warps (MMA + softmax)
2. **Pipeline stages** — how TMA loads and MMA consumption overlap
3. **Register allocation** — producer uses fewer registers (warpgroup_reg_dealloc), consumers use more
4. **Epilogue** — how the final output moves from registers → smem → gmem

### Step 3: Mixed-Input FMHA

**File:** `csrc/cutlass/examples/python/CuTeDSL/blackwell/mixed_input_fmha/mixed_input_fmha_prefill_d256.py`

Shows FP8 → BF16 conversion in attention, which is related to SA3's FP4 quantization approach.

---

## 2.3 Mapping CuTeDSL FMHA to SA3 C++ Code

This is the critical bridge. Here's how each section of the FMHA algorithm maps:

### High-Level Structure

| Algorithm Step | CuTeDSL FMHA (`fmha.py`) | SA3 C++ Code |
|---|---|---|
| **Kernel launch** | Grid/block dims, smem size | `launch.h` (lines 33-101) |
| **Warp specialization** | Producer/consumer role assignment | `kernel_ws.h` (lines 70-92) |
| **Register redistribution** | `warpgroup_reg_dealloc/alloc` | `kernel_ws.h` (lines 141, 169) |
| **TMA descriptor creation** | `make_tma_copy(...)` | `mainloop_tma_ws.h::to_underlying_arguments()` (lines 200-265) |
| **TMA load loop** | Producer warp loads Q,K,V via TMA | `mainloop_tma_ws.h::load()` (lines 432-554) |
| **GEMM-I (QK)** | `gemm(mma, Q_frag, K_frag, S)` | `mainloop_tma_ws.h::mma()` (lines 716-729) |
| **Causal masking** | `S[mask] = -inf` | `mainloop_tma_ws.h::mma()` (lines 732-749) |
| **Softmax** | `online_softmax(S, m, l)` | `softmax_fused.h::online_softmax_with_quant()` |
| **P quantization** | *(not in standard FMHA)* | `mainloop_tma_ws.h::quantize()` (lines 750-797) |
| **GEMM-II (PV)** | `gemm(mma, P_frag, V_frag, O)` | `mainloop_tma_ws.h::mma()` (lines 801-815) |
| **O rescaling** | `O = O * scale + O_new` | `softmax_fused.h::rescale_o()` |
| **Output store** | smem → gmem via TMA | `epilogue_tma_ws.h` |
| **Tile scheduler** | Persistent: grid_dim = num_SMs | `tile_scheduler.h::StaticPersistentTileScheduler` |

### Detailed Mapping: The Main Loop

**CuTeDSL FMHA pattern** (conceptual):
```python
# Consumer warp
for tile_info in scheduler:
    m_block, bidh, bidb = tile_info.get_block_coord()
    n_block_max = compute_n_block_max(m_block)

    # Initialize accumulators
    O = zeros(kBlockM, kHeadDim)
    m = fill(-inf, kBlockM)
    l = zeros(kBlockM)

    for n_block in range(n_block_max-1, -1, -1):
        # Wait for K,V to arrive in smem
        pipeline_k.consumer_wait(stage)

        # GEMM-I: S = Q × K^T
        S = zeros(kBlockM, kBlockN)
        for k_block in range(kHeadDim // atom_k):
            copy_k_from_smem_to_reg(k_block)
            gemm(Q_reg, K_reg, S)
        pipeline_k.consumer_release(stage)

        # Masking (causal)
        apply_causal_mask(S, m_block, n_block)

        # Online softmax
        m_new = max(m, rowmax(S))
        scale = exp2((m - m_new) * softmax_scale)
        P = exp2(S * softmax_scale - m_new * softmax_scale)
        l = l * scale + rowsum(P)

        # GEMM-II: O_new = P × V
        pipeline_v.consumer_wait(stage)
        O_new = zeros(kBlockM, kHeadDim)
        for v_block in range(kBlockN // atom_k):
            copy_v_from_smem_to_reg(v_block)
            gemm(P_reg, V_reg, O_new)
        pipeline_v.consumer_release(stage)

        # Rescale previous O and add new
        O = O * scale + O_new
        m = m_new

    # Finalize
    O = O / l

    # Store O to smem, then TMA to gmem
    store_to_smem(O)
    barrier_o.arrive()  # signal producer for TMA store
```

**SA3 C++ differences from standard FMHA:**

1. **FP4 quantization after softmax**: SA3 inserts a `quantize()` step between softmax and GEMM-II
2. **Block-scaled MMA**: Both GEMMs use `make_zip_tensor(data, scale_factors)`
3. **Delta-S correction**: `add_delta_s(tSrS)` adds smooth-Q correction before GEMM-I
4. **Reversed tile order**: SA3 iterates from `n_block_max-1` down to 0 (matches FlashAttention convention)
5. **Two-level P scaling**: The softmax computes AbsMaxP (per-16-element-block max) simultaneously with the row max

---

## 2.4 The SA3-Specific Innovation: Fused Softmax + FP4 Quantization

This is what makes SA3 different from standard FMHA. In `softmax_fused.h`:

### Standard Online Softmax

```python
# For each row:
max_new = max(row_max, max(S_row))
P_row = exp2(S_row * scale - max_new * scale)
row_sum += sum(P_row)
```

### SA3's Fused Version

```python
# For each row, simultaneously:
# 1. Find row max (for softmax)
# 2. Find per-16-element-block AbsMax (for FP4 microscaling)
for ni in range(num_blocks_of_16):
    block_max = max(S_row[ni*16 : (ni+1)*16])  # 8 elements per thread

    # Exchange with neighbor thread to get 16-element max
    peer_max = shfl_xor(block_max, 1)
    AbsMaxP[mi, ni] = max(block_max, peer_max)

    # Update row max
    row_max[mi] = max(row_max[mi], AbsMaxP[mi, ni])

# Exchange row max with quad-mate
peer_row_max = shfl_xor(row_max[mi], 2)
row_max[mi] = max(row_max[mi], peer_row_max)

# Now compute softmax and scale factors simultaneously
max_scaled = row_max * softmax_scale_log2 + fp8xfp4_scale_log2

for ni in range(total_elements):
    S[mi, ni] = exp2(S[mi, ni] * softmax_scale_log2 - max_scaled)

for sfi in range(num_sf_blocks):
    AbsMaxP[mi, sfi] = exp2(AbsMaxP[mi, sfi] * softmax_scale_log2 - max_scaled + fp4_scale_log2)

# Then normalize and quantize
for i in range(total):
    S_normalized[i] = S[i] / AbsMaxP[block_of(i)]  # Normalize to [-1, 1] for FP4
```

### Why This Works

1. **AbsMaxP is the FP4 microscale**: Each 16-element block of P has its own E4M3 scale factor
2. **The max is already computed**: Softmax requires row-max anyway, so per-block-max is nearly free
3. **The exp2 is fused**: Both softmax values and scale factors go through the same `exp2` computation
4. **Two-level scaling**: The constant `fp8_scalexfp4_scale_log2 = log2(1 / (448 × 6))` absorbs both the FP8 range (448) and FP4 range (6) into the exponent computation

### The Quantize Lambda in mainloop_tma_ws.h

After softmax produces P in FP32, the `quantize` lambda converts it to FP4 + E4M3 scales:

```cpp
auto quantize = [&](auto mma_k, auto acc_conversion_view) {
    // 1. Convert AbsMaxP to E4M3 scale factors
    for (i = 0; i < size(AbsMaxP); i += 4)
        packed_float_to_ue4m3(AbsMaxP[i..i+3], SFP[i/4]);

    // 2. Convert P values to E2M1 (FP4)
    for (mma_m in range(...))
        for (i in range(4))
            packed_float_to_e2m1(P[0..7], tOrP[i, mma_m]);  // 8 floats → 32 bits

    // 3. Shuffle scale factors between thread pairs
    // Threads come in pairs (quad_id & 1). Each thread computes SF for 8 elements,
    // but the MMA needs SF for 16 elements. shfl_xor(_, 2) exchanges between pairs.
    local_sfp = SFP[mma_m];
    peer_sfp = __shfl_xor_sync(-1, local_sfp, 2);
    // Interleave local and peer SFs
    tOrSFP[mma_m] = (local_sfp & MASK) | (peer_sfp & MASK);
};
```

---

## 2.5 CuTeDSL Study Exercises

### Exercise 1: Run the Ampere FA2

```bash
cd sageattention3_blackwell/csrc/cutlass/examples/python/CuTeDSL/ampere
python flash_attention_v2.py
```

Study the output. Then answer:
1. What are the tile sizes used?
2. How many pipeline stages?
3. Where is the online softmax computed?

### Exercise 2: Study the NVFP4 GEMM

```bash
cd sageattention3_blackwell/csrc/cutlass/examples/python/CuTeDSL/blackwell/tutorial_gemm
python nvfp4_gemm_0.py
```

This is directly relevant to SA3 because:
- It uses the same NVFP4 MMA instruction
- It has block-scaled data + scale factors
- It demonstrates `make_zip_tensor(data, sf)`

### Exercise 3: Study the Blackwell FMHA

```bash
cd sageattention3_blackwell/csrc/cutlass/examples/python/CuTeDSL/blackwell
python fmha.py  # May need specific GPU
```

Map each section to the SA3 files as described in section 2.3 above.

### Exercise 4: Compare CuTeDSL FMHA with SA3 C++

Open these side by side:
- `csrc/cutlass/examples/python/CuTeDSL/blackwell/fmha.py`
- `sageattention3_blackwell/sageattn3/blackwell/mainloop_tma_ws.h`

For each of these elements, find the corresponding code in both files:
1. TMA descriptor creation
2. Pipeline initialization
3. The QK GEMM loop
4. Online softmax
5. The PV GEMM loop
6. Output rescaling
7. Output store

---

## 2.6 The Complete FMHA Data Flow in SA3

```
Input: Q[B,H,M,D], K[B,H,N,D], V[B,H,D,N], SFQ, SFK, SFV, DeltaS
       All pre-quantized to FP4+E4M3 on the Python side

┌─────── Per tile (m_block) ───────┐
│                                   │
│  Q loaded once from gmem → smem   │  ← pipeline_q (1 stage)
│  Q + SFQ copied to registers      │
│                                   │
│  ┌── For n = n_max-1 downto 0 ──┐│
│  │                                ││
│  │  K[n] + SFK[n] + DS[n]        ││  ← pipeline_k (3 stages)
│  │    loaded from gmem → smem     ││
│  │                                ││
│  │  S = DeltaS                    ││  ← add_delta_s (smooth-Q correction)
│  │  for k_block:                  ││
│  │    copy K[k_block] smem→reg    ││
│  │    S += Q[k_block] × K[k_block]││  ← GEMM-I (block-scaled NVFP4 MMA)
│  │                                ││
│  │  if causal: mask S             ││
│  │                                ││
│  │  online_softmax_with_quant(S)  ││  ← fused softmax + FP4 quantize
│  │    → P (FP4 in registers)      ││
│  │    → SFP (E4M3 scale factors)  ││
│  │    → update m, l, scale        ││
│  │                                ││
│  │  V[n] + SFV[n]                 ││  ← pipeline_v (3 stages)
│  │    loaded from gmem → smem     ││
│  │                                ││
│  │  O_new = 0                     ││
│  │  for v_block:                  ││
│  │    copy V[v_block] smem→reg    ││
│  │    O_new += P[v_block]×V[v_block]│  ← GEMM-II (block-scaled NVFP4 MMA)
│  │                                ││
│  │  O = O * scale + O_new         ││  ← rescale_o
│  │                                ││
│  └────────────────────────────────┘│
│                                   │
│  O = O / row_sum                  │  ← finalize
│  O (FP32 regs) → BF16 → smem     │  ← epilogue.mma_store
│  smem → gmem via TMA              │  ← epilogue.tma_store
│                                   │
└───────────────────────────────────┘
```

---

## Summary of Phase 2 Key Takeaways

1. **Online softmax** = process K,V tiles one at a time, rescaling previous results when max changes
2. **SA3's innovation** = fuse softmax with FP4 microscaling quantization, reusing the max computation
3. **Two-level P scaling** = row-level scale (from softmax) × block-level scale (FP4 microscale)
4. **The quantize step** is unique to SA3 — standard FMHA keeps P in FP16/FP32
5. **CuTeDSL FMHA** shares identical structure with SA3 C++ — study it first for faster C++ comprehension

## What to Read Next

- Phase 3: SA3 C++ Architecture (kernel_ws.h, kernel_traits.h)
- Phase 4: SA3 Mainloop Line-by-Line (mainloop_tma_ws.h)
