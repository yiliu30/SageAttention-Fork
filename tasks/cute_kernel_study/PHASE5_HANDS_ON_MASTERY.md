# Phase 5: Hands-on Mastery — Exercises and Verification

## 5.1 CuTeDSL Exercises

### Exercise 1: Layout Exploration Script

Create a Python script that prints CuTe layouts interactively:

```python
#!/usr/bin/env python3
"""CuTe Layout Explorer - Run on any GPU with CuTeDSL installed."""

# Try to import CuTeDSL
try:
    import cutlass
    import cutlass.cute as cute
    CUTEDSL_AVAILABLE = True
except ImportError:
    CUTEDSL_AVAILABLE = False
    print("CuTeDSL not available. Install from csrc/cutlass/python/")

def explore_layout_basics():
    """Exercise 1.1: Create and print basic layouts."""
    # Row-major 4x8
    shape = (4, 8)
    stride = (8, 1)
    print(f"Row-major {shape}: stride {stride}")
    for i in range(4):
        offsets = [i * stride[0] + j * stride[1] for j in range(8)]
        print(f"  row {i}: offsets = {offsets}")

    # Column-major 4x8
    stride_cm = (1, 4)
    print(f"\nColumn-major {shape}: stride {stride_cm}")
    for i in range(4):
        offsets = [i * stride_cm[0] + j * stride_cm[1] for j in range(8)]
        print(f"  row {i}: offsets = {offsets}")

    # Swizzled example: what SA3 actually uses
    # Layout<Shape<_128, _128>, Stride<_128, _1>> but swizzled
    print("\nFor SA3: Q smem is 128x128 with swizzling for bank-conflict-free access")

def explore_hierarchical_layout():
    """Exercise 1.2: Nested shapes."""
    # Shape<Shape<_2, _4>, _8> = 2x4x8 = 64 elements
    # This represents: 2 groups of 4, repeated 8 times
    print("Hierarchical shape: ((2, 4), 8)")
    print("  Total elements: 2 * 4 * 8 = 64")
    print("  If stride is ((1, 2), 8): offset(((i,j), k)) = i*1 + j*2 + k*8")
    for k in range(2):  # Just show first 2 of 8
        for j in range(4):
            for i in range(2):
                offset = i * 1 + j * 2 + k * 8
                print(f"    ((i={i}, j={j}), k={k}) -> offset {offset}")

def explore_mma_thread_layout():
    """Exercise 1.3: Understand MMA thread-value mapping.

    For SM120_16x32x64_TN_VS_NVFP4:
    CLayout: (T32, V16) -> (M16, N32)
      Shape:  ((4, 8), ((2, 4), 2))
      Stride: ((32, 1), ((16, 128), 8))
    """
    print("\n=== MMA CLayout: How threads map to output elements ===")
    print("Shape:  ((4, 8), ((2, 4), 2))")
    print("Stride: ((32, 1), ((16, 128), 8))")
    print()

    # For each thread, compute which (M, N) elements it owns
    for thread_id in range(32):
        t0 = thread_id % 4   # Inner thread dim
        t1 = thread_id // 4  # Outer thread dim

        elements = []
        for v_outer in range(2):       # V outer dim
            for v_mid in range(4):     # V middle dim
                for v_inner in range(2):  # V inner dim
                    offset = t0 * 32 + t1 * 1 + v_inner * 16 + v_mid * 128 + v_outer * 8
                    # Decode to (M, N)
                    m = offset % 16
                    n = offset // 16
                    if n < 32:  # Valid
                        elements.append((m, n))

        if thread_id < 4:  # Just show first 4 threads
            print(f"  Thread {thread_id:2d}: owns elements at (M, N) = {elements[:8]}...")

def explore_sa3_tile_sizes():
    """Exercise 1.4: SA3 concrete dimensions."""
    kBlockM = 128
    kBlockN = 128
    kHeadDim = 128
    kStages = 3
    SFVectorSize = 16

    print(f"\n=== SA3 Tile Configuration ===")
    print(f"Tile: {kBlockM}×{kBlockN}×{kHeadDim}")
    print(f"MMA atom: 16×32×64 (NVFP4)")
    print(f"Atoms along M: 8 (8×16 = {8*16})")
    print(f"k_block iterations for GEMM-I: {kHeadDim // 64}")
    print(f"v_block iterations for GEMM-II: {kBlockN // 64}")
    print(f"Threads: 12 warps × 32 = 384")
    print(f"  Producer WG: 128 threads (24 regs each)")
    print(f"  Consumer WG0: 128 threads (232 regs each)")
    print(f"  Consumer WG1: 128 threads (232 regs each)")
    print(f"Pipeline stages for K,V: {kStages}")
    print(f"SF blocks per head dim: {kHeadDim // SFVectorSize}")
    print(f"SF blocks per block N: {kBlockN // SFVectorSize}")

    # Memory calculation
    q_bytes = kBlockM * kHeadDim // 2  # FP4 = 0.5 bytes
    k_bytes = kBlockN * kHeadDim // 2 * kStages
    v_bytes = kHeadDim * kBlockN // 2 * kStages
    sfq_bytes = kBlockM * (kHeadDim // 16)  # E4M3 = 1 byte each
    sfk_bytes = kBlockN * (kHeadDim // 16) * kStages
    sfv_bytes = kHeadDim * (kBlockN // 16) * kStages  # This is for V transposed
    ds_bytes = kBlockM * kBlockN * 4 * kStages  # float32
    o_bytes = kBlockM * kHeadDim * 2  # BF16 output

    total = q_bytes + k_bytes + v_bytes + sfq_bytes + sfk_bytes + sfv_bytes + ds_bytes + o_bytes
    print(f"\nSmem usage estimate:")
    print(f"  Q:    {q_bytes:6d} bytes ({q_bytes/1024:.1f} KB)")
    print(f"  K×3:  {k_bytes:6d} bytes ({k_bytes/1024:.1f} KB)")
    print(f"  V×3:  {v_bytes:6d} bytes ({v_bytes/1024:.1f} KB)")
    print(f"  SFQ:  {sfq_bytes:6d} bytes")
    print(f"  SFK×3:{sfk_bytes:6d} bytes")
    print(f"  SFV×3:{sfv_bytes:6d} bytes")
    print(f"  DS×3: {ds_bytes:6d} bytes ({ds_bytes/1024:.1f} KB)")
    print(f"  O:    {o_bytes:6d} bytes ({o_bytes/1024:.1f} KB)")
    print(f"  Total: ~{total/1024:.1f} KB (actual may differ due to padding/alignment)")

if __name__ == "__main__":
    explore_layout_basics()
    explore_hierarchical_layout()
    explore_mma_thread_layout()
    explore_sa3_tile_sizes()
```

**Run it**: `python tasks/cute_kernel_study/layout_explorer.py`

### Exercise 2: Modify CuTeDSL NVFP4 GEMM

```bash
cd sageattention3_blackwell/csrc/cutlass/examples/python/CuTeDSL/blackwell/tutorial_gemm
cp nvfp4_gemm_0.py nvfp4_gemm_modified.py
```

Modifications to try:
1. Change tile sizes (e.g., M=64 instead of 128)
2. Print the layouts at key points
3. Add timing measurement
4. Change the number of pipeline stages

### Exercise 3: Trace the Softmax Data Flow

Write a Python simulation of `softmax_fused.h`:

```python
import numpy as np

def sa3_fused_softmax_quantize(S, softmax_scale):
    """
    Simulate SA3's fused online softmax + FP4 quantization.

    S: attention scores (M, N) for one tile
    softmax_scale: 1/sqrt(d)
    """
    M, N = S.shape
    SF_BLOCK_SIZE = 16
    FP4_MAX = 6.0    # max of E2M1
    FP8_MAX = 448.0   # max of E4M3

    scale_log2 = softmax_scale * np.log2(np.e)
    fp8xfp4_scale_log2 = np.log2(1.0 / (FP8_MAX * FP4_MAX))
    fp4_scale_log2 = np.log2(1.0 / FP4_MAX)

    num_sf_blocks = N // SF_BLOCK_SIZE

    # 1. Compute per-block AbsMax and row max
    AbsMaxP = np.zeros((M, num_sf_blocks))
    row_max = np.full(M, -np.inf)

    for mi in range(M):
        for ni in range(num_sf_blocks):
            block = S[mi, ni*SF_BLOCK_SIZE : (ni+1)*SF_BLOCK_SIZE]
            AbsMaxP[mi, ni] = np.max(block)
            row_max[mi] = max(row_max[mi], AbsMaxP[mi, ni])

    # 2. Apply softmax
    max_scaled = row_max * scale_log2 + fp8xfp4_scale_log2
    P = np.power(2.0, S * scale_log2 - max_scaled[:, None])

    # 3. Compute SF for microscaling
    SFP = np.power(2.0, AbsMaxP * scale_log2 - max_scaled[:, None] + fp4_scale_log2)

    # 4. Normalize P by block SF
    P_normalized = np.zeros_like(P)
    for mi in range(M):
        for ni in range(num_sf_blocks):
            block = P[mi, ni*SF_BLOCK_SIZE : (ni+1)*SF_BLOCK_SIZE]
            P_normalized[mi, ni*SF_BLOCK_SIZE : (ni+1)*SF_BLOCK_SIZE] = block / SFP[mi, ni]

    # 5. Row sum (for final normalization)
    row_sum = np.sum(P, axis=1)

    # P_normalized is in [-1, 1] range, ready for FP4 conversion
    # SFP contains E4M3 scale factors
    return P_normalized, SFP, row_max, row_sum


# Test
np.random.seed(42)
S = np.random.randn(128, 128).astype(np.float32) * 3
softmax_scale = 1.0 / np.sqrt(128)

P_norm, SFP, row_max, row_sum = sa3_fused_softmax_quantize(S, softmax_scale)
print(f"P_normalized range: [{P_norm.min():.4f}, {P_norm.max():.4f}]")
print(f"SFP range: [{SFP.min():.6f}, {SFP.max():.6f}]")
print(f"Row sum range: [{row_sum.min():.4f}, {row_sum.max():.4f}]")

# Compare with standard softmax
import torch
S_torch = torch.from_numpy(S)
P_standard = torch.softmax(S_torch * softmax_scale, dim=-1).numpy()
print(f"\nStandard softmax rowsum: {P_standard.sum(axis=1).mean():.4f} (should be ~1)")
```

---

## 5.2 C++ Exercises

### Exercise 1: Add Debug Prints

Add shape/layout prints to `mainloop_tma_ws.h::mma()` for one thread:

```cpp
// Add after line 609 (after creating thread_mma slices)
if (thread_idx == 0 && m_block == 0) {
    if constexpr (cute::is_static_v<decltype(shape(tSrQ))>) {
        printf("tSrQ shape: static\n");
    }
    printf("tSrQ size: %d, tSrK size: %d, tSrS size: %d\n",
           int(size(tSrQ)), int(size(tSrK)), int(size(tSrS)));
    printf("tOrP size: %d, tOrSFP size: %d\n",
           int(size(tOrP)), int(size(tOrSFP)));
    printf("n_block_count: %d, k_blocks: %d, v_blocks: %d\n",
           n_block_count, int(size<2>(tSrQ)), int(size<2>(tOrP)));
}
```

Rebuild and run to see the actual tensor dimensions at runtime.

### Exercise 2: Change kBlockN from 128 to 64

In `launch.h`, line 110:
```cpp
// Change this:
run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 128, 3, 1, per_block, T, O>, ...>
// To this:
run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 3, 1, per_block, T, O>, ...>
```

**Expected effects:**
- Half the elements per K,V tile → twice as many tiles to process
- GEMM-I output is 128×64 instead of 128×128
- GEMM-II has fewer v_blocks
- Less smem used per stage → could potentially increase stages
- May improve occupancy but reduce arithmetic intensity

**To verify:** Run the test:
```bash
cd sageattention3_blackwell
pip install -e . --no-build-isolation
cd examples
python sageattn3_demo.py  # Check correctness
```

### Exercise 3: Add a New Head Dimension (d=256)

Trace what needs to change:

1. `api.cu` line 193: Add `else if (params.d == 256)` case
2. `launch.h` line 108: Add `Headdim == 256` branch
3. `kernel_traits.h`:
   - `kHeadDim = 256`
   - `NumSFQK = 256/16 = 16` scale factors along K
   - More k_block iterations in GEMM-I: 256/64 = 4
   - Larger smem for Q: 128×256/2 = 16KB
   - May need to reduce kStages or kBlockN to fit in smem
4. `blockscaled_layout.h`: SF layouts should auto-adapt via templates
5. `mainloop_tma_ws.h`: Auto-adapts via TileShape_MNK

**Key concern:** Shared memory capacity. With d=256:
- Q: 128 × 256 / 2 = 16,384 bytes (vs 8,192 for d=128)
- K per stage: 128 × 256 / 2 = 16,384 bytes
- V per stage: 256 × 128 / 2 = 16,384 bytes
- Total for K+V with 3 stages: ~98KB — may exceed smem limits

### Exercise 4: Benchmark Modifications

```python
#!/usr/bin/env python3
"""Benchmark SA3 kernel modifications."""
import torch
import time

def benchmark_attention(seqlen_q, seqlen_k, num_heads, head_dim, num_runs=100, warmup=10):
    """Benchmark the SA3 kernel."""
    try:
        from sageattn3 import sageattn3_blackwell
    except ImportError:
        print("sageattn3 not installed. Run: cd sageattention3_blackwell && pip install -e . --no-build-isolation")
        return

    batch_size = 1
    device = 'cuda'
    dtype = torch.bfloat16

    q = torch.randn(batch_size, num_heads, seqlen_q, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, num_heads, seqlen_k, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, num_heads, seqlen_k, head_dim, device=device, dtype=dtype)

    # Warmup
    for _ in range(warmup):
        out = sageattn3_blackwell(q, k, v)
    torch.cuda.synchronize()

    # Benchmark
    start = time.time()
    for _ in range(num_runs):
        out = sageattn3_blackwell(q, k, v)
    torch.cuda.synchronize()
    elapsed = (time.time() - start) / num_runs * 1000  # ms

    # Calculate TFLOPS
    flops = 2 * batch_size * num_heads * seqlen_q * seqlen_k * head_dim  # QK
    flops += 2 * batch_size * num_heads * seqlen_q * seqlen_k * head_dim  # PV
    tflops = flops / (elapsed / 1000) / 1e12

    print(f"seqlen_q={seqlen_q}, seqlen_k={seqlen_k}, heads={num_heads}, d={head_dim}")
    print(f"  Time: {elapsed:.3f} ms, TFLOPS: {tflops:.1f}")

if __name__ == "__main__":
    for seq_len in [1024, 2048, 4096, 8192]:
        benchmark_attention(seq_len, seq_len, 32, 128)
    print()
    for seq_len in [1024, 2048, 4096, 8192]:
        benchmark_attention(seq_len, seq_len, 32, 64)
```

---

## 5.3 Verification Milestones

After each phase, verify your understanding with these checkpoints:

### Phase 1 Verification ✓

- [ ] Can explain what `Layout<Shape<_4, _8>, Stride<_8, _1>>` means
- [ ] Can compute the offset for any coordinate in a layout
- [ ] Can explain what `zipped_divide` does (conceptually)
- [ ] Can explain the difference between `gmem_ptr`, `smem_ptr`, and register fragments
- [ ] Can explain the TMA pipeline pattern (acquire → copy → release)
- [ ] Know what `SM120_16x32x64_TN_VS_NVFP4` means (each part of the name)

### Phase 2 Verification ✓

- [ ] Can draw the FMHA data flow: QK → softmax → PV
- [ ] Can explain the online softmax algorithm (why we need rescaling)
- [ ] Can explain SA3's fused softmax + quantization (how AbsMaxP is computed)
- [ ] Can explain two-level P scaling (fp8_scalexfp4_scale)
- [ ] Can map sections of CuTeDSL `fmha.py` to SA3 C++ files

### Phase 3 Verification ✓

- [ ] Can draw the warp specialization diagram (3 warp groups, roles)
- [ ] Can explain why producer uses 24 registers and consumers use 232
- [ ] Can explain the persistent tile scheduler (grid = num_SMs, tiles loop)
- [ ] Can list all 7 TMA descriptors and what they load
- [ ] Can explain the `OrderedSequenceBarrier` protocol for epilogue

### Phase 4 Verification ✓

- [ ] Can trace a single attention tile through every stage of `mma()`
- [ ] Can explain `add_delta_s`: why delta_s is added to the accumulator before GEMM
- [ ] Can explain the masking: how `make_identity_tensor` is used for position-based masking
- [ ] Can explain the quantize lambda: float → FP4 conversion + warp shuffle for SFs
- [ ] Can explain the 3 loop sections: first tile, causal masking tiles, full tiles
- [ ] Can explain `finalize`: row_sum reduction across 4 threads, then division

### Phase 5 Verification ✓

- [ ] Successfully ran the layout explorer script
- [ ] Successfully ran a CuTeDSL example (even if not on Blackwell)
- [ ] Can predict the effect of changing kBlockN from 128 to 64
- [ ] Understand what would need to change for d=256 support
- [ ] Can benchmark and measure actual kernel performance

---

## 5.4 Deep Dive Topics (Advanced)

Once you've completed the main phases, explore these:

### Topic 1: TMEM and Scale Factor Layout

The Blackwell Tensor Memory (TMEM) has specific constraints on how scale factors must be laid out. Study `blockscaled_layout.h` to understand:
- Why `SfAtom` uses `Shape<Shape<_16, _4>, Shape<_16, _4>>`
- Why stride `_0` is used (broadcasting)
- How `tile_to_shape` extends the atom to full tile dimensions

### Topic 2: Swizzling for Bank-Conflict-Free Access

The smem layouts use swizzling (`sm120_rr_smem_selector`). Study:
- What swizzling means (XOR-based address transformation)
- Why `as_position_independent_swizzle_tensor` is needed for copies
- How it prevents bank conflicts in the 32-wide smem banks

### Topic 3: Producer-Consumer Overlapping

The key performance optimization is overlapping:
- TMA loads (producer) with MMA compute (consumer)
- smem→reg copy with MMA compute
- FP4 quantization with PV GEMM

Study how the pipeline depth (kStages=3) enables this and whether more stages would help.

### Topic 4: Register Pressure Analysis

With 232 registers per thread and 128 consumer threads per warp group:
- Q fragment: ~64 registers
- K fragment: ~128 registers (per stage)
- S accumulator: ~64 registers
- P (FP4): ~16 registers
- V fragment: ~128 registers
- O accumulator: ~64 registers
- Softmax state: ~4 registers
- Total: 232 registers → very tight!

Study whether the kernel is register-limited and how register allocation affects occupancy.

### Topic 5: Port SA3 to CuTeDSL

The ultimate exercise: write a minimal SA3 in CuTeDSL:

```python
# Pseudocode for a CuTeDSL SA3:
@cute.kernel
def sage3_attention(Q, K, V, SFQ, SFK, SFV, DeltaS, O):
    # 1. Setup warp specialization
    # 2. Create TMA descriptors
    # 3. Producer loop: TMA loads
    # 4. Consumer loop:
    #    - GEMM-I with block-scaled NVFP4
    #    - Fused softmax + FP4 quantization
    #    - GEMM-II with block-scaled NVFP4
    #    - Online rescaling
    # 5. Epilogue: store O via TMA
```

This would validate your understanding of every concept and give you a functional prototype for experimenting with algorithmic changes.

---

## 5.5 Recommended Study Schedule

| Day | Activity | Time |
|-----|----------|------|
| 1 | Read Phase 1, run layout explorer | 2-3 hours |
| 2 | Run CuTeDSL examples (nvfp4_gemm_0.py, etc.) | 2-3 hours |
| 3 | Read Phase 2, write softmax simulation | 2-3 hours |
| 4 | Read Phase 3 (architecture), skim all 14 files | 3-4 hours |
| 5-6 | Read Phase 4 (mainloop line-by-line), with code open | 4-6 hours |
| 7 | Add debug prints, modify kBlockN, benchmark | 3-4 hours |
| 8-10 | Deep dive topics, start CuTeDSL port | 6-8 hours |

**Total estimated time**: 25-35 hours of focused study.

---

## Key External Resources

1. **CuTe docs** (in repo): `csrc/cutlass/media/docs/cute/` (if the submodule has them)
2. **CuTeDSL tutorial notebooks**: `csrc/cutlass/examples/python/CuTeDSL/notebooks/`
3. **GTC talks**: Search "CuTe Cute Layouts Tensors Atoms" on YouTube
4. **FlashAttention papers**: FA1 (2022), FA2 (2023), FA3 (2024)
5. **SageAttention3 paper**: `papers/sageattention/SageAttention3-2505.11594.pdf`
6. **PTX ISA reference**: NVIDIA PTX ISA documentation for `mma.sync`, `cvt.rn.satfinite.e2m1x2`, etc.
7. **Paper summaries**: `papers/sageattention/paper_summaries.md`
8. **Existing pseudocode**: `tasks/sage3_impl_pseudocode/sageattention3_pseudocode_v3.py`
9. **Flow visualization**: `tasks/sage3_flow_vis/sage3-flash-flow-v1-annotated.excalidraw`
