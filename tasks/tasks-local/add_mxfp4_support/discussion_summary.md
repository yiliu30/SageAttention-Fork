# Discussion Summary: Chunked Fake-Quant SDPA Approach

## Context

Goal: Add MXFP4 (block size 32, E8M0 scales) support to SageAttention.  
Before writing any GPU kernel, we need to verify accuracy by simulating FP4 quantization in pure PyTorch.

---

## Original Plan vs Revised Plan

The original plan had 4 steps:
1. NVFP4 quant-dequant (no P quant)
2. Tiled attention with online softmax + P quant
3. Triton kernel (NVFP4)
4. Triton kernel (MXFP4)

**Problem:** Steps 2-3 required implementing FlashAttention-style online softmax in Python/Triton — complex, error-prone, and Triton lacks native FP4 support anyway.

**Revised approach:** Use chunked SDPA with two `torch.matmul` calls and fake-quant. This collapses Steps 1-3 into a single, simple implementation.

---

## Chunked SDPA Design

### Core idea

Chunk along Q's sequence dimension (N_q). Each chunk produces full rows of the attention matrix, so softmax is exact — no online softmax needed.

```python
for i in range(0, N, chunk_size):
    q_c = fake_quant(Q[:, :, i:i+chunk_size])       # [B, H, C, d]
    S_c = q_c @ fake_quant(K).T                      # [B, H, C, N_kv] ← full row
    P_c = softmax(S_c * sm_scale, dim=-1)             # exact, not approximate
    O_c = fake_quant(P_c) @ fake_quant(V)             # [B, H, C, d]
```

### Why chunk Q, not K

- Softmax operates along the K dimension (last dim): `softmax(S, dim=-1)`
- Chunking Q → each chunk has full rows `[C, N_kv]` → softmax is exact
- Chunking K → partial rows `[N_q, C]` → can't compute softmax without seeing all tokens → requires online softmax

This is the same dimension used in prefill chunking.

### Memory analysis

For CogVideoX shape `[2, 30, 17776, 64]`:

| chunk_size | Persistent | Chunk temp | Peak VRAM | Iterations |
|---|---|---|---|---|
| 128 | 1.09 GB | 1.10 GB | **2.19 GB** | 139 |
| 256 | 1.09 GB | 2.19 GB | **3.28 GB** | 70 |
| 512 | 1.09 GB | 4.38 GB | **5.48 GB** | 35 |
| 1024 | 1.09 GB | 8.77 GB | **9.86 GB** | 18 |

The dominant cost is the `S` and `P` matrices at `[B, H, C, N_kv]` in fp32 (~99% of chunk memory). With chunk_size=128, peak is only 2.2 GB — fits trivially even with the full CogVideoX model loaded (~16 GB).

---

## Normalized P vs Unnormalized P̃

### Key difference from SageAttention3

SageAttention3 uses online softmax and quantizes **unnormalized** $\tilde{P}_j = e^{S_j - m_{\text{running}}}$ per tile.  
Chunked SDPA uses exact softmax and quantizes **normalized** $P = \text{softmax}(S)$.

### Empirical comparison (CogVideoX shapes, random Q/K)

| Statistic | Normalized P | Unnorm P̃ (per tile) |
|---|---|---|
| Max value | 0.016 | 1.0 |
| Rowmax avg | 0.002 | 0.300 |
| Median | 3.4e-5 | 2.5e-2 |
| Values < 0.01 | 100% | 21% |
| Dynamic range | 15 bits | 14 bits |

**Normalized P is ~150x smaller** because dividing by $\ell$ (sum over all N_kv=17776 tokens) crushes all values toward zero.

### Implications for quantization

1. **Normalized P makes quantization harder.** Values cluster near zero, so scale factors must be tiny — E4M3 handles this (has mantissa bits), but E8M0 (power-of-2 only) struggles.

2. **Two-level scaling is more critical for normalized P.** The technique `sP1 = rowmax / (448 × 6)` was designed to expand P's range into E4M3's representable values. With normalized P (rowmax ~0.002 vs ~0.3), the gap is even larger.

3. **Accuracy on normalized P is a pessimistic estimate.** If MXFP4 accuracy is acceptable on normalized P, it will definitely be acceptable in the real kernel (which quantizes the easier unnormalized P̃). This is a useful sanity-check property.

### Online softmax rescaling

In FlashAttention's online softmax, when a new tile introduces a larger max:

```
α = exp(m_old - m_new)     # ≤ 1.0
O *= α                      # damp all previous accumulation
ℓ *= α                      # damp previous sum
P̃_j = exp(S_j - m_new)     # THIS gets quantized
O += P̃_j @ V_j
```

The cascading rescaling means earlier tiles' quantization errors get exponentially damped — errors from irrelevant tiles (low attention weight) are naturally suppressed. This is another reason the real kernel may have better accuracy than our simulation.

---

## Levels of Fidelity

| Level | Softmax | P quant granularity | P@V matmul | Complexity |
|---|---|---|---|---|
| **A (Simple)** | Exact, full-row | Full-row | Single matmul | ~30 lines |
| **B (Tiled P)** | Exact, full-row | Per-64-tile blocks | Tiled accumulation | ~50 lines |
| **C (Full match)** | Online (running max/sum) | Per-64-tile | Tiled accumulation | ~80 lines |

- **Level A** is sufficient for NVFP4 vs MXFP4 relative comparison (pessimistic baseline)
- **Level B** adds tile-level P quantization via simple reshape
- **Level C** reproduces SageAttention3's exact data flow for paper-number matching

**Recommendation:** Start with Level A. The NVFP4-vs-MXFP4 gap is driven by block size (16 vs 32) and scale type (E4M3 vs E8M0), not by softmax implementation details.

---

## Implementation Status

- [x] `chunked_sdpa.py` — bf16 chunked SDPA baseline (smoke-tested, cos_sim=0.999997 vs `sdpa()`)
- [ ] `nvfp4_quant_dequant()` — NVFP4 fake quantization (block=16, E4M3 scales)
- [ ] `mxfp4_quant_dequant()` — MXFP4 fake quantization (block=32, E8M0 scales)
- [ ] Q/K smoothing + δS correction
- [ ] CogVideoX QKV tensor capture
- [ ] Accuracy evaluation across all layers

---

## Key Takeaways

1. **Chunked Q-dim SDPA works.** Exact softmax, trivial memory (~2 GB for chunk_size=128), chunk-size independent results.

2. **Evaluating normalized P is pessimistic but safe.** If accuracy passes here, it passes in the real kernel.

3. **MXFP4 should be evaluated alongside NVFP4 from the start** — only ~10 lines difference in the quant function.

4. **Triton kernel is unnecessary for accuracy evaluation** — pure PyTorch is simpler, correct, and sufficient.

5. **Level A (simple) is the right starting point.** Tile-level P quant and online softmax matching can be added later if needed.
