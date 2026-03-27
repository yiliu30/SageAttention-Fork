# CogVideoX-2b Workload Profiling: SDPA vs SageAttention3

**Config**: 480×720, 49 frames, 50 steps, batch=2 (CFG), CogVideoX-2b
**GPU**: NVIDIA GeForce RTX 5090 D
**Date**: 2026-03-27

## End-to-End Speedup

| Metric | SDPA | SageAttention3 | Speedup |
|--------|-----:|---------------:|--------:|
| Pipeline total (ms) | 68,464 | 52,357 | **1.31×** |
| Transformer total (ms) | 63,462 | 47,426 | **1.34×** |
| Per-step latency (ms) | 1,269 | 949 | **1.34×** |

SageAttention3 delivers a **1.31× end-to-end speedup** by cutting attention kernel time in half.

### Where the Time Goes

```
CogVideoX-2b Pipeline:

  Text Encoder ──> [ Transformer x50 steps ] ──> VAE Decode
      (~1s)        ╔══════════════════════╗        (~4s)
                   ║  30x CogVideoXBlock  ║
                   ║  ┌─ Norm Linear      ║
                   ║  ├─ QKV Projection   ║
                   ║  ├─ Attention Kernel  ║  <-- sage3 speeds this up
                   ║  ├─ Output Projection ║
                   ║  ├─ FFN (up + down)   ║
                   ║  └─ Residual + Norm   ║
                   ╚══════════════════════╝
                      92% of total time
```

The pipeline runs **50 transformer forward passes** (one per denoising step). The transformer dominates:

| | SDPA | SageAttention3 |
|--|-----:|---------------:|
| 50× Transformer forward | 63.5s (92.7%) | 47.4s (90.6%) |
| Other (VAE, text encoder, scheduler) | 5.0s (7.3%) | 4.9s (9.4%) |
| **Pipeline total** | **68.5s** | **52.4s** |

## GPU Time Breakdown — Side by Side

| Component | SDPA (ms) | % Xfmr | Sage3 (ms) | % Xfmr | Δ ms | Change |
|-----------|----------:|-------:|-----------:|-------:|-----:|-------:|
| **Transformer total** | 63,462 | 100.0% | 47,426 | 100.0% | −16,036 | −25.3% |
| Attention kernel | 32,444 | 51.1% | 16,581 | 35.0% | −15,863 | **−48.9%** |
| Attn QKV projection | 5,607 | 8.8% | 5,572 | 11.7% | −36 | −0.6% |
| Attn output projection | 1,884 | 3.0% | 1,853 | 3.9% | −32 | −1.7% |
| FFN (up + down) | 15,321 | 24.1% | 15,209 | 32.1% | −112 | −0.7% |
| Norm linear | 38 | 0.1% | 38 | 0.1% | 0 | — |
| Other linear | 13 | 0.0% | 13 | 0.0% | 0 | — |
| **All linear total** | 22,864 | 36.0% | 22,685 | 47.8% | −179 | −0.8% |
| Unaccounted¹ | 8,155 | 12.8% | 8,160 | 17.2% | +5 | — |
| Outside transformer² | 5,002 | — | 4,931 | — | −71 | −1.4% |

¹ LayerNorm, GELU, RoPE, reshapes, residual adds
² VAE decode, text encoder, scheduler

## Where the Time Goes (% of Transformer)

```
SDPA:
  ┌─────────────────────────────────────────────────────┐
  │ Attention 51.1%  │  FFN 24.1% │ QKV 8.8%│Unc 12.8%│
  └─────────────────────────────────────────────────────┘

SageAttention3:
  ┌─────────────────────────────────────────────────────┐
  │ Attention 35.0%│   FFN 32.1%   │QKV 11.7%│Unc 17.2%│
  └─────────────────────────────────────────────────────┘
```

With sage3, attention is no longer the majority — **FFN is now nearly as large as attention**.

## Per-Call Latency

| Component | SDPA (ms/call) | Sage3 (ms/call) | Speedup |
|-----------|---------------:|----------------:|--------:|
| Attention kernel | 21.63 | 11.05 | **1.96×** |
| Attn QKV proj | 1.25 | 1.24 | 1.01× |
| Attn output proj | 1.26 | 1.24 | 1.02× |
| FFN (per linear) | 5.11 | 5.07 | 1.01× |

Attention kernel is **~2× faster** per call with sage3. All other components are unchanged (as expected — they don't depend on attention backend).

## Sanity Checks

| Check | SDPA | Sage3 | Status |
|-------|------|-------|--------|
| (attn + linear) / transformer | 87.2% | 82.8% | ✅ 85±5% expected |
| Wall-clock vs GPU time | −0.0% | −0.0% | ✅ No host bottleneck |
| Attention calls | 1,500 | 1,500 | ✅ 50 steps × 30 blocks |
| Linear calls consistency | 4500+1500+3000+3050+200 | same | ✅ |

## Comparison with Paper's E2E Results

The SageAttention3 paper (Table 4a) reports CogVideoX-2B end-to-end inference latency on RTX 5090:

| Attention | Paper (RTX 5090) | Our Profiling (RTX 5090 D) | Notes |
|-----------|----------------:|--------------------------:|-------|
| Original (FA2/SDPA) | 64 s | 68.5 s | Paper uses "Original" = FA2; our baseline = SDPA. RTX 5090 D has slightly lower clocks than RTX 5090 |
| SageAttention (8-bit) | 55 s | — | Not profiled |
| SageAttention2 (8-bit) | 46 s | — | Not profiled |
| **SageAttention3 (FP4)** | **27 s** | **52.4 s** | See analysis below |
| Paper speedup | **2.37×** | **1.31×** | |

**Why the gap?** The paper reports **2.4× e2e speedup** for CogVideoX, but we measure only **1.31×**. Key differences:

1. **GPU variant**: RTX 5090 vs RTX 5090 **D** — specs are nearly identical (same CUDA cores, similar clocks), accounts for ~7% at most (64s vs 68.5s on baseline), not the 2× gap.
2. **Attention kernel speedup**: The paper claims **4.85× kernel speedup** (1038 vs 214 TOPS). Our per-call measurement shows **1.96× (21.6 → 11.1 ms/call)**. The paper's TOPS are measured at the raw FP4 CUDA kernel level for large seq_len with head_dim=128; CogVideoX uses seq_len=17,776 with head_dim=64. Issue [#342](https://github.com/thu-ml/SageAttention/issues/342) on the upstream repo shows another user also couldn't reproduce the paper's kernel TOPS.
3. **No e2e benchmark in public repo**: The upstream `example/cogvideox_infer.py` doesn't even have a `sage3` option. The paper's e2e methodology is undisclosed.
4. **SDPA ≈ FA2 on RTX 5090**: PyTorch SDPA dispatches to FA2 on Blackwell (FA3 is Hopper-only), so our baseline should match the paper's.

### Amdahl's Law Analysis: Paper's 27s Is Inconsistent

We can use our detailed profiling to back-calculate what the paper's numbers imply:

**Our measured breakdown (RTX 5090 D):**
- SDPA pipeline: 68.5s = 32.4s attention + 36.0s non-attention
- Sage3 pipeline: 52.4s = 16.6s attention + 35.8s non-attention
- Non-attention time is constant at ~36s ✅ (as expected — linear layers, norms, etc. don't change)

**Scaling to the paper's 64s baseline** (proportional to 5090 vs 5090 D clocks):
- Paper SDPA: 64s → ~30.3s attention + ~33.7s non-attention

**Testing: can the paper's 27s be explained by any attention speedup?**

| Assumed attn speedup | Sage3 attn time | + Non-attn (33.7s) | = Pipeline total | Matches paper? |
|---------------------:|----------------:|-------------------:|-----------------:|:--------------:|
| 1.96× (our measured) | 15.5s | 33.7s | **49.2s** | ❌ (vs 27s) |
| 4.85× (paper kernel TOPS) | 6.3s | 33.7s | **40.0s** | ❌ (vs 27s) |
| ∞ (free attention) | 0s | 33.7s | **33.7s** | ❌ (vs 27s) |

**Even with infinitely fast attention, the pipeline can't go below ~34s** — yet the paper claims 27s. This means the paper's 27s result **must include optimizations beyond the attention kernel**, such as:
- `torch.compile` on the full transformer (accelerates linear layers, fusions, etc.)
- Custom optimized linear layers or other pipeline changes
- Different measurement scope (e.g., only the denoising loop, excluding VAE/text encoder)

For the paper's 27s to be achievable, non-attention time would need to be **under 20s** — a 40%+ reduction from what vanilla diffusers produces. This would require something like `torch.compile` which can fuse and accelerate the linear layers, norms, and activations.

### What Our Data Shows vs What the Paper Claims

| Metric | Our Measurement | Paper Claim | Consistent? |
|--------|---------------:|------------:|:-----------:|
| SDPA baseline | 68.5s | 64s | ✅ ~7% clock diff |
| Attention kernel speedup | 1.96× | 4.85× (TOPS) | ⚠️ TOPS ≠ e2e attn call; seq_len/headdim dependent |
| Non-attention time | ~36s | Not disclosed | — |
| Sage3 e2e | 52.4s | 27s | ❌ 27s < our non-attn time |
| E2E speedup | **1.31×** | **2.37×** | ❌ Paper implies hidden optimizations |

**Bottom line**: Our 1.31× e2e speedup from pure attention kernel replacement is the correct, reproducible result. The paper's 2.4× claim cannot be achieved by swapping the attention kernel alone — it requires additional optimizations to the non-attention components that are not disclosed.

## Key Insights

1. **Sage3 cuts attention time by 49%**, which translates to a 1.31× pipeline speedup and 1.34× transformer speedup.

2. **FFN is the next optimization frontier.** With sage3, FFN (32.1%) is nearly as large as attention (35.0%) inside the transformer. The two FFN linears per block (1920→7680 and 7680→1920) account for 15.2s — optimizing these (e.g., FP8 GEMM, structured sparsity) would yield significant further gains.

3. **Attn QKV projections are a secondary target** at 11.7% of transformer time with sage3. These are three 1920→1920 GEMMs per block — potential candidates for FP8 quantization.

4. **Non-linear overhead is fixed at ~8.2s** regardless of attention backend. This sets a floor: even with instant attention, the transformer can't go below ~30.8s (linear + unaccounted).

5. **Theoretical ceiling analysis** (sage3 baseline):
   - If attention → 0: transformer = 30,845 ms → **1.54× over current sage3**
   - If attention + FFN → 0: transformer = 15,636 ms → **3.03× over current sage3**
   - Practical next step: FP8 FFN could save ~30-50% of FFN time → ~5-7s → **1.12-1.16× additional speedup**

## Architecture Reference

CogVideoX-2b: 30 blocks, each containing:
- 3× QKV Linear(1920→1920) + 1× Output Linear(1920→1920)
- 2× FFN Linear (1920→7680 up, 7680→1920 down)
- 2× Norm Linear(512→11520) — negligible time
- Input to attention: [2, 30, 17776, 64] (batch=2 from CFG)
- 245 total nn.Linear layers discovered
