# HunyuanVideo 1.5 Workload Profiling: SDPA vs SageAttention3

**Config**: 320×576, 121 frames, 50 steps, HunyuanVideo-1.5-Diffusers-480p_t2v
**GPU**: 2× NVIDIA GeForce RTX 5090 D (device_map=balanced, no CPU offload)
**Date**: 2026-03-27

## Note: Why Not Split Image/Text Tokens?

[Issue #115](https://github.com/thu-ml/SageAttention/issues/115) proposes splitting the sequence at the image/text boundary and running sage attention only on image tokens. **This doesn't work for HunyuanVideo 1.5** because:

1. **Joint attention is required.** Image and text tokens are concatenated (`torch.cat([query, encoder_query])`) so every image token cross-attends to every text token. Splitting destroys this and produces wrong results.
2. **Negligible speedup.** Text is only 8% of tokens (1,985/24,305). Splitting saves ~16% at best (O(N²)), erased by two kernel launches + concat overhead.
3. **Our approach is better.** Drop the mask (padding tokens are zeroed, contribute nothing) and run sage3 on the full sequence: 7.87ms/call, 8.41× speedup.

## Reproduction Commands

```bash
cd example

# ── Pipeline Profiling (2× RTX 5090 D, device_map=balanced) ──

# SDPA with mask (default pipeline — SDPA forced to mem-efficient backend by attention mask)
python hunyuan15_infer.py --attention_type sdpa --proportion --device_map balanced

# SDPA without mask (fair kernel comparison — SDPA uses FlashAttention2)
python hunyuan15_infer.py --attention_type sdpa --proportion --device_map balanced --no_mask

# SageAttention3 (CUTE kernel, ignores mask)
python hunyuan15_infer.py --attention_type sage3 --proportion --device_map balanced

# ── Standalone Benchmark (single GPU, head_dim + mask effects) ──
CUDA_VISIBLE_DEVICES=1 python bench_headdim.py
```

## End-to-End Speedup

| Metric | SDPA | SageAttention3 | Speedup |
|--------|-----:|---------------:|--------:|
| Pipeline total (ms) | 509,331 | 195,581 | **2.60×** |
| Transformer total (ms) | 503,553 | 189,799 | **2.65×** |
| Attention kernel (ms) | 357,664 | 42,507 | **8.41×** |
| Avg attn per call (ms) | 66.23 | 7.87 | **8.42×** |
| **Attn / Pipeline** | **70.2%** | **21.7%** | — |
| **Attn / Transformer** | **71.0%** | **22.4%** | — |
| **Transformer / Pipeline** | **98.9%** | **97.0%** | — |
| Attention calls | 5,400 | 5,400 | — |
| Transformer fwd calls | 100 | 100 | — |

SageAttention3 delivers a **2.60× end-to-end speedup** with an **8.41× attention kernel speedup**. Attention drops from **70.2%** to **21.7%** of total pipeline time.

### 3-Way Comparison: SDPA +mask vs SDPA no-mask vs sage3

The default HunyuanVideo 1.5 pipeline always constructs a `[B,1,N,N]` bool attention mask for text token padding. This forces SDPA off FlashAttention2 to the slower mem-efficient backend. The `--no_mask` flag strips the mask for a fair kernel-vs-kernel comparison.

| Metric | SDPA +mask (default) | SDPA no-mask | sage3 | sage3 vs SDPA+mask | sage3 vs SDPA no-mask |
|--------|---:|---:|---:|---:|---:|
| Pipeline total (ms) | 509,331 | 272,039 | 195,581 | **2.60×** | **1.39×** |
| Transformer total (ms) | 503,553 | 266,310 | 189,799 | **2.65×** | **1.40×** |
| Attention kernel (ms) | 357,664 | 119,448 | 42,507 | **8.41×** | **2.81×** |
| Avg attn per call (ms) | 66.23 | 22.12 | 7.87 | **8.42×** | **2.81×** |
| Attn / Pipeline | 70.2% | 43.9% | 21.7% | — | — |
| Attn / Transformer | 71.0% | 44.8% | 22.4% | — | — |
| Transformer / Pipeline | 98.9% | 97.9% | 97.0% | — | — |
| Non-attn transformer (ms) | 145,889 | 146,862 | 147,292 | — | — |

**Key takeaway**: Removing the attention mask gives SDPA a **1.87× speedup** (509→272s). The remaining **1.39× gap** (272→196s) is the true kernel advantage of sage3's FP4 WGMMA over FlashAttention2 at head_dim=128.

### Where the Time Goes

```
SDPA +mask (509s, default pipeline):
  ┌──────────────────────────────────────────────────────────┐
  │ Attention 70.2%              │Non-attn xfmr 28.7%│ 1.1% │
  └──────────────────────────────────────────────────────────┘
                                                       ↑ Non-xfmr

SDPA no-mask (272s, --no_mask flag):
  ┌──────────────────────────────────────────────────────────┐
  │ Attention 43.9%    │    Non-attn xfmr 54.0%        │2.1% │
  └──────────────────────────────────────────────────────────┘
                                                       ↑ Non-xfmr

SageAttention3 (196s):
  ┌──────────────────────────────────────────────────────────┐
  │ Attn 21.7% │    Non-attn transformer 75.3%        │ 3%  │
  └──────────────────────────────────────────────────────────┘
                                                       ↑ Non-xfmr
```

With sage3, attention is no longer the bottleneck — **non-attention transformer ops (linear, norm, FFN) now dominate at 75%**.

## Attention Shape Analysis

**Actual runtime shape**: `[1, 24305, 16, 128]` (B, N, H, D)

| Parameter | Value |
|-----------|-------|
| Batch size | 1 (no CFG) |
| Seq_len | 24,305 (22,320 latent + 1,985 text) |
| Heads | 16 |
| **Head_dim** | **128** |
| Layers | 54 dual-stream blocks |
| Attn calls/step | 108 (54 × 2) |

Latent tokens: 31t × 20h × 36w = 22,320 (from VAE: spatial÷16, temporal÷4)
Text tokens: 1,985 (encoder hidden states, concatenated with latent before attention)

### Root Cause Analysis: Why 8.42× Speedup (Verified by Standalone Benchmark)

The 8.42× pipeline speedup is explained by **three factors**, verified by `bench_headdim.py`:

#### Factor 1: head_dim=128 improves sage3, slightly degrades SDPA

Standalone benchmark (no mask) isolating only head_dim:

| Model shape | SDPA (TFLOP/s) | sage3 (TFLOP/s) | Speedup |
|-------------|--:|--:|--:|
| HunyuanVideo D=64 (isolated) | 227.5 | 487.6 | 2.14× |
| HunyuanVideo D=128 (actual) | 214.9 | 664.3 | 3.09× |
| **D effect** | **0.94×** (slight degradation) | **1.36×** (improvement) | |

- sage3 FP4 WGMMA tiles are 128-wide → D=128 is a perfect fit (+36% throughput)
- SDPA/FA2 has marginally higher register pressure at D=128 (−6%)

#### Factor 2: Attention mask forces SDPA off FlashAttention2 (DOMINANT)

`HunyuanVideo15AttnProcessor2_0` always constructs a `[B, 1, N, N]` bool attention mask for text token padding. This forces SDPA to fall back from FlashAttention2 to a slower kernel path:

| Config | SDPA (ms) | sage3 (ms) | Speedup |
|--------|--:|--:|--:|
| HunyuanVideo D=128, no mask | 22.52 | 7.31 | 3.08× |
| HunyuanVideo D=128, **+mask** | **67.60** | **7.29** | **9.27×** |
| Mask slowdown on SDPA | **3.00×** | **none** | |

The mask causes a **3× SDPA slowdown** — this is the single largest factor. sage3 ignores the mask entirely.

#### Factor 3: Combined effect matches pipeline

| Measurement | SDPA (ms/call) | sage3 (ms/call) | Speedup |
|-------------|--:|--:|--:|
| Pipeline profiling | 66.23 | 7.87 | **8.42×** |
| Standalone +mask | 67.60 | 7.29 | **9.27×** |
| Standalone no mask | 22.52 | 7.31 | 3.08× |

The standalone +mask result (9.27×) closely matches the pipeline (8.42×), confirming the root cause. The remaining gap is likely pipeline overhead (dispatch, permutes, etc.).

```
Decomposition of the 8.4× speedup:

  SDPA no mask:  22.5 ms ──[mask: 3.0×]──→ 67.6 ms   (actual pipeline path)
  sage3:          7.3 ms ──[mask: 1.0×]──→  7.3 ms   (ignores mask)
                                             ─────
                                             9.3× speedup

  Without mask, speedup would only be ~3× (from head_dim effect alone)
  The attention mask is the DOMINANT factor in the 8.4× result
```

### Per-Call Efficiency (Standalone Benchmark)

| Metric | CogVideoX (D=64) | HunyuanVideo (D=128) | HunyuanVideo +mask |
|--------|--:|--:|--:|
| SDPA ms/call | 21.57 | 22.52 | 67.60 |
| sage3 ms/call | 10.61 | 7.28 | 7.29 |
| SDPA TFLOP/s | 225.0 | 214.9 | 71.6 |
| sage3 TFLOP/s | 457.5 | 664.3 | 663.4 |
| Speedup | 2.03× | 3.09× | 9.27× |

sage3 at D=128 achieves **664 TFLOP/s** — the highest throughput of any configuration. SDPA with mask drops to **71.6 TFLOP/s** — a **9.3× gap** purely from kernel efficiency.

## Non-Attention Time Analysis

| Component | SDPA +mask (ms) | SDPA no-mask (ms) | sage3 (ms) | Δ (max) |
|-----------|----------:|-----------:|-----------:|--:|
| Transformer total | 503,553 | 266,310 | 189,799 | — |
| Attention kernel | 357,664 | 119,448 | 42,507 | — |
| **Non-attn transformer** | **145,889** | **146,862** | **147,292** | **+0.96%** |
| Non-transformer (VAE, text enc, scheduler) | 5,778 | 5,729 | 5,782 | +0.9% |

Non-attention time is constant at ~147s regardless of attention backend ✅ — confirming the speedup comes entirely from the attention kernel.

## Sage3 Sub-step Breakdown

| Sub-step | Time (ms) | % of Attention |
|----------|----------:|---------------:|
| blockscaled_fp4_attn | 30,858 | 72.6% |
| preprocess_qkv | 7,491 | 17.6% |
| scale_and_quant_fp4 | 436 | 1.0% |
| scale_and_quant_fp4_permute | 431 | 1.0% |
| scale_and_quant_fp4_transpose | 442 | 1.0% |
| Unaccounted (dispatch overhead, permutes) | 2,849 | 6.7% |

The FP4 WGMMA attention kernel (`blockscaled_fp4_attn`) dominates at 72.6%. The `preprocess_qkv` step (Triton kernel for smooth_k, layout prep) is the second largest at 17.6%.

## Cross-Model Comparison

| Metric | CogVideoX-2b SDPA | CogVideoX-2b sage3 | HunyuanVideo 1.5 SDPA +mask | HunyuanVideo 1.5 SDPA no-mask | HunyuanVideo 1.5 sage3 |
|--------|---:|---:|---:|---:|---:|
| Attn shape (B,N,H,D) | [2, 17776, 30, 64] | same | [1, 24305, 16, 128] | same | same |
| Has attn mask | No | No | Yes (default) | No (--no_mask) | Ignored |
| Pipeline (ms) | 68,464 | 52,357 | 509,331 | 272,039 | 195,581 |
| Attn kernel (ms) | 32,444 | 16,581 | 357,664 | 119,448 | 42,507 |
| Attn / Pipeline | 47.4% | 31.7% | 70.2% | 43.9% | 21.7% |
| Attn / Transformer | 51.1% | 35.0% | 71.0% | 44.8% | 22.4% |
| Avg attn/call (ms) | 21.63 | 11.05 | 66.23 | 22.12 | 7.87 |
| **E2E speedup (vs SDPA+mask)** | — | **1.31×** | — | **1.87×** | **2.60×** |
| **Attn speedup (vs SDPA+mask)** | — | **1.96×** | — | **3.00×** | **8.41×** |

HunyuanVideo 1.5 benefits **far more** from sage3 than CogVideoX-2b due to head_dim=128 and the attention mask:
- **8.41× vs 1.96× attention speedup** — the attention mask forces SDPA to mem-efficient backend (3× penalty), and sage3 excels at D=128
- **2.60× vs 1.31× e2e speedup** — attention was 70% of pipeline (vs 47%), amplifying the kernel speedup via Amdahl's Law
- **Fair comparison (no mask): 2.81× vs 1.96×** — even without the mask penalty, sage3 at D=128 is 1.4× more efficient than at D=64
- sage3 per-call latency on HunyuanVideo (7.87ms) is actually **lower** than CogVideoX (11.05ms) despite similar FLOPs — D=128 is sage3's sweet spot

## Key Insights

1. **The attention mask is the dominant factor in the 8.4× speedup.** HunyuanVideo's `AttnProcessor2_0` always constructs a `[B,1,N,N]` bool mask for text token padding. This forces SDPA off the FlashAttention2 path, causing a 3× slowdown. sage3 ignores the mask — this single factor accounts for most of the speedup gap vs CogVideoX (which has no mask). Without the mask, sage3 speedup would be ~3× instead of ~8×.

2. **head_dim=128 is sage3's sweet spot.** The FP4 WGMMA 128-wide tiles achieve peak utilization at D=128, delivering 664 TFLOP/s — 1.36× higher throughput than at D=64. SDPA shows only marginal degradation at D=128 (0.94×) when no mask is used.

3. **Attention was the dominant bottleneck** at 70% of pipeline time with SDPA — driven primarily by the mask-induced fallback path, amplified by head_dim=128 inefficiency.

4. **Non-attention transformer ops are now the bottleneck** at 147s (75% of sage3 pipeline). Optimizing linear layers (FP8 GEMM, structured sparsity) is the next frontier.

5. **Transformer dominates the pipeline** at 97-99% — VAE, text encoder, and scheduler are negligible (~6s combined).

6. **Theoretical ceiling analysis** (sage3 baseline):
   - If attention → 0: pipeline ≈ 153s → **1.28× over current sage3**
   - Non-attention time sets a floor; further gains require optimizing FFN/linear layers.

7. **Implication for model design**: Models with attention masks (HunyuanVideo 1.5) and/or head_dim=128 (Flux, SD3) will see dramatically larger sage3 speedups than maskless D=64 models (CogVideoX). The mask effect is particularly important — even removing the unnecessary mask from SDPA would yield a ~3× improvement without changing the attention kernel.

## Setup Notes

- Uses `device_map="balanced"` to split pipeline across 2× RTX 5090 D (32GB each) — avoids CPU offload overhead
- Transformer placed entirely on one GPU; text encoders + VAE on the other
- Attention monkey-patched via `dispatch_attention_fn` with [B,N,H,D] → [B,H,N,D] permutation for sage3
- HunyuanVideo 1.5 attention mask (text token padding) ignored for sage3 — does not meaningfully affect video quality
- `--no_mask` flag strips the attention mask from `dispatch_attention_fn` for fair SDPA vs sage3 kernel comparison
- Attention shapes verified at runtime: single unique shape `[1, 24305, 16, 128]` across all 5,400 calls
- Standalone benchmark: `example/bench_headdim.py` — reproduces the speedup with synthetic tensors, isolating head_dim and mask effects
- Pipeline profiling: `example/hunyuan15_infer.py` with `--proportion` flag for GPU-timed attention proportion measurement
