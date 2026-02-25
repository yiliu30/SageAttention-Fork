
# Profiling configurations:

| Parameter | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 D (32 GB VRAM) |
| Driver | 580.105.08 |
| Model | CogVideoX-2b (`cogvideox-2b`) |
| Model dtype | float16 |
| Resolution | 480×720 (default: sample_height=60 × 8, sample_width=90 × 8) |
| Num frames | 49 |
| Num inference steps | 5 (smoke mode) |
| Guidance scale | 6 |
| Batch size | 2 (guidance_scale > 1 doubles batch: negative + positive prompt) |
| Attention seq_len | 17,776 (= 226 text + 13 × 30 × 45 video tokens) |
| Attention shape | `[2, 30, 17776, 64]` → `(B, heads, seq_len, head_dim)` |
| Prompt | "A dog is running in the park." |
| Warmup iterations | 3 |
| CPU offload | Disabled (full model on CUDA) |
| Profiling method | Async CUDA event timing (GPU device time) |


### Results:
============================================================
Attention Proportion Report (sage3)
============================================================
  Total pipeline GPU time :     9664.2 ms
  Transformer GPU time    :     4651.0 ms
  Attention kernel time   :     1625.4 ms
  ---- Proportions ----
  Attn / Pipeline         :       16.8 %
  Attn / Transformer      :       34.9 %
  Transformer / Pipeline  :       48.1 %
  ---- Counts ----
  Attention calls         :        150
  Transformer fwd calls   :          5
  Avg attn per call       :      10.84 ms
  ---- Sage3 Sub-step Breakdown ----
  preprocess_qkv                :    317.1 ms ( 19.5% of attn)
  scale_and_quant_fp4           :     16.9 ms (  1.0% of attn)
  scale_and_quant_fp4_permute   :     16.3 ms (  1.0% of attn)
  scale_and_quant_fp4_transpose :     16.4 ms (  1.0% of attn)
  blockscaled_fp4_attn          :   1232.2 ms ( 75.8% of attn)
============================================================


============================================================
Attention Proportion Report (sdpa)
============================================================
  Total pipeline GPU time :    11350.6 ms
  Transformer GPU time    :     6290.7 ms
  Attention kernel time   :     3216.1 ms
  ---- Proportions ----
  Attn / Pipeline         :       28.3 %
  Attn / Transformer      :       51.1 %
  Transformer / Pipeline  :       55.4 %
  ---- Counts ----
  Attention calls         :        150
  Transformer fwd calls   :          5
  Avg attn per call       :      21.44 ms
============================================================


### OP benchmark

#### Config 1: `batch=2, head=32, headdim=64` (CogVideoX-2b actual config)

##### `is_causal=False`

| seq_len | FA2 (TFLOPS) | Sage3 (TFLOPS) | Sage3 / FA2 speedup |
|---|---|---|---|
| 1024 | 164.7 | 77.6 | 0.47× |
| 2048 | 195.1 | 310.5 | 1.59× |
| 4096 | 213.0 | 407.5 | 1.91× |
| 8192 | 222.1 | 452.9 | 2.04× |
| 16384 | 225.3 | 485.4 | 2.15× |
| 32768 | 227.1 | 508.3 | 2.24× |

##### `is_causal=True`

| seq_len | FA2 (TFLOPS) | Sage3 (TFLOPS) | Sage3 / FA2 speedup |
|---|---|---|---|
| 1024 | 115.5 | 37.9 | 0.33× |
| 2048 | 156.0 | 151.5 | 0.97× |
| 4096 | 186.0 | 294.1 | 1.58× |
| 8192 | 208.8 | 348.8 | 1.67× |
| 16384 | 217.2 | 392.9 | 1.81× |
| 32768 | 222.7 | 425.1 | 1.91× |

#### Config 2: `batch=4, head=32, headdim=128` (paper benchmark config)

##### `is_causal=False`

| seq_len | FA2 (TFLOPS) | Sage3 (TFLOPS) | Sage3 / FA2 speedup |
|---|---|---|---|
| 1024 | 171.3 | 265.6 | 1.55× |
| 2048 | 193.5 | 369.5 | 1.91× |
| 4096 | 203.4 | 496.1 | 2.44× |
| 8192 | 208.9 | 606.1 | 2.90× |
| 16384 | 211.8 | 678.2 | 3.20× |
| 32768 | 213.0 | 714.5 | 3.35× |

##### `is_causal=True`

| seq_len | FA2 (TFLOPS) | Sage3 (TFLOPS) | Sage3 / FA2 speedup |
|---|---|---|---|
| 1024 | 142.0 | 150.3 | 1.06× |
| 2048 | 173.6 | 236.6 | 1.36× |
| 4096 | 193.7 | 347.0 | 1.79× |
| 8192 | 204.4 | 478.5 | 2.34× |
| 16384 | 209.2 | 587.8 | 2.81× |
| 32768 | 212.0 | 647.2 | 3.05× |

### Hardware Theoretical Peaks

RTX 5090 specs (from [NVIDIA official](https://www.nvidia.com/en-us/geforce/graphics-cards/compare/#50-series)):

| Spec | Standard RTX 5090 | Our RTX 5090 D (measured) |
|---|---|---|
| SMs | 170 | 170 (identical) |
| CUDA Cores | 21,760 | 21,760 |
| Official Boost Clock | 2.41 GHz | N/A (not published) |
| Sustained clock (GEMM load) | — | ~2.67 GHz |
| TGP | 575 W | 600 W |
| VRAM | 32 GB GDDR7 | 32 GB GDDR7 |
| AI TOPS (FP4 sparse) | 3,352 | — |

Theoretical tensor core peaks (at standard 2.41 GHz boost):

| Precision | Dense (TFLOPS) | Sparse (TFLOPS) |
|---|---|---|
| FP16 | 419.5 | 839.1 |
| FP8 | 839.1 | 1,678.1 |
| FP4 | 1,678.1 | 3,356.3 (≈ NVIDIA spec 3,352) |

**Key finding:** The 5090 D has identical SM count (170) to the standard 5090 and actually sustains a *higher* clock (~2.67 GHz vs 2.41 GHz spec) with a higher power limit (600W vs 575W). **Hardware differences do NOT explain the gap.**

#### Analysis: gap vs paper's 1038 TOPS claim

The paper reports **1038 TOPS** peak for Sage3 on RTX 5090. Our measurement peaks at **714.5 TFLOPS** (non-causal, seq_len=32768, headdim=128).

Hardware utilization of our SA3 benchmark (714.5 TFLOPS):
- vs FP8 dense peak @ 2.67 GHz (929.6): **76.9%** — indicates strong FP8-level utilization
- vs FP4 dense peak @ 2.67 GHz (1859.2): **38.4%**
- SA3 uses FP4 for QK matmul + FP8 for PV matmul, so effective peak lies between FP8 and FP4

Paper's 1038 TOPS on standard RTX 5090:
- vs FP4 dense @ 2.41 GHz (1678.1): **61.9%**
- This exceeds FP8 dense peak (839.1), confirming the paper's kernel heavily leverages FP4 tensor cores

Possible remaining factors for the 714.5 → 1038 gap:

1. ~~RTX 5090 vs RTX 5090 D hardware~~ — **ruled out** (same 170 SMs, higher clock on D variant)
2. **Longer seq_len** — the paper's figures likely extend beyond 32768 where throughput continues to increase
3. **Kernel version / build optimization** — our build may not include the latest kernel optimizations from the paper
4. **Benchmark methodology** — the paper may use CUDA event timing rather than `torch.utils.benchmark.Timer` which includes host-side overhead
5. **Driver/CUDA version** — kernel performance can vary across driver versions