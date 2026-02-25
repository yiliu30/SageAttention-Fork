
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

#### Analysis: gap vs paper's 1038 TOPS claim

The paper reports **1038 TOPS** peak for Sage3 on RTX 5090. Our measurement peaks at **714.5 TFLOPS** (non-causal, seq_len=32768, headdim=128). Possible remaining factors:

1. **RTX 5090 vs RTX 5090 D** — our GPU is the "D" variant (China-specific, potentially lower clocks or fewer SMs). The paper benchmarks on a standard RTX 5090.
2. **Longer seq_len** — the paper's figures likely extend beyond 32768 where throughput continues to increase.
3. **Benchmark methodology** — the paper may use CUDA event timing rather than `torch.utils.benchmark.Timer` which includes host-side overhead.
4. **Driver/CUDA version** — kernel performance can vary across driver versions.