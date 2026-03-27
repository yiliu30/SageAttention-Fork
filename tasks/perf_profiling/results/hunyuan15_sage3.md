# HunyuanVideo 1.5 Workload Profile — SageAttention3
Config: 320×576, 121 frames, 50 steps, attention=sage3 (CUTE/Blackwell)
GPU: 2× NVIDIA GeForce RTX 5090 D (device_map=balanced, no CPU offload)
Date: 2026-03-27

## Timing Summary
| Metric | Value |
|--------|------:|
| **Pipeline total (ms)** | 195,581 |
| **Transformer total (ms)** | 189,799 |
| Attention kernel (ms) | 42,507 |
| **Attn / Pipeline** | **21.7%** |
| **Attn / Transformer** | **22.4%** |
| **Transformer / Pipeline** | **97.0%** |

## Per-Call Statistics
| Metric | Value |
|--------|------:|
| Attention calls | 5,400 |
| Transformer fwd calls | 100 |
| Avg attn per call (ms) | 7.87 |

## Sage3 Sub-step Breakdown
| Sub-step | Time (ms) | % of Attention |
|----------|----------:|---------------:|
| preprocess_qkv | 7,491 | 17.6% |
| scale_and_quant_fp4 | 436 | 1.0% |
| scale_and_quant_fp4_permute | 431 | 1.0% |
| scale_and_quant_fp4_transpose | 442 | 1.0% |
| blockscaled_fp4_attn | 30,858 | 72.6% |
