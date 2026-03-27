# HunyuanVideo 1.5 Workload Profiling: SDPA vs SageAttention3

**Config**: 320×576, 121 frames, 50 steps, HunyuanVideo-1.5-Diffusers-480p_t2v
**GPU**: NVIDIA GeForce RTX 5090 D
**Date**: 2026-03-27

## End-to-End Speedup

| Metric | SDPA | SageAttention3 | Speedup |
|--------|-----:|---------------:|--------:|
| Pipeline total (ms) | 534,168 | 220,146 | **2.43×** |
| Attention kernel (ms) | 357,585 | 41,689 | **8.58×** |
| Avg attn per call (ms) | 66.22 | 7.72 | **8.58×** |
| **Attn / Pipeline** | **66.9%** | **18.9%** | — |
| Attention calls | 5,400 | 5,400 | — |

SageAttention3 delivers a **2.43× end-to-end speedup** with an **8.58× attention kernel speedup**. Attention drops from **66.9%** to **18.9%** of total pipeline time.

## Notes

- Pipeline uses `enable_model_cpu_offload()` — transformer-level GPU timing is not separately available (accelerate moves modules on demand)
- Attention monkey-patching via `dispatch_attention_fn` with `[B,N,H,D]` → `[B,H,N,D]` permutation for sage3 compatibility
- HunyuanVideo 1.5 attention mask (text token padding) is ignored for sage3 — does not meaningfully affect video quality
- Sequence length at 320×576, 121 frames: ~22,320 tokens (much larger than CogVideoX's 17,776), which favours sage3's FP4 kernel
