# HunyuanVideo 1.5 Workload Profile
Config: 320×576, 121 frames, 50 steps, attention=sage3 (SageAttention3 CUTE/Blackwell)
GPU: NVIDIA GeForce RTX 5090 D

## Timing Summary
| Metric | Value |
|--------|------:|
| **Pipeline total (ms)** | 220,146 |
| Attention kernel (ms) | 41,689 |
| Attention calls | 5,400 |
| Avg attn per call (ms) | 7.72 |
| **Attn / Pipeline** | **18.9%** |
