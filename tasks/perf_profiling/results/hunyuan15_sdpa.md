# HunyuanVideo 1.5 Workload Profile
Config: 320×576, 121 frames, 50 steps, attention=sdpa (PyTorch SDPA)
GPU: NVIDIA GeForce RTX 5090 D

## Timing Summary
| Metric | Value |
|--------|------:|
| **Pipeline total (ms)** | 534,168 |
| Attention kernel (ms) | 357,585 |
| Attention calls | 5,400 |
| Avg attn per call (ms) | 66.22 |
| **Attn / Pipeline** | **66.9%** |
