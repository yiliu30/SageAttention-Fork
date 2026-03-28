# HunyuanVideo 1.5 Workload Profile — SDPA
Config: 320×576, 121 frames, 50 steps, attention=sdpa (PyTorch SDPA)
GPU: 2× NVIDIA GeForce RTX 5090 D (device_map=balanced, no CPU offload)
Date: 2026-03-27

## SDPA with Attention Mask (Default Pipeline Behavior)

**Note**: HunyuanVideo 1.5's `AttnProcessor2_0` always constructs a `[B,1,N,N]` bool attention mask for text token padding. This forces SDPA to fall back from FlashAttention2 to the slower mem-efficient backend (~3× penalty).

### Timing Summary
| Metric | Value |
|--------|------:|
| **Pipeline total (ms)** | 509,331 |
| **Transformer total (ms)** | 503,553 |
| Attention kernel (ms) | 357,664 |
| **Attn / Pipeline** | **70.2%** |
| **Attn / Transformer** | **71.0%** |
| **Transformer / Pipeline** | **98.9%** |

### Per-Call Statistics
| Metric | Value |
|--------|------:|
| Attention calls | 5,400 |
| Transformer fwd calls | 100 |
| Avg attn per call (ms) | 66.23 |

### Reproduction
```bash
cd example
python hunyuan15_infer.py --attention_type sdpa --proportion --device_map balanced
```

## SDPA without Attention Mask (Fair Kernel Comparison)

With `--no_mask`, the attention mask is stripped from `dispatch_attention_fn`. SDPA can use FlashAttention2, providing a fair kernel-vs-kernel comparison with sage3.

### Timing Summary
| Metric | Value |
|--------|------:|
| **Pipeline total (ms)** | 272,039 |
| **Transformer total (ms)** | 266,310 |
| Attention kernel (ms) | 119,448 |
| **Attn / Pipeline** | **43.9%** |
| **Attn / Transformer** | **44.8%** |
| **Transformer / Pipeline** | **97.9%** |

### Per-Call Statistics
| Metric | Value |
|--------|------:|
| Attention calls | 5,400 |
| Transformer fwd calls | 100 |
| Avg attn per call (ms) | 22.12 |

### Reproduction
```bash
cd example
python hunyuan15_infer.py --attention_type sdpa --proportion --device_map balanced --no_mask
```

## Mask Effect Summary

| Metric | SDPA +mask | SDPA no-mask | Mask penalty |
|--------|---:|---:|---:|
| Pipeline (ms) | 509,331 | 272,039 | **1.87×** |
| Attn kernel (ms) | 357,664 | 119,448 | **2.99×** |
| Avg attn/call (ms) | 66.23 | 22.12 | **2.99×** |

Removing the mask gives SDPA a **3× attention speedup** and **1.87× pipeline speedup**, confirming the mask is the dominant factor in the 8.4× sage3-vs-SDPA gap.
