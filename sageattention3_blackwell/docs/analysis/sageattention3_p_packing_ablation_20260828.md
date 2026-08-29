# SageAttention 3 P Packing Ablation

## Question

Does the FP32-to-E2M1 conversion and packing of the softmax probability tensor
P materially limit the baseline SageAttention 3 NVFP4 kernel?

## Method

A diagnostic baseline kernel replaces only `packed_float_to_e2m1` with:

- an empty volatile inline-assembly dependency sink for all eight FP32 inputs,
  which prevents the compiler from deleting the softmax computation; and
- a constant packed FP4 word consumed by the unchanged P-by-V MMA.

Scale conversion and packing, QK MMA, P-by-V MMA, TMA traffic, scheduling, and
the epilogue remain enabled. The diagnostic output is intentionally invalid
for attention and is used only for performance attribution.

The normal and bypass kernels were timed in alternating order with 15 warmup
iterations and 75 measured iterations at B=1, H=40, S=16384, D=128, BF16,
non-causal.

## CUDA-event result

| Variant | Median latency |
|---|---:|
| Normal baseline | 6.976 ms |
| P conversion/packing bypass | 6.194 ms |
| Exposed conversion/packing cost | 0.783 ms |

The exposed cost is 11.22% of baseline latency, and the bypass is 1.126x
faster.

## NCU validation

| Metric | Normal | Bypass |
|---|---:|---:|
| Threads/CTA | 384 | 384 |
| Grid CTAs | 170 | 170 |
| Registers/thread | 168 | 168 |
| Dynamic shared memory/CTA | 100.35 KB | 100.35 KB |
| Achieved occupancy | 20.82% | 20.83% |
| Global TMA read bytes | 12.462 GB | 12.462 GB |
| TMA load instructions | 3,287,040 | 3,287,040 |
| Executed OMMA instructions | 335,544,320 | 335,544,320 |
| Executed E2M1 conversion instructions | 167,772,160 | 0 |
| Total executed instructions | 4,543,580,841 | 4,180,240,943 |
| NCU replay duration | 7.34 ms | 6.52 ms |

The resource usage, tensor-core work, and TMA work are unchanged. Total
instructions fall by 363.34 million, or 8.0%, and tensor-pipe utilization rises
from 45.05% to 50.87% because the same tensor work completes in less time.

## Conclusion

P conversion and packing is a major optimization target. Removing it exposes
an 11.22% latency reduction under the 16K workload, above the 10% threshold
chosen for a strong target.

The expensive operation is not a standalone nibble-copy instruction. The
native `F2FP.SATFINITE.E2M1.F32.PACK_AB_MERGE_C` instruction combines numeric
conversion with pair packing. The next optimization should therefore target
the number, scheduling, or overlap of these conversion-and-pack instructions,
not only the final packed-word move.

##
```text
GPU: NVIDIA GeForce RTX 5090 D
Config: B=1, H=40, S=16384, D=128, BF16, causal=False
Materialized QK^T score tensor: 20.00 GiB

Method                                      Median (ms)       TOPS
-------------------------------------------------------
PyTorch SDPA (Flash)                             25.877      212.5
SageAttention 3 NVFP4                             8.386      655.6
SageAttention 3 NVFP4 (two CTA)                  10.004      549.5
SageAttention 3 FP4 kernel only                   7.023      782.8
SageAttention 3 FP4 kernel only (two CTA)         7.982      688.7
PyTorch matmul (QK^T, K transposed)              44.580       61.7
PyTorch matmul (16384^3)                         39.885      220.5
PyTorch NVFP4 scaled_mm (16384^3)                 7.938     1108.1

Ratios relative to the large GEMM references:
  PyTorch SDPA effective TOPS / GEMM TOPS: 0.96x
  SageAttention effective TOPS / GEMM TOPS: 2.97x
  SageAttention FP4 kernel-only effective TOPS / GEMM TOPS: 3.55x
  PyTorch SDPA effective TOPS / NVFP4 GEMM TOPS: 0.19x
  SageAttention effective TOPS / NVFP4 GEMM TOPS: 0.59x
  SageAttention FP4 kernel-only effective TOPS / NVFP4 GEMM TOPS: 0.71x
  Two-CTA FP4 kernel-only effective TOPS / NVFP4 GEMM TOPS: 0.62x
  SageAttention / PyTorch SDPA: 3.09x

Two-CTA speedup over baseline:
  End-to-end: 0.838x
  FP4 kernel only: 0.880x

P conversion/packing ablation (interleaved):
  Normal FP4 kernel: 7.036 ms
  Bypassed P packing: 6.243 ms
  Bypassed P packing effective TOPS: 880.5
  Exposed P packing cost: 0.793 ms
  Fraction of normal latency: 11.3%
  Bypass speedup: 1.127x

```