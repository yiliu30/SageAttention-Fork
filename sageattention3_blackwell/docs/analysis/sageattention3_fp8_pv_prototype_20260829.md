# SageAttention 3 FP8 P×V Prototype

## Design

The experimental `fp8_pv` variant keeps SageAttention 3's NVFP4 Q×K path and
changes the P×V path:

```text
BF16/FP16 V
    -> per-channel E4M3 quantization and transpose
    -> E4M3 V plus FP32 scale [B, Hkv, D]

NVFP4 Q×K
    -> online softmax with an implicit per-row factor of 448
    -> E4M3 P
    -> SM120 m16n8k32 E4M3×E4M3 MMA
    -> FP32 accumulation
    -> divide by the softmax denominator
    -> multiply by the per-channel V scale
    -> BF16/FP16 epilogue
```

The P factor cancels between the P×V numerator and softmax denominator, so P
does not need a stored scale. V uses one scale across the sequence for each
batch, KV head, and output channel. The quantizer sets that scale to
`amax(V[:, d]) / 448`.

E4M3 V doubles the staged V storage relative to packed E2M1. Three mainloop
stages would raise shared-memory use above the SM120 limit, so this variant
uses two stages. The compiled kernel uses 95.23 KiB of dynamic shared memory
and 168 registers per thread.

The variant is opt-in through `kernel_variant="fp8_pv"`. The existing NVFP4
kernel remains the default. FP8 P×V rejects the two-CTA diagnostic,
P-packing bypass, attention masks, and causal calls with unequal query and key
lengths.

## Correctness coverage

`benchmarks/validate_fp8_pv.py` checks:

- BF16 and FP16 inputs;
- causal and non-causal attention;
- head dimensions 64 and 128;
- sequence lengths around 32-, 64-, and 128-token boundaries;
- GQA, including different means for each query head;
- zero-valued V channels;
- persistent CTA reuse at 16K;
- CUDA graph replay after input buffers change;
- explicit rejection of unsupported masks and rectangular causal attention.

The checks compare output with PyTorch SDPA and pass on the RTX 5090 D.

## 16K performance

The benchmark used B=1, H=40, S=16384, D=128, BF16, 15 warmups, and 75
measured iterations.

| Variant | Median latency | Effective TOPS |
|---|---:|---:|
| NVFP4 end-to-end | 8.396 ms | 654.8 |
| FP8 P×V end-to-end | 20.649 ms | 266.2 |
| NVFP4 kernel-only | 6.995 ms | 785.9 |
| FP8 P×V kernel-only | 18.096 ms | 303.8 |

FP8 P×V reaches 0.407× the end-to-end speed and 0.387× the kernel-only speed
of the NVFP4 baseline. It does not meet the 3% promotion threshold and remains
an experimental accuracy and architecture reference.

## Nsight Compute result

Nsight Compute 2026.2.1 collected a 41-pass full report:

```text
/home/yiliu7/.local/state/sageattention3/sageattn3_pure_16k_fp8_pv.ncu-rep
```

| Metric | FP8 P×V |
|---|---:|
| Profile duration | 20.76 ms |
| Registers per thread | 168 |
| Dynamic shared memory per CTA | 95.23 KiB |
| Achieved occupancy | 20.83% |
| Eligible warps per scheduler | 0.18 |
| Cycles with no eligible warp | 85.40% |
| Tensor pipeline utilization | 38.5% |
| DRAM throughput | 1.83% |
| L2 hit rate | 99.09% |

The FP8 kernel spends most cycles without an eligible warp. The smaller
m16n8k32 MMA shape, two-stage pipeline, extra V bytes, and FP8 conversion work
outweigh the removal of FP4 P block scales.

