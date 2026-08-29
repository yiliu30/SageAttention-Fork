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

## Direct register-remap variant

`kernel_variant="fp8_pv_register"` retains the complete FP8 numerical path but
replaces the P exchange with a direct register remap. CuTe identity tensors
establish the QK-C and PV-A coordinates, including the inverse K
preprocessing permutation. Each destination PV-A word gathers four converted
E4M3 values from lanes in the same warp.

For this variant the compiler removes the full P shared-memory store, LDSM
reload, and both 256-consumer-thread named barriers per KV tile. The epilogue
still requires the aliased `smem_o` allocation, so dynamic shared memory stays
at 95.23 KiB. The shared-exchange `fp8_pv` variant remains available unchanged
as the reference.

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

Every category runs for both FP8 variants and directly compares their outputs.
The 144 original/category checks plus three direct C++ boundary checks pass on
the RTX 5090 D; register-remap output is bitwise identical to shared exchange.

## 16K performance

The benchmark used B=1, H=40, S=16384, D=128, BF16, 15 warmups, and 75
measured iterations.

| Variant | Median latency | Effective TOPS |
|---|---:|---:|
| NVFP4 end-to-end | 8.386 ms | 655.6 |
| FP8 P×V shared end-to-end | 20.421 ms | 269.2 |
| FP8 P×V register end-to-end | 11.467 ms | 479.4 |
| NVFP4 kernel-only | 7.026 ms | 782.5 |
| FP8 P×V shared kernel-only | 17.928 ms | 306.6 |
| FP8 P×V register kernel-only | 8.954 ms | 614.0 |

Register remap is 1.781× faster end-to-end and 2.002× faster kernel-only than
shared FP8. It still reaches only 0.731× the end-to-end speed and 0.785× the
kernel-only speed of NVFP4, so NVFP4 remains the default.

## Nsight Compute result

Nsight Compute 2026.2.1 collected a 41-pass full report:

```text
/home/yiliu7/.local/state/sageattention3/sageattn3_pure_16k_fp8_pv_register.ncu-rep
```

| Metric | FP8 P×V register |
|---|---:|
| Profile duration | 9.48 ms |
| Registers per thread | 168 |
| Dynamic shared memory per CTA | 95.23 KiB |
| Achieved occupancy | 20.83% |
| Eligible warps per scheduler | 0.34 |
| Cycles with no eligible warp | 71.34% |
| Tensor pipeline utilization | 83.14% |
| DRAM throughput | 3.95% |
| L2 throughput | 27.76% |
| L2 hit rate | 96.06% |

The register variant remains one-CTA limited, but removing shared exchange
raises tensor utilization substantially and almost halves event-measured
kernel latency. NCU reports one allocated barrier instead of 15 and
251.74 million LDSM instructions instead of 272.71 million for shared FP8.
