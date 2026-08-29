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

## Next optimization while retaining E4M3 P and V

The register-remap kernel moves the bottleneck from P exchange to the tensor
pipeline. NCU reports 83.14% tensor-pipeline utilization, 2.06
math-pipe-throttle warp-cycles per issued instruction, and 2.21 wait
warp-cycles per issued instruction. DRAM throughput is 3.95%, so reducing V
traffic cannot address the primary limit.

Each logical m16n32k64 P×V region requires eight native E4M3×E4M3
`m16n8k32` instructions:

```text
four N=8 output slices × two K=32 slices = eight instructions
```

The two K slices update the same accumulator registers. Those read-after-write
dependencies contribute to tensor-pipeline throttling and wait stalls. The
kernel also holds FP32 accumulator fragments for the complete output tile,
which contributes to its 168-register count and one-CTA residency.

### Recommended experiment: FP16 instruction buffer

SM120 provides this E4M3×E4M3 atom:

```cpp
cute::SM120_16x8x32_TN<
    cutlass::float_e4m3_t,
    cutlass::float_e4m3_t,
    cutlass::half_t>
```

Use it for a short partial sum while retaining a persistent FP32 output
accumulator:

```text
E4M3 P × E4M3 V
    -> FP16 temporary accumulator for one 64- or 128-token interval
    -> convert the temporary fragment to FP32
    -> add it to the persistent FP32 output accumulator
```

The FP16 atom stores four accumulator values in two 32-bit registers. The
FP32 atom stores the same four values in four registers. A temporary FP16
fragment can therefore reduce the live MMA accumulator footprint, create room
for independent accumulator groups, and increase the distance between
dependent QMMAs.

P and V remain E4M3. This experiment changes only partial-sum precision.
Folding the temporary fragment into FP32 after each KV tile bounds the length
of the FP16 accumulation chain. The implementation should test both 64-token
and 128-token fold intervals because a shorter interval improves numerical
accuracy but adds conversion and FP32-add instructions.

### Scheduling after the FP16-buffer experiment

Within each logical m16n32k64 region, issue independent N slices before the
second K slice updates the same accumulators:

```text
N0 K0
N1 K0
N2 K0
N3 K0
N0 K1
N1 K1
N2 K1
N3 K1
```

If the FP16 buffer lowers register pressure, keep two score tiles live and run
QK ahead of softmax and P×V:

```text
QK tile i
QK tile i+1
softmax, conversion, and P×V for tile i
QK tile i+2
softmax, conversion, and P×V for tile i+1
```

Online softmax must consume tiles in sequence, but QK computation can run
ahead. This schedule gives the tensor pipeline useful work while scalar
softmax and E4M3 conversion execute.

### Experimental variants and gates

Keep the FP32-accumulator register-remap kernel as the reference. Add the FP16
buffer as a separate opt-in variant so accuracy and latency can be compared
without changing default behavior.

Evaluate:

- fold intervals of 64 and 128 tokens;
- BF16 and FP16 inputs;
- causal and non-causal attention;
- head dimensions 64 and 128;
- GQA and sequence-boundary cases;
- CUDA graph replay and persistent CTA reuse;
- the 16K, 40-head benchmark with 15 warmups and 75 measured iterations.

Report error against BF16 SDPA and against the FP32-accumulator FP8 kernel.
Promote the FP16-buffer variant only if it improves kernel latency and keeps
FP8 output error below the NVFP4 baseline. NCU should confirm lower register
pressure or lower tensor dependency stalls. If register use stays at 168 and
math-pipe throttle does not fall, the FP16 buffer has not changed the limiting
execution schedule.

Shared-memory conflict removal remains secondary. NCU estimates 3.21% local
speedup from excessive shared wavefronts, while FP32 instruction fusion has a
1.27% estimate. These changes should follow the accumulator and scheduling
experiments.

## Standalone SM120 accumulator microbenchmark

`benchmarks/sm120_fp8_mma_accum_bench.cu` directly issues the native
E4M3×E4M3 `QMMA.16832` FP32- and FP16-accumulator instructions. The final run
used GPU 0, 170 SMs, 15 warmups, 75 CUDA-event samples, 65,536 dependent
iterations, 8,192 raw iterations, 4,096 folds, and 60 blocks per SM. Each
native instruction counts as 8,192 operations; setup loads and output stores
are timed but excluded from that count. All checksums were finite.

The run used an NVIDIA GeForce RTX 5090 D with driver 595.84, CUDA 13.0 build
36424714, and CUTLASS commit
`dc45f979ae336a235da1676b311f35efeb30149a`. Reproduce it with:

```bash
./benchmarks/run_sm120_fp8_mma_accum.sh --build-only
CUDA_VISIBLE_DEVICES=0 ./benchmarks/sm120_fp8_mma_accum_bench \
  --mode all --warmup 15 --iterations 75 --latency-inner 65536 \
  --inner 8192 --folds 4096 --blocks-per-sm 60
```

The one-warp-per-SM dependent test measured 35.004 cycles / 12.097 ns per FP32
instruction and 29.754 cycles / 10.308 ns per FP16 instruction, a 1.174×
FP16 reduction in dependent-chain time.

| Chains/warp | FP32 regs, max warps/SM | FP32 TOPS | FP16 regs, max warps/SM | FP16 TOPS | FP16 speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 14, 48 | 504.773 | 12, 48 | 574.836 | 1.139× |
| 2 | 18, 48 | 503.781 | 16, 48 | 573.566 | 1.139× |
| 4 | 26, 48 | 504.583 | 24, 48 | 575.538 | 1.141× |
| 8 | 42, 40 | 504.968 | 32, 48 | 575.660 | 1.140× |

The realistic folded mode used four independent N slices. Runtime-distinct,
rotating valid E4M3 operand words prevent the compiler from coalescing
equivalent short partial chains; NCU verified that the executed tensor
instruction count exactly matches the requested work.

| Fold | Accumulator | Registers/thread | Max blocks/SM | Median | Native MMA/s | Tensor TOPS | FP16 speedup |
|---:|---|---:|---:|---:|---:|---:|---:|
| K=64 | FP32 persistent | 56 | 4 | 43.3794 ms | 61.639 G/s | 504.948 | — |
| K=64 | FP16 partial → FP32 | 45 | 5 | 38.1567 ms | 70.076 G/s | 574.063 | 1.137× |
| K=128 | FP32 persistent | 56 | 4 | 87.1458 ms | 61.365 G/s | 502.705 | — |
| K=128 | FP16 partial → FP32 | 45 | 5 | 76.1909 ms | 70.189 G/s | 574.985 | 1.144× |

SASS contains both required forms:

```text
QMMA.16832.F32.E4M3.E4M3
QMMA.16832.F16.E4M3.E4M3
```

NCU on the best eight-chain raw configurations reported 5,347,737,600 tensor
instructions for each launch. FP32 versus FP16 respectively measured 105.64
versus 76.60 ms profile duration, 99.2% versus 99.7% tensor-pipeline active,
6.42% versus 8.86% issue active, 0.34 versus 0.51 eligible warps per scheduler,
80.19% versus 95.83% achieved occupancy, and 42 versus 32 registers/thread.
CUDA-event timing remains authoritative because NCU replay changes timing.

The folded result retains a 13.7–14.4% gain after FP16-to-FP32 conversion and
persistent FP32 addition, while reducing this standalone kernel by 11
registers/thread. This is enough to justify an opt-in buffered attention
prototype, but not to predict a 1.14× attention speedup: the integrated kernel
has softmax, data movement, scheduling, and a much larger live register set,
so the prototype must still demonstrate lower total registers or lower tensor
dependency stalls and an end-to-end kernel gain.
