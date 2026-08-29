# SageAttention 3 Pure-Kernel NCU Bottleneck Analysis

## Profile

- GPU: NVIDIA GeForce RTX 5090 D
- Workload: batch 1, 40 heads, sequence 16384, head dimension 128, non-causal
- Kernel: `compute_attn_ws`
- Kernel duration under NCU: 7.36 ms
- Report: `~/.local/state/sageattention3/sageattn3_pure_16k.ncu-rep`

## Primary Bottleneck: Insufficient Latency Hiding

The kernel is limited primarily by low residency and a shortage of eligible
warps, not by DRAM bandwidth or peak tensor throughput.

NCU reports:

- 384 threads, or 12 warps, per CTA
- 168 registers per thread
- 100.35 KiB dynamic shared memory per CTA
- one resident CTA per SM
- 25.00% theoretical occupancy and 20.82% achieved occupancy
- 2.50 active warps and 0.55 eligible warps per scheduler
- no eligible warp in 60.20% of scheduler cycles
- one instruction issued every 2.5 cycles
- estimated local speedup from improving scheduler utilization: 54.99%

Both register use and shared-memory use independently limit residency to one
CTA. Reducing only one resource will therefore not increase occupancy.

The corresponding kernel choices are:

- `sageattn3/blackwell/kernel_ws.h:41`: 12-warp launch bound
- `sageattn3/blackwell/kernel_ws.h:166`: 232-register consumer warp-group
  allocation
- `sageattn3/blackwell/launch.h:105`: 128x128 tile with three pipeline stages
- `sageattn3/blackwell/kernel_traits.h:136-146`: triple-buffered K, V, and
  related shared-memory layouts

The most promising experiment is a coordinated lower-resource kernel variant,
not a single register-only or shared-memory-only change. Test a 64x128 query
tile with fewer consumer warps and two pipeline stages, with the explicit goal
of enabling two resident CTAs per SM. This can trade some per-CTA tensor
parallelism for enough additional ready warps to cover pipeline and
scoreboard latency. The variant must be benchmarked because a smaller tile may
increase scheduling, softmax, and epilogue overhead.

## Secondary Bottlenecks

### Shared-memory bank conflicts

NCU found 83,968,000 excessive shared-memory wavefronts, 8% of
1,008,435,200 total wavefronts, and estimates up to 7.983% speedup. The load
path accounts for nearly all conflicts:

- shared-load bank conflicts: 86,849,907
- shared-store bank conflicts: 31,156

The likely inspection points are the LDSM-based Q/K/V copies and their
swizzled layouts in `kernel_traits.h:129-142` and
`mainloop_tma_ws.h:622-660`. This is worthwhile after the residency variant,
or in parallel if source-correlated profiling is enabled.

### Scalar FP32 instruction mix

NCU estimates 3.251% speedup from replacing eligible non-fused FP32
instruction pairs with fused operations. This is materially smaller than the
residency and shared-memory opportunities.

## Ruled Out as Primary Limits

- DRAM bandwidth: 4.79% of peak
- overall memory throughput: 36.46% of peak
- tensor pipeline: 45.01% of peak
- local or shared-memory spilling: zero
- branch divergence: branch efficiency is 100%

The high L2 hit rate of 94.92% and low DRAM utilization further indicate that
off-chip bandwidth is not the main constraint.

## Priority

1. Create and benchmark a lower-resource 64x128, two-stage kernel variant that
   targets two resident CTAs per SM.
2. Remove shared-load bank conflicts in the Q/K/V shared-memory copy layouts.
3. Fuse the remaining eligible FP32 instruction pairs.

NCU's 54.99% estimate is a local upper-bound diagnostic, not a guaranteed
end-to-end gain. The measured pure-attention rate is already about 71% of the
large native NVFP4 GEMM reference, so a realistic first target is a 10-25%
kernel improvement rather than the full NCU estimate.
