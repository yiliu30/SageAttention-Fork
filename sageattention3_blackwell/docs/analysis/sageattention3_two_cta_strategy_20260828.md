# SageAttention 3 Two-CTA Strategy

## Objective

Increase the pure FP4 attention kernel from one to two resident CTAs per SM on
the RTX 5090 D, improving the number of active and eligible warps available to
hide pipeline latency.

## Current Kernel

The head-dimension-128 kernel instantiated in `blackwell/launch.h` uses:

- tile M = 128
- tile N = 128
- head dimension K = 128
- three K/V pipeline stages
- 12 warps, or 384 threads, per CTA
- one producer warp-group and two consumer warp-groups
- 24 registers requested for producer threads
- 232 registers requested for consumer threads
- a persistent grid fixed at 170 CTAs, exactly one CTA per SM

The compiled and measured resources are:

| Resource | Kernel use | RTX 5090 D per-SM limit |
|---|---:|---:|
| Registers | 168/thread, about 64,512/CTA | 65,536 |
| Dynamic shared memory | 100,352 bytes/CTA | 102,400 bytes |
| Driver shared memory | 1,024 bytes/CTA | included in the SM limit |
| Threads | 384/CTA | 1,536 |

The kernel therefore consumes nearly the entire register file and the maximum
opt-in shared-memory allocation. Even if the scheduler launched more blocks,
only one could reside on each SM.

The persistent scheduler is a second independent restriction:
`StaticPersistentTileScheduler::get_grid_dim()` always returns 170 blocks.
Resource reduction without increasing this grid still leaves one CTA per SM.

## Resource Targets

For two CTAs per SM:

- total registers per CTA must be at most 32,768
- dynamic shared memory per CTA must be at most 50,176 bytes after reserving
  the 1,024-byte driver allocation for each CTA
- use a practical dynamic-shared-memory target of at most 48 KiB

With 384 threads, the register limit is about 85 registers/thread, which is not
realistic for this kernel. The CTA must shrink to 256 threads, raising the
hard average limit to 128 registers/thread.

## Candidate Comparison

The table below counts the main Q/K/V, scale, delta, and output shared-memory
payloads. The current kernel has 98,816 bytes of payload and 100,352 bytes of
measured dynamic shared memory, so layout, barrier, and alignment overhead is
approximately 1.5 KiB.

| Tile M | Tile N | Stages | Payload | Predicted dynamic shared memory | Two-CTA result |
|---:|---:|---:|---:|---:|---|
| 128 | 128 | 3 | 98,816 B | 100,352 B measured | impossible |
| 128 | 128 | 2 | 79,872 B | about 81.5 KiB | impossible |
| 64 | 128 | 2 | 58,880 B | about 59 KiB | impossible |
| 64 | 64 | 3 | 49,408 B | about 50 KiB | too large after driver overhead |
| 64 | 64 | 2 | 39,936 B | about 40.5 KiB | feasible |

The selected first candidate is therefore M64 x N64 with two stages.

## Why the Candidate Can Meet the Register Target

M64 selects eight warps in `kernel_traits.h`, producing one 128-thread
producer warp-group and one 128-thread consumer warp-group. Reducing N from
128 to 64 halves the score accumulator and temporary quantized-P fragments
per consumer thread. The output accumulator remains the same size because
M and the number of consumer threads both halve.

Start with:

- producer `setmaxnreg`: 24
- consumer `setmaxnreg`: 208
- launch bound: 256 threads and two minimum blocks per SM

The requested weighted average is `(24 + 208) / 2 = 116` registers/thread.
Allowing for compiler overhead observed in the current binary gives a
prediction near 121 registers/thread, below the 128-register hard limit.
This is only a prediction; the compiled binary must be the acceptance test.

## Required Code Changes

### 1. Parameterize the kernel configuration

In `blackwell/launch.h`, retain the current M128/N128/stage-3 kernel and add an
experimental M64/N64/stage-2 instantiation for head dimension 128. Do not
replace the baseline until the candidate passes correctness and performance
gates.

Add trait constants for:

- minimum CTAs per SM
- producer register allocation
- consumer register allocation

Use these constants in `kernel_ws.h` rather than fixed values 1, 24, and 232.

### 2. Generalize accumulator initialization

`mainloop_tma_ws.h` currently initializes `delta_s` with a manually unrolled
layout that assumes the M128/N128 score fragment. Replace this with a
tile-derived C-fragment partition or add a separate M64/N64 specialization.
This is the main correctness risk of changing tile shape.

The remaining mainloop code is mostly shape-derived:

- K/V iteration uses `kBlockN`
- causal bounds use `kBlockM` and `kBlockN`
- score and scale fragments derive from `TileShape_MNK`
- the softmax row count remains two for the selected candidate

### 3. Launch enough persistent CTAs

Change the persistent grid from `num_sm` to
`num_sm * KernelTraits::kMinBlocksPerSm`. For the candidate this is 340 CTAs.
The existing scheduler stride through work is `gridDim.x`, so no algorithmic
scheduler change is required.

At sequence length 16,384 and 40 heads:

- baseline: 5,120 M tiles distributed over 170 CTAs
- candidate: 10,240 M tiles distributed over 340 CTAs

Both configurations assign approximately 30 M tiles to each persistent CTA.

### 4. Preserve an A/B path

Compile both variants and select them through a temporary benchmark-only
switch. This permits correctness and latency comparisons in the same build
and provides immediate rollback.

## Acceptance Gates

### Compile-time and resource gates

Inspect the candidate with ptxas, `cuobjdump`, and NCU:

- no local-memory spills
- compiled register use at most 128 registers/thread
- dynamic shared memory at most 50,176 bytes/CTA, preferably at most 48 KiB
- NCU `Block Limit Registers` at least 2
- NCU `Block Limit Shared Mem` at least 2
- two active CTAs per SM

If either resource limit remains one block, stop; changing the grid alone
cannot improve residency.

### Correctness gates

Compare candidate output with the current FP4 kernel and BF16 SDPA for:

- non-causal and causal attention
- sequence lengths around tile boundaries: 63, 64, 65, 127, 128, and 129
- sequence lengths 8,192 and 16,384
- head dimension 128
- deterministic random inputs and adversarial high-magnitude inputs

Track maximum absolute error, RMSE, cosine similarity, and NaN/Inf counts.
Tile-order changes can alter FP4 rounding, so bitwise equality is not
expected.

### Performance gates

Use the saved pure-kernel benchmark with identical inputs:

- median of at least 75 timed iterations
- candidate must improve pure-kernel latency by at least 5%
- NCU eligible warps per scheduler must exceed the current 0.55
- NCU no-eligible cycles must fall below the current 60.20%
- tensor-pipeline utilization should exceed the current 45.01%

The main 16K target is a 10-25% reduction from the current roughly 7.0 ms.

## Expected Tradeoffs

The candidate doubles the number of M tiles and N-loop iterations. It
therefore increases scheduler, softmax-loop, and epilogue frequency while
reducing per-CTA tensor work. Two-CTA residency only wins if improved latency
hiding is larger than this extra overhead.

If M64/N64/stage-2 reaches two CTAs but is slower, keep the current kernel and
optimize the measured shared-load bank conflicts instead. NCU estimates that
path at about 8%, and it does not require changing tile-level numerical order.

## Conclusion

Do not begin by lowering only the current `warpgroup_reg_alloc<232>` value.
The present shared-memory allocation would still force one CTA per SM and a
lower register cap could introduce spills with no concurrency benefit.

The smallest coherent experiment is:

1. M64/N64 tile
2. two pipeline stages
3. 256 threads
4. producer/consumer register targets of 24/208
5. 340-block persistent grid
6. generalized `delta_s` fragment initialization

All six changes are required to make the two-CTA hypothesis testable.

## Experiment Results

The opt-in candidate was implemented and profiled on the RTX 5090 D. The
baseline remains the default.

### Resource and scheduler results

| Metric | Baseline | M64/N64 two-CTA |
|---|---:|---:|
| Threads/CTA | 384 | 256 |
| Persistent grid | 170 | 340 |
| Registers/thread | 168 | 128 |
| Dynamic shared memory/CTA | 100.35 KB | 41.98 KB |
| Register block limit | 1 | 2 |
| Shared-memory block limit | 1 | 2 |
| Theoretical occupancy | 25.00% | 33.33% |
| Achieved occupancy | 20.82% | 23.73% |
| Active warps/scheduler | 2.50 | 2.85 |
| Eligible warps/scheduler | 0.55 | 0.76 |
| Cycles with no eligible warp | 60.20% | 48.27% |
| Compute throughput | 45.01% | 50.62% |

The candidate therefore achieved the intended two-CTA resource limits and
improved latency hiding. NCU also measured 130,480 local-memory spill
instructions. A diagnostic build without the two-block launch bound used 208
registers/thread, showing that the 128-register residency ceiling forces the
candidate below its natural consumer register demand. The spill instructions
are only about 0.0024% of the candidate's 5.51 billion executed instructions,
so they are a secondary symptom rather than the primary performance cost.

### 16K performance results

The final benchmark used 75 timed iterations with B=1, H=40, S=16,384, D=128,
and BF16 inputs.

| Method | Median | Effective TOPS |
|---|---:|---:|
| Baseline end-to-end | 8.385 ms | 655.6 |
| Two-CTA end-to-end | 10.024 ms | 548.4 |
| Baseline pure FP4 kernel | 7.027 ms | 782.3 |
| Two-CTA pure FP4 kernel | 7.867 ms | 698.8 |

The candidate is 12.0% slower for the pure kernel and 19.5% slower
end-to-end. It fails the required 5% improvement gate despite better scheduler
metrics. The added tile, softmax, epilogue, and scheduling work, combined with
duplicated K/V movement, costs more than the extra latency hiding saves.

### Why the candidate is slower

The full NCU reports identify duplicated data movement and per-tile control
work as the dominant costs:

| Work metric | Baseline | Two-CTA | Change |
|---|---:|---:|---:|
| Global TMA read bytes | 12.46 GB | 24.88 GB | +99.6% |
| TMA load instructions | 3.29 million | 13.13 million | +299.4% |
| Shared-load instructions | 125.91 million | 241.25 million | +91.6% |
| Branch instructions | 34.73 million | 111.60 million | +221.3% |
| Total executed instructions | 4.54 billion | 5.51 billion | +21.4% |
| L2 throughput | 26.16% | 73.12% | +179.5% |
| Overall memory throughput | 36.46% | 73.12% | +100.5% |

Halving M doubles the number of query tiles. Each query tile still traverses
the complete K/V sequence, so total K/V TMA traffic nearly doubles. Halving N
also doubles the number of K/V loop iterations; combined with twice as many M
tiles, this produces approximately four times as many TMA load instructions.
The data loaded per TMA instruction is smaller, which is why bytes double
rather than quadruple.

The additional CTA does improve issue efficiency:

- executed IPC rises from 1.52 to 2.01
- eligible warps/scheduler rise from 0.55 to 0.76
- math-pipe-throttle samples fall substantially
- compute throughput rises from 45.01% to 50.62%

Those gains are real, but achieved occupancy reaches only 23.73%, not the
33.33% theoretical value, and average active warps rise only from 9.99 to
11.39 per SM. The kernel therefore receives 14% more active warps while doing
21% more instructions and nearly twice the TMA data movement.

NCU's individual report durations show 7.36 ms for the baseline and 7.13 ms
for the candidate, but that replay-derived timing conflicts with repeated
CUDA-event measurements. Reversing benchmark order still gives a candidate
speedup of only 0.90x; baseline-first gives 0.893x; and an interleaved run gives
0.927x. The slowdown is therefore not launch-order or thermal bias. Use the
75-iteration CUDA-event result for latency and use NCU for workload attribution,
not cross-report timing.

### Decision

Keep M128/N128/stage-3 as the default. The two-CTA variant remains opt-in for
experimentation and profiling, but is not suitable as the production default.
The next optimization target should be the baseline shared-memory bank
conflicts. If the two-CTA design is continued, the most direct response to the
profile is pairing adjacent M64 CTAs and multicasting each K/V TMA tile across
the pair, so the second query tile does not reload the same K/V data. This
requires a cluster-aware scheduler and TMA multicast path and is a larger
architectural change than reducing register use alone.

Candidate NCU report:
`~/.local/state/sageattention3/sageattn3_pure_16k_two_cta.ncu-rep`.
