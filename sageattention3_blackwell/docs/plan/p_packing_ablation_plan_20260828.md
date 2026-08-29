# P Packing Ablation Plan

## Goal

Measure the exposed latency of the FP32-to-E2M1 conversion and packing used for
the softmax probability tensor P, without removing softmax, changing TMA
traffic, or changing the P-by-V tensor-core operation.

## Chosen approach

Add a diagnostic baseline-kernel instantiation that replaces only
`packed_float_to_e2m1` with an empty inline-assembly dependency sink and a
constant packed FP4 output. The dependency sink forces every softmax value to
remain live, while the constant output removes the E2M1 conversion instructions.
Both the normal and ablated kernels will be present in the same extension so
their CUDA-event timings can be interleaved.

This diagnostic mode will not be exposed as a normal high-level
`sageattn3_blackwell` variant. It will be available only through the low-level
benchmark path and clearly labeled as producing invalid attention results.

## Alternatives considered

1. Rebuild-time macro: smallest code change, but separate builds prevent a
   controlled interleaved comparison and can introduce compiler/build drift.
2. Same-binary compile-time ablation: moderate change, low measurement risk,
   and supports direct interleaved A/B timing. This is the selected approach.
3. `clock64` instrumentation only: useful as supporting evidence, but summed
   warp cycles do not measure the packing contribution to wall-clock latency.

## Implementation

1. Parameterize the mainloop's P packing call with a compile-time diagnostic
   flag.
2. In diagnostic mode, pass all eight FP32 inputs to an empty volatile inline
   assembly statement, then emit a constant packed FP4 word. Keep scale
   conversion and scale rearrangement unchanged.
3. Instantiate the diagnostic mode only for the baseline M128 x N128,
   head-dimension-128 kernel.
4. Add a private low-level dispatch flag with a default of false so existing
   Python and direct extension callers remain compatible.
5. Extend the benchmark with an interleaved pure-kernel comparison using the
   existing 15 warmups and 75 measured iterations.
6. Report normal latency, bypass latency, exposed packing latency, packing
   fraction, and speedup.

## Validation

1. Build the extension in the existing UV environment with `--no-deps`.
2. Confirm the normal path still produces the prior output and that the
   diagnostic path launches and returns finite data; diagnostic numerical
   accuracy is intentionally invalid.
3. Compare compiled resources for both kernels. Record registers, shared
   memory, threads, and launch grid.
4. Profile both variants with NCU and confirm equal QK and P-by-V tensor
   instruction counts, equal TMA bytes, equal CTA counts, and equal shared
   memory.
5. Run the 16K, 40-head, BF16 interleaved benchmark for at least 75 iterations.

## Interpretation

- Under 5% of normal latency: P conversion/packing is not the primary target.
- 5% to 10%: secondary optimization target.
- Over 10%: strong optimization target.

The timing delta includes the conversion instructions and the dependency stalls
they expose. It does not attribute cost to the final `mov.b32` separately,
because the native `cvt.e2m1x2.f32` instruction already combines conversion
with pair packing.

## Risk and rollback

The main risk is compiler drift caused by different register allocation or
instruction scheduling. Resource and NCU comparisons are required before
accepting the timing delta. The change is diagnostic-only and can be rolled
back by removing the extra instantiation and benchmark flag; it does not touch
persistent data or require credentials, external APIs, or new dependencies.
