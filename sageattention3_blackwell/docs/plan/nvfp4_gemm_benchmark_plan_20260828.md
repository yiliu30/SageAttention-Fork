# NVFP4 GEMM Benchmark Plan

## Goal

Add a large square NVFP4 GEMM peak reference to
`benchmarks/bench_sageattn3.py` and compare both SageAttention measurements
against it.

## Recommended Approach

Use PyTorch 2.9.1's native `torch._scaled_mm` on the RTX 5090 D:

- Allocate packed E2M1 operands directly as
  `torch.float4_e2m1fn_x2`.
- Allocate contiguous E4M3 block-scale buffers in the layout and size required
  by the native NVFP4 GEMM.
- Keep allocation and quantized-data preparation outside the timed region.
- Time only `_scaled_mm`, matching the existing BF16 GEMM reference policy.
- Report latency, dense GEMM TOPS (`2*M*N*K/time`), and attention ratios
  relative to both BF16 and NVFP4 GEMM.

This approach adds no dependency and measures the device's native NVFP4 GEMM
kernel rather than quantization overhead.

## Alternatives

1. Install TorchAO and quantize BF16 inputs with `nvfp4_quantize`. This provides
   meaningful numerical inputs but adds a dependency and does not improve the
   pure-kernel peak measurement.
2. Build CUTLASS Profiler and invoke it as a subprocess. This provides more
   kernel-selection control but complicates the Python benchmark and makes the
   result less directly comparable to PyTorch operations.

## Validation

- Reject dimensions incompatible with native NVFP4 packing/scaling.
- Run a small-shape smoke test that exercises `_scaled_mm`.
- Run the requested large GEMM benchmark on GPU 0.
- Confirm all reported ratios use the corresponding GEMM TOPS denominator.

## Scope

Modify the benchmark and its directly related benchmark plan only. No new
packages, compiled extensions, or system configuration are required.
