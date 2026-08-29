# SageAttention 3 Benchmark Plan

## Goal

Save a reproducible benchmark for SageAttention 3 NVFP4, PyTorch Flash SDPA,
and a materialized PyTorch QK transpose matrix multiplication.

## Decisions

- Default to the requested BF16, batch-1, 40-head, 8192-token workload on
  CUDA device 0 with head dimension 128.
- Time GPU work with CUDA events after a configurable warm-up period.
- Report full-attention throughput as QK transpose plus PV matrix
  multiplication work, using two FLOPs per fused multiply-add.
- Report the QK reference separately because it materializes the full
  attention-score tensor, unlike the fused attention implementations.
- Restrict head dimensions to the two NVFP4 kernel variants, 64 and 128, and
  use triangular FLOP accounting when causal attention is selected.
- Prepare the QK reference key tensor in transposed, contiguous form before
  timing, so the measurement contains only the per-head matrix multiplication.
- Include an independent 8192-cubed BF16 GEMM as the GPU peak reference; it
  intentionally does not model attention dimensions.
- Include a native square NVFP4 GEMM using packed E2M1 operands and E4M3
  block scales. Prepare operands and scales outside the timed region.
- Report both attention methods relative to the BF16 and NVFP4 GEMM TOPS and
  their direct ratio. The attention rates are effective operation rates.
- Report an additional SageAttention 3 FP4-kernel-only result after preparing
  the preprocessed and quantized Q/K/V inputs outside the timed region.
- Keep an Nsight Compute harness and shell launcher in `benchmarks/`; the
  launcher starts profiling after preprocessing and quantization complete.

## Validation

Run the script in the NVFP4 environment and confirm all attention and GEMM
methods produce timing and throughput results on GPU 0.
