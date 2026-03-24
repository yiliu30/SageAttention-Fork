---
name: bench
description: Run kernel benchmarks for SageAttention3 and baselines. Use to measure kernel performance (TFLOPS) for different attention configurations.
---

Run kernel benchmarks to compare SageAttention3 against baselines.

## Primary Benchmarks (Sage3 focus)

**Sage3 CUTE kernel vs baselines** (`bench/bench_baseline.py`):
```bash
cd bench
# Sage3 Blackwell kernel
python bench_baseline.py --method sage3 --head_dim 128
# FA2 baseline
python bench_baseline.py --method fa2 --head_dim 128
# PyTorch SDPA baseline
python bench_baseline.py --method torch --head_dim 128
```
Options: `--method {sage3,fa2,torch,xformers}`, `--batch_size`, `--num_heads`, `--head_dim`

**Sage3 demo with correctness check** (`sageattention3_blackwell/examples/sageattn3_demo.py`):
```bash
cd sageattention3_blackwell/examples
python sageattn3_demo.py --batch 1 --heads 8 --q-len 1024 --head-dim 128
```

## Legacy v2 Benchmarks (in `bench/`, use only if explicitly asked)

- `bench_qk_int8_pv_fp8_cuda.py` — SM89 INT8 QK + FP8 PV
- `bench_qk_int8_pv_fp8_cuda_sm90.py` — SM90 INT8 QK + FP8 PV (Hopper)
- `bench_qk_int8_pv_fp16_cuda.py` / `bench_qk_int8_pv_fp16_triton.py` — INT8 QK + FP16 PV
- `bench_fa3.py` / `bench_fa3_fp8.py` — FlashAttention3

## Usage

Parse `$ARGUMENTS` for options:
- Kernel/method: `sage3`, `fa2`, `torch`, `xformers` (default: run sage3 and fa2 for comparison)
- `--head_dim` (default: 128)
- `--batch_size` (default: 2)
- `--num_heads` (default: 32)

If no arguments given, run `sage3` and `fa2` back-to-back for a quick comparison.

After completion, summarize the TFLOPS results in a table grouped by sequence length, with both causal and non-causal results.
