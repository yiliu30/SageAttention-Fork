# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SageAttention is a high-performance CUDA kernel library providing plug-and-play attention with INT8/FP8/FP4 quantization for GPU inference acceleration. It includes:
- ~~SageAttention/SageAttention2(deprecated)~~ (v2.2.0): INT8 QK + FP16/FP8 PV kernels for Ampere/Ada/Hopper GPUs (`sageattention/`)
- ~~SageAttention2++(deprecated)~~: Same codebase, uses `pv_accum_dtype="fp32+fp16"` two-level accumulation for higher speed
- **SageAttention3**: FP4 microscaling kernels for Blackwell GPUs only (`sageattention3_blackwell/`, separate package `sageattn3`)
    - cute version: `sageattention3_blackwell/`
    - triton version: A fake-quantization versio for fast test and experiment: `standalone/`

## Build and Development Commands

<!-- ### Build SageAttention2/2++ from source
```bash
# Editable install (preferred for development)
EXT_PARALLEL=16 NVCC_APPEND_FLAGS="--threads 32" MAX_JOBS=128 pip install -e . -v --no-build-isolation

# Or use the quick build script
./build_cmd.sh

# Or classic install
export EXT_PARALLEL=16 NVCC_APPEND_FLAGS="--threads 32" MAX_JOBS=128
python setup.py install
``` -->

### Build SageAttention3 (Blackwell only, separate package)
```bash
cd sageattention3_blackwell
# CUTLASS headers required - cloned automatically by setup.py if missing
pip install -e . --no-build-isolation
# Verify: python -c "from sageattn3 import sageattn3_blackwell; print('OK')"
```

### GPU Architecture Targeting
Build auto-detects GPUs. To cross-compile or override:
```bash
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;10.0;12.0;12.1"
```
Skip CUDA build entirely (CI/sdist): `SAGEATTN_SKIP_CUDA_BUILD=1`

### Python Environment
```bash
source /mnt/disk1/yiliu7/sage/bin/activate
```

<!-- ### Benchmarking
```bash
# SM89 (Ada) kernel benchmark
cd bench
python bench_qk_int8_pv_fp8_cuda.py --pv_accum_dtype fp32+fp16 --quant_gran per_warp

# End-to-end inference benchmark
cd example
python cogvideox_infer.py --model cogvideox-2b --compile --attention_type sage
# Sage3 on Blackwell:
python cogvideox_infer.py --model cogvideox-2b --compile --attention_type sage3
``` -->

### Standalone Tests
```bash
cd standalone
python test_standalone.py [--verbose] [--performance] [--accuracy-only]

# Sage3 Blackwell kernel demo
cd sageattention3_blackwell/examples
python sageattn3_demo.py
```


### Examples
#### Generate one flame for quick test
- sdpa
```bash
python cogvideox_infer.py --model cogvideox-2b --attention_type sdpa -q -i -n 1
```
- cute kernel
```bash
python cogvideox_infer.py --model cogvideox-2b --attention_type sage3 -q -i -n 1
```
- triton kernel
```bash
python cogvideox_infer.py --model cogvideox-2b --attention_type sage3_standalone -q -i -n 1
```

## Architecture Overview

<!-- ### Two Separate Packages, One Repo

**`sageattention` (v2.2.0)** — The main package for SM80-SM121:
- `sageattention/core.py`: **Entry point**. The `sageattn()` function auto-selects the optimal kernel by reading `torch.cuda.get_device_capability()`. This is the most important file for understanding how kernels are dispatched.
- `sageattention/quant.py`: CUDA-backed quantization utilities (per-block INT8, per-warp INT8, sub-mean, per-channel FP8). Wraps the `_fused` C extension.
- `sageattention/sm{80,89,90}_compile.py`: `torch.library.custom_op` wrappers around CUDA extensions (`_qattn_sm*`). These register kernels with `torch.compile` via fake implementations.
- `sageattention/triton/`: Triton-based attention kernels (non-causal, causal, varlen) and quantization kernels (per-block, per-thread). Used as fallback or for SM86.
- `sageattention/fa3_wrapper.py`: FlashAttention3 compatibility wrapper for benchmarking. -->

**`sageattn3` (v1.0.0)** — Blackwell-only package in `sageattention3_blackwell/`:
- `sageattn3/api.py`: High-level API (`sageattn3_blackwell()`), FP4 quantization functions, and Triton preprocessing kernels
- `sageattn3/blackwell/`: CUTLASS-based CUDA kernel (TMA + WGMMA, `api.cu` + headers)
- `sageattn3/quantization/`: FP4 quantization CUDA kernel
- Requires CUDA 12.8+, CUTLASS headers (auto-cloned to `csrc/cutlass/`)

<!-- ### Kernel Selection Logic (core.py `sageattn()`)

| GPU Arch | Kernel Path | Notes |
|----------|------------|-------|
| SM80 (A100) | `sageattn_qk_int8_pv_fp16_cuda` | INT8 QK, FP16 PV, FP32 accum |
| SM86 (A6000) | `sageattn_qk_int8_pv_fp16_triton` | Triton fallback |
| SM89 (RTX 4090) | `sageattn_qk_int8_pv_fp8_cuda` | FP8 PV, `fp32+fp16` accum (2++) |
| SM90 (H100) | `sageattn_qk_int8_pv_fp8_cuda_sm90` | WGMMA-optimized, `fp32+fp32` accum |
| SM100 (B200) | `sageattn_qk_int8_pv_fp16_cuda` | Falls back to SM80 path |
| SM120/121 (RTX 5090) | `sageattn_qk_int8_pv_fp8_cuda` | FP8 with per_warp, `fp32+fp16` | -->
<!-- 
### CUDA Extension Structure

The build produces these C extensions:
- `sageattention._fused`: Quantization and preprocessing kernels (all architectures)
- `sageattention._qattn_sm80`: INT8 QK + FP16 PV attention (SM80+)
- `sageattention._qattn_sm89`: INT8 QK + FP8 PV attention with inst_buf variants (SM89+)
- `sageattention._qattn_sm90`: Hopper-specific WGMMA attention (SM90 only)
- `fp4attn_cuda` / `fp4quant_cuda`: Sage3 Blackwell FP4 kernels (separate package) -->

### Key Quantization Concepts

- **Outlier smoothing**: Subtracting K's mean (`smooth_k`) before quantization reduces outlier impact
- **Quantization granularities**: `per_block` > `per_warp` > `per_thread` (finer = more accurate, slightly slower)
- **Two-level accumulation** (`fp32+fp16`): Short-term FP16 accumulator flushed to FP32 buffer every few iterations — this is what makes SageAttention2++ faster than plain FP32 accumulation
- **V smoothing**: For FP8 PV path, V is mean-subtracted and the correction is added back post-attention
- **Tensor layouts**: `"HND"` = `[B, H, N, D]`, `"NHD"` = `[B, N, H, D]`. Internally represented as int (0=NHD, 1=HND)

### Build System Specifics

- `BuildExtensionSeparateDir`: Custom build class in `setup.py` that isolates object files per-extension to prevent conflicts during parallel compilation
- `EXT_PARALLEL` controls extension-level parallelism (how many extensions compile simultaneously)
- `MAX_JOBS` controls file-level parallelism within each extension
- `NVCC_APPEND_FLAGS="--threads N"` controls NVCC internal parallelism

### Integration Patterns

**Global drop-in replacement** (works for most models):
```python
from sageattention import sageattn
import torch.nn.functional as F
F.scaled_dot_product_attention = sageattn
```

**Model-specific replacement** (recommended for DiT models):
See `example/modify_model/modify_mochi.py` for how to replace attention only in specific transformer blocks.

## Reference Materials

- SageAttention3 paper: `papers/sageattention/SageAttention3-2505.11594.pdf`
- Paper LaTeX source: `papers/sageattention/sage_v3/src/`
- Paper summaries: `papers/sageattention/paper_summaries.md`
- Research tasks/experiments: `tasks/` (pseudocode implementations, Triton reference kernels, benchmarks, analysis)

## Git Best Practices

- **Create small, focused commits** with a single clear purpose (feat:, test:, docs:, fix:, refactor:)
- Separate commits for implementation, tests, and documentation
