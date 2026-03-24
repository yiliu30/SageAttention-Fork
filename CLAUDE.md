# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Fork of [thu-ml/SageAttention](https://github.com/thu-ml/SageAttention) — **focused exclusively on SageAttention3 and its variants**. The v1/v2 code (`sageattention/`, `csrc/qattn/`) is inherited from upstream but not our active development area; ignore it unless explicitly asked.

**SageAttention3** — FP4 microscaling attention for Blackwell GPUs:
- **CUTE kernel** (`sageattention3_blackwell/`, package `sageattn3`): CUTLASS-based CUDA kernel (TMA + WGMMA), production target
- **Triton kernel** (`standalone/`): Fake-quantization implementation for fast experimentation and iteration

Current experiment variants (see `example/sage3_next.md`):
- `sage3` — CUTE kernel reference
- `sage3_standalone` — Triton NVFP4
- `sage3_standalone_mxfp4` / `sage3_standalone_mxfp4_s1` — MXFP4 variants (accuracy improvement is the key focus)
- `sage3_standalone_mxfp8_s1` — MXFP8 (comparable accuracy to NVFP4)

## Build and Development Commands

### Python Environment
```bash
# B200:
source /home/yiliu7/workspace/envs/sage-local/bin/activate
# Other machines:
source /mnt/disk1/yiliu7/sage/bin/activate
```

### Build SageAttention3 (Blackwell only)
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

### Standalone Tests
```bash
cd standalone
python test_standalone.py [--verbose] [--performance] [--accuracy-only]
```

### Quick E2E Regression Test
```bash
cd example
bash test_quick_e2e.sh
# Compares sage3_standalone (triton) vs sage3 (CUTE kernel) via PSNR and cosine similarity
# Env vars: PSNR_THRESHOLD (default 20), COS_THRESHOLD (default 0.95), NUM_FRAMES (default 1)
```

### Example Inference (single frame, quick test)
```bash
cd example
python cogvideox_infer.py --model cogvideox-2b --attention_type sage3_standalone -q -i -n 1
python cogvideox_infer.py --model cogvideox-2b --attention_type sage3 -q -i -n 1
python cogvideox_infer.py --model cogvideox-2b --attention_type sdpa -q -i -n 1
```

## Architecture Overview

**`sageattn3`** — Blackwell-only package in `sageattention3_blackwell/`:
- `sageattn3/api.py`: High-level API (`sageattn3_blackwell()`), FP4 quantization functions, and Triton preprocessing kernels
- `sageattn3/blackwell/`: CUTLASS-based CUDA kernel (TMA + WGMMA, `api.cu` + headers)
- `sageattn3/quantization/`: FP4 quantization CUDA kernel
- Requires CUDA 12.8+, CUTLASS headers (auto-cloned to `csrc/cutlass/`)

**Triton implementation** (`standalone/`):
- `sageattention3_standalone.py`: Main Triton kernel — the primary file for experimenting with new quantization schemes
- `test_standalone.py`: Accuracy and performance tests

### Key Quantization Concepts (Sage3 focus)

- **Microscaling FP4 (MXFP4)**: Block-level FP4 quantization with shared exponents — main accuracy challenge vs NVFP4
- **MXFP8**: Higher-precision variant, accuracy comparable to NVFP4 — useful as a baseline or mixed-precision component
- **Outlier smoothing**: Subtracting K's mean (`smooth_k`) before quantization reduces outlier impact
- **V smoothing**: V is mean-subtracted and the correction is added back post-attention
- **Tensor layouts**: `"HND"` = `[B, H, N, D]`, `"NHD"` = `[B, N, H, D]`. Internally represented as int (0=NHD, 1=HND)

### Build System Specifics (for sage3 CUTE kernel)

- Requires CUDA 12.8+ and SM100 (Blackwell)
- CUTLASS headers auto-cloned to `sageattention3_blackwell/csrc/cutlass/` on first build

### Integration (example inference scripts)

The `example/` directory has plug-and-play inference scripts for diffusion models (CogVideoX, WAN, HunyuanVideo, Mochi, LTX-Video). They accept `--attention_type` to swap between `sdpa`, `sage3`, `sage3_standalone`, and variants.

## Reference Materials

- SageAttention3 paper: `papers/sageattention/SageAttention3-2505.11594.pdf`
- Paper LaTeX source: `papers/sageattention/sage_v3/src/`
- Research tasks/experiments: `tasks/`
- Current experiment status: `example/sage3_next.md`

## Git Conventions

- **Branch naming**: `exp/<experiment-name>` for experiment branches
- **Commit style**: Small, focused commits with prefixes: `feat:`, `test:`, `docs:`, `fix:`, `refactor:`
- Separate commits for implementation, tests, and documentation
