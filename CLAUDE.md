# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SageAttention is a high-performance CUDA kernel library that provides plug-and-play attention implementations with INT8/FP8 quantization for GPU inference acceleration. The project includes multiple versions (SageAttention, SageAttention2, SageAttention2++, SageAttention3) with optimized kernels for different GPU architectures (Ampere, Ada, Hopper, Blackwell).

## Build and Development Commands

### Installation and Build
```bash
# Install from PyPI (latest stable)
pip install sageattention==2.2.0 --no-build-isolation

# Build from source (optimized for parallel compilation)
export EXT_PARALLEL=16 NVCC_APPEND_FLAGS="--threads 32" MAX_JOBS=128
python setup.py install

# Quick build script
./build_cmd.sh
```

### GPU Architecture Support
The build system auto-detects GPU compute capabilities or uses `TORCH_CUDA_ARCH_LIST`:
```bash
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;10.0;12.0;12.1"
```

### Testing and Benchmarking
```bash
# Benchmark against FlashAttention2/3
cd bench
python bench_qk_int8_pv_fp8_cuda.py --pv_accum_dtype fp32+fp16 --quant_gran per_warp

# Run example inference
cd example
python cogvideox_infer.py --model cogvideox-2b --compile --attention_type sage
```

## Architecture Overview

### Core Components

**Kernel Architecture**: Multi-backend system with CUDA and Triton implementations
- `csrc/qattn/`: CUDA kernels optimized for specific architectures (SM80, SM89, SM90)
- `sageattention/triton/`: Triton-based kernels for broader compatibility
- `sageattention/core.py`: Unified API that selects optimal kernels based on hardware

**Quantization Strategy**: Two-stage quantization approach
- QK^T: INT8 quantization with outlier smoothing (per-block, per-warp, per-thread granularities)
- PV: FP8 quantization with two-level accumulation strategy (FP32+FP16 for SageAttention2++)

**GPU-Specific Optimizations**:
- SM80/SM86 (Ampere): Base INT8/FP16 kernels
- SM89/Ada (RTX 40xx): FP8 support with CUDA 12.4+
- SM90/Hopper (H100): Native FP8 tensor cores and WGMMA
- SM100+/Blackwell (RTX 50xx): Advanced FP4 microscaling (SageAttention3)

### Module Structure

```
sageattention/
├── core.py              # Main API and kernel selection logic
├── quant.py            # CUDA quantization utilities
├── triton/             # Triton kernel implementations
│   ├── attn_*.py       # Attention kernels (causal, non-causal, varlen)
│   └── quant_*.py      # Quantization kernels (per-block, per-thread)
├── sm{80,89,90}_compile.py  # Architecture-specific kernel compilation
└── fa3_wrapper.py      # FlashAttention3 compatibility wrapper
```

### Key APIs

**Primary Interface**:
```python
from sageattention import sageattn
output = sageattn(q, k, v, tensor_layout="HND", is_causal=False)
```

**Specialized Kernels**:
- `sageattn_qk_int8_pv_fp16_cuda`: INT8 QK, FP16 PV (best accuracy)
- `sageattn_qk_int8_pv_fp8_cuda`: INT8 QK, FP8 PV (SageAttention2)
- `sageattn_qk_int8_pv_fp8_cuda_sm90`: Hopper-optimized version
- `sageattn_varlen`: Variable sequence length support

### Build System Details

**CUDA Compilation Pipeline**:
- Multi-threaded compilation with configurable parallelism (`EXT_PARALLEL`, `MAX_JOBS`)
- Architecture-specific extensions compiled conditionally based on target GPUs
- Custom `BuildExtensionSeparateDir` to prevent object file conflicts during parallel builds

**Dependencies**:
- CUDA 12.0+ (12.3+ for Hopper, 12.4+ for Ada FP8, 12.8+ for Blackwell)
- PyTorch 2.3.0+, Triton 3.0+
- Optional: `flash-attn` for benchmarking comparisons

### Integration Patterns

**Plug-and-play Replacement**:
```python
import torch.nn.functional as F
from sageattention import sageattn
F.scaled_dot_product_attention = sageattn  # Global replacement
```

**Model-specific Integration**: Modify attention processors in Diffusers models (see `example/modify_model/` for targeted replacements in transformer blocks)

**Distributed Inference**: Compatible with `torch.compile` (non-cudagraphs mode) and model parallelism frameworks like xDiT

## Development Notes

- The codebase prioritizes hardware-specific optimizations over generality
- Kernel selection in `core.py` uses runtime GPU detection and capability matching
- Quantization parameters (`quant_gran`, `pv_accum_dtype`) significantly impact performance/accuracy tradeoffs
- Two-level accumulation (`fp32+fp16`) is key to SageAttention2++ performance gains
- Variable-length attention (`sageattn_varlen`) requires different kernel paths for batched inference

### Test
- python envs: /mnt/disk1/yiliu7/sage/bin/python
- sage3 real kerenl demo: SageAttention-Fork/sageattention3_blackwell/examples/sageattn3_demo.py
