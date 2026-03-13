# SageAttention3 Pure-Torch Implementation

This directory contains a pure PyTorch educational implementation of SageAttention3 with detailed comments and tensor shapes for study purposes.

## Overview

SageAttention3 introduces NVFP4 microscaling quantization for attention mechanisms on Blackwell GPUs. This implementation simulates the algorithm using PyTorch operations to help understand the core concepts.

## Key Features

- **NVFP4 E2M1 Quantization**: 4-bit quantization with 1×16 microscaling blocks
- **Two-Level P Scaling**: Optimized scaling strategy for probability matrices
- **Tiled Online Attention**: Memory-efficient processing without materializing full attention matrix
- **QK Smoothing**: Per-block mean subtraction for outlier handling
- **Educational Focus**: Heavy commenting with tensor shapes throughout

## Files

- `sageattn3_torch.py`: Main implementation with all algorithm components
- `demo_sageattn3.py`: Demonstration script with usage examples
- `README.md`: This documentation file

## Algorithm Components

### 1. Input Quantization
```python
# Quantize Q, K, V to NVFP4 E2M1 with 1×16 microscaling
q_fp4, q_scales = nvfp4_quantize(q_smoothed, block_size=(1, 16))
k_fp4, k_scales = nvfp4_quantize(k_smoothed, block_size=(1, 16))
v_fp4, v_scales = nvfp4_quantize(v, block_size=(1, 16))
```

### 2. QK Smoothing
```python
# Apply per-block mean subtraction to reduce outliers
q_smoothed, k_smoothed, q_correction = smooth_qk_tensors(q, k)
```

### 3. Tiled Online Attention
```python
# Process attention in tiles with running statistics
# Never materializes full P matrix
output, lse = tiled_online_attention(
    q_fp4, k_fp4, v_fp4, q_scales, k_scales, v_scales,
    sm_scale=sm_scale, is_causal=is_causal,
    tile_size_q=64, tile_size_k=64
)
```

### 4. Two-Level P Scaling
```python
# First level: scale to [0, 448×6] with FP32 per-token scales
p_scaled, first_level_scales = two_level_p_scaling(attention_scores)

# Second level: NVFP4 quantization with microscaling
p_fp4, p_fp4_scales = nvfp4_quantize(p_scaled, block_size=(1, 16))
```

## Usage

### Basic Usage
```python
from sageattn3_torch import sageattn3_torch

# Create input tensors [B, H, N, D]
q = torch.randn(1, 8, 128, 64, dtype=torch.float16, device='cuda')
k = torch.randn(1, 8, 128, 64, dtype=torch.float16, device='cuda')
v = torch.randn(1, 8, 128, 64, dtype=torch.float16, device='cuda')

# Run SageAttention3
output = sageattn3_torch(q, k, v,
                        per_block_mean=True,
                        is_causal=False,
                        tile_size_q=64,
                        tile_size_k=64)
```

### Using the Triton-Optimized Version
```python
from sage3_triton_wrapper import sage3_triton_sdpa_wrapper

# Drop-in replacement for F.scaled_dot_product_attention
output = sage3_triton_sdpa_wrapper(q, k, v, is_causal=False)

# Enable debug logging if needed
output = sage3_triton_sdpa_wrapper(q, k, v, is_causal=False, debug=True)
```

### Demo Script
```bash
# Run basic demo
python demo_sageattn3.py --seq-len 128 --heads 8 --head-dim 64

# Compare with PyTorch SDPA
python demo_sageattn3.py --compare-pytorch --verbose

# Test causal attention
python demo_sageattn3.py --causal --seq-len 256

# Show all options
python demo_sageattn3.py --help
```

### CogVideoX Integration
```bash
# Clean inference (no debug logs)
python cogvideox_infer.py --model cogvideox-2b --attention_type sage3_triton --smoke

# Debug mode (with verbose Triton logs) - set debug=True in wrapper if needed
# For debugging, modify sage3_triton_wrapper.py or add debug parameter support
```

## Implementation Notes

### NVFP4 E2M1 Format
- 4-bit floating point with 2 exponent bits, 1 mantissa bit, 1 sign bit
- Representable values: ±{0, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, ∞}
- Microscaling uses 1×16 blocks with FP8 E4M3 scale factors

### Memory Optimization
- Tiled processing reduces attention matrix memory from O(N²) to O(tile_size²)
- Online softmax maintains running statistics without storing full probability matrix
- FP4 quantization reduces memory usage by 4× compared to FP16

### Accuracy Considerations
- QK smoothing is critical for maintaining accuracy with aggressive quantization
- Two-level P scaling improves cosine similarity from 93.32% to 99.52%
- Q smoothing correction via GEMV maintains mathematical correctness

## Comparison with Real Kernel

This PyTorch implementation simulates the behavior of actual SageAttention3 CUDA kernels:

| Aspect | Real Kernel | PyTorch Implementation |
|--------|-------------|----------------------|
| Quantization | Hardware FP4 | Simulated via lookup tables |
| Matrix Ops | FP4MM instruction | torch.matmul on dequantized tensors |
| Memory | Optimized SRAM usage | Standard PyTorch memory |
| Performance | ~5x faster than FA2 | Educational speed (slower) |
| Accuracy | Hardware precision | Close approximation |

## Requirements

- PyTorch 2.3.0+
- CUDA 12.0+ (optional, can run on CPU)
- Python 3.8+

## Reference

Based on "SageAttention3: Microscaling FP4 Attention for Inference" (NeurIPS 2025)
- Paper: https://arxiv.org/abs/2505.11594
- Original implementation: https://github.com/thu-ml/SageAttention