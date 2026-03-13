#!/usr/bin/env python3
"""
Test minimal two-level quantization.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def apply_nvfp4_e2m1_quantization_triton(x):
    """Apply NVFP4 E2M1 quantization."""
    x_abs = tl.abs(x)
    sign = tl.where(x >= 0.0, 1.0, -1.0)

    quantized_abs = tl.where(x_abs < 0.25, 0.0,
                    tl.where(x_abs < 0.625, 0.5,
                    tl.where(x_abs < 0.875, 0.75,
                    tl.where(x_abs < 1.25, 1.0,
                    tl.where(x_abs < 1.75, 1.5,
                    tl.where(x_abs < 2.5, 2.0,
                    tl.where(x_abs < 3.5, 3.0,
                    tl.where(x_abs < 5.0, 4.0, 6.0))))))))

    return quantized_abs * sign

@triton.jit
def minimal_quantization_triton(p_tile, BLOCK_N: tl.constexpr):
    """Minimal quantization with fixed BLOCK_N."""
    MICROSCALE_BLOCK_SIZE = 16

    # Level 1: Global scaling
    row_max = tl.max(tl.abs(p_tile), axis=1)
    global_scales = tl.maximum(row_max / 2688.0, 1e-8)
    p_level1 = p_tile / global_scales[:, None]

    # Level 2: Block structure
    col_indices = tl.arange(0, BLOCK_N)
    block_ids = col_indices // MICROSCALE_BLOCK_SIZE

    # Microscaling
    row_max_level2 = tl.max(tl.abs(p_level1), axis=1)
    base_microscale = tl.maximum(row_max_level2 / 6.0, 1e-8)
    microscale_rounded = tl.floor(base_microscale * 256.0 + 0.5) / 256.0

    # Block coefficients
    block_coefficient = 1.0 + 0.05 * tl.sin(block_ids.to(tl.float32) * 0.5)
    microscale_final = microscale_rounded[:, None] * block_coefficient[None, :]
    microscale_final = tl.maximum(microscale_final, 1e-8)

    # Apply
    p_microscaled = p_level1 / microscale_final
    p_quantized = apply_nvfp4_e2m1_quantization_triton(p_microscaled)
    p_final = p_quantized * microscale_final * global_scales[:, None]

    return p_final

@triton.jit
def test_minimal_kernel(input_ptr, output_ptr, M: tl.constexpr, N: tl.constexpr):
    """Test minimal quantization."""
    p_tile = tl.load(input_ptr + tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :])
    result = minimal_quantization_triton(p_tile, BLOCK_N=N)
    tl.store(output_ptr + tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :], result)

def test_minimal():
    """Test minimal quantization."""
    print("🧪 Testing minimal quantization")

    device = torch.device('cuda')
    M, N = 32, 128

    x = torch.randn(M, N, device=device, dtype=torch.float32) * 0.1
    y = torch.zeros_like(x)

    try:
        grid = (1,)
        test_minimal_kernel[grid](x, y, M=M, N=N)

        print("✅ Minimal quantization test successful")
        print(f"Input range: [{x.min().item():.6f}, {x.max().item():.6f}]")
        print(f"Output range: [{y.min().item():.6f}, {y.max().item():.6f}]")
        return True

    except Exception as e:
        print(f"❌ Minimal quantization test failed: {e}")
        return False

if __name__ == "__main__":
    test_minimal()