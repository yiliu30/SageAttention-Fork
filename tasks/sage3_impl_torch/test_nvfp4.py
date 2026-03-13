#!/usr/bin/env python3
"""
Test NVFP4 quantization function.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def apply_nvfp4_e2m1_quantization_triton(x):
    """Apply NVFP4 E2M1 quantization."""
    x_abs = tl.abs(x)
    sign = tl.where(x >= 0.0, 1.0, -1.0)

    # Find nearest FP4 E2M1 level
    quantized_abs = tl.where(x_abs < 0.25, 0.0,
                    tl.where(x_abs < 0.625, 0.5,
                    tl.where(x_abs < 0.875, 0.75,
                    tl.where(x_abs < 1.25, 1.0,
                    tl.where(x_abs < 1.75, 1.5,
                    tl.where(x_abs < 2.5, 2.0,
                    tl.where(x_abs < 3.5, 3.0,
                    tl.where(x_abs < 5.0, 4.0,
                                          6.0))))))))

    return quantized_abs * sign

@triton.jit
def test_nvfp4_kernel(input_ptr, output_ptr, M: tl.constexpr, N: tl.constexpr):
    """Test NVFP4 quantization."""
    p_tile = tl.load(input_ptr + tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :])
    result = apply_nvfp4_e2m1_quantization_triton(p_tile)
    tl.store(output_ptr + tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :], result)

def test_nvfp4():
    """Test NVFP4 quantization."""
    print("🧪 Testing NVFP4 quantization")

    device = torch.device('cuda')
    M, N = 32, 64

    x = torch.randn(M, N, device=device, dtype=torch.float32) * 2.0
    y = torch.zeros_like(x)

    try:
        grid = (1,)
        test_nvfp4_kernel[grid](x, y, M=M, N=N)

        print("✅ NVFP4 test successful")
        print(f"Input range: [{x.min().item():.3f}, {x.max().item():.3f}]")
        print(f"Output range: [{y.min().item():.3f}, {y.max().item():.3f}]")
        return True

    except Exception as e:
        print(f"❌ NVFP4 test failed: {e}")
        return False

if __name__ == "__main__":
    test_nvfp4()