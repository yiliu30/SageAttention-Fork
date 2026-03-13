#!/usr/bin/env python3
"""
Debug the E4M3 rounding function.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def round_to_e4m3_triton(scale):
    """Round scale to E4M3 precision using Triton's float8 support."""
    scale_fp8 = scale.to(tl.float8e4nv)  # Triton's E4M3 float8 type
    return scale_fp8.to(tl.float32)      # Convert back to FP32 for computation

@triton.jit
def test_e4m3_kernel(input_ptr, output_ptr, N: tl.constexpr):
    """Test E4M3 rounding."""
    idx = tl.arange(0, N)
    x = tl.load(input_ptr + idx)
    y = round_to_e4m3_triton(x)
    tl.store(output_ptr + idx, y)

def test_e4m3_rounding():
    """Test E4M3 rounding in isolation."""
    print("🧪 Testing E4M3 rounding function")

    device = torch.device('cuda')
    N = 16

    # Test data
    x = torch.linspace(0.1, 2.0, N, device=device, dtype=torch.float32)
    y = torch.zeros_like(x)

    try:
        grid = (1,)
        test_e4m3_kernel[grid](x, y, N=N)

        print("✅ E4M3 rounding test successful")
        print(f"Input:  {x.tolist()}")
        print(f"Output: {y.tolist()}")
        return True

    except Exception as e:
        print(f"❌ E4M3 rounding test failed: {e}")
        return False

if __name__ == "__main__":
    test_e4m3_rounding()