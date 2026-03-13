#!/usr/bin/env python3
"""
Test the simplest possible quantization.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def simple_test_kernel(input_ptr, output_ptr, M: tl.constexpr, N: tl.constexpr):
    """Simple test without calling complex functions."""
    # Load input
    p_tile = tl.load(input_ptr + tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :])

    # Simple operation without calling other functions
    result = p_tile * 2.0

    # Store output
    tl.store(output_ptr + tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :], result)

def test_simple():
    """Test simple kernel."""
    print("🧪 Testing simple kernel")

    device = torch.device('cuda')
    M, N = 32, 64

    x = torch.randn(M, N, device=device, dtype=torch.float32)
    y = torch.zeros_like(x)

    try:
        grid = (1,)
        simple_test_kernel[grid](x, y, M=M, N=N)

        print("✅ Simple test successful")
        print(f"Input range: [{x.min().item():.3f}, {x.max().item():.3f}]")
        print(f"Output range: [{y.min().item():.3f}, {y.max().item():.3f}]")
        return True

    except Exception as e:
        print(f"❌ Simple test failed: {e}")
        return False

if __name__ == "__main__":
    test_simple()