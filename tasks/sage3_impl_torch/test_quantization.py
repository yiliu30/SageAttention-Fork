#!/usr/bin/env python3
"""
Test the two-level P quantization function separately.
"""

import torch
import triton
import triton.language as tl
from sageattn3_torch_triton import (
    round_to_e4m3_triton,
    apply_nvfp4_e2m1_quantization_triton,
    two_level_p_quantization_triton
)

@triton.jit
def test_quantization_kernel(p_ptr, out_ptr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """Test kernel for quantization."""
    # Load input
    p_tile = tl.load(p_ptr + tl.arange(0, BLOCK_M)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :])

    # Apply quantization
    p_quantized = two_level_p_quantization_triton(p_tile, BLOCK_N)

    # Store output
    tl.store(out_ptr + tl.arange(0, BLOCK_M)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :], p_quantized)

def test_quantization():
    """Test the quantization function."""
    print("🧪 Testing two-level P quantization")

    device = torch.device('cuda')
    B, H, N, D = 1, 8, 128, 64

    # Create test data
    torch.manual_seed(42)
    p = torch.randn(128, 128, device=device, dtype=torch.float32) * 0.1

    # Test simple quantization
    output = torch.zeros_like(p)

    try:
        # Launch test kernel
        grid = (1,)
        test_quantization_kernel[grid](
            p, output,
            BLOCK_M=128,
            BLOCK_N=128
        )

        print(f"✅ Quantization test successful")
        print(f"Input range: [{p.min().item():.6f}, {p.max().item():.6f}]")
        print(f"Output range: [{output.min().item():.6f}, {output.max().item():.6f}]")

        return True

    except Exception as e:
        print(f"❌ Quantization test failed: {e}")
        return False

if __name__ == "__main__":
    test_quantization()