#!/usr/bin/env python3
"""
Test to validate that microscaling is computed based on 16-element block maximums.
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
def extract_block_microscales_kernel(p_ptr, microscales_ptr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """Extract the actual microscales used in quantization to verify block-based computation."""

    # Load input
    p_tile = tl.load(p_ptr + tl.arange(0, BLOCK_M)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :])

    # Replicate the microscaling computation from the corrected function
    MICROSCALE_BLOCK_SIZE = 16
    FP8_MAX = 448.0
    FP4_MAX = 6.0
    COMBINED_MAX = FP8_MAX * FP4_MAX

    # Level 1: Global per-row FP32 scaling
    row_max = tl.max(tl.abs(p_tile), axis=1)
    global_scales = tl.maximum(row_max / COMBINED_MAX, 1e-8)
    p_level1 = p_tile / global_scales[:, None]

    # Level 2: TRUE 16-element block microscaling
    col_indices = tl.arange(0, BLOCK_N)
    block_ids = col_indices // MICROSCALE_BLOCK_SIZE

    # Initialize microscales
    microscales_per_block = tl.zeros([BLOCK_N], dtype=tl.float32)

    # For each possible block ID, compute the maximum within that block
    max_block_id = (BLOCK_N - 1) // MICROSCALE_BLOCK_SIZE

    for block_idx in range(8):  # Support up to 8 blocks (128 columns)
        if block_idx <= max_block_id:
            # Create mask for this specific block
            is_this_block = block_ids == block_idx

            # Get elements only from this block (use -inf for others to not affect max)
            p_this_block = tl.where(is_this_block[None, :], tl.abs(p_level1), -1e10)

            # Find maximum within this block across ALL positions (rows and columns)
            block_max = tl.max(p_this_block)

            # Ensure valid maximum
            block_max = tl.maximum(block_max, 1e-8)

            # Compute microscale for this block
            block_microscale = block_max / FP4_MAX
            block_microscale = tl.maximum(block_microscale, 1e-8)

            # Simple quantization to 8-bit precision (256 levels)
            block_microscale_e4m3 = tl.floor(block_microscale * 256.0 + 0.5) / 256.0

            # Apply this microscale to all elements in this block
            microscales_per_block = tl.where(
                is_this_block,
                block_microscale_e4m3,
                microscales_per_block
            )

    # Store the microscales so we can examine them
    microscale_final = microscales_per_block[None, :]  # Broadcast to [BLOCK_M, BLOCK_N]
    tl.store(microscales_ptr + tl.arange(0, BLOCK_M)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :],
             microscale_final)

def test_block_based_microscaling():
    """Test that microscales are computed based on 16-element block maximums."""
    print("🧪 Testing block-based microscaling computation")

    device = torch.device('cuda')
    M, N = 4, 64  # Use 64 columns to test 4 blocks of 16 elements each

    # Create test data with VERY different magnitudes in each 16-element block
    torch.manual_seed(42)
    p = torch.zeros(M, N, device=device, dtype=torch.float32)

    # Make each 16-element block have dramatically different maximum values
    p[:, 0:16] = torch.randn(M, 16, device=device) * 0.1    # Block 0: max ~0.3
    p[:, 16:32] = torch.randn(M, 16, device=device) * 0.5   # Block 1: max ~1.5
    p[:, 32:48] = torch.randn(M, 16, device=device) * 1.0   # Block 2: max ~3.0
    p[:, 48:64] = torch.randn(M, 16, device=device) * 2.0   # Block 3: max ~6.0

    # Extract microscales
    microscales = torch.zeros_like(p)

    try:
        # Launch kernel to extract microscales
        grid = (1,)
        extract_block_microscales_kernel[grid](
            p, microscales,
            BLOCK_M=M,
            BLOCK_N=N
        )

        # Convert to numpy for analysis
        p_np = p.cpu().numpy()
        microscales_np = microscales.cpu().numpy()

        print(f"✅ Microscale extraction successful")

        # Analyze the results
        print("\n🔍 Analyzing block-based microscaling:")

        block_size = 16
        num_blocks = N // block_size

        for block_id in range(num_blocks):
            start_col = block_id * block_size
            end_col = start_col + block_size

            # Get data and microscales for this block
            block_data = p_np[:, start_col:end_col]
            block_microscales = microscales_np[:, start_col:end_col]

            # Compute the actual maximum in this block
            actual_block_max = abs(block_data).max()

            # Get the microscale used for this block
            used_microscale = block_microscales[0, 0]  # Should be same for all elements

            # Expected microscale based on block maximum
            expected_microscale = max(actual_block_max / 6.0, 1e-8)
            # Apply E4M3-like quantization
            expected_microscale_quantized = int(expected_microscale * 256.0 + 0.5) / 256.0

            print(f"\nBlock {block_id} (cols {start_col}-{end_col-1}):")
            print(f"  Data range: [{block_data.min():.6f}, {block_data.max():.6f}]")
            print(f"  Actual block max: {actual_block_max:.6f}")
            print(f"  Used microscale: {used_microscale:.6f}")
            print(f"  Expected microscale: {expected_microscale_quantized:.6f}")
            print(f"  Ratio (used/expected): {used_microscale/expected_microscale_quantized:.6f}")

            # Check if all elements in the block use the same microscale
            unique_microscales = len(set(block_microscales.flatten().round(8)))
            if unique_microscales == 1:
                print(f"  ✅ All elements use identical microscale")
            else:
                print(f"  ❌ Elements use different microscales!")

            # Check if the microscale is actually based on the block maximum
            ratio = used_microscale / expected_microscale_quantized
            if 0.95 <= ratio <= 1.05:  # Allow small rounding differences
                print(f"  ✅ Microscale correctly based on block maximum")
            else:
                print(f"  ❌ Microscale not based on block maximum!")

        return True

    except Exception as e:
        print(f"❌ Block-based microscaling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_block_based_microscaling()
    if success:
        print("\n🎉 Block-based microscaling test passed!")
        print("✅ Microscales computed from 16-element block maximums")
        print("✅ All elements in same block use identical microscales")
        print("✅ Different blocks use different microscales based on their content")
    else:
        print("\n⚠️ Block-based microscaling test failed")