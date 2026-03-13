@triton.jit
def two_level_p_quantization_triton_fixed(p_tile):
    """
    FIXED: Two-level P quantization with proper 16-element microscaling blocks.

    Level 1: FP8 global scale per attention row (max = 448)
    Level 2: FP4 microscaling per 16-element block (max = 6)
    Combined scale factor: 448 × 6 = 2688

    Args:
        p_tile: [BLOCK_M, BLOCK_N] attention probabilities

    Returns:
        p_quantized: [BLOCK_M, BLOCK_N] quantized probabilities
    """
    FP8_MAX = 448.0
    FP4_MAX = 6.0
    COMBINED_SCALE = FP8_MAX * FP4_MAX  # 2688
    MICROSCALE_BLOCK_SIZE = 16

    BLOCK_M = p_tile.shape[0]
    BLOCK_N = p_tile.shape[1]

    # Level 1: Global FP8 scaling per attention row
    row_max = tl.max(tl.abs(p_tile), axis=1)  # [BLOCK_M]
    global_fp8_scales = tl.maximum(row_max / COMBINED_SCALE, 1e-8)

    # Apply level 1 scaling
    p_level1_scaled = p_tile / global_fp8_scales[:, None]

    # Level 2: FP4 microscaling per 16-element blocks along K dimension
    # Number of 16-element blocks per row
    num_microscale_blocks = (BLOCK_N + MICROSCALE_BLOCK_SIZE - 1) // MICROSCALE_BLOCK_SIZE

    # Initialize output
    p_quantized = tl.zeros_like(p_tile)

    # Process each row
    for m in range(BLOCK_M):
        # Process each 16-element block in this row
        for block_idx in range(num_microscale_blocks):
            block_start = block_idx * MICROSCALE_BLOCK_SIZE
            block_end = tl.minimum(block_start + MICROSCALE_BLOCK_SIZE, BLOCK_N)

            if block_start < BLOCK_N:
                # Extract 16-element block (or remaining elements)
                block_indices = block_start + tl.arange(0, MICROSCALE_BLOCK_SIZE)
                block_mask = block_indices < block_end

                # Get values for this block
                p_block = tl.where(
                    block_mask,
                    p_level1_scaled[m, block_indices],
                    0.0
                )

                # Compute microscale for this 16-element block
                block_max = tl.max(tl.abs(p_block))
                microscale_fp4_scale = tl.maximum(block_max / FP4_MAX, 1e-8)

                # Apply FP4 microscaling
                p_microscaled = p_block / microscale_fp4_scale

                # Apply FP4 quantization (using the existing function)
                p_block_quantized = apply_nvfp4_e2m1_quantization_triton(p_microscaled)

                # Reconstruct with microscaling
                p_block_final = p_block_quantized * microscale_fp4_scale * global_fp8_scales[m]

                # Store back in output tensor
                p_quantized = tl.where(
                    block_mask[None, :] & (tl.arange(0, BLOCK_M)[:, None] == m),
                    p_block_final[None, :],
                    p_quantized
                )

    return p_quantized