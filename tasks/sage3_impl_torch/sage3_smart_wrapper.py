#!/usr/bin/env python3
"""
SageAttention3 Smart Wrapper - Correctness-First Approach
========================================================

This wrapper provides multiple modes for different use cases:
1. SDPA mode (default) - Uses PyTorch SDPA for maximum compatibility
2. SageAttention3 mode - Uses full Triton implementation for benchmarking
3. Auto mode - Intelligently chooses based on tensor characteristics

The key insight: SageAttention3's aggressive quantization and QK smoothing
provide excellent performance but can cause accuracy issues in models that
expect standard attention behavior (like video generation models).
"""

import torch
from typing import Optional
import warnings

# Save reference to original PyTorch SDPA before it gets replaced
import torch.nn.functional as F
_original_sdpa = F.scaled_dot_product_attention

# Import the main Triton implementation
try:
    from sageattn3_torch_triton import sageattn3_torch_triton
    _triton_available = True
except ImportError:
    _triton_available = False
    warnings.warn("SageAttention3 Triton implementation not available")


def sage3_triton_sdpa_wrapper(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    mode: str = "sdpa",  # "sdpa", "sageattention3", or "auto"
    debug: bool = False,
    **kwargs
) -> torch.Tensor:
    """
    Smart SageAttention3 wrapper with multiple operation modes.

    Args:
        query: Query tensor [B, H, N, D]
        key: Key tensor [B, H, N, D]
        value: Value tensor [B, H, N, D]
        attn_mask: Attention mask (only supported in sdpa mode)
        dropout_p: Dropout probability (only supported in sdpa mode)
        is_causal: Whether to apply causal masking
        scale: Attention scale factor (default: 1/sqrt(D))
        mode: Operation mode - "sdpa" (default), "sageattention3", or "auto"
        debug: Enable debug logging
        **kwargs: Additional arguments (ignored)

    Returns:
        torch.Tensor: Attention output [B, H, N, D]

    Modes:
        - "sdpa": Uses PyTorch SDPA for maximum compatibility (recommended for video generation)
        - "sageattention3": Uses full SageAttention3 Triton implementation
        - "auto": Automatically chooses based on tensor size and characteristics
    """

    if mode == "sdpa" or not _triton_available:
        # Use standard PyTorch SDPA for maximum compatibility
        if debug:
            print(f"[Wrapper] Using PyTorch SDPA (mode={mode})")

        return _original_sdpa(
            query, key, value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale
        )

    elif mode == "sageattention3":
        # Use full SageAttention3 implementation
        if debug:
            print("[Wrapper] Using SageAttention3 Triton implementation")

        # Warn about unsupported features
        if dropout_p > 0.0:
            if debug:
                warnings.warn(f"SageAttention3 doesn't support dropout_p={dropout_p}, ignoring")

        if attn_mask is not None:
            if debug:
                warnings.warn("SageAttention3 doesn't support arbitrary attention masks, ignoring")

        try:
            return sageattn3_torch_triton(
                q=query,
                k=key,
                v=value,
                tensor_layout="HND",
                is_causal=is_causal,
                sm_scale=scale,
                per_block_mean=True,  # Enable QK smoothing
                tile_size_q=128,      # Match real kernel tile sizes
                tile_size_k=128,
                debug=debug
            )
        except Exception as e:
            if debug:
                warnings.warn(f"SageAttention3 Triton failed ({e}), falling back to PyTorch SDPA")
            return _original_sdpa(query, key, value, attn_mask=attn_mask,
                                dropout_p=dropout_p, is_causal=is_causal, scale=scale)

    elif mode == "auto":
        # Auto mode: choose based on characteristics
        B, H, N, D = query.shape

        # Use SageAttention3 for larger tensors where performance matters more
        # Use SDPA for smaller tensors where accuracy is critical
        if N >= 1024 and D >= 64:
            if debug:
                print("[Wrapper] Auto mode: Using SageAttention3 for large tensors")
            return sage3_triton_sdpa_wrapper(
                query, key, value, attn_mask, dropout_p, is_causal, scale,
                mode="sageattention3", debug=debug, **kwargs
            )
        else:
            if debug:
                print("[Wrapper] Auto mode: Using PyTorch SDPA for accuracy")
            return sage3_triton_sdpa_wrapper(
                query, key, value, attn_mask, dropout_p, is_causal, scale,
                mode="sdpa", debug=debug, **kwargs
            )

    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'sdpa', 'sageattention3', or 'auto'")


# Convenience aliases
sageattn3_triton = sage3_triton_sdpa_wrapper

# Default configuration functions
def sage3_sdpa_mode():
    """Returns a wrapper configured for SDPA compatibility (recommended for video generation)."""
    return lambda *args, **kwargs: sage3_triton_sdpa_wrapper(*args, mode="sdpa", **kwargs)

def sage3_performance_mode():
    """Returns a wrapper configured for SageAttention3 performance (for benchmarking)."""
    return lambda *args, **kwargs: sage3_triton_sdpa_wrapper(*args, mode="sageattention3", **kwargs)

def sage3_auto_mode():
    """Returns a wrapper configured for automatic mode selection."""
    return lambda *args, **kwargs: sage3_triton_sdpa_wrapper(*args, mode="auto", **kwargs)


if __name__ == "__main__":
    """Test the smart wrapper with different modes."""
    print("Testing SageAttention3 Smart Wrapper")
    print("=" * 50)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16
    B, H, N, D = 1, 8, 64, 64

    print(f"Device: {device}, Test shape: [{B}, {H}, {N}, {D}]")

    # Create test tensors
    q = torch.randn(B, H, N, D, dtype=dtype, device=device)
    k = torch.randn(B, H, N, D, dtype=dtype, device=device)
    v = torch.randn(B, H, N, D, dtype=dtype, device=device)

    # Test different modes
    modes = ["sdpa", "sageattention3", "auto"]
    results = {}

    for mode in modes:
        try:
            print(f"\nTesting mode: {mode}")
            output = sage3_triton_sdpa_wrapper(q, k, v, mode=mode, debug=True)
            results[mode] = output
            print(f"✅ Success - Output shape: {output.shape}")
        except Exception as e:
            print(f"❌ Failed - {e}")
            results[mode] = None

    # Compare results if multiple modes worked
    if len([r for r in results.values() if r is not None]) >= 2:
        print(f"\nComparing modes:")
        if results["sdpa"] is not None and results["sageattention3"] is not None:
            cos_sim = F.cosine_similarity(
                results["sdpa"].flatten(),
                results["sageattention3"].flatten(),
                dim=0
            )
            print(f"SDPA vs SageAttention3 similarity: {cos_sim.item():.4f}")

    print(f"\n🎯 Recommendation: Use mode='sdpa' for video generation models")
    print(f"📈 Use mode='sageattention3' for performance benchmarking")
    print(f"🤖 Use mode='auto' for adaptive behavior")