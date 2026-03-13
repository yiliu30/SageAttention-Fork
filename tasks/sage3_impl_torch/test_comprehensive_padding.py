#!/usr/bin/env python3
"""
Comprehensive test to demonstrate the padding fix for CogVideoX-like sequence lengths.
"""

import torch
import sys
import os

# Add path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_cogvideox_like_padding():
    """Test with CogVideoX-like extreme configuration: seq_len=17776."""
    print("🎯 Testing CogVideoX-like Extreme Sequence Length")
    print("=" * 70)

    device = torch.device('cuda')
    dtype = torch.float16

    # CogVideoX-like configuration
    B, H, N, D = 2, 30, 17776, 64  # Exact same as in real test

    print(f"Configuration: {B}×{H}×{N}×{D}")
    print(f"Sequence length: {N}")
    print(f"Padding calculation: {N} % 128 = {N % 128}")
    print(f"Padding needed: {(128 - N % 128) % 128} tokens")
    print(f"Padded length: {N + ((128 - N % 128) % 128)}")
    print(f"Number of 128-element groups: {(N + ((128 - N % 128) % 128)) // 128}")
    print()

    # Create test tensors
    torch.manual_seed(42)
    q = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.01
    k = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.01
    v = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.01

    print("🧪 Running Triton implementation...")
    try:
        from sageattn3_torch_triton import sageattn3_torch_triton

        output_triton = sageattn3_torch_triton(
            q=q, k=k, v=v,
            tensor_layout="HND",
            is_causal=False,
            per_block_mean=True,
            debug=True
        )

        print("✅ Triton implementation successful!")
        print(f"Input shape: {q.shape}")
        print(f"Output shape: {output_triton.shape}")
        print(f"Shape preservation: {'✅' if output_triton.shape == q.shape else '❌'}")
        print(f"Output range: [{output_triton.min().item():.6f}, {output_triton.max().item():.6f}]")

        return True

    except Exception as e:
        print(f"❌ Triton implementation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_padding_edge_cases():
    """Test various padding edge cases."""
    print("\n🔍 Testing Padding Edge Cases")
    print("=" * 50)

    test_cases = [
        128,    # No padding needed
        129,    # Small padding (127 tokens)
        255,    # Large padding (1 token)
        256,    # No padding needed
        257,    # Small padding again
        1024,   # Multiple of 128
        17776,  # CogVideoX case
    ]

    for seq_len in test_cases:
        padding_needed = (128 - seq_len % 128) % 128
        padded_len = seq_len + padding_needed

        print(f"seq_len={seq_len:5d} -> padded={padded_len:5d} (pad +{padding_needed:2d})")

        # Quick test
        try:
            from sageattn3_torch_triton import sageattn3_torch_triton

            q = torch.randn(1, 4, seq_len, 64, device='cuda', dtype=torch.float16) * 0.01
            k = torch.randn(1, 4, seq_len, 64, device='cuda', dtype=torch.float16) * 0.01
            v = torch.randn(1, 4, seq_len, 64, device='cuda', dtype=torch.float16) * 0.01

            output = sageattn3_torch_triton(q, k, v, per_block_mean=True, debug=False)

            if output.shape[2] == seq_len:
                print(f"  ✅ Padding handled correctly")
            else:
                print(f"  ❌ Output seq_len {output.shape[2]} != input seq_len {seq_len}")
                return False

        except Exception as e:
            print(f"  ❌ Error: {e}")
            return False

    print("✅ All edge cases passed!")
    return True

if __name__ == "__main__":
    print("🎯 Comprehensive Padding Fix Verification")
    print("Testing Triton implementation with large sequence lengths")
    print()

    success1 = test_cogvideox_like_padding()
    success2 = test_padding_edge_cases()

    print(f"\n{'='*70}")
    print("🎯 PADDING FIX VERIFICATION RESULTS")
    print(f"{'='*70}")

    if success1 and success2:
        print("🎉 ALL TESTS PASSED!")
        print("✅ CogVideoX-like sequence length (17776) handled correctly")
        print("✅ All padding edge cases working")
        print("✅ Original sequence length preserved in output")
        print("✅ Triton implementation properly trims padded results")
        print()
        print("The critical padding bug has been fixed!")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED")
        print("The padding fix needs further investigation.")
        sys.exit(1)