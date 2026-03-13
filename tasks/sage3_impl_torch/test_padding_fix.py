#!/usr/bin/env python3
"""
Test script to verify padding behavior in Triton implementation.
"""
import torch
import sys
import os

# Add path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_padding_behavior():
    print("🔍 Testing Padding Behavior in Triton Implementation")
    print("=" * 60)

    device = torch.device('cuda')
    dtype = torch.float16

    # Test various sequence lengths that require padding
    test_cases = [
        {"seq_len": 240, "expected_pad": 256, "name": "240->256"},
        {"seq_len": 17776, "expected_pad": 17792, "name": "17776->17792"},
        {"seq_len": 100, "expected_pad": 128, "name": "100->128"},
        {"seq_len": 256, "expected_pad": 256, "name": "256->256 (no pad)"},
    ]

    for i, case in enumerate(test_cases):
        seq_len = case["seq_len"]
        expected_pad = case["expected_pad"]
        name = case["name"]

        print(f"\n🧪 Test {i+1}: {name}")
        print("-" * 30)

        # Create test tensors
        B, H, D = 1, 8, 64
        q = torch.randn(B, H, seq_len, D, device=device, dtype=dtype) * 0.01
        k = torch.randn(B, H, seq_len, D, device=device, dtype=dtype) * 0.01
        v = torch.randn(B, H, seq_len, D, device=device, dtype=dtype) * 0.01

        print(f"Original shapes: Q={q.shape}, K={k.shape}, V={v.shape}")

        try:
            # Import and test
            from sageattn3_torch_triton import sageattn3_torch_triton

            output = sageattn3_torch_triton(
                q=q, k=k, v=v,
                tensor_layout="HND",
                is_causal=False,
                per_block_mean=True,
                debug=True
            )

            print(f"Final output shape: {output.shape}")
            print(f"Expected shape: {(B, H, seq_len, D)}")
            print(f"Shapes match: {'✅' if output.shape == (B, H, seq_len, D) else '❌'}")

            if output.shape[2] != seq_len:
                print(f"❌ PADDING BUG: Output seq_len {output.shape[2]} != input seq_len {seq_len}")
                return False
            else:
                print(f"✅ Padding handled correctly")

        except Exception as e:
            print(f"❌ Error: {e}")
            return False

    print(f"\n✅ All padding tests passed!")
    return True

if __name__ == "__main__":
    try:
        success = test_padding_behavior()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)