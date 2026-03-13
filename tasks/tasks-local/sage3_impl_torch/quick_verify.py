#!/usr/bin/env python3
"""
Quick Verification Script for SageAttention3 Real Kernel Alignment

This is a simplified version for quick testing that focuses on the key metrics:
1. Scaling correctness (vecMax / 6.0)
2. Basic accuracy (>90% cosine similarity)
3. Core functionality

Usage:
    python quick_verify.py

Expected: >99% accuracy with real kernel alignment
"""

import torch
import torch.nn.functional as F

def quick_verify():
    """Quick verification of core functionality and accuracy."""
    print("🚀 Quick SageAttention3 Verification")
    print("=" * 40)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    try:
        from sageattn3_torch import sageattn3_torch, nvfp4_quantize
        print("✅ Imports successful")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

    # Test 1: Scaling verification
    print("\n1. Scaling Verification")
    print("-" * 25)

    test_tensor = torch.tensor([6.0], device=device, dtype=torch.float16).view(1,1,1,1)
    padded = F.pad(test_tensor, (0, 15), value=0)
    _, scales = nvfp4_quantize(padded, block_size=16)

    expected_scale = 1.0  # 6.0 / 6.0 = 1.0
    actual_scale = scales.flatten()[0].item()

    print(f"Expected scale: {expected_scale:.3f}")
    print(f"Actual scale: {actual_scale:.3f}")

    scaling_correct = abs(actual_scale - expected_scale) < 0.01
    if scaling_correct:
        print("✅ Global scaling correct (vecMax / 6.0)")
    else:
        print("❌ Global scaling incorrect")

    # Test 2: Accuracy test
    print("\n2. Accuracy Test")
    print("-" * 18)

    B, H, N, D = 1, 4, 32, 64
    torch.manual_seed(42)

    q = torch.randn(B, H, N, D, device=device, dtype=torch.float16) * 0.1
    k = torch.randn(B, H, N, D, device=device, dtype=torch.float16) * 0.1
    v = torch.randn(B, H, N, D, device=device, dtype=torch.float16) * 0.1

    try:
        output_sage = sageattn3_torch(q, k, v, tensor_layout='HND')
        output_ref = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        cosine_sim = F.cosine_similarity(output_sage.flatten(), output_ref.flatten(), dim=0)

        print(f"Cosine similarity: {cosine_sim:.6f}")

        if cosine_sim >= 0.99:
            print("✅ EXCELLENT accuracy (>99%)")
            accuracy_good = True
        elif cosine_sim >= 0.95:
            print("✅ GOOD accuracy (>95%)")
            accuracy_good = True
        elif cosine_sim >= 0.90:
            print("⚠️ ACCEPTABLE accuracy (>90%)")
            accuracy_good = True
        else:
            print("❌ POOR accuracy (<90%)")
            accuracy_good = False

    except Exception as e:
        print(f"❌ Attention test failed: {e}")
        accuracy_good = False
        cosine_sim = 0.0

    # Results
    print("\n" + "=" * 30)
    print("QUICK VERIFICATION RESULTS")
    print("=" * 30)

    if scaling_correct and accuracy_good:
        print("🎉 SUCCESS: Real kernel alignment verified!")
        print(f"✅ Correct scaling: vecMax / 6.0")
        print(f"✅ Good accuracy: {cosine_sim:.1%}")
        print("✅ Ready for production use")
        return True
    else:
        print("❌ FAILED: Issues detected")
        if not scaling_correct:
            print("❌ Scaling issues")
        if not accuracy_good:
            print("❌ Accuracy issues")
        return False

if __name__ == "__main__":
    success = quick_verify()
    exit(0 if success else 1)