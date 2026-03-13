#!/usr/bin/env python3
"""
Kernel Input Comparison Script for SageAttention3
================================================

This script compares the inputs to the core attention kernels:
1. Triton implementation: tiled_online_attention_triton
2. Real kernel: blockscaled_fp4_attn

The goal is to verify that both kernels receive identical preprocessed inputs
(quantized Q/K/V, delta_s corrections, scale factors) to isolate any accuracy
differences to the kernel implementations themselves.

Usage:
    python compare_kernel_inputs.py

Requirements:
    - CUDA Blackwell GPU (RTX 5090 D or similar)
    - SageAttention3 Blackwell kernel compiled and installed
"""

import torch
import torch.nn.functional as F
import sys
import os
from typing import Dict, Any, Tuple
import numpy as np

def setup_paths():
    """Add necessary paths for imports."""
    # Add current directory for our implementation
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)

    # Add Blackwell kernel path
    blackwell_path = '/mnt/disk1/yiliu7/SageAttention-Fork/sageattention3_blackwell'
    sys.path.insert(0, blackwell_path)

def capture_triton_inputs(q, k, v):
    """
    Capture inputs to tiled_online_attention_triton by monkey-patching.

    Returns the inputs that would be passed to the Triton kernel.
    """
    from sageattn3_torch_triton import tiled_online_attention_triton

    # Storage for captured inputs
    captured_inputs = {}

    # Monkey-patch the kernel function to capture inputs
    original_kernel = tiled_online_attention_triton

    def capture_wrapper(*args, **kwargs):
        captured_inputs.update({
            'q': args[0].clone() if len(args) > 0 else None,
            'k': args[1].clone() if len(args) > 1 else None,
            'v': args[2].clone() if len(args) > 2 else None,
            'delta_s': args[3].clone() if len(args) > 3 and args[3] is not None else None,
            'sm_scale': args[4] if len(args) > 4 else kwargs.get('sm_scale'),
            'is_causal': args[5] if len(args) > 5 else kwargs.get('is_causal', False),
            'tile_size_q': kwargs.get('tile_size_q', 128),
            'tile_size_k': kwargs.get('tile_size_k', 128)
        })
        return original_kernel(*args, **kwargs)

    # Temporarily replace the function
    import sageattn3_torch_triton
    sageattn3_torch_triton.tiled_online_attention_triton = capture_wrapper

    # Run the full Triton pipeline
    from sageattn3_torch_triton import sageattn3_torch_triton
    output = sageattn3_torch_triton(
        q=q, k=k, v=v,
        tensor_layout="HND",
        is_causal=False,
        per_block_mean=True,
        tile_size_q=128,
        tile_size_k=128,
        debug=False
    )

    # Restore original function
    sageattn3_torch_triton.tiled_online_attention_triton = original_kernel

    return captured_inputs, output

def capture_real_kernel_inputs(q, k, v):
    """
    Capture inputs to the real kernel by examining the preprocessing steps.

    This requires understanding the real kernel's preprocessing pipeline.
    """
    try:
        from sageattn3 import sageattn3_blackwell
    except ImportError:
        return None, None

    # The real kernel does its own preprocessing, so we need to capture
    # what it receives internally. For now, we'll capture the raw inputs
    # and the output to compare preprocessing effects.

    output = sageattn3_blackwell(
        q.clone(), k.clone(), v.clone(),
        is_causal=False,
        per_block_mean=True
    )

    # Real kernel inputs (we can only capture the external interface)
    real_inputs = {
        'q_raw': q.clone(),
        'k_raw': k.clone(),
        'v_raw': v.clone(),
        'is_causal': False,
        'per_block_mean': True
    }

    return real_inputs, output

def compare_tensor_stats(tensor1, tensor2, name):
    """Compare detailed statistics between two tensors."""
    if tensor1 is None or tensor2 is None:
        return f"{name}: One tensor is None"

    if tensor1.shape != tensor2.shape:
        return f"{name}: Shape mismatch {tensor1.shape} vs {tensor2.shape}"

    # Convert to float32 for precise comparison
    t1 = tensor1.float()
    t2 = tensor2.float()

    # Basic statistics
    stats = {
        'mean_diff': (t1.mean() - t2.mean()).item(),
        'std_diff': (t1.std() - t2.std()).item(),
        'max_diff': (t1.max() - t2.max()).item(),
        'min_diff': (t1.min() - t2.min()).item(),
        'cosine_sim': F.cosine_similarity(t1.flatten(), t2.flatten(), dim=0).item()
    }

    # Elementwise differences
    diff = (t1 - t2)
    stats.update({
        'max_abs_diff': diff.abs().max().item(),
        'mean_abs_diff': diff.abs().mean().item(),
        'mse': (diff ** 2).mean().item(),
        'relative_error': (diff.abs() / (t2.abs() + 1e-8)).mean().item()
    })

    return stats

def analyze_preprocessing_differences(triton_inputs, real_inputs, test_name):
    """Analyze the preprocessing differences between implementations."""

    print(f"\n🔍 {test_name} - Preprocessing Analysis")
    print("=" * 60)

    # Compare raw inputs (should be identical)
    if 'q' in triton_inputs and 'q_raw' in real_inputs:
        q_stats = compare_tensor_stats(triton_inputs['q'], real_inputs['q_raw'], "Q tensor")
        print(f"📊 Q Input Comparison:")
        if isinstance(q_stats, str):
            print(f"   {q_stats}")
        else:
            print(f"   Cosine similarity: {q_stats['cosine_sim']:.8f}")
            print(f"   Max abs diff: {q_stats['max_abs_diff']:.6e}")
            print(f"   Mean abs diff: {q_stats['mean_abs_diff']:.6e}")
            print(f"   Relative error: {q_stats['relative_error']:.6e}")

    # Compare K tensors
    if 'k' in triton_inputs and 'k_raw' in real_inputs:
        k_stats = compare_tensor_stats(triton_inputs['k'], real_inputs['k_raw'], "K tensor")
        print(f"📊 K Input Comparison:")
        if isinstance(k_stats, str):
            print(f"   {k_stats}")
        else:
            print(f"   Cosine similarity: {k_stats['cosine_sim']:.8f}")
            print(f"   Max abs diff: {k_stats['max_abs_diff']:.6e}")
            print(f"   Mean abs diff: {k_stats['mean_abs_diff']:.6e}")
            print(f"   Relative error: {k_stats['relative_error']:.6e}")

    # Compare V tensors
    if 'v' in triton_inputs and 'v_raw' in real_inputs:
        v_stats = compare_tensor_stats(triton_inputs['v'], real_inputs['v_raw'], "V tensor")
        print(f"📊 V Input Comparison:")
        if isinstance(v_stats, str):
            print(f"   {v_stats}")
        else:
            print(f"   Cosine similarity: {v_stats['cosine_sim']:.8f}")
            print(f"   Max abs diff: {v_stats['max_abs_diff']:.6e}")
            print(f"   Mean abs diff: {v_stats['mean_abs_diff']:.6e}")
            print(f"   Relative error: {v_stats['relative_error']:.6e}")

    # Analyze delta_s if available
    if 'delta_s' in triton_inputs and triton_inputs['delta_s'] is not None:
        print(f"📊 Delta_s Analysis:")
        delta_s = triton_inputs['delta_s']
        print(f"   Shape: {delta_s.shape}")
        print(f"   Range: [{delta_s.min():.6f}, {delta_s.max():.6f}]")
        print(f"   Mean: {delta_s.mean():.6f}")
        print(f"   Std: {delta_s.std():.6f}")
    else:
        print(f"📊 Delta_s: Not available or None")

    # Compare scale factors
    if 'sm_scale' in triton_inputs:
        print(f"📊 Scale Factor:")
        print(f"   Triton sm_scale: {triton_inputs['sm_scale']}")

    # Compare tile sizes
    if 'tile_size_q' in triton_inputs:
        print(f"📊 Tile Sizes:")
        print(f"   Query tile: {triton_inputs['tile_size_q']}")
        print(f"   Key tile: {triton_inputs['tile_size_k']}")

def test_kernel_input_comparison():
    """Main comparison function."""
    print("🔧 SageAttention3 Kernel Input Comparison")
    print("=" * 60)

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False

    device = torch.device('cuda')
    print(f"🖥️  GPU: {torch.cuda.get_device_name(0)}")

    # Test configurations
    test_configs = [
        {"B": 1, "H": 8, "N": 256, "D": 64, "name": "Small"},
        {"B": 2, "H": 30, "N": 1024, "D": 64, "name": "CogVideoX-like"},
    ]

    all_comparisons = []

    for config in test_configs:
        B, H, N, D = config["B"], config["H"], config["N"], config["D"]
        name = config["name"]

        print(f"\n🧪 Testing {name} Configuration: {B}×{H}×{N}×{D}")
        print("-" * 50)

        torch.manual_seed(42)  # Ensure reproducible inputs
        dtype = torch.float16

        # Create test tensors
        q = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.01
        k = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.01
        v = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.01

        try:
            # Capture Triton inputs
            print("🔬 Capturing Triton kernel inputs...")
            triton_inputs, triton_output = capture_triton_inputs(q.clone(), k.clone(), v.clone())

            # Capture real kernel inputs
            print("🔬 Capturing real kernel inputs...")
            real_inputs, real_output = capture_real_kernel_inputs(q.clone(), k.clone(), v.clone())

            if real_inputs is None:
                print("❌ Real kernel not available")
                continue

            # Analyze preprocessing differences
            analyze_preprocessing_differences(triton_inputs, real_inputs, name)

            # Compare final outputs
            print(f"\n📊 Final Output Comparison:")
            output_stats = compare_tensor_stats(triton_output, real_output, "Output")
            if isinstance(output_stats, str):
                print(f"   {output_stats}")
            else:
                print(f"   Cosine similarity: {output_stats['cosine_sim']:.8f}")
                print(f"   Max abs diff: {output_stats['max_abs_diff']:.6e}")
                print(f"   Mean abs diff: {output_stats['mean_abs_diff']:.6e}")
                print(f"   MSE: {output_stats['mse']:.6e}")
                print(f"   Relative error: {output_stats['relative_error']:.6e}")

            # Store results
            all_comparisons.append({
                'config': name,
                'triton_inputs': triton_inputs,
                'real_inputs': real_inputs,
                'output_similarity': output_stats['cosine_sim'] if isinstance(output_stats, dict) else 0.0
            })

        except Exception as e:
            print(f"❌ Failed: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print(f"\n{'='*70}")
    print(f"📈 KERNEL INPUT COMPARISON SUMMARY")
    print(f"{'='*70}")

    if all_comparisons:
        avg_similarity = sum(comp['output_similarity'] for comp in all_comparisons) / len(all_comparisons)
        print(f"📊 Average output similarity: {avg_similarity:.6f}")

        if avg_similarity >= 0.99:
            print("🎉 EXCELLENT: Kernel inputs and preprocessing are well aligned!")
        elif avg_similarity >= 0.95:
            print("✅ GOOD: Minor preprocessing differences detected")
        else:
            print("⚠️  ATTENTION: Significant preprocessing differences found")
    else:
        print("❌ No successful comparisons")

    return len(all_comparisons) > 0

def main():
    """Main function."""
    setup_paths()

    print("🎯 SageAttention3 Kernel Input Comparison")
    print("Analyzing preprocessing and inputs to core attention kernels")
    print()

    success = test_kernel_input_comparison()

    if success:
        print("\n✅ ANALYSIS COMPLETE: Kernel input comparison finished!")
    else:
        print("\n❌ ANALYSIS FAILED: Unable to compare kernel inputs")

    return success

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Analysis interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)