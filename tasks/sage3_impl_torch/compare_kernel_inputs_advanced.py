#!/usr/bin/env python3
"""
Advanced Kernel Input Comparison Script for SageAttention3
=========================================================

This script performs deep inspection of the actual inputs that reach the
core attention computation kernels by hooking into the preprocessing pipeline
and capturing the quantized tensors, scaling factors, and other parameters
that are passed to the lowest-level kernel functions.

Key comparisons:
1. Quantized Q, K, V tensors after educational quantization
2. Delta_s QK smoothing corrections
3. Scale factors and quantization parameters
4. Tile processing parameters

Usage:
    python compare_kernel_inputs_advanced.py
"""

import torch
import torch.nn.functional as F
import sys
import os
from typing import Dict, Any, Tuple, Optional
import numpy as np
from contextlib import contextmanager

def setup_paths():
    """Add necessary paths for imports."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    blackwell_path = '/mnt/disk1/yiliu7/SageAttention-Fork/sageattention3_blackwell'
    sys.path.insert(0, blackwell_path)

class KernelInputCapture:
    """Context manager to capture kernel inputs through monkey patching."""

    def __init__(self):
        self.captured_data = {}
        self.original_functions = {}

    def capture_triton_pipeline(self, q, k, v):
        """Capture the complete Triton preprocessing pipeline."""
        print("🔍 Capturing Triton preprocessing pipeline...")

        # Import modules
        import sageattn3_torch
        import sageattn3_torch_triton

        # Storage for captured intermediate values
        self.captured_data['triton'] = {}

        # Hook into apply_qk_smoothing
        original_smoothing = sageattn3_torch.apply_qk_smoothing
        def capture_smoothing(*args, **kwargs):
            print("   📎 Capturing QK smoothing...")
            result = original_smoothing(*args, **kwargs)
            self.captured_data['triton']['qk_smoothing'] = {
                'q_smoothed': result[0].clone(),
                'k_smoothed': result[1].clone(),
                'delta_s': result[2].clone() if result[2] is not None else None
            }
            print(f"      Q smoothed shape: {result[0].shape}, range: [{result[0].min():.6f}, {result[0].max():.6f}]")
            print(f"      K smoothed shape: {result[1].shape}, range: [{result[1].min():.6f}, {result[1].max():.6f}]")
            if result[2] is not None:
                print(f"      Delta_s shape: {result[2].shape}, range: [{result[2].min():.6f}, {result[2].max():.6f}]")
            return result

        # Hook into educational_quantize
        original_quantize = sageattn3_torch.educational_quantize
        quantize_call_count = 0
        def capture_quantize(tensor):
            nonlocal quantize_call_count
            print(f"   📎 Capturing quantization call {quantize_call_count}...")
            result = original_quantize(tensor)

            tensor_name = ['q', 'k', 'v'][quantize_call_count % 3]
            if 'quantized' not in self.captured_data['triton']:
                self.captured_data['triton']['quantized'] = {}

            self.captured_data['triton']['quantized'][tensor_name] = {
                'input': tensor.clone(),
                'output': result.clone(),
                'input_range': [tensor.min().item(), tensor.max().item()],
                'output_range': [result.min().item(), result.max().item()],
                'quantization_error': (tensor - result).abs().mean().item()
            }
            print(f"      {tensor_name.upper()} quantization error: {(tensor - result).abs().mean().item():.6e}")
            quantize_call_count += 1
            return result

        # Hook into tiled_online_attention_triton
        original_kernel = sageattn3_torch_triton.tiled_online_attention_triton
        def capture_kernel(*args, **kwargs):
            print("   📎 Capturing kernel inputs...")
            self.captured_data['triton']['kernel_inputs'] = {
                'q': args[0].clone() if len(args) > 0 else None,
                'k': args[1].clone() if len(args) > 1 else None,
                'v': args[2].clone() if len(args) > 2 else None,
                'delta_s': args[3].clone() if len(args) > 3 and args[3] is not None else None,
                'sm_scale': args[4] if len(args) > 4 else kwargs.get('sm_scale'),
                'is_causal': args[5] if len(args) > 5 else kwargs.get('is_causal', False),
                'tile_size_q': kwargs.get('tile_size_q', 128),
                'tile_size_k': kwargs.get('tile_size_k', 128)
            }
            # Print kernel input summary
            for name, tensor in [('q', args[0]), ('k', args[1]), ('v', args[2])]:
                if tensor is not None:
                    print(f"      Kernel {name.upper()}: shape={tensor.shape}, dtype={tensor.dtype}, range=[{tensor.min():.6f}, {tensor.max():.6f}]")
            return original_kernel(*args, **kwargs)

        # Apply patches
        sageattn3_torch.apply_qk_smoothing = capture_smoothing
        sageattn3_torch.educational_quantize = capture_quantize
        sageattn3_torch_triton.tiled_online_attention_triton = capture_kernel

        try:
            # Run the Triton implementation
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
            self.captured_data['triton']['output'] = output.clone()

        finally:
            # Restore original functions
            sageattn3_torch.apply_qk_smoothing = original_smoothing
            sageattn3_torch.educational_quantize = original_quantize
            sageattn3_torch_triton.tiled_online_attention_triton = original_kernel

        return output

    def capture_real_kernel_pipeline(self, q, k, v):
        """Capture what we can from the real kernel pipeline."""
        print("🔍 Capturing real kernel pipeline...")

        try:
            from sageattn3 import sageattn3_blackwell

            # Store the raw inputs
            self.captured_data['real'] = {
                'raw_inputs': {
                    'q': q.clone(),
                    'k': k.clone(),
                    'v': v.clone()
                },
                'parameters': {
                    'is_causal': False,
                    'per_block_mean': True
                }
            }

            # Run the real kernel
            output = sageattn3_blackwell(
                q.clone(), k.clone(), v.clone(),
                is_causal=False,
                per_block_mean=True
            )
            self.captured_data['real']['output'] = output.clone()

            print(f"   📎 Real kernel output shape: {output.shape}, range: [{output.min():.6f}, {output.max():.6f}]")
            return output

        except ImportError:
            print("   ❌ Real kernel not available")
            return None

def compare_quantization_effects(triton_data, real_data):
    """Compare the quantization effects between implementations."""
    print(f"\n🔬 Quantization Analysis")
    print("=" * 50)

    if 'quantized' in triton_data:
        print("📊 Triton Quantization Effects:")
        for tensor_name, data in triton_data['quantized'].items():
            error = data['quantization_error']
            input_range = data['input_range']
            output_range = data['output_range']
            print(f"   {tensor_name.upper()}:")
            print(f"      Input range:  [{input_range[0]:.6f}, {input_range[1]:.6f}]")
            print(f"      Output range: [{output_range[0]:.6f}, {output_range[1]:.6f}]")
            print(f"      Quantization error: {error:.6e}")

            # Compare input vs output statistics
            input_tensor = data['input']
            output_tensor = data['output']
            cos_sim = F.cosine_similarity(input_tensor.flatten(), output_tensor.flatten(), dim=0)
            print(f"      Pre/Post quantization similarity: {cos_sim.item():.6f}")

    print(f"\n📊 Real Kernel (Black Box):")
    print(f"   Input preprocessing: Internal (not visible)")
    print(f"   Quantization: Hardware-optimized FP4/FP8")

def compare_qk_smoothing(triton_data, real_data):
    """Compare QK smoothing effects."""
    print(f"\n🔬 QK Smoothing Analysis")
    print("=" * 50)

    if 'qk_smoothing' in triton_data:
        smoothing = triton_data['qk_smoothing']

        print("📊 Triton QK Smoothing:")
        print(f"   Q smoothing applied: {'Yes' if smoothing['q_smoothed'] is not None else 'No'}")
        print(f"   K smoothing applied: {'Yes' if smoothing['k_smoothed'] is not None else 'No'}")
        print(f"   Delta_s correction: {'Yes' if smoothing['delta_s'] is not None else 'No'}")

        if smoothing['delta_s'] is not None:
            delta_s = smoothing['delta_s']
            print(f"   Delta_s shape: {delta_s.shape}")
            print(f"   Delta_s range: [{delta_s.min():.6f}, {delta_s.max():.6f}]")
            print(f"   Delta_s mean: {delta_s.mean():.6f}")
            print(f"   Delta_s std: {delta_s.std():.6f}")

    print(f"\n📊 Real Kernel QK Smoothing:")
    print(f"   QK smoothing: Enabled (per_block_mean=True)")
    print(f"   Implementation: Hardware-optimized (internal)")

def compare_kernel_inputs_detailed(triton_data, real_data):
    """Detailed comparison of what actually reaches the kernels."""
    print(f"\n🔬 Kernel Input Comparison")
    print("=" * 50)

    if 'kernel_inputs' in triton_data:
        kernel_inputs = triton_data['kernel_inputs']

        print("📊 Triton Kernel Inputs:")
        for name in ['q', 'k', 'v']:
            if name in kernel_inputs and kernel_inputs[name] is not None:
                tensor = kernel_inputs[name]
                print(f"   {name.upper()}: shape={tensor.shape}, dtype={tensor.dtype}")
                print(f"       range=[{tensor.min():.6f}, {tensor.max():.6f}]")
                print(f"       mean={tensor.mean():.6f}, std={tensor.std():.6f}")

        print(f"   Scale factor: {kernel_inputs.get('sm_scale', 'Not set')}")
        print(f"   Causal: {kernel_inputs.get('is_causal', False)}")
        print(f"   Tile sizes: Q={kernel_inputs.get('tile_size_q', 'N/A')}, K={kernel_inputs.get('tile_size_k', 'N/A')}")

        if kernel_inputs.get('delta_s') is not None:
            delta_s = kernel_inputs['delta_s']
            print(f"   Delta_s: shape={delta_s.shape}, range=[{delta_s.min():.6f}, {delta_s.max():.6f}]")

    print(f"\n📊 Real Kernel Inputs:")
    print(f"   Input processing: Internal preprocessing")
    print(f"   Tile sizes: kBlockM=128, kBlockN=128 (hardcoded)")
    print(f"   Quantization: Hardware FP4/FP8 with microscaling")

def test_advanced_comparison():
    """Main comparison test."""
    print("🎯 Advanced Kernel Input Comparison")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False

    device = torch.device('cuda')
    print(f"🖥️  GPU: {torch.cuda.get_device_name(0)}")

    # Test configuration
    B, H, N, D = 2, 30, 1024, 64
    print(f"\n🧪 Test Configuration: {B}×{H}×{N}×{D}")
    print("-" * 40)

    torch.manual_seed(42)
    dtype = torch.float16

    # Create test tensors
    q = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.01
    k = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.01
    v = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.01

    # Capture both pipelines
    capturer = KernelInputCapture()

    try:
        # Capture Triton pipeline
        triton_output = capturer.capture_triton_pipeline(q.clone(), k.clone(), v.clone())

        # Capture real kernel pipeline
        real_output = capturer.capture_real_kernel_pipeline(q.clone(), k.clone(), v.clone())

        if real_output is None:
            print("❌ Real kernel not available")
            return False

        # Perform detailed analysis
        triton_data = capturer.captured_data.get('triton', {})
        real_data = capturer.captured_data.get('real', {})

        compare_quantization_effects(triton_data, real_data)
        compare_qk_smoothing(triton_data, real_data)
        compare_kernel_inputs_detailed(triton_data, real_data)

        # Final output comparison
        print(f"\n🔬 Final Output Comparison")
        print("=" * 50)
        cos_sim = F.cosine_similarity(triton_output.flatten(), real_output.flatten(), dim=0)
        diff = (triton_output - real_output).float()
        max_abs_diff = diff.abs().max()
        mean_abs_diff = diff.abs().mean()

        print(f"📊 Output Similarity: {cos_sim.item():.8f}")
        print(f"📊 Max absolute difference: {max_abs_diff.item():.6e}")
        print(f"📊 Mean absolute difference: {mean_abs_diff.item():.6e}")

        if cos_sim > 0.99:
            print("🎉 EXCELLENT: Kernel processing is highly aligned!")
        elif cos_sim > 0.95:
            print("✅ GOOD: Minor differences in kernel processing")
        else:
            print("⚠️  ATTENTION: Significant differences in kernel processing")

        return True

    except Exception as e:
        print(f"❌ Error during comparison: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    setup_paths()

    print("🎯 SageAttention3 Advanced Kernel Input Analysis")
    print("Deep inspection of preprocessing and kernel inputs")
    print()

    success = test_advanced_comparison()

    if success:
        print(f"\n✅ ADVANCED ANALYSIS COMPLETE!")
        print("Review the preprocessing differences above to understand accuracy gaps.")
    else:
        print(f"\n❌ ANALYSIS FAILED")

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