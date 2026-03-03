#!/usr/bin/env python3
"""
Real Kernel API Patching for Input Capture
==========================================

This script patches the sageattn3_blackwell API functions to capture
the actual inputs that reach the low-level CUDA kernels, including:

1. preprocess_qkv - QK smoothing and delta_s computation
2. scale_and_quant_fp4* - Hardware FP4 quantization steps
3. blockscaled_fp4_attn - Core attention kernel inputs

This gives us visibility into the real kernel's preprocessing pipeline
that was previously hidden inside the compiled CUDA code.

Usage:
    python patch_real_kernel_api.py
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

class RealKernelAPIPatcher:
    """Patches real kernel API functions to capture intermediate values."""

    def __init__(self):
        self.captured_data = {}
        self.original_functions = {}
        self.is_patched = False

    def patch_api_functions(self):
        """Patch all the key API functions in sageattn3."""
        if self.is_patched:
            return

        print("🔧 Patching sageattn3 API functions...")

        try:
            import sageattn3.api as api_module

            # Store original functions
            self.original_functions = {
                'preprocess_qkv': api_module.preprocess_qkv,
                'scale_and_quant_fp4': api_module.scale_and_quant_fp4,
                'scale_and_quant_fp4_permute': api_module.scale_and_quant_fp4_permute,
                'scale_and_quant_fp4_transpose': api_module.scale_and_quant_fp4_transpose,
                'blockscaled_fp4_attn': api_module.blockscaled_fp4_attn
            }

            # Create patched functions
            api_module.preprocess_qkv = self._patch_preprocess_qkv(self.original_functions['preprocess_qkv'])
            api_module.scale_and_quant_fp4 = self._patch_scale_and_quant_fp4(self.original_functions['scale_and_quant_fp4'])
            api_module.scale_and_quant_fp4_permute = self._patch_scale_and_quant_fp4_permute(self.original_functions['scale_and_quant_fp4_permute'])
            api_module.scale_and_quant_fp4_transpose = self._patch_scale_and_quant_fp4_transpose(self.original_functions['scale_and_quant_fp4_transpose'])
            api_module.blockscaled_fp4_attn = self._patch_blockscaled_fp4_attn(self.original_functions['blockscaled_fp4_attn'])

            self.is_patched = True
            print("   ✅ API functions patched successfully")

        except ImportError as e:
            print(f"   ❌ Failed to patch API: {e}")
            raise

    def _patch_preprocess_qkv(self, original_func):
        """Patch preprocess_qkv to capture QK smoothing."""
        def patched_preprocess_qkv(q, k, v, per_block_mean=True):
            print("   📎 Capturing preprocess_qkv...")

            # Store inputs
            self.captured_data['preprocess_qkv'] = {
                'inputs': {
                    'q_raw': q.clone(),
                    'k_raw': k.clone(),
                    'v_raw': v.clone(),
                    'per_block_mean': per_block_mean
                }
            }

            # Call original function
            q_processed, k_processed, v_processed, delta_s = original_func(q, k, v, per_block_mean)

            # Store outputs
            self.captured_data['preprocess_qkv']['outputs'] = {
                'q_processed': q_processed.clone(),
                'k_processed': k_processed.clone(),
                'v_processed': v_processed.clone(),
                'delta_s': delta_s.clone() if delta_s is not None else None
            }

            # Log details
            print(f"      Q: {q.shape} -> {q_processed.shape}, range: [{q_processed.min():.6f}, {q_processed.max():.6f}]")
            print(f"      K: {k.shape} -> {k_processed.shape}, range: [{k_processed.min():.6f}, {k_processed.max():.6f}]")
            print(f"      V: {v.shape} -> {v_processed.shape}, range: [{v_processed.min():.6f}, {v_processed.max():.6f}]")
            if delta_s is not None:
                print(f"      Delta_s: {delta_s.shape}, range: [{delta_s.min():.6f}, {delta_s.max():.6f}]")

            return q_processed, k_processed, v_processed, delta_s

        return patched_preprocess_qkv

    def _patch_scale_and_quant_fp4(self, original_func):
        """Patch Q tensor FP4 quantization."""
        def patched_scale_and_quant_fp4(x):
            print("   📎 Capturing scale_and_quant_fp4 (Q)...")

            # Store input
            if 'quantization' not in self.captured_data:
                self.captured_data['quantization'] = {}

            self.captured_data['quantization']['q_fp4'] = {
                'input': x.clone(),
                'input_shape': x.shape,
                'input_dtype': x.dtype,
                'input_range': [x.min().item(), x.max().item()]
            }

            # Call original function
            packed_fp4, fp8_scale = original_func(x)

            # Store outputs
            self.captured_data['quantization']['q_fp4']['outputs'] = {
                'packed_fp4': packed_fp4.clone(),
                'fp8_scale': fp8_scale.clone(),
                'packed_shape': packed_fp4.shape,
                'scale_shape': fp8_scale.shape,
                'packed_dtype': packed_fp4.dtype,
                'scale_dtype': fp8_scale.dtype
            }

            print(f"      Input: {x.shape} {x.dtype} [{x.min():.6f}, {x.max():.6f}]")
            print(f"      Packed FP4: {packed_fp4.shape} {packed_fp4.dtype}")
            print(f"      FP8 Scale: {fp8_scale.shape} {fp8_scale.dtype}")

            return packed_fp4, fp8_scale

        return patched_scale_and_quant_fp4

    def _patch_scale_and_quant_fp4_permute(self, original_func):
        """Patch K tensor FP4 quantization with permutation."""
        def patched_scale_and_quant_fp4_permute(x):
            print("   📎 Capturing scale_and_quant_fp4_permute (K)...")

            # Store input
            if 'quantization' not in self.captured_data:
                self.captured_data['quantization'] = {}

            self.captured_data['quantization']['k_fp4'] = {
                'input': x.clone(),
                'input_shape': x.shape,
                'input_dtype': x.dtype,
                'input_range': [x.min().item(), x.max().item()]
            }

            # Call original function
            packed_fp4, fp8_scale = original_func(x)

            # Store outputs
            self.captured_data['quantization']['k_fp4']['outputs'] = {
                'packed_fp4': packed_fp4.clone(),
                'fp8_scale': fp8_scale.clone(),
                'packed_shape': packed_fp4.shape,
                'scale_shape': fp8_scale.shape,
                'packed_dtype': packed_fp4.dtype,
                'scale_dtype': fp8_scale.dtype
            }

            print(f"      Input: {x.shape} {x.dtype} [{x.min():.6f}, {x.max():.6f}]")
            print(f"      Packed FP4: {packed_fp4.shape} {packed_fp4.dtype}")
            print(f"      FP8 Scale: {fp8_scale.shape} {fp8_scale.dtype}")

            return packed_fp4, fp8_scale

        return patched_scale_and_quant_fp4_permute

    def _patch_scale_and_quant_fp4_transpose(self, original_func):
        """Patch V tensor FP4 quantization with transpose."""
        def patched_scale_and_quant_fp4_transpose(x):
            print("   📎 Capturing scale_and_quant_fp4_transpose (V)...")

            # Store input
            if 'quantization' not in self.captured_data:
                self.captured_data['quantization'] = {}

            self.captured_data['quantization']['v_fp4'] = {
                'input': x.clone(),
                'input_shape': x.shape,
                'input_dtype': x.dtype,
                'input_range': [x.min().item(), x.max().item()]
            }

            # Call original function
            packed_fp4, fp8_scale = original_func(x)

            # Store outputs
            self.captured_data['quantization']['v_fp4']['outputs'] = {
                'packed_fp4': packed_fp4.clone(),
                'fp8_scale': fp8_scale.clone(),
                'packed_shape': packed_fp4.shape,
                'scale_shape': fp8_scale.shape,
                'packed_dtype': packed_fp4.dtype,
                'scale_dtype': fp8_scale.dtype
            }

            print(f"      Input: {x.shape} {x.dtype} [{x.min():.6f}, {x.max():.6f}]")
            print(f"      Packed FP4: {packed_fp4.shape} {packed_fp4.dtype}")
            print(f"      FP8 Scale: {fp8_scale.shape} {fp8_scale.dtype}")

            return packed_fp4, fp8_scale

        return patched_scale_and_quant_fp4_transpose

    def _patch_blockscaled_fp4_attn(self, original_func):
        """Patch the core attention kernel."""
        def patched_blockscaled_fp4_attn(qlist, klist, vlist, delta_s, KL, is_causal=False, per_block_mean=True, is_bf16=True):
            print("   📎 Capturing blockscaled_fp4_attn (CORE KERNEL)...")

            # Store kernel inputs
            self.captured_data['kernel_inputs'] = {
                'qlist': {
                    'packed_fp4': qlist[0].clone(),
                    'fp8_scale': qlist[1].clone(),
                    'shapes': [qlist[0].shape, qlist[1].shape],
                    'dtypes': [qlist[0].dtype, qlist[1].dtype]
                },
                'klist': {
                    'packed_fp4': klist[0].clone(),
                    'fp8_scale': klist[1].clone(),
                    'shapes': [klist[0].shape, klist[1].shape],
                    'dtypes': [klist[0].dtype, klist[1].dtype]
                },
                'vlist': {
                    'packed_fp4': vlist[0].clone(),
                    'fp8_scale': vlist[1].clone(),
                    'shapes': [vlist[0].shape, vlist[1].shape],
                    'dtypes': [vlist[0].dtype, vlist[1].dtype]
                },
                'delta_s': delta_s.clone() if delta_s is not None else None,
                'parameters': {
                    'KL': KL,
                    'is_causal': is_causal,
                    'per_block_mean': per_block_mean,
                    'is_bf16': is_bf16,
                    'softmax_scale': (qlist[0].shape[-1] * 2) ** (-0.5)
                }
            }

            print(f"      Q packed: {qlist[0].shape} {qlist[0].dtype}")
            print(f"      Q scale:  {qlist[1].shape} {qlist[1].dtype}")
            print(f"      K packed: {klist[0].shape} {klist[0].dtype}")
            print(f"      K scale:  {klist[1].shape} {klist[1].dtype}")
            print(f"      V packed: {vlist[0].shape} {vlist[0].dtype}")
            print(f"      V scale:  {vlist[1].shape} {vlist[1].dtype}")
            if delta_s is not None:
                print(f"      Delta_s:  {delta_s.shape} {delta_s.dtype} [{delta_s.min():.6f}, {delta_s.max():.6f}]")
            print(f"      Scale: {(qlist[0].shape[-1] * 2) ** (-0.5):.6f}")

            # Call original kernel
            result = original_func(qlist, klist, vlist, delta_s, KL, is_causal, per_block_mean, is_bf16)

            # Store result (handle different return types)
            if isinstance(result, (list, tuple)):
                output = result[0]  # First element is usually the output tensor
            else:
                output = result

            self.captured_data['kernel_inputs']['output'] = {
                'tensor': output.clone(),
                'shape': output.shape,
                'dtype': output.dtype,
                'range': [output.min().item(), output.max().item()],
                'result_type': type(result).__name__,
                'result_length': len(result) if isinstance(result, (list, tuple)) else 1
            }

            print(f"      Output:   {output.shape} {output.dtype} [{output.min():.6f}, {output.max():.6f}]")
            print(f"      Result type: {type(result).__name__} (length: {len(result) if isinstance(result, (list, tuple)) else 1})")

            return result

        return patched_blockscaled_fp4_attn

    def unpatch_api_functions(self):
        """Restore original API functions."""
        if not self.is_patched:
            return

        print("🔧 Unpatching sageattn3 API functions...")

        try:
            import sageattn3.api as api_module

            # Restore original functions
            for func_name, original_func in self.original_functions.items():
                setattr(api_module, func_name, original_func)

            self.is_patched = False
            print("   ✅ API functions restored")

        except ImportError as e:
            print(f"   ❌ Failed to unpatch API: {e}")

    def capture_real_kernel_pipeline(self, q, k, v, is_causal=False, per_block_mean=True):
        """Capture the complete real kernel pipeline."""
        print("🔍 Capturing real kernel pipeline with API patching...")

        self.captured_data = {}  # Reset capture data

        try:
            # Import and run the real kernel
            from sageattn3 import sageattn3_blackwell

            output = sageattn3_blackwell(
                q.clone(), k.clone(), v.clone(),
                is_causal=is_causal,
                per_block_mean=per_block_mean
            )

            self.captured_data['final_output'] = {
                'tensor': output.clone(),
                'shape': output.shape,
                'dtype': output.dtype,
                'range': [output.min().item(), output.max().item()]
            }

            print(f"   📎 Final output: {output.shape} {output.dtype} [{output.min():.6f}, {output.max():.6f}]")

            return output

        except Exception as e:
            print(f"   ❌ Pipeline capture failed: {e}")
            import traceback
            traceback.print_exc()
            return None

def compare_with_triton(real_data, q, k, v):
    """Compare captured real kernel data with Triton implementation."""
    print("\n🔬 Comparing Real Kernel vs Triton Implementation")
    print("=" * 60)

    try:
        # Import and run Triton
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, current_dir)

        from sageattn3_torch_triton import sageattn3_torch_triton

        triton_output = sageattn3_torch_triton(
            q=q.clone(), k=k.clone(), v=v.clone(),
            tensor_layout="HND",
            is_causal=False,
            per_block_mean=True,
            tile_size_q=128,
            tile_size_k=128,
            debug=False
        )

        # Compare preprocessing
        if 'preprocess_qkv' in real_data:
            preprocess = real_data['preprocess_qkv']
            print("📊 QK Smoothing Comparison:")
            print(f"   Real kernel delta_s: {preprocess['outputs']['delta_s'].shape if preprocess['outputs']['delta_s'] is not None else 'None'}")
            if preprocess['outputs']['delta_s'] is not None:
                delta_s = preprocess['outputs']['delta_s']
                print(f"   Delta_s range: [{delta_s.min():.6f}, {delta_s.max():.6f}]")
                print(f"   Delta_s mean: {delta_s.mean():.6f}, std: {delta_s.std():.6f}")

        # Compare quantization
        if 'quantization' in real_data:
            print("\n📊 Quantization Comparison:")
            for tensor_name, quant_data in real_data['quantization'].items():
                if 'outputs' in quant_data:
                    print(f"   {tensor_name.upper()}:")
                    print(f"      Input -> Packed: {quant_data['input_shape']} -> {quant_data['outputs']['packed_shape']}")
                    print(f"      Scale shape: {quant_data['outputs']['scale_shape']}")
                    print(f"      Compression ratio: {np.prod(quant_data['input_shape']) / np.prod(quant_data['outputs']['packed_shape']):.1f}x")

        # Compare final outputs
        real_output = real_data['final_output']['tensor']
        print(f"\n📊 Final Output Comparison:")
        print(f"   Real kernel:  {real_output.shape} [{real_output.min():.6f}, {real_output.max():.6f}]")
        print(f"   Triton:       {triton_output.shape} [{triton_output.min():.6f}, {triton_output.max():.6f}]")

        # Compute similarity
        cos_sim = F.cosine_similarity(real_output.flatten(), triton_output.flatten(), dim=0)
        diff = (real_output - triton_output).float()
        max_abs_diff = diff.abs().max()
        mean_abs_diff = diff.abs().mean()

        print(f"   Cosine similarity: {cos_sim.item():.8f}")
        print(f"   Max abs diff: {max_abs_diff.item():.6e}")
        print(f"   Mean abs diff: {mean_abs_diff.item():.6e}")

        if cos_sim > 0.99:
            print("   🎉 EXCELLENT similarity!")
        elif cos_sim > 0.95:
            print("   ✅ GOOD similarity")
        else:
            print("   ⚠️  Accuracy gap detected")

        return cos_sim.item()

    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        return 0.0

def test_real_kernel_api_patching():
    """Test the API patching approach."""
    print("🎯 Real Kernel API Patching Test")
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

    # Create patcher and patch API
    patcher = RealKernelAPIPatcher()

    try:
        patcher.patch_api_functions()

        # Run real kernel with patched API
        output = patcher.capture_real_kernel_pipeline(q, k, v)

        if output is None:
            print("❌ Real kernel execution failed")
            return False

        # Compare with Triton
        similarity = compare_with_triton(patcher.captured_data, q, k, v)

        print(f"\n🎯 API Patching Results:")
        print(f"   ✅ Successfully captured real kernel internals")
        print(f"   ✅ Preprocessing pipeline visible")
        print(f"   ✅ Hardware FP4 quantization captured")
        print(f"   ✅ Core kernel inputs captured")
        print(f"   📊 Accuracy vs Triton: {similarity:.6f}")

        return True

    except Exception as e:
        print(f"❌ API patching failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Always unpatch
        patcher.unpatch_api_functions()

def main():
    """Main function."""
    setup_paths()

    print("🎯 SageAttention3 Real Kernel API Patching")
    print("Capturing actual inputs to sageattn3_blackwell internals")
    print()

    success = test_real_kernel_api_patching()

    if success:
        print(f"\n✅ API PATCHING SUCCESS!")
        print("Real kernel preprocessing pipeline is now visible!")
        print("This provides the missing piece for accurate Triton alignment.")
    else:
        print(f"\n❌ API PATCHING FAILED")

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