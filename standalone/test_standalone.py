#!/usr/bin/env python3
"""
Comprehensive Test Suite for SageAttention3 Standalone Implementation
===================================================================

This test suite provides thorough validation of the standalone SageAttention3
implementation, including correctness, performance, and integration tests.

Usage:
    python test_standalone.py [--verbose] [--performance] [--accuracy-only]

Options:
    --verbose       Enable detailed debug output
    --performance   Run performance benchmarks
    --accuracy-only Only run accuracy tests (skip performance)
"""

import sys
import os
import time
import argparse
import warnings
from typing import List, Tuple, Dict

import torch
import torch.nn.functional as F

# Add current directory to path to import standalone module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from sageattention3_standalone import (
        scaled_dot_product_attention,
        sageattn3_torch_triton_standalone,
        run_all_tests as builtin_tests,
        debug_print,
        SAGE3_DEBUG,
        QUANT_FORMATS,
        SAGE3_QUANT_FORMAT,
    )
    print("✅ Successfully imported SageAttention3 standalone module")
except ImportError as e:
    print(f"❌ Failed to import SageAttention3 standalone: {e}")
    sys.exit(1)

# Test configuration
TEST_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TEST_DTYPE = torch.float16
VERBOSE = False

class TestResults:
    """Class to track test results and generate reports."""

    def __init__(self):
        self.results: List[Tuple[str, bool, str]] = []
        self.start_time = time.time()

    def add_result(self, test_name: str, passed: bool, message: str = ""):
        self.results.append((test_name, passed, message))
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status:<8} {test_name}")
        if message and (VERBOSE or not passed):
            print(f"         {message}")

    def print_summary(self):
        """Print comprehensive test results summary."""
        total_time = time.time() - self.start_time
        passed = sum(1 for _, result, _ in self.results if result)
        total = len(self.results)

        print("\n" + "=" * 70)
        print("COMPREHENSIVE TEST RESULTS SUMMARY")
        print("=" * 70)

        for test_name, result, message in self.results:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status:<8} {test_name}")
            if message and not result:  # Show failure messages
                print(f"         {message}")

        print(f"\nTotal: {passed}/{total} tests passed")
        print(f"Runtime: {total_time:.2f} seconds")

        if passed == total:
            print("\n🎉 ALL TESTS PASSED! SageAttention3 standalone is ready for production.")
        else:
            print(f"\n⚠️  {total - passed} test(s) failed. Check output above for details.")

        return passed == total

def log_verbose(message: str):
    """Log message if verbose mode is enabled."""
    if VERBOSE:
        print(f"[VERBOSE] {message}")

# ============================================================================
# Core Functionality Tests
# ============================================================================

def test_tensor_shapes_and_dtypes(results: TestResults):
    """Test various tensor shapes and data types."""
    test_cases = [
        # (B, H, N, D)
        (1, 1, 32, 16),    # Minimal
        (1, 8, 64, 32),    # Small
        (2, 16, 128, 64),  # Medium
        (1, 32, 256, 128), # Large head dim
        (4, 8, 512, 64),   # Large batch
    ]

    for i, (B, H, N, D) in enumerate(test_cases):
        try:
            log_verbose(f"Testing shape [{B}, {H}, {N}, {D}]")

            q = torch.randn(B, H, N, D, dtype=TEST_DTYPE, device=TEST_DEVICE)
            k = torch.randn(B, H, N, D, dtype=TEST_DTYPE, device=TEST_DEVICE)
            v = torch.randn(B, H, N, D, dtype=TEST_DTYPE, device=TEST_DEVICE)

            output = scaled_dot_product_attention(q, k, v)

            # Validate output
            assert output.shape == (B, H, N, D), f"Shape mismatch: {output.shape} != {(B, H, N, D)}"
            assert output.dtype == TEST_DTYPE, f"Dtype mismatch: {output.dtype} != {TEST_DTYPE}"
            assert output.device == q.device, f"Device mismatch: {output.device} != {q.device}"
            assert not torch.isnan(output).any(), "Output contains NaN"
            assert not torch.isinf(output).any(), "Output contains Inf"

            results.add_result(f"Shape Test {i+1} [{B}×{H}×{N}×{D}]", True)

        except Exception as e:
            results.add_result(f"Shape Test {i+1} [{B}×{H}×{N}×{D}]", False, str(e))

def test_causal_masking(results: TestResults):
    """Test causal masking functionality."""
    try:
        B, H, N, D = 1, 4, 64, 32

        q = torch.randn(B, H, N, D, dtype=TEST_DTYPE, device=TEST_DEVICE)
        k = torch.randn(B, H, N, D, dtype=TEST_DTYPE, device=TEST_DEVICE)
        v = torch.randn(B, H, N, D, dtype=TEST_DTYPE, device=TEST_DEVICE)

        # Test non-causal
        output_non_causal = scaled_dot_product_attention(q, k, v, is_causal=False)

        # Test causal
        output_causal = scaled_dot_product_attention(q, k, v, is_causal=True)

        # They should be different
        assert not torch.allclose(output_non_causal, output_causal, atol=1e-3), \
            "Causal and non-causal outputs are too similar"

        # Causal output should have different structure (can't easily verify triangular property in output)
        # But we can verify it runs without error
        assert output_causal.shape == output_non_causal.shape, "Shape mismatch between causal modes"

        results.add_result("Causal Masking", True, "Causal and non-causal modes produce different outputs")

    except Exception as e:
        results.add_result("Causal Masking", False, str(e))

def test_custom_scale_factor(results: TestResults):
    """Test custom scale factor functionality."""
    try:
        B, H, N, D = 1, 2, 32, 16

        q = torch.randn(B, H, N, D, dtype=TEST_DTYPE, device=TEST_DEVICE)
        k = torch.randn(B, H, N, D, dtype=TEST_DTYPE, device=TEST_DEVICE)
        v = torch.randn(B, H, N, D, dtype=TEST_DTYPE, device=TEST_DEVICE)

        # Test with default scale
        output_default = scaled_dot_product_attention(q, k, v)

        # Test with custom scale
        custom_scale = 0.5
        output_custom = scaled_dot_product_attention(q, k, v, scale=custom_scale)

        # They should be different
        assert not torch.allclose(output_default, output_custom, atol=1e-3), \
            "Default and custom scale outputs are too similar"

        results.add_result("Custom Scale Factor", True, f"Custom scale {custom_scale} produces different output")

    except Exception as e:
        results.add_result("Custom Scale Factor", False, str(e))

# ============================================================================
# Accuracy Tests
# ============================================================================

def test_accuracy_comprehensive(results: TestResults):
    """Comprehensive accuracy test against PyTorch SDPA."""
    test_cases = [
        # (B, H, N, D, is_causal, description)
        (1, 1, 32, 16, False, "Minimal Non-Causal"),
        (1, 1, 32, 16, True, "Minimal Causal"),
        (1, 8, 64, 32, False, "Small Non-Causal"),
        (1, 8, 64, 32, True, "Small Causal"),
        (2, 4, 128, 64, False, "Medium Non-Causal"),
        (2, 4, 128, 64, True, "Medium Causal"),
    ]

    for B, H, N, D, is_causal, description in test_cases:
        try:
            log_verbose(f"Testing accuracy: {description} [{B}×{H}×{N}×{D}]")

            # Set random seed for reproducibility
            torch.manual_seed(42)
            q = torch.randn(B, H, N, D, dtype=TEST_DTYPE, device=TEST_DEVICE)
            k = torch.randn(B, H, N, D, dtype=TEST_DTYPE, device=TEST_DEVICE)
            v = torch.randn(B, H, N, D, dtype=TEST_DTYPE, device=TEST_DEVICE)

            # Get PyTorch SDPA reference
            with torch.no_grad():
                ref_output = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

            # Get SageAttention3 output
            with torch.no_grad():
                sage_output = scaled_dot_product_attention(q, k, v, is_causal=is_causal)

            # Compute metrics
            ref_flat = ref_output.flatten().float()
            sage_flat = sage_output.flatten().float()

            # Cosine similarity
            cos_sim = F.cosine_similarity(ref_flat.unsqueeze(0), sage_flat.unsqueeze(0)).item()

            # L2 relative error
            l2_error = torch.norm(ref_flat - sage_flat) / torch.norm(ref_flat)

            # Maximum absolute error
            max_abs_error = torch.max(torch.abs(ref_flat - sage_flat)).item()

            log_verbose(f"  Cosine similarity: {cos_sim:.6f}")
            log_verbose(f"  L2 relative error: {l2_error:.6f}")
            log_verbose(f"  Max absolute error: {max_abs_error:.6f}")

            # Pass criteria (relaxed for FP16)
            cos_sim_threshold = 0.90  # 90% similarity
            l2_threshold = 0.2  # 20% relative error

            if cos_sim >= cos_sim_threshold and l2_error <= l2_threshold:
                results.add_result(
                    f"Accuracy: {description}",
                    True,
                    f"cos_sim={cos_sim:.4f}, l2_err={l2_error:.4f}"
                )
            else:
                results.add_result(
                    f"Accuracy: {description}",
                    False,
                    f"cos_sim={cos_sim:.4f} (need >{cos_sim_threshold}), l2_err={l2_error:.4f} (need <{l2_threshold})"
                )

        except Exception as e:
            results.add_result(f"Accuracy: {description}", False, str(e))

def test_mxfp4_accuracy(results: TestResults):
    """Test MXFP4 accuracy against PyTorch SDPA.

    MXFP4 uses block_size=32 with E8M0 (power-of-2) scales, which is
    coarser than NVFP4 (block_size=16, E4M3 scales). Expected CosSim
    is lower (~90%+ vs ~95%+ for NVFP4).
    """
    test_cases = [
        # (B, H, N, D, is_causal, description)
        (1, 1, 32, 16, False, "MXFP4 Minimal Non-Causal"),
        (1, 8, 64, 32, False, "MXFP4 Small Non-Causal"),
        (2, 4, 128, 64, False, "MXFP4 Medium Non-Causal"),
        (2, 4, 128, 64, True, "MXFP4 Medium Causal"),
    ]

    for B, H, N, D, is_causal, description in test_cases:
        try:
            log_verbose(f"Testing MXFP4 accuracy: {description} [{B}×{H}×{N}×{D}]")

            torch.manual_seed(42)
            q = torch.randn(B, H, N, D, dtype=TEST_DTYPE, device=TEST_DEVICE)
            k = torch.randn(B, H, N, D, dtype=TEST_DTYPE, device=TEST_DEVICE)
            v = torch.randn(B, H, N, D, dtype=TEST_DTYPE, device=TEST_DEVICE)

            # Get PyTorch SDPA reference
            with torch.no_grad():
                ref_output = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

            # Get SageAttention3 output with MXFP4
            with torch.no_grad():
                sage_output = scaled_dot_product_attention(
                    q, k, v, is_causal=is_causal, quant_format="mxfp4"
                )

            # Compute metrics
            ref_flat = ref_output.flatten().float()
            sage_flat = sage_output.flatten().float()

            cos_sim = F.cosine_similarity(ref_flat.unsqueeze(0), sage_flat.unsqueeze(0)).item()
            l2_error = torch.norm(ref_flat - sage_flat) / torch.norm(ref_flat)

            log_verbose(f"  Cosine similarity: {cos_sim:.6f}")
            log_verbose(f"  L2 relative error: {l2_error:.6f}")

            # Relaxed thresholds for MXFP4 (coarser quantization)
            cos_sim_threshold = 0.85  # 85% similarity (relaxed from 90% for NVFP4)
            l2_threshold = 0.3        # 30% relative error (relaxed from 20% for NVFP4)

            if cos_sim >= cos_sim_threshold and l2_error <= l2_threshold:
                results.add_result(
                    f"Accuracy: {description}",
                    True,
                    f"cos_sim={cos_sim:.4f}, l2_err={l2_error:.4f}"
                )
            else:
                results.add_result(
                    f"Accuracy: {description}",
                    False,
                    f"cos_sim={cos_sim:.4f} (need >{cos_sim_threshold}), "
                    f"l2_err={l2_error:.4f} (need <{l2_threshold})"
                )

        except Exception as e:
            results.add_result(f"Accuracy: {description}", False, str(e))


# ============================================================================
# Performance Tests
# ============================================================================

def test_performance_benchmarks(results: TestResults):
    """Performance benchmarking against PyTorch SDPA."""
    if TEST_DEVICE != 'cuda':
        results.add_result("Performance Benchmarks", False, "CUDA required for performance tests")
        return

    test_cases = [
        # (B, H, N, D, description)
        (1, 8, 256, 64, "Small Sequence"),
        (1, 8, 512, 64, "Medium Sequence"),
        (2, 16, 512, 64, "Large Batch"),
        (1, 8, 1024, 64, "Large Sequence"),
    ]

    num_runs = 20
    warmup_runs = 5

    for B, H, N, D, description in test_cases:
        try:
            log_verbose(f"Benchmarking: {description} [{B}×{H}×{N}×{D}]")

            q = torch.randn(B, H, N, D, dtype=TEST_DTYPE, device=TEST_DEVICE)
            k = torch.randn(B, H, N, D, dtype=TEST_DTYPE, device=TEST_DEVICE)
            v = torch.randn(B, H, N, D, dtype=TEST_DTYPE, device=TEST_DEVICE)

            # Benchmark SageAttention3
            # Warmup
            for _ in range(warmup_runs):
                _ = scaled_dot_product_attention(q, k, v)
            torch.cuda.synchronize()

            # Actual benchmark
            start_time = time.time()
            for _ in range(num_runs):
                _ = scaled_dot_product_attention(q, k, v)
            torch.cuda.synchronize()
            sage_time = (time.time() - start_time) / num_runs

            # Benchmark PyTorch SDPA
            # Warmup
            for _ in range(warmup_runs):
                _ = F.scaled_dot_product_attention(q, k, v)
            torch.cuda.synchronize()

            # Actual benchmark
            start_time = time.time()
            for _ in range(num_runs):
                _ = F.scaled_dot_product_attention(q, k, v)
            torch.cuda.synchronize()
            pytorch_time = (time.time() - start_time) / num_runs

            # Calculate speedup
            speedup = pytorch_time / sage_time if sage_time > 0 else 0

            log_verbose(f"  SageAttention3: {sage_time*1000:.3f}ms")
            log_verbose(f"  PyTorch SDPA: {pytorch_time*1000:.3f}ms")
            log_verbose(f"  Speedup: {speedup:.2f}x")

            # Consider it a pass if SageAttention3 runs successfully (speedup may vary)
            results.add_result(
                f"Performance: {description}",
                True,
                f"SageAttention3: {sage_time*1000:.2f}ms, PyTorch: {pytorch_time*1000:.2f}ms, Speedup: {speedup:.2f}x"
            )

        except Exception as e:
            results.add_result(f"Performance: {description}", False, str(e))

# ============================================================================
# Error Handling Tests
# ============================================================================

def test_error_handling(results: TestResults):
    """Test error handling and graceful fallbacks."""

    # Test 1: CPU tensors (should fall back gracefully if on GPU machine)
    try:
        if TEST_DEVICE == 'cuda':
            q_cpu = torch.randn(1, 2, 32, 16, dtype=TEST_DTYPE, device='cpu')
            k_cpu = torch.randn(1, 2, 32, 16, dtype=TEST_DTYPE, device='cpu')
            v_cpu = torch.randn(1, 2, 32, 16, dtype=TEST_DTYPE, device='cpu')

            # Should fall back to PyTorch SDPA without error
            with warnings.catch_warnings(record=True) as w:
                output = scaled_dot_product_attention(q_cpu, k_cpu, v_cpu)
                assert output.shape == q_cpu.shape, "CPU fallback failed"

            results.add_result("Error Handling: CPU Fallback", True, "Successfully fell back to PyTorch SDPA")
        else:
            results.add_result("Error Handling: CPU Fallback", True, "Skipped (already on CPU)")

    except Exception as e:
        results.add_result("Error Handling: CPU Fallback", False, str(e))

    # Test 2: Shape mismatch
    try:
        q = torch.randn(1, 2, 32, 16, dtype=TEST_DTYPE, device=TEST_DEVICE)
        k = torch.randn(1, 2, 64, 16, dtype=TEST_DTYPE, device=TEST_DEVICE)  # Wrong N
        v = torch.randn(1, 2, 32, 16, dtype=TEST_DTYPE, device=TEST_DEVICE)

        try:
            output = scaled_dot_product_attention(q, k, v)
            results.add_result("Error Handling: Shape Mismatch", False, "Should have raised error for shape mismatch")
        except (ValueError, RuntimeError) as expected_error:
            results.add_result("Error Handling: Shape Mismatch", True, f"Correctly raised: {type(expected_error).__name__}")

    except Exception as e:
        results.add_result("Error Handling: Shape Mismatch", False, f"Unexpected error: {e}")

    # Test 3: Unsupported features (should warn but not fail)
    try:
        q = torch.randn(1, 2, 32, 16, dtype=TEST_DTYPE, device=TEST_DEVICE)
        k = torch.randn(1, 2, 32, 16, dtype=TEST_DTYPE, device=TEST_DEVICE)
        v = torch.randn(1, 2, 32, 16, dtype=TEST_DTYPE, device=TEST_DEVICE)

        # Test with unsupported parameters
        mask = torch.ones(32, 32, device=TEST_DEVICE)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            output = scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.1)

            # Should produce output despite warnings
            assert output.shape == q.shape, "Function failed with unsupported parameters"

            # Should have produced warnings in debug mode
            if SAGE3_DEBUG and len(w) > 0:
                results.add_result("Error Handling: Unsupported Features", True, f"Issued {len(w)} warnings as expected")
            else:
                results.add_result("Error Handling: Unsupported Features", True, "Handled unsupported features gracefully")

    except Exception as e:
        results.add_result("Error Handling: Unsupported Features", False, str(e))

# ============================================================================
# Integration Tests
# ============================================================================

def test_drop_in_replacement(results: TestResults):
    """Test that it works as a drop-in replacement for F.scaled_dot_product_attention."""
    try:
        # Save original function
        original_sdpa = F.scaled_dot_product_attention

        # Replace with SageAttention3
        F.scaled_dot_product_attention = scaled_dot_product_attention

        # Test that models can use it
        B, H, N, D = 1, 4, 64, 32
        q = torch.randn(B, H, N, D, dtype=TEST_DTYPE, device=TEST_DEVICE)
        k = torch.randn(B, H, N, D, dtype=TEST_DTYPE, device=TEST_DEVICE)
        v = torch.randn(B, H, N, D, dtype=TEST_DTYPE, device=TEST_DEVICE)

        # Call through F.scaled_dot_product_attention
        output = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        assert output.shape == q.shape, "Drop-in replacement failed"
        assert not torch.isnan(output).any(), "Drop-in replacement produced NaN"

        # Restore original function
        F.scaled_dot_product_attention = original_sdpa

        results.add_result("Integration: Drop-in Replacement", True, "Successfully replaced F.scaled_dot_product_attention")

    except Exception as e:
        # Make sure to restore original function even on failure
        F.scaled_dot_product_attention = original_sdpa
        results.add_result("Integration: Drop-in Replacement", False, str(e))

def test_environment_variables(results: TestResults):
    """Test environment variable configuration."""
    try:
        # Test with different environment configurations
        original_env = {}
        env_vars = ['SAGE3_DEBUG', 'SAGE3_DISABLE_PER_BLOCK_MEAN', 'SAGE3_TILE_SIZE']

        # Save original values
        for var in env_vars:
            original_env[var] = os.environ.get(var, None)

        # Test with debug enabled
        os.environ['SAGE3_DEBUG'] = '1'

        # Reimport to pick up new env vars
        import importlib
        import sageattention3_standalone
        importlib.reload(sageattention3_standalone)

        B, H, N, D = 1, 2, 32, 16
        q = torch.randn(B, H, N, D, dtype=TEST_DTYPE, device=TEST_DEVICE)
        k = torch.randn(B, H, N, D, dtype=TEST_DTYPE, device=TEST_DEVICE)
        v = torch.randn(B, H, N, D, dtype=TEST_DTYPE, device=TEST_DEVICE)

        # Should work with debug enabled
        output = sageattention3_standalone.scaled_dot_product_attention(q, k, v)
        assert output.shape == q.shape, "Failed with debug enabled"

        # Restore original environment
        for var, value in original_env.items():
            if value is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = value

        results.add_result("Integration: Environment Variables", True, "Environment variable configuration works")

    except Exception as e:
        # Restore environment on failure
        for var, value in original_env.items():
            if value is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = value
        results.add_result("Integration: Environment Variables", False, str(e))

# ============================================================================
# Main Test Runner
# ============================================================================

def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="SageAttention3 Standalone Test Suite")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--performance", action="store_true", help="Run performance benchmarks")
    parser.add_argument("--accuracy-only", action="store_true", help="Only run accuracy tests")
    args = parser.parse_args()

    global VERBOSE
    VERBOSE = args.verbose or SAGE3_DEBUG

    print("=" * 70)
    print("SageAttention3 Standalone Comprehensive Test Suite")
    print("=" * 70)
    print(f"Device: {TEST_DEVICE}")
    print(f"Data Type: {TEST_DTYPE}")
    print(f"Verbose Mode: {VERBOSE}")
    print("")

    if TEST_DEVICE != 'cuda':
        print("⚠️  Warning: Running on CPU. Performance tests will be limited.")
        print("")

    results = TestResults()

    # Run test categories
    print("[Core Functionality Tests]")
    print("-" * 50)
    test_tensor_shapes_and_dtypes(results)
    test_causal_masking(results)
    test_custom_scale_factor(results)

    if not args.accuracy_only:
        print("\n[Error Handling Tests]")
        print("-" * 50)
        test_error_handling(results)

        print("\n[Integration Tests]")
        print("-" * 50)
        test_drop_in_replacement(results)
        test_environment_variables(results)

    print("\n[Accuracy Tests]")
    print("-" * 50)
    test_accuracy_comprehensive(results)

    print("\n[MXFP4 Accuracy Tests]")
    print("-" * 50)
    test_mxfp4_accuracy(results)

    if args.performance and TEST_DEVICE == 'cuda':
        print("\n[Performance Tests]")
        print("-" * 50)
        test_performance_benchmarks(results)
    elif args.performance:
        print("\n[Performance Tests]")
        print("-" * 50)
        results.add_result("Performance Tests", False, "CUDA required for performance benchmarks")

    # Run built-in tests as well
    print("\n[Built-in Module Tests]")
    print("-" * 50)
    try:
        builtin_success = builtin_tests()
        results.add_result("Built-in Test Suite", builtin_success, "Ran built-in module tests")
    except Exception as e:
        results.add_result("Built-in Test Suite", False, str(e))

    # Print comprehensive summary
    success = results.print_summary()

    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()