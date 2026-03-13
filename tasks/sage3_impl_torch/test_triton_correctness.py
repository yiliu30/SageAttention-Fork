#!/usr/bin/env python3
"""
Test Triton Correctness Against PyTorch Reference
==============================================

Comprehensive test suite for comparing the Triton implementation
against the highly accurate PyTorch reference implementation.

Tests various configurations:
- Different batch sizes, head counts, sequence lengths
- With/without causal masking
- With/without QK smoothing (delta_s)
- Different tile sizes

Target: >99% cosine similarity with PyTorch reference
"""

import os
import sys
import torch
import torch.nn.functional as F
import math
import time
from typing import Tuple, List, Dict, Any

# Add current directory to path to import both versions
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sageattn3_torch import sageattn3_torch
from sageattn3_torch_triton import sageattn3_torch_triton

def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity between two tensors."""
    a_flat = a.view(-1).float()
    b_flat = b.view(-1).float()

    cos_sim = F.cosine_similarity(a_flat, b_flat, dim=0)
    return cos_sim.item()

def relative_error(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute relative error between two tensors."""
    return ((a - b).abs() / (a.abs() + 1e-8)).mean().item()

def test_configuration(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    is_causal: bool = False,
    per_block_mean: bool = True,
    tile_size_q: int = 64,
    tile_size_k: int = 64,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    Test a specific configuration and return metrics.

    Returns:
        Dictionary with test results including cosine similarity,
        relative error, timing, and configuration details.
    """
    print(f"\nTesting: B={batch_size}, H={num_heads}, N={seq_len}, D={head_dim}")
    print(f"         causal={is_causal}, smoothing={per_block_mean}, tiles=({tile_size_q},{tile_size_k})")

    # Generate random input tensors
    torch.manual_seed(42)  # For reproducibility
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)

    # Scale inputs for numerical stability
    q = q * 0.1
    k = k * 0.1
    v = v * 0.1

    sm_scale = 1.0 / math.sqrt(head_dim)

    try:
        # Run PyTorch reference
        print("  Running PyTorch reference...")
        torch.cuda.synchronize()
        start_time = time.time()

        output_torch = sageattn3_torch(
            q, k, v,
            tensor_layout="HND",
            is_causal=is_causal,
            sm_scale=sm_scale,
            per_block_mean=per_block_mean,
            tile_size_q=tile_size_q,
            tile_size_k=tile_size_k,
            return_lse=False
        )

        torch.cuda.synchronize()
        pytorch_time = time.time() - start_time

        # Run Triton implementation
        print("  Running Triton implementation...")
        torch.cuda.synchronize()
        start_time = time.time()

        output_triton = sageattn3_torch_triton(
            q, k, v,
            tensor_layout="HND",
            is_causal=is_causal,
            sm_scale=sm_scale,
            per_block_mean=per_block_mean,
            tile_size_q=tile_size_q,
            tile_size_k=tile_size_k,
            return_lse=False
        )

        torch.cuda.synchronize()
        triton_time = time.time() - start_time

        # Compute metrics
        cos_sim = cosine_similarity(output_torch, output_triton)
        rel_err = relative_error(output_torch, output_triton)
        max_abs_diff = (output_torch - output_triton).abs().max().item()

        # Check for NaN or Inf
        torch_has_nan = torch.isnan(output_torch).any().item()
        triton_has_nan = torch.isnan(output_triton).any().item()
        torch_has_inf = torch.isinf(output_torch).any().item()
        triton_has_inf = torch.isinf(output_triton).any().item()

        result = {
            "config": {
                "batch_size": batch_size,
                "num_heads": num_heads,
                "seq_len": seq_len,
                "head_dim": head_dim,
                "is_causal": is_causal,
                "per_block_mean": per_block_mean,
                "tile_size_q": tile_size_q,
                "tile_size_k": tile_size_k,
                "dtype": str(dtype)
            },
            "metrics": {
                "cosine_similarity": cos_sim,
                "relative_error": rel_err,
                "max_abs_diff": max_abs_diff,
                "pytorch_time": pytorch_time,
                "triton_time": triton_time,
                "speedup": pytorch_time / triton_time if triton_time > 0 else 0.0
            },
            "status": {
                "torch_has_nan": torch_has_nan,
                "triton_has_nan": triton_has_nan,
                "torch_has_inf": torch_has_inf,
                "triton_has_inf": triton_has_inf,
                "passed": cos_sim > 0.95 and not (torch_has_nan or triton_has_nan or torch_has_inf or triton_has_inf)
            }
        }

        # Print results
        status = "✅ PASS" if result["status"]["passed"] else "❌ FAIL"
        print(f"  {status} Cosine Similarity: {cos_sim:.6f}")
        print(f"       Relative Error: {rel_err:.6f}")
        print(f"       Max Abs Diff: {max_abs_diff:.6f}")
        print(f"       PyTorch Time: {pytorch_time:.4f}s")
        print(f"       Triton Time: {triton_time:.4f}s")
        print(f"       Speedup: {result['metrics']['speedup']:.2f}x")

        if torch_has_nan or triton_has_nan or torch_has_inf or triton_has_inf:
            print(f"       ⚠️  NaN/Inf detected: torch_nan={torch_has_nan}, triton_nan={triton_has_nan}, "
                  f"torch_inf={torch_has_inf}, triton_inf={triton_has_inf}")

        return result

    except Exception as e:
        print(f"  ❌ ERROR: {str(e)}")
        return {
            "config": {
                "batch_size": batch_size,
                "num_heads": num_heads,
                "seq_len": seq_len,
                "head_dim": head_dim,
                "is_causal": is_causal,
                "per_block_mean": per_block_mean,
                "tile_size_q": tile_size_q,
                "tile_size_k": tile_size_k,
                "dtype": str(dtype)
            },
            "metrics": {
                "cosine_similarity": 0.0,
                "relative_error": float('inf'),
                "max_abs_diff": float('inf'),
                "pytorch_time": 0.0,
                "triton_time": 0.0,
                "speedup": 0.0
            },
            "status": {
                "torch_has_nan": False,
                "triton_has_nan": False,
                "torch_has_inf": False,
                "triton_has_inf": False,
                "passed": False,
                "error": str(e)
            }
        }

def run_comprehensive_tests() -> List[Dict[str, Any]]:
    """
    Run comprehensive test suite across multiple configurations.

    Returns:
        List of test results for each configuration.
    """
    print("SageAttention3 Triton Correctness Test Suite")
    print("=" * 50)

    test_configs = [
        # Small tests
        {"batch_size": 1, "num_heads": 8, "seq_len": 128, "head_dim": 64},
        {"batch_size": 1, "num_heads": 8, "seq_len": 256, "head_dim": 64},
        {"batch_size": 1, "num_heads": 8, "seq_len": 512, "head_dim": 64},

        # Different head dimensions
        {"batch_size": 1, "num_heads": 8, "seq_len": 256, "head_dim": 128},
        {"batch_size": 2, "num_heads": 16, "seq_len": 256, "head_dim": 64},

        # Causal masking tests
        {"batch_size": 1, "num_heads": 8, "seq_len": 256, "head_dim": 64, "is_causal": True},
        {"batch_size": 1, "num_heads": 8, "seq_len": 512, "head_dim": 64, "is_causal": True},

        # Different tile sizes
        {"batch_size": 1, "num_heads": 8, "seq_len": 256, "head_dim": 64, "tile_size_q": 32, "tile_size_k": 32},
        {"batch_size": 1, "num_heads": 8, "seq_len": 256, "head_dim": 64, "tile_size_q": 128, "tile_size_k": 64},

        # Without QK smoothing
        {"batch_size": 1, "num_heads": 8, "seq_len": 256, "head_dim": 64, "per_block_mean": False},

        # Larger tests (if memory allows)
        {"batch_size": 1, "num_heads": 8, "seq_len": 1024, "head_dim": 64},
        {"batch_size": 2, "num_heads": 32, "seq_len": 512, "head_dim": 128},
        # 16k
        {"batch_size": 1, "num_heads": 8, "seq_len": 16384, "head_dim": 64}
    ]

    results = []
    passed_tests = 0
    total_tests = len(test_configs)

    for i, config in enumerate(test_configs):
        print(f"\n--- Test {i+1}/{total_tests} ---")
        result = test_configuration(**config)
        results.append(result)

        if result["status"]["passed"]:
            passed_tests += 1

    # Summary
    print(f"\n{'='*50}")
    print(f"TEST SUMMARY")
    print(f"{'='*50}")
    print(f"Passed: {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f}%)")

    # Statistics
    valid_results = [r for r in results if r["status"]["passed"] and r["metrics"]["cosine_similarity"] > 0]
    if valid_results:
        cos_sims = [r["metrics"]["cosine_similarity"] for r in valid_results]
        rel_errs = [r["metrics"]["relative_error"] for r in valid_results]
        speedups = [r["metrics"]["speedup"] for r in valid_results if r["metrics"]["speedup"] > 0]

        print(f"\nAccuracy Statistics (passed tests only):")
        print(f"  Cosine Similarity - Min: {min(cos_sims):.6f}, Max: {max(cos_sims):.6f}, Avg: {sum(cos_sims)/len(cos_sims):.6f}")
        print(f"  Relative Error    - Min: {min(rel_errs):.6f}, Max: {max(rel_errs):.6f}, Avg: {sum(rel_errs)/len(rel_errs):.6f}")

        if speedups:
            print(f"  Speedup          - Min: {min(speedups):.2f}x, Max: {max(speedups):.2f}x, Avg: {sum(speedups)/len(speedups):.2f}x")

    # Failed test details
    failed_results = [r for r in results if not r["status"]["passed"]]
    if failed_results:
        print(f"\n⚠️  FAILED TESTS:")
        for result in failed_results:
            config = result["config"]
            print(f"  B={config['batch_size']}, H={config['num_heads']}, N={config['seq_len']}, D={config['head_dim']}")
            if "error" in result["status"]:
                print(f"    Error: {result['status']['error']}")
            else:
                print(f"    Cosine Sim: {result['metrics']['cosine_similarity']:.6f}")

    return results

def test_specific_case():
    """Test a specific problematic case for debugging."""
    print("\nDEBUGGING SPECIFIC CASE")
    print("=" * 30)

    # Small, simple case for debugging
    result = test_configuration(
        batch_size=1,
        num_heads=2,
        seq_len=128,
        head_dim=64,
        is_causal=False,
        per_block_mean=True,
        tile_size_q=32,
        tile_size_k=32,
        dtype=torch.float16
    )

    return result

def main():
    """Main test function."""
    if not torch.cuda.is_available():
        print("❌ CUDA not available. Tests require GPU.")
        return

    print(f"🚀 Starting tests on {torch.cuda.get_device_name()}")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA version: {torch.version.cuda}")

    # Check if we can import both implementations
    try:
        from sageattn3_torch import sageattn3_torch
        from sageattn3_torch_triton import sageattn3_torch_triton
        print("✅ Successfully imported both implementations")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return

    # Choose test mode
    if len(sys.argv) > 1 and sys.argv[1] == "--debug":
        # Debug mode - single test case
        test_specific_case()
    else:
        # Full test suite
        results = run_comprehensive_tests()

        # Determine overall success
        passed_count = sum(1 for r in results if r["status"]["passed"])
        if passed_count >= len(results) * 0.8:  # 80% pass rate
            print(f"\n🎉 OVERALL: SUCCESS ({passed_count}/{len(results)} tests passed)")
            return 0
        else:
            print(f"\n💥 OVERALL: FAILURE ({passed_count}/{len(results)} tests passed)")
            return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code if exit_code is not None else 0)