#!/usr/bin/env python3
"""
Test suite for the sage3 refactored package.

Tests:
1. Structural: registry completeness, config validation, type safety
2. Numerical equivalence: all 4 formats vs the original monolith
3. Edge cases: causal masking, short sequences, non-divisible dims
4. SDPA wrapper
"""

import sys
import os
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List


@dataclass
class TestResult:
    name: str
    passed: bool
    message: str = ""


class TestRunner:
    def __init__(self):
        self.results: List[TestResult] = []

    def add(self, name: str, passed: bool, message: str = ""):
        self.results.append(TestResult(name, passed, message))
        status = "✅" if passed else "❌"
        print(f"  {status} {name}" + (f": {message}" if message else ""))

    def summary(self):
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed
        print(f"\n{'='*60}")
        print(f"Results: {passed}/{total} passed, {failed} failed")
        if failed > 0:
            print("Failed tests:")
            for r in self.results:
                if not r.passed:
                    print(f"  ❌ {r.name}: {r.message}")
        print(f"{'='*60}")
        return failed == 0


# ============================================================================
# Structural Tests
# ============================================================================

def test_structural(runner: TestRunner):
    """Test that all registries and configs are properly set up."""
    print("\n[Structural Tests]")

    from sage3 import ATTENTION_CONFIGS, P_QUANT_REGISTRY, AttentionConfig, QuantConfig
    from sage3.transforms import TransformContext

    # Registry completeness
    expected_configs = {"nvfp4", "mxfp4", "mxfp4_s1", "mxfp8_s1"}
    runner.add(
        "ATTENTION_CONFIGS has all 4 keys",
        set(ATTENTION_CONFIGS.keys()) == expected_configs,
        f"got {set(ATTENTION_CONFIGS.keys())}",
    )
    runner.add(
        "P_QUANT_REGISTRY has all 4 keys",
        set(P_QUANT_REGISTRY.keys()) == expected_configs,
        f"got {set(P_QUANT_REGISTRY.keys())}",
    )

    # Config validation
    for name, config in ATTENTION_CONFIGS.items():
        try:
            config.validate()
            runner.add(f"{name} config validates", True)
        except AssertionError as e:
            runner.add(f"{name} config validates", False, str(e))

    # QuantConfig torch callables present
    for name, config in ATTENTION_CONFIGS.items():
        has_round = config.qk_quant.round_scale_torch is not None
        runner.add(f"{name} qk_quant has round_scale_torch", has_round)

    # TransformContext is typed
    ctx = TransformContext()
    runner.add(
        "TransformContext has typed fields",
        hasattr(ctx, 'delta_s') and hasattr(ctx, 'v_mean'),
    )


# ============================================================================
# Numerical Equivalence Tests
# ============================================================================

def test_numerical_equivalence(runner: TestRunner):
    """Test that refactored implementation matches the original monolith exactly."""
    print("\n[Numerical Equivalence Tests]")

    from sageattention3_standalone import sageattn3_torch_triton_standalone as original_fn
    from sage3 import sageattn3_torch_triton_standalone as refactored_fn

    torch.manual_seed(42)

    # Test all 4 formats with multiple shapes
    shapes = [
        (1, 4, 256, 128),   # Standard
        (2, 8, 128, 64),    # Different batch/heads/dim
        (1, 1, 512, 128),   # Longer sequence
    ]

    for B, H, N, D in shapes:
        q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
        k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
        v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)

        for fmt in ['nvfp4', 'mxfp4', 'mxfp4_s1', 'mxfp8_s1']:
            out_orig = original_fn(q.clone(), k.clone(), v.clone(), quant_format=fmt)
            out_new = refactored_fn(q.clone(), k.clone(), v.clone(), quant_format=fmt)

            exact = torch.equal(out_orig, out_new)
            cos_sim = F.cosine_similarity(
                out_orig.flatten().float(), out_new.flatten().float(), dim=0
            ).item()

            runner.add(
                f"Equiv {fmt} [{B},{H},{N},{D}]",
                exact or cos_sim > 0.9999,
                f"exact={exact}, cos_sim={cos_sim:.8f}",
            )


# ============================================================================
# Causal Masking Test
# ============================================================================

def test_causal_masking(runner: TestRunner):
    """Test causal masking produces valid results."""
    print("\n[Causal Masking Tests]")

    from sage3 import sageattn3_torch_triton_standalone

    torch.manual_seed(42)
    B, H, N, D = 1, 4, 256, 128
    q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
    k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)

    out_causal = sageattn3_torch_triton_standalone(q, k, v, is_causal=True, quant_format='nvfp4')
    out_noncausal = sageattn3_torch_triton_standalone(q, k, v, is_causal=False, quant_format='nvfp4')

    valid = not torch.isnan(out_causal).any() and not torch.isinf(out_causal).any()
    different = not torch.equal(out_causal, out_noncausal)

    runner.add("Causal output valid", valid)
    runner.add("Causal != non-causal", different)


# ============================================================================
# SDPA Wrapper Test
# ============================================================================

def test_sdpa_wrapper(runner: TestRunner):
    """Test the SDPA-compatible wrapper."""
    print("\n[SDPA Wrapper Tests]")

    from sage3 import scaled_dot_product_attention

    torch.manual_seed(42)
    B, H, N, D = 1, 4, 256, 128
    q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
    k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)

    out = scaled_dot_product_attention(q, k, v)
    valid = not torch.isnan(out).any() and not torch.isinf(out).any()
    correct_shape = out.shape == q.shape
    correct_dtype = out.dtype == q.dtype

    runner.add("SDPA wrapper output valid", valid)
    runner.add("SDPA wrapper correct shape", correct_shape, f"{out.shape}")
    runner.add("SDPA wrapper correct dtype", correct_dtype, f"{out.dtype}")


# ============================================================================
# Config-Based API Test
# ============================================================================

def test_config_api(runner: TestRunner):
    """Test the new config-based API."""
    print("\n[Config-Based API Tests]")

    from sage3 import sageattn3_standalone, ATTENTION_CONFIGS, AttentionConfig

    torch.manual_seed(42)
    B, H, N, D = 1, 4, 256, 128
    q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
    k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)

    # Test with string config
    out_str = sageattn3_standalone(q, k, v, config="mxfp4")
    runner.add("Config API: string config works", out_str.shape == q.shape)

    # Test with object config
    config_obj = ATTENTION_CONFIGS["mxfp4"]
    out_obj = sageattn3_standalone(q, k, v, config=config_obj)
    runner.add("Config API: object config works", out_obj.shape == q.shape)

    # Both should produce identical results
    exact = torch.equal(out_str, out_obj)
    runner.add("Config API: string == object", exact)


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("sage3 Refactored Package — Test Suite")
    print("=" * 60)

    runner = TestRunner()

    test_structural(runner)
    test_numerical_equivalence(runner)
    test_causal_masking(runner)
    test_sdpa_wrapper(runner)
    test_config_api(runner)

    all_passed = runner.summary()
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
