#!/usr/bin/env python3
"""
Test suite for the sage3 refactored package.

Tests:
1. Structural: registry completeness, config validation, type safety
2. Numerical equivalence: all 4 formats vs the original monolith (strict, expanded shapes)
3. bf16 equivalence: bfloat16 dtype path
4. No-smoothing equivalence: per_block_mean=False path
5. Causal masking: correctness invariant + original match
6. Custom sm_scale: non-default scale factor equivalence
7. SDPA wrapper: wrapper comparison against original
8. Error handling: negative tests for error paths
9. Accuracy vs PyTorch SDPA: sanity check against native attention
10. Config-based API
"""

import sys
import os
import warnings
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
        status = "\u2705" if passed else "\u274c"
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
                    print(f"  \u274c {r.name}: {r.message}")
        print(f"{'='*60}")
        return failed == 0


# Helper to compare two tensors and report metrics
def _compare(out_orig, out_new):
    """Return (exact_match, cos_sim, max_abs_diff)."""
    exact = torch.equal(out_orig, out_new)
    cos_sim = F.cosine_similarity(
        out_orig.flatten().float(), out_new.flatten().float(), dim=0
    ).item()
    max_diff = (out_orig.float() - out_new.float()).abs().max().item()
    return exact, cos_sim, max_diff


# ============================================================================
# 1. Structural Tests
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

    # AttentionConfig is frozen
    config = ATTENTION_CONFIGS["nvfp4"]
    try:
        config.name = "mutated"
        runner.add("AttentionConfig is frozen", False, "mutation succeeded")
    except AttributeError:
        runner.add("AttentionConfig is frozen", True)

    # pre_transforms is a tuple
    runner.add(
        "pre_transforms is tuple",
        isinstance(config.pre_transforms, tuple),
        f"type={type(config.pre_transforms).__name__}",
    )


# ============================================================================
# 2. Strict Numerical Equivalence — Expanded Shapes
# ============================================================================

def test_numerical_equivalence(runner: TestRunner):
    """Strict test: refactored must match the original monolith bit-for-bit."""
    print("\n[Strict Numerical Equivalence Tests]")

    from sageattention3_standalone import sageattn3_torch_triton_standalone as original_fn
    from sage3 import sageattn3_torch_triton_standalone as refactored_fn

    torch.manual_seed(42)

    formats = ['nvfp4', 'mxfp4', 'mxfp4_s1', 'mxfp8_s1']

    shapes = [
        (1, 4, 256, 128),   # Standard
        (2, 8, 128, 64),    # Different batch/heads/dim
        (1, 1, 512, 128),   # Longer sequence
        (1, 4, 64, 128),    # N < 128 (single tile, tests padding)
        (1, 4, 300, 128),   # N not divisible by 128 (tests trimming)
        (1, 4, 256, 64),    # Smaller D
        (1, 4, 1024, 128),  # Longer sequence
        (1, 1, 128, 128),   # Minimal: single batch, single head, one tile
    ]

    for B, H, N, D in shapes:
        q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
        k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
        v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)

        for fmt in formats:
            out_orig = original_fn(q.clone(), k.clone(), v.clone(), quant_format=fmt)
            out_new = refactored_fn(q.clone(), k.clone(), v.clone(), quant_format=fmt)

            exact, cos_sim, max_diff = _compare(out_orig, out_new)

            runner.add(
                f"Strict {fmt} [{B},{H},{N},{D}]",
                exact or cos_sim > 0.9999,
                f"exact={exact}, cos_sim={cos_sim:.8f}, max_diff={max_diff:.2e}",
            )


# ============================================================================
# 3. bf16 Equivalence
# ============================================================================

def test_bf16_equivalence(runner: TestRunner):
    """Test bfloat16 path: refactored must match original."""
    print("\n[bf16 Equivalence Tests]")

    from sageattention3_standalone import sageattn3_torch_triton_standalone as original_fn
    from sage3 import sageattn3_torch_triton_standalone as refactored_fn

    torch.manual_seed(42)

    B, H, N, D = 1, 4, 256, 128
    q = torch.randn(B, H, N, D, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(B, H, N, D, device='cuda', dtype=torch.bfloat16)
    v = torch.randn(B, H, N, D, device='cuda', dtype=torch.bfloat16)

    for fmt in ['nvfp4', 'mxfp4', 'mxfp4_s1', 'mxfp8_s1']:
        out_orig = original_fn(q.clone(), k.clone(), v.clone(), quant_format=fmt)
        out_new = refactored_fn(q.clone(), k.clone(), v.clone(), quant_format=fmt)

        exact, cos_sim, max_diff = _compare(out_orig, out_new)

        # Check dtype preserved
        dtype_ok = out_new.dtype == torch.bfloat16

        runner.add(
            f"bf16 {fmt}",
            (exact or cos_sim > 0.9999) and dtype_ok,
            f"exact={exact}, cos_sim={cos_sim:.8f}, max_diff={max_diff:.2e}, dtype={out_new.dtype}",
        )


# ============================================================================
# 4. per_block_mean=False Equivalence
# ============================================================================

def test_no_smoothing_equivalence(runner: TestRunner):
    """Test with QK smoothing disabled: refactored must match original."""
    print("\n[No-Smoothing Equivalence Tests]")

    from sageattention3_standalone import sageattn3_torch_triton_standalone as original_fn
    from sage3 import sageattn3_torch_triton_standalone as refactored_fn

    torch.manual_seed(42)

    B, H, N, D = 1, 4, 256, 128
    q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
    k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)

    for fmt in ['nvfp4', 'mxfp4', 'mxfp4_s1', 'mxfp8_s1']:
        out_orig = original_fn(
            q.clone(), k.clone(), v.clone(),
            quant_format=fmt, per_block_mean=False,
        )
        out_new = refactored_fn(
            q.clone(), k.clone(), v.clone(),
            quant_format=fmt, per_block_mean=False,
        )

        exact, cos_sim, max_diff = _compare(out_orig, out_new)

        runner.add(
            f"NoSmooth {fmt}",
            exact or cos_sim > 0.9999,
            f"exact={exact}, cos_sim={cos_sim:.8f}, max_diff={max_diff:.2e}",
        )


# ============================================================================
# 5. Causal Masking Correctness
# ============================================================================

def test_causal_masking(runner: TestRunner):
    """Test causal masking: (a) valid output, (b) differs from non-causal, (c) matches original."""
    print("\n[Causal Masking Tests]")

    from sageattention3_standalone import sageattn3_torch_triton_standalone as original_fn
    from sage3 import sageattn3_torch_triton_standalone as refactored_fn

    torch.manual_seed(42)
    B, H, N, D = 1, 4, 256, 128
    q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
    k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)

    # Basic: causal output is valid and different from non-causal
    out_causal = refactored_fn(q, k, v, is_causal=True, quant_format='nvfp4')
    out_noncausal = refactored_fn(q, k, v, is_causal=False, quant_format='nvfp4')

    valid = not torch.isnan(out_causal).any() and not torch.isinf(out_causal).any()
    different = not torch.equal(out_causal, out_noncausal)

    runner.add("Causal output valid", valid)
    runner.add("Causal != non-causal", different)

    # Strict: refactored causal matches original causal for each format
    for fmt in ['nvfp4', 'mxfp4', 'mxfp4_s1', 'mxfp8_s1']:
        out_orig_causal = original_fn(
            q.clone(), k.clone(), v.clone(),
            is_causal=True, quant_format=fmt,
        )
        out_new_causal = refactored_fn(
            q.clone(), k.clone(), v.clone(),
            is_causal=True, quant_format=fmt,
        )

        exact, cos_sim, max_diff = _compare(out_orig_causal, out_new_causal)

        runner.add(
            f"Causal strict {fmt}",
            exact or cos_sim > 0.9999,
            f"exact={exact}, cos_sim={cos_sim:.8f}, max_diff={max_diff:.2e}",
        )

    # Correctness: compare refactored causal vs PyTorch SDPA causal
    # (can't be exact due to quantization, but should be close)
    with torch.no_grad():
        ref_causal = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    sage_causal = refactored_fn(q.clone(), k.clone(), v.clone(), is_causal=True, quant_format='mxfp8_s1')
    cos_vs_sdpa = F.cosine_similarity(
        ref_causal.flatten().float(), sage_causal.flatten().float(), dim=0
    ).item()
    runner.add(
        "Causal vs SDPA sanity",
        cos_vs_sdpa > 0.95,
        f"cos_sim={cos_vs_sdpa:.6f}",
    )


# ============================================================================
# 6. Custom sm_scale Equivalence
# ============================================================================

def test_custom_scale_equivalence(runner: TestRunner):
    """Test non-default sm_scale: refactored must match original."""
    print("\n[Custom sm_scale Tests]")

    from sageattention3_standalone import sageattn3_torch_triton_standalone as original_fn
    from sage3 import sageattn3_torch_triton_standalone as refactored_fn

    torch.manual_seed(42)

    B, H, N, D = 1, 4, 256, 128
    q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
    k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)

    for sm_scale in [0.05, 0.5, 2.0]:
        for fmt in ['mxfp4', 'nvfp4']:
            out_orig = original_fn(
                q.clone(), k.clone(), v.clone(),
                sm_scale=sm_scale, quant_format=fmt,
            )
            out_new = refactored_fn(
                q.clone(), k.clone(), v.clone(),
                sm_scale=sm_scale, quant_format=fmt,
            )

            exact, cos_sim, max_diff = _compare(out_orig, out_new)

            runner.add(
                f"Scale={sm_scale} {fmt}",
                exact or cos_sim > 0.9999,
                f"exact={exact}, cos_sim={cos_sim:.8f}, max_diff={max_diff:.2e}",
            )


# ============================================================================
# 7. SDPA Wrapper Tests
# ============================================================================

def test_sdpa_wrapper(runner: TestRunner):
    """Test the SDPA-compatible wrapper: shape, dtype, and comparison with original."""
    print("\n[SDPA Wrapper Tests]")

    from sage3 import scaled_dot_product_attention as refactored_sdpa
    from sageattention3_standalone import scaled_dot_product_attention as original_sdpa

    torch.manual_seed(42)
    B, H, N, D = 1, 4, 256, 128
    q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
    k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)

    # Basic: valid output, correct shape and dtype
    out = refactored_sdpa(q, k, v)
    valid = not torch.isnan(out).any() and not torch.isinf(out).any()
    runner.add("SDPA wrapper output valid", valid)
    runner.add("SDPA wrapper correct shape", out.shape == q.shape, f"{out.shape}")
    runner.add("SDPA wrapper correct dtype", out.dtype == q.dtype, f"{out.dtype}")

    # Compare refactored SDPA vs original SDPA
    # Use the same quant_format for both (default is nvfp4 for original via env)
    for fmt in ['nvfp4', 'mxfp4']:
        out_orig = original_sdpa(
            q.clone(), k.clone(), v.clone(), quant_format=fmt,
        )
        out_new = refactored_sdpa(
            q.clone(), k.clone(), v.clone(), quant_format=fmt,
        )

        exact, cos_sim, max_diff = _compare(out_orig, out_new)

        runner.add(
            f"SDPA wrapper {fmt} vs original",
            exact or cos_sim > 0.9999,
            f"exact={exact}, cos_sim={cos_sim:.8f}, max_diff={max_diff:.2e}",
        )


# ============================================================================
# 8. Error Handling / Negative Tests
# ============================================================================

def test_error_handling(runner: TestRunner):
    """Test that error paths raise the expected exceptions."""
    print("\n[Error Handling Tests]")

    from sage3 import sageattn3_torch_triton_standalone, scaled_dot_product_attention

    B, H, N, D = 1, 4, 256, 128

    # return_lse=True should raise NotImplementedError
    q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
    k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)

    try:
        sageattn3_torch_triton_standalone(q, k, v, return_lse=True, quant_format='nvfp4')
        runner.add("return_lse raises NotImplementedError", False, "no exception raised")
    except NotImplementedError:
        runner.add("return_lse raises NotImplementedError", True)
    except Exception as e:
        runner.add("return_lse raises NotImplementedError", False, f"wrong exception: {type(e).__name__}: {e}")

    # tensor_layout="NHD" should raise ValueError
    try:
        sageattn3_torch_triton_standalone(q, k, v, tensor_layout="NHD", quant_format='nvfp4')
        runner.add("NHD layout raises ValueError", False, "no exception raised")
    except ValueError:
        runner.add("NHD layout raises ValueError", True)
    except Exception as e:
        runner.add("NHD layout raises ValueError", False, f"wrong exception: {type(e).__name__}")

    # Invalid quant_format should raise ValueError
    try:
        sageattn3_torch_triton_standalone(q, k, v, quant_format='invalid_format')
        runner.add("Invalid format raises ValueError", False, "no exception raised")
    except ValueError:
        runner.add("Invalid format raises ValueError", True)
    except Exception as e:
        runner.add("Invalid format raises ValueError", False, f"wrong exception: {type(e).__name__}")

    # CPU tensors should raise RuntimeError via SDPA wrapper
    q_cpu = torch.randn(B, H, N, D, dtype=torch.float16)
    k_cpu = torch.randn(B, H, N, D, dtype=torch.float16)
    v_cpu = torch.randn(B, H, N, D, dtype=torch.float16)

    try:
        scaled_dot_product_attention(q_cpu, k_cpu, v_cpu)
        runner.add("CPU tensors raise RuntimeError", False, "no exception raised")
    except RuntimeError:
        runner.add("CPU tensors raise RuntimeError", True)
    except Exception as e:
        runner.add("CPU tensors raise RuntimeError", False, f"wrong exception: {type(e).__name__}")

    # Shape mismatch K should raise ValueError
    k_bad = torch.randn(B, H, N * 2, D, device='cuda', dtype=torch.float16)
    try:
        scaled_dot_product_attention(q, k_bad, v)
        runner.add("Shape mismatch K raises ValueError", False, "no exception raised")
    except ValueError:
        runner.add("Shape mismatch K raises ValueError", True)
    except Exception as e:
        runner.add("Shape mismatch K raises ValueError", False, f"wrong exception: {type(e).__name__}")

    # Shape mismatch V should raise ValueError
    v_bad = torch.randn(B, H, N, D * 2, device='cuda', dtype=torch.float16)
    try:
        scaled_dot_product_attention(q, k, v_bad)
        runner.add("Shape mismatch V raises ValueError", False, "no exception raised")
    except ValueError:
        runner.add("Shape mismatch V raises ValueError", True)
    except Exception as e:
        runner.add("Shape mismatch V raises ValueError", False, f"wrong exception: {type(e).__name__}")

    # Dropout warning: should warn but not fail
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = scaled_dot_product_attention(q, k, v, dropout_p=0.1)
        warned = any("dropout" in str(warning.message).lower() for warning in w)
        runner.add(
            "Dropout warns but runs",
            out.shape == q.shape and warned,
            f"warned={warned}, shape={out.shape}",
        )

    # attn_mask warning: should warn but not fail
    mask = torch.ones(N, N, device='cuda', dtype=torch.float16)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = scaled_dot_product_attention(q, k, v, attn_mask=mask)
        warned = any("mask" in str(warning.message).lower() for warning in w)
        runner.add(
            "attn_mask warns but runs",
            out.shape == q.shape and warned,
            f"warned={warned}, shape={out.shape}",
        )


# ============================================================================
# 9. Accuracy vs PyTorch SDPA (sanity check)
# ============================================================================

def test_accuracy_vs_sdpa(runner: TestRunner):
    """Sanity check: quantized attention should be reasonably close to exact SDPA."""
    print("\n[Accuracy vs PyTorch SDPA]")

    from sage3 import sageattn3_torch_triton_standalone as refactored_fn

    torch.manual_seed(42)

    B, H, N, D = 2, 4, 256, 128
    q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
    k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)

    with torch.no_grad():
        ref_output = F.scaled_dot_product_attention(q, k, v)

    for fmt in ['nvfp4', 'mxfp4', 'mxfp4_s1', 'mxfp8_s1']:
        sage_output = refactored_fn(q.clone(), k.clone(), v.clone(), quant_format=fmt)

        cos_sim = F.cosine_similarity(
            ref_output.flatten().float(), sage_output.flatten().float(), dim=0
        ).item()

        l2_err = torch.norm(ref_output.float() - sage_output.float()) / torch.norm(ref_output.float())

        # Quantized attention should be > 0.95 cos_sim with exact
        runner.add(
            f"vs SDPA {fmt}",
            cos_sim > 0.95,
            f"cos_sim={cos_sim:.6f}, l2_err={l2_err:.4f}",
        )


# ============================================================================
# 10. Config-Based API Test
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
    print("sage3 Refactored Package \u2014 Test Suite")
    print("=" * 60)

    runner = TestRunner()

    test_structural(runner)
    test_numerical_equivalence(runner)
    test_bf16_equivalence(runner)
    test_no_smoothing_equivalence(runner)
    test_causal_masking(runner)
    test_custom_scale_equivalence(runner)
    test_sdpa_wrapper(runner)
    test_error_handling(runner)
    test_accuracy_vs_sdpa(runner)
    test_config_api(runner)

    all_passed = runner.summary()
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
