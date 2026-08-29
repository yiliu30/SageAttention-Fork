#!/usr/bin/env python3
"""Benchmark SageAttention 3 against PyTorch Flash SDPA and GEMM references.

The QK^T reference creates an [batch, heads, sequence, sequence] tensor.
Unlike the fused attention implementations, it writes this tensor to global
memory, so its throughput is intentionally reported as a separate metric. Its
key tensor is pre-transposed and contiguous outside the timed region. Large
square BF16 and native NVFP4 GEMMs provide independent peak references.
"""

import argparse
import statistics
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from sageattn3 import sageattn3_blackwell
from sageattn3.api import (
    blockscaled_fp4_attn,
    preprocess_qkv,
    scale_and_quant_fp4,
    scale_and_quant_fp4_permute,
    scale_and_quant_fp4_transpose,
    scale_and_quant_fp8_transpose,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--heads", type=int, default=40)
    parser.add_argument("--sequence-length", type=int, default=16384)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--iterations", type=int, default=75)
    parser.add_argument(
        "--peak-matmul-size",
        type=int,
        default=16384,
        help="Matrix dimension for the independent square GEMM peak reference.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--causal", action="store_true")
    return parser.parse_args()


def benchmark(
    operation: Callable[[], object],
    warmup: int,
    iterations: int,
    device: torch.device,
) -> float:
    with torch.cuda.device(device):
        for _ in range(warmup):
            operation()
        torch.cuda.synchronize(device)

        timings_ms = []
        for _ in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            operation()
            end.record()
            end.synchronize()
            timings_ms.append(start.elapsed_time(end))
    return statistics.median(timings_ms)


def benchmark_interleaved(
    operations: tuple[Callable[[], object], Callable[[], object]],
    warmup: int,
    iterations: int,
    device: torch.device,
) -> tuple[float, float]:
    with torch.cuda.device(device):
        for _ in range(warmup):
            operations[0]()
            operations[1]()
        torch.cuda.synchronize(device)

        timings_ms = [[], []]
        for iteration in range(iterations):
            order = (0, 1) if iteration % 2 == 0 else (1, 0)
            for operation_index in order:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                operations[operation_index]()
                end.record()
                end.synchronize()
                timings_ms[operation_index].append(start.elapsed_time(end))
    return tuple(statistics.median(values) for values in timings_ms)


def make_nvfp4_gemm_inputs(
    size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    packed_size = size // 2
    a = torch.randint(
        0,
        256,
        (size, packed_size),
        device=device,
        dtype=torch.uint8,
    ).view(torch.float4_e2m1fn_x2)
    b = (
        torch.randint(
            0,
            256,
            (size, packed_size),
            device=device,
            dtype=torch.uint8,
        )
        .view(torch.float4_e2m1fn_x2)
        .t()
    )

    k_blocks = (size + 15) // 16
    padded_k_blocks = (k_blocks + 3) // 4 * 4
    scale_elements = 128 * ((size + 127) // 128) * padded_k_blocks
    scale_a = torch.ones(
        scale_elements,
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    scale_b = torch.ones_like(scale_a)
    return a, b, scale_a, scale_b


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError(f"CUDA device required, got {device}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    for name, value in {
        "batch size": args.batch_size,
        "heads": args.heads,
        "sequence length": args.sequence_length,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "peak matmul size": args.peak_matmul_size,
    }.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    if args.head_dim not in (64, 128):
        raise ValueError(
            f"SageAttention 3 NVFP4 supports head dimensions 64 or 128, got {args.head_dim}"
        )
    if args.head_dim != 128:
        raise ValueError("The two-CTA benchmark requires head dimension 128")
    if args.peak_matmul_size % 32 != 0:
        raise ValueError(
            "peak matmul size must be divisible by 32 for packed NVFP4 GEMM, "
            f"got {args.peak_matmul_size}"
        )

    torch.manual_seed(args.seed)
    q = torch.randn(
        args.batch_size,
        args.heads,
        args.sequence_length,
        args.head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    k_transposed = k.transpose(-2, -1).contiguous()
    pure_q, pure_k, pure_v, delta_s = preprocess_qkv(q.clone(), k.clone(), v.clone())
    pure_q_list = scale_and_quant_fp4(pure_q)
    pure_k_list = scale_and_quant_fp4_permute(pure_k)
    pure_v_list = scale_and_quant_fp4_transpose(pure_v)
    pure_fp8_v_list = scale_and_quant_fp8_transpose(pure_v)
    candidate_q, candidate_k, candidate_v, candidate_delta_s = preprocess_qkv(
        q.clone(),
        k.clone(),
        v.clone(),
        q_group_size=64,
    )
    candidate_q_list = scale_and_quant_fp4(candidate_q)
    candidate_k_list = scale_and_quant_fp4_permute(candidate_k)
    candidate_v_list = scale_and_quant_fp4_transpose(candidate_v)
    peak_a = torch.randn(
        args.peak_matmul_size,
        args.peak_matmul_size,
        device=device,
        dtype=torch.bfloat16,
    )
    peak_b = torch.randn_like(peak_a)
    nvfp4_a, nvfp4_b, nvfp4_scale_a, nvfp4_scale_b = make_nvfp4_gemm_inputs(
        args.peak_matmul_size,
        device,
    )

    score_elements = (
        args.sequence_length * (args.sequence_length + 1) // 2
        if args.causal
        else args.sequence_length**2
    )
    qk_flops = 2 * args.batch_size * args.heads * score_elements * args.head_dim
    attention_flops = 2 * qk_flops
    peak_matmul_flops = 2 * args.peak_matmul_size**3
    score_tensor_gib = (
        args.batch_size
        * args.heads
        * args.sequence_length**2
        * q.element_size()
        / 2**30
    )

    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        sdpa_ms = benchmark(
            lambda: F.scaled_dot_product_attention(q, k, v, is_causal=args.causal),
            args.warmup,
            args.iterations,
            device,
        )
    sage_ms = benchmark(
        lambda: sageattn3_blackwell(q, k, v, is_causal=args.causal),
        args.warmup,
        args.iterations,
        device,
    )
    candidate_sage_ms = benchmark(
        lambda: sageattn3_blackwell(
            q,
            k,
            v,
            is_causal=args.causal,
            kernel_variant="two_cta",
        ),
        args.warmup,
        args.iterations,
        device,
    )
    fp8_pv_sage_ms, fp8_pv_register_sage_ms = benchmark_interleaved(
        (
            lambda: sageattn3_blackwell(
                q,
                k,
                v,
                is_causal=args.causal,
                kernel_variant="fp8_pv",
            ),
            lambda: sageattn3_blackwell(
                q,
                k,
                v,
                is_causal=args.causal,
                kernel_variant="fp8_pv_register",
            ),
        ),
        args.warmup,
        args.iterations,
        device,
    )
    pure_sage_ms = benchmark(
        lambda: blockscaled_fp4_attn(
            pure_q_list,
            pure_k_list,
            pure_v_list,
            delta_s,
            args.sequence_length,
            is_causal=args.causal,
        ),
        args.warmup,
        args.iterations,
        device,
    )
    candidate_pure_sage_ms = benchmark(
        lambda: blockscaled_fp4_attn(
            candidate_q_list,
            candidate_k_list,
            candidate_v_list,
            candidate_delta_s,
            args.sequence_length,
            is_causal=args.causal,
            use_two_cta=True,
        ),
        args.warmup,
        args.iterations,
        device,
    )
    fp8_pv_pure_sage_ms, fp8_pv_register_pure_sage_ms = benchmark_interleaved(
        (
            lambda: blockscaled_fp4_attn(
                pure_q_list,
                pure_k_list,
                pure_fp8_v_list,
                delta_s,
                args.sequence_length,
                is_causal=args.causal,
                fp8_pv=True,
            ),
            lambda: blockscaled_fp4_attn(
                pure_q_list,
                pure_k_list,
                pure_fp8_v_list,
                delta_s,
                args.sequence_length,
                is_causal=args.causal,
                fp8_pv_register=True,
            ),
        ),
        args.warmup,
        args.iterations,
        device,
    )
    pure_sage_interleaved_ms, bypass_p_packing_ms = benchmark_interleaved(
        (
            lambda: blockscaled_fp4_attn(
                pure_q_list,
                pure_k_list,
                pure_v_list,
                delta_s,
                args.sequence_length,
                is_causal=args.causal,
            ),
            lambda: blockscaled_fp4_attn(
                pure_q_list,
                pure_k_list,
                pure_v_list,
                delta_s,
                args.sequence_length,
                is_causal=args.causal,
                bypass_p_packing=True,
            ),
        ),
        args.warmup,
        args.iterations,
        device,
    )
    matmul_ms = benchmark(
        lambda: torch.matmul(q, k_transposed),
        args.warmup,
        args.iterations,
        device,
    )
    peak_matmul_ms = benchmark(
        lambda: torch.matmul(peak_a, peak_b),
        args.warmup,
        args.iterations,
        device,
    )
    nvfp4_matmul_ms = benchmark(
        lambda: torch._scaled_mm(
            nvfp4_a,
            nvfp4_b,
            nvfp4_scale_a,
            nvfp4_scale_b,
            out_dtype=torch.bfloat16,
        ),
        args.warmup,
        args.iterations,
        device,
    )
    sdpa_tops = attention_flops / (sdpa_ms * 1e9)
    sage_tops = attention_flops / (sage_ms * 1e9)
    candidate_sage_tops = attention_flops / (candidate_sage_ms * 1e9)
    fp8_pv_sage_tops = attention_flops / (fp8_pv_sage_ms * 1e9)
    fp8_pv_register_sage_tops = attention_flops / (
        fp8_pv_register_sage_ms * 1e9
    )
    pure_sage_tops = attention_flops / (pure_sage_ms * 1e9)
    candidate_pure_sage_tops = attention_flops / (
        candidate_pure_sage_ms * 1e9
    )
    fp8_pv_pure_sage_tops = attention_flops / (
        fp8_pv_pure_sage_ms * 1e9
    )
    fp8_pv_register_pure_sage_tops = attention_flops / (
        fp8_pv_register_pure_sage_ms * 1e9
    )
    qk_tops = qk_flops / (matmul_ms * 1e9)
    peak_matmul_tops = peak_matmul_flops / (peak_matmul_ms * 1e9)
    nvfp4_matmul_tops = peak_matmul_flops / (nvfp4_matmul_ms * 1e9)

    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(
        "Config: "
        f"B={args.batch_size}, H={args.heads}, S={args.sequence_length}, "
        f"D={args.head_dim}, BF16, causal={args.causal}"
    )
    print(f"Materialized QK^T score tensor: {score_tensor_gib:.2f} GiB")
    print()
    print(f"{'Method':<42} {'Median (ms)':>12} {'TOPS':>10}")
    print("-" * 55)
    print(
        f"{'PyTorch SDPA (Flash)':<42} {sdpa_ms:>12.3f} {sdpa_tops:>10.1f}"
    )
    print(
        f"{'SageAttention 3 NVFP4':<42} {sage_ms:>12.3f} {sage_tops:>10.1f}"
    )
    print(
        f"{'SageAttention 3 NVFP4 (two CTA)':<42} "
        f"{candidate_sage_ms:>12.3f} {candidate_sage_tops:>10.1f}"
    )
    print(
        f"{'SageAttention 3 FP8 P x V (shared)':<42} "
        f"{fp8_pv_sage_ms:>12.3f} {fp8_pv_sage_tops:>10.1f}"
    )
    print(
        f"{'SageAttention 3 FP8 P x V (register)':<42} "
        f"{fp8_pv_register_sage_ms:>12.3f} "
        f"{fp8_pv_register_sage_tops:>10.1f}"
    )
    print(
        f"{'SageAttention 3 FP4 kernel only':<42} {pure_sage_ms:>12.3f} "
        f"{pure_sage_tops:>10.1f}"
    )
    print(
        f"{'SageAttention 3 FP4 kernel only (two CTA)':<42} "
        f"{candidate_pure_sage_ms:>12.3f} {candidate_pure_sage_tops:>10.1f}"
    )
    print(
        f"{'SageAttention 3 FP8 P x V kernel (shared)':<42} "
        f"{fp8_pv_pure_sage_ms:>12.3f} {fp8_pv_pure_sage_tops:>10.1f}"
    )
    print(
        f"{'SageAttention 3 FP8 P x V kernel (register)':<42} "
        f"{fp8_pv_register_pure_sage_ms:>12.3f} "
        f"{fp8_pv_register_pure_sage_tops:>10.1f}"
    )
    print(
        f"{'PyTorch matmul (QK^T, K transposed)':<42} {matmul_ms:>12.3f} {qk_tops:>10.1f}"
    )
    print(
        f"{f'PyTorch matmul ({args.peak_matmul_size}^3)':<42} {peak_matmul_ms:>12.3f} "
        f"{peak_matmul_tops:>10.1f}"
    )
    print(
        f"{f'PyTorch NVFP4 scaled_mm ({args.peak_matmul_size}^3)':<42} "
        f"{nvfp4_matmul_ms:>12.3f} {nvfp4_matmul_tops:>10.1f}"
    )
    print("\nRatios relative to the large GEMM references:")
    print(f"  PyTorch SDPA effective TOPS / GEMM TOPS: {sdpa_tops / peak_matmul_tops:.2f}x")
    print(f"  SageAttention effective TOPS / GEMM TOPS: {sage_tops / peak_matmul_tops:.2f}x")
    print(
        "  SageAttention FP4 kernel-only effective TOPS / GEMM TOPS: "
        f"{pure_sage_tops / peak_matmul_tops:.2f}x"
    )
    print(
        "  PyTorch SDPA effective TOPS / NVFP4 GEMM TOPS: "
        f"{sdpa_tops / nvfp4_matmul_tops:.2f}x"
    )
    print(
        "  SageAttention effective TOPS / NVFP4 GEMM TOPS: "
        f"{sage_tops / nvfp4_matmul_tops:.2f}x"
    )
    print(
        "  SageAttention FP4 kernel-only effective TOPS / NVFP4 GEMM TOPS: "
        f"{pure_sage_tops / nvfp4_matmul_tops:.2f}x"
    )
    print(
        "  Two-CTA FP4 kernel-only effective TOPS / NVFP4 GEMM TOPS: "
        f"{candidate_pure_sage_tops / nvfp4_matmul_tops:.2f}x"
    )
    print(
        "  Shared FP8 P x V kernel-only effective TOPS / NVFP4 GEMM TOPS: "
        f"{fp8_pv_pure_sage_tops / nvfp4_matmul_tops:.2f}x"
    )
    print(
        "  Register FP8 P x V kernel-only effective TOPS / NVFP4 GEMM TOPS: "
        f"{fp8_pv_register_pure_sage_tops / nvfp4_matmul_tops:.2f}x"
    )
    print(f"  SageAttention / PyTorch SDPA: {sage_tops / sdpa_tops:.2f}x")
    print("\nTwo-CTA speedup over baseline:")
    print(f"  End-to-end: {sage_ms / candidate_sage_ms:.3f}x")
    print(f"  FP4 kernel only: {pure_sage_ms / candidate_pure_sage_ms:.3f}x")
    print("\nShared FP8 P x V speedup over NVFP4:")
    print(f"  End-to-end: {sage_ms / fp8_pv_sage_ms:.3f}x")
    print(f"  Kernel only: {pure_sage_ms / fp8_pv_pure_sage_ms:.3f}x")
    print("\nRegister-remap FP8 P x V speedup:")
    print(
        "  Versus shared FP8 end-to-end: "
        f"{fp8_pv_sage_ms / fp8_pv_register_sage_ms:.3f}x"
    )
    print(
        "  Versus shared FP8 kernel only: "
        f"{fp8_pv_pure_sage_ms / fp8_pv_register_pure_sage_ms:.3f}x"
    )
    print(
        "  Versus NVFP4 end-to-end: "
        f"{sage_ms / fp8_pv_register_sage_ms:.3f}x"
    )
    print(
        "  Versus NVFP4 kernel only: "
        f"{pure_sage_ms / fp8_pv_register_pure_sage_ms:.3f}x"
    )
    exposed_p_packing_ms = pure_sage_interleaved_ms - bypass_p_packing_ms
    exposed_p_packing_fraction = exposed_p_packing_ms / pure_sage_interleaved_ms
    bypass_p_packing_tops = attention_flops / (bypass_p_packing_ms * 1e9)
    print("\nP conversion/packing ablation (interleaved):")
    print(f"  Normal FP4 kernel: {pure_sage_interleaved_ms:.3f} ms")
    print(f"  Bypassed P packing: {bypass_p_packing_ms:.3f} ms")
    print(f"  Bypassed P packing effective TOPS: {bypass_p_packing_tops:.1f}")
    print(f"  Exposed P packing cost: {exposed_p_packing_ms:.3f} ms")
    print(f"  Fraction of normal latency: {exposed_p_packing_fraction:.1%}")
    print(f"  Bypass speedup: {pure_sage_interleaved_ms / bypass_p_packing_ms:.3f}x")


if __name__ == "__main__":
    main()
