#!/usr/bin/env python3
"""Targeted correctness checks for the opt-in FP8 P x V kernel."""

import argparse

import torch
import torch.nn.functional as F

import fp4quant_cuda
from sageattn3 import sageattn3_blackwell


SEQUENCE_LENGTHS = (31, 32, 33, 63, 64, 65, 127, 128, 129)
HEAD_DIMS = (64, 128)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    label: str,
    max_error_limit: float = 0.75,
    mean_error_limit: float = 0.06,
) -> None:
    if not torch.isfinite(actual).all():
        raise AssertionError(f"{label}: output contains non-finite values")
    error = (actual - expected).float().abs()
    max_error = error.max().item()
    mean_error = error.mean().item()
    if max_error > max_error_limit or mean_error > mean_error_limit:
        raise AssertionError(
            f"{label}: max error {max_error:.6f}, "
            f"mean error {mean_error:.6f}"
        )
    print(
        f"PASS {label}: max error {max_error:.6f}, "
        f"mean error {mean_error:.6f}"
    )


def make_inputs(
    batch: int,
    q_heads: int,
    kv_heads: int,
    sequence_length: int,
    head_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = torch.randn(
        batch,
        q_heads,
        sequence_length,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    k = torch.randn(
        batch,
        kv_heads,
        sequence_length,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    v = torch.randn_like(k)
    return q, k, v


def validate_quantizer(device: torch.device) -> None:
    for dtype in (torch.float16, torch.bfloat16):
        hnd = torch.zeros((2, 3, 17, 64), device=device, dtype=dtype)
        hnd_out = torch.empty(
            (2, 3, 64, 128),
            device=device,
            dtype=torch.float8_e4m3fn,
        )
        hnd_scale = torch.empty((2, 3, 64), device=device)
        fp4quant_cuda.scaled_fp8_quant_trans(
            hnd,
            hnd_out,
            hnd_scale,
            1,
        )

        nhd = hnd.permute(0, 2, 1, 3).contiguous()
        nhd_out = torch.empty(
            (2, 64, 3, 128),
            device=device,
            dtype=torch.float8_e4m3fn,
        )
        nhd_scale = torch.empty((2, 64, 3), device=device)
        fp4quant_cuda.scaled_fp8_quant_trans(
            nhd,
            nhd_out,
            nhd_scale,
            0,
        )
        torch.cuda.synchronize(device)
        if (
            hnd_out.float().abs().max().item() != 0.0
            or hnd_scale.abs().max().item() != 0.0
            or nhd_out.float().abs().max().item() != 0.0
            or nhd_scale.abs().max().item() != 0.0
        ):
            raise AssertionError(f"zero-scale quantization failed for {dtype}")
        print(f"PASS quantizer dtype={dtype}")


def validate_boundaries(device: torch.device) -> None:
    for head_dim in HEAD_DIMS:
        for sequence_length in SEQUENCE_LENGTHS:
            q, k, v = make_inputs(
                1,
                2,
                2,
                sequence_length,
                head_dim,
                device,
            )
            for is_causal in (False, True):
                expected = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    is_causal=is_causal,
                )
                actual = sageattn3_blackwell(
                    q,
                    k,
                    v,
                    is_causal=is_causal,
                    kernel_variant="fp8_pv",
                )
                assert_close(
                    actual,
                    expected,
                    f"S={sequence_length} D={head_dim} causal={is_causal}",
                )


def validate_gqa(device: torch.device) -> None:
    for kv_heads in (1, 2):
        q, k, v = make_inputs(1, 4, kv_heads, 65, 128, device)
        for is_causal in (False, True):
            expected = F.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=is_causal,
                enable_gqa=True,
            )
            actual = sageattn3_blackwell(
                q,
                k,
                v,
                is_causal=is_causal,
                kernel_variant="fp8_pv",
            )
            assert_close(
                actual,
                expected,
                f"GQA Hq=4 Hkv={kv_heads} causal={is_causal}",
            )

    q, k, v = make_inputs(1, 4, 1, 128, 64, device)
    head_offsets = torch.tensor(
        [2.0, -2.0, 4.0, -4.0],
        device=device,
        dtype=q.dtype,
    ).view(1, 4, 1, 1)
    q = q * 0.1 + head_offsets
    expected = F.scaled_dot_product_attention(
        q,
        k,
        v,
        enable_gqa=True,
    )
    actual = sageattn3_blackwell(
        q,
        k,
        v,
        kernel_variant="fp8_pv",
    )
    assert_close(actual, expected, "GQA per-query-head delta correction")


def validate_fp16(device: torch.device) -> None:
    q = torch.randn((1, 2, 65, 128), device=device, dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    expected = F.scaled_dot_product_attention(q, k, v)
    actual = sageattn3_blackwell(
        q,
        k,
        v,
        kernel_variant="fp8_pv",
    )
    assert_close(actual, expected, "FP16")


def validate_persistent_reuse(device: torch.device) -> None:
    q, k, v = make_inputs(1, 2, 2, 16384, 64, device)
    expected = F.scaled_dot_product_attention(q, k, v)
    actual = sageattn3_blackwell(
        q,
        k,
        v,
        kernel_variant="fp8_pv",
    )
    assert_close(actual, expected, "persistent CTA reuse at S=16384")


def validate_mask_rejection(device: torch.device) -> None:
    q, k, v = make_inputs(1, 1, 1, 32, 64, device)
    try:
        sageattn3_blackwell(
            q,
            k,
            v,
            attn_mask=torch.zeros((32, 32), device=device),
            kernel_variant="fp8_pv",
        )
    except NotImplementedError:
        print("PASS all-masked rows are explicitly unsupported")
        return
    raise AssertionError("attn_mask must fail explicitly")

def validate_rectangular_causal_rejection(device: torch.device) -> None:
    q = torch.randn((1, 2, 129, 64), device=device, dtype=torch.bfloat16)
    k = torch.randn((1, 2, 65, 64), device=device, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    try:
        sageattn3_blackwell(
            q,
            k,
            v,
            is_causal=True,
            kernel_variant="fp8_pv",
        )
    except ValueError:
        print("PASS rectangular causal attention is explicitly unsupported")
        return
    raise AssertionError("rectangular causal attention must fail explicitly")


def validate_cuda_graph(device: torch.device) -> None:
    q, k, v = make_inputs(1, 2, 2, 128, 64, device)
    for _ in range(2):
        sageattn3_blackwell(q, k, v, kernel_variant="fp8_pv")
    torch.cuda.synchronize(device)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output = sageattn3_blackwell(
            q,
            k,
            v,
            kernel_variant="fp8_pv",
        )
    graph.replay()
    torch.cuda.synchronize(device)
    first_output = captured_output.clone()

    q.copy_(torch.randn_like(q))
    k.copy_(torch.randn_like(k))
    v.copy_(torch.randn_like(v))
    graph.replay()
    torch.cuda.synchronize(device)
    replay_output = captured_output.clone()
    eager_output = sageattn3_blackwell(
        q,
        k,
        v,
        kernel_variant="fp8_pv",
    )
    if torch.equal(first_output, replay_output):
        raise AssertionError("CUDA graph replay ignored changed input contents")
    assert_close(
        replay_output,
        eager_output,
        "CUDA graph replay",
        max_error_limit=0.0,
        mean_error_limit=0.0,
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("A CUDA device is required")
    torch.manual_seed(args.seed)
    with torch.cuda.device(device):
        validate_quantizer(device)
        validate_boundaries(device)
        validate_gqa(device)
        validate_fp16(device)
        validate_persistent_reuse(device)
        validate_mask_rejection(device)
        validate_rectangular_causal_rejection(device)
        validate_cuda_graph(device)
    print("All FP8 P x V validation checks passed")


if __name__ == "__main__":
    main()
