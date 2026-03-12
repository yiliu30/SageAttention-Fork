#!/usr/bin/env python3
"""Quick SageAttention3 demo matching the README usage snippet.

The script seeds random Q/K/V tensors, calls ``sageattn3_blackwell`` just
like the README example, and compares the output against PyTorch's
``scaled_dot_product_attention`` for a simple correctness smoke test.
"""

from __future__ import annotations

import argparse
import sys
from typing import Tuple

import torch
from torch import Tensor
from torch.nn.functional import scaled_dot_product_attention

try:
    from sageattn3 import sageattn3_blackwell
except ImportError as exc:  # pragma: no cover - tutorial script
    raise SystemExit(
        "Could not import sageattn3. Did you run setup.py install under "
        "third_party/SageAttention/sageattention3_blackwell?"
    ) from exc


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--heads", type=int, default=8, help="Attention heads")
    parser.add_argument("--q-len", type=int, default=1024, help="Query sequence length")
    parser.add_argument(
        "--kv-len",
        type=int,
        default=None,
        help="Key/value sequence length (defaults to q-len)",
    )
    parser.add_argument(
        "--head-dim",
        type=int,
        default=128,
        help="Head dimension (must be <256 for SageAttention3)",
    )
    parser.add_argument(
        "--dtype",
        choices=("fp16", "bf16"),
        default="fp16",
        help="Input dtype",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device to run on (SageAttention3 expects a CUDA Blackwell GPU)",
    )
    parser.add_argument(
        "--causal",
        action="store_true",
        help="Enable causal masking",
    )
    parser.add_argument(
        "--no-per-block-mean",
        action="store_true",
        help="Disable additional per-block mean subtraction for Q",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Torch RNG seed for reproducibility",
    )
    return parser.parse_args()


def _build_inputs(
    *,
    batch: int,
    heads: int,
    q_len: int,
    kv_len: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    if head_dim >= 256:
        raise ValueError("sageattn3_blackwell only supports head_dim < 256")
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    shape_q = (batch, heads, q_len, head_dim)
    shape_kv = (batch, heads, kv_len, head_dim)
    q = torch.randn(shape_q, dtype=dtype, device=device, generator=generator)
    k = torch.randn(shape_kv, dtype=dtype, device=device, generator=generator)
    v = torch.randn(shape_kv, dtype=dtype, device=device, generator=generator)
    return q, k, v


def main() -> None:
    args = _parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available but --device cuda was requested")

    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    kv_len = args.kv_len if args.kv_len is not None else args.q_len

    q, k, v = _build_inputs(
        batch=args.batch,
        heads=args.heads,
        q_len=args.q_len,
        kv_len=kv_len,
        head_dim=args.head_dim,
        dtype=dtype,
        device=device,
        seed=args.seed,
    )

    print(
        f"Running SageAttention3 demo with q.shape={tuple(q.shape)}, dtype={dtype}, device={device}"
    )

    # Clone tensors because sageattn3 modifies them in-place during preprocessing.
    fast_out = sageattn3_blackwell(
        q.clone(),
        k.clone(),
        v.clone(),
        is_causal=args.causal,
        per_block_mean=not args.no_per_block_mean,
    )
    
    print("sageattn3_blackwell output:", fast_out.shape, fast_out.dtype)

    ref_out = scaled_dot_product_attention(q, k, v, is_causal=args.causal)
    print("PyTorch SDPA output:", ref_out.shape, ref_out.dtype)

    diff = (fast_out - ref_out).float()
    max_abs = diff.abs().max().item()
    mean_abs = diff.abs().mean().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        fast_out.reshape(1, -1).float(), ref_out.reshape(1, -1).float(), dim=-1
    ).item()

    print(f"Max abs diff: {max_abs:.3e}\nMean abs diff: {mean_abs:.3e}\nCosine sim: {cos_sim:.6f}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:  # pragma: no cover - convenience for demos
        sys.exit(130)
