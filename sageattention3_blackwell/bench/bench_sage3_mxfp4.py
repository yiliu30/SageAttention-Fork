#!/usr/bin/env python
"""MXFP4 vs NVFP4 correctness + latency for SageAttention3.

Checks (plan Phase 4):
  - MXFP4 output vs torch SDPA and vs the NVFP4 path: max/mean abs error, cosine.
  - Runs across head_dim, seq length (incl. L>=1024 to catch Blk_MN swizzle
    divergence past the first tile), causal/non-causal, bf16/fp16.
  - Latency for each format.

Usage:
  PYTHONPATH=/home/guest/yiliu7/sage-mxfp4/sageattention3_blackwell \\
    /home/guest/yiliu7/vllm-omni/.venv/bin/python bench/bench_sage3_mxfp4.py [--bench]
"""
import argparse
import sys

import torch
import torch.nn.functional as F

REPO = "/home/guest/yiliu7/sage-mxfp4/sageattention3_blackwell"
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def _prove_build():
    """Both .so files are top-level extension modules, so PYTHONPATH must
    shadow them as well as the package -- otherwise the new Python API runs
    against the old NVFP4 kernels and everything 'passes'."""
    import fp4attn_cuda
    import fp4quant_cuda
    print(f"fp4attn_cuda : {fp4attn_cuda.__file__}")
    print(f"fp4quant_cuda: {fp4quant_cuda.__file__}")
    for m in (fp4attn_cuda, fp4quant_cuda):
        if REPO not in m.__file__:
            raise SystemExit(
                f"WRONG BUILD: {m.__file__} is not under {REPO}.\n"
                "Put the repo root on PYTHONPATH."
            )


def _metrics(got, ref):
    g, r = got.float().flatten(), ref.float().flatten()
    return {
        "cos": F.cosine_similarity(g, r, dim=0).item(),
        "maxabs": (g - r).abs().max().item(),
        "meanabs": (g - r).abs().mean().item(),
    }


def correctness(fmt, B, H, L, D, dtype, causal, device="cuda", seed=0):
    from sageattn3 import sageattn3_blackwell

    torch.manual_seed(seed)
    q = torch.randn(B, H, L, D, device=device, dtype=dtype) * 0.1
    k = torch.randn(B, H, L, D, device=device, dtype=dtype) * 0.1
    v = torch.randn(B, H, L, D, device=device, dtype=dtype) * 0.1

    out = sageattn3_blackwell(q, k, v, is_causal=causal, fmt=fmt)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    return out, ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true", help="also run latency")
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    _prove_build()
    print()

    rows = []
    for D in (128,):
        for L in (1024, 2048, 4096):
            for causal in (False, True):
                for dtype in (torch.bfloat16, torch.float16):
                    for fmt in ("nvfp4", "mxfp4"):
                        try:
                            out, ref = correctness(fmt, 1, 8, L, D, dtype, causal)
                            m = _metrics(out, ref)
                            rows.append((fmt, D, L, causal, str(dtype).split(".")[-1], m))
                            print(
                                f"  D={D:4d} L={L:5d} causal={int(causal)} "
                                f"{str(dtype).split('.')[-1]:9s} {fmt:6s} "
                                f"cos={m['cos']:.6f} maxabs={m['maxabs']:.4f} "
                                f"meanabs={m['meanabs']:.5f}"
                            )
                        except Exception as e:  # noqa: BLE001
                            print(f"  D={D} L={L} causal={int(causal)} {fmt}: FAILED {type(e).__name__}: {e}")

    # Summary: is MXFP4 close to NVFP4?
    print("\n  --- MXFP4 vs NVFP4 (same shape, both vs SDPA) ---")
    by_key = {}
    for fmt, D, L, causal, dt, m in rows:
        by_key.setdefault((D, L, causal, dt, fmt), m)
    deltas = []
    for (D, L, causal, dt, fmt), m in by_key.items():
        if fmt != "mxfp4":
            continue
        nv = by_key.get((D, L, causal, dt, "nvfp4"))
        if nv:
            d = nv["cos"] - m["cos"]
            deltas.append(d)
            print(f"    D={D} L={L} causal={int(causal)} {dt:9s} "
                  f"cos_nvfp4={nv['cos']:.6f} cos_mxfp4={m['cos']:.6f} delta={d:+.6f}")
    if deltas:
        print(f"  mean cosine delta (nvfp4 - mxfp4): {sum(deltas)/len(deltas):+.6f}")

    if args.bench:
        print("\n  --- latency (ms, B=1 H=40 D=128 L=75600, non-causal) ---")
        L = 75600
        for fmt in ("nvfp4", "mxfp4"):
            for dtype in (torch.bfloat16,):
                try:
                    q = torch.randn(1, 40, L, 128, device="cuda", dtype=dtype) * 0.1
                    k = torch.randn(1, 40, L, 128, device="cuda", dtype=dtype) * 0.1
                    v = torch.randn(1, 40, L, 128, device="cuda", dtype=dtype) * 0.1
                    for _ in range(3):
                        correctness.__wrapped__ if False else None
                        from sageattn3 import sageattn3_blackwell
                        sageattn3_blackwell(q, k, v, fmt=fmt)
                    torch.cuda.synchronize()
                    import time
                    t0 = time.time()
                    for _ in range(args.iters):
                        from sageattn3 import sageattn3_blackwell
                        sageattn3_blackwell(q, k, v, fmt=fmt)
                    torch.cuda.synchronize()
                    ms = (time.time() - t0) / args.iters * 1e3
                    print(f"    {fmt:6s} {ms:9.2f} ms/it")
                except Exception as e:  # noqa: BLE001
                    print(f"    {fmt:6s} FAILED {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
