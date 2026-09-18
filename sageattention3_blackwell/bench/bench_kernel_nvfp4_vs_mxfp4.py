#!/usr/bin/env python
"""Apples-to-apples NVFP4 vs MXFP4 attention-kernel benchmark.

Target shape (Wan2.2 720p production): B=1, H=40, L=75600, D=128, non-causal.

Both formats run the exact same Python entry point (sageattn3_blackwell) so the
measured time covers the whole pipeline -- Q/K/V pre-process, quantisation, and
the attention kernel.  That is the honest end-to-end comparison; a kernel-only
number would need separate instrumentation and would understate the MXFP4 cost
(its quantiser is a different kernel).

Fairness controls:
  - same GPU, same process, same tensors (re-created per format to equalise
    allocator state)
  - CUDA events, not wall clock
  - warmup iterations excluded
  - the loaded build is asserted to support fmt= (a path check is NOT enough:
    the old site-packages api.py silently swallows fmt= via **kwargs)

Usage:
  PYTHONPATH=<repo-root> CUDA_VISIBLE_DEVICES=4 \
    python bench/bench_kernel_nvfp4_vs_mxfp4.py [--iters 30] [--warmup 5]
"""
import argparse
import inspect
import os
import json
import sys
import time

import torch
import torch.nn.functional as F

# Analyse the build in THIS checkout: the repo root holds sageattn3/ plus the
# built fp4attn_cuda / fp4quant_cuda extensions, so putting it on sys.path
# shadows any installed copy.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


def assert_build():
    """Verify CAPABILITY, not just path. The old site-packages sageattn3 has
    sageattn3_blackwell(..., **kwargs), which swallows fmt= and silently runs
    NVFP4 -- it would look like a valid MXFP4 result."""
    import fp4attn_cuda
    import fp4quant_cuda
    import sageattn3
    import sageattn3.api as api

    print(f"  fp4attn_cuda : {fp4attn_cuda.__file__}")
    print(f"  fp4quant_cuda: {fp4quant_cuda.__file__}")
    print(f"  sageattn3    : {api.__file__}")
    missing = [
        n for n, f in (("sageattn3_blackwell", api.sageattn3_blackwell),
                       ("scale_and_quant_fp4", api.scale_and_quant_fp4))
        if "fmt" not in inspect.signature(f).parameters
    ]
    if missing:
        raise SystemExit(f"FATAL: {missing} lack fmt= -> would silently run NVFP4")
    print("  capability guard: fmt= present on both entry points OK")
    return sageattn3.sageattn3_blackwell


def bench(fn, q, k, v, fmt, warmup, iters):
    for _ in range(warmup):
        fn(q, k, v, is_causal=False, fmt=fmt)
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        out = fn(q, k, v, is_causal=False, fmt=fmt)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))

    t = torch.tensor(times)
    return {
        "mean_ms": t.mean().item(),
        "median_ms": t.median().item(),
        "min_ms": t.min().item(),
        "max_ms": t.max().item(),
        "std_ms": t.std().item(),
        "samples": len(times),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--heads", type=int, default=40)
    ap.add_argument("--seqlen", type=int, default=75600)
    ap.add_argument("--headdim", type=int, default=128)
    ap.add_argument("--causal", action="store_true")
    ap.add_argument("--out", default=None, help="write JSON results here")
    args = ap.parse_args()

    kernel = assert_build()
    B, H, L, D = args.batch, args.heads, args.seqlen, args.headdim
    Lp = ((L + 127) // 128) * 128  # the kernel pads to a multiple of 128

    # FLOPs for the two QK^T and PV matmuls, 2*M*N*K each.
    flops = 2.0 * 2.0 * B * H * L * Lp * D
    print(f"\nshape: B={B} H={H} L={L} (padded {Lp}) D={D} causal={args.causal}")
    print(f"  = {flops/1e12:.2f} TFLOP per call (2 matmuls x 2*B*H*L*Lp*D)")

    results = {}
    for fmt in ("nvfp4", "mxfp4"):
        torch.manual_seed(0)
        q = torch.randn(B, H, L, D, device="cuda", dtype=torch.bfloat16) * 0.1
        k = torch.randn(B, H, L, D, device="cuda", dtype=torch.bfloat16) * 0.1
        v = torch.randn(B, H, L, D, device="cuda", dtype=torch.bfloat16) * 0.1
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        try:
            r = bench(kernel, q, k, v, fmt, args.warmup, args.iters)
        except Exception as exc:  # noqa: BLE001
            print(f"  {fmt:6s} FAILED: {type(exc).__name__}: {exc}")
            results[fmt] = {"error": f"{type(exc).__name__}: {exc}"}
            del q, k, v
            torch.cuda.empty_cache()
            continue

        peak = torch.cuda.max_memory_allocated() / 2**30
        r.update(fmt=fmt, tflops=flops / (r["mean_ms"] / 1e3) / 1e12, peak_gib=peak)
        results[fmt] = r
        print(f"  {fmt:6s} mean={r['mean_ms']:8.2f} ms  median={r['median_ms']:8.2f}  "
              f"min={r['min_ms']:8.2f}  max={r['max_ms']:8.2f}  "
              f"{r['tflops']:7.1f} TFLOP/s  peak={peak:5.1f} GiB")

        del q, k, v
        torch.cuda.empty_cache()

    if "nvfp4" in results and "mxfp4" in results \
            and "mean_ms" in results["nvfp4"] and "mean_ms" in results["mxfp4"]:
        n, m = results["nvfp4"], results["mxfp4"]
        d = m["mean_ms"] - n["mean_ms"]
        print(f"\n  MXFP4 vs NVFP4: {d:+.2f} ms  ({100*d/n['mean_ms']:+.2f}%)")
        print(f"    TFLOP/s: {n['tflops']:.1f} -> {m['tflops']:.1f} "
              f"({100*(m['tflops']/n['tflops']-1):+.2f}%)")
        print(f"    peak GiB: {n['peak_gib']:.1f} -> {m['peak_gib']:.1f} "
              f"({100*(m['peak_gib']/n['peak_gib']-1):+.2f}%)")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"shape": {"B": B, "H": H, "L": L, "L_padded": Lp, "D": D,
                                 "causal": args.causal},
                       "tflop_per_call": flops / 1e12,
                       "results": results}, fh, indent=2)
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
