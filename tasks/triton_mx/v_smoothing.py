#!/usr/bin/env python3
"""
V-Smoothing Experiment: Mean-Subtracting V Before FP4 Quantization

Hypothesis: Quantization error on V can be reduced by centering V (subtracting
its mean along the sequence dimension) before passing it to the quantized
attention kernel.  Because softmax weights sum to 1, the mean can be added back
to the output exactly:

    v_mean      = V.mean(dim=-2, keepdim=True)   # [B, H, 1, D]
    v_centered  = V - v_mean
    output      = quantized_attention(Q, K, v_centered)
    output     += v_mean                          # exact correction

This is mathematically equivalent to standard attention but presents a
lower-dynamic-range V to the quantizer, which should improve accuracy at
the cost of one extra mean + add per call.

Two parts:
  A) Synthetic single-call accuracy on CogVideoX-realistic shapes
  B) End-to-end latent divergence using CogVideoX-2b

Usage:
  # Part A only (fast, no model loading)
  python v_smoothing.py --parts A

  # Part B (needs CogVideoX model + GPU)
  python v_smoothing.py --parts B --num-steps 5 --num-frames 1

  # Full run with JSON output
  python v_smoothing.py --parts A B --output-json v_smoothing_results.json
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import functools
import gc
import json
import sys
import time
from dataclasses import asdict
from typing import Callable, Dict, List, Tuple

import torch
import torch.nn.functional as F

# Add standalone to path for sageattention3_standalone imports
_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_standalone_path = os.path.join(_repo_root, "standalone")
if _standalone_path not in sys.path:
    sys.path.insert(0, _standalone_path)

from ablation import (
    AccuracyMetrics,
    compute_metrics,
    build_sage_fn,
    print_header,
    print_metrics_table,
    _load_cogvideox_pipe,
    _run_pipe,
    LatentCapture,
    COGVIDEOX_SHAPES,
    ALL_FORMATS,
)


# ============================================================================
# V-Smoothed Attention Wrapper
# ============================================================================

def build_v_smoothed_sage_fn(fmt: str) -> Callable:
    """Create an attention function with V smoothing for a specific quant format.

    V smoothing subtracts the sequence-dimension mean of V before quantized
    attention, then adds it back to the output.  Because softmax weights sum
    to 1, this is mathematically exact and presents a lower-dynamic-range V
    to the quantizer.

    Parameters
    ----------
    fmt : str
        Quantization format (e.g. "nvfp4", "mxfp4", "mxfp4_s1", "mxfp8_s1").

    Returns
    -------
    Callable
        A function with the same signature as F.scaled_dot_product_attention.
    """
    from sageattention3_standalone import scaled_dot_product_attention as sage_sdpa

    def _v_smoothed_attention(query, key, value, *args, **kwargs):
        # 1. Compute per-head mean of V along the sequence dimension
        v_mean = value.mean(dim=-2, keepdim=True)  # [B, H, 1, D]

        # 2. Center V
        value_centered = value - v_mean

        # 3. Run quantized attention on the centered V
        output = sage_sdpa(query, key, value_centered, *args, quant_format=fmt, **kwargs)

        # 4. Add back the mean (exact because sum(softmax_weights) == 1)
        output = output + v_mean

        return output

    return _v_smoothed_attention


# ============================================================================
# Part A: Synthetic Single-Call Accuracy
# ============================================================================

def run_part_a(formats: List[str], seed: int) -> Dict:
    """Part A: Synthetic accuracy comparison — baseline vs V-smoothed.

    For each CogVideoX-realistic shape and each quantization format, compares:
      - baseline: raw quantized attention
      - V-smoothed: mean-subtracted V before quantization

    Both are measured against SDPA reference.
    """
    print_header("Part A: Synthetic Accuracy — Baseline vs V-Smoothed")
    print(f"  Formats: {formats}")
    print(f"  Seed: {seed}")
    print(f"  Shapes: {list(COGVIDEOX_SHAPES.keys())}")

    # Pre-build both baseline and V-smoothed functions
    sage_fns = {fmt: build_sage_fn(fmt) for fmt in formats}
    vs_fns = {fmt: build_v_smoothed_sage_fn(fmt) for fmt in formats}

    results = {}

    for shape_name, (B, H, N, D) in COGVIDEOX_SHAPES.items():
        print(f"\n  --- Shape: {shape_name}  (B={B}, H={H}, N={N}, D={D}) ---")

        try:
            # Generate random FP16 inputs (seeded for reproducibility)
            gen = torch.Generator(device="cuda").manual_seed(seed)
            q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16, generator=gen)
            k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16, generator=gen)
            v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16, generator=gen)

            # Reference: PyTorch SDPA
            with torch.no_grad():
                ref = F.scaled_dot_product_attention(q, k, v)

            # Test each format: baseline + V-smoothed
            shape_results = {}
            rows = []

            for fmt in formats:
                # --- Baseline ---
                try:
                    with torch.no_grad():
                        out_base = sage_fns[fmt](q, k, v)
                    m_base = compute_metrics(ref, out_base)
                    shape_results[fmt] = asdict(m_base)
                    rows.append((fmt, m_base))
                    del out_base
                except torch.cuda.OutOfMemoryError:
                    print(f"    {fmt}: OOM — skipped")
                    shape_results[fmt] = "OOM"
                except Exception as e:
                    print(f"    {fmt}: ERROR — {e}")
                    shape_results[fmt] = f"ERROR: {e}"

                # --- V-smoothed ---
                vs_key = f"{fmt}_vs"
                try:
                    with torch.no_grad():
                        out_vs = vs_fns[fmt](q, k, v)
                    m_vs = compute_metrics(ref, out_vs)
                    shape_results[vs_key] = asdict(m_vs)
                    rows.append((vs_key, m_vs))
                    del out_vs
                except torch.cuda.OutOfMemoryError:
                    print(f"    {vs_key}: OOM — skipped")
                    shape_results[vs_key] = "OOM"
                except Exception as e:
                    print(f"    {vs_key}: ERROR — {e}")
                    shape_results[vs_key] = f"ERROR: {e}"

            if rows:
                print_metrics_table(rows)

            results[shape_name] = shape_results

            # Free GPU memory
            del q, k, v, ref
            torch.cuda.empty_cache()
            gc.collect()

        except torch.cuda.OutOfMemoryError:
            print("    Entire shape OOM — skipped")
            results[shape_name] = "OOM"
            torch.cuda.empty_cache()
            gc.collect()

    # --- Summary: V-smoothing delta for each format ---
    print_header("Part A Summary — V-Smoothing Improvement (delta)")
    print(f"\n  {'Shape':<16s} {'Format':<12s} {'Base SNR':>10s} {'VS SNR':>10s} "
          f"{'delta_SNR':>10s} {'Base CosSim':>12s} {'VS CosSim':>12s} {'delta_Cos':>10s}")
    print(f"  {'-'*16} {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*12} {'-'*12} {'-'*10}")

    for shape_name in COGVIDEOX_SHAPES:
        if shape_name not in results or results[shape_name] == "OOM":
            continue
        shape_data = results[shape_name]
        for fmt in formats:
            vs_key = f"{fmt}_vs"
            base_data = shape_data.get(fmt)
            vs_data = shape_data.get(vs_key)
            if (isinstance(base_data, dict) and isinstance(vs_data, dict)
                    and "snr_db" in base_data and "snr_db" in vs_data):
                d_snr = vs_data["snr_db"] - base_data["snr_db"]
                d_cos = vs_data["cos_sim"] - base_data["cos_sim"]
                print(f"  {shape_name:<16s} {fmt:<12s} "
                      f"{base_data['snr_db']:>10.2f} {vs_data['snr_db']:>10.2f} "
                      f"{d_snr:>+10.2f} "
                      f"{base_data['cos_sim']:>12.6f} {vs_data['cos_sim']:>12.6f} "
                      f"{d_cos:>+10.6f}")
    print()

    return {"synthetic": results}


# ============================================================================
# Part B: End-to-End Latent Divergence
# ============================================================================

def run_part_b(
    formats: List[str],
    model: str,
    seed: int,
    num_steps: int,
    num_frames: int,
) -> Dict:
    """Part B: End-to-end latent divergence using CogVideoX.

    Compares final denoising latent for baseline vs V-smoothed attention
    against SDPA reference.
    """
    print_header("Part B: E2E Latent Divergence — Baseline vs V-Smoothed")
    print(f"  Model: {model}, Steps: {num_steps}, Frames: {num_frames}")
    print(f"  Formats: {formats}")
    print(f"  Seed: {seed}")

    prompt = "A dog is running in the park."

    # Load pipeline once
    print("\n  Loading CogVideoX pipeline...")
    pipe, default_frames, _torch_dtype = _load_cogvideox_pipe(model)
    if num_frames is None:
        num_frames = default_frames
    print(f"  Pipeline loaded. Using {num_frames} frames, {num_steps} steps.")

    orig_sdpa = F.scaled_dot_product_attention

    # --- Reference run with SDPA ---
    print("\n  Running SDPA reference...")
    ref_capture = LatentCapture()
    ref_capture.attach(pipe, num_steps)
    F.scaled_dot_product_attention = orig_sdpa
    _run_pipe(pipe, prompt, num_steps, num_frames, seed)
    ref_capture.detach(pipe)

    ref_latent = ref_capture.final_latent
    if ref_latent is None:
        print("  ERROR: Could not capture reference latent!")
        F.scaled_dot_product_attention = orig_sdpa
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
        return {"e2e_latent": {"error": "Failed to capture reference latent"}}
    print(f"  Reference latent shape: {ref_latent.shape}")

    results = {}
    rows = []

    # --- Per-format runs: baseline + V-smoothed ---
    for fmt in formats:
        # --- Baseline ---
        print(f"\n  Running baseline: {fmt}...")
        sage_fn = build_sage_fn(fmt)

        base_capture = LatentCapture()
        base_capture.attach(pipe, num_steps)
        F.scaled_dot_product_attention = sage_fn
        _run_pipe(pipe, prompt, num_steps, num_frames, seed)
        base_capture.detach(pipe)
        F.scaled_dot_product_attention = orig_sdpa

        base_latent = base_capture.final_latent
        if base_latent is not None:
            m_base = compute_metrics(ref_latent, base_latent)
            results[fmt] = asdict(m_base)
            rows.append((fmt, m_base))
            print(f"    SNR={m_base.snr_db:.2f}dB  CosSim={m_base.cos_sim:.8f}")
        else:
            print(f"    WARNING: Could not capture {fmt} baseline latent!")
            results[fmt] = {"error": "Failed to capture latent"}

        del base_latent
        gc.collect()
        torch.cuda.empty_cache()

        # --- V-smoothed ---
        vs_key = f"{fmt}_vs"
        print(f"  Running V-smoothed: {vs_key}...")
        vs_fn = build_v_smoothed_sage_fn(fmt)

        vs_capture = LatentCapture()
        vs_capture.attach(pipe, num_steps)
        F.scaled_dot_product_attention = vs_fn
        _run_pipe(pipe, prompt, num_steps, num_frames, seed)
        vs_capture.detach(pipe)
        F.scaled_dot_product_attention = orig_sdpa

        vs_latent = vs_capture.final_latent
        if vs_latent is not None:
            m_vs = compute_metrics(ref_latent, vs_latent)
            results[vs_key] = asdict(m_vs)
            rows.append((vs_key, m_vs))
            print(f"    SNR={m_vs.snr_db:.2f}dB  CosSim={m_vs.cos_sim:.8f}")
        else:
            print(f"    WARNING: Could not capture {vs_key} latent!")
            results[vs_key] = {"error": "Failed to capture latent"}

        del vs_latent
        gc.collect()
        torch.cuda.empty_cache()

    # --- Summary table ---
    if rows:
        print("\n  Summary — E2E Latent Divergence:")
        print_metrics_table(rows)

    # --- Delta summary ---
    print_header("Part B Summary — V-Smoothing Improvement (delta)")
    print(f"\n  {'Format':<12s} {'Base SNR':>10s} {'VS SNR':>10s} "
          f"{'delta_SNR':>10s} {'Base CosSim':>12s} {'VS CosSim':>12s} {'delta_Cos':>10s}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*12} {'-'*12} {'-'*10}")

    for fmt in formats:
        vs_key = f"{fmt}_vs"
        base_data = results.get(fmt)
        vs_data = results.get(vs_key)
        if (isinstance(base_data, dict) and isinstance(vs_data, dict)
                and "snr_db" in base_data and "snr_db" in vs_data):
            d_snr = vs_data["snr_db"] - base_data["snr_db"]
            d_cos = vs_data["cos_sim"] - base_data["cos_sim"]
            print(f"  {fmt:<12s} "
                  f"{base_data['snr_db']:>10.2f} {vs_data['snr_db']:>10.2f} "
                  f"{d_snr:>+10.2f} "
                  f"{base_data['cos_sim']:>12.6f} {vs_data['cos_sim']:>12.6f} "
                  f"{d_cos:>+10.6f}")
    print()

    # Clean up
    F.scaled_dot_product_attention = orig_sdpa
    del ref_latent, pipe
    gc.collect()
    torch.cuda.empty_cache()

    return {"e2e_latent": results}


# ============================================================================
# CLI and Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="V-Smoothing Experiment: Mean-Subtracting V Before FP4 Quantization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--parts", nargs="+", default=["A"],
        choices=["A", "B"],
        help="Which parts to run (default: A only)",
    )
    parser.add_argument(
        "--formats", nargs="+", default=ALL_FORMATS,
        choices=ALL_FORMATS,
        help=f"Quant formats to test (default: all {len(ALL_FORMATS)})",
    )
    parser.add_argument("--model", default="cogvideox-2b",
                        choices=["cogvideox-2b", "cogvideox1.5-5b"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-steps", type=int, default=5,
                        help="Denoising steps (default: 5)")
    parser.add_argument("--num-frames", type=int, default=1,
                        help="Frames to generate (default: 1)")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Save results to JSON file")

    args = parser.parse_args()

    print_header("V-Smoothing Experiment")
    print(f"  Parts:   {args.parts}")
    print(f"  Formats: {args.formats}")
    print(f"  Model:   {args.model}")
    print(f"  Seed:    {args.seed}")
    if "B" in args.parts:
        print(f"  Steps:   {args.num_steps}")
        print(f"  Frames:  {args.num_frames}")

    all_results = {}
    start_time = time.time()

    if "A" in args.parts:
        result = run_part_a(args.formats, args.seed)
        all_results.update(result)

    if "B" in args.parts:
        result = run_part_b(
            args.formats, args.model, args.seed,
            args.num_steps, args.num_frames,
        )
        all_results.update(result)

    elapsed = time.time() - start_time

    # --- Final summary ---
    print_header("Final Summary")
    print(f"  Total time: {elapsed:.1f}s")

    # Print overall delta summary across all parts
    if "synthetic" in all_results:
        print("\n  Synthetic (Part A) — average delta_SNR by format:")
        synth = all_results["synthetic"]
        for fmt in args.formats:
            vs_key = f"{fmt}_vs"
            deltas = []
            for shape_name in COGVIDEOX_SHAPES:
                if shape_name not in synth or synth[shape_name] == "OOM":
                    continue
                sd = synth[shape_name]
                b = sd.get(fmt)
                v = sd.get(vs_key)
                if (isinstance(b, dict) and isinstance(v, dict)
                        and "snr_db" in b and "snr_db" in v):
                    deltas.append(v["snr_db"] - b["snr_db"])
            if deltas:
                avg_delta = sum(deltas) / len(deltas)
                print(f"    {fmt}: avg delta_SNR = {avg_delta:+.2f} dB "
                      f"(across {len(deltas)} shapes)")

    if "e2e_latent" in all_results:
        print("\n  E2E Latent (Part B) — delta_SNR by format:")
        e2e = all_results["e2e_latent"]
        for fmt in args.formats:
            vs_key = f"{fmt}_vs"
            b = e2e.get(fmt)
            v = e2e.get(vs_key)
            if (isinstance(b, dict) and isinstance(v, dict)
                    and "snr_db" in b and "snr_db" in v):
                d_snr = v["snr_db"] - b["snr_db"]
                print(f"    {fmt}: delta_SNR = {d_snr:+.2f} dB")

    if args.output_json:
        all_results["_metadata"] = {
            "formats": args.formats,
            "model": args.model,
            "seed": args.seed,
            "num_steps": args.num_steps,
            "num_frames": args.num_frames,
            "elapsed_s": elapsed,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "gpu": (torch.cuda.get_device_name(0)
                    if torch.cuda.is_available() else "N/A"),
        }
        with open(args.output_json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  Results saved to: {args.output_json}")


if __name__ == "__main__":
    main()
