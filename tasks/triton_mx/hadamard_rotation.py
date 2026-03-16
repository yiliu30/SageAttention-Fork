#!/usr/bin/env python3
"""
Hadamard Rotation Experiment: Spreading Outliers Before FP4 Quantization

Hypothesis: Applying an orthogonal Hadamard rotation to Q and K before
quantization spreads outliers uniformly across dimensions, reducing the
per-block quantization max and improving FP4 attention accuracy.

Key insight: QK^T is invariant under orthogonal transforms:
    (QH)(KH)^T = QHH^TK^T = QIK^T = QK^T
So the attention output is mathematically identical. However, quantization
error changes because outliers are redistributed. QuaRot showed 1-2 dB
improvement for INT4 LLM quantization using this technique.

Two parts:
  A) Synthetic single-call accuracy on CogVideoX-realistic shapes,
     with rotation diagnostics (max, std, kurtosis before/after).
  B) End-to-end latent divergence using CogVideoX-2b pipeline.

Usage:
  # Part A only (fast, no model loading)
  python hadamard_rotation.py --parts A

  # Part B (needs CogVideoX model + GPU)
  python hadamard_rotation.py --parts B --num-steps 5 --num-frames 1

  # Full run with JSON output
  python hadamard_rotation.py --parts A B --output-json hadamard_results.json
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import gc
import json
import math
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
# Hadamard Matrix Construction
# ============================================================================

def hadamard_matrix(n: int) -> torch.Tensor:
    """Construct normalized Hadamard matrix of size n (must be power of 2).

    Uses recursive Sylvester construction:
    H_1 = [1]
    H_{2n} = [[H_n, H_n], [H_n, -H_n]] / sqrt(2)

    The result satisfies H @ H.T = I (orthogonal).
    """
    if n == 1:
        return torch.ones(1, 1)
    assert n > 0 and (n & (n - 1)) == 0, f"n must be power of 2, got {n}"

    H_half = hadamard_matrix(n // 2)
    H = torch.cat([
        torch.cat([H_half, H_half], dim=1),
        torch.cat([H_half, -H_half], dim=1),
    ], dim=0) / math.sqrt(2)
    return H


# Pre-compute the Hadamard matrix for D=64 (deterministic, same every time)
_H64 = hadamard_matrix(64)

# Verify orthogonality: H @ H.T ≈ I
_identity_check = _H64 @ _H64.T
_identity_error = (_identity_check - torch.eye(64)).abs().max().item()
assert _identity_error < 1e-5, (
    f"Hadamard matrix is not orthogonal! max|HH^T - I| = {_identity_error}"
)


# ============================================================================
# Hadamard-Rotated Attention Builder
# ============================================================================

def build_hadamard_sage_fn(fmt: str, H: torch.Tensor) -> Callable:
    """Build attention fn with Hadamard rotation on Q and K.

    Only Q and K are rotated. V is NOT rotated because
    softmax(QK^T) @ V would not be invariant — the output would need
    inverse rotation, which changes the error profile.
    """
    from sageattention3_standalone import scaled_dot_product_attention as sage_sdpa

    def hadamard_attn(query, key, value, **kwargs):
        # Rotate Q and K: QK^T is invariant under orthogonal transform
        H_dev = H.to(device=query.device, dtype=query.dtype)
        q_rot = query @ H_dev    # [B,H,N,D] @ [D,D]
        k_rot = key @ H_dev
        return sage_sdpa(q_rot, k_rot, value, quant_format=fmt, **kwargs)

    return hadamard_attn


# ============================================================================
# Rotation Diagnostics
# ============================================================================

def _compute_kurtosis(x: torch.Tensor) -> float:
    """Compute kurtosis: E[(x-mu)^4] / E[(x-mu)^2]^2.

    For a Gaussian distribution, kurtosis = 3.0.
    Higher values indicate heavier tails (more outliers).
    """
    x_flat = x.flatten().float()
    mu = x_flat.mean()
    centered = x_flat - mu
    var = (centered * centered).mean()
    if var < 1e-12:
        return 0.0
    kurt = (centered ** 4).mean() / (var ** 2)
    return kurt.item()


def _rotation_diagnostics(
    q: torch.Tensor,
    k: torch.Tensor,
    H: torch.Tensor,
) -> Dict:
    """Compute distribution diagnostics before and after Hadamard rotation."""
    H_dev = H.to(device=q.device, dtype=q.dtype)
    q_rot = q @ H_dev
    k_rot = k @ H_dev

    return {
        "q_max_before": q.abs().max().item(),
        "q_max_after": q_rot.abs().max().item(),
        "q_std_before": q.float().std().item(),
        "q_std_after": q_rot.float().std().item(),
        "q_kurtosis_before": _compute_kurtosis(q),
        "q_kurtosis_after": _compute_kurtosis(q_rot),
        "k_max_before": k.abs().max().item(),
        "k_max_after": k_rot.abs().max().item(),
        "k_std_before": k.float().std().item(),
        "k_std_after": k_rot.float().std().item(),
        "k_kurtosis_before": _compute_kurtosis(k),
        "k_kurtosis_after": _compute_kurtosis(k_rot),
    }


# ============================================================================
# Part A: Synthetic Single-Call Accuracy
# ============================================================================

def run_part_a(formats: List[str], seed: int) -> Dict:
    """Part A: Synthetic accuracy comparison — baseline vs Hadamard-rotated.

    For each CogVideoX shape and format, computes:
    - Baseline: sage_fn(q, k, v) vs SDPA reference
    - Hadamard: hadamard_sage_fn(q, k, v) vs SDPA reference
    Also prints rotation diagnostics (max, std, kurtosis).
    """
    print_header("Part A: Synthetic Accuracy — Baseline vs Hadamard Rotation")
    print(f"  Formats: {formats}")
    print(f"  Seed: {seed}")
    print(f"  Shapes: {list(COGVIDEOX_SHAPES.keys())}")
    print(f"  Hadamard matrix: {_H64.shape}, orthogonality error: {_identity_error:.2e}")

    # Pre-build sage functions
    sage_fns = {fmt: build_sage_fn(fmt) for fmt in formats}
    hadamard_fns = {fmt: build_hadamard_sage_fn(fmt, _H64) for fmt in formats}

    synthetic_results = {}
    diagnostic_results = {}

    for shape_name, (B, H, N, D) in COGVIDEOX_SHAPES.items():
        print(f"\n  --- Shape: {shape_name}  (B={B}, H={H}, N={N}, D={D}) ---")

        try:
            # Generate random FP16 inputs (seeded for reproducibility)
            gen = torch.Generator(device="cuda").manual_seed(seed)
            q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16, generator=gen)
            k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16, generator=gen)
            v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16, generator=gen)

            # SDPA reference
            with torch.no_grad():
                ref = F.scaled_dot_product_attention(q, k, v)

            # Rotation diagnostics
            with torch.no_grad():
                diag = _rotation_diagnostics(q, k, _H64)
            diagnostic_results[shape_name] = diag

            print(f"\n  Rotation diagnostics:")
            print(f"    Q: max {diag['q_max_before']:.4f} -> {diag['q_max_after']:.4f}  "
                  f"std {diag['q_std_before']:.4f} -> {diag['q_std_after']:.4f}  "
                  f"kurtosis {diag['q_kurtosis_before']:.2f} -> {diag['q_kurtosis_after']:.2f}")
            print(f"    K: max {diag['k_max_before']:.4f} -> {diag['k_max_after']:.4f}  "
                  f"std {diag['k_std_before']:.4f} -> {diag['k_std_after']:.4f}  "
                  f"kurtosis {diag['k_kurtosis_before']:.2f} -> {diag['k_kurtosis_after']:.2f}")

            # Test each format: baseline and Hadamard-rotated
            shape_results = {}
            rows = []

            for fmt in formats:
                # Baseline
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

                # Hadamard-rotated
                fmt_had = f"{fmt}_had"
                try:
                    with torch.no_grad():
                        out_had = hadamard_fns[fmt](q, k, v)
                    m_had = compute_metrics(ref, out_had)
                    shape_results[fmt_had] = asdict(m_had)
                    rows.append((fmt_had, m_had))
                    del out_had
                except torch.cuda.OutOfMemoryError:
                    print(f"    {fmt_had}: OOM — skipped")
                    shape_results[fmt_had] = "OOM"
                except Exception as e:
                    print(f"    {fmt_had}: ERROR — {e}")
                    shape_results[fmt_had] = f"ERROR: {e}"

            if rows:
                print_metrics_table(rows)

            synthetic_results[shape_name] = shape_results

            # Free GPU memory
            del q, k, v, ref
            torch.cuda.empty_cache()
            gc.collect()

        except torch.cuda.OutOfMemoryError:
            print("    Entire shape OOM — skipped")
            synthetic_results[shape_name] = "OOM"
            torch.cuda.empty_cache()
            gc.collect()

    # --- Summary: delta table (Hadamard improvement per format) ---
    print_header("Part A Summary — Hadamard Improvement (delta SNR dB)")
    print(f"\n  {'Shape':<16s}", end="")
    for fmt in formats:
        print(f"  {fmt:>12s}", end="")
    print()
    print(f"  {'-'*16}", end="")
    for _ in formats:
        print(f"  {'-'*12}", end="")
    print()

    for shape_name in COGVIDEOX_SHAPES:
        sr = synthetic_results.get(shape_name)
        if sr is None or isinstance(sr, str):
            continue
        print(f"  {shape_name:<16s}", end="")
        for fmt in formats:
            base = sr.get(fmt)
            had = sr.get(f"{fmt}_had")
            if (isinstance(base, dict) and isinstance(had, dict)
                    and "snr_db" in base and "snr_db" in had):
                delta = had["snr_db"] - base["snr_db"]
                print(f"  {delta:>+12.2f}", end="")
            else:
                print(f"  {'N/A':>12s}", end="")
        print()
    print()

    return {
        "synthetic": synthetic_results,
        "rotation_diagnostics": diagnostic_results,
    }


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
    """Part B: End-to-end latent divergence — baseline vs Hadamard-rotated.

    Runs CogVideoX pipeline with SDPA reference, then for each format runs
    both baseline and Hadamard-rotated versions. Compares final latents.
    """
    print_header("Part B: E2E Latent Divergence — Baseline vs Hadamard")
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

    # --- Reference run with SDPA ---
    print("\n  Running SDPA reference...")
    orig_sdpa = F.scaled_dot_product_attention

    ref_capture = LatentCapture()
    ref_capture.attach(pipe, num_steps)
    F.scaled_dot_product_attention = orig_sdpa
    _run_pipe(pipe, prompt, num_steps, num_frames, seed)
    ref_capture.detach(pipe)

    ref_latent = ref_capture.final_latent
    if ref_latent is None:
        print("  WARNING: Could not capture reference latent!")
        return {"e2e_latent": {"error": "Failed to capture reference latent"}}
    print(f"  Reference latent shape: {ref_latent.shape}")

    results = {}
    rows = []

    # --- Per-format runs: baseline and Hadamard ---
    for fmt in formats:
        sage_fn = build_sage_fn(fmt)
        had_fn = build_hadamard_sage_fn(fmt, _H64)

        # --- Baseline run ---
        print(f"\n  Running baseline: {fmt}...")
        base_capture = LatentCapture()
        base_capture.attach(pipe, num_steps)
        F.scaled_dot_product_attention = sage_fn
        _run_pipe(pipe, prompt, num_steps, num_frames, seed)
        base_capture.detach(pipe)
        F.scaled_dot_product_attention = orig_sdpa

        base_latent = base_capture.final_latent
        if base_latent is None:
            print(f"    WARNING: Could not capture {fmt} baseline latent!")
            results[fmt] = {"error": "Failed to capture latent"}
        else:
            m_base = compute_metrics(ref_latent, base_latent)
            results[fmt] = asdict(m_base)
            rows.append((fmt, m_base))
            print(f"    SNR={m_base.snr_db:.2f}dB  CosSim={m_base.cos_sim:.8f}  "
                  f"L2Rel={m_base.l2_rel:.6f}")
            del base_latent

        gc.collect()
        torch.cuda.empty_cache()

        # --- Hadamard run ---
        fmt_had = f"{fmt}_had"
        print(f"  Running Hadamard: {fmt_had}...")
        had_capture = LatentCapture()
        had_capture.attach(pipe, num_steps)
        F.scaled_dot_product_attention = had_fn
        _run_pipe(pipe, prompt, num_steps, num_frames, seed)
        had_capture.detach(pipe)
        F.scaled_dot_product_attention = orig_sdpa

        had_latent = had_capture.final_latent
        if had_latent is None:
            print(f"    WARNING: Could not capture {fmt_had} latent!")
            results[fmt_had] = {"error": "Failed to capture latent"}
        else:
            m_had = compute_metrics(ref_latent, had_latent)
            results[fmt_had] = asdict(m_had)
            rows.append((fmt_had, m_had))
            print(f"    SNR={m_had.snr_db:.2f}dB  CosSim={m_had.cos_sim:.8f}  "
                  f"L2Rel={m_had.l2_rel:.6f}")
            del had_latent

        gc.collect()
        torch.cuda.empty_cache()

    # --- Summary table ---
    if rows:
        print("\n  Summary — E2E Latent Divergence:")
        print_metrics_table(rows)

    # --- Delta summary ---
    print_header("Part B Summary — Hadamard Improvement (delta SNR dB)")
    print(f"\n  {'Format':<16s} {'Baseline SNR':>14s} {'Hadamard SNR':>14s} {'Delta':>10s}")
    print(f"  {'-'*16} {'-'*14} {'-'*14} {'-'*10}")
    for fmt in formats:
        base = results.get(fmt)
        had = results.get(f"{fmt}_had")
        if (isinstance(base, dict) and isinstance(had, dict)
                and "snr_db" in base and "snr_db" in had):
            delta = had["snr_db"] - base["snr_db"]
            print(f"  {fmt:<16s} {base['snr_db']:>14.2f} {had['snr_db']:>14.2f} "
                  f"{delta:>+10.2f}")
        else:
            print(f"  {fmt:<16s} {'N/A':>14s} {'N/A':>14s} {'N/A':>10s}")
    print()

    # Restore SDPA and clean up
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
        description="Hadamard Rotation Experiment: Spreading Outliers Before FP4 Quantization",
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
                        help="Denoising steps (5 for quick, 50 for full)")
    parser.add_argument("--num-frames", type=int, default=1,
                        help="Frames to generate (1 for quick, 49 for full)")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Save results to JSON file")

    args = parser.parse_args()

    print_header("Hadamard Rotation Experiment")
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

    print_header("Done")
    print(f"  Total time: {elapsed:.1f}s")

    if args.output_json:
        all_results["_metadata"] = {
            "experiment": "hadamard_rotation",
            "parts": args.parts,
            "formats": args.formats,
            "model": args.model,
            "seed": args.seed,
            "num_steps": args.num_steps,
            "num_frames": args.num_frames,
            "hadamard_dim": 64,
            "hadamard_orthogonality_error": _identity_error,
            "elapsed_s": elapsed,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "gpu": (torch.cuda.get_device_name(0)
                    if torch.cuda.is_available() else "N/A"),
        }
        with open(args.output_json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  Results saved to: {args.output_json}")


if __name__ == "__main__":
    main()
