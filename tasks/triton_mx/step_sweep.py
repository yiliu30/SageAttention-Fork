#!/usr/bin/env python3
"""
Step-Count Proxy Validation — Is 5 Steps Enough?

Validates that the format ranking and relative divergence magnitudes observed
at 5 denoising steps are representative of the converged behavior at higher
step counts (10, 20, 50).

For each step count, runs SDPA reference + all specified formats, captures
the final-step latent, and computes accuracy metrics (SNR, CosSim, L1_rel).
Prints a cross-step comparison table and ranking stability check.

Usage:
  # Full sweep (~42 min)
  python step_sweep.py --output-json step_sweep_results.json

  # Quick test with 2 step counts and 2 formats (~5 min)
  python step_sweep.py --steps 5 10 --formats nvfp4 mxfp8_s1
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import gc
import json
import sys
import time
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# Add parent so we can import from ablation.py
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from ablation import (
    AccuracyMetrics,
    LatentCapture,
    build_sage_fn,
    compute_metrics,
    print_header,
    _load_cogvideox_pipe,
    _run_pipe,
)

ALL_FORMATS = ["nvfp4", "mxfp4", "mxfp4_s1", "mxfp8_s1"]
DEFAULT_STEPS = [5, 10, 20, 50]


def run_step_sweep(
    formats: List[str],
    step_counts: List[int],
    model: str,
    seed: int,
    num_frames: int,
) -> Dict[int, Dict[str, AccuracyMetrics]]:
    """Run latent divergence at multiple step counts to validate proxy.

    Loads the pipeline once and reuses it for all step counts and formats.

    Returns:
        results[step_count][format] = AccuracyMetrics
    """
    print_header("Step-Count Proxy Validation")
    print(f"  Model:   {model}")
    print(f"  Formats: {formats}")
    print(f"  Steps:   {step_counts}")
    print(f"  Frames:  {num_frames}")
    print(f"  Seed:    {seed}")

    prompt = "A dog is running in the park."

    # Load pipeline once
    print("\n  Loading CogVideoX pipeline...")
    pipe, default_frames, _torch_dtype = _load_cogvideox_pipe(model)
    if num_frames is None:
        num_frames = default_frames
    print(f"  Pipeline loaded. Using {num_frames} frames.")

    orig_sdpa = F.scaled_dot_product_attention

    # Pre-build sage functions (triggers import once)
    sage_fns = {fmt: build_sage_fn(fmt) for fmt in formats}

    results: Dict[int, Dict[str, AccuracyMetrics]] = {}

    for step_idx, num_steps in enumerate(step_counts):
        step_start = time.time()
        print_header(f"Step count: {num_steps}  [{step_idx + 1}/{len(step_counts)}]")

        # --- Reference run with SDPA ---
        print(f"  Running SDPA reference ({num_steps} steps)...")
        ref_capture = LatentCapture()
        ref_capture.attach(pipe, num_steps)
        F.scaled_dot_product_attention = orig_sdpa
        _run_pipe(pipe, prompt, num_steps, num_frames, seed)
        ref_capture.detach(pipe)
        ref_latent = ref_capture.final_latent

        if ref_latent is None:
            print("  WARNING: Could not capture reference latent! Skipping.")
            continue
        print(f"  Reference latent shape: {ref_latent.shape}")

        step_results: Dict[str, AccuracyMetrics] = {}

        # --- Per-format runs ---
        for fmt in formats:
            print(f"  Running {fmt} ({num_steps} steps)...", end="", flush=True)
            fmt_start = time.time()

            try:
                fmt_capture = LatentCapture()
                fmt_capture.attach(pipe, num_steps)
                F.scaled_dot_product_attention = sage_fns[fmt]
                _run_pipe(pipe, prompt, num_steps, num_frames, seed)
                fmt_capture.detach(pipe)
                F.scaled_dot_product_attention = orig_sdpa

                fmt_latent = fmt_capture.final_latent
                if fmt_latent is None:
                    print(f" FAILED (no latent captured)")
                    continue

                m = compute_metrics(ref_latent, fmt_latent)
                step_results[fmt] = m
                fmt_elapsed = time.time() - fmt_start
                print(f"  SNR={m.snr_db:.2f}dB  CosSim={m.cos_sim:.6f}"
                      f"  L1Rel={m.l1_rel:.5f}  ({fmt_elapsed:.1f}s)")

                del fmt_latent
            except torch.cuda.OutOfMemoryError:
                print(f" OOM")
                F.scaled_dot_product_attention = orig_sdpa
            except Exception as e:
                print(f" ERROR: {e}")
                F.scaled_dot_product_attention = orig_sdpa

            gc.collect()
            torch.cuda.empty_cache()

        results[num_steps] = step_results
        step_elapsed = time.time() - step_start
        print(f"\n  Step count {num_steps} completed in {step_elapsed:.1f}s")

        del ref_latent
        gc.collect()
        torch.cuda.empty_cache()

    # Restore SDPA and clean up
    F.scaled_dot_product_attention = orig_sdpa
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    return results


def _get_snr_ranking(step_results: Dict[str, AccuracyMetrics]) -> List[str]:
    """Return format names sorted by SNR descending (best first)."""
    valid = {fmt: m for fmt, m in step_results.items()
             if isinstance(m, AccuracyMetrics)}
    return sorted(valid.keys(), key=lambda f: valid[f].snr_db, reverse=True)


def _get_l1_ranking(step_results: Dict[str, AccuracyMetrics]) -> List[str]:
    """Return format names sorted by L1_rel ascending (best first)."""
    valid = {fmt: m for fmt, m in step_results.items()
             if isinstance(m, AccuracyMetrics)}
    return sorted(valid.keys(), key=lambda f: valid[f].l1_rel)


def print_comparison_table(
    results: Dict[int, Dict[str, AccuracyMetrics]],
    formats: List[str],
    step_counts: List[int],
):
    """Print cross-step comparison table."""
    print_header("Cross-Step Comparison — SNR (dB) and L1 Relative Error")

    # Step count headers
    print(f"\n  {'':14s}", end="")
    for s in step_counts:
        label = f"{s} steps"
        print(f"  {label:^19s}", end="")
    print()

    # Sub-header: SNR / L1Rel
    print(f"  {'Format':<14s}", end="")
    for _ in step_counts:
        print(f"  {'SNR':>8s} {'L1Rel':>8s}", end="")
    print()

    # Separator
    print(f"  {'-'*14}", end="")
    for _ in step_counts:
        print(f"  {'-'*8} {'-'*8}", end="")
    print()

    # Data rows
    for fmt in formats:
        print(f"  {fmt:<14s}", end="")
        for s in step_counts:
            m = results.get(s, {}).get(fmt)
            if isinstance(m, AccuracyMetrics):
                print(f"  {m.snr_db:>8.2f} {m.l1_rel:>8.5f}", end="")
            else:
                print(f"  {'N/A':>8s} {'N/A':>8s}", end="")
        print()

    # CosSim / L2Rel supplementary table
    print(f"\n  {'':14s}", end="")
    for s in step_counts:
        label = f"{s} steps"
        print(f"  {label:^19s}", end="")
    print()

    print(f"  {'Format':<14s}", end="")
    for _ in step_counts:
        print(f"  {'CosSim':>8s} {'L2Rel':>8s}", end="")
    print()

    print(f"  {'-'*14}", end="")
    for _ in step_counts:
        print(f"  {'-'*8} {'-'*8}", end="")
    print()

    for fmt in formats:
        print(f"  {fmt:<14s}", end="")
        for s in step_counts:
            m = results.get(s, {}).get(fmt)
            if isinstance(m, AccuracyMetrics):
                print(f"  {m.cos_sim:>8.6f} {m.l2_rel:>8.6f}", end="")
            else:
                print(f"  {'N/A':>8s} {'N/A':>8s}", end="")
        print()

    print()


def print_ranking_analysis(
    results: Dict[int, Dict[str, AccuracyMetrics]],
    formats: List[str],
    step_counts: List[int],
):
    """Print ranking stability analysis."""
    print_header("Ranking Stability Analysis")

    # --- SNR ranking ---
    print("\n  Ranking by SNR (higher = better):")
    snr_rankings = {}
    for s in step_counts:
        if s not in results:
            continue
        ranking = _get_snr_ranking(results[s])
        snr_rankings[s] = ranking

        parts = []
        for fmt in ranking:
            m = results[s][fmt]
            parts.append(f"{fmt}({m.snr_db:.1f}dB)")
        print(f"  {s:>3d} steps: {' > '.join(parts)}")

    # Check SNR ranking stability
    if len(snr_rankings) >= 2:
        step_list = sorted(snr_rankings.keys())
        ref_ranking = snr_rankings[step_list[0]]
        snr_stable = all(snr_rankings[s] == ref_ranking for s in step_list[1:])
        print(f"\n  SNR rankings stable: {'YES' if snr_stable else 'NO'}")
        if not snr_stable:
            for s in step_list[1:]:
                if snr_rankings[s] != ref_ranking:
                    print(f"    Divergence at {s} steps: "
                          f"{' > '.join(snr_rankings[s])} vs "
                          f"{' > '.join(ref_ranking)} at {step_list[0]} steps")

    # --- L1_rel ranking ---
    print(f"\n  Ranking by L1 Relative Error (lower = better):")
    l1_rankings = {}
    for s in step_counts:
        if s not in results:
            continue
        ranking = _get_l1_ranking(results[s])
        l1_rankings[s] = ranking

        parts = []
        for fmt in ranking:
            m = results[s][fmt]
            parts.append(f"{fmt}({m.l1_rel:.5f})")
        print(f"  {s:>3d} steps: {' < '.join(parts)}")

    if len(l1_rankings) >= 2:
        step_list = sorted(l1_rankings.keys())
        ref_ranking = l1_rankings[step_list[0]]
        l1_stable = all(l1_rankings[s] == ref_ranking for s in step_list[1:])
        print(f"\n  L1_rel rankings stable: {'YES' if l1_stable else 'NO'}")

    # --- SNR drift analysis ---
    if len(step_counts) >= 2:
        base_step = step_counts[0]
        final_step = step_counts[-1]
        if base_step in results and final_step in results:
            print(f"\n  SNR drift from {base_step} to {final_step} steps:")
            for fmt in formats:
                base_m = results.get(base_step, {}).get(fmt)
                final_m = results.get(final_step, {}).get(fmt)
                if isinstance(base_m, AccuracyMetrics) and isinstance(final_m, AccuracyMetrics):
                    drift = final_m.snr_db - base_m.snr_db
                    print(f"    {fmt:<14s}: {base_m.snr_db:.2f} -> {final_m.snr_db:.2f} dB"
                          f"  (delta = {drift:+.2f} dB)")

    print()


def results_to_json(
    results: Dict[int, Dict[str, AccuracyMetrics]],
    step_counts: List[int],
    formats: List[str],
    metadata: Dict,
) -> Dict:
    """Convert results to JSON-serializable dict."""
    json_results = {"step_sweep": {}}

    for s in step_counts:
        if s not in results:
            continue
        step_data = {}
        for fmt in formats:
            m = results[s].get(fmt)
            if isinstance(m, AccuracyMetrics):
                step_data[fmt] = asdict(m)
            else:
                step_data[fmt] = None
        json_results["step_sweep"][str(s)] = step_data

    # Rankings by SNR
    rankings_snr = {}
    for s in step_counts:
        if s in results:
            rankings_snr[str(s)] = _get_snr_ranking(results[s])
    json_results["rankings_by_snr"] = rankings_snr

    # Rankings by L1_rel
    rankings_l1 = {}
    for s in step_counts:
        if s in results:
            rankings_l1[str(s)] = _get_l1_ranking(results[s])
    json_results["rankings_by_l1_rel"] = rankings_l1

    # Stability checks
    if len(rankings_snr) >= 2:
        step_keys = sorted(rankings_snr.keys(), key=int)
        first = rankings_snr[step_keys[0]]
        json_results["snr_rankings_stable"] = all(
            rankings_snr[k] == first for k in step_keys[1:]
        )
    else:
        json_results["snr_rankings_stable"] = None

    if len(rankings_l1) >= 2:
        step_keys = sorted(rankings_l1.keys(), key=int)
        first = rankings_l1[step_keys[0]]
        json_results["l1_rankings_stable"] = all(
            rankings_l1[k] == first for k in step_keys[1:]
        )
    else:
        json_results["l1_rankings_stable"] = None

    json_results["_metadata"] = metadata
    return json_results


def main():
    parser = argparse.ArgumentParser(
        description="Step-Count Proxy Validation — Is 5 Steps Enough?",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--formats", nargs="+", default=ALL_FORMATS,
        choices=ALL_FORMATS,
        help=f"Quant formats to test (default: all {len(ALL_FORMATS)})",
    )
    parser.add_argument(
        "--steps", nargs="+", type=int, default=DEFAULT_STEPS,
        help=f"Step counts to test (default: {DEFAULT_STEPS})",
    )
    parser.add_argument("--model", default="cogvideox-2b",
                        choices=["cogvideox-2b", "cogvideox1.5-5b"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-frames", type=int, default=1,
                        help="Frames to generate (default: 1 for speed)")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Save results to JSON file")

    args = parser.parse_args()

    step_counts = sorted(args.steps)

    print_header("Step-Count Proxy Validation — Is 5 Steps Enough?")
    print(f"  Steps:   {step_counts}")
    print(f"  Formats: {args.formats}")
    print(f"  Model:   {args.model}")
    print(f"  Seed:    {args.seed}")
    print(f"  Frames:  {args.num_frames}")

    start_time = time.time()

    results = run_step_sweep(
        formats=args.formats,
        step_counts=step_counts,
        model=args.model,
        seed=args.seed,
        num_frames=args.num_frames,
    )

    elapsed = time.time() - start_time

    # Print cross-step comparison table
    print_comparison_table(results, args.formats, step_counts)

    # Print ranking stability analysis
    print_ranking_analysis(results, args.formats, step_counts)

    print_header("Done")
    print(f"  Total time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")

    # Save JSON
    if args.output_json:
        metadata = {
            "step_counts": step_counts,
            "formats": args.formats,
            "model": args.model,
            "seed": args.seed,
            "num_frames": args.num_frames,
            "elapsed_s": elapsed,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        }
        json_data = results_to_json(results, step_counts, args.formats, metadata)
        with open(args.output_json, "w") as f:
            json.dump(json_data, f, indent=2)
        print(f"  Results saved to: {args.output_json}")


if __name__ == "__main__":
    main()
