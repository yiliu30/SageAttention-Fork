#!/usr/bin/env python3
"""
E2E pixel-level comparison: original monolith vs refactored sage3 package.

For each quant format (nvfp4, mxfp4, mxfp4_s1, mxfp8_s1), generates one frame
with the original sageattention3_standalone.py and one with the sage3/ package,
then compares the outputs at the pixel level.

The refactored package intentionally diverges from the original monolith for
`nvfp4` and `mxfp4` after RFC #8, so monolith-vs-refactored image similarity is
informational only for those two formats. The monolith remains the reference for
`mxfp4_s1` and `mxfp8_s1`, where direct parity is still expected aside from
minor Triton compilation non-determinism.

Expected results:
  - NVFP4/MXFP4: informational comparison only (expected to differ from monolith)
  - MXFP4_S1/MXFP8_S1: PSNR ~25-40 dB and CosSim > 0.995

Usage:
    cd example
    python test_refactored_e2e.py [--formats nvfp4 mxfp4 mxfp4_s1 mxfp8_s1]
                                   [--num-frames 1]
                                   [--psnr-threshold 25]
"""

import argparse
import gc
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Add paths
standalone_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'standalone')
if standalone_path not in sys.path:
    sys.path.insert(0, standalone_path)


def load_pipeline(model_path, torch_dtype):
    """Load CogVideoX pipeline once, shared across all runs."""
    from diffusers import CogVideoXPipeline
    pipe = CogVideoXPipeline.from_pretrained(model_path, torch_dtype=torch_dtype)
    pipe.to("cuda")
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    return pipe


def generate_frame(pipe, attn_fn, num_frames=1, seed=42, num_inference_steps=5):
    """Generate a single frame with the given attention function."""
    # Monkey-patch SDPA
    orig_sdpa = F.scaled_dot_product_attention
    F.scaled_dot_product_attention = attn_fn

    try:
        video = pipe(
            prompt="A dog is running in the park.",
            num_videos_per_prompt=1,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            guidance_scale=6,
            generator=torch.Generator(device="cuda").manual_seed(seed),
        ).frames[0]

        # Extract first frame as numpy array
        frame = np.array(video[0].convert("RGB"))
        return frame
    finally:
        # Restore original SDPA
        F.scaled_dot_product_attention = orig_sdpa


def compute_metrics(img_a, img_b):
    """Compute PSNR, cosine similarity, MSE, max abs difference."""
    assert img_a.shape == img_b.shape, f"Shape mismatch: {img_a.shape} vs {img_b.shape}"

    a = img_a.astype(np.float64)
    b = img_b.astype(np.float64)
    diff = a - b

    mse = np.mean(diff ** 2)
    max_abs_diff = np.max(np.abs(diff))
    mean_abs_diff = np.mean(np.abs(diff))

    if mse == 0:
        psnr = float("inf")
    else:
        psnr = 10.0 * np.log10(255.0 ** 2 / mse)

    a_flat = a.ravel()
    b_flat = b.ravel()
    dot = np.dot(a_flat, b_flat)
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    cos_sim = float(dot / (norm_a * norm_b)) if (norm_a > 0 and norm_b > 0) else 0.0

    return {
        "psnr": psnr,
        "cos_sim": cos_sim,
        "mse": mse,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
    }


def make_attn_fn_original(quant_format):
    """Create attention function using the original monolith.

    The original monolith captures SAGE3_QUANT_FORMAT at import time (H2 bug),
    so we must pass quant_format= explicitly via functools.partial to override.
    """
    import functools
    from sageattention3_standalone import scaled_dot_product_attention
    return functools.partial(scaled_dot_product_attention, quant_format=quant_format)


def make_attn_fn_refactored(quant_format):
    """Create attention function using the refactored sage3 package.

    The refactored package reads env vars lazily (H2 fix), but we pass
    quant_format= explicitly for consistency with the original monolith test.
    """
    import functools
    from sage3 import scaled_dot_product_attention as sage3_sdpa
    return functools.partial(sage3_sdpa, quant_format=quant_format)


def save_frame(frame, path):
    """Save a numpy frame to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(frame).save(path)


def save_diff_image(img_a, img_b, path, amplify=10):
    """Save an amplified difference image for visual inspection."""
    diff = np.abs(img_a.astype(np.float64) - img_b.astype(np.float64))
    diff_amplified = np.clip(diff * amplify, 0, 255).astype(np.uint8)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(diff_amplified).save(path)


def main():
    parser = argparse.ArgumentParser(
        description="E2E pixel-level comparison: original monolith vs refactored sage3"
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["nvfp4", "mxfp4", "mxfp4_s1", "mxfp8_s1"],
        help="Quant formats to test",
    )
    parser.add_argument("--num-frames", type=int, default=1, help="Frames per video")
    parser.add_argument("--num-steps", type=int, default=5, help="Inference steps (5 for smoke test)")
    parser.add_argument("--psnr-threshold", type=float, default=25.0,
                        help="Minimum PSNR in dB (25 = visually similar, 40+ = nearly identical)")
    parser.add_argument("--cos-threshold", type=float, default=0.995,
                        help="Minimum cosine similarity")
    parser.add_argument("--save-images", action="store_true",
                        help="Save generated frames and diff images to disk")
    args = parser.parse_args()

    # Suppress debug output for clean E2E test
    os.environ['SAGE3_DEBUG'] = '0'
    os.environ['SAGE3_BENCHMARK'] = '0'

    model_path = "/storage/yiliu7/THUDM/CogVideoX-2b"
    torch_dtype = torch.float16

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = f"videos/cogvideox-2b/e2e_comparison/{timestamp}"

    print("=" * 70)
    print("E2E Pixel-Level Comparison: Original Monolith vs Refactored sage3/")
    print("=" * 70)
    print(f"  Formats     : {args.formats}")
    print(f"  Num frames  : {args.num_frames}")
    print(f"  Num steps   : {args.num_steps}")
    print(f"  PSNR thresh : {args.psnr_threshold} dB")
    print(f"  CosSim thresh: {args.cos_threshold}")
    print(f"  Save images : {args.save_images}")
    print()

    # Load pipeline once
    print("[0/N] Loading CogVideoX-2b pipeline...")
    pipe = load_pipeline(model_path, torch_dtype)
    print("      Pipeline loaded.\n")

    results = {}
    all_passed = True
    info_only_formats = {"nvfp4", "mxfp4"}

    for i, fmt in enumerate(args.formats, 1):
        print(f"[{i}/{len(args.formats)}] Testing format: {fmt}")
        print(f"  Generating frame with ORIGINAL monolith ({fmt})...")

        # Generate with original
        os.environ['SAGE3_QUANT_FORMAT'] = fmt
        attn_orig = make_attn_fn_original(fmt)
        frame_orig = generate_frame(
            pipe, attn_orig,
            num_frames=args.num_frames,
            num_inference_steps=args.num_steps,
        )

        # Clear GPU cache between runs
        gc.collect()
        torch.cuda.empty_cache()

        print(f"  Generating frame with REFACTORED sage3/ ({fmt})...")

        # Generate with refactored
        os.environ['SAGE3_QUANT_FORMAT'] = fmt
        attn_ref = make_attn_fn_refactored(fmt)
        frame_ref = generate_frame(
            pipe, attn_ref,
            num_frames=args.num_frames,
            num_inference_steps=args.num_steps,
        )

        gc.collect()
        torch.cuda.empty_cache()

        # Compute metrics
        metrics = compute_metrics(frame_orig, frame_ref)
        results[fmt] = metrics

        passed = (
            (metrics["psnr"] >= args.psnr_threshold or metrics["psnr"] == float("inf"))
            and metrics["cos_sim"] >= args.cos_threshold
        )
        if fmt in info_only_formats:
            status = "ℹ️ INFO"
        else:
            status = "✅ PASS" if passed else "❌ FAIL"
            if not passed:
                all_passed = False

        print(f"  {status}")
        print(f"    PSNR             : {metrics['psnr']:.2f} dB")
        print(f"    Cosine Similarity: {metrics['cos_sim']:.8f}")
        print(f"    MSE              : {metrics['mse']:.4f}")
        print(f"    Max Abs Diff     : {metrics['max_abs_diff']:.1f}")
        print(f"    Mean Abs Diff    : {metrics['mean_abs_diff']:.4f}")
        if fmt in info_only_formats:
            print("    Note             : informational only after RFC #8 denominator change")

        # Save images if requested
        if args.save_images:
            save_frame(frame_orig, f"{output_dir}/{fmt}_original.png")
            save_frame(frame_ref, f"{output_dir}/{fmt}_refactored.png")
            save_diff_image(frame_orig, frame_ref, f"{output_dir}/{fmt}_diff_10x.png", amplify=10)
            print(f"    Saved to: {output_dir}/{fmt}_*.png")

        print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Format':<12} {'PSNR':>10} {'CosSim':>12} {'MaxDiff':>10} {'MeanDiff':>10} {'Status':>8}")
    print("-" * 70)
    for fmt, m in results.items():
        psnr_str = "inf" if m["psnr"] == float("inf") else f"{m['psnr']:.2f}"
        passed = (
            (m["psnr"] >= args.psnr_threshold or m["psnr"] == float("inf"))
            and m["cos_sim"] >= args.cos_threshold
        )
        if fmt in info_only_formats:
            status = "INFO"
        else:
            status = "PASS" if passed else "FAIL"
        print(f"{fmt:<12} {psnr_str:>10} {m['cos_sim']:>12.8f} {m['max_abs_diff']:>10.1f} {m['mean_abs_diff']:>10.4f} {status:>8}")
    print("-" * 70)

    overall = "ALL PASSED ✅" if all_passed else "SOME FAILED ❌"
    print(f"\nOverall: {overall}")
    print("=" * 70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
