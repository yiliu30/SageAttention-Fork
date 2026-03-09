#!/usr/bin/env python3
"""Compare two images using PSNR and Cosine Similarity.

Usage:
    python compare_frames.py <image_a> <image_b> [--threshold 20.0] [--cos-threshold 0.95]

Exit codes:
    0 — PASS (both metrics meet thresholds)
    1 — FAIL (any metric below threshold, or error)
"""

import argparse
import sys

import numpy as np
from PIL import Image


def compute_psnr(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """Compute PSNR between two uint8 images (higher is better, inf if identical)."""
    assert img_a.shape == img_b.shape, (
        f"Shape mismatch: {img_a.shape} vs {img_b.shape}"
    )
    mse = np.mean((img_a.astype(np.float64) - img_b.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(255.0**2 / mse)


def compute_cosine_similarity(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """Compute cosine similarity between two images flattened to 1-D vectors.

    Returns a value in [-1, 1]; 1.0 means identical direction.
    """
    assert img_a.shape == img_b.shape, (
        f"Shape mismatch: {img_a.shape} vs {img_b.shape}"
    )
    a = img_a.astype(np.float64).ravel()
    b = img_b.astype(np.float64).ravel()
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare two images via PSNR and Cosine Similarity"
    )
    parser.add_argument("image_a", help="Path to first image")
    parser.add_argument("image_b", help="Path to second image")
    parser.add_argument(
        "--threshold",
        type=float,
        default=20.0,
        help="Minimum PSNR (dB) to pass (default: 20.0)",
    )
    parser.add_argument(
        "--cos-threshold",
        type=float,
        default=0.95,
        help="Minimum cosine similarity to pass (default: 0.95)",
    )
    args = parser.parse_args()

    try:
        a = np.array(Image.open(args.image_a).convert("RGB"))
        b = np.array(Image.open(args.image_b).convert("RGB"))
    except Exception as e:
        print(f"Error loading images: {e}", file=sys.stderr)
        return 1

    psnr = compute_psnr(a, b)
    cos_sim = compute_cosine_similarity(a, b)

    print(f"PSNR:              {psnr:.2f} dB")
    print(f"Cosine Similarity: {cos_sim:.6f}")

    passed = True

    if psnr < args.threshold:
        print(f"\u274c FAIL  PSNR {psnr:.2f} < {args.threshold:.2f} dB")
        passed = False
    else:
        print(f"\u2705 PASS  PSNR >= {args.threshold:.2f} dB")

    if cos_sim < args.cos_threshold:
        print(f"\u274c FAIL  CosSim {cos_sim:.6f} < {args.cos_threshold}")
        passed = False
    else:
        print(f"\u2705 PASS  CosSim {cos_sim:.6f} >= {args.cos_threshold}")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
