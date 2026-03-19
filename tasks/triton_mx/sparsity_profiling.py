#!/usr/bin/env python3
"""
Sparsity Profiling — Tile-Level Attention Pattern Analysis for CogVideoX-2b

Measures real attention sparsity patterns to validate whether tile-level
FP4/FP8 mixed-precision routing is viable.

Four measurements:
  1) Tile-level attention mass distribution (Gini, coverage curves)
  2) Per-head sparsity variation (sharpness ranking)
  3) Cheap proxy accuracy for predicting tile importance
  4) FP4 quantization error concentration in important tiles

Architecture note:
  CogVideoX-2b has 30 transformer blocks. Each block makes one
  F.scaled_dot_product_attention call per denoising step with shape
  (B=2, H=30, N=17776, D=64). All 30 heads are in the H dimension.

  call_count maps to:
    step_idx  = call_count // calls_per_step  (denoising step)
    layer_idx = call_count % calls_per_step   (transformer block)

Usage:
  # Full profiling (50 steps, GPU 6)
  CUDA_VISIBLE_DEVICES=6 python sparsity_profiling.py \\
      --output-json sparsity_profiling_results.json

  # Quick smoke test (5 steps, 2 sampled steps × 2 layers)
  CUDA_VISIBLE_DEVICES=6 python sparsity_profiling.py \\
      --num-steps 5 --sample-steps 1 3 --sample-layers 0 14

  # Include quantization error analysis (slower)
  CUDA_VISIBLE_DEVICES=6 python sparsity_profiling.py \\
      --measure-error --output-json sparsity_profiling_results.json
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import gc
import json
import math
import sys
import time
from dataclasses import dataclass, asdict, field
from typing import Callable, Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from scipy import stats as scipy_stats

# Add standalone + tasks/triton_mx to path
_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_standalone_path = os.path.join(_repo_root, "standalone")
if _standalone_path not in sys.path:
    sys.path.insert(0, _standalone_path)

# Reuse utilities from ablation.py
from ablation import (
    _load_cogvideox_pipe,
    _run_pipe,
    print_header,
)


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class TileStats:
    """Per-(call, head, batch) tile-level statistics."""
    step: int
    layer: int
    head: int
    batch: int
    # Sparsity
    gini: float = 0.0
    frac_lt_half_uniform: float = 0.0   # fraction of tiles < 0.5× uniform mass
    frac_lt_tenth_uniform: float = 0.0  # fraction of tiles < 0.1× uniform mass
    mean_tiles_for_90pct: float = 0.0   # mean across query rows
    median_tiles_for_90pct: float = 0.0
    mean_tiles_for_95pct: float = 0.0
    mean_tiles_for_99pct: float = 0.0
    # Proxies (populated later)
    proxy_rho_max_product: float = 0.0
    proxy_rho_norm_product: float = 0.0
    proxy_rho_sampled_dot: float = 0.0
    proxy_precision_at_20_max_product: float = 0.0
    proxy_precision_at_20_norm_product: float = 0.0
    proxy_precision_at_20_sampled_dot: float = 0.0
    proxy_roc_auc_max_product: float = 0.0
    proxy_roc_auc_norm_product: float = 0.0
    proxy_roc_auc_sampled_dot: float = 0.0


# ============================================================================
# Sparsity computation helpers
# ============================================================================

def compute_gini(values: np.ndarray) -> float:
    """Compute Gini coefficient. 0=uniform, 1=max concentration."""
    if len(values) == 0 or values.sum() == 0:
        return 0.0
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    cumsum = np.cumsum(sorted_vals)
    return (2 * np.sum((np.arange(1, n + 1) * sorted_vals)) / (n * cumsum[-1])) - (n + 1) / n


def compute_tiles_for_coverage(row_tile_masses: np.ndarray, threshold: float) -> np.ndarray:
    """For each query row, find how many tiles (sorted desc) cover `threshold` of total mass.

    Args:
        row_tile_masses: [num_q_rows, num_kv_tiles] — per-row, per-KV-tile attention mass
        threshold: fraction of total mass to cover (e.g., 0.9)

    Returns:
        [num_q_rows] — number of tiles needed per row
    """
    num_rows, num_tiles = row_tile_masses.shape
    # Sort each row descending
    sorted_masses = np.sort(row_tile_masses, axis=1)[:, ::-1]
    cumsum = np.cumsum(sorted_masses, axis=1)
    row_totals = row_tile_masses.sum(axis=1, keepdims=True)
    # Threshold: cumsum >= threshold * total
    target = threshold * row_totals
    # Find first index where cumsum >= target (add 1 for count)
    # Use argmax on boolean — finds first True
    reached = cumsum >= target
    # If never reached (shouldn't happen with threshold < 1), return num_tiles
    tiles_needed = np.where(
        reached.any(axis=1),
        np.argmax(reached, axis=1) + 1,
        num_tiles
    )
    return tiles_needed


def compute_proxy_metrics(
    proxy_scores: np.ndarray,
    actual_masses: np.ndarray,
    top_k_frac: float = 0.2,
) -> Tuple[float, float, float]:
    """Compute proxy accuracy metrics.

    Args:
        proxy_scores: [num_tiles] — proxy importance score per KV-tile
        actual_masses: [num_tiles] — actual attention mass per KV-tile
        top_k_frac: fraction for precision@k and ROC

    Returns:
        (spearman_rho, precision_at_k, roc_auc)
    """
    n = len(proxy_scores)
    k = max(1, int(n * top_k_frac))

    # Spearman rank correlation
    if np.std(proxy_scores) < 1e-12 or np.std(actual_masses) < 1e-12:
        rho = 0.0
    else:
        rho, _ = scipy_stats.spearmanr(proxy_scores, actual_masses)
        if np.isnan(rho):
            rho = 0.0

    # Precision@k: overlap of top-k by proxy vs top-k by actual
    top_k_proxy = set(np.argsort(proxy_scores)[-k:])
    top_k_actual = set(np.argsort(actual_masses)[-k:])
    precision_at_k = len(top_k_proxy & top_k_actual) / k if k > 0 else 0.0

    # ROC AUC: binary classification "is tile in top-k by actual mass?"
    labels = np.zeros(n, dtype=int)
    labels[np.argsort(actual_masses)[-k:]] = 1
    # Simple AUC via sorted ranks
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(labels, proxy_scores)
    except Exception:
        # Fallback: manual AUC via Wilcoxon-Mann-Whitney
        pos_scores = proxy_scores[labels == 1]
        neg_scores = proxy_scores[labels == 0]
        if len(pos_scores) == 0 or len(neg_scores) == 0:
            auc = 0.5
        else:
            auc = np.mean(np.array([
                np.mean(pos_scores > t) + 0.5 * np.mean(pos_scores == t)
                for t in neg_scores
            ]))

    return rho, precision_at_k, auc


# ============================================================================
# SparsityProfiler — hooks attention calls
# ============================================================================

class SparsityProfiler:
    """Hooks F.scaled_dot_product_attention to capture tile-level
    attention statistics on sampled calls.

    CogVideoX-2b: 30 transformer blocks × 50 steps = 1500 calls.
    Each call: (B=2, H=30, N=17776, D=64).
    """

    def __init__(
        self,
        sample_steps: Set[int],
        sample_layers: Set[int],
        tile_size: int = 128,
        calls_per_step: Optional[int] = None,
        measure_error: bool = False,
    ):
        self.sample_steps = set(sample_steps)
        self.sample_layers = set(sample_layers)
        self.tile_size = tile_size
        self.calls_per_step = calls_per_step  # auto-detected if None
        self.measure_error = measure_error

        self._call_count = 0
        self._detecting_cps = calls_per_step is None
        self._first_step_shapes: List[Tuple] = []

        self.results: List[TileStats] = []
        self._orig_sdpa = None

        # For error measurement
        self._sage_fp4_fn = None
        self._sage_fp8_fn = None
        if measure_error:
            from sageattention3_standalone import scaled_dot_product_attention as sage_sdpa
            import functools
            self._sage_fp4_fn = functools.partial(sage_sdpa, quant_format="mxfp4")
            self._sage_fp8_fn = functools.partial(sage_sdpa, quant_format="mxfp8_s1")

    def attach(self, orig_sdpa):
        """Save original SDPA and return self as replacement."""
        self._orig_sdpa = orig_sdpa
        return self

    def reset(self):
        self._call_count = 0

    def __call__(self, *args, **kwargs):
        # Extract q, k, v from args
        q = args[0] if len(args) > 0 else kwargs.get('query')
        k = args[1] if len(args) > 1 else kwargs.get('key')
        v = args[2] if len(args) > 2 else kwargs.get('value')

        # Auto-detect calls_per_step
        if self._detecting_cps:
            self._first_step_shapes.append(q.shape)
            # We'll finalize after the first step completes

        if self.calls_per_step is not None:
            step_idx = self._call_count // self.calls_per_step
            layer_idx = self._call_count % self.calls_per_step

            if step_idx in self.sample_steps and layer_idx in self.sample_layers:
                print(f"  [Profiler] Capturing step={step_idx}, layer={layer_idx}, "
                      f"shape={q.shape}")
                self._profile_call(q, k, v, step_idx, layer_idx)

        self._call_count += 1
        return self._orig_sdpa(*args, **kwargs)

    def finalize_detection(self):
        """Called after first step to determine calls_per_step."""
        if self._detecting_cps and len(self._first_step_shapes) > 0:
            self.calls_per_step = len(self._first_step_shapes)
            print(f"  [Profiler] Auto-detected calls_per_step = {self.calls_per_step}")
            self._detecting_cps = False
            # Reset call count — the first step was detection-only
            self._call_count = 0

    def _profile_call(self, q, k, v, step_idx, layer_idx):
        """Compute full attention matrix per (head, batch), extract tile stats."""
        B, H, N, D = q.shape
        scale = 1.0 / math.sqrt(D)
        T = self.tile_size
        num_tiles = math.ceil(N / T)
        uniform_mass = 1.0 / num_tiles  # expected mass per tile if uniform

        t0 = time.time()

        for h in range(H):
            for b in range(B):
                # One head, one batch — peak ~2.4 GB for N=17776
                qh = q[b, h, :, :].float()  # [N, D]
                kh = k[b, h, :, :].float()  # [N, D]

                # Full attention matrix in float32
                scores = torch.matmul(qh, kh.T) * scale  # [N, N]
                attn = torch.softmax(scores, dim=-1)      # [N, N]

                # === Measurement 1 & 2: Tile-level mass distribution ===
                tile_stats = self._compute_tile_sparsity(
                    attn, num_tiles, T, N, uniform_mass
                )

                # === Measurement 3: Proxy accuracy ===
                proxy_stats = self._compute_proxy_accuracy(
                    qh, kh, attn, num_tiles, T, N
                )

                # Merge stats
                ts = TileStats(
                    step=step_idx, layer=layer_idx, head=h, batch=b,
                    **tile_stats, **proxy_stats,
                )
                self.results.append(ts)

                del scores, attn
                torch.cuda.empty_cache()

        elapsed = time.time() - t0
        print(f"    step={step_idx}, layer={layer_idx}: "
              f"{H}h × {B}b = {H*B} pairs in {elapsed:.1f}s")

    def _compute_tile_sparsity(
        self, attn: torch.Tensor, num_tiles: int, T: int, N: int,
        uniform_mass: float,
    ) -> dict:
        """Compute tile-level sparsity metrics from attention matrix.

        Args:
            attn: [N, N] float32 attention weights (one head, one batch)
        """
        # Compute per-row, per-KV-tile mass
        # row_tile_mass[q_idx, kv_tile_j] = sum of attn[q_idx, j*T:(j+1)*T]
        # We do this efficiently using unfold or reshape
        attn_np = attn.cpu().numpy()

        # Pad to exact multiple of T if needed
        if N % T != 0:
            pad_n = num_tiles * T - N
            attn_padded = np.pad(attn_np, ((0, pad_n), (0, pad_n)), mode='constant')
        else:
            attn_padded = attn_np

        N_padded = num_tiles * T

        # Reshape to [N_orig, num_kv_tiles, T] and sum over T
        # But query rows are unpadded (only N original rows matter)
        attn_for_kv = attn_padded[:N, :].reshape(N, num_tiles, T)
        row_tile_mass = attn_for_kv.sum(axis=2)  # [N, num_kv_tiles]

        # Gini: compute per row, average
        ginis = np.array([compute_gini(row_tile_mass[i]) for i in range(N)])
        mean_gini = float(ginis.mean())

        # Fraction of tiles below thresholds (relative to uniform)
        frac_lt_half = float((row_tile_mass < 0.5 * uniform_mass).mean())
        frac_lt_tenth = float((row_tile_mass < 0.1 * uniform_mass).mean())

        # Coverage: tiles needed for 90%, 95%, 99%
        tiles_90 = compute_tiles_for_coverage(row_tile_mass, 0.9)
        tiles_95 = compute_tiles_for_coverage(row_tile_mass, 0.95)
        tiles_99 = compute_tiles_for_coverage(row_tile_mass, 0.99)

        return {
            "gini": mean_gini,
            "frac_lt_half_uniform": frac_lt_half,
            "frac_lt_tenth_uniform": frac_lt_tenth,
            "mean_tiles_for_90pct": float(tiles_90.mean()),
            "median_tiles_for_90pct": float(np.median(tiles_90)),
            "mean_tiles_for_95pct": float(tiles_95.mean()),
            "mean_tiles_for_99pct": float(tiles_99.mean()),
        }

    def _compute_proxy_accuracy(
        self, qh: torch.Tensor, kh: torch.Tensor,
        attn: torch.Tensor, num_tiles: int, T: int, N: int,
    ) -> dict:
        """Compute proxy accuracy metrics.

        Tests three proxies for predicting per-KV-tile importance.
        Proxies are evaluated per Q-tile (averaged across rows in the tile).
        """
        # Aggregate actual tile mass: average across all query rows per KV-tile
        attn_np = attn.cpu().numpy()
        if N % T != 0:
            pad_n = num_tiles * T - N
            attn_padded = np.pad(attn_np, ((0, pad_n), (0, pad_n)), mode='constant')
        else:
            attn_padded = attn_np
        # Mean mass per KV-tile across all query rows
        attn_for_kv = attn_padded[:N, :].reshape(N, num_tiles, T)
        mean_kv_tile_mass = attn_for_kv.sum(axis=2).mean(axis=0)  # [num_kv_tiles]

        qh_np = qh.cpu().numpy()  # [N, D]
        kh_np = kh.cpu().numpy()  # [N, D]
        D = qh_np.shape[1]

        # Pre-compute per-tile statistics
        q_tile_max = np.zeros(num_tiles)
        q_tile_norm = np.zeros(num_tiles)
        k_tile_max = np.zeros(num_tiles)
        k_tile_norm = np.zeros(num_tiles)

        for t in range(num_tiles):
            s, e = t * T, min((t + 1) * T, N)
            q_block = qh_np[s:e]
            k_block = kh_np[s:e]
            q_tile_max[t] = np.abs(q_block).max() if len(q_block) > 0 else 0
            q_tile_norm[t] = np.linalg.norm(q_block) if len(q_block) > 0 else 0
            k_tile_max[t] = np.abs(k_block).max() if len(k_block) > 0 else 0
            k_tile_norm[t] = np.linalg.norm(k_block) if len(k_block) > 0 else 0

        # Proxy A: max-product — for each KV-tile, take max across Q-tiles
        # (simplified: use global Q max × per-KV-tile K max)
        proxy_max = q_tile_max.max() * k_tile_max  # [num_kv_tiles]

        # Proxy B: norm-product — similar simplification
        proxy_norm = q_tile_norm.mean() * k_tile_norm  # [num_kv_tiles]

        # Proxy C: sampled dot — subsample query rows, compute small matmul
        sample_stride = 16
        q_sampled = qh_np[::sample_stride]  # [N//16, D]
        proxy_sampled = np.zeros(num_tiles)
        for t in range(num_tiles):
            s, e = t * T, min((t + 1) * T, N)
            k_block = kh_np[s:e]  # [T, D]
            if len(k_block) > 0 and len(q_sampled) > 0:
                dots = q_sampled @ k_block.T  # [N//16, T]
                proxy_sampled[t] = dots.max()

        # Compute metrics for each proxy
        actual = mean_kv_tile_mass
        results = {}
        for name, proxy in [("max_product", proxy_max),
                            ("norm_product", proxy_norm),
                            ("sampled_dot", proxy_sampled)]:
            rho, prec, auc = compute_proxy_metrics(proxy, actual)
            results[f"proxy_rho_{name}"] = rho
            results[f"proxy_precision_at_20_{name}"] = prec
            results[f"proxy_roc_auc_{name}"] = auc

        return results


# ============================================================================
# Aggregation and reporting
# ============================================================================

def aggregate_results(results: List[TileStats], config: dict) -> dict:
    """Aggregate per-(call, head, batch) results into summaries."""
    num_tiles = config["num_tiles"]

    # Group by various dimensions
    by_head: Dict[int, List[TileStats]] = {}
    by_step: Dict[int, List[TileStats]] = {}
    by_layer: Dict[int, List[TileStats]] = {}

    for r in results:
        by_head.setdefault(r.head, []).append(r)
        by_step.setdefault(r.step, []).append(r)
        by_layer.setdefault(r.layer, []).append(r)

    def avg_field(items, field):
        vals = [getattr(r, field) for r in items]
        return sum(vals) / len(vals) if vals else 0.0

    # Per-head aggregation
    agg_head = {}
    for h, items in sorted(by_head.items()):
        agg_head[str(h)] = {
            "mean_gini": avg_field(items, "gini"),
            "mean_tiles_for_90pct": avg_field(items, "mean_tiles_for_90pct"),
            "mean_tiles_for_90pct_frac": avg_field(items, "mean_tiles_for_90pct") / num_tiles,
            "mean_frac_lt_half_uniform": avg_field(items, "frac_lt_half_uniform"),
        }

    # Per-step aggregation
    agg_step = {}
    for s, items in sorted(by_step.items()):
        agg_step[str(s)] = {
            "mean_gini": avg_field(items, "gini"),
            "mean_tiles_for_90pct": avg_field(items, "mean_tiles_for_90pct"),
        }

    # Per-layer aggregation
    agg_layer = {}
    for l, items in sorted(by_layer.items()):
        agg_layer[str(l)] = {
            "mean_gini": avg_field(items, "gini"),
            "mean_tiles_for_90pct": avg_field(items, "mean_tiles_for_90pct"),
        }

    # Overall summary
    all_gini = [r.gini for r in results]
    all_t90 = [r.mean_tiles_for_90pct for r in results]
    all_frac_half = [r.frac_lt_half_uniform for r in results]
    all_frac_tenth = [r.frac_lt_tenth_uniform for r in results]

    # Find sharpest/most diffuse heads
    head_t90 = {h: avg_field(items, "mean_tiles_for_90pct")
                for h, items in by_head.items()}
    sharpest_head = min(head_t90, key=head_t90.get) if head_t90 else 0
    diffuse_head = max(head_t90, key=head_t90.get) if head_t90 else 0

    # Best proxy
    proxy_names = ["max_product", "norm_product", "sampled_dot"]
    proxy_rhos = {}
    proxy_aucs = {}
    proxy_threshold_stds = {}
    for name in proxy_names:
        rhos = [getattr(r, f"proxy_rho_{name}") for r in results]
        aucs = [getattr(r, f"proxy_roc_auc_{name}") for r in results]
        proxy_rhos[name] = sum(rhos) / len(rhos) if rhos else 0
        proxy_aucs[name] = sum(aucs) / len(aucs) if aucs else 0
        # Threshold stability: std of rho across calls (proxy for threshold std)
        proxy_threshold_stds[name] = float(np.std(rhos)) if rhos else 0

    best_proxy = max(proxy_rhos, key=proxy_rhos.get) if proxy_rhos else "none"

    summary = {
        "mean_gini": sum(all_gini) / len(all_gini) if all_gini else 0,
        "mean_tiles_for_90pct": sum(all_t90) / len(all_t90) if all_t90 else 0,
        "mean_tiles_for_90pct_frac": (sum(all_t90) / len(all_t90) / num_tiles) if all_t90 else 0,
        "frac_tiles_lt_half_uniform": sum(all_frac_half) / len(all_frac_half) if all_frac_half else 0,
        "frac_tiles_lt_tenth_uniform": sum(all_frac_tenth) / len(all_frac_tenth) if all_frac_tenth else 0,
        "sharpest_head": int(sharpest_head),
        "sharpest_head_tiles_90": head_t90.get(sharpest_head, 0),
        "most_diffuse_head": int(diffuse_head),
        "most_diffuse_head_tiles_90": head_t90.get(diffuse_head, 0),
        "head_tiles_90_std": float(np.std(list(head_t90.values()))) if head_t90 else 0,
        "best_proxy": best_proxy,
        "best_proxy_rho": proxy_rhos.get(best_proxy, 0),
        "best_proxy_roc_auc": proxy_aucs.get(best_proxy, 0),
        "best_proxy_threshold_std": proxy_threshold_stds.get(best_proxy, 0),
        "proxy_details": {
            name: {
                "mean_rho": proxy_rhos[name],
                "mean_roc_auc": proxy_aucs[name],
                "rho_std": proxy_threshold_stds[name],
                "mean_precision_at_20": avg_field(results, f"proxy_precision_at_20_{name}"),
            }
            for name in proxy_names
        },
    }

    return {
        "summary": summary,
        "aggregated_by_head": agg_head,
        "aggregated_by_step": agg_step,
        "aggregated_by_layer": agg_layer,
    }


def print_report(summary: dict, config: dict):
    """Print formatted sparsity report."""
    num_tiles = config["num_tiles"]
    uniform_mass_pct = 100.0 / num_tiles

    print_header("Attention Sparsity Report — CogVideoX-2b")

    print(f"\n  Tile-Level Sparsity (T={config['tile_size']}, "
          f"{num_tiles} tiles/dim, uniform={uniform_mass_pct:.2f}%/tile):")
    print(f"    Mean tiles for 90% coverage: "
          f"{summary['mean_tiles_for_90pct']:.1f} / {num_tiles} "
          f"({summary['mean_tiles_for_90pct_frac']*100:.1f}%)")
    print(f"    Mean Gini coefficient: {summary['mean_gini']:.4f} "
          f"(0=uniform, 1=max concentration)")
    print(f"    {summary['frac_tiles_lt_half_uniform']*100:.1f}% of tiles carry "
          f"<0.5× uniform mass (<{uniform_mass_pct*0.5:.2f}%)")
    print(f"    {summary['frac_tiles_lt_tenth_uniform']*100:.1f}% of tiles carry "
          f"<0.1× uniform mass (<{uniform_mass_pct*0.1:.3f}%)")

    print(f"\n  Per-Head Variation:")
    print(f"    Sharpest:  head {summary['sharpest_head']} — "
          f"{summary['sharpest_head_tiles_90']:.1f} tiles for 90%")
    print(f"    Diffuse:   head {summary['most_diffuse_head']} — "
          f"{summary['most_diffuse_head_tiles_90']:.1f} tiles for 90%")
    print(f"    Std across heads: {summary['head_tiles_90_std']:.1f} tiles")

    print(f"\n  Proxy Accuracy (ρ / P@20% / ROC AUC / ρ_std):")
    for name, details in summary["proxy_details"].items():
        marker = " ← best" if name == summary["best_proxy"] else ""
        print(f"    {name:15s}: ρ={details['mean_rho']:.3f} / "
              f"P@20={details['mean_precision_at_20']:.3f} / "
              f"AUC={details['mean_roc_auc']:.3f} / "
              f"σ={details['rho_std']:.3f}{marker}")

    # Decision
    t90 = summary["mean_tiles_for_90pct"]
    best_rho = summary["best_proxy_rho"]
    best_auc = summary["best_proxy_roc_auc"]

    print(f"\n  Decision Evaluation:")
    print(f"    tiles_for_90pct = {t90:.1f} (threshold: <30=GO, 30-50=EXPLORE, >50=STOP)")
    print(f"    best_proxy ρ = {best_rho:.3f} (threshold: >0.8=good, 0.6-0.8=weak, <0.6=poor)")
    print(f"    best_proxy AUC = {best_auc:.3f} (threshold: >0.9=good)")

    if t90 < 30 and best_rho > 0.8 and best_auc > 0.9:
        conclusion = "GO — strong sparsity, predictable proxies"
    elif t90 < 30 and best_rho >= 0.6:
        conclusion = "EXPLORE — strong sparsity, proxy needs refinement"
    elif t90 < 50 and best_rho > 0.8:
        conclusion = "EXPLORE — moderate sparsity, good proxy"
    elif t90 < 50:
        conclusion = "CAUTION — moderate sparsity, weak proxy"
    else:
        conclusion = "STOP — attention too diffuse for tile-level mixing"

    print(f"\n  *** Conclusion: {conclusion} ***\n")


# ============================================================================
# Main pipeline
# ============================================================================

def run_profiling(
    num_steps: int = 50,
    num_frames: Optional[int] = None,
    sample_steps: Optional[List[int]] = None,
    sample_layers: Optional[List[int]] = None,
    tile_size: int = 128,
    seed: int = 42,
    model: str = "cogvideox-2b",
    measure_error: bool = False,
    output_json: Optional[str] = None,
):
    """Run the full sparsity profiling experiment."""
    if sample_steps is None:
        sample_steps = [1, 5, 10, 25, 49]
    if sample_layers is None:
        sample_layers = [0, 7, 14, 18, 21, 28]

    # Filter sample_steps to valid range
    sample_steps = [s for s in sample_steps if s < num_steps]

    print_header("Sparsity Profiling — CogVideoX-2b Attention Patterns")
    print(f"  Model: {model}")
    print(f"  Steps: {num_steps}, Seed: {seed}")
    print(f"  Tile size: {tile_size}")
    print(f"  Sample steps: {sample_steps}")
    print(f"  Sample layers: {sample_layers}")
    print(f"  Measure error: {measure_error}")

    # Load pipeline
    print("\n  Loading CogVideoX pipeline...")
    pipe, default_frames, torch_dtype = _load_cogvideox_pipe(model)
    if num_frames is None:
        num_frames = default_frames
    print(f"  Loaded. num_frames={num_frames}")

    prompt = "A dog is running in the park."

    # === Step 1: Detect calls_per_step ===
    print("\n  Step 1: Detecting calls_per_step...")
    orig_sdpa = F.scaled_dot_product_attention
    detect_count = [0]

    def counting_sdpa(*args, **kwargs):
        detect_count[0] += 1
        return orig_sdpa(*args, **kwargs)

    F.scaled_dot_product_attention = counting_sdpa
    _run_pipe(pipe, prompt, num_steps=1, num_frames=num_frames, seed=seed)
    calls_per_step = detect_count[0]
    F.scaled_dot_product_attention = orig_sdpa
    print(f"  Detected calls_per_step = {calls_per_step}")

    # Filter sample_layers
    sample_layers = [l for l in sample_layers if l < calls_per_step]
    print(f"  Valid sample layers: {sample_layers}")

    gc.collect()
    torch.cuda.empty_cache()

    # Compute config
    seq_len = 17776  # CogVideoX-2b default
    num_tiles = math.ceil(seq_len / tile_size)

    config = {
        "model": model,
        "num_steps": num_steps,
        "num_frames": num_frames,
        "tile_size": tile_size,
        "seq_len": seq_len,
        "num_tiles": num_tiles,
        "seed": seed,
        "sampled_steps": sample_steps,
        "sampled_layers": sample_layers,
        "calls_per_step": calls_per_step,
        "uniform_baseline_mass": 1.0 / num_tiles,
    }

    # === Step 2: Run profiling ===
    print(f"\n  Step 2: Running profiling ({len(sample_steps)} steps × "
          f"{len(sample_layers)} layers × {calls_per_step} heads × 2 batch = "
          f"{len(sample_steps) * len(sample_layers) * calls_per_step * 2} pairs)...")

    # Wait — calls_per_step here is the number of transformer blocks (30),
    # NOT the number of heads. Heads are H=30 in the tensor dimension.
    # The profiler processes all H heads within each sampled call.
    num_profiled_pairs = (len(sample_steps) * len(sample_layers) * 30 * 2)  # H=30, B=2
    print(f"  (Corrected: {len(sample_steps)} steps × {len(sample_layers)} layers "
          f"× 30 heads × 2 batch = {num_profiled_pairs} pairs)")
    est_time = num_profiled_pairs * 5  # ~5s per (head, batch)
    print(f"  Estimated time: ~{est_time // 60} min")

    profiler = SparsityProfiler(
        sample_steps=sample_steps,
        sample_layers=sample_layers,
        tile_size=tile_size,
        calls_per_step=calls_per_step,
        measure_error=measure_error,
    )
    profiler.attach(orig_sdpa)

    F.scaled_dot_product_attention = profiler

    t_start = time.time()
    _run_pipe(pipe, prompt, num_steps=num_steps, num_frames=num_frames, seed=seed)
    t_elapsed = time.time() - t_start

    F.scaled_dot_product_attention = orig_sdpa

    print(f"\n  Profiling complete: {len(profiler.results)} stats captured "
          f"in {t_elapsed:.1f}s ({t_elapsed/60:.1f} min)")

    # === Step 3: Aggregate and report ===
    print("\n  Step 3: Aggregating results...")
    agg = aggregate_results(profiler.results, config)

    print_report(agg["summary"], config)

    # === Step 4: Save results ===
    if output_json:
        output = {
            "config": config,
            **agg,
            "per_call": [asdict(r) for r in profiler.results],
        }
        with open(output_json, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"  Results saved to {output_json}")

    return agg


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Sparsity profiling for tile-level mixed-precision attention")
    parser.add_argument("--num-steps", type=int, default=50,
                        help="Number of denoising steps (default: 50)")
    parser.add_argument("--num-frames", type=int, default=None,
                        help="Number of video frames (default: model default)")
    parser.add_argument("--sample-steps", type=int, nargs="+",
                        default=None,
                        help="Denoising steps to sample (default: 1 5 10 25 49)")
    parser.add_argument("--sample-layers", type=int, nargs="+",
                        default=None,
                        help="Layers to sample (default: 0 7 14 18 21 28)")
    parser.add_argument("--tile-size", type=int, default=128,
                        help="Tile size (default: 128)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--model", type=str, default="cogvideox-2b",
                        help="Model name (default: cogvideox-2b)")
    parser.add_argument("--measure-error", action="store_true",
                        help="Also measure FP4/FP8 quantization error concentration")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Output JSON file path")
    args = parser.parse_args()

    run_profiling(
        num_steps=args.num_steps,
        num_frames=args.num_frames,
        sample_steps=args.sample_steps,
        sample_layers=args.sample_layers,
        tile_size=args.tile_size,
        seed=args.seed,
        model=args.model,
        measure_error=args.measure_error,
        output_json=args.output_json,
    )


if __name__ == "__main__":
    main()
