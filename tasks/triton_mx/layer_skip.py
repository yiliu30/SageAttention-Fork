#!/usr/bin/env python3
"""
Layer-Skip Experiment: SDPA Fallback for Sensitive Layers

Hypothesis: By falling back to SDPA (full precision) on the most error-sensitive
layers and using the quantized kernel on the remaining layers, we can significantly
improve end-to-end latent accuracy with minimal speed cost.

Consensus worst layers from Part B ablation (ranked by error across all formats):
  layer_18 (score=40), layer_17 (36), layer_15 (29),
  layer_16 (25), layer_14 (23), layer_19 (22), layer_13, layer_12

Usage:
  # Quick test — 1 format (~1 min)
  python layer_skip.py --formats nvfp4

  # Full run — all formats (~5 min)
  python layer_skip.py --output-json layer_skip_results.json

  # Custom skip layers
  python layer_skip.py --formats nvfp4 --skip-layers 18 17 15 16 14
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import gc
import json
import time
from dataclasses import asdict
from typing import Callable, Dict, List, Optional, Set

import torch
import torch.nn.functional as F

# Reuse utilities from ablation.py (no duplication)
from ablation import (
    AccuracyMetrics,
    AttentionCapture,
    LatentCapture,
    _average_metrics,
    build_sage_fn,
    compute_metrics,
    print_header,
    print_metrics_table,
    _load_cogvideox_pipe,
    _run_pipe,
)


# ============================================================================
# Constants — Worst layers from Part B consensus ranking
# ============================================================================

WORST_LAYERS_RANKED = [18, 17, 15, 16, 14, 19, 13, 12]

SKIP_CONFIGS = {
    "baseline":  set(),                                    # 0 skipped — pure quantized
    "skip_top3": {18, 17, 15},                             # 3/30 = 10% SDPA
    "skip_top5": {18, 17, 15, 16, 14},                     # 5/30 = 17% SDPA
    "skip_top8": {18, 17, 15, 16, 14, 19, 13, 12},         # 8/30 = 27% SDPA
}

ALL_FORMATS = ["nvfp4", "mxfp4", "mxfp4_s1", "mxfp8_s1"]


# ============================================================================
# SelectiveAttention Hook
# ============================================================================

class SelectiveAttention:
    """Monkey-patches F.sdpa to route per-layer: quantized or SDPA fallback.

    Tracks a call counter that resets every `calls_per_step` calls (=30 for
    CogVideoX-2b). If the current layer index is in `skip_layers`, uses
    original SDPA; otherwise uses the quantized sage_fn.

    When capture=True, also records each attention output (CPU float32) for
    per-layer divergence analysis against an SDPA reference run.
    """

    def __init__(
        self,
        sage_fn: Callable,
        orig_sdpa: Callable,
        skip_layers: Set[int],
        calls_per_step: int = 30,
        capture: bool = False,
    ):
        self.sage_fn = sage_fn
        self.orig_sdpa = orig_sdpa
        self.skip_layers = skip_layers
        self.calls_per_step = calls_per_step
        self._call_count = 0
        self.capture = capture
        self.outputs: List[torch.Tensor] = []  # CPU float32 copies

    def __call__(self, *args, **kwargs):
        layer_idx = self._call_count % self.calls_per_step
        self._call_count += 1
        if layer_idx in self.skip_layers:
            result = self.orig_sdpa(*args, **kwargs)
        else:
            result = self.sage_fn(*args, **kwargs)
        if self.capture:
            self.outputs.append(result.detach().float().cpu())
        return result

    def reset(self):
        self._call_count = 0

    def clear(self):
        """Free captured outputs."""
        self.outputs.clear()


# ============================================================================
# Main Experiment
# ============================================================================

def run_layer_skip_experiment(
    formats: List[str],
    model: str,
    seed: int,
    num_steps: int,
    num_frames: Optional[int],
    skip_configs: Dict[str, Set[int]],
) -> Dict:
    """Run the layer-skip experiment for all formats and configurations."""

    print_header("Layer-Skip Experiment — SDPA Fallback for Sensitive Layers")
    print(f"  Model: {model}, Steps: {num_steps}")
    print(f"  Formats: {formats}")
    print(f"  Seed: {seed}")
    print(f"  Skip configurations:")
    for name, layers in skip_configs.items():
        if layers:
            print(f"    {name}: {sorted(layers)} ({len(layers)}/30 = {100*len(layers)/30:.0f}% SDPA)")
        else:
            print(f"    {name}: none (pure quantized)")

    prompt = "A dog is running in the park."

    # Load pipeline once
    print("\n  Loading CogVideoX pipeline...")
    pipe, default_frames, _torch_dtype = _load_cogvideox_pipe(model)
    if num_frames is None:
        num_frames = default_frames
    print(f"  Pipeline loaded. Using {num_frames} frames, {num_steps} steps.")

    # --- Reference run with SDPA ---
    print("\n  Running SDPA reference (latent + per-layer attention)...")
    orig_sdpa = F.scaled_dot_product_attention

    # Capture per-layer attention outputs for reference
    with AttentionCapture(orig_sdpa) as ref_attn_cap:
        ref_latent_capture = LatentCapture()
        ref_latent_capture.attach(pipe, num_steps)
        _run_pipe(pipe, prompt, num_steps, num_frames, seed)
        ref_latent_capture.detach(pipe)

    ref_latent = ref_latent_capture.final_latent
    ref_attn_outputs = ref_attn_cap.outputs
    num_attn_calls = len(ref_attn_outputs)
    calls_per_step = num_attn_calls // num_steps if num_steps > 0 else num_attn_calls

    if ref_latent is None:
        print("  ERROR: Could not capture reference latent!")
        return {"error": "Failed to capture reference latent"}
    print(f"  Reference latent shape: {ref_latent.shape}")
    print(f"  Captured {num_attn_calls} attention calls "
          f"({calls_per_step} layers/step, {num_steps} steps)")

    results = {}

    # --- Per-format experiment ---
    for fmt in formats:
        print_header(f"Layer-Skip — {fmt}")
        sage_fn = build_sage_fn(fmt)

        fmt_results = {}
        fmt_rows = []
        baseline_snr = None

        for config_name, skip_set in skip_configs.items():
            print(f"\n  Config: {config_name} "
                  f"({len(skip_set)}/30 layers skipped)...")

            selective = SelectiveAttention(
                sage_fn, orig_sdpa, skip_set, calls_per_step=30,
                capture=True,
            )

            capture = LatentCapture()
            capture.attach(pipe, num_steps)
            F.scaled_dot_product_attention = selective
            _run_pipe(pipe, prompt, num_steps, num_frames, seed)
            capture.detach(pipe)
            F.scaled_dot_product_attention = orig_sdpa

            fmt_latent = capture.final_latent
            if fmt_latent is None:
                print(f"    WARNING: Could not capture latent for {config_name}!")
                fmt_results[config_name] = {"error": "Failed to capture latent"}
                continue

            # --- Final-step latent divergence ---
            m = compute_metrics(ref_latent, fmt_latent)
            print(f"    Latent: SNR={m.snr_db:.2f}dB  CosSim={m.cos_sim:.8f}  "
                  f"L1Rel={m.l1_rel:.6f}")

            if config_name == "baseline":
                baseline_snr = m.snr_db

            # --- Per-layer attention divergence ---
            cfg_attn_outputs = selective.outputs
            n = min(len(ref_attn_outputs), len(cfg_attn_outputs))

            per_call_metrics = []
            for i in range(n):
                per_call_metrics.append(compute_metrics(ref_attn_outputs[i], cfg_attn_outputs[i]))

            # Group by layer index
            layer_metrics: Dict[int, List[AccuracyMetrics]] = {}
            for i, lm in enumerate(per_call_metrics):
                layer_idx = i % calls_per_step
                layer_metrics.setdefault(layer_idx, []).append(lm)

            layer_avg = {
                idx: _average_metrics(mlist)
                for idx, mlist in sorted(layer_metrics.items())
            }

            # Print per-layer table
            print(f"\n    Per-layer attention divergence ({calls_per_step} layers):")
            layer_rows = [(f"layer_{idx:02d}", lm) for idx, lm in sorted(layer_avg.items())]
            print_metrics_table(layer_rows, label_header="Layer")

            # Worst/best layers summary
            sorted_by_snr = sorted(layer_avg.items(), key=lambda x: x[1].snr_db)
            print("    Worst 3 layers (by SNR):")
            for idx, lm in sorted_by_snr[:3]:
                skipped = "SDPA" if idx in skip_set else "quant"
                print(f"      layer_{idx:02d} [{skipped}]: "
                      f"SNR={lm.snr_db:.2f}dB  CosSim={lm.cos_sim:.6f}")

            # Global attention average
            global_attn_avg = _average_metrics(per_call_metrics)
            print(f"    Attention avg: SNR={global_attn_avg.snr_db:.2f}dB  "
                  f"CosSim={global_attn_avg.cos_sim:.6f}")

            fmt_results[config_name] = {
                "skipped": len(skip_set),
                "skip_layers": sorted(skip_set),
                "latent_metrics": asdict(m),
                "attention_global_avg": asdict(global_attn_avg),
                "attention_per_layer": {
                    f"layer_{idx:02d}": asdict(lm)
                    for idx, lm in sorted(layer_avg.items())
                },
                "attention_worst_layers": [
                    {"layer": f"layer_{idx:02d}",
                     "mode": "sdpa" if idx in skip_set else "quantized",
                     **asdict(lm)}
                    for idx, lm in sorted_by_snr[:5]
                ],
            }
            fmt_rows.append((config_name, m))

            # Free captured attention outputs
            selective.clear()
            del fmt_latent, cfg_attn_outputs, per_call_metrics
            gc.collect()
            torch.cuda.empty_cache()

        # Print summary table for this format
        if fmt_rows:
            print(f"\n  Summary — {fmt}, {num_steps} steps")
            print(f"\n  {'Config':<16s} {'Skipped':>8s} {'Latent SNR':>11s} "
                  f"{'L1Rel':>9s} {'CosSim':>10s} {'Δ SNR':>8s}")
            print(f"  {'-'*16} {'-'*8} {'-'*11} {'-'*9} {'-'*10} {'-'*8}")
            for config_name, m in fmt_rows:
                skip_count = len(skip_configs[config_name])
                delta = ""
                if baseline_snr is not None and config_name != "baseline":
                    d = m.snr_db - baseline_snr
                    delta = f"+{d:.2f}" if d >= 0 else f"{d:.2f}"
                else:
                    delta = "—"
                print(f"  {config_name:<16s} {skip_count:>4d}/30  {m.snr_db:>11.2f} "
                      f"{m.l1_rel:>9.6f} {m.cos_sim:>10.6f} {delta:>8s}")
            print()

        results[fmt] = fmt_results

    # --- Cross-format summary ---
    print_header("Cross-Format Summary — Best Cost/Benefit Ratio")
    for fmt in formats:
        if fmt not in results or isinstance(results[fmt], str):
            continue
        fmt_data = results[fmt]
        baseline = fmt_data.get("baseline", {})
        baseline_latent = baseline.get("latent_metrics", {})
        baseline_snr = baseline_latent.get("snr_db", None)
        baseline_attn = baseline.get("attention_global_avg", {})
        baseline_attn_snr = baseline_attn.get("snr_db", None)
        if baseline_snr is None:
            continue

        print(f"\n  {fmt}:")
        for config_name in skip_configs:
            if config_name == "baseline":
                continue
            cfg = fmt_data.get(config_name, {})
            cfg_latent = cfg.get("latent_metrics", {})
            cfg_snr = cfg_latent.get("snr_db", None)
            cfg_attn = cfg.get("attention_global_avg", {})
            cfg_attn_snr = cfg_attn.get("snr_db", None)
            if cfg_snr is None:
                continue
            delta = cfg_snr - baseline_snr
            skip_count = cfg.get("skipped", 0)
            efficiency = delta / skip_count if skip_count > 0 else 0
            attn_delta = ""
            if baseline_attn_snr is not None and cfg_attn_snr is not None:
                ad = cfg_attn_snr - baseline_attn_snr
                attn_delta = f", attn Δ={ad:+.2f} dB"
            print(f"    {config_name}: latent Δ SNR = {delta:+.2f} dB "
                  f"({skip_count} layers, "
                  f"{efficiency:+.2f} dB/layer{attn_delta})")

    # Clean up
    F.scaled_dot_product_attention = orig_sdpa
    del ref_latent, ref_attn_outputs, pipe
    ref_attn_cap.clear()
    gc.collect()
    torch.cuda.empty_cache()

    return results


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Layer-Skip Experiment: SDPA Fallback for Sensitive Layers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
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
    parser.add_argument("--skip-layers", nargs="+", type=int, default=None,
                        help="Custom set of layers to skip (overrides defaults)")

    args = parser.parse_args()

    # Build skip configs
    if args.skip_layers is not None:
        # Custom skip config: just baseline + custom
        skip_configs = {
            "baseline": set(),
            "custom": set(args.skip_layers),
        }
    else:
        skip_configs = SKIP_CONFIGS

    start_time = time.time()

    results = run_layer_skip_experiment(
        formats=args.formats,
        model=args.model,
        seed=args.seed,
        num_steps=args.num_steps,
        num_frames=args.num_frames,
        skip_configs=skip_configs,
    )

    elapsed = time.time() - start_time

    print_header("Done")
    print(f"  Total time: {elapsed:.1f}s")

    if args.output_json:
        output = {
            "layer_skip": results,
            "_metadata": {
                "formats": args.formats,
                "model": args.model,
                "seed": args.seed,
                "num_steps": args.num_steps,
                "num_frames": args.num_frames,
                "skip_configs": {
                    k: sorted(v) for k, v in skip_configs.items()
                },
                "worst_layers_ranked": WORST_LAYERS_RANKED,
                "elapsed_s": elapsed,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "gpu": (torch.cuda.get_device_name(0)
                        if torch.cuda.is_available() else "N/A"),
            },
        }
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"  Results saved to: {args.output_json}")


if __name__ == "__main__":
    main()
