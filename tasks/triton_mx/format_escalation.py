#!/usr/bin/env python3
"""
Format Escalation Experiment: Three-Tier FP4 -> FP8 -> SDPA Routing

Hypothesis: The step-skip experiment showed that falling back to SDPA on worst
steps recovers up to +8.8 dB. But MXFP8 achieves 0.9985 CosSim (nearly lossless)
while being much faster than SDPA. Many "moderate" steps may be fine with FP8
instead of full SDPA.

Three-tier routing:
  - Critical steps  -> SDPA (full precision)
  - Moderate steps  -> MXFP8_S1 (nearly lossless, fast)
  - Easy steps      -> FP4 (fastest)

This extends step_skip.py's binary skip (FP4 vs SDPA) to a three-tier
escalation (FP4 -> FP8 -> SDPA) for finer-grained accuracy/speed tradeoffs.

Usage:
  # Quick test with one format
  python format_escalation.py --formats nvfp4

  # Full run with all FP4 formats
  python format_escalation.py --output-json format_escalation_results.json

  # With custom step rankings file
  python format_escalation.py --step-rankings step_skip_results.json
"""

import os
import sys

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_standalone_path = os.path.join(_repo_root, "standalone")
if _standalone_path not in sys.path:
    sys.path.insert(0, _standalone_path)

import argparse
import gc
import json
import time
from dataclasses import asdict
from typing import Callable, Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F

# Reuse utilities from ablation.py (no duplication)
from ablation import (
    AccuracyMetrics,
    LatentCapture,
    compute_metrics,
    build_sage_fn,
    print_header,
    print_metrics_table,
    _load_cogvideox_pipe,
    _run_pipe,
)


ALL_FORMATS = ["nvfp4", "mxfp4", "mxfp4_s1"]

# Hardcoded consensus worst steps from step_skip profiling (fallback)
DEFAULT_CONSENSUS_WORST_STEPS = [5, 4, 1, 6, 3, 2, 0, 7, 8, 9]

# Escalation configurations: (n_critical_sdpa, n_moderate_fp8)
ESCALATION_CONFIGS = {
    "baseline":       (0, 0),   # 0 critical, 0 moderate -- all FP4
    "escalate_1_2":   (1, 2),   # 1 critical (SDPA), 2 moderate (FP8)
    "escalate_2_3":   (2, 3),   # 2 critical, 3 moderate
    "escalate_3_5":   (3, 5),   # 3 critical, 5 moderate
    "step_skip_5":    (5, 0),   # 5 critical, 0 moderate -- binary skip reference
    # Scaled for 50-step runs (matching ~25-40% coverage that worked at 20 steps)
    "escalate_5_8":   (5, 8),   # 5 SDPA + 8 FP8 = 26% non-FP4
    "escalate_5_13":  (5, 13),  # 5 SDPA + 13 FP8 = 36% non-FP4
    "escalate_8_12":  (8, 12),  # 8 SDPA + 12 FP8 = 40% non-FP4 (matches 3_5 at 20 steps)
    "step_skip_13":   (13, 0),  # 13 SDPA binary = 26% (matches skip_5 at 20 steps)
}


# ============================================================================
# EscalationAttention -- three-tier routing by denoising step index
# ============================================================================

class EscalationAttention:
    """Three-tier attention routing: FP4 -> FP8 -> SDPA by step index.

    Extends SelectiveStepAttention from step_skip.py to support an intermediate
    FP8 tier. Critical steps use SDPA, moderate steps use MXFP8, easy steps use FP4.
    """

    def __init__(
        self,
        fp4_fn: Callable,
        fp8_fn: Callable,
        sdpa_fn: Callable,
        critical_steps: Set[int],
        moderate_steps: Set[int],
        calls_per_step: int = 30,
    ):
        self.fp4_fn = fp4_fn
        self.fp8_fn = fp8_fn
        self.sdpa_fn = sdpa_fn
        self.critical_steps = critical_steps
        self.moderate_steps = moderate_steps
        self.calls_per_step = calls_per_step
        self._call_count = 0

    def __call__(self, *args, **kwargs):
        step_idx = self._call_count // self.calls_per_step
        self._call_count += 1
        if step_idx in self.critical_steps:
            return self.sdpa_fn(*args, **kwargs)
        elif step_idx in self.moderate_steps:
            return self.fp8_fn(*args, **kwargs)
        return self.fp4_fn(*args, **kwargs)

    def reset(self):
        self._call_count = 0


# ============================================================================
# Step Rankings Loading
# ============================================================================

def load_step_rankings(results_path: str = "step_skip_results.json") -> Dict[str, List[int]]:
    """Load profiled step rankings from step_skip experiment results.

    Returns a dict mapping format name -> list of step indices sorted from
    worst (most error-contributing) to best.

    Falls back to hardcoded consensus if the file is not found.
    """
    if not os.path.exists(results_path):
        print(f"  WARNING: {results_path} not found, using hardcoded consensus.")
        return {}

    with open(results_path) as f:
        data = json.load(f)

    rankings = {}
    for fmt, prof in data.get("profiling", {}).items():
        if fmt.startswith("_"):
            continue
        ranked = prof.get("worst_steps_ranked", [])
        if ranked:
            rankings[fmt] = ranked

    return rankings


def _get_worst_steps(
    fmt: str,
    rankings: Dict[str, List[int]],
) -> List[int]:
    """Get worst steps for a format, falling back to consensus if unavailable."""
    if fmt in rankings:
        return rankings[fmt]

    # Try consensus from rankings
    if "_consensus" in rankings:
        return rankings["_consensus"]

    # Fall back to hardcoded consensus
    return DEFAULT_CONSENSUS_WORST_STEPS


# ============================================================================
# Build Escalation Configs from Ranked Worst Steps
# ============================================================================

def build_escalation_configs(
    worst_steps: List[int],
    num_steps: int,
) -> Dict[str, Tuple[Set[int], Set[int]]]:
    """Build escalation configurations from ranked worst steps.

    Returns:
        Dict mapping config_name -> (critical_steps_set, moderate_steps_set)
    """
    configs = {}

    for config_name, (n_crit, n_mod) in ESCALATION_CONFIGS.items():
        # Take worst_steps[:n_crit] as critical (SDPA)
        critical = set(worst_steps[:n_crit]) if n_crit > 0 else set()
        # Take worst_steps[n_crit:n_crit+n_mod] as moderate (FP8)
        moderate = set(worst_steps[n_crit:n_crit + n_mod]) if n_mod > 0 else set()
        configs[config_name] = (critical, moderate)

    return configs


# ============================================================================
# Main Experiment
# ============================================================================

def run_escalation(
    formats: List[str],
    model: str,
    seed: int,
    num_steps: int,
    num_frames: Optional[int],
    rankings: Dict[str, List[int]],
) -> Dict:
    """Run the three-tier format escalation experiment.

    For each FP4 format, tests all escalation configurations and compares
    final latent accuracy against SDPA reference.
    """
    print_header("Format Escalation Experiment: FP4 -> FP8 -> SDPA")
    print(f"  Model: {model}, Steps: {num_steps}")
    print(f"  Formats: {formats}")
    print(f"  Seed: {seed}")
    print(f"  FP8 intermediate: mxfp8_s1")
    print(f"  Configs: {list(ESCALATION_CONFIGS.keys())}")

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
        print("  ERROR: Could not capture reference latent!")
        return {"error": "Failed to capture reference latent"}
    print(f"  Reference latent shape: {ref_latent.shape}")

    # --- Determine calls_per_step empirically ---
    call_counter = {"count": 0}

    def _counting_sdpa(*args, **kwargs):
        call_counter["count"] += 1
        return orig_sdpa(*args, **kwargs)

    F.scaled_dot_product_attention = _counting_sdpa
    _run_pipe(pipe, prompt, num_steps, num_frames, seed)
    F.scaled_dot_product_attention = orig_sdpa
    total_calls = call_counter["count"]
    calls_per_step = total_calls // num_steps if num_steps > 0 else total_calls
    print(f"  Detected {total_calls} attention calls "
          f"({calls_per_step} calls/step, {num_steps} steps)")

    results = {}

    # Pre-build the FP8 function (shared across all FP4 formats)
    fp8_fn = build_sage_fn("mxfp8_s1")

    # --- Per-format experiment ---
    for fmt in formats:
        print_header(f"Format Escalation: {fmt}")

        fp4_fn = build_sage_fn(fmt)
        worst_steps = _get_worst_steps(fmt, rankings)

        print(f"  Worst steps (ranked): {worst_steps[:8]}...")

        # Build all escalation configs for this format
        esc_configs = build_escalation_configs(worst_steps, num_steps)

        print(f"  Escalation configurations:")
        for config_name, (crit, mod) in esc_configs.items():
            n_crit = len(crit)
            n_mod = len(mod)
            n_fp4 = num_steps - n_crit - n_mod
            print(f"    {config_name}: "
                  f"{n_crit} SDPA + {n_mod} FP8 + {n_fp4} FP4 steps")

        fmt_results = {}
        fmt_rows = []
        baseline_snr = None

        for config_name, (critical_steps, moderate_steps) in esc_configs.items():
            n_crit = len(critical_steps)
            n_mod = len(moderate_steps)
            n_fp4 = num_steps - n_crit - n_mod
            print(f"\n  Config: {config_name} "
                  f"({n_crit} SDPA + {n_mod} FP8 + {n_fp4} FP4)...")

            esc_attn = EscalationAttention(
                fp4_fn, fp8_fn, orig_sdpa,
                critical_steps, moderate_steps,
                calls_per_step=calls_per_step,
            )

            capture = LatentCapture()
            capture.attach(pipe, num_steps)
            F.scaled_dot_product_attention = esc_attn
            with torch.no_grad():
                _run_pipe(pipe, prompt, num_steps, num_frames, seed)
            capture.detach(pipe)
            F.scaled_dot_product_attention = orig_sdpa

            fmt_latent = capture.final_latent
            if fmt_latent is None:
                print(f"    WARNING: Could not capture latent for {config_name}!")
                fmt_results[config_name] = {"error": "Failed to capture latent"}
                continue

            m = compute_metrics(ref_latent, fmt_latent)
            print(f"    Latent: SNR={m.snr_db:.2f}dB  CosSim={m.cos_sim:.8f}  "
                  f"L1Rel={m.l1_rel:.6f}")

            if config_name == "baseline":
                baseline_snr = m.snr_db

            fmt_results[config_name] = {
                "critical": n_crit,
                "moderate": n_mod,
                "critical_steps": sorted(critical_steps),
                "moderate_steps": sorted(moderate_steps),
                "metrics": asdict(m),
            }
            fmt_rows.append((config_name, m, n_crit, n_mod))

            del fmt_latent
            gc.collect()
            torch.cuda.empty_cache()

        # --- Summary table for this format ---
        if fmt_rows:
            print(f"\n  Summary -- {fmt}, {num_steps} steps")
            print(f"\n  {'Config':<16s} {'SDPA':>5s} {'FP8':>5s} {'FP4':>5s} "
                  f"{'Latent SNR':>11s} {'CosSim':>10s} {'L1Rel':>9s} "
                  f"{'delta_SNR':>10s}")
            print(f"  {'-'*16} {'-'*5} {'-'*5} {'-'*5} "
                  f"{'-'*11} {'-'*10} {'-'*9} {'-'*10}")
            for config_name, m, n_crit, n_mod in fmt_rows:
                n_fp4 = num_steps - n_crit - n_mod
                if baseline_snr is not None and config_name != "baseline":
                    d = m.snr_db - baseline_snr
                    delta = f"{d:+.2f}"
                else:
                    delta = "--"
                print(f"  {config_name:<16s} {n_crit:>5d} {n_mod:>5d} {n_fp4:>5d} "
                      f"{m.snr_db:>11.2f} {m.cos_sim:>10.6f} {m.l1_rel:>9.6f} "
                      f"{delta:>10s}")
            print()

        results[fmt] = fmt_results

    # --- Cross-format summary ---
    print_header("Cross-Format Summary -- Escalation Results")

    # Collect all configs across formats for comparison
    all_config_names = list(ESCALATION_CONFIGS.keys())

    for fmt in formats:
        if fmt not in results or isinstance(results[fmt], str):
            continue
        fmt_data = results[fmt]
        baseline = fmt_data.get("baseline", {})
        baseline_metrics = baseline.get("metrics", {})
        baseline_snr = baseline_metrics.get("snr_db")
        if baseline_snr is None:
            continue

        print(f"\n  {fmt}:")
        for config_name in all_config_names:
            if config_name == "baseline":
                print(f"    baseline: SNR={baseline_snr:.2f}dB (all FP4)")
                continue
            cfg = fmt_data.get(config_name, {})
            cfg_metrics = cfg.get("metrics", {})
            cfg_snr = cfg_metrics.get("snr_db")
            if cfg_snr is None:
                continue
            delta = cfg_snr - baseline_snr
            n_crit = cfg.get("critical", 0)
            n_mod = cfg.get("moderate", 0)
            n_fp4 = num_steps - n_crit - n_mod
            pct_sdpa = 100 * n_crit / num_steps
            pct_fp8 = 100 * n_mod / num_steps
            pct_fp4 = 100 * n_fp4 / num_steps
            print(f"    {config_name}: delta_SNR={delta:+.2f}dB  "
                  f"(SDPA:{pct_sdpa:.0f}% FP8:{pct_fp8:.0f}% FP4:{pct_fp4:.0f}%)")

    # --- Key comparison: escalation vs pure step-skip ---
    print_header("Escalation vs Binary Step-Skip")
    print(f"\n  Comparing escalate_2_3 (2 SDPA + 3 FP8) vs step_skip_5 (5 SDPA):")
    print(f"  Both use 5 'non-FP4' steps, but escalation replaces 3 SDPA with FP8.\n")

    for fmt in formats:
        if fmt not in results or isinstance(results[fmt], str):
            continue
        fmt_data = results[fmt]
        esc = fmt_data.get("escalate_2_3", {})
        skip = fmt_data.get("step_skip_5", {})
        baseline = fmt_data.get("baseline", {})

        esc_snr = esc.get("metrics", {}).get("snr_db")
        skip_snr = skip.get("metrics", {}).get("snr_db")
        base_snr = baseline.get("metrics", {}).get("snr_db")

        if all(v is not None for v in [esc_snr, skip_snr, base_snr]):
            esc_delta = esc_snr - base_snr
            skip_delta = skip_snr - base_snr
            overhead_saved = 3  # 3 steps changed from SDPA to FP8
            print(f"  {fmt}:")
            print(f"    step_skip_5:   delta_SNR={skip_delta:+.2f}dB (5 SDPA steps)")
            print(f"    escalate_2_3:  delta_SNR={esc_delta:+.2f}dB (2 SDPA + 3 FP8)")
            print(f"    quality gap:   {esc_delta - skip_delta:+.2f}dB "
                  f"(negative = escalation slightly worse)")
            print(f"    speed benefit: {overhead_saved} steps faster "
                  f"(FP8 vs SDPA)")

    # Clean up
    F.scaled_dot_product_attention = orig_sdpa
    del ref_latent, pipe
    gc.collect()
    torch.cuda.empty_cache()

    return results


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Format Escalation: Three-Tier FP4 -> FP8 -> SDPA Routing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--formats", nargs="+", default=ALL_FORMATS,
        choices=ALL_FORMATS + ["mxfp8_s1"],
        help=f"FP4 quant formats to test (default: {ALL_FORMATS}). "
             f"Note: mxfp8_s1 is the intermediate format, not tested as base.",
    )
    parser.add_argument("--model", default="cogvideox-2b",
                        choices=["cogvideox-2b", "cogvideox1.5-5b"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-steps", type=int, default=20,
                        help="Denoising steps (default: 20, matching step_skip)")
    parser.add_argument("--num-frames", type=int, default=1,
                        help="Frames to generate (default: 1)")
    parser.add_argument("--step-rankings", type=str, default="step_skip_results.json",
                        help="Path to step_skip_results.json for worst step rankings")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Save results to JSON file")

    args = parser.parse_args()

    # Filter out mxfp8_s1 if accidentally included (it's the intermediate format)
    formats = [f for f in args.formats if f != "mxfp8_s1"]
    if len(formats) != len(args.formats):
        print("  NOTE: mxfp8_s1 removed from test formats (it IS the intermediate format)")

    print_header("Format Escalation Experiment")
    print(f"  Formats (FP4 base): {formats}")
    print(f"  FP8 intermediate:   mxfp8_s1")
    print(f"  Model:              {args.model}")
    print(f"  Steps:              {args.num_steps}")
    print(f"  Frames:             {args.num_frames}")
    print(f"  Seed:               {args.seed}")
    print(f"  Step rankings:      {args.step_rankings}")

    start_time = time.time()

    # Load step rankings from profiling
    print(f"\n  Loading step rankings from: {args.step_rankings}")
    rankings = load_step_rankings(args.step_rankings)
    if rankings:
        print(f"  Found rankings for formats: {list(rankings.keys())}")
    else:
        print(f"  No rankings found, using hardcoded consensus: "
              f"{DEFAULT_CONSENSUS_WORST_STEPS}")

    # Run escalation experiment
    escalation_results = run_escalation(
        formats=formats,
        model=args.model,
        seed=args.seed,
        num_steps=args.num_steps,
        num_frames=args.num_frames,
        rankings=rankings,
    )

    elapsed = time.time() - start_time

    print_header("Done")
    print(f"  Total time: {elapsed:.1f}s")

    if args.output_json:
        all_results = {
            "escalation": escalation_results,
            "_metadata": {
                "formats": formats,
                "fp8_intermediate": "mxfp8_s1",
                "model": args.model,
                "seed": args.seed,
                "num_steps": args.num_steps,
                "num_frames": args.num_frames,
                "step_rankings_file": args.step_rankings,
                "escalation_configs": {
                    k: {"n_critical": v[0], "n_moderate": v[1]}
                    for k, v in ESCALATION_CONFIGS.items()
                },
                "elapsed_s": elapsed,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "gpu": (torch.cuda.get_device_name(0)
                        if torch.cuda.is_available() else "N/A"),
            },
        }
        with open(args.output_json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  Results saved to: {args.output_json}")


if __name__ == "__main__":
    main()
