#!/usr/bin/env python3
"""
Step-Skip Experiment: SDPA Fallback for Sensitive Denoising Steps

Hypothesis: Quantization error accumulates unevenly across denoising steps.
By falling back to SDPA on the most error-sensitive steps and using the
quantized kernel on the remaining steps, we can recover end-to-end accuracy
with minimal speed cost.

Two phases:
  Phase 1 — Per-step error profiling: capture scheduler latent at every step,
            compute cumulative divergence and marginal delta to rank steps.
  Phase 2 — Step-skip validation: use SDPA for worst-N steps (from Phase 1),
            quantized for the rest. Measure final latent SNR improvement.

Usage:
  # Quick test — Phase 1 only, 1 format
  python step_skip.py --phases 1 --formats nvfp4

  # Full run — both phases, all formats (~42 min)
  python step_skip.py --output-json step_skip_results.json

  # Custom step count
  python step_skip.py --phases 1 2 --num-steps 50 --formats nvfp4 mxfp4_s1
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

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


ALL_FORMATS = ["nvfp4", "mxfp4", "mxfp4_s1", "mxfp8_s1"]


# ============================================================================
# PerStepLatentCapture — hooks scheduler.step to capture every step's latent
# ============================================================================

class PerStepLatentCapture:
    """Hooks pipe.scheduler.step to capture the latent at every denoising step.

    After a pipeline run, self.latents contains one CPU float32 tensor per
    scheduler step, in order [step_0, step_1, ..., step_{N-1}].
    """

    def __init__(self):
        self.latents: List[torch.Tensor] = []  # one per step, CPU float32
        self._orig_step = None

    def attach(self, pipe):
        self._orig_step = pipe.scheduler.step
        capture = self

        def _hooked_step(*args, **kwargs):
            result = capture._orig_step(*args, **kwargs)
            # Handle both return_dict=True (dataclass) and
            # return_dict=False (tuple) — CogVideoX uses tuple.
            if isinstance(result, tuple):
                latent = result[0]
            else:
                latent = result.prev_sample
            capture.latents.append(latent.detach().float().cpu())
            return result

        pipe.scheduler.step = _hooked_step

    def detach(self, pipe):
        if self._orig_step is not None:
            pipe.scheduler.step = self._orig_step
            self._orig_step = None

    def clear(self):
        """Free captured latents."""
        self.latents.clear()


# ============================================================================
# SelectiveStepAttention — routes attention by denoising step index
# ============================================================================

class SelectiveStepAttention:
    """Monkey-patches F.sdpa to route per-step: SDPA for skipped steps,
    quantized otherwise.

    Tracks the current denoising step by counting attention calls and dividing
    by calls_per_step (30 for CogVideoX-2b). When the current step index is
    in skip_steps, all attention calls in that step use SDPA.

    Key difference from layer_skip.py's SelectiveAttention:
      - Layer-skip:  call_count % calls_per_step  -> layer index within a step
      - Step-skip:   call_count // calls_per_step  -> step index across the run
    """

    def __init__(
        self,
        sage_fn: Callable,
        orig_sdpa: Callable,
        skip_steps: Set[int],
        calls_per_step: int = 30,
    ):
        self.sage_fn = sage_fn
        self.orig_sdpa = orig_sdpa
        self.skip_steps = skip_steps
        self.calls_per_step = calls_per_step
        self._call_count = 0

    def __call__(self, *args, **kwargs):
        step_idx = self._call_count // self.calls_per_step
        self._call_count += 1
        if step_idx in self.skip_steps:
            return self.orig_sdpa(*args, **kwargs)
        return self.sage_fn(*args, **kwargs)

    def reset(self):
        self._call_count = 0


# ============================================================================
# Phase 1: Per-Step Error Profiling
# ============================================================================

def run_profiling(
    formats: List[str],
    model: str,
    seed: int,
    num_steps: int,
    num_frames: Optional[int],
) -> Tuple[Dict, Dict[str, List[int]]]:
    """Phase 1: Profile per-step cumulative error to identify sensitive steps.

    Returns:
        (results_dict, worst_steps_by_format)
        where worst_steps_by_format maps format -> list of step indices
        sorted from most to least error-contributing (by marginal SNR drop).
    """
    print_header("Phase 1: Per-Step Error Profiling")
    print(f"  Model: {model}, Steps: {num_steps}")
    print(f"  Formats: {formats}")
    print(f"  Seed: {seed}")

    prompt = "A dog is running in the park."

    # Load pipeline once
    print("\n  Loading CogVideoX pipeline...")
    pipe, default_frames, _torch_dtype = _load_cogvideox_pipe(model)
    if num_frames is None:
        num_frames = default_frames
    print(f"  Pipeline loaded. Using {num_frames} frames, {num_steps} steps.")

    # --- Reference run with SDPA, capturing latent at every step ---
    print("\n  Running SDPA reference (capturing per-step latents)...")
    orig_sdpa = F.scaled_dot_product_attention

    ref_capture = PerStepLatentCapture()
    ref_capture.attach(pipe)
    F.scaled_dot_product_attention = orig_sdpa
    _run_pipe(pipe, prompt, num_steps, num_frames, seed)
    ref_capture.detach(pipe)

    ref_latents = ref_capture.latents
    print(f"  Captured {len(ref_latents)} reference latents "
          f"(shape: {ref_latents[0].shape})")

    if len(ref_latents) != num_steps:
        print(f"  WARNING: Expected {num_steps} latents, got {len(ref_latents)}")

    results = {}
    worst_steps_by_format: Dict[str, List[int]] = {}

    # --- Per-format profiling ---
    for fmt in formats:
        print_header(f"Phase 1 — Profiling: {fmt}")
        sage_fn = build_sage_fn(fmt)

        fmt_capture = PerStepLatentCapture()
        fmt_capture.attach(pipe)
        F.scaled_dot_product_attention = sage_fn
        _run_pipe(pipe, prompt, num_steps, num_frames, seed)
        fmt_capture.detach(pipe)
        F.scaled_dot_product_attention = orig_sdpa

        fmt_latents = fmt_capture.latents
        n = min(len(ref_latents), len(fmt_latents))

        if n != num_steps:
            print(f"  WARNING: Expected {num_steps} latents, got {len(fmt_latents)}")

        # --- Compute cumulative metrics at each step ---
        per_step_data = []
        prev_snr = None

        for t in range(n):
            m = compute_metrics(ref_latents[t], fmt_latents[t])
            delta_snr = None
            if prev_snr is not None:
                delta_snr = m.snr_db - prev_snr
            per_step_data.append({
                "step": t,
                **asdict(m),
                "delta_snr": delta_snr,
            })
            prev_snr = m.snr_db

        # --- Print per-step table ---
        print(f"\n  Per-step cumulative divergence ({n} steps):")
        print(f"  {'Step':>6s}  {'CosSim':>10s}  {'SNR(dB)':>9s}  "
              f"{'delta_SNR':>10s}  {'L1 Rel':>9s}  {'L2 Rel':>9s}")
        print(f"  {'-'*6}  {'-'*10}  {'-'*9}  "
              f"{'-'*10}  {'-'*9}  {'-'*9}")
        for d in per_step_data:
            delta_str = f"{d['delta_snr']:+.2f}" if d['delta_snr'] is not None else "—"
            print(f"  {d['step']:>6d}  {d['cos_sim']:>10.6f}  {d['snr_db']:>9.2f}  "
                  f"{delta_str:>10s}  {d['l1_rel']:>9.6f}  {d['l2_rel']:>9.6f}")

        # --- Rank steps by marginal SNR drop (most negative = worst) ---
        # Step 0 has no delta, so we use its absolute SNR inverted as a proxy:
        # lower absolute SNR at step 0 = worse initial quantization
        steps_with_delta = [
            (d["step"], d["delta_snr"])
            for d in per_step_data
            if d["delta_snr"] is not None
        ]
        # Sort by delta_snr ascending (most negative = biggest error amplification)
        steps_with_delta.sort(key=lambda x: x[1])
        worst_steps_ranked = [s for s, _ in steps_with_delta]

        # Also compute error growth rate by phase
        third = max(1, n // 3)
        early_deltas = [d["delta_snr"] for d in per_step_data[1:third+1]
                        if d["delta_snr"] is not None]
        mid_deltas = [d["delta_snr"] for d in per_step_data[third+1:2*third+1]
                      if d["delta_snr"] is not None]
        late_deltas = [d["delta_snr"] for d in per_step_data[2*third+1:]
                       if d["delta_snr"] is not None]

        def _safe_avg(lst):
            return sum(lst) / len(lst) if lst else 0.0

        error_growth = {
            "early_avg_delta": _safe_avg(early_deltas),
            "mid_avg_delta": _safe_avg(mid_deltas),
            "late_avg_delta": _safe_avg(late_deltas),
        }

        print(f"\n  Worst steps (by marginal SNR drop):")
        for rank, (step, delta) in enumerate(steps_with_delta[:5]):
            print(f"    #{rank+1}: step {step} (delta_SNR = {delta:+.2f} dB)")

        print(f"\n  Error growth by phase:")
        print(f"    Early (steps 1-{third}):   avg delta = {error_growth['early_avg_delta']:+.2f} dB/step")
        print(f"    Mid   (steps {third+1}-{2*third}): avg delta = {error_growth['mid_avg_delta']:+.2f} dB/step")
        print(f"    Late  (steps {2*third+1}-{n-1}):  avg delta = {error_growth['late_avg_delta']:+.2f} dB/step")

        results[fmt] = {
            "per_step": per_step_data,
            "worst_steps_ranked": worst_steps_ranked,
            "error_growth_rate": error_growth,
        }
        worst_steps_by_format[fmt] = worst_steps_ranked

        # Free format latents
        fmt_capture.clear()
        gc.collect()
        torch.cuda.empty_cache()

    # --- Cross-format consensus: which steps are consistently worst? ---
    print_header("Phase 1 — Cross-Format Consensus")
    step_scores: Dict[int, float] = {}
    for fmt, ranked in worst_steps_by_format.items():
        for rank, step in enumerate(ranked):
            # Higher score = worse (rank 0 = worst step gets highest score)
            score = len(ranked) - rank
            step_scores[step] = step_scores.get(step, 0) + score

    consensus_ranked = sorted(step_scores.items(), key=lambda x: -x[1])
    print(f"\n  Step sensitivity consensus (across {len(formats)} formats):")
    for step, score in consensus_ranked[:8]:
        fmts_in_top5 = [fmt for fmt, ranked in worst_steps_by_format.items()
                        if step in ranked[:5]]
        print(f"    step {step:>3d}: score={score:>5.0f}  "
              f"(top-5 in: {', '.join(fmts_in_top5)})")

    results["_consensus"] = {
        "step_scores": {str(s): sc for s, sc in consensus_ranked},
        "consensus_worst_ranked": [s for s, _ in consensus_ranked],
    }

    # Clean up
    ref_capture.clear()
    F.scaled_dot_product_attention = orig_sdpa
    del ref_latents, pipe
    gc.collect()
    torch.cuda.empty_cache()

    return results, worst_steps_by_format


# ============================================================================
# Phase 2: Step-Skip Validation
# ============================================================================

def _build_skip_configs(
    worst_steps: List[int],
    num_steps: int,
) -> Dict[str, Set[int]]:
    """Build skip configurations from profiled worst steps."""
    configs = {"baseline": set()}

    # Data-driven worst-N
    if len(worst_steps) >= 1:
        configs["skip_worst_1"] = {worst_steps[0]}
    if len(worst_steps) >= 3:
        configs["skip_worst_3"] = set(worst_steps[:3])
    if len(worst_steps) >= 5:
        configs["skip_worst_5"] = set(worst_steps[:5])

    return configs


def run_step_skip_validation(
    formats: List[str],
    model: str,
    seed: int,
    num_steps: int,
    num_frames: Optional[int],
    worst_steps_by_format: Dict[str, List[int]],
) -> Dict:
    """Phase 2: Validate step-skip configurations using profiled worst steps."""

    print_header("Phase 2: Step-Skip Validation")
    print(f"  Model: {model}, Steps: {num_steps}")
    print(f"  Formats: {formats}")
    print(f"  Seed: {seed}")
    for fmt, ws in worst_steps_by_format.items():
        print(f"  {fmt} worst steps: {ws[:5]}")

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

    # Determine calls_per_step empirically: count attention calls in one run
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

    # --- Per-format experiment ---
    for fmt in formats:
        print_header(f"Phase 2 — Step-Skip: {fmt}")
        sage_fn = build_sage_fn(fmt)

        # Build skip configs from this format's worst steps
        worst_steps = worst_steps_by_format.get(fmt, [])
        skip_configs = _build_skip_configs(worst_steps, num_steps)

        print(f"  Skip configurations:")
        for name, steps in skip_configs.items():
            if steps:
                pct = 100 * len(steps) / num_steps
                print(f"    {name}: steps {sorted(steps)} "
                      f"({len(steps)}/{num_steps} = {pct:.0f}% SDPA)")
            else:
                print(f"    {name}: none (pure quantized)")

        fmt_results = {}
        fmt_rows = []
        baseline_snr = None

        for config_name, skip_set in skip_configs.items():
            print(f"\n  Config: {config_name} "
                  f"({len(skip_set)}/{num_steps} steps skipped)...")

            selective = SelectiveStepAttention(
                sage_fn, orig_sdpa, skip_set,
                calls_per_step=calls_per_step,
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

            m = compute_metrics(ref_latent, fmt_latent)
            print(f"    Latent: SNR={m.snr_db:.2f}dB  CosSim={m.cos_sim:.8f}  "
                  f"L1Rel={m.l1_rel:.6f}")

            if config_name == "baseline":
                baseline_snr = m.snr_db

            fmt_results[config_name] = {
                "skipped": len(skip_set),
                "skip_steps": sorted(skip_set),
                "metrics": asdict(m),
            }
            fmt_rows.append((config_name, m))

            del fmt_latent
            gc.collect()
            torch.cuda.empty_cache()

        # --- Summary table for this format ---
        if fmt_rows:
            print(f"\n  Summary — {fmt}, {num_steps} steps")
            print(f"\n  {'Config':<16s} {'Skipped':>8s} {'Latent SNR':>11s} "
                  f"{'L1Rel':>9s} {'CosSim':>10s} {'delta_SNR':>10s}")
            print(f"  {'-'*16} {'-'*8} {'-'*11} {'-'*9} {'-'*10} {'-'*10}")
            for config_name, m in fmt_rows:
                skip_count = len(skip_configs[config_name])
                if baseline_snr is not None and config_name != "baseline":
                    d = m.snr_db - baseline_snr
                    delta = f"{d:+.2f}"
                else:
                    delta = "—"
                print(f"  {config_name:<16s} {skip_count:>4d}/{num_steps:<3d} "
                      f"{m.snr_db:>11.2f} {m.l1_rel:>9.6f} "
                      f"{m.cos_sim:>10.6f} {delta:>10s}")
            print()

        results[fmt] = fmt_results

    # --- Cross-format summary ---
    print_header("Cross-Format Summary — Step-Skip Results")
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
        for config_name in sorted(fmt_data.keys()):
            if config_name == "baseline":
                continue
            cfg = fmt_data[config_name]
            cfg_metrics = cfg.get("metrics", {})
            cfg_snr = cfg_metrics.get("snr_db")
            if cfg_snr is None:
                continue
            delta = cfg_snr - baseline_snr
            skip_count = cfg.get("skipped", 0)
            efficiency = delta / skip_count if skip_count > 0 else 0
            print(f"    {config_name}: latent delta_SNR = {delta:+.2f} dB "
                  f"({skip_count} steps skipped, "
                  f"{efficiency:+.2f} dB/step)")

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
        description="Step-Skip Experiment: SDPA Fallback for Sensitive Denoising Steps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--phases", nargs="+", type=int, default=[1, 2],
        choices=[1, 2],
        help="Which phases to run (default: both)",
    )
    parser.add_argument(
        "--formats", nargs="+", default=ALL_FORMATS,
        choices=ALL_FORMATS,
        help=f"Quant formats to test (default: all {len(ALL_FORMATS)})",
    )
    parser.add_argument("--model", default="cogvideox-2b",
                        choices=["cogvideox-2b", "cogvideox1.5-5b"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-steps", type=int, default=20,
                        help="Denoising steps (default: 20)")
    parser.add_argument("--num-frames", type=int, default=1,
                        help="Frames to generate (default: 1)")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Save results to JSON file")

    args = parser.parse_args()

    print_header("Step-Skip Experiment")
    print(f"  Phases:  {args.phases}")
    print(f"  Formats: {args.formats}")
    print(f"  Model:   {args.model}")
    print(f"  Steps:   {args.num_steps}")
    print(f"  Frames:  {args.num_frames}")
    print(f"  Seed:    {args.seed}")

    start_time = time.time()
    all_results = {}

    # --- Phase 1: Per-step error profiling ---
    worst_steps_by_format = {}
    if 1 in args.phases:
        profiling_results, worst_steps_by_format = run_profiling(
            formats=args.formats,
            model=args.model,
            seed=args.seed,
            num_steps=args.num_steps,
            num_frames=args.num_frames,
        )
        all_results["profiling"] = profiling_results

    # --- Phase 2: Step-skip validation ---
    if 2 in args.phases:
        if not worst_steps_by_format:
            # Phase 1 was skipped — cannot build data-driven skip configs
            print("\n  WARNING: Phase 1 was not run. Cannot determine worst steps.")
            print("  Please run with --phases 1 2, or --phases 1 first.")
            return

        step_skip_results = run_step_skip_validation(
            formats=args.formats,
            model=args.model,
            seed=args.seed,
            num_steps=args.num_steps,
            num_frames=args.num_frames,
            worst_steps_by_format=worst_steps_by_format,
        )
        all_results["step_skip"] = step_skip_results

    elapsed = time.time() - start_time

    print_header("Done")
    print(f"  Total time: {elapsed:.1f}s")

    if args.output_json:
        all_results["_metadata"] = {
            "phases": args.phases,
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
        print(f"  Results saved to: {args.output_json}")


if __name__ == "__main__":
    main()
