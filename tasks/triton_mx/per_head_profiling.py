#!/usr/bin/env python3
"""
Per-Head Format Selection Profiler — CogVideoX-2b

Motivation: Sparsity profiling (sparsity_profiling.py) showed that tile-level
FP4/FP8 mixing is NOT viable for CogVideoX-2b (tiles_for_90pct=64.9 > STOP
threshold, best proxy ρ=0.225). However, per-head variation is very large:
  - Head 16: 21.1 tiles for 90% (sharp → good FP4 candidate)
  - Head 28: 95.7 tiles for 90% (diffuse → needs FP8)
  - Std across heads: 16.4 tiles

This script profiles per-head quantization error to determine which heads can
tolerate FP4 and which need FP8, enabling head-level mixed-precision routing.

Two phases:
  Phase 1: Per-head error profiling
    - Run full pipeline with SDPA reference → capture reference latent
    - For each head h (0..29): run with FP4 on head h only, SDPA on all others
    - Measure per-head marginal error contribution to final latent
    - Rank heads from worst (highest error) to best (lowest error)

  Phase 2: Head allocation sweep
    - Given the ranking, sweep allocations: top-K worst heads → FP8, rest → FP4
    - Measure E2E latent quality for K = 0, 5, 10, 15, 20, 25, 30
    - Compare against uniform mxfp4 baseline and format escalation results
    - Can compose with step-level escalation for even finer control

Architecture:
  CogVideoX-2b: 30 transformer blocks, each makes one SDPA call per step.
  Each call: (B=2, H=30, N=17776, D=64). All 30 attention heads in H dimension.
  call_count → step_idx = call_count // 30, layer_idx = call_count % 30

Usage:
  # Phase 1 only (profiling) — ~30 min
  CUDA_VISIBLE_DEVICES=6 python per_head_profiling.py --phase 1

  # Phase 2 only (head allocation sweep, needs Phase 1 results) — ~15 min
  CUDA_VISIBLE_DEVICES=6 python per_head_profiling.py --phase 2

  # Both phases
  CUDA_VISIBLE_DEVICES=6 python per_head_profiling.py

  # Quick smoke test
  CUDA_VISIBLE_DEVICES=6 python per_head_profiling.py --num-steps 5 --phase 1
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import gc
import json
import math
import sys
import time
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F

# Add standalone + tasks/triton_mx to path
_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_standalone_path = os.path.join(_repo_root, "standalone")
if _standalone_path not in sys.path:
    sys.path.insert(0, _standalone_path)

# Reuse utilities from ablation.py
from ablation import (
    AccuracyMetrics,
    LatentCapture,
    compute_metrics,
    build_sage_fn,
    print_header,
    _load_cogvideox_pipe,
    _run_pipe,
)


# ============================================================================
# HeadSplitAttention — per-head format routing within one SDPA call
# ============================================================================

class HeadSplitAttention:
    """Routes individual heads to different attention backends.

    Given a mapping head_idx -> backend_fn, splits the Q/K/V along H dimension,
    runs each group through its assigned backend, and concatenates results.

    For efficiency, groups consecutive heads that share the same backend into
    a single call (avoids 30 separate kernel launches).
    """

    def __init__(
        self,
        head_assignments: Dict[int, str],  # head_idx -> "fp4" | "fp8" | "sdpa"
        fp4_fn: Callable,
        fp8_fn: Callable,
        sdpa_fn: Callable,
        num_heads: int = 30,
    ):
        self.num_heads = num_heads
        self.fp4_fn = fp4_fn
        self.fp8_fn = fp8_fn
        self.sdpa_fn = sdpa_fn

        # Build groups of consecutive heads with the same assignment
        self.groups = self._build_groups(head_assignments)

    def _build_groups(self, assignments: Dict[int, str]) -> List[Tuple[str, List[int]]]:
        """Group consecutive heads by assignment for efficient batched calls."""
        groups = []
        current_fmt = None
        current_heads = []

        for h in range(self.num_heads):
            fmt = assignments.get(h, "fp4")
            if fmt == current_fmt:
                current_heads.append(h)
            else:
                if current_heads:
                    groups.append((current_fmt, current_heads))
                current_fmt = fmt
                current_heads = [h]
        if current_heads:
            groups.append((current_fmt, current_heads))

        return groups

    def _get_fn(self, fmt: str) -> Callable:
        if fmt == "fp4":
            return self.fp4_fn
        elif fmt == "fp8":
            return self.fp8_fn
        else:
            return self.sdpa_fn

    def __call__(self, *args, **kwargs):
        q = args[0] if len(args) > 0 else kwargs.get('query')
        k = args[1] if len(args) > 1 else kwargs.get('key')
        v = args[2] if len(args) > 2 else kwargs.get('value')
        # Pass through remaining kwargs (is_causal, scale, etc.)

        B, H, N, D = q.shape
        outputs = []

        for fmt, heads in self.groups:
            fn = self._get_fn(fmt)
            head_slice = slice(heads[0], heads[-1] + 1)
            q_sub = q[:, head_slice, :, :]
            k_sub = k[:, head_slice, :, :]
            v_sub = v[:, head_slice, :, :]

            # Call with sub-tensor
            if len(args) > 2:
                out = fn(q_sub, k_sub, v_sub, *args[3:], **kwargs)
            else:
                out = fn(q_sub, k_sub, v_sub, **kwargs)
            outputs.append(out)

        return torch.cat(outputs, dim=1)


# ============================================================================
# StepAwareHeadSplitAttention — combines step-level + head-level routing
# ============================================================================

class StepAwareHeadSplitAttention:
    """Combines step-level format escalation with head-level routing.

    For "critical" steps: use SDPA for all heads.
    For "moderate" steps: use FP8 for all heads.
    For "easy" steps: use per-head routing (some FP4, some FP8).
    """

    def __init__(
        self,
        head_split_fn: Callable,  # HeadSplitAttention instance
        sdpa_fn: Callable,
        fp8_fn: Callable,
        critical_steps: Set[int],
        moderate_steps: Set[int],
        calls_per_step: int = 30,
    ):
        self.head_split_fn = head_split_fn
        self.sdpa_fn = sdpa_fn
        self.fp8_fn = fp8_fn
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
        return self.head_split_fn(*args, **kwargs)

    def reset(self):
        self._call_count = 0


# ============================================================================
# Phase 1: Per-Head Error Profiling
# ============================================================================

def run_phase1(
    pipe,
    prompt: str,
    num_steps: int,
    num_frames: int,
    seed: int,
    ref_latent: torch.Tensor,
    orig_sdpa: Callable,
    fp4_fn: Callable,
    calls_per_step: int,
    quant_format: str = "mxfp4",
) -> Dict:
    """Profile per-head FP4 error contribution.

    For each head h: run full pipeline with FP4 on head h only + SDPA on rest.
    Measure how much that single head's FP4 error contributes to latent divergence.
    """
    print_header(f"Phase 1: Per-Head {quant_format.upper()} Error Profiling")
    print(f"  Strategy: For each head h, run FP4 on head h + SDPA on all others")
    print(f"  Num heads: 30, Steps: {num_steps}")
    print(f"  Baseline: reference latent from SDPA")

    num_heads = 30
    per_head_results = {}
    t0 = time.time()

    for h in range(num_heads):
        print(f"\n  Profiling head {h}/29...", end=" ", flush=True)

        # Assignment: head h → FP4, all others → SDPA
        assignments = {i: "sdpa" for i in range(num_heads)}
        assignments[h] = "fp4"

        head_attn = HeadSplitAttention(
            head_assignments=assignments,
            fp4_fn=fp4_fn,
            fp8_fn=fp4_fn,  # not used in Phase 1
            sdpa_fn=orig_sdpa,
            num_heads=num_heads,
        )

        capture = LatentCapture()
        capture.attach(pipe, num_steps)
        F.scaled_dot_product_attention = head_attn
        with torch.no_grad():
            _run_pipe(pipe, prompt, num_steps, num_frames, seed)
        capture.detach(pipe)
        F.scaled_dot_product_attention = orig_sdpa

        latent = capture.final_latent
        if latent is None:
            print("FAILED")
            per_head_results[h] = {"error": "no latent"}
            continue

        m = compute_metrics(ref_latent, latent)
        print(f"SNR={m.snr_db:.2f}dB  CosSim={m.cos_sim:.6f}")

        per_head_results[h] = {
            "head": h,
            "snr_db": m.snr_db,
            "cos_sim": m.cos_sim,
            "l1_rel": m.l1_rel,
            "l2_rel": m.l2_rel,
            "metrics": asdict(m),
        }

        del latent
        gc.collect()
        torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"\n  Phase 1 complete: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Rank heads by SNR (lower SNR = more error = worse head for FP4)
    valid_heads = [(h, r) for h, r in per_head_results.items()
                   if isinstance(r, dict) and "snr_db" in r]
    ranked = sorted(valid_heads, key=lambda x: x[1]["snr_db"])

    print(f"\n  Head Ranking (worst → best for {quant_format.upper()}):")
    print(f"  {'Head':>6s}  {'SNR (dB)':>10s}  {'CosSim':>10s}  {'L1Rel':>10s}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}")
    for h, r in ranked:
        print(f"  {h:>6d}  {r['snr_db']:>10.2f}  {r['cos_sim']:>10.6f}  "
              f"{r['l1_rel']:>10.6f}")

    worst_to_best = [h for h, _ in ranked]
    print(f"\n  Worst heads (FP4): {worst_to_best[:10]}")
    print(f"  Best heads (FP4):  {worst_to_best[-10:]}")

    return {
        "per_head": per_head_results,
        "worst_to_best": worst_to_best,
        "elapsed_s": elapsed,
    }


# ============================================================================
# Phase 2: Head Allocation Sweep
# ============================================================================

def run_phase2(
    pipe,
    prompt: str,
    num_steps: int,
    num_frames: int,
    seed: int,
    ref_latent: torch.Tensor,
    orig_sdpa: Callable,
    fp4_fn: Callable,
    fp8_fn: Callable,
    calls_per_step: int,
    worst_to_best: List[int],
    quant_format: str = "mxfp4",
) -> Dict:
    """Sweep head allocations: top-K worst heads → FP8, rest → FP4.

    Tests K = 0, 5, 10, 15, 20, 25, 30 (where K=0 = all FP4, K=30 = all FP8).
    Also tests "all FP4" and "all FP8" baselines for comparison.
    """
    print_header(f"Phase 2: Head Allocation Sweep ({quant_format.upper()})")

    num_heads = 30
    # Sweep configurations: (name, n_fp8_heads)
    sweep_configs = [
        ("all_fp4", 0),
        ("fp8_worst_5", 5),
        ("fp8_worst_10", 10),
        ("fp8_worst_15", 15),
        ("fp8_worst_20", 20),
        ("fp8_worst_25", 25),
        ("all_fp8", 30),
    ]

    results = {}
    t0 = time.time()

    for config_name, n_fp8 in sweep_configs:
        print(f"\n  Config: {config_name} ({n_fp8} FP8 heads, "
              f"{num_heads - n_fp8} FP4 heads)...", end=" ", flush=True)

        if n_fp8 == 0:
            # All FP4 — use fp4_fn for everything
            F.scaled_dot_product_attention = fp4_fn
        elif n_fp8 == num_heads:
            # All FP8
            F.scaled_dot_product_attention = fp8_fn
        else:
            # Mixed: worst n_fp8 heads → FP8, rest → FP4
            fp8_heads = set(worst_to_best[:n_fp8])
            assignments = {}
            for h in range(num_heads):
                assignments[h] = "fp8" if h in fp8_heads else "fp4"

            head_attn = HeadSplitAttention(
                head_assignments=assignments,
                fp4_fn=fp4_fn,
                fp8_fn=fp8_fn,
                sdpa_fn=orig_sdpa,
                num_heads=num_heads,
            )
            F.scaled_dot_product_attention = head_attn

        capture = LatentCapture()
        capture.attach(pipe, num_steps)
        with torch.no_grad():
            _run_pipe(pipe, prompt, num_steps, num_frames, seed)
        capture.detach(pipe)
        F.scaled_dot_product_attention = orig_sdpa

        latent = capture.final_latent
        if latent is None:
            print("FAILED")
            results[config_name] = {"error": "no latent"}
            continue

        m = compute_metrics(ref_latent, latent)
        print(f"SNR={m.snr_db:.2f}dB  CosSim={m.cos_sim:.6f}")

        fp8_head_list = worst_to_best[:n_fp8] if n_fp8 < num_heads else list(range(num_heads))
        results[config_name] = {
            "n_fp8": n_fp8,
            "n_fp4": num_heads - n_fp8,
            "fp8_heads": sorted(fp8_head_list),
            "metrics": asdict(m),
        }

        del latent
        gc.collect()
        torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"\n  Phase 2 complete: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Summary table
    baseline_snr = None
    print(f"\n  Head Allocation Summary — {quant_format.upper()}, {num_steps} steps")
    print(f"\n  {'Config':<18s} {'FP8':>5s} {'FP4':>5s} "
          f"{'Latent SNR':>11s} {'CosSim':>10s} {'L1Rel':>9s} "
          f"{'delta_SNR':>10s}")
    print(f"  {'-'*18} {'-'*5} {'-'*5} "
          f"{'-'*11} {'-'*10} {'-'*9} {'-'*10}")

    for config_name, n_fp8 in sweep_configs:
        r = results.get(config_name, {})
        met = r.get("metrics", {})
        snr = met.get("snr_db")
        cos = met.get("cos_sim")
        l1r = met.get("l1_rel")
        if snr is None:
            continue
        if config_name == "all_fp4":
            baseline_snr = snr
        if baseline_snr is not None and config_name != "all_fp4":
            delta = f"{snr - baseline_snr:+.2f}"
        else:
            delta = "--"
        print(f"  {config_name:<18s} {n_fp8:>5d} {num_heads - n_fp8:>5d} "
              f"{snr:>11.2f} {cos:>10.6f} {l1r:>9.6f} {delta:>10s}")

    return {
        "sweep": results,
        "elapsed_s": elapsed,
    }


# ============================================================================
# Phase 3: Composition with Step-Level Escalation (bonus)
# ============================================================================

def run_phase3(
    pipe,
    prompt: str,
    num_steps: int,
    num_frames: int,
    seed: int,
    ref_latent: torch.Tensor,
    orig_sdpa: Callable,
    fp4_fn: Callable,
    fp8_fn: Callable,
    calls_per_step: int,
    worst_to_best: List[int],
    quant_format: str = "mxfp4",
) -> Dict:
    """Test composition: step-level escalation + per-head routing on easy steps.

    On critical steps → all SDPA
    On moderate steps → all FP8
    On easy steps → per-head split (worst heads FP8, rest FP4)
    """
    print_header(f"Phase 3: Composed Step + Head Routing ({quant_format.upper()})")

    num_heads = 30

    # Step rankings from 50-step profiling (consensus worst steps)
    worst_steps_50 = [6, 2, 1, 4, 5, 8, 7, 9, 3, 10, 11, 14, 15, 13, 12,
                      16, 17, 18, 19, 49]
    # For 20-step, use the known consensus
    worst_steps_20 = [5, 4, 1, 6, 3, 2, 0, 7, 8, 9, 10, 11, 12, 13, 14,
                      15, 16, 17, 18, 19]

    worst_steps = worst_steps_50 if num_steps == 50 else worst_steps_20

    # Configurations: (name, n_critical, n_moderate, n_fp8_heads)
    configs = [
        # Pure head-split (no step escalation)
        ("head_only_10", 0, 0, 10),
        ("head_only_15", 0, 0, 15),
        # Step escalation + head split
        ("step3_5_head10", 3, 5, 10),
        ("step3_5_head15", 3, 5, 15),
        # Reference: escalate_3_5 (no head split)
        ("escalate_3_5", 3, 5, 0),
        # Full composition
        ("step5_8_head10", 5, 8, 10),
        ("step5_8_head15", 5, 8, 15),
    ]

    results = {}
    t0 = time.time()

    for config_name, n_crit, n_mod, n_fp8_heads in configs:
        critical = set(worst_steps[:n_crit]) if n_crit > 0 else set()
        moderate = set(worst_steps[n_crit:n_crit + n_mod]) if n_mod > 0 else set()
        n_fp4 = num_steps - n_crit - n_mod

        print(f"\n  Config: {config_name} "
              f"({n_crit} SDPA-steps + {n_mod} FP8-steps + {n_fp4} easy-steps, "
              f"{n_fp8_heads} FP8-heads on easy steps)...", end=" ", flush=True)

        if n_fp8_heads == 0 and n_crit == 0 and n_mod == 0:
            # All FP4
            F.scaled_dot_product_attention = fp4_fn
        elif n_fp8_heads == 0:
            # Step escalation only (no head split)
            from format_escalation import EscalationAttention
            esc = EscalationAttention(
                fp4_fn, fp8_fn, orig_sdpa,
                critical, moderate,
                calls_per_step=calls_per_step,
            )
            F.scaled_dot_product_attention = esc
        else:
            # Per-head split on easy steps
            fp8_heads = set(worst_to_best[:n_fp8_heads])
            assignments = {}
            for h in range(num_heads):
                assignments[h] = "fp8" if h in fp8_heads else "fp4"

            head_split = HeadSplitAttention(
                head_assignments=assignments,
                fp4_fn=fp4_fn,
                fp8_fn=fp8_fn,
                sdpa_fn=orig_sdpa,
                num_heads=num_heads,
            )

            step_head_attn = StepAwareHeadSplitAttention(
                head_split_fn=head_split,
                sdpa_fn=orig_sdpa,
                fp8_fn=fp8_fn,
                critical_steps=critical,
                moderate_steps=moderate,
                calls_per_step=calls_per_step,
            )
            F.scaled_dot_product_attention = step_head_attn

        capture = LatentCapture()
        capture.attach(pipe, num_steps)
        with torch.no_grad():
            _run_pipe(pipe, prompt, num_steps, num_frames, seed)
        capture.detach(pipe)
        F.scaled_dot_product_attention = orig_sdpa

        latent = capture.final_latent
        if latent is None:
            print("FAILED")
            results[config_name] = {"error": "no latent"}
            continue

        m = compute_metrics(ref_latent, latent)
        print(f"SNR={m.snr_db:.2f}dB  CosSim={m.cos_sim:.6f}")

        results[config_name] = {
            "critical_steps": n_crit,
            "moderate_steps": n_mod,
            "fp8_heads": n_fp8_heads,
            "fp4_heads": num_heads - n_fp8_heads,
            "easy_steps": n_fp4,
            "metrics": asdict(m),
        }

        del latent
        gc.collect()
        torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"\n  Phase 3 complete: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Summary table
    print(f"\n  Composed Routing Summary — {quant_format.upper()}, {num_steps} steps")
    print(f"\n  {'Config':<22s} {'SDPA-S':>6s} {'FP8-S':>6s} {'Easy-S':>6s} "
          f"{'FP8-H':>6s} {'FP4-H':>6s} {'SNR':>8s} {'CosSim':>10s}")
    print(f"  {'-'*22} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*8} {'-'*10}")

    for config_name, n_crit, n_mod, n_fp8_heads in configs:
        r = results.get(config_name, {})
        met = r.get("metrics", {})
        snr = met.get("snr_db")
        cos = met.get("cos_sim")
        if snr is None:
            continue
        n_fp4 = num_steps - n_crit - n_mod
        print(f"  {config_name:<22s} {n_crit:>6d} {n_mod:>6d} {n_fp4:>6d} "
              f"{n_fp8_heads:>6d} {num_heads - n_fp8_heads:>6d} "
              f"{snr:>8.2f} {cos:>10.6f}")

    return {
        "composed": results,
        "elapsed_s": elapsed,
    }


# ============================================================================
# Main Experiment
# ============================================================================

def run_experiment(
    phases: List[int],
    model: str,
    seed: int,
    num_steps: int,
    num_frames: Optional[int],
    quant_format: str = "mxfp4",
    output_json: Optional[str] = None,
    phase1_json: Optional[str] = None,
):
    """Run the per-head format selection experiment."""
    print_header("Per-Head Format Selection Profiler")
    print(f"  Model: {model}, Steps: {num_steps}")
    print(f"  Quant format: {quant_format}")
    print(f"  Phases: {phases}")
    print(f"  Seed: {seed}")

    prompt = "A dog is running in the park."

    # Load pipeline
    print("\n  Loading CogVideoX pipeline...")
    pipe, default_frames, _torch_dtype = _load_cogvideox_pipe(model)
    if num_frames is None:
        num_frames = default_frames
    print(f"  Loaded. num_frames={num_frames}, num_steps={num_steps}")

    # Save original SDPA
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
        return {"error": "Failed to capture reference latent"}
    print(f"  Reference latent shape: {ref_latent.shape}")

    # --- Detect calls_per_step ---
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

    # Build quant functions
    fp4_fn = build_sage_fn(quant_format)
    fp8_fn = build_sage_fn("mxfp8_s1")

    all_results = {
        "_metadata": {
            "model": model,
            "seed": seed,
            "num_steps": num_steps,
            "num_frames": num_frames,
            "quant_format": quant_format,
            "calls_per_step": calls_per_step,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "gpu": (torch.cuda.get_device_name(0)
                    if torch.cuda.is_available() else "N/A"),
        }
    }

    worst_to_best = None

    # --- Phase 1: Per-head error profiling ---
    if 1 in phases:
        phase1_results = run_phase1(
            pipe, prompt, num_steps, num_frames, seed,
            ref_latent, orig_sdpa, fp4_fn, calls_per_step, quant_format,
        )
        all_results["phase1"] = phase1_results
        worst_to_best = phase1_results["worst_to_best"]
    elif phase1_json:
        # Load Phase 1 results from file
        print(f"\n  Loading Phase 1 results from: {phase1_json}")
        with open(phase1_json) as f:
            loaded = json.load(f)
        if "phase1" in loaded:
            worst_to_best = loaded["phase1"]["worst_to_best"]
            print(f"  Loaded head ranking: {worst_to_best[:10]}... (worst first)")
        else:
            print("  ERROR: No phase1 data in file")

    # --- Phase 2: Head allocation sweep ---
    if 2 in phases:
        if worst_to_best is None:
            print("\n  ERROR: Phase 2 requires Phase 1 results (--phase 1 or --phase1-json)")
        else:
            phase2_results = run_phase2(
                pipe, prompt, num_steps, num_frames, seed,
                ref_latent, orig_sdpa, fp4_fn, fp8_fn, calls_per_step,
                worst_to_best, quant_format,
            )
            all_results["phase2"] = phase2_results

    # --- Phase 3: Composition with step escalation ---
    if 3 in phases:
        if worst_to_best is None:
            print("\n  ERROR: Phase 3 requires Phase 1 results")
        else:
            phase3_results = run_phase3(
                pipe, prompt, num_steps, num_frames, seed,
                ref_latent, orig_sdpa, fp4_fn, fp8_fn, calls_per_step,
                worst_to_best, quant_format,
            )
            all_results["phase3"] = phase3_results

    # Save results
    if output_json:
        with open(output_json, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n  Results saved to: {output_json}")

    # Clean up
    F.scaled_dot_product_attention = orig_sdpa
    del ref_latent, pipe
    gc.collect()
    torch.cuda.empty_cache()

    return all_results


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Per-Head Format Selection Profiler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--phase", type=int, nargs="+", default=[1, 2, 3],
        choices=[1, 2, 3],
        help="Phases to run (default: 1 2 3). "
             "1=profiling, 2=allocation sweep, 3=step+head composition",
    )
    parser.add_argument("--model", default="cogvideox-2b",
                        choices=["cogvideox-2b", "cogvideox1.5-5b"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-steps", type=int, default=20,
                        help="Denoising steps (default: 20)")
    parser.add_argument("--num-frames", type=int, default=1,
                        help="Frames to generate (default: 1 for speed)")
    parser.add_argument("--quant-format", type=str, default="mxfp4",
                        choices=["mxfp4", "mxfp4_s1", "nvfp4"],
                        help="FP4 format to test (default: mxfp4)")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Save results to JSON file")
    parser.add_argument("--phase1-json", type=str, default=None,
                        help="Load Phase 1 results from JSON file (skip Phase 1)")

    args = parser.parse_args()

    start_time = time.time()

    run_experiment(
        phases=args.phase,
        model=args.model,
        seed=args.seed,
        num_steps=args.num_steps,
        num_frames=args.num_frames,
        quant_format=args.quant_format,
        output_json=args.output_json,
        phase1_json=args.phase1_json,
    )

    elapsed = time.time() - start_time
    print_header("Done")
    print(f"  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
