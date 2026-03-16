#!/usr/bin/env python3
"""
MX Format Accuracy Ablation — Phase 1

Systematically measures accuracy across quantization formats (NVFP4, MXFP4,
MXFP4_S1, MXFP8_S1) vs SDPA reference to identify degradation sources.

Three parts:
  A) Synthetic accuracy comparison on CogVideoX-realistic shapes
  B) Per-layer accuracy in CogVideoX inference
  C) Final-step latent divergence (closest proxy for E2E quality)

Usage:
  # Part A only (fast, no model loading)
  python ablation.py --parts A

  # Part B + C (needs CogVideoX model + GPU)
  python ablation.py --parts B C --num-steps 5 --num-frames 1

  # Full run with JSON output
  python ablation.py --parts A B C --output-json results.json
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import functools
import gc
import json
import math
import sys
import time
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# Add standalone to path for sageattention3_standalone imports
_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_standalone_path = os.path.join(_repo_root, "standalone")
if _standalone_path not in sys.path:
    sys.path.insert(0, _standalone_path)


# ============================================================================
# Shared Utilities
# ============================================================================

@dataclass
class AccuracyMetrics:
    """Accuracy metrics comparing an output tensor to a reference."""
    cos_sim: float
    l2_rel: float    # ||out - ref||_2 / ||ref||_2
    max_abs: float   # max |out - ref|
    mean_abs: float  # mean |out - ref|
    snr_db: float    # 10 * log10(||ref||² / ||err||²)  — higher = better
    rmse: float      # sqrt(mean((out - ref)²))
    l1_rel: float    # Σ|out - ref| / Σ|ref|  (paper's exact formula)


def compute_metrics(ref: torch.Tensor, out: torch.Tensor) -> AccuracyMetrics:
    """Compute accuracy metrics between reference and output tensors."""
    ref_flat = ref.flatten().float()
    out_flat = out.flatten().float()

    diff = out_flat - ref_flat

    # Cosine similarity
    cos = F.cosine_similarity(ref_flat.unsqueeze(0), out_flat.unsqueeze(0)).item()

    # Relative L2 norm
    ref_norm = ref_flat.norm().item()
    l2_rel = diff.norm().item() / max(ref_norm, 1e-12)

    # Absolute error stats
    abs_diff = diff.abs()
    max_abs = abs_diff.max().item()
    mean_abs = abs_diff.mean().item()

    # SNR in dB
    signal_power = (ref_flat * ref_flat).sum().item()
    noise_power = (diff * diff).sum().item()
    snr_db = 10.0 * math.log10(signal_power / max(noise_power, 1e-30))

    # RMSE
    rmse = math.sqrt(noise_power / ref_flat.numel())

    # Relative L1 (paper formula: Σ|O - O'| / Σ|O|)
    l1_rel = abs_diff.sum().item() / max(ref_flat.abs().sum().item(), 1e-12)

    return AccuracyMetrics(
        cos_sim=cos, l2_rel=l2_rel, max_abs=max_abs, mean_abs=mean_abs,
        snr_db=snr_db, rmse=rmse, l1_rel=l1_rel,
    )


def build_sage_fn(fmt: str) -> Callable:
    """Create an attention function for a specific quant format.

    Uses the quant_format= kwarg directly — no env var hacks needed.
    The module is imported once; format switching is per-call via the kwarg.
    """
    from sageattention3_standalone import scaled_dot_product_attention as sage_sdpa
    return functools.partial(sage_sdpa, quant_format=fmt)


def print_header(title: str):
    """Print a formatted section header."""
    width = 72
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def print_metrics_table(rows: List[Tuple[str, AccuracyMetrics]], label_header: str = "Format"):
    """Print a formatted table of accuracy metrics."""
    print(f"\n  {label_header:<20s} {'CosSim':>10s} {'SNR(dB)':>9s} {'L1 Rel':>9s} "
          f"{'RMSE':>10s} {'L2 Rel':>9s} {'MaxAbs':>10s}")
    print(f"  {'-'*20} {'-'*10} {'-'*9} {'-'*9} {'-'*10} {'-'*9} {'-'*10}")
    for label, m in rows:
        print(f"  {label:<20s} {m.cos_sim:>10.6f} {m.snr_db:>9.2f} {m.l1_rel:>9.6f} "
              f"{m.rmse:>10.6f} {m.l2_rel:>9.6f} {m.max_abs:>10.6f}")
    print()


def _average_metrics(metrics_list: List[AccuracyMetrics]) -> AccuracyMetrics:
    """Average a list of AccuracyMetrics into a single summary."""
    n = len(metrics_list)
    return AccuracyMetrics(
        cos_sim=sum(m.cos_sim for m in metrics_list) / n,
        l2_rel=sum(m.l2_rel for m in metrics_list) / n,
        max_abs=sum(m.max_abs for m in metrics_list) / n,
        mean_abs=sum(m.mean_abs for m in metrics_list) / n,
        snr_db=sum(m.snr_db for m in metrics_list) / n,
        rmse=sum(m.rmse for m in metrics_list) / n,
        l1_rel=sum(m.l1_rel for m in metrics_list) / n,
    )


# ============================================================================
# Part A: Synthetic Accuracy Comparison
# ============================================================================

# CogVideoX-2b realistic shapes: (B, H, N, D)
COGVIDEOX_SHAPES = {
    "text_cross":  (1, 24, 226, 64),
    "video_short": (1, 24, 4561, 64),
    "video_full":  (1, 24, 17281, 64),
}


def run_part_a(formats: List[str], seed: int, is_causal: bool = False) -> Dict:
    """Part A: Synthetic accuracy comparison on CogVideoX-realistic shapes."""
    print_header("Part A: Synthetic Accuracy — All Formats vs SDPA")
    print(f"  Formats: {formats}")
    print(f"  Seed: {seed}, Causal: {is_causal}")
    print(f"  Shapes: {list(COGVIDEOX_SHAPES.keys())}")

    # Pre-build sage functions (triggers import once)
    sage_fns = {fmt: build_sage_fn(fmt) for fmt in formats}

    results = {}

    for shape_name, (B, H, N, D) in COGVIDEOX_SHAPES.items():
        print(f"\n  --- Shape: {shape_name}  (B={B}, H={H}, N={N}, D={D}) ---")

        try:
            # Generate random BF16 inputs (seeded for reproducibility)
            gen = torch.Generator(device="cuda").manual_seed(seed)
            q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16, generator=gen)
            k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16, generator=gen)
            v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16, generator=gen)

            # Reference: PyTorch SDPA
            with torch.no_grad():
                ref = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

            # Test each format
            shape_results = {}
            rows = []
            for fmt in formats:
                try:
                    with torch.no_grad():
                        out = sage_fns[fmt](q, k, v, is_causal=is_causal)
                    m = compute_metrics(ref, out)
                    shape_results[fmt] = asdict(m)
                    rows.append((fmt, m))
                    del out
                except torch.cuda.OutOfMemoryError:
                    print(f"    {fmt}: OOM — skipped")
                    shape_results[fmt] = "OOM"
                except Exception as e:
                    print(f"    {fmt}: ERROR — {e}")
                    shape_results[fmt] = f"ERROR: {e}"

            if rows:
                print_metrics_table(rows)

            # Per-head analysis — compute metrics for each head independently
            if H <= 32:
                head_rows_by_fmt: Dict[str, List[AccuracyMetrics]] = {}
                for fmt in formats:
                    if shape_results.get(fmt) in ("OOM",) or isinstance(shape_results.get(fmt), str):
                        continue
                    try:
                        with torch.no_grad():
                            out = sage_fns[fmt](q, k, v, is_causal=is_causal)
                        head_metrics = []
                        for h in range(H):
                            hm = compute_metrics(ref[:, h], out[:, h])
                            head_metrics.append(hm)
                        head_rows_by_fmt[fmt] = head_metrics
                        del out
                    except Exception:
                        pass

                # Print per-head summary: worst/best heads per format
                if head_rows_by_fmt:
                    print(f"\n  Per-head analysis ({H} heads):")
                    for fmt in formats:
                        if fmt not in head_rows_by_fmt:
                            continue
                        hm_list = head_rows_by_fmt[fmt]
                        sorted_heads = sorted(enumerate(hm_list), key=lambda x: x[1].snr_db)
                        worst3 = sorted_heads[:3]
                        best3 = sorted_heads[-3:][::-1]
                        spread = sorted_heads[-1][1].snr_db - sorted_heads[0][1].snr_db
                        print(f"    {fmt}:  worst heads: {[(h, f'{m.snr_db:.1f}dB') for h,m in worst3]}  "
                              f"best: {[(h, f'{m.snr_db:.1f}dB') for h,m in best3]}  "
                              f"spread: {spread:.1f}dB")

                    # Store per-head data in results
                    shape_results["per_head"] = {
                        fmt: {f"head_{h:02d}": asdict(hm) for h, hm in enumerate(hm_list)}
                        for fmt, hm_list in head_rows_by_fmt.items()
                    }

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

    return {"part_a": results}


# ============================================================================
# Part B: Per-Layer CogVideoX Accuracy
# ============================================================================

class AttentionCapture:
    """Context manager that monkey-patches F.sdpa, records outputs as CPU tensors.

    Captures the attention output from each call for later comparison.
    Outputs are stored as float32 on CPU to save GPU memory.
    """

    def __init__(self, attn_fn: Callable, max_calls: Optional[int] = None):
        self.attn_fn = attn_fn
        self.outputs: List[torch.Tensor] = []  # CPU float32 copies
        self.shapes: List[Tuple] = []           # q.shape for each call
        self.max_calls = max_calls
        self._orig_sdpa = None

    def __enter__(self):
        self._orig_sdpa = F.scaled_dot_product_attention
        F.scaled_dot_product_attention = self._hooked
        return self

    def _hooked(self, *args, **kwargs):
        result = self.attn_fn(*args, **kwargs)
        if self.max_calls is None or len(self.outputs) < self.max_calls:
            self.outputs.append(result.detach().float().cpu())
            # args[0] is query
            if len(args) > 0:
                self.shapes.append(tuple(args[0].shape))
            else:
                self.shapes.append(())
        return result

    def __exit__(self, *_exc):
        F.scaled_dot_product_attention = self._orig_sdpa
        return False

    def clear(self):
        """Free captured outputs."""
        self.outputs.clear()
        self.shapes.clear()


def _load_cogvideox_pipe(model: str) -> Tuple:
    """Load CogVideoX pipeline. Returns (pipe, num_frames, torch_dtype)."""
    from diffusers import CogVideoXPipeline

    if model == "cogvideox-2b":
        model_path = "/storage/yiliu7/THUDM/CogVideoX-2b"
        num_frames = 49
        torch_dtype = torch.float16
    else:
        model_path = "THUDM/CogVideoX1.5-5B"
        num_frames = 81
        torch_dtype = torch.bfloat16

    pipe = CogVideoXPipeline.from_pretrained(model_path, torch_dtype=torch_dtype)
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    return pipe, num_frames, torch_dtype


def _run_pipe(pipe, prompt: str, num_steps: int, num_frames: int, seed: int):
    """Run the pipeline with deterministic seed."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    result = pipe(
        prompt=prompt,
        num_videos_per_prompt=1,
        num_inference_steps=num_steps,
        num_frames=num_frames,
        guidance_scale=6,
        generator=generator,
    )
    return result


def run_part_b(
    formats: List[str],
    model: str,
    seed: int,
    num_steps: int,
    num_frames: int,
) -> Dict:
    """Part B: Per-layer accuracy in CogVideoX inference."""
    print_header("Part B: Per-Layer CogVideoX Accuracy")
    print(f"  Model: {model}, Steps: {num_steps}, Frames: {num_frames}")
    print(f"  Formats: {formats}")
    print(f"  Seed: {seed}")

    prompt = "A dog is running in the park."

    # Load pipeline (once)
    print("\n  Loading CogVideoX pipeline...")
    pipe, default_frames, _torch_dtype = _load_cogvideox_pipe(model)
    if num_frames is None:
        num_frames = default_frames
    print(f"  Pipeline loaded. Using {num_frames} frames, {num_steps} steps.")

    # --- Reference run with SDPA ---
    print("\n  Running SDPA reference...")
    orig_sdpa = F.scaled_dot_product_attention
    with AttentionCapture(orig_sdpa) as ref_cap:
        _run_pipe(pipe, prompt, num_steps, num_frames, seed)
    ref_outputs = ref_cap.outputs
    num_calls = len(ref_outputs)
    print(f"  Captured {num_calls} attention calls from SDPA reference.")

    results = {}

    # --- Per-format runs ---
    for fmt in formats:
        print(f"\n  Running format: {fmt}...")
        sage_fn = build_sage_fn(fmt)

        with AttentionCapture(sage_fn) as fmt_cap:
            _run_pipe(pipe, prompt, num_steps, num_frames, seed)

        fmt_outputs = fmt_cap.outputs
        n = min(len(ref_outputs), len(fmt_outputs))
        if n != num_calls:
            print(f"    WARNING: {fmt} produced {len(fmt_outputs)} calls vs {num_calls} reference calls")

        # Per-call metrics
        per_call_metrics = []
        for i in range(n):
            m = compute_metrics(ref_outputs[i], fmt_outputs[i])
            per_call_metrics.append(m)

        # Estimate num_layers: CogVideoX-2b has 30 transformer blocks.
        # Each step calls attention in each block. With classifier-free guidance
        # there may be 2x calls. We'll infer from the data.
        # Group by layer index (call_idx % num_layers_per_step)
        calls_per_step = num_calls // num_steps if num_steps > 0 else num_calls
        if calls_per_step == 0:
            calls_per_step = num_calls

        # Per-layer average (layer = call_idx % calls_per_step)
        layer_metrics: Dict[int, List[AccuracyMetrics]] = {}
        for i, m in enumerate(per_call_metrics):
            layer_idx = i % calls_per_step
            layer_metrics.setdefault(layer_idx, []).append(m)

        # Average metrics per layer
        layer_avg = {
            layer_idx: _average_metrics(metrics_list)
            for layer_idx, metrics_list in sorted(layer_metrics.items())
        }

        # Print per-layer table
        print(f"\n  Per-layer accuracy for {fmt} ({calls_per_step} layers/step, {num_steps} steps):")
        rows = [(f"layer_{idx:02d}", m) for idx, m in sorted(layer_avg.items())]
        print_metrics_table(rows, label_header="Layer")

        # Summary: worst/best layers (by SNR)
        sorted_by_snr = sorted(layer_avg.items(), key=lambda x: x[1].snr_db)
        print("  Worst 3 layers (by SNR):")
        for idx, m in sorted_by_snr[:3]:
            print(f"    layer_{idx:02d}: SNR={m.snr_db:.2f}dB  CosSim={m.cos_sim:.8f}  L1Rel={m.l1_rel:.6f}")

        # Global average
        global_avg = _average_metrics(per_call_metrics)
        print(f"  Global avg: SNR={global_avg.snr_db:.2f}dB  CosSim={global_avg.cos_sim:.8f}  L2Rel={global_avg.l2_rel:.6f}")

        # Per-head metrics (average across all calls)
        H = ref_outputs[0].shape[1] if ref_outputs[0].ndim == 4 else None
        if H is not None:
            head_metrics_accum: Dict[int, List[AccuracyMetrics]] = {}
            for i in range(n):
                for h in range(H):
                    hm = compute_metrics(ref_outputs[i][:, h], fmt_outputs[i][:, h])
                    head_metrics_accum.setdefault(h, []).append(hm)
            head_avg = {h: _average_metrics(mlist) for h, mlist in sorted(head_metrics_accum.items())}

            # Print per-head summary
            sorted_heads = sorted(head_avg.items(), key=lambda x: x[1].snr_db)
            print(f"\n  Per-head accuracy for {fmt} ({H} heads, averaged over {n} calls):")
            print("    Worst 5 heads (by SNR):")
            for h, m in sorted_heads[:5]:
                print(f"      head_{h:02d}: SNR={m.snr_db:.2f}dB  CosSim={m.cos_sim:.6f}  L1Rel={m.l1_rel:.6f}")
            print("    Best 5 heads (by SNR):")
            for h, m in sorted_heads[-5:][::-1]:
                print(f"      head_{h:02d}: SNR={m.snr_db:.2f}dB  CosSim={m.cos_sim:.6f}  L1Rel={m.l1_rel:.6f}")

        results[fmt] = {
            "num_calls": n,
            "calls_per_step": calls_per_step,
            "global_avg": asdict(global_avg),
            "per_layer": {
                f"layer_{idx:02d}": asdict(m) for idx, m in sorted(layer_avg.items())
            },
            "worst_layers": [
                {"layer": f"layer_{idx:02d}", **asdict(m)} for idx, m in sorted_by_snr[:5]
            ],
        }

        # Add per-head results if available
        if H is not None:
            results[fmt]["per_head"] = {f"head_{h:02d}": asdict(m) for h, m in sorted(head_avg.items())}
            results[fmt]["worst_heads"] = [
                {"head": f"head_{h:02d}", **asdict(m)} for h, m in sorted_heads[:5]
            ]

        # Free format outputs
        del fmt_outputs
        fmt_cap.clear()
        gc.collect()
        torch.cuda.empty_cache()

    # Free reference outputs
    del ref_outputs
    ref_cap.clear()
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    return {"part_b": results}


# ============================================================================
# Part C: Final-Step Latent Divergence
# ============================================================================

class LatentCapture:
    """Hooks pipe.scheduler.step to capture the final latent."""

    def __init__(self):
        self.final_latent: Optional[torch.Tensor] = None
        self._orig_step = None
        self._call_count = 0
        self._total_steps = 0

    def attach(self, pipe, total_steps: int):
        self._orig_step = pipe.scheduler.step
        self._call_count = 0
        self._total_steps = total_steps
        capture = self

        def _hooked_step(*args, **kwargs):
            result = capture._orig_step(*args, **kwargs)
            capture._call_count += 1
            if capture._call_count == capture._total_steps:
                # Handle both return_dict=True (dataclass) and
                # return_dict=False (tuple) — CogVideoX pipeline uses
                # return_dict=False so result is (prev_sample, ...).
                if isinstance(result, tuple):
                    latent = result[0]
                else:
                    latent = result.prev_sample
                capture.final_latent = latent.detach().float().cpu()
            return result

        pipe.scheduler.step = _hooked_step

    def detach(self, pipe):
        if self._orig_step is not None:
            pipe.scheduler.step = self._orig_step
            self._orig_step = None


def run_part_c(
    formats: List[str],
    model: str,
    seed: int,
    num_steps: int,
    num_frames: int,
) -> Dict:
    """Part C: Final-step latent divergence."""
    print_header("Part C: Final-Step Latent Divergence")
    print(f"  Model: {model}, Steps: {num_steps}, Frames: {num_frames}")
    print(f"  Formats: {formats}")
    print(f"  Seed: {seed}")

    prompt = "A dog is running in the park."

    # Load pipeline (once)
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
    # Ensure SDPA is used (restore original)
    F.scaled_dot_product_attention = orig_sdpa
    _run_pipe(pipe, prompt, num_steps, num_frames, seed)
    ref_capture.detach(pipe)
    ref_latent = ref_capture.final_latent
    if ref_latent is None:
        print("  WARNING: Could not capture reference latent!")
        return {"part_c": {"error": "Failed to capture reference latent"}}
    print(f"  Reference latent shape: {ref_latent.shape}")

    results = {}
    rows = []

    # --- Per-format runs ---
    for fmt in formats:
        print(f"\n  Running format: {fmt}...")
        sage_fn = build_sage_fn(fmt)

        fmt_capture = LatentCapture()
        fmt_capture.attach(pipe, num_steps)
        F.scaled_dot_product_attention = sage_fn
        _run_pipe(pipe, prompt, num_steps, num_frames, seed)
        fmt_capture.detach(pipe)

        # Restore SDPA
        F.scaled_dot_product_attention = orig_sdpa

        fmt_latent = fmt_capture.final_latent
        if fmt_latent is None:
            print(f"    WARNING: Could not capture {fmt} latent!")
            results[fmt] = {"error": "Failed to capture latent"}
            continue

        m = compute_metrics(ref_latent, fmt_latent)
        results[fmt] = asdict(m)
        rows.append((fmt, m))
        print(f"    SNR={m.snr_db:.2f}dB  CosSim={m.cos_sim:.8f}  L2Rel={m.l2_rel:.6f}")

        del fmt_latent
        gc.collect()
        torch.cuda.empty_cache()

    if rows:
        print("\n  Summary — Final Latent Divergence:")
        print_metrics_table(rows)

    # Restore SDPA and clean up
    F.scaled_dot_product_attention = orig_sdpa
    del ref_latent, pipe
    gc.collect()
    torch.cuda.empty_cache()

    return {"part_c": results}


# ============================================================================
# CLI and Main
# ============================================================================

ALL_FORMATS = ["nvfp4", "mxfp4", "mxfp4_s1", "mxfp8_s1"]


def main():
    parser = argparse.ArgumentParser(
        description="MX Format Accuracy Ablation — Phase 1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--parts", nargs="+", default=["A"],
        choices=["A", "B", "C"],
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
    parser.add_argument("--causal", action="store_true",
                        help="Use causal masking in Part A")

    args = parser.parse_args()

    print_header("MX Format Accuracy Ablation — Phase 1")
    print(f"  Parts:   {args.parts}")
    print(f"  Formats: {args.formats}")
    print(f"  Model:   {args.model}")
    print(f"  Seed:    {args.seed}")
    if "B" in args.parts or "C" in args.parts:
        print(f"  Steps:   {args.num_steps}")
        print(f"  Frames:  {args.num_frames}")

    all_results = {}
    start_time = time.time()

    if "A" in args.parts:
        result = run_part_a(args.formats, args.seed, args.causal)
        all_results.update(result)

    if "B" in args.parts:
        result = run_part_b(args.formats, args.model, args.seed,
                            args.num_steps, args.num_frames)
        all_results.update(result)

    if "C" in args.parts:
        result = run_part_c(args.formats, args.model, args.seed,
                            args.num_steps, args.num_frames)
        all_results.update(result)

    elapsed = time.time() - start_time

    print_header("Done")
    print(f"  Total time: {elapsed:.1f}s")

    if args.output_json:
        # Add metadata
        all_results["_metadata"] = {
            "formats": args.formats,
            "model": args.model,
            "seed": args.seed,
            "num_steps": args.num_steps,
            "num_frames": args.num_frames,
            "causal": args.causal,
            "elapsed_s": elapsed,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        }
        with open(args.output_json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  Results saved to: {args.output_json}")


if __name__ == "__main__":
    main()
