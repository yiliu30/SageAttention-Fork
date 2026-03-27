"""Reusable GPU profiling utilities for diffusion model workload analysis.

Classes:
    AttentionProportionProfiler — wraps a callable with CUDA event timing
    ModuleForwardTimer — wraps a module's forward() with CUDA event timing
    LinearCategoryProfiler — auto-discovers and categorizes all nn.Linear layers
"""

import torch
import torch.nn as nn
from collections import defaultdict
from typing import Callable, Dict, Optional


class AttentionProportionProfiler:
    """Wraps an attention function with async CUDA event timing to measure
    the total GPU time spent in attention without serializing the pipeline."""

    def __init__(self, attn_fn):
        self.attn_fn = attn_fn
        self.events = []  # list of (start_event, end_event)

    def __call__(self, *args, **kwargs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = self.attn_fn(*args, **kwargs)
        end.record()
        self.events.append((start, end))
        return result

    def total_attn_ms(self):
        torch.cuda.synchronize()
        return sum(s.elapsed_time(e) for s, e in self.events)

    def call_count(self):
        return len(self.events)

    def reset(self):
        self.events.clear()


class ModuleForwardTimer:
    """Wraps a module's forward method with async CUDA event timing.
    Use as a context manager or call attach/detach manually."""

    def __init__(self, module):
        self.module = module
        self.events = []  # list of (start_event, end_event)
        self._orig_forward = None

    def attach(self):
        self._orig_forward = self.module.forward
        timer = self
        orig = self._orig_forward

        def timed_forward(*args, **kwargs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = orig(*args, **kwargs)
            end.record()
            timer.events.append((start, end))
            return result

        self.module.forward = timed_forward

    def detach(self):
        if self._orig_forward is not None:
            self.module.forward = self._orig_forward
            self._orig_forward = None

    def total_ms(self):
        torch.cuda.synchronize()
        return sum(s.elapsed_time(e) for s, e in self.events)

    def call_count(self):
        return len(self.events)

    def reset(self):
        self.events.clear()


# ---------------------------------------------------------------------------
# Default linear-layer classification heuristics
# ---------------------------------------------------------------------------

def _default_classify_linear(name: str, _module: nn.Linear) -> str:
    """Classify a nn.Linear layer by its named_modules() path.

    Works across diffusers transformer models (CogVideoX, WAN, HunyuanVideo,
    Mochi, LTX-Video) without model-specific hardcoding.

    Categories:
        attn_qkv    — name contains to_q, to_k, or to_v
        attn_out    — name contains to_out
        ffn         — name contains ff, mlp, or feed_forward
        norm_linear — name contains norm (adaptive norm scale/shift)
        other_linear — anything else
    """
    name_lower = name.lower()
    # Check attention QKV projections
    if "to_q" in name_lower or "to_k" in name_lower or "to_v" in name_lower:
        return "attn_qkv"
    # Check attention output projection
    if "to_out" in name_lower:
        return "attn_out"
    # Check feedforward layers
    if "ff" in name_lower or "mlp" in name_lower or "feed_forward" in name_lower:
        return "ffn"
    # Check norm-related linear layers (adaptive norm scale/shift)
    if "norm" in name_lower:
        return "norm_linear"
    return "other_linear"


class LinearCategoryProfiler:
    """Model-agnostic profiler that auto-discovers all nn.Linear layers in a
    transformer module and categorizes them by name-pattern heuristics.

    Args:
        module: Any nn.Module (e.g., ``pipe.transformer``).
        classify_fn: Optional callback ``(name: str, module: nn.Linear) -> str``
            for custom categorization. Defaults to built-in heuristics that
            work for CogVideoX, WAN, HunyuanVideo, Mochi, LTX-Video, etc.
    """

    # Canonical category display order
    CATEGORY_ORDER = ["attn_qkv", "attn_out", "ffn", "norm_linear", "other_linear"]

    def __init__(
        self,
        module: nn.Module,
        classify_fn: Optional[Callable[[str, nn.Linear], str]] = None,
    ):
        self._classify_fn = classify_fn or _default_classify_linear
        # {category: [(name, ModuleForwardTimer), ...]}
        self._timers: Dict[str, list] = defaultdict(list)
        self._all_timers: list = []  # flat list for bulk ops

        for name, submod in module.named_modules():
            if isinstance(submod, nn.Linear):
                category = self._classify_fn(name, submod)
                timer = ModuleForwardTimer(submod)
                timer.attach()
                self._timers[category].append((name, timer))
                self._all_timers.append(timer)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self):
        """Clear all recorded events across every timer."""
        for timer in self._all_timers:
            timer.reset()

    def total_ms_by_category(self) -> Dict[str, float]:
        """Return ``{category: total_gpu_ms}`` for all categories."""
        torch.cuda.synchronize()
        result = {}
        for cat in self.CATEGORY_ORDER:
            timers = self._timers.get(cat, [])
            result[cat] = sum(t.total_ms() for _, t in timers)
        return result

    def call_count_by_category(self) -> Dict[str, int]:
        """Return ``{category: total_call_count}`` for all categories."""
        result = {}
        for cat in self.CATEGORY_ORDER:
            timers = self._timers.get(cat, [])
            result[cat] = sum(t.call_count() for _, t in timers)
        return result

    def layer_count_by_category(self) -> Dict[str, int]:
        """Return ``{category: num_linear_layers}`` for all categories."""
        return {cat: len(self._timers.get(cat, [])) for cat in self.CATEGORY_ORDER}

    def total_linear_ms(self) -> float:
        """Sum of all linear-layer GPU time across all categories."""
        torch.cuda.synchronize()
        return sum(t.total_ms() for t in self._all_timers)

    def detach_all(self):
        """Remove timing hooks from all wrapped modules."""
        for timer in self._all_timers:
            timer.detach()

    def summary(self) -> str:
        """Human-readable summary of all categories."""
        lines = ["Linear Layer Profiling Summary", "=" * 50]
        layer_counts = self.layer_count_by_category()
        ms_by_cat = self.total_ms_by_category()
        calls_by_cat = self.call_count_by_category()
        total_ms = sum(ms_by_cat.values())

        for cat in self.CATEGORY_ORDER:
            ms = ms_by_cat.get(cat, 0.0)
            calls = calls_by_cat.get(cat, 0)
            layers = layer_counts.get(cat, 0)
            pct = (ms / total_ms * 100) if total_ms > 0 else 0.0
            lines.append(
                f"  {cat:15s}: {ms:>9.1f} ms  ({pct:>5.1f}%)  "
                f"[{layers} layers, {calls} calls]"
            )

        lines.append(f"  {'TOTAL':15s}: {total_ms:>9.1f} ms")
        lines.append(
            f"\nDiscovered {sum(layer_counts.values())} nn.Linear layers total"
        )
        return "\n".join(lines)
