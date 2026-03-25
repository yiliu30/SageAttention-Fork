"""
Quantization configuration: enums, QuantConfig, AttentionConfig, and the ATTENTION_CONFIGS registry.

This is the central data model for the composable architecture. Each quantization
scheme is fully described by an AttentionConfig that composes:
- QK quantization config (host-side Q/K preprocessing)
- PV quantization config (host-side V preprocessing + P-quant format)
- P-quant function (in-kernel, passed as constexpr)
- Pre-transforms pipeline (smoothing, rotation, etc.)
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Optional, List, TYPE_CHECKING

import triton

if TYPE_CHECKING:
    from .transforms import TransformFn

from .quant_primitives import (
    apply_e2m1_quantization_torch,
    apply_e4m3_quantization_torch,
    round_to_e4m3_torch,
    round_to_e8m0_torch,
)


# ============================================================================
# Enums
# ============================================================================

class DataFormat(IntEnum):
    """Data quantization format."""
    E2M1 = 0    # FP4 (±{0, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6})
    E4M3 = 1    # FP8 ([-448, 448])
    E5M2 = 2    # FP8 alternate
    INT8 = 3    # 8-bit integer
    INT4 = 4    # 4-bit integer


class ScaleFormat(IntEnum):
    """Scale rounding format."""
    E4M3 = 0    # NVFP4 style (3-bit mantissa)
    E8M0 = 1    # MXFP4/MXFP8 style (power-of-2 shared exponent)
    NONE = 2    # No scale rounding (INT schemes, standalone FP8)


class RoundingMode(IntEnum):
    """Rounding mode for scale and/or data quantization."""
    STOCHASTIC = 0
    RNE = 1          # Round-to-nearest-even
    CEIL = 2
    FLOOR = 3


# ============================================================================
# QuantConfig — per-stage quantization configuration
# ============================================================================

@dataclass(frozen=True)
class QuantConfig:
    """
    Configuration for one quantization stage (QK or PV).

    Carries both the abstract format description (enums) and the concrete
    callable implementations needed by host-side quantization (quantize.py).
    """
    name: str
    data_format: DataFormat
    scale_format: ScaleFormat
    rounding_mode: RoundingMode = RoundingMode.STOCHASTIC
    block_size: int = 16
    scale_levels: int = 2       # 1 = single-level, 2 = two-level
    fp_max: float = 6.0         # Max representable value for the data format

    # Torch callables for host-side quantization (used by quantize.py)
    round_scale_torch: Optional[Callable] = None   # e.g., round_to_e4m3_torch
    quant_data_torch: Optional[Callable] = None     # e.g., apply_e2m1_quantization_torch; None → E2M1 default


# ============================================================================
# AttentionConfig — full attention configuration
# ============================================================================

@dataclass
class AttentionConfig:
    """
    Composes QK quant, PV quant, P-quant kernel, and pre-transforms.

    This is the single object that fully describes an attention quantization scheme.
    """
    name: str
    qk_quant: QuantConfig               # Host-side Q/K quantization config
    pv_quant: QuantConfig               # Host-side V quantization config
    p_quant_fn: triton.JITFunction      # @triton.jit function for in-kernel P quantization
    pre_transforms: List = field(default_factory=list)  # Ordered pipeline of TransformFn

    def validate(self):
        """Validate config consistency."""
        assert self.qk_quant.round_scale_torch is not None, \
            f"{self.name}: qk_quant missing round_scale_torch"
        assert self.pv_quant.round_scale_torch is not None, \
            f"{self.name}: pv_quant missing round_scale_torch"
        assert self.p_quant_fn is not None, \
            f"{self.name}: missing p_quant_fn"


# ============================================================================
# ATTENTION_CONFIGS registry
# ============================================================================
# Populated after imports to avoid circular dependencies.
# p_quant_kernels.py registers its functions into P_QUANT_REGISTRY on import.

ATTENTION_CONFIGS: dict[str, AttentionConfig] = {}


def _register_builtin_configs():
    """Register all built-in attention configurations."""
    from .p_quant_registry import P_QUANT_REGISTRY
    # Ensure P-quant kernels are registered (import triggers @register_p_quant)
    from . import p_quant_kernels as _  # noqa: F401
    from .transforms import qk_smoothing

    # ── NVFP4: Two-level, E4M3 scales, block_size=16 ──
    nvfp4_quant = QuantConfig(
        name="nvfp4",
        data_format=DataFormat.E2M1,
        scale_format=ScaleFormat.E4M3,
        block_size=16,
        scale_levels=2,
        fp_max=6.0,
        round_scale_torch=round_to_e4m3_torch,
        quant_data_torch=None,  # Default E2M1
    )
    ATTENTION_CONFIGS["nvfp4"] = AttentionConfig(
        name="nvfp4",
        qk_quant=nvfp4_quant,
        pv_quant=nvfp4_quant,
        p_quant_fn=P_QUANT_REGISTRY["nvfp4"],
        pre_transforms=[qk_smoothing],
    )

    # ── MXFP4: Two-level, E8M0 scales, block_size=32 ──
    mxfp4_quant = QuantConfig(
        name="mxfp4",
        data_format=DataFormat.E2M1,
        scale_format=ScaleFormat.E8M0,
        block_size=32,
        scale_levels=2,
        fp_max=6.0,
        round_scale_torch=round_to_e8m0_torch,
        quant_data_torch=None,  # Default E2M1
    )
    ATTENTION_CONFIGS["mxfp4"] = AttentionConfig(
        name="mxfp4",
        qk_quant=mxfp4_quant,
        pv_quant=mxfp4_quant,
        p_quant_fn=P_QUANT_REGISTRY["mxfp4"],
        pre_transforms=[qk_smoothing],
    )

    # ── MXFP4_S1: Single-level, E8M0 scales, block_size=32 ──
    mxfp4_s1_quant = QuantConfig(
        name="mxfp4_s1",
        data_format=DataFormat.E2M1,
        scale_format=ScaleFormat.E8M0,
        block_size=32,
        scale_levels=1,
        fp_max=6.0,
        round_scale_torch=round_to_e8m0_torch,
        quant_data_torch=None,  # Default E2M1
    )
    ATTENTION_CONFIGS["mxfp4_s1"] = AttentionConfig(
        name="mxfp4_s1",
        qk_quant=mxfp4_s1_quant,
        pv_quant=mxfp4_s1_quant,
        p_quant_fn=P_QUANT_REGISTRY["mxfp4_s1"],
        pre_transforms=[qk_smoothing],
    )

    # ── MXFP8_S1: Single-level, E8M0 scales, FP8 data, block_size=32 ──
    mxfp8_s1_quant = QuantConfig(
        name="mxfp8_s1",
        data_format=DataFormat.E4M3,
        scale_format=ScaleFormat.E8M0,
        block_size=32,
        scale_levels=1,
        fp_max=448.0,
        round_scale_torch=round_to_e8m0_torch,
        quant_data_torch=apply_e4m3_quantization_torch,
    )
    ATTENTION_CONFIGS["mxfp8_s1"] = AttentionConfig(
        name="mxfp8_s1",
        qk_quant=mxfp8_s1_quant,
        pv_quant=mxfp8_s1_quant,
        p_quant_fn=P_QUANT_REGISTRY["mxfp8_s1"],
        pre_transforms=[qk_smoothing],
    )

    # Validate all configs
    for config in ATTENTION_CONFIGS.values():
        config.validate()


# Auto-register on module import
_register_builtin_configs()
