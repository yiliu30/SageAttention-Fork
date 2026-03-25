# Standalone Triton Kernel — Architecture Refactoring Design

**Date**: 2026-03-24
**Scope**: New `standalone/sage3/` package — the original `standalone/sageattention3_standalone.py` stays as the reference implementation.
**Goal**: Make it trivial to add new quantization schemes (INT8, INT4, FP8, rounding modes) and pre-quantization transforms (Hadamard rotation) without touching the attention kernel.

---

## Problems Solved

See [problems.md](./problems.md) for the full problem list. In brief:

- **P1**: Adding a scheme requires ~6 touch points → reduced to 1 P-quant function + 1 config registration
- **P2**: 4 duplicated P-quant kernels (~380 lines) → composable functions from shared primitives
- **P3**: No extension point for pre-quant transforms → ordered `pre_transforms` pipeline
- **P4**: Boolean constexpr flags don't scale → zero dispatch via function-as-constexpr (verified on Triton)
- **P5**: Single 1400-line file → 8 focused modules, each independently testable

## Must-Have Schemes

The architecture must support all of these without structural changes:

| Scheme | Data format | Scale format | Notes |
|--------|------------|-------------|-------|
| NVFP4 (existing) | E2M1 | E4M3 | Two-level scales |
| MXFP4 (existing) | E2M1 | E8M0 | Two-level and single-level variants |
| MXFP8 (existing) | E4M3 | E8M0 | Single-level |
| **INT8** (new) | 8-bit integer | per-block | For Q/K or P quantization |
| **INT4** (new) | 4-bit integer | per-block | Lower precision alternative to FP4 |
| **FP8** (new) | E4M3 or E5M2 | per-block | Standalone (not microscaling) |
| **Rounding modes** (new) | — | — | ceil, floor, RNE as composable options for any format |
| **Mixed precision** | e.g., MXFP8 QK + MXFP4 PV | — | Different quant for QK vs PV stages |

---

## Core Design: Composable QuantConfig + Zero-Dispatch Kernel

### Key Insight (Verified)

Triton supports passing `@triton.jit` functions as `tl.constexpr` parameters. The function is inlined at compile time — no dispatch overhead, no if/elif chain. Each unique function produces a separately compiled kernel.

Verified with tests on this repo's Triton installation (see test results in the design discussion).

### Data Model

```python
# ── Enums ──

class DataFormat(IntEnum):
    E2M1 = 0    # FP4
    E4M3 = 1    # FP8
    E5M2 = 2    # FP8 (alternate)
    INT8 = 3
    INT4 = 4

class ScaleFormat(IntEnum):
    E4M3 = 0    # NVFP4 style
    E8M0 = 1    # MXFP4/MXFP8 style (shared exponent)
    NONE = 2    # No scale rounding (INT schemes, standalone FP8)

class RoundingMode(IntEnum):
    STOCHASTIC = 0
    RNE = 1          # Round-to-nearest-even
    CEIL = 2
    FLOOR = 3

# ── Per-stage config ──

@dataclass
class QuantConfig:
    """Configuration for one quantization stage (QK or PV)."""
    name: str
    data_format: DataFormat
    scale_format: ScaleFormat
    rounding_mode: RoundingMode = RoundingMode.STOCHASTIC
    block_size: int = 16
    scale_levels: int = 2      # 1 = single-level, 2 = two-level
    fp_max: float = 6.0        # Max representable value

# ── Full attention config ──

@dataclass
class AttentionConfig:
    """Composes QK quant, PV quant, P-quant kernel, and pre-transforms."""
    name: str
    qk_quant: QuantConfig              # Host-side Q/K quantization
    pv_quant: QuantConfig              # Host-side V quantization + P-quant format
    p_quant_fn: triton.JITFunction     # @triton.jit function for in-kernel P quantization
    pre_transforms: list = field(default_factory=list)  # Ordered pipeline: smoothing, rotation, etc.
```

### P-Quant Registry

P-quant functions self-register via decorator:

```python
# p_quant_registry.py
P_QUANT_REGISTRY: dict[str, triton.JITFunction] = {}

def register_p_quant(name: str):
    def wrapper(fn):
        jit_fn = triton.jit(fn)
        P_QUANT_REGISTRY[name] = jit_fn
        return jit_fn
    return wrapper
```

```python
# p_quant_kernels.py
@register_p_quant("mxfp4_s1")
def p_quant_mxfp4_s1(p_tile, BLOCK_N: tl.constexpr):
    block_max = tl.max(tl.abs(p_tile), axis=1)
    scale = round_e8m0(block_max / 6.0)          # shared primitive
    scaled = p_tile / (scale[:, None] + 1e-12)
    quantized = quant_e2m1(scaled)                # shared primitive
    return quantized * scale[:, None]
```

`AttentionConfig` references P-quant functions by registry lookup:

```python
ATTENTION_CONFIGS["mxfp4_s1"] = AttentionConfig(
    name="mxfp4_s1",
    ...,
    p_quant_fn=P_QUANT_REGISTRY["mxfp4_s1"],
)
```

### Zero-Dispatch Attention Kernel

The attention kernel takes the P-quant function as a constexpr — no dispatch logic at all:

```python
@triton.jit
def tiled_online_attention_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    # ... strides, dimensions, sm_scale ...
    P_QUANT_FN: tl.constexpr,        # ← pluggable P-quant
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    # ... other constexprs ...
):
    # ... load Q tile, iterate K/V tiles, compute QK^T, softmax ...

    # Single function call — Triton inlines at compile time
    p_quantized = P_QUANT_FN(p_tile, BLOCK_N)

    # ... PV matmul, accumulate, store ...
```

### Pre-Transform Pipeline

Each transform is a function with signature `(q, k, v, ctx) → (q, k, v, ctx)`. The `ctx` dict carries metadata between transforms and to the kernel launcher (e.g., `delta_s` from smoothing, `v_mean` from V-smoothing).

```python
def qk_smoothing(q, k, v, ctx):
    """Subtract K's mean to reduce outlier impact."""
    k_mean = k.mean(dim=-2, keepdim=True)
    k = k - k_mean
    ctx["delta_s"] = compute_delta_s(q, k_mean)
    return q, k, v, ctx

def hadamard_rotation(q, k, v, ctx):
    """Apply Hadamard rotation before quantization."""
    H = get_hadamard_matrix(k.shape[-1])
    k = k @ H
    q = q @ H
    return q, k, v, ctx
```

Configuring smoothing on/off or adding rotation:

```python
# With smoothing (default):
AttentionConfig(..., pre_transforms=[qk_smoothing, v_smoothing])

# Without smoothing:
AttentionConfig(..., pre_transforms=[])

# With rotation + smoothing:
AttentionConfig(..., pre_transforms=[hadamard_rotation, qk_smoothing, v_smoothing])
```

### Mixed Precision

Different quant for QK vs PV is just two different `QuantConfig`s:

```python
ATTENTION_CONFIGS["mixed_mxfp8qk_mxfp4pv"] = AttentionConfig(
    name="mixed_mxfp8qk_mxfp4pv",
    qk_quant=QuantConfig("mxfp8_qk", DataFormat.E4M3, ScaleFormat.E8M0,
                          block_size=32, scale_levels=1, fp_max=448.0),
    pv_quant=QuantConfig("mxfp4_pv", DataFormat.E2M1, ScaleFormat.E8M0,
                          block_size=32, scale_levels=1, fp_max=6.0),
    p_quant_fn=P_QUANT_REGISTRY["mxfp4_s1"],   # P uses MXFP4
    pre_transforms=[qk_smoothing],
)
```

---

## Call Chain

```
User: sageattn3_standalone(q, k, v, attention_type="mxfp4_s1")
  │
  ▼
api.py: sageattn3_standalone()
  │  config = ATTENTION_CONFIGS["mxfp4_s1"]
  │
  ├─▶ transforms.py: for t in config.pre_transforms: q,k,v,ctx = t(q,k,v,ctx)
  │
  ├─▶ quantize.py: quantize_kv(k, v, config) → (k_q, k_s, v_q, v_s)
  │     Uses quant_primitives based on config.qk_quant / config.pv_quant
  │
  └─▶ attention_kernel.py: tiled_online_attention_kernel[grid](
         ..., P_QUANT_FN=config.p_quant_fn, ...)
              │
              └─▶ p_quant_kernels.py: p_quant_mxfp4_s1(p_tile, BLOCK_N)
                    │  (inlined by Triton at compile time)
                    └─▶ quant_primitives.py: round_e8m0(), quant_e2m1()
```

---

## File Structure

```
standalone/
├── sageattention3_standalone.py    # ORIGINAL — kept as reference, not modified
│
├── sage3/                          # NEW — refactored implementation
│   ├── __init__.py                 # Exports: sageattn3_standalone, scaled_dot_product_attention
│   ├── quant_primitives.py         # round/quant functions (torch + triton pairs)
│   ├── p_quant_registry.py         # P_QUANT_REGISTRY + @register_p_quant decorator (~20 lines)
│   ├── p_quant_kernels.py          # All P-quant @triton.jit functions (self-registering)
│   ├── quant_config.py             # QuantConfig, AttentionConfig, ATTENTION_CONFIGS
│   ├── transforms.py               # Pre-quant transforms (smoothing, rotation)
│   ├── quantize.py                 # Host-side K/V quantization
│   ├── attention_kernel.py         # Tiled online attention Triton kernel
│   └── api.py                      # Entry points
│
├── test_standalone.py              # Existing tests (updated to test both original and sage3/)
└── debug_mxfp8_p_quant.py         # Existing debug script
```

Module dependency flow (no cycles):

```
quant_primitives  ←── p_quant_kernels  ←── quant_config
                  ←── quantize               ↑
                                             │
p_quant_registry ←── p_quant_kernels   ── api ──→ attention_kernel
                                             │
transforms ──────────────────────────────────┘
```

---

## Adding a New Scheme — Complete Recipe

**Example: Adding INT4 quantization**

### Step 1: Add primitive (if data format is new)

In `quant_primitives.py`:
```python
@triton.jit
def quant_int4_triton(x):
    return tl.clamp(tl.nearbyint(x), -7.0, 7.0)

def quant_int4_torch(x):
    return torch.clamp(torch.round(x), -7, 7)
```

### Step 2: Write P-quant function

In `p_quant_kernels.py`:
```python
@register_p_quant("int4")
def p_quant_int4(p_tile, BLOCK_N: tl.constexpr):
    block_max = tl.max(tl.abs(p_tile), axis=1)
    scale = block_max / 7.0
    scaled = p_tile / (scale[:, None] + 1e-12)
    quantized = quant_int4_triton(scaled)
    return quantized * scale[:, None]
```

### Step 3: Register AttentionConfig

In `quant_config.py`:
```python
ATTENTION_CONFIGS["int4"] = AttentionConfig(
    name="int4",
    qk_quant=QuantConfig("int4_qk", DataFormat.INT4, ScaleFormat.NONE, block_size=32, fp_max=7.0),
    pv_quant=QuantConfig("int4_pv", DataFormat.INT4, ScaleFormat.NONE, block_size=32, fp_max=7.0),
    p_quant_fn=P_QUANT_REGISTRY["int4"],
    pre_transforms=[qk_smoothing],
)
```

### Step 4: Done

`--attention_type int4` works in example scripts. No changes to the attention kernel, launcher, or any other file.

---

## Rounding Modes — Composable, Not Per-Scheme

Rounding mode is a field on `QuantConfig`. A P-quant function can accept it as a further constexpr, or a separate P-quant variant can be registered:

**Option A (constexpr parameter on P-quant)**:
```python
@register_p_quant("mxfp4_s1_rne")
def p_quant_mxfp4_s1_rne(p_tile, BLOCK_N: tl.constexpr):
    # Same as mxfp4_s1 but using RNE rounding
    ...
    quantized = quant_e2m1_rne(scaled)   # RNE variant of the primitive
    ...
```

**Option B (parameterized primitive)**:
The quant primitive itself takes a `ROUND_MODE: tl.constexpr`:
```python
@triton.jit
def quant_e2m1(x, ROUND_MODE: tl.constexpr = 0):
    if ROUND_MODE == 0:    # Stochastic
        ...
    elif ROUND_MODE == 1:  # RNE
        ...
```

Both work. Option A is simpler (one function per combination, explicit). Option B is more composable (fewer functions, but the primitive is more complex). Recommend starting with Option A and moving to Option B if the combinatorial explosion becomes real.

---

## Backward Compatibility

- `standalone/sageattention3_standalone.py` is **not modified** — it remains the reference
- `standalone/sage3/` is the new implementation
- Tests should validate that both produce identical results for existing schemes
- Example scripts can be updated to import from `sage3/` with a feature flag or CLI option

---

## Open Questions

1. **V-smoothing as a transform**: Currently V-smoothing is tightly coupled to the FP8 PV path. Should it be a generic transform in `pre_transforms`, or does it remain coupled to `pv_quant` config?
2. **Compile time**: With N schemes, Triton compiles N kernels on first use. Is cold-start compile time a concern? (Can be mitigated with Triton's caching.)
3. **Test strategy**: Should `test_standalone.py` test both the original and refactored implementations in parallel (numerical equivalence), or should the refactored version fully replace tests?
