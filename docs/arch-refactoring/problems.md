# Standalone Triton Kernel — Architecture Refactoring: Problems to Resolve

Scope: `standalone/sageattention3_standalone.py` (Triton kernel only, CUTE kernel untouched)

---

## P1: Adding a new quantization scheme requires touching too many places

Currently, adding a new scheme (e.g., INT8, FP8, new MXFP4 variant) requires:
1. Write torch + triton round/quant primitives
2. Write a new ~100-line P-quantization Triton kernel (copy-pasting 70% from an existing one)
3. Add a new `tl.constexpr` boolean flag to the attention kernel signature
4. Add a new branch to the if/elif chain inside the kernel (line 817-824)
5. Thread the flag through the launcher (`tiled_online_attention_triton`)
6. Thread it through the high-level API (`sageattn3_torch_triton_standalone`)
7. Add pre-processing logic (K/V quantization path) if the scheme needs different preprocessing
8. Register the `--attention_type` string in example scripts

**Impact**: ~6 files/functions touched, high risk of missing a step or introducing bugs in the copy-paste.

---

## P2: P-quantization kernels duplicate most of their logic

The 4 P-quant Triton functions share the same structure:
- Reshape tile into blocks
- Compute per-block max/scale
- Round the scale (differs by scheme)
- Clamp data to format range (differs by scheme)
- Dequantize back to float
- Optionally handle two-level vs single-level scales

| Kernel | Lines | Unique logic |
|--------|-------|-------------|
| `two_level_p_quantization_triton` (NVFP4) | ~130 | E4M3 scale rounding, two-level scales |
| `two_level_p_quantization_mxfp4_triton` | ~90 | E8M0 scale rounding, two-level scales |
| `single_level_p_quantization_mxfp4_triton` | ~80 | E8M0 scale rounding, single-level |
| `single_level_p_quantization_mxfp8_triton` | ~80 | E8M0 scale rounding, E4M3 data quant, single-level |

Only the scale rounding, data clamping, and one-vs-two-level logic differ. ~70% of each kernel is shared boilerplate.

**Impact**: Bug fixes or optimizations to the shared pattern must be applied 4+ times. Easy to have subtle divergences.

---

## P3: No clean extension point for pre-quantization transforms

Techniques like Hadamard rotation need to run on K/V *before* quantization. Currently, pre-processing (smoothing, K/V quantization) is interleaved in `sageattn3_torch_triton_standalone` with no clear hook point. Adding rotation means inserting it into the middle of a 100-line function and hoping it doesn't break the existing flow.

**Impact**: Experimental transforms are hard to plug in/out cleanly, making A/B comparisons messy.

---

## P4: The attention kernel's constexpr dispatch doesn't scale

The attention kernel uses boolean constexpr flags (`USE_MXFP4`, `USE_MXFP4_S1`, `USE_MXFP8_S1`) to select which P-quant function to call. Each new scheme adds another flag and another branch. With N schemes, this becomes N booleans and an N-way if/elif.

**Impact**: Triton compiles a separate kernel for each combination of constexpr values. The flag approach works for 4 schemes but gets unwieldy at 8+.

---

## P5: Single 1400-line file hinders navigation and independent testing

Everything lives in one file: primitives, format config, 4 P-quant kernels, the attention kernel, preprocessing, and 3 entry points. There's no way to test P-quant logic without importing the attention kernel, or to iterate on preprocessing without the full pipeline.

**Impact**: Slower development iteration; harder to reason about changes in isolation.

---

## Must-Have New Schemes

The refactored architecture must support all of these without major restructuring:

| Scheme | Data format | Scale format | Notes |
|--------|------------|-------------|-------|
| **INT8** | 8-bit integer | per-block or per-warp | For Q/K or P quantization |
| **INT4** | 4-bit integer | per-block | Lower precision alternative to FP4 |
| **FP8** | E4M3 or E5M2 | per-block | Standalone (not microscaling) — differs from MXFP8 which uses E8M0 shared exponents |
| **Rounding modes** | — | — | ceil, floor, RNE (round-to-nearest-even) as configurable options for any format |

Combined with existing schemes (NVFP4, MXFP4, MXFP4_S1, MXFP8_S1), this means **8+ total schemes** — the constexpr bool-per-scheme approach (P4) will definitely not scale.

Rounding modes are cross-cutting: they apply to any format's scale rounding or data quantization step. The architecture should treat rounding as a composable parameter, not a per-scheme fork.

---

## Priority Order

1. **P1** — Easy to add new schemes (primary goal)
2. **P3** — Pre-quantization transform hooks (needed for rotation experiments)
3. **P2** — Reduce P-quant duplication (enables P1)
4. **P4** — Constexpr dispatch scaling (critical at 8+ schemes)
5. **P5** — File splitting (natural consequence of solving P1-P4)
