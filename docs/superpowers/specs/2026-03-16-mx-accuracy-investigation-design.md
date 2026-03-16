# MX Format Accuracy Investigation and Improvement

**Date:** 2026-03-16
**Status:** Draft
**Goal:** Diagnose why MXFP4_S1 attention produces degraded end-to-end CogVideoX video quality, and develop techniques to improve MX format accuracy.

---

## Problem Statement

The Triton standalone implementation (`standalone/sageattention3_standalone.py`) supports four quantization formats: NVFP4, MXFP4, MXFP4_S1, and MXFP8_S1. NVFP4 achieves high accuracy (~99%+ CosSim vs SDPA). However, MXFP4_S1 shows visible quality degradation in end-to-end CogVideoX video generation, even though per-layer CosSim may pass the relaxed 0.85 threshold in unit tests.

The MX formats differ from NVFP4 in two ways:
1. **E8M0 scales** (powers-of-2 only) vs NVFP4's E4M3 scales (fine-grained FP8) — up to 2x scale error per block
2. **Block size 32** vs NVFP4's block size 16 — fewer scales, wider range per block

Block size 32 is fixed per the OCP MX specification and will not be changed.

### Format Reference

| Format | Data type | Scale type | Block size | P quantization |
|--------|-----------|------------|------------|----------------|
| NVFP4 | E2M1 | E4M3 | 16 | Two-level (FP32 global + E4M3 micro) |
| MXFP4 | E2M1 | E8M0 | 32 | Two-level (FP32 global + E8M0 micro) |
| MXFP4_S1 | E2M1 | E8M0 | 32 | Single-level (E8M0 micro only) |
| MXFP8_S1 | E4M3 | E8M0 | 32 | Single-level (E8M0 micro only) |

Note: MXFP4_S1 and MXFP8_S1 both use single-level P quantization. MXFP4 uses two-level. When comparing MXFP8_S1 vs MXFP4_S1, we isolate data precision (E4M3 vs E2M1) while holding P quantization structure constant (both single-level). When comparing MXFP4 vs MXFP4_S1, we isolate P quantization structure (two-level vs single-level) while holding data precision constant.

## Investigation Strategy

**MXFP8 first, then MXFP4.** MXFP8_S1 uses E4M3 data (high precision) with E8M0 scales and single-level P quantization. Comparing MXFP8_S1 vs MXFP4_S1 isolates the data precision variable (E4M3 vs E2M1) while keeping everything else the same (E8M0 scales, block=32, single-level P). If MXFP8_S1 also has accuracy problems, the core issue is E8M0 scale precision and/or block=32. If MXFP8_S1 works well, the E2M1 data precision is the bottleneck for MXFP4_S1.

To also isolate the P quantization level count, Phase 1 includes an MXFP4 (two-level) experiment alongside MXFP4_S1 (single-level).

---

## Phase 1: Per-GEMM Ablation (Diagnosis)

### Design

Attention has two GEMMs:
- **GEMM1 (QK):** `S = Q @ K^T` — uses quantized Q and K
- **GEMM2 (PV):** `O = P @ V` — uses quantized P and V

Each GEMM can independently run in **MX mode** (quantized) or **BF16 mode** (full precision). This gives a 2x2 ablation matrix:

| Mode | GEMM1 (QK) | GEMM2 (PV) | What it isolates |
|------|------------|------------|------------------|
| 0 | bf16 | bf16 | Baseline (reference) |
| 1 | mx | bf16 | GEMM1 error only |
| 2 | bf16 | mx | GEMM2 error only |
| 3 | mx | mx | Full MX (current) |

Run this matrix with three MX formats: **MXFP8_S1**, **MXFP4_S1**, and **MXFP4** (two-level).

Full experiment matrix: 3 formats x 3 non-baseline modes + 1 shared baseline = 10 experiments.

(The baseline Mode 0 is format-independent — run once and reuse.)

### Implementation

Add a `quant_components` parameter to `scaled_dot_product_attention` in `sageattention3_standalone.py`:
- `quant_components={"QK", "PV"}` (default, current behavior)
- `quant_components={"QK"}` — only GEMM1 uses MX, GEMM2 runs in BF16
- `quant_components={"PV"}` — only GEMM2 uses MX, GEMM1 runs in BF16
- `quant_components=set()` — baseline, all BF16

**Implementation note for GEMM1 BF16 path (Mode 2):** Currently, Q and K are always pre-quantized before the Triton kernel in `sage3_attention_standalone`. There is no kernel-level flag to skip QK quantization. Implementing Mode 2 requires:
1. Passing raw BF16 Q/K tensors to the kernel (bypassing `fp4_quantize_standalone`)
2. Skipping Q/K smoothing and delta_s computation (irrelevant when Q/K are full precision)
3. Adding a new kernel `tl.constexpr` flag (e.g., `QUANTIZE_QK`) to gate the GEMM1 quantization path and use `tl.dot` on raw BF16 inputs instead

When QK is not in `quant_components`: pass raw BF16 Q/K to the kernel, skip smoothing/delta_s, compute GEMM1 via BF16 `tl.dot`.
When PV is not in `quant_components`: skip P quantization and V quantization, compute GEMM2 via BF16 `tl.dot`.

### Measurements

For each experiment:
1. **Per-layer CosSim** vs baseline (Mode 0)
2. **Cumulative CosSim** (output after all layers up to L vs baseline up to L)
3. **Final-step latent CosSim** — `CosSim(latent_mx, latent_sdpa)` at the last denoising step. This is deterministic, fast, and already runnable without additional tooling.

E2E VQA scores (using EvalCrafter/DOVER) are deferred to Phase 3 where full multi-frame generation is performed. Per-layer and latent CosSim are sufficient for Phase 1 diagnosis.

### Analysis

Key questions the ablation answers:
- Which GEMM dominates the accuracy loss?
- Is MXFP8_S1 accuracy acceptable? (If yes → E2M1 data precision is the bottleneck)
- Does MXFP4 (two-level) improve over MXFP4_S1 (single-level)? (If yes → P quantization structure matters for MX)
- Are errors additive or compounding? (Compare Mode 3 loss vs sum of Mode 1 + Mode 2 losses. If Mode 3 loss > 1.2x the sum, there is significant interaction between the GEMMs.)

---

## Phase 2: Targeted Improvement Techniques

Based on Phase 1 results, apply the relevant techniques below.

### 2a. Stochastic E8M0 Rounding

**Applicable to:** Both GEMMs (always worth trying).

**Current behavior:** `ceil(log2(x))` — deterministic, rounds UP to next power of 2 when x is not already a power of 2. This systematically overestimates scales, underestimates quantized values. (When x is already a power of 2, ceil is exact and introduces no error.)

**Proposed change:** Probabilistic rounding in log-space:
```python
log2_val = log2(abs_scale)
floor_exp = floor(log2_val)       # integer exponent, rounds DOWN
frac = log2_val - floor_exp       # fractional part in [0, 1)
# Stochastic: pick floor_exp with prob (1-frac), (floor_exp+1) with prob frac
exp = floor_exp if random() > frac else (floor_exp + 1)
rounded_scale = 2 ** exp          # final power-of-2 scale value
```

**Expected benefit:** Makes quantization unbiased in expectation. Across 50 denoising steps and 30 layers, removing the systematic ceil bias should reduce error accumulation.

**Risk:** Adds per-block randomness. Run each experiment with 5 different seeds to measure variance. Compare mean and worst-case latent CosSim vs deterministic ceil.

### 2b. Two-Level P Quantization Redesign for E8M0

**Applicable to:** GEMM2 (PV), if Phase 1 shows PV dominates error.

**Current state:** MXFP4_S1 and MXFP8_S1 use single-level P quantization (no global scale). MXFP4 uses two-level with `COMBINED_MAX = 448 * 6 = 2688`, inherited from NVFP4.

**Problem:** The NVFP4 `COMBINED_MAX` was tuned for E4M3 microscales. With E8M0 microscales (powers-of-2 grid), the optimal value is different.

**Motivation for sweep values:** P's row-max after unnormalized softmax typically lies in (0, 1]. The global scale `row_max / COMBINED_MAX` maps this into the microscale's quantizable range. For E8M0 microscales (powers-of-2 only), the ideal `COMBINED_MAX` should be a power-of-2 multiple of `fp_max`, so that the global scale itself lands near a clean fraction. This minimizes the compounding of two coarse quantization steps.

**Proposed experiments:** Sweep `COMBINED_MAX` with power-of-2 multiples of `fp_max`:

For MXFP8 (fp_max = 448):
- `COMBINED_MAX` in `{448, 448*2, 448*4, 448*8, 448*16, 448*32}`

For MXFP4 (fp_max = 6):
- `COMBINED_MAX` in `{6, 6*4, 6*16, 6*64, 6*256, 2688}`

Measure per-layer CosSim for each value. If no clear optimum is found within these ranges, extend the sweep by another 2-3 values in the direction of improvement.

### 2c. V Smoothing

**Applicable to:** GEMM2 (PV), if Phase 1 shows PV dominates error.

**Technique:** Mean-subtract V before quantization, correct post-attention:
```python
v_mean = v.mean(dim=-2, keepdim=True)  # [B, H, 1, D]
v_smooth = v - v_mean
v_quant = mx_quantize(v_smooth)
# After attention: output += attn_weights_sum * v_mean
```

**Why it helps for MX:** E8M0's coarse scale is especially wasteful when V has outlier channels. Smoothing reduces V's dynamic range within each block=32 group, so the power-of-2 scale covers the data more tightly.

**Cost:** One reduction + broadcast per layer. Negligible.

---

## Phase 3: Denoising Step Error Propagation (E2E Validation)

After finding the best techniques from Phase 2, validate that they fix the E2E quality issue.

### 3a. Step-by-Step Divergence Tracking

Run CogVideoX with both SDPA and the best MX configuration (same seed). Capture intermediate latent tensors at each denoising step. Plot `CosSim(latent_mx[t], latent_sdpa[t])` over t = 0..49.

This shows whether error grows linearly (manageable) or exponentially (compounding).

### 3b. Selective Step Quantization

Use MX attention for only a subset of denoising steps, SDPA for the rest:
- First half MX, second half SDPA
- First half SDPA, second half MX
- Every other step MX

Compare final latent CosSim and VQA scores.

**Decision rule:** If "first half MX + second half SDPA" scores significantly better than "first half SDPA + second half MX", the later (detail-refining) denoising steps are more sensitive to quantization error, and improvement efforts should focus on reducing error in those steps (e.g., per-step adaptive precision). The reverse result would indicate early steps are sensitive. If alternating scores similarly to both halves, errors distribute uniformly.

### 3c. Per-Layer Mixed Precision

Use Phase 1 ablation results to identify "hard" layers (lowest CosSim). Run those layers with SDPA, rest with MX. This is also a practical deployment strategy — most of the MX speedup with maintained quality.

### 3d. Full E2E VQA Evaluation

Run full CogVideoX video generation (49 frames) with the best MX configuration from Phase 2. Evaluate using the existing EvalCrafter pipeline (DOVER/Q-Align VQA metrics). Compare against SDPA baseline and NVFP4.

---

## Execution Order

```
Phase 1: Ablation (diagnose)
  1.1 Add quant_components control to standalone kernel (incl. new kernel flag)
  1.2 Run 2x2 ablation with MXFP8_S1 (4 modes)
  1.3 Run 2x2 ablation with MXFP4_S1 (4 modes)
  1.4 Run 2x2 ablation with MXFP4 two-level (4 modes)
  1.5 Analyze results → identify dominant error source and format interactions

Phase 2: Targeted improvements (fix)
  2a Stochastic E8M0 rounding (always worth trying)
  2b Two-level P quant redesign for E8M0 (if PV dominates)
  2c V smoothing (if PV dominates)
  2d Re-run ablation with each fix to measure impact

Phase 3: Denoising propagation (validate E2E)
  3a Step-by-step divergence tracking
  3b Selective step quantization
  3c Per-layer mixed precision (if needed)
  3d Full E2E VQA evaluation
```

**Decision points:**
- After Phase 1.5: If MXFP8_S1 shows good accuracy but MXFP4_S1 doesn't → E2M1 data precision is the bottleneck.
- After Phase 1.5: If MXFP8_S1 also shows poor accuracy → E8M0 scales are the bottleneck. Focus on 2a, 2b.
- After Phase 1.5: If MXFP4 (two-level) is significantly better than MXFP4_S1 (single-level) → P quantization structure matters. Focus on 2b.
- After Phase 2d: Pick the best combination and validate with full E2E generation in Phase 3.

---

## Success Criteria

1. **Ablation isolates per-GEMM contributions:** Mode 0 matches SDPA exactly. Mode 1 + Mode 2 individual losses sum to within 20% of Mode 3 combined loss (confirming errors are approximately additive and the framework is working correctly).
2. **Latent-space accuracy target:** The best technique combination achieves final-step latent CosSim > 0.95 vs SDPA baseline for MXFP8. For MXFP4, latent CosSim > 0.90.
3. **MXFP4 viability decision:** A documented conclusion: either "MXFP4 is viable for E2E generation with technique X (specify which)" or "MXFP4 is not viable; MXFP8 is the practical floor for MX-format attention quality."
4. **E2E validation (Phase 3):** VQA scores (DOVER/Q-Align) for the best MX configuration are within 2% of the SDPA baseline on the CogVideoX benchmark set.
