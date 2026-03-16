# Accuracy Improvement Ideas — FP4/FP8 Attention Quantization

Date: 2026-03-16
Status: Brainstorming catalog — no implementation yet

## Context

Based on experiments in `tasks/triton_mx/` on CogVideoX-2b (B200 GPU),
we have established baseline accuracy and two recovery mechanisms:

- **Step-skip** (SDPA fallback on worst denoising steps): up to +8.8 dB SNR
- **Layer-skip** (SDPA fallback on worst transformer layers): up to +3.6 dB SNR
- Step-skip is 4-15x more effective than layer-skip for FP4

Format ranking: mxfp8_s1 (0.9985 CosSim) >> nvfp4 (0.9822) > mxfp4_s1 (0.9688) ~ mxfp4 (0.9692)

This document catalogs additional ideas to improve accuracy, organized by
category with feasibility/impact estimates.

---

## Category 1: Quantization Granularity & Smoothing (kernel-level)

Changes to *how* we quantize, within the existing kernel framework.

### 1a. Per-Head Adaptive Format Selection

**Idea**: Some attention heads may tolerate FP4 while others need FP8. At
inference time, cheaply estimate per-head dynamic range (e.g., max absolute
value of Q and K), and route each head to FP4 vs FP8 independently.

**Evidence**: Ablation Part A shows per-head SNR spread of ~2-3 dB in synthetic
tests. Worst heads consistently underperform the average.

**Mechanism**:
- Before quantization, compute `max(|Q[:,h,:,:]|)` and `max(|K[:,h,:,:]|)` per head
- If either exceeds a calibrated threshold, use MXFP8 for that head; otherwise FP4
- Both kernel variants are already compiled, just need per-head dispatch

**Impact**: Medium. Targets the ~2-3 dB worst-to-best head spread.
**Feasibility**: Medium. Requires per-head dispatch logic. Shapes are uniform
across heads so the kernel itself is unchanged, but the calling code needs to
split heads into FP4 and FP8 groups and run two kernels.
**Cost**: Per-head max computation (~negligible), plus overhead of two kernel
launches instead of one when mixed formats are needed.

### 1b. V Smoothing for FP4 PV Path

**Idea**: The SageAttention3 CUDA kernel applies V smoothing for the FP8 PV
path: V is mean-subtracted before quantization, and the correction
`mean(V) * sum(attn_weights)` is added back post-attention. Our Triton FP4
implementation does NOT do this — V is quantized raw.

**Evidence**: V quantization error flows directly into the output via the PV
matmul. V has no smoothing at all in our current Triton code (only Q and K
get smoothed via `apply_qk_smoothing_standalone`).

**Mechanism**:
```
v_mean = V.mean(dim=-2, keepdim=True)  # [B, H, 1, D]
v_centered = V - v_mean                # reduces V outliers
v_quant = fp4_quantize(v_centered)
output = attention(Q, K, v_quant)
output += v_mean                        # correction (since sum(softmax) = 1)
```

**Impact**: Potentially high. V quantization is currently unsmoothed, so this
is low-hanging fruit. The technique is proven in the FP8 PV path.
**Feasibility**: High. Already exists in the CUDA codebase — just port to Triton.
**Cost**: One mean computation + subtraction + post-attention add-back. Negligible.

### 1c. Finer Quantization Granularity

**Idea**: Currently we use per-block quantization (block_size=16 for NVFP4,
32 for MX). Finer granularity (per-warp or per-thread scale factors) produces
more accurate quantization at the cost of more scale storage and bandwidth.

**Evidence**: CLAUDE.md notes: "per_block > per_warp > per_thread (finer = more
accurate, slightly slower)." SageAttention2 showed meaningful accuracy gains
from finer granularity.

**Impact**: Medium-high based on SageAttention2 experience.
**Feasibility**: Low-medium for Triton. per_warp and per_thread are CUDA
abstractions that don't map cleanly to Triton's programming model. Would
require rethinking the scale factor layout.
**Cost**: More scale factors = more memory + memory bandwidth.

### 1d. Two-Level PV Accumulation as Speed/Accuracy Knob

**Idea**: SageAttention2++ uses `fp32+fp16` two-level accumulation: a short-term
FP16 accumulator is flushed to an FP32 buffer every few iterations. Our Triton
kernel accumulates in FP32 throughout. This gives us a speed knob: full FP32
for sensitive steps, fp32+fp16 for easy steps.

**Mechanism**: Combine with step-skip: on "easy" steps, use FP4 + fp32+fp16 accum
(fastest). On "medium" steps, use FP4 + FP32 accum. On "hard" steps, use SDPA.

**Impact**: Primarily a speed optimization, not accuracy. But enables finer
speed/accuracy tradeoff when combined with step-skip.
**Feasibility**: Medium. Triton supports mixed-precision accumulation.
**Cost**: Slight accuracy loss on easy steps (expected to be within noise).

---

## Category 2: Selective Fallback Strategies (system-level)

Building on step-skip/layer-skip findings to create smarter routing.

### 2a. Combined Step + Layer Skip

**Idea**: Step-skip gives +8.8 dB and layer-skip gives +3.6 dB independently.
Combining them: skip worst steps entirely (all 30 layers use SDPA), and within
non-skipped steps, also skip worst layers.

**Evidence**: The two dimensions are orthogonal — step sensitivity comes from
diffusion dynamics, layer sensitivity from transformer block structure.

**Mechanism**: Extend `SelectiveStepAttention` to also check layer index:
```python
step_idx = call_count // calls_per_step
layer_idx = call_count % calls_per_step
if step_idx in skip_steps or layer_idx in skip_layers:
    return sdpa(...)
return quant_attn(...)
```

**Impact**: Potentially high — could compound gains. Expected to exceed both
individual methods. Best case: most of step-skip (+8.8 dB) plus incremental
layer-skip gains (+1-2 dB) on non-skipped steps.
**Feasibility**: High. Trivial to implement — combine two existing mechanisms.
**Cost**: More SDPA calls = slower, but fine-grained control over the budget.

### 2b. Adaptive Per-Call Sensitivity Routing

**Idea**: Instead of static skip sets determined offline, dynamically decide
per attention call whether to use quantized or SDPA based on a cheap
"difficulty score" computed from the inputs.

**Candidate metrics**:
- `max(|Q|) * max(|K|)` — proxy for dynamic range (high = harder to quantize)
- kurtosis of Q/K distributions — peaky = more outliers = worse quantization
- Head dimension entropy — low entropy = concentrated, easier to quantize

**Mechanism**: Compute the metric before quantization. If above threshold, fall
back to SDPA. Threshold calibrated on a small set of representative inputs.

**Impact**: Could be optimal — only falls back when truly needed, no wasted SDPA.
**Feasibility**: Medium. Need to identify a metric that (a) correlates well with
actual quantization error and (b) is cheap enough that computing it doesn't
negate the speed benefit of quantization.
**Cost**: Per-call metric overhead (small if using max/kurtosis on existing tensors).

### 2c. Format Escalation Ladder (FP4 -> FP8 -> SDPA)

**Idea**: Instead of binary FP4-or-SDPA, introduce an intermediate rung:
FP4 -> MXFP8 -> SDPA. If a step/layer is "medium-sensitive," use MXFP8 instead
of falling all the way back to SDPA. Only truly bad cases get SDPA.

**Evidence**: MXFP8 achieves 0.9985 CosSim — nearly perfect — while being
much faster than SDPA. Many calls that "need" SDPA may actually be fine with
FP8.

**Mechanism**: Three-tier routing based on profiled sensitivity:
```
if step_idx in critical_steps:
    return sdpa(...)          # ~5% of calls
elif step_idx in moderate_steps:
    return fp8_attn(...)      # ~15% of calls
else:
    return fp4_attn(...)      # ~80% of calls
```

**Impact**: High. Gets most of the accuracy benefit of step-skip with much less
speed penalty, since FP8 is much faster than SDPA.
**Feasibility**: High. We already have MXFP8_S1 implemented and profiled. Just
need routing logic.
**Cost**: Two quantized kernel variants loaded, but both are already compiled.
Marginal overhead of format selection logic.

---

## Category 3: Preprocessing & Calibration (pre-inference)

Techniques applied before or around the attention kernel.

### 3a. Better Smoothing Strategies

**Problem**: Current smoothing (K-mean subtraction + Q per-block centering) barely
helps for MXFP4 — mxfp4 vs mxfp4_s1 are nearly identical (0.9692 vs 0.9688 CosSim).
The smoothing isn't targeting the right error source.

**Sub-ideas**:

**3a-i. Per-channel smoothing (SmoothQuant-style)**:
Multiply K by per-channel scales `s_d`, divide Q by the same scales:
`Q' = Q / s_d, K' = K * s_d`, where `s_d = max(|K[:,d]|)^alpha / max(|Q[:,d]|)^(1-alpha)`.
This equalizes dynamic range per channel (head_dim dimension), making quantization
more uniform.
- *Feasibility*: High. Simple preprocessing.
- *Impact*: Medium. Proven in LLM weight quantization (SmoothQuant).

**3a-ii. Outlier-aware smoothing**:
Replace mean subtraction with median or trimmed mean (exclude top/bottom 1% values).
Mean is sensitive to outliers; median is not. If a few extreme values dominate
the mean, current smoothing shifts everything toward the outlier direction.
- *Feasibility*: High. torch.median is available.
- *Impact*: Low-medium. Depends on outlier distribution.

**3a-iii. Learned per-head smoothing coefficients**:
Instead of uniform mean subtraction, calibrate optimal per-head smoothing
factors `alpha_h` on a small representative dataset: `Q'[:,h,:,:] = Q[:,h,:,:] - alpha_h * mean(Q[:,h,:,:])`.
- *Feasibility*: Medium. Needs a calibration pass.
- *Impact*: Medium. Personalizes smoothing to each head's characteristics.

### 3b. Rotation-Based Outlier Suppression (Hadamard / QuaRot)

**Idea**: Recent LLM quantization work (QuaRot, SpinQuant) applies random
Hadamard rotations to activations before quantization. This spreads outliers
uniformly across dimensions, making quantization error more uniform and reducing
worst-case error.

**Mechanism**:
```
H_d = hadamard_matrix(D)  # D=64, so 64x64 orthogonal matrix
Q_rot = Q @ H_d           # spreads outliers across dimensions
K_rot = K @ H_d           # same rotation
# Quantize Q_rot and K_rot to FP4
# Attention output is the same because Q_rot @ K_rot^T = Q @ H @ H^T @ K^T = Q @ K^T
```

Since H is orthogonal (H @ H^T = I), the QK^T product is mathematically
identical. But the quantization error is different — outliers that were
concentrated in one dimension are now spread across all dimensions, reducing
per-block max values and improving quantization fidelity.

**Evidence**: QuaRot showed 1-2 dB improvement for INT4 LLM quantization.
Head dimension D=64 means the Hadamard matrix is small and the transform is
very cheap.

**Impact**: Potentially high. Theoretically optimal for structured outlier
distributions. The D=64 case is especially favorable (small matrix).
**Feasibility**: Medium. The transform itself is trivial, but it changes the
quantization landscape and may interact with existing smoothing. Need to
verify it doesn't conflict with delta_s correction. Also, the Hadamard matrix
must be pre-computed once (it's deterministic for a given D).
**Cost**: Two matrix multiplies per attention call (Q @ H and K @ H),
but D=64 makes these very cheap: [B, H, N, 64] @ [64, 64].

---

## Priority Ranking (recommended experiment order)

| Priority | Idea | Impact | Feasibility | Rationale |
|----------|------|--------|-------------|-----------|
| **P0** | **1b. V smoothing** | High | High | Low-hanging fruit. Already proven in CUDA FP8 path. Zero-cost to try. |
| **P0** | **2c. Format escalation (FP4->FP8->SDPA)** | High | High | Leverages existing FP8 kernel. Much better speed/accuracy tradeoff than binary skip. |
| **P1** | **2a. Combined step+layer skip** | High | High | Trivial to implement, directly compounds two proven mechanisms. |
| **P1** | **3b. Hadamard rotation** | High | Medium | Theoretically sound, cheap for D=64. Could be a fundamental improvement. |
| **P2** | **1a. Per-head adaptive format** | Medium | Medium | Targets head-level variance but adds dispatch complexity. |
| **P2** | **3a-i. Per-channel smoothing** | Medium | High | Simple to try, but unclear if channel imbalance is the bottleneck. |
| **P3** | **2b. Adaptive per-call routing** | Optimal | Medium | Needs metric research. High reward if the right metric is found. |
| **P3** | **1c. Finer granularity** | Medium-high | Low-medium | Significant kernel changes in Triton. Better suited for CUDA. |
| **P3** | **1d. Two-level accum** | Speed knob | Medium | Not an accuracy improvement per se, but enables new tradeoff points. |
| **P4** | **3a-ii. Outlier-aware smoothing** | Low-medium | High | Easy to try, low expected impact. |
| **P4** | **3a-iii. Learned smoothing** | Medium | Medium | Needs calibration infrastructure. |
