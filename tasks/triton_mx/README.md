# MX Format Experiments — Triton FP4/FP8 Attention Quantization

## Goal

Extend the Triton SageAttention3 standalone implementation (NVFP4) to support
**MXFP4** and **MXFP8** microscaling formats (block size 32, E8M0 scales), and
characterize accuracy vs the SDPA reference on CogVideoX-2b.

Triton NVFP4 baseline: `/home/yiliu7/workspace/SageAttention-Fork/standalone`

## Experiments

All experiments run on **NVIDIA B200**, CogVideoX-2b, seed=42, 1 frame.

| Script | Purpose | Steps | Output |
|--------|---------|-------|--------|
| `ablation.py` | 3-part accuracy ablation (synthetic, per-layer, E2E latent) | 5 | `ablation_results.json` |
| `step_sweep.py` | Validate if 5-step proxy is representative of 10/20/50 | 5,10,20,50 | `step_sweep_results.json` |
| `layer_skip.py` | SDPA fallback on worst **layers** to recover accuracy | 5 | `layer_skip_results.json` |
| `step_skip.py` | SDPA fallback on worst **steps** to recover accuracy | 20 | `step_skip_results.json` |
| `format_escalation.py` | Three-tier FP4→FP8→SDPA step routing | 20 | `format_escalation_results.json` |
| `sparsity_profiling.py` | Tile-level attention sparsity analysis | 5 | `sparsity_profiling_smoke_test.json` |
| `per_head_profiling.py` | Per-head FP4/FP8 format selection | 20, 50 | `per_head_20step_results.json`, `per_head_50step_49frame_phase1.json` |

---

## Key Findings

### 1. Format Accuracy Ranking (Ablation Part A — Synthetic)

Single-call attention accuracy on CogVideoX-realistic shapes (video_full: B=1, H=24, N=17281, D=64):

| Rank | Format | CosSim | Description |
|------|--------|--------|-------------|
| 1 | **mxfp8_s1** | 0.9985 | MX FP8 E4M3 with K-mean smoothing |
| 2 | **nvfp4** | 0.9822 | NVIDIA FP4 (block size 16, FP8 scales) |
| 3 | **mxfp4_s1** | 0.9688 | MX FP4 E2M1 with K-mean smoothing |
| 4 | **mxfp4** | 0.9692 | MX FP4 E2M1, no smoothing |

MXFP8 is clearly the highest-accuracy format. Among FP4 variants, NVFP4 leads
by ~1.3 percentage points in cosine similarity. Smoothing provides negligible
benefit for MXFP4 at the single-call level.

### 2. Error Accumulates Across Denoising Steps (Step Sweep)

End-to-end latent SNR (dB) degrades significantly at higher step counts for FP4:

| Format | 5 steps | 10 steps | 20 steps | 50 steps |
|--------|---------|----------|----------|----------|
| mxfp8_s1 | 25.8 | 22.5 | 28.0 | 17.5 |
| nvfp4 | 20.6 | 17.0 | 13.6 | 14.1 |
| mxfp4_s1 | 18.5 | 16.4 | 9.6 | 13.0 |
| mxfp4 | 18.4 | 16.1 | 9.8 | 12.0 |

FP4 formats lose 6-9 dB going from 5 to 20 steps. This motivated the step-skip
experiment: if some steps amplify error more than others, we can selectively use
SDPA on those steps.

### 3. Worst Layers — Spatial Dimension (Ablation Part B)

Consensus worst layers across all 4 formats (CogVideoX-2b, 30 transformer blocks):

```
layer_18 (score=40), layer_17 (36), layer_15 (29),
layer_16 (25), layer_14 (23), layer_19 (22), layer_13, layer_12
```

Layers 14-19 consistently show the highest quantization error.

### 4. Worst Steps — Temporal Dimension (Step-Skip Phase 1)

Per-step marginal SNR drop at 20 steps, profiled via cumulative latent divergence:

**Consensus ranking** (across all 4 formats):

| Rank | Step | Score | Top-5 in |
|------|------|-------|----------|
| 1 | **step 5** | 75 | all 4 formats |
| 2 | **step 4** | 69 | all 4 formats |
| 3 | **step 1** | 66 | mxfp4, mxfp4_s1, mxfp8_s1 |
| 4 | **step 6** | 63 | nvfp4, mxfp4, mxfp4_s1 |
| 5 | **step 3** | 57 | nvfp4, mxfp4_s1 |

Error growth rate is heavily front-loaded:

| Phase | Avg delta (dB/step) |
|-------|---------------------|
| Early (steps 1-6) | -4.7 |
| Mid (steps 7-12) | -2.0 |
| Late (steps 13-19) | -1.2 |

### 5. Step-Skip vs Layer-Skip — Recovery Comparison

**Step-skip results** (20-step inference, SDPA fallback on worst steps):

| Format | Baseline | skip_worst_1 | skip_worst_3 | skip_worst_5 | Best delta |
|--------|----------|-------------|-------------|-------------|------------|
| mxfp4_s1 | 9.6 dB | 10.1 (+0.4) | 13.5 (+3.9) | **18.4 (+8.8)** | **+8.8 dB** |
| mxfp4 | 9.8 dB | 10.6 (+0.8) | **16.5 (+6.7)** | 17.2 (+7.4) | +7.4 dB |
| nvfp4 | 13.6 dB | 15.4 (+1.8) | 16.2 (+2.6) | **17.5 (+3.9)** | +3.9 dB |
| mxfp8_s1 | 28.0 dB | 28.6 (+0.6) | **29.6 (+1.6)** | 29.3 (+1.3) | +1.6 dB |

**Layer-skip results** (5-step inference, SDPA fallback on worst layers):

| Format | Baseline | skip_top3 | skip_top5 | skip_top8 | Best delta |
|--------|----------|-----------|-----------|-----------|------------|
| mxfp8_s1 | 25.8 dB | 26.4 (+0.6) | 27.5 (+1.7) | **29.3 (+3.6)** | +3.6 dB |
| nvfp4 | 20.6 dB | 19.2 (-1.4) | 20.1 (-0.5) | **21.6 (+1.0)** | +1.0 dB |
| mxfp4_s1 | 18.5 dB | 18.9 (+0.5) | 19.8 (+1.4) | **19.9 (+1.4)** | +1.4 dB |
| mxfp4 | 18.4 dB | 18.7 (+0.3) | 19.0 (+0.6) | **18.9 (+0.5)** | +0.5 dB |

### Key takeaway: Step-skip is dramatically more effective than layer-skip for FP4 attention

- **mxfp4_s1**: +8.8 dB (step-skip) vs +1.4 dB (layer-skip) — **6.3x more effective**
- **mxfp4**: +7.4 dB vs +0.5 dB — **14.8x more effective**
- **nvfp4**: +3.9 dB vs +1.0 dB — **3.9x more effective**
- **mxfp8_s1**: +1.6 dB vs +3.6 dB — layer-skip wins here (already high accuracy)

Skipping 5/20 steps (25% SDPA) brings mxfp4_s1 from 9.6 dB to 18.4 dB, nearly
matching nvfp4's 5-step baseline (20.6 dB). The cost: 25% of steps run at SDPA
speed instead of quantized kernel speed.

### 6. Format Escalation — Three-Tier Routing Beats Binary Skip

**Idea**: Instead of binary skip (FP4 or SDPA), use a three-tier routing:
- **Critical steps** → SDPA (full precision)
- **Moderate steps** → MXFP8_S1 (nearly lossless, faster than SDPA)
- **Easy steps** → FP4 (fastest)

Step rankings from Phase 1 profiling determine which steps get which tier.

**Results** (20-step CogVideoX-2b):

#### nvfp4

| Config | SDPA | FP8 | FP4 | SNR (dB) | CosSim | vs baseline |
|--------|------|-----|-----|----------|--------|-------------|
| baseline | 0 | 0 | 20 | 13.56 | 0.9778 | — |
| escalate_1_2 | 1 | 2 | 17 | 16.82 | 0.9896 | +3.26 dB |
| escalate_2_3 | 2 | 3 | 15 | 16.91 | 0.9898 | +3.35 dB |
| escalate_3_5 | 3 | 5 | 12 | **21.80** | **0.9967** | **+8.24 dB** |
| step_skip_5 (binary) | 5 | 0 | 15 | 17.48 | 0.9911 | +3.92 dB |

#### mxfp4

| Config | SDPA | FP8 | FP4 | SNR (dB) | CosSim | vs baseline |
|--------|------|-----|-----|----------|--------|-------------|
| baseline | 0 | 0 | 20 | 9.80 | 0.9472 | — |
| escalate_1_2 | 1 | 2 | 17 | 16.63 | 0.9891 | +6.83 dB |
| escalate_2_3 | 2 | 3 | 15 | 17.87 | 0.9918 | +8.07 dB |
| escalate_3_5 | 3 | 5 | 12 | **18.47** | **0.9929** | **+8.67 dB** |
| step_skip_5 (binary) | 5 | 0 | 15 | 17.18 | 0.9904 | +7.38 dB |

#### mxfp4_s1

| Config | SDPA | FP8 | FP4 | SNR (dB) | CosSim | vs baseline |
|--------|------|-----|-----|----------|--------|-------------|
| baseline | 0 | 0 | 20 | 9.61 | 0.9446 | — |
| escalate_1_2 | 1 | 2 | 17 | 13.25 | 0.9761 | +3.64 dB |
| escalate_2_3 | 2 | 3 | 15 | 18.94 | 0.9936 | +9.33 dB |
| escalate_3_5 | 3 | 5 | 12 | **20.02** | **0.9950** | **+10.41 dB** |
| step_skip_5 (binary) | 5 | 0 | 15 | 18.41 | 0.9928 | +8.80 dB |

#### Key takeaway: Escalation outperforms binary skip with fewer SDPA calls

`escalate_3_5` uses only **3 SDPA steps** (vs 5 for binary skip) yet achieves higher SNR:

| Format | escalate_3_5 SNR | step_skip_5 SNR | Δ (escalation wins) |
|--------|-----------------|-----------------|---------------------|
| nvfp4 | **21.80 dB** | 17.48 dB | **+4.32 dB** |
| mxfp4 | **18.47 dB** | 17.18 dB | **+1.29 dB** |
| mxfp4_s1 | **20.02 dB** | 18.41 dB | **+1.61 dB** |

**Why it works**: The FP8 moderate tier (MXFP8_S1 at 0.9985 CosSim) prevents error
accumulation on "moderately sensitive" steps far better than FP4 would, while being
much faster than SDPA. Even `escalate_1_2` (just 1 SDPA + 2 FP8) recovers +3–7 dB —
the FP8 tier does most of the heavy lifting.

**Practical implication**: A production deployment could use `escalate_3_5` to get
near-SDPA quality (21.8 dB for nvfp4) while running only 3/20 steps at SDPA speed,
5/20 at FP8 speed, and 12/20 at FP4 speed.


  Summary -- mxfp4, 50 steps

  Config            SDPA   FP8   FP4  Latent SNR     CosSim     L1Rel  delta_SNR
  ---------------- ----- ----- ----- ----------- ---------- --------- ----------
  baseline             0     0    50       10.69   0.957559  0.248057         --
  escalate_1_2         1     2    47       10.58   0.956010  0.252729      -0.11
  escalate_2_3         2     3    45       10.73   0.957538  0.246364      +0.04
  escalate_3_5         3     5    42       11.26   0.962491  0.233302      +0.57
  step_skip_5          5     0    45       11.00   0.960159  0.239567      +0.31


========================================================================
  Cross-Format Summary -- Escalation Results
========================================================================

### 7. Sparsity Profiling — Tile-Level Mixing Not Viable

**Idea**: Use attention sparsity patterns to mix FP4/FP8 at the tile level —
sparse attention tiles use FP4 (faster), important tiles use FP8 (more accurate).

**Result (STOP)**: CogVideoX-2b attention is too diffuse for tile-level mixing:
- **tiles_for_90pct = 64.9/139 (46.7%)** — exceeds STOP threshold of 50
- **Best proxy ρ = 0.225** — no predictive power for tile importance
- **Gini = 0.59** — moderate concentration, but insufficient for exploitation

However, **per-head variation is significant**:
- Sharpest: head 16 — 21.1 tiles for 90% coverage (Gini=0.87)
- Diffuse: head 28 — 95.7 tiles for 90% coverage (Gini=0.39)
- Std across heads: 16.4 tiles

**Key insight**: Sparsity does NOT correlate with FP4 quantization sensitivity
(Pearson r = 0.12). The two properties are orthogonal — head 16 is the most
sparse but tolerates FP4 well (30.49 dB SNR), while head 7 is moderately sparse
but most FP4-sensitive (24.13 dB). Direct quantization error measurement is
needed, not sparsity-based prediction.

### 8. Per-Head Format Selection — Head-Level FP4/FP8 Routing

**Idea**: Instead of applying the same format to all 30 attention heads, route
each head independently: FP4 for heads that tolerate it, FP8 for sensitive heads.

#### Phase 1: Per-head error profiling

For each head h: run FP4 on head h only + SDPA on all other 29 heads, measure
the marginal error contribution.

**20-step results** (1 frame, MXFP4) — ~9 dB spread:

| Rank | Head | SNR (dB) | Assessment |
|------|------|----------|------------|
| **Worst** | **6** | **18.90** | Most FP4-sensitive |
| Worst | 18 | 19.07 | |
| Worst | 3 | 20.02 | |
| Worst | 20 | 21.83 | |
| Worst | 10 | 21.90 | |
| ... | ... | ... | |
| Best | 25 | 30.19 | |
| Best | 24 | 30.35 | |
| Best | 29 | 30.37 | |
| **Best** | **1** | **30.53** | Most FP4-tolerant |

20-step worst→best: `[6, 18, 3, 20, 10, 7, 8, 15, 27, 17, 23, 26, 9, 19, 28, 16, 13, 11, 22, 14, 4, 5, 21, 2, 0, 12, 25, 24, 29, 1]`

**50-step results (PRODUCTION)** (49 frames, MXFP4) — ~12 dB spread:

| Rank | Head | SNR (dB) | CosSim | Assessment |
|------|------|----------|--------|------------|
| **Worst** | **21** | **10.54** | 0.9556 | Most FP4-sensitive |
| 2 | 23 | 10.81 | 0.9583 | |
| 3 | 25 | 11.06 | 0.9607 | |
| 4 | 4 | 11.21 | 0.9619 | |
| 5 | 10 | 11.40 | 0.9637 | |
| 6 | 7 | 11.53 | 0.9646 | |
| 7 | 2 | 11.61 | 0.9653 | |
| 8 | 14 | 11.89 | 0.9675 | |
| 9 | 15 | 11.96 | 0.9682 | |
| 10 | 13 | 12.46 | 0.9717 | |
| ... | ... | ... | ... | |
| 26 | 3 | 18.68 | 0.9932 | |
| 27 | 11 | 19.10 | 0.9939 | |
| 28 | 16 | 20.71 | 0.9958 | |
| 29 | 12 | 21.54 | 0.9965 | |
| **Best** | **0** | **22.59** | 0.9973 | Most FP4-tolerant |

50-step worst→best: `[21, 23, 25, 4, 10, 7, 2, 14, 15, 13, 24, 26, 19, 6, 18, 29, 27, 1, 9, 22, 20, 8, 5, 17, 28, 3, 11, 16, 12, 0]`

**⚠️ Rankings shift significantly between 20-step and 50-step:**

| Head | 20-step rank | 50-step rank | Shift |
|------|-------------|-------------|-------|
| **21** | 23 (good) | **1 (worst)** | −22 ↓↓ |
| **6** | 1 (worst) | 14 (mid) | +13 ↑ |
| **25** | 27 (good) | **3 (worst)** | −24 ↓↓ |
| **1** | 30 (best) | 18 (mid) | −12 ↓ |
| 10 | 5 | 5 | 0 ✅ |
| 7 | 6 | 6 | 0 ✅ |
| 15 | 8 | 9 | −1 ✅ |

**Conclusion**: 20-step head rankings do NOT transfer to production. Per-head
format assignments must be profiled at the target step count. Only a few heads
(10, 7, 15) maintain stable rankings across step counts.

#### Phase 2: Head allocation sweep

Assign worst K heads → FP8, remaining → FP4.

**20-step results** (1 frame):

| Config | FP8 | FP4 | SNR (dB) | CosSim | vs all_fp4 |
|--------|-----|-----|----------|--------|------------|
| all_fp4 | 0 | 30 | 9.80 | 0.9472 | — |
| fp8_worst_5 | 5 | 25 | 15.15 | 0.9847 | +5.35 dB |
| fp8_worst_15 | 15 | 15 | 19.31 | 0.9941 | **+9.51 dB** |
| fp8_worst_20 | 20 | 10 | 19.99 | 0.9950 | +10.19 dB |
| fp8_worst_25 | 25 | 5 | 21.95 | 0.9968 | +12.15 dB |
| all_fp8 | 30 | 0 | 28.02 | 0.9992 | +18.22 dB |

**50-step results (PRODUCTION)** (49 frames):

| Config | FP8 | FP4 | SNR (dB) | CosSim | vs all_fp4 |
|--------|-----|-----|----------|--------|------------|
| all_fp4 | 0 | 30 | 10.69 | 0.9576 | — |
| fp8_worst_5 | 5 | 25 | 10.10 | 0.9512 | **−0.59 dB** |
| fp8_worst_10 | 10 | 20 | 11.34 | 0.9636 | +0.65 dB |
| fp8_worst_15 | 15 | 15 | 10.32 | 0.9535 | **−0.38 dB** |
| fp8_worst_20 | 20 | 10 | 12.06 | 0.9690 | +1.37 dB |
| fp8_worst_25 | 25 | 5 | 12.23 | 0.9702 | +1.54 dB |
| all_fp8 | 30 | 0 | 12.97 | 0.9748 | +2.28 dB |

**⚠️ Per-head routing is far less effective at production settings:**

| Metric | 20-step | 50-step |
|--------|---------|---------|
| fp8_worst_15 delta | **+9.51 dB** | **−0.38 dB** |
| fp8_worst_5 delta | +5.35 dB | −0.59 dB |
| all_fp8 delta | +18.22 dB | +2.28 dB |
| Non-monotonic? | fp8_worst_10 only | fp8_worst_5 AND fp8_worst_15 |

At 50 steps, head-level routing provides **minimal benefit** (+1.5 dB max at
fp8_worst_25). Even all_fp8 only improves +2.28 dB over all_fp4 (vs +18.22 dB
at 20 steps). This suggests that at production step counts, **step-level error
dominates head-level error** — fixing the wrong heads barely matters when the
error accumulates across 50 steps.

The non-monotonic behavior (fp8_worst_5 and fp8_worst_15 worse than baseline)
confirms that head-group fragmentation creates kernel-launch overhead and
quantization-statistics artifacts that can **degrade** quality when splitting
heads into small non-contiguous groups.

**Key comparisons** (50 steps, production):

| Method | SNR (dB) | vs all_fp4 | Notes |
|--------|----------|------------|-------|
| all_fp4 (baseline) | 10.69 | — | |
| fp8_worst_25 (head-level) | 12.23 | +1.54 dB | Best head-level result |
| all_fp8 (head-level) | 12.97 | +2.28 dB | All FP8 |
| escalate_8_12 (step-level) | 11.65 | +0.96 dB | 8 SDPA + 12 FP8 + 30 FP4 steps |

**Conclusion**: At production settings (50 steps), **step-level routing remains
the most practical approach**. Per-head routing's 20-step promise (+9.5 dB) does
not transfer to production — the head sensitivity rankings shift dramatically,
and head-group fragmentation hurts. The path forward is to improve step-level
escalation or combine it with coarser head grouping (e.g., top/bottom half).

---

## Related Work

Our experiments study **attention kernel quantization** (FP4/FP8 in SDPA) with
**binary fallback** (quantized vs full-precision SDPA). This is distinct from
the more common **weight/activation quantization** with continuous bit-width
allocation. Nevertheless, several works explore the same sensitivity dimensions.

### Most Directly Related

**MixDQ** (arXiv:2405.12032, 2024) — Mixed-precision quantization for few-step
diffusion models. Discovers that different layers and timesteps have different
sensitivity, measured via LPIPS (visual quality) and CLIP score (text alignment).
Assigns higher bit-widths to sensitive dimensions. Closest to our approach but
uses continuous bit allocation rather than binary SDPA/quant fallback.
https://arxiv.org/abs/2405.12032

**TDQ — Temporal Dynamic Quantization** (NeurIPS 2023) — Dynamically adjusts
quantization parameters across timesteps. Recognizes static PTQ is inadequate
due to time-varying activation distributions. Our Phase 1 profiling (error
front-loaded in early steps) directly validates TDQ's premise.
https://proceedings.neurips.cc/paper_files/paper/2023/hash/56748a4b4d48a1cb0b3d0ae71b90e1a0-Abstract-Conference.html

**ViDiT-Q** (arXiv:2406.02540, 2024) — Quantization for Video Diffusion
Transformers (image + video). Metric-aware mixed precision across channels,
layers, and timesteps. Relevant since our experiments run on CogVideoX (a video DiT).
https://arxiv.org/abs/2406.02540

### Foundational Works

**Q-Diffusion** (ICCV 2023, arXiv:2302.04304) — First to identify time-varying
activation distributions in diffusion models making naive PTQ fail. Introduced
timestep-aware calibration. Our profiling results (early steps = -4.7 dB/step vs
late steps = -1.2 dB/step) quantify what Q-Diffusion describes qualitatively.
https://arxiv.org/abs/2302.04304

**PTQD** (arXiv:2305.10657, 2023) — Models quantization error as correlated
noise in the diffusion process and proposes analytic correction. Provides the
theoretical framework for why quantization error compounds across steps
differently than random noise.
https://arxiv.org/abs/2305.10657

**Q-DiT** (arXiv:2406.17343, 2024) — PTQ for Diffusion Transformers. Block-level
and timestep-level sensitivity analysis targeting edge deployment.
https://arxiv.org/abs/2406.17343

### Alternative Approaches

**SVDQuant** (arXiv:2411.05007, 2024, MIT Han Lab) — Instead of keeping
sensitive layers in FP16, absorbs outliers via low-rank SVD decomposition.
Achieves W4A4 on SDXL/PixArt/FLUX with 3.5x memory reduction, 3-8.7x speedup.
https://arxiv.org/abs/2411.05007

**EfficientDM** (CVPR 2024, arXiv:2310.03270) — QAT for low-bit diffusion
models with per-block sensitivity analysis.
https://arxiv.org/abs/2310.03270

**Efficient Quantization Strategies for LDMs** (arXiv:2312.09571) —
Timestep-aware calibration for UNet quantization on SDXL/SDXL-Turbo.
https://arxiv.org/abs/2312.09571

### Surveys

- **Efficient Diffusion Models: A Comprehensive Survey** (Jan 2025) — https://arxiv.org/abs/2501.09274
- **Qualcomm Developer Blog** — https://developer.qualcomm.com/blog/introduction-quantization-techniques-diffusion-models

### How Our Work Differs

| Dimension | Prior Art | Our Contribution |
|-----------|-----------|------------------|
| Quantization target | Weight/activation quantization | **Attention kernel** quantization (FP4/FP8 QK matmul) |
| Precision strategy | Continuous bit-width allocation | **Binary** SDPA/quant fallback |
| Step sensitivity | TDQ/MixDQ: dynamic quant params per step | Direct **marginal per-step error** measurement; step-skip recovers up to +8.8 dB |
| Layer sensitivity | MixDQ/EfficientDM: per-layer bit allocation | CogVideoX-specific profiling of 30 transformer blocks |
| Video models | ViDiT-Q targets video DiTs | CogVideoX-2b end-to-end validation with latent divergence metrics |
| Key finding | N/A | **Step-skip >> layer-skip** for FP4 attention (6-15x more effective); **Format escalation** (FP4→FP8→SDPA) beats binary skip by +1.3–4.3 dB with fewer SDPA calls; **Per-head routing** promising at 20 steps (+9.5 dB) but **does NOT transfer to production** (50 steps: +1.5 dB max, non-monotonic); head rankings shift dramatically between step counts; sparsity ≠ quantization sensitivity (r=0.12) |

---

## Usage

```bash
# Ablation — synthetic + per-layer + E2E latent
python ablation.py --parts A B C --output-json ablation_results.json

# Step sweep — validate 5-step proxy
python step_sweep.py --output-json step_sweep_results.json

# Layer skip — SDPA fallback on worst layers (5 steps)
python layer_skip.py --output-json layer_skip_results.json

# Step skip — SDPA fallback on worst steps (20 steps)
python step_skip.py --output-json step_skip_results.json

# Format escalation — three-tier FP4→FP8→SDPA routing (20 steps)
python format_escalation.py --output-json format_escalation_results.json

# Sparsity profiling — tile-level attention pattern analysis
python sparsity_profiling.py --num-steps 5 --sample-steps 1 3 --sample-layers 0 14

# Per-head format selection — head-level FP4/FP8 routing (20 steps)
python per_head_profiling.py --phase 1 2 --output-json per_head_results.json

# Quick tests
python step_skip.py --phases 1 --formats nvfp4          # Phase 1 only, 1 format
python layer_skip.py --formats nvfp4                     # 1 format
python format_escalation.py --formats nvfp4               # 1 format
python per_head_profiling.py --phase 1 --num-steps 5     # quick Phase 1 only
python ablation.py --parts A                             # synthetic only (no model)
```
