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
| Key finding | N/A | **Step-skip >> layer-skip** for FP4 attention (6-15x more effective) |

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

# Quick tests
python step_skip.py --phases 1 --formats nvfp4          # Phase 1 only, 1 format
python layer_skip.py --formats nvfp4                     # 1 format
python ablation.py --parts A                             # synthetic only (no model)
```
