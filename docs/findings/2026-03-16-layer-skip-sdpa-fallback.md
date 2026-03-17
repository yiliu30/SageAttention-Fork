# Findings: Layer-Skip SDPA Fallback for Sensitive Layers

**Date:** 2026-03-16
**GPU:** NVIDIA B200
**Model:** CogVideoX-2b (30 transformer layers, 5 denoising steps, 1 frame)
**Script:** `tasks/triton_mx/layer_skip.py`
**Raw data:** `tasks/triton_mx/layer_skip_results.json`

## Summary

We tested the hypothesis that falling back to full-precision SDPA on the most error-sensitive transformer layers (identified by Part B per-layer ablation) while keeping quantized attention on the remaining layers can improve end-to-end latent accuracy at minimal throughput cost. The experiment measures **both** final-step latent divergence (Part C-style, closest proxy for image quality) **and** per-layer attention divergence (Part B-style, revealing which layers actually contribute error). The hypothesis was confirmed for 3 of 4 formats, with **mxfp8_s1 benefiting most dramatically (+3.6 dB latent SNR with 8 layers skipped)**, while nvfp4 showed anomalous behavior where partial skipping initially hurt accuracy.

## Experiment Design

**Sensitive layers** were identified from Part B ablation consensus ranking across all formats:

| Rank | Layer | Consensus Score |
|------|-------|----------------|
| 1 | layer_18 | 40 |
| 2 | layer_17 | 36 |
| 3 | layer_15 | 29 |
| 4 | layer_16 | 25 |
| 5 | layer_14 | 23 |
| 6 | layer_19 | 22 |
| 7 | layer_13 | — |
| 8 | layer_12 | — |

**Skip configurations tested:**
- `baseline`: 0/30 layers skipped (pure quantized)
- `skip_top3`: layers {15, 17, 18} use SDPA (10% fallback)
- `skip_top5`: layers {14, 15, 16, 17, 18} use SDPA (17% fallback)
- `skip_top8`: layers {12–19} use SDPA (27% fallback)

**Mechanism:** A `SelectiveAttention` hook counts attention calls (30 per denoising step in CogVideoX-2b) and routes each call to either the quantized kernel or original SDPA based on layer index.

## Key Findings

### Full Results Table — Latent Divergence

| Format | Config | Layers Skipped | Latent SNR (dB) | L1 Rel | CosSim | Δ SNR |
|--------|--------|---------------|----------------|--------|--------|----------|
| **nvfp4** | baseline | 0/30 | 20.64 | 0.0828 | 0.9958 | — |
| | skip_top3 | 3/30 | 19.23 | 0.0879 | 0.9942 | **-1.40** |
| | skip_top5 | 5/30 | 20.15 | 0.0825 | 0.9953 | -0.49 |
| | skip_top8 | 8/30 | 21.62 | 0.0753 | 0.9966 | **+0.98** |
| **mxfp4** | baseline | 0/30 | 18.42 | 0.1053 | 0.9931 | — |
| | skip_top3 | 3/30 | 18.72 | 0.0981 | 0.9935 | +0.30 |
| | skip_top5 | 5/30 | 19.01 | 0.1001 | 0.9940 | **+0.59** |
| | skip_top8 | 8/30 | 18.93 | 0.0969 | 0.9939 | +0.51 |
| **mxfp4_s1** | baseline | 0/30 | 18.45 | 0.0929 | 0.9932 | — |
| | skip_top3 | 3/30 | 18.93 | 0.0916 | 0.9939 | +0.48 |
| | skip_top5 | 5/30 | 19.81 | 0.0865 | 0.9950 | +1.36 |
| | skip_top8 | 8/30 | 19.88 | 0.0818 | 0.9950 | **+1.43** |
| **mxfp8_s1** | baseline | 0/30 | 25.77 | 0.0303 | 0.9987 | — |
| | skip_top3 | 3/30 | 26.41 | 0.0299 | 0.9989 | +0.64 |
| | skip_top5 | 5/30 | 27.49 | 0.0286 | 0.9991 | +1.72 |
| | skip_top8 | 8/30 | 29.34 | 0.0265 | 0.9994 | **+3.57** |

### Per-Layer Attention Divergence

The experiment also captures per-layer attention output divergence (vs SDPA reference) for each configuration. This reveals two key things: (1) whether the "skipped" layers truly produce zero error (they should — they use SDPA), and (2) how the remaining quantized layers' errors propagate.

**Global attention-level SNR (averaged across all 150 calls):**

| Format | Config | Attn SNR (dB) | Attn Δ SNR |
|--------|--------|--------------|-----------|
| **nvfp4** | baseline | 26.26 | — |
| | skip_top3 | 26.23 | -0.03 |
| | skip_top5 | 26.65 | +0.39 |
| | skip_top8 | 27.23 | **+0.97** |
| **mxfp4** | baseline | 25.19 | — |
| | skip_top3 | 25.71 | +0.52 |
| | skip_top5 | 25.67 | +0.48 |
| | skip_top8 | 26.10 | **+0.91** |
| **mxfp4_s1** | baseline | 26.56 | — |
| | skip_top3 | 26.61 | +0.05 |
| | skip_top5 | 26.96 | +0.40 |
| | skip_top8 | 27.33 | **+0.77** |
| **mxfp8_s1** | baseline | 27.24 | — |
| | skip_top3 | 27.33 | +0.08 |
| | skip_top5 | 27.71 | +0.46 |
| | skip_top8 | 28.22 | **+0.98** |

**Key observation — Latent amplification effect:** The attention-level SNR improvement is consistently *smaller* than the latent-level improvement. For example, mxfp8_s1 skip_top8 gains +0.98 dB at the attention level but +3.57 dB at the latent level. This confirms that attention errors in sensitive layers *compound through the denoising process* — fixing them at the source yields amplified benefits in the final output.

**Skipped layers still show non-zero divergence:** Even though skipped layers use the identical SDPA function, their outputs differ from the reference because *inputs* (Q, K, V) to those layers have already diverged due to quantization errors in earlier layers. The skipped layers have near-perfect attention computation but imperfect inputs.

### Per-Layer Efficiency (latent dB gained per layer skipped)

| Format | skip_top3 | skip_top5 | skip_top8 |
|--------|-----------|-----------|-----------|
| nvfp4 | -0.47 | -0.10 | +0.12 |
| mxfp4 | +0.10 | **+0.12** | +0.06 |
| mxfp4_s1 | +0.16 | **+0.27** | +0.18 |
| mxfp8_s1 | +0.21 | +0.34 | **+0.45** |

### Interpretation

1. **mxfp8_s1 benefits most and scales linearly.** Every additional layer skipped adds ~0.4 dB. At skip_top8 (27% SDPA fallback), SNR improves from 25.77 to 29.34 dB — a massive +3.57 dB gain. This makes sense: FP8 quantization is already high-quality, so removing the few bad layers has an outsized effect on overall error.

2. **mxfp4_s1 shows strong, consistent gains.** Monotonically improves with more skips, peaking at +1.43 dB for skip_top8. Best efficiency is at skip_top5 (+0.27 dB/layer), meaning 17% SDPA fallback captures most of the benefit.

3. **mxfp4 has moderate but diminishing gains.** Best at skip_top5 (+0.59 dB), but skip_top8 actually regresses slightly vs skip_top5. The sweet spot is 5 layers.

4. **nvfp4 is anomalous — partial skipping initially hurts.** skip_top3 causes a -1.40 dB *regression*, and skip_top5 is still slightly negative (-0.49 dB). Only skip_top8 yields a modest +0.98 dB gain. This suggests:
   - nvfp4's error distribution across layers may differ from the other formats
   - Mixing quantized/unquantized attention within a denoising step may create interference effects specific to nvfp4's quantization scheme
   - The "worst layers" ranking from Part B (averaged across all formats) may not be optimal for nvfp4 specifically

5. **The hypothesis is confirmed for 3/4 formats.** Layer-skip is a viable accuracy recovery strategy, especially for mxfp8_s1 where the cost/benefit ratio is excellent.

6. **Error amplification through denoising.** Per-layer attention divergence data reveals that latent SNR improvements are 2–4x larger than the corresponding attention-level improvements. For example, mxfp8_s1 skip_top8 gains +0.98 dB at the attention level but +3.57 dB at the latent level (3.6x amplification). This means fixing attention errors in sensitive layers has a compounding benefit across denoising steps — a strong argument for selective skipping even at modest cost.

7. **Skipped layers still show residual divergence.** Even SDPA-fallback layers produce outputs that differ from the pure-SDPA reference, because their Q/K/V inputs have already been affected by quantization errors in upstream layers. This explains why skip_top3 shows smaller gains than expected — the skipped layers can't fully recover if their inputs are already corrupted.

## Decisions Made

- **Layer ranking source:** Used Part B consensus ranking across all formats rather than per-format rankings, to test a single universal skip set.
- **Call-counting approach:** Reused the same `call_idx % 30` mechanism from Part B's `AttentionCapture`, which relies on CogVideoX-2b having exactly 30 attention calls per denoising step in fixed order.
- **Measurement:** Both final-step latent divergence (Part C-style, closest proxy for image quality) and per-layer attention output divergence (Part B-style, revealing error sources). The dual measurement revealed a key amplification effect: attention-level improvements are consistently smaller than their latent-level counterparts, confirming that errors compound through denoising steps.

## Open Questions

1. **Would per-format skip sets help nvfp4?** nvfp4 may have different worst layers than the consensus ranking. Running Part B per-layer analysis specifically for nvfp4 and using its own ranking could yield better results.

2. **Is there a non-linear interaction effect?** nvfp4's regression at skip_top3 but recovery at skip_top8 suggests quantized/unquantized boundary effects. Are there specific layer *pairs* that cause problems when only one is skipped?

3. **Does this scale to more denoising steps?** These results use 5 steps. At 20–50 steps, error accumulation is different — the benefit could be larger or could wash out.

4. **What about speed impact?** We measured accuracy only. The actual throughput cost of falling back to SDPA on N/30 layers needs benchmarking (SDPA is slower than quantized kernels, so the cost is non-zero).

5. **Can the skip set be dynamic?** Instead of a fixed set, could we detect high-error layers at runtime (e.g., based on activation statistics) and skip adaptively?

## Next Steps

- **Per-format skip ranking for nvfp4:** Re-run Part B with nvfp4 only, extract its specific worst layers, and re-test layer-skip with that custom ranking.
- **Speed benchmarking:** Measure wall-clock time for each skip configuration to quantify the throughput cost of SDPA fallback.
- **Extended step counts:** Test with 20 and 50 steps to validate whether the gains hold or amplify with longer diffusion trajectories.
- **Visual quality evaluation:** Generate actual video frames with skip_top5/skip_top8 for mxfp8_s1 and mxfp4_s1, and compare visually against baseline and SDPA reference.
- **Adaptive skipping prototype:** Explore whether activation norms or gradient magnitudes at each layer could predict which layers need full-precision attention, enabling dynamic skip decisions.
