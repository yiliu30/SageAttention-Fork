# Sparsity-Aware Mixed-Precision Attention

**Date:** 2026-03-17
**Status:** Draft
**Goal:** Profile real attention sparsity patterns in CogVideoX-2b, then design a tile-level FP4/FP8 mixed-precision attention kernel that routes precision based on attention importance.

---

## Problem Statement

MXFP4 attention achieves only ~10.69 dB SNR (CosSim 0.957) at 50 denoising steps on CogVideoX-2b. Format escalation at the denoising-step level improves this modestly — `escalate_3_5` achieves 11.26 dB (+0.57 dB over baseline at 50 steps). But step-level granularity is coarse: it treats all 30 transformer blocks within a step identically, and all ~19,000 tile-pairs within each attention call identically.

**Hypothesis:** Attention matrices are inherently sparse — most softmax weights are near-zero. Within a single attention call, only a fraction of KV-tiles carry significant attention mass. If we route important tiles to FP8 and unimportant tiles to FP4, we can approach FP8 accuracy at mostly-FP4 speed.

### Architecture Context

CogVideoX-2b has **30 transformer blocks** (layers). Each block makes **one** `F.scaled_dot_product_attention` call per denoising step. Each call operates on all **30 attention heads** simultaneously in the H dimension:

```
Attention tensor shape: (B=2, H=30, N=17776, D=64)
                         │     │      │       └─ head dimension
                         │     │      └─ sequence length (226 text + 17550 video tokens)
                         │     └─ 30 attention heads (within each call)
                         └─ batch (2 for guidance_scale > 1)
```

- **30 attention calls per denoising step** = 30 transformer blocks (layers), NOT 30 heads
- Each call processes all 30 heads as the H dimension of the tensor
- With tile_size=128: N/128 = 139 tiles per dimension → ~19,321 tile pairs per head per call
- 50 denoising steps × 30 layers = 1,500 total attention calls per video

The `call_count` tracking (used by `SelectiveStepAttention`, `EscalationAttention`) maps to:
- `step_idx = call_count // 30` (which denoising step)
- `layer_idx = call_count % 30` (which transformer block within the step)

### Research Direction

Two-level precision mixing:
1. **Tile-level (within one kernel):** Sparse tiles → FP4, important tiles → FP8
2. **Step/layer-level (existing escalation):** Compose with existing step routing

Head balancing: control the FP4/FP8 tile budget per head so aggregate compute is predictable.

---

## Phase 0: Sparsity Profiling (this spec)

Before building any kernel, we need ground-truth measurements of attention sparsity in CogVideoX-2b. This phase is pure analysis — no kernel changes.

### What We Measure

#### Measurement 1: Tile-Level Attention Mass Distribution

For each attention call in the sample set, compute the full attention weight matrix `A = softmax(Q @ K^T / sqrt(d))` and aggregate at tile granularity. Statistics are computed **per query row** within each Q-tile, not by summing across rows (since each query row's softmax independently sums to 1):

```python
T = 128  # tile size
num_tiles = ceil(N / T)  # 139 for N=17776

# For each query row q_idx within Q-tile i, compute per-KV-tile mass:
row_tile_mass[q_idx, j] = A[q_idx, j*T:(j+1)*T].sum()  # mass of KV-tile j for this query

# Then aggregate per Q-tile (average across rows in the tile):
tile_mass[i, j] = mean over q_idx in [i*T..(i+1)*T] of row_tile_mass[q_idx, j]
tile_max[i, j]  = A[i*T:(i+1)*T, j*T:(j+1)*T].max()
```

Sparsity is measured **relative to the uniform baseline**. With 139 KV-tiles, uniform attention gives each tile 1/139 ≈ 0.72% of row mass. Raw percentage thresholds (<1%, <5%) would always read ~100% "sparse" and are meaningless.

**Metrics reported:**
- **Relative sparsity:** fraction of tiles carrying <0.1× uniform (< 0.072% mass), <0.5× uniform (< 0.36% mass)
- **Cumulative coverage curve (primary metric):** how many KV-tiles (sorted by mass, descending) are needed to cover 90%, 95%, 99% of total attention per query row. Reported as a fraction of total tiles (e.g., "top 15% of tiles cover 90% of attention mass").
- **Gini coefficient:** single-number sparsity summary (0 = perfectly uniform, 1 = maximally concentrated). Computed per query row, averaged.
- **Cross-dimension consistency:** is sparsity stable across heads, layers, steps, or highly variable?

#### Measurement 2: Per-Head Sparsity Variation

For each head h, compute the effective number of important KV-tiles per query row:

```python
# Per query row q_idx, per head h:
# Sort KV-tile masses descending, find k such that cumsum >= 0.9 * total
tiles_for_90pct[h, q_idx] = argmin_k { cumsum(sorted_masses[:k]) >= 0.9 * total_mass }
```

This quantifies head "sharpness" — sharp heads have few important tiles (high sparsity, good FP4 candidates), diffuse heads have many (need FP8).

Summary statistics:
- Per-head: mean/median/std of tiles_for_90pct across query rows
- Head ranking: sharpest → most diffuse
- Variation within a head vs between heads

#### Measurement 3: Cheap Proxy Accuracy for Tile Importance

Test whether we can predict tile importance without computing the full attention matrix. Three candidate proxies (computed from Q and K tiles before softmax):

| Proxy | Formula | Cost |
|-------|---------|------|
| A: max-product | `max(\|Q_tile\|) * max(\|K_tile\|)` | 2 reductions per tile-pair |
| B: norm-product | `\|\|Q_tile\|\|_F * \|\|K_tile\|\|_F` | 2 norms per tile-pair |
| C: sampled dot | `max(Q_tile[::S] @ K_tile[::S].T)`, S=16 | Small matmul (8×D)×(D×8) per pair |

For each proxy, measure:
- **Spearman rank correlation (ρ)** with actual per-row tile mass (averaged across rows)
- **Precision@k:** if we pick the top-20% tiles by proxy score, what fraction of actual top-20% mass tiles do we catch?
- **ROC AUC** for the binary classification "tile is in top-k% by mass" (tests whether a single threshold can separate important from unimportant)
- **Threshold stability:** does the optimal threshold vary significantly across calls/heads/layers? Report std of optimal threshold across the sample set.
- **False negative rate at fixed FP8 budget:** if we allocate the top-20% of tiles to FP8 using the proxy, what fraction of truly-important tiles (top-20% by actual mass) do we miss?

Threshold for usability: ρ > 0.8, precision@20% > 0.85, ROC AUC > 0.9 means the proxy is usable for kernel routing.

#### Measurement 4: Quantization Error Concentration

Does FP4 quantization error concentrate in the important tiles? This determines whether tile-level mixing actually helps accuracy.

FP4 quantization in SageAttention3 is applied to **Q and K** (the QK matmul), not V. The output error is:
```
O_error = A_sdpa @ V - A_fp4 @ V = (A_sdpa - A_fp4) @ V
```
The error comes entirely from the quantized attention weights `A_fp4 ≠ A_sdpa`.

**Primary approach — QK-level error per tile:**

```python
# Compute pre-softmax scores at both precisions
S_sdpa = (Q @ K^T) / sqrt(d)                             # float16 full precision
S_fp4  = (Q_fp4_dequant @ K_fp4_dequant^T) / sqrt(d)     # FP4 quantized then dequantized

# Per tile QK error (Frobenius norm)
qk_error[i, j] = ||S_sdpa[i*T:(i+1)*T, j*T:(j+1)*T] - S_fp4[i*T:(i+1)*T, j*T:(j+1)*T]||_F
```

Correlate `tile_mass` with `qk_error`:
- If high-mass tiles have proportionally higher QK error → tile-level mixing is maximally valuable (FP8 exactly where errors are worst AND where they matter most)
- If error is uniform across tiles → tile-level mixing still helps (FP8 reduces error on tiles that contribute most to the output), but the benefit comes from output weighting rather than error concentration

**Metric:** Error concentration ratio = mean(qk_error in top-20% mass tiles) / mean(qk_error in bottom-80% mass tiles). Values > 1.5 indicate meaningful concentration.

### Sampling Strategy

Computing the full attention matrix is expensive: `O(B × N² × D)` per head. We can't profile all 1,500 calls.

**Sample selection:**
- **Steps:** 5 denoising steps — [1, 5, 10, 25, 49] (early/mid/late, covering the known sensitivity spectrum from step-skip profiling)
- **Layers:** 6 transformer blocks — [0, 7, 14, 18, 21, 28] (evenly spaced + layer 18, the worst layer from per-layer profiling in README.md; this ensures the known-hard range layers 14-19 has two representatives)
- **Heads:** All 30 heads (processed one at a time within each call for memory management)
- **Total calls sampled:** 5 steps × 6 layers = 30 calls, each profiling 30 heads

**Memory management:**

Full attention matrix per head: `B × N × N × sizeof(dtype)`.
- `scores` must be computed in **float32** for numerically stable softmax: `2 × 17776² × 4 bytes ≈ 2.4 GB`
- `attn` after softmax can stay in float32: `2 × 17776² × 4 bytes ≈ 2.4 GB`
- **Peak per head (scores + attn alive simultaneously): ~4.8 GB**

To reduce peak memory, process one batch element at a time:
- `scores_b = qh[b] @ kh[b].T`: `17776² × 4 ≈ 1.2 GB`
- `attn_b = softmax(scores_b)`: `17776² × 4 ≈ 1.2 GB`
- **Peak per (head, batch): ~2.4 GB** — comfortably fits on B200 (192 GB) or 5090 (32 GB)

Store only tile-level aggregated statistics (139×139 per head = ~77 KB per call-head), not the raw A matrix. Total stored data across all samples: 30 calls × 30 heads × ~77 KB ≈ ~69 MB.

**Implementation approach:**
- Hook `F.scaled_dot_product_attention`
- On sampled calls: compute A manually via `torch.matmul` + softmax, extract tile stats, then call original SDPA for the actual output
- On non-sampled calls: pass through to original SDPA directly
- Track `call_count` to derive `step_idx = call_count // calls_per_step` and `layer_idx = call_count % calls_per_step`
- Detect `calls_per_step` empirically during a warmup pass (same approach as `format_escalation.py`) rather than hardcoding 30
- Explicitly cast scores to float32 before softmax: `scores = torch.matmul(qh.float(), kh.float().transpose(-1, -2)) * scale`

### Output

**File:** `tasks/triton_mx/sparsity_profiling_results.json`

Structure:
```json
{
  "config": {
    "model": "cogvideox-2b",
    "num_steps": 50,
    "num_frames": 49,
    "tile_size": 128,
    "seq_len": 17776,
    "num_tiles": 139,
    "seed": 42,
    "sampled_steps": [1, 5, 10, 25, 49],
    "sampled_layers": [0, 7, 14, 18, 21, 28],
    "calls_per_step": 30,
    "uniform_baseline_mass": 0.00719
  },
  "summary": {
    "mean_tiles_for_90pct": 0.0,
    "mean_tiles_for_90pct_frac": 0.0,
    "mean_gini": 0.0,
    "frac_tiles_lt_half_uniform": 0.0,
    "frac_tiles_lt_tenth_uniform": 0.0,
    "sharpest_head": 0,
    "most_diffuse_head": 0,
    "best_proxy": "norm_product",
    "best_proxy_rho": 0.0,
    "best_proxy_roc_auc": 0.0,
    "best_proxy_threshold_std": 0.0,
    "error_concentration_ratio": 0.0
  },
  "per_call": [
    {
      "step": 1,
      "layer": 0,
      "per_head": {
        "0": {
          "gini": 0.0,
          "frac_lt_half_uniform": 0.0,
          "frac_lt_tenth_uniform": 0.0,
          "mean_tiles_for_90pct": 0.0,
          "median_tiles_for_90pct": 0.0,
          "tiles_for_95pct": 0.0,
          "tiles_for_99pct": 0.0,
          "proxy_rho_max_product": 0.0,
          "proxy_rho_norm_product": 0.0,
          "proxy_rho_sampled_dot": 0.0,
          "proxy_precision_at_20_max_product": 0.0,
          "proxy_precision_at_20_norm_product": 0.0,
          "proxy_precision_at_20_sampled_dot": 0.0,
          "proxy_roc_auc_max_product": 0.0,
          "proxy_roc_auc_norm_product": 0.0,
          "proxy_roc_auc_sampled_dot": 0.0
        }
      }
    }
  ],
  "aggregated_by_head": {
    "0": {
      "mean_gini": 0.0,
      "mean_tiles_for_90pct": 0.0,
      "mean_frac_lt_half_uniform": 0.0
    }
  },
  "aggregated_by_step": {
    "1": {"mean_gini": 0.0, "mean_tiles_for_90pct": 0.0}
  },
  "aggregated_by_layer": {
    "0": {"mean_gini": 0.0, "mean_tiles_for_90pct": 0.0}
  }
}
```

**Printed summary:**
```
Attention Sparsity Report — CogVideoX-2b
==========================================

Tile-Level Sparsity (T=128, 139 tiles/dim, uniform=0.72%/tile):
  Mean tiles for 90% coverage: XX.X / 139 (XX.X%)
  Mean tiles for 95% coverage: XX.X / 139 (XX.X%)
  Mean Gini coefficient: X.XXX (0=uniform, 1=max concentration)
  XX.X% of tiles carry <0.5× uniform mass (< 0.36%)
  XX.X% of tiles carry <0.1× uniform mass (< 0.072%)

Per-Head Variation:
  Sharpest:  head XX — XX tiles for 90%, Gini=X.XX
  Diffuse:   head XX — XX tiles for 90%, Gini=X.XX
  Std across heads: XX.X tiles

Per-Step Variation:
  Step  1: mean XX tiles for 90% (Gini=X.XX)
  Step 49: mean XX tiles for 90% (Gini=X.XX)

Per-Layer Variation:
  Layer  0: mean XX tiles for 90% (Gini=X.XX)
  Layer 18: mean XX tiles for 90% (Gini=X.XX)

Proxy Accuracy (ρ / P@20% / ROC AUC / threshold σ):
  max-product:  ρ=X.XX / P@20=X.XX / AUC=X.XX / σ=X.XX
  norm-product: ρ=X.XX / P@20=X.XX / AUC=X.XX / σ=X.XX
  sampled-dot:  ρ=X.XX / P@20=X.XX / AUC=X.XX / σ=X.XX

Error Concentration:
  QK error in top-20% mass tiles: X.XX× vs bottom-80%
  → Error IS / IS NOT concentrated in important tiles

Conclusion: [GO / EXPLORE / CAUTION / STOP] for tile-level mixed precision
```

### Decision Criteria

Sparsity is measured by **tiles_for_90pct** — the number of KV-tiles (out of 139) needed to cover 90% of attention mass per query row. Lower = sparser = more FP4 opportunity.

| Finding | Decision |
|---------|----------|
| tiles_for_90pct < 30 (< 22% of tiles) AND best proxy ρ > 0.8, AUC > 0.9 | ✅ **GO** — strong sparsity, predictable. Build tile-level mixed kernel |
| tiles_for_90pct < 30 AND proxy ρ 0.6-0.8 | ⚠️ **EXPLORE** — sparsity exists but proxy needs refinement. Try combining proxies or using static maps |
| tiles_for_90pct 30-50 (22-36%) AND proxy ρ > 0.8 | ⚠️ **EXPLORE** — moderate sparsity. Tile-level mixing helps but FP8 budget is larger. May combine with per-head routing |
| tiles_for_90pct 30-50 AND proxy ρ < 0.6 | ⚠️ **CAUTION** — moderate sparsity, hard to predict. Pivot to per-head format selection (idea 1a from ideas.md) |
| tiles_for_90pct > 50 (> 36%) | ❌ **STOP** for tile-level. Attention is too diffuse. Pivot to per-head or per-layer routing only |
| Error concentration ratio > 2.0 | ✅ Extra confirmation — FP8 on important tiles maximally valuable |
| Error concentration ratio < 1.2 | ⚠️ Tile-level mixing helps via output weighting, but less than expected |
| Proxy threshold σ > 0.3 × mean threshold | ⚠️ Threshold varies too much across calls — may need per-head or per-layer thresholds |

---

## Future Phases (after profiling)

### Phase 1: Tile-Level Mixed-Precision Kernel (if GO)

Design a Triton kernel where the inner KV-tile loop branches on a per-tile importance decision:
- Prediction pass: compute cheap proxy for each KV-tile
- Route: proxy > threshold → load FP8-quantized K/V; else → load FP4-quantized K/V
- Rest of online attention (softmax rescaling, PV accumulation) unchanged

Key kernel design question: how to handle the two quantization formats within the same tile loop without excessive branching overhead. Options:
- `tl.where`-based inline branching (simple, some warp divergence)
- Two-pass: first pass FP4 on all tiles, second pass re-compute important tiles at FP8 (no branching, but redundant work)
- Sorted-tile: sort tiles by importance, process FP4 batch then FP8 batch (best throughput, complex bookkeeping)

### Phase 2: Head Balancing

Set a per-head FP8 tile budget (e.g., 25% of tiles). Heads with more sparsity use fewer FP8 tiles. The budget can be:
- Fixed globally (simple)
- Adaptive per-layer based on profiled sensitivity (uses layer-skip data from README.md)
- Adaptive per-step (uses step-skip data)

### Phase 3: Integration with Step/Layer Escalation

Combine tile-level mixing with the existing format_escalation.py framework:
- Critical steps: full SDPA (no change)
- Moderate steps: FP8 everywhere (no change)
- Easy steps: tile-level FP4/FP8 mixing (new)

This creates a three-level hierarchy: step → layer → tile.

---

## Implementation: `sparsity_profiling.py`

**Location:** `tasks/triton_mx/sparsity_profiling.py`

**Dependencies:** Reuses `ablation.py` utilities (`_load_cogvideox_pipe`, `_run_pipe`, `build_sage_fn`, `print_header`).

**Usage:**
```bash
cd tasks/triton_mx

# Full profiling (50 steps, 5 sampled steps × 6 layers × 30 heads)
python sparsity_profiling.py --output-json sparsity_profiling_results.json

# Quick smoke test (5 steps, fewer samples)
python sparsity_profiling.py --num-steps 5 --sample-steps 1 3 --sample-layers 0 14

# Include quantization error concentration (slower)
python sparsity_profiling.py --measure-error --output-json sparsity_profiling_results.json
```

**Key classes:**

```python
class SparsityProfiler:
    """Hooks F.scaled_dot_product_attention to capture tile-level
    attention statistics on sampled calls.

    Architecture note: CogVideoX-2b has 30 transformer blocks (layers),
    each making one attention call per denoising step. Each call has shape
    (B=2, H=30, N=17776, D=64) — all 30 heads are in the H dimension.
    call_count maps to: step_idx = call_count // calls_per_step,
    layer_idx = call_count % calls_per_step.
    """

    def __init__(self, sample_steps, sample_layers, tile_size=128,
                 calls_per_step=None):
        # calls_per_step auto-detected if None (via warmup pass)
        ...

    def __call__(self, q, k, v, *args, **kwargs):
        """Intercepts attention calls. On sampled calls, computes
        full attention matrix and extracts tile stats."""
        step_idx = self._call_count // self.calls_per_step
        layer_idx = self._call_count % self.calls_per_step
        self._call_count += 1

        if step_idx in self.sample_steps and layer_idx in self.sample_layers:
            self._profile_call(q, k, v, step_idx, layer_idx)

        return self._orig_sdpa(q, k, v, *args, **kwargs)

    def _profile_call(self, q, k, v, step_idx, layer_idx):
        """Compute full attention matrix per head, extract tile stats.

        Processes one (head, batch) at a time to limit peak memory to ~2.4 GB.
        Scores are computed in float32 for numerically stable softmax.
        """
        B, H, N, D = q.shape
        scale = 1.0 / math.sqrt(D)
        num_tiles = math.ceil(N / self.tile_size)

        for h in range(H):
            for b in range(B):
                # One head, one batch element at a time
                qh = q[b, h, :, :].float()   # [N, D] in float32
                kh = k[b, h, :, :].float()   # [N, D] in float32
                scores = torch.matmul(qh, kh.T) * scale  # [N, N] in float32
                attn = torch.softmax(scores, dim=-1)      # [N, N] in float32

                # Extract tile-level statistics (per query row)
                self._extract_tile_stats(attn, h, b, step_idx, layer_idx, num_tiles)

                del scores, attn  # free ~2.4 GB immediately
```

**Runtime estimate:**
- 30 sampled calls × 30 heads × 2 batch elements = 1,800 (head, batch) pairs
- Per (head, batch): matmul [17776, 64] × [64, 17776] → [17776, 17776] + softmax: ~4-6 seconds on B200
- Total profiling overhead: 1,800 × 5s = ~2.5 hours
- Non-sampled calls: zero overhead (pass-through)
- Pipeline run itself (50 steps): ~3-4 minutes

**Total estimated runtime: ~2.5-3 hours**

This is significantly longer than previous experiments. For faster iteration, a smoke-test mode with fewer samples (2 steps × 3 layers = 6 calls → ~15 minutes) is provided.
