# How Video Becomes Tokens in CogVideoX

## Overview

In CogVideoX, the model doesn't work on raw video pixels directly — that would be far too expensive. Instead, the video is **compressed into a flat list of tokens** through two stages before the attention kernel sees it.

---

## The Compression Pipeline

```
Raw Video (49 frames, 480×720 pixels each)
         │
         ▼
   ┌─────────────────────────────────────────────┐
   │  Stage 1: VAE (Variational Autoencoder)      │
   │  • Compress time:  ÷4  (every ~4 frames → 1) │
   │  • Compress space: ÷8  (shrink H & W by 8×)  │
   └─────────────────────────────────────────────┘
         │
         ▼
   Small latent cube: 13 frames × 60 × 90
         │
         ▼
   ┌─────────────────────────────────────────────┐
   │  Stage 2: Patch Embedding                    │
   │  • Group every 2×2 spatial block → 1 token   │
   │  • Each latent frame → 30 × 45 = 1350 tokens │
   └─────────────────────────────────────────────┘
         │
         ▼
   Flat list: 13 × 1350 = 17,550 video tokens
         │
         ▼
   ┌─────────────────────────────────────────────┐
   │  Concatenate with 226 text tokens            │
   │  → seq_len = 226 + 17,550 = 17,776          │
   └─────────────────────────────────────────────┘
         │
         ▼
   Attention input: (B, num_heads, 17776, head_dim)
```

---

## The Three Factors of Video Tokens

### Factor 1: Temporal — Number of Latent Frames (`T_latent`)

**Formal name:** "latent temporal length" or "number of latent frames" (`num_latent_frames` in Diffusers code).

**Formula:**

```
T_latent = (num_frames - 1) / temporal_compression_ratio + 1
         = (num_frames - 1) / 4 + 1
```

**Why the "−1 then +1" pattern?** Think of it like fence posts:

```
Frame:  1   2   3   4   5   6   7   8   9  ...  49
        |---+---+---+---|---+---+---+---|--      --|
Group:  [     1st       ] [     2nd     ]    ...   [12th]
        + the 1st frame itself as anchor

Gaps between frames = 49 − 1 = 48
Compressed steps    = 48 / 4  = 12
Plus starting frame = 12 + 1  = 13 latent frames
```

It's the same math as: "How many fence posts for a 48-meter fence with a post every 4 meters?" → 12 sections + 1 post at the start = **13 posts**.

| `num_frames` | Calculation | `T_latent` |
|---|---|---|
| 49 (CogVideoX-2b default) | (49−1)/4 + 1 | **13** |
| 81 (CogVideoX1.5-5B default) | (81−1)/4 + 1 | **21** |

> **Why 49 frames?** Not a magic number — it's a practical default (~6s at 8 FPS) that fits the 2B model's memory budget. `num_frames` is configurable but must satisfy `(num_frames − 1) % 4 == 0` due to the VAE's temporal compression kernel. CogVideoX1.5-5B uses 81 frames (~10s) as its default.

### Factor 2: Spatial — Patches per Latent Frame (`H_patch × W_patch`)

Each latent frame is divided into a grid of tokens. Two compressions stack:

| Stage | What it does | Factor |
|---|---|---|
| VAE spatial | Shrinks each dimension by 8× | ÷8 |
| Patch embedding | Groups 2×2 blocks into 1 token | ÷2 |
| **Combined** | **Total spatial compression** | **÷16** |

```
H_patch = height / 16
W_patch = width  / 16
```

For the default 480×720 resolution:

```
H_patch = 480 / 16 = 30
W_patch = 720 / 16 = 45
Spatial tokens per latent frame = 30 × 45 = 1,350
```

### Factor 3: Multiply Them Together

```
video_tokens = T_latent × H_patch × W_patch
             = 13       × 30      × 45
             = 17,550
```

---

## Full Sequence Length Formula

The attention kernel operates on text + video tokens concatenated:

```
seq_len = S_text + T_latent × H_patch × W_patch

        = max_sequence_length + ((num_frames - 1) / 4 + 1) × (H / 16) × (W / 16)
```

### Concrete Example (CogVideoX-2b defaults)

| Component | Value |
|---|---|
| Text tokens (`S_text`) | 226 (always fixed, regardless of prompt length) |
| Latent frames (`T_latent`) | 13 |
| Spatial tokens per frame | 30 × 45 = 1,350 |
| **Video tokens** | 13 × 1,350 = **17,550** |
| **Total `seq_len`** | 226 + 17,550 = **17,776** |

---

## What Affects Token Count?

| Config | Effect on `seq_len` | Example |
|---|---|---|
| **Prompt length** | **No effect** — always padded to 226 | "A cat" and "A long description..." both → 226 |
| **`num_frames`** | **Linear** — each +4 frames adds `spatial_tokens` | +4 frames → +1,350 tokens |
| **Resolution (H×W)** | **Quadratic** — doubling H and W → ~4× tokens | 480×720 → 960×1440 = ~4× |
| **`guidance_scale`** | No effect on `seq_len` (doubles batch dim instead) | — |

**Resolution is the biggest lever** — doubling it roughly quadruples the work for the attention kernel.

---

## Naming Conventions

| Term | Where it appears |
|---|---|
| Latent temporal length / `num_latent_frames` | HuggingFace Diffusers code (`prepare_latents()`) |
| `temporal_compression_ratio` (= 4) | CogVideoX model config |
| `vae_temporal_compression` | Pipeline code |
| `vae_spatial_compression` (= 8) | Pipeline code |
| `patch_size` (= 2) | `CogVideoXPatchEmbed` |
| `max_sequence_length` (= 226) | Tokenizer / pipeline config |

---

## Code Path Reference

1. **Latent shape**: `CogVideoXPipeline.prepare_latents()` creates latents of shape `(B, T_latent, C, H_latent, W_latent)`
2. **Patchification**: `CogVideoXPatchEmbed.forward()` applies Conv2d (patch_size=2) and flattens → `(B, T_latent × H_patch × W_patch, dim)`, then concatenates text embeddings → `(B, S_text + video_tokens, dim)`
3. **Attention**: `CogVideoXAttnProcessor2_0.__call__()` concatenates `encoder_hidden_states` (text) with `hidden_states` (video), projects to q/k/v, reshapes to `(B, heads, seq_len, head_dim)`, and calls `F.scaled_dot_product_attention`
