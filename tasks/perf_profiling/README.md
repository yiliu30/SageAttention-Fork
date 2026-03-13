Task Description

Goal:
1. Understand the proportion of the attention kernel in the whole pipeline.
2. We may want to compare different attention kernels (sdpa/sage3), so the measurement/profiling approach should be programmable.


Steps:
1. Investigate the existing profiling methods, like torch profiler, nsight systems, NVTX, etc.
2. Read the codebase and suggest how to insert profiling hooks.

Source:
- Codebase:
    - Sage v3: /mnt/disk1/yiliu7/SageAttention-Fork/sageattention3_blackwell
- Ref, torch profiling:
    ```bash
     python cogvideox_infer.py --model cogvideox-2b  --attention_type sage3 -d -p
    ```
Notes:
1. It would be better to provide a simple example to demonstrate the profiling usage.
2. Please focus compare the sdpa or sage3 time proportion, not the inner kernel time of sage3

---

## Summary

### CogVideoX Pipeline Components

| Component | Class | Role |
|---|---|---|
| tokenizer | T5Tokenizer | Converts text prompt → token IDs |
| text_encoder | T5EncoderModel | Encodes token IDs → text embeddings (conditioning signal) |
| transformer | CogVideoXTransformer3DModel | Denoising backbone, runs iteratively for N steps. All `F.scaled_dot_product_attention` calls happen here. **Dominates runtime.** |
| vae | AutoencoderKLCogVideoX | Decodes final latent → pixel-space video frames |
| scheduler | CogVideoXDDIMScheduler | Noise schedule / denoising step logic (lightweight, CPU-side) |

### Profiling Approach

We use **async CUDA event timing** to measure GPU-side attention time without serializing the pipeline:

- `AttentionProportionProfiler` wraps `F.scaled_dot_product_attention` with `torch.cuda.Event` pairs
- Events are recorded inline (no `synchronize()` per call); synced once at the end
- Both attention time and total pipeline time are **device (GPU) time**, not host (CPU) wall-clock
- CPU offload is disabled during proportion measurement to avoid inflating total time with CPU↔GPU transfers

### How the attention is monkey-patched

In `cogvideox_infer.py`, the attention backend is swapped at the `F.scaled_dot_product_attention` level:
- `sdpa`: default PyTorch SDPA (no override, uses cuDNN/FlashAttention backend)
- `sage3`: `sageattn3_blackwell` from `sageattention3_blackwell/sageattn3/api.py`
- `sage`: `sageattn` from `sageattention/core.py`

The profiler wraps whichever function is assigned, so it works identically for any attention backend.

### Usage

```bash
# Compare attention proportion across backends
python cogvideox_infer.py --model cogvideox-2b --attention_type sdpa  --proportion -s
python cogvideox_infer.py --model cogvideox-2b --attention_type sage3 --proportion -s
```

Example output:
```
============================================================
Attention Proportion Report (sage3)
============================================================
  Total pipeline GPU time :     2200.0 ms
  Attention kernel time   :      400.0 ms
  Non-attention time      :     1800.0 ms
  Attention proportion    :       18.2 %
  Attention calls         :       1500
  Avg per call            :        0.27 ms
============================================================
```



### Key design decisions
1. **GPU time, not CPU time** — `torch.cuda.Event.elapsed_time()` measures device-side elapsed time, immune to Python/CPU overhead.
2. **Async event recording** — no per-call `synchronize()`, so pipeline throughput is unaffected.
3. **Separate from `--profile`** — existing torch profiler code path (`-p`) is untouched; `--proportion` is a new independent branch.

---

## How Prompt Length & Pipeline Configs Impact Q/K/V Shape (`seq_len`)

In CogVideoX, attention operates on **text tokens + video tokens concatenated** along the sequence dimension. The q/k/v shape entering `F.scaled_dot_product_attention` is:

```
(B, num_heads, seq_len, head_dim)
```

where:

```
seq_len = S_text + T_latent × H_patch × W_patch
```

### Text tokens (`S_text`)

- **Always fixed** at `max_sequence_length` (default **226** for CogVideoX-2b), regardless of actual prompt length.
- The T5 tokenizer pads to `max_length=max_sequence_length` with `padding="max_length"`.
- **Prompt length does NOT affect `seq_len`** — it is always padded/truncated to 226.

### Video tokens — three factors

#### Temporal: `T_latent`

```
T_latent = (num_frames - 1) / vae_temporal_compression + 1
         = (num_frames - 1) / 4 + 1
```

| `num_frames` | `T_latent` |
|---|---|
| 49 (CogVideoX-2b default) | 13 |
| 81 (CogVideoX1.5-5B default) | 21 |

#### Spatial: `H_patch × W_patch`

```
H_patch = height / (vae_spatial_compression × patch_size) = height / (8 × 2) = height / 16
W_patch = width  / (vae_spatial_compression × patch_size) = width  / (8 × 2) = width  / 16
```

For the default 480×720 resolution:
- `H_patch = 480 / 16 = 30`
- `W_patch = 720 / 16 = 45`
- Spatial tokens = `30 × 45 = 1350`

### Full formula

```
seq_len = max_seq_len + ((num_frames - 1) / 4 + 1) × (H / 16) × (W / 16)
```

### Concrete example (CogVideoX-2b defaults)

| Component | Value |
|---|---|
| Text tokens | 226 |
| `T_latent` | (49−1)/4 + 1 = **13** |
| `H_patch × W_patch` | 30 × 45 = **1350** |
| **Video tokens** | 13 × 1350 = **17,550** |
| **Total `seq_len`** | 226 + 17,550 = **17,776** |

### Impact summary

| Config | Effect on `seq_len` |
|---|---|
| **Prompt length** | **No effect** — always padded to `max_sequence_length` (226) |
| **`num_frames`** | **Linear** — each +4 frames → +1 latent frame × spatial_tokens (e.g. +1350) |
| **Resolution (H×W)** | **Quadratic** — doubling resolution → ~4× spatial tokens |
| **`max_sequence_length`** | Direct additive (but typically fixed per model checkpoint) |

### Notes on `guidance_scale`

With `guidance_scale > 1` (classifier-free guidance), the **batch dimension doubles** (negative + positive prompts are concatenated along dim 0), but `seq_len` stays the same.

### Code path reference

1. **Latent shape**: `CogVideoXPipeline.prepare_latents()` creates latents of shape `(B, T_latent, C, H_latent, W_latent)`
2. **Patchification**: `CogVideoXPatchEmbed.forward()` applies Conv2d (patch_size=2) and flattens → `(B, T_latent × H_patch × W_patch, dim)`, then concatenates text embeddings → `(B, S_text + video_tokens, dim)`
3. **Attention**: `CogVideoXAttnProcessor2_0.__call__()` concatenates `encoder_hidden_states` (text) with `hidden_states` (video), projects to q/k/v, reshapes to `(B, heads, seq_len, head_dim)`, and calls `F.scaled_dot_product_attention`

---
