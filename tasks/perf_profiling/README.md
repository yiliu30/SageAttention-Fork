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
