---
name: verify
description: Quick E2E regression test comparing sage3_standalone (triton) vs sage3 (CUTE kernel) via PSNR and cosine similarity. Only available on RTX 5090 (SM120) — the CUTE kernel reference requires this GPU. Use when you want to check if a change broke end-to-end accuracy.
---

**Prerequisites**: This test requires an **RTX 5090 (SM120)** GPU. The sage3 CUTE kernel reference only runs on this architecture. Before running, verify the GPU:

```bash
nvidia-smi --query-gpu=name --format=csv,noheader | head -1
```

If the output does not contain "5090", abort and inform the user:
> "/verify is only available on RTX 5090. The sage3 CUTE kernel (used as the reference) requires SM120. Current GPU: <detected GPU>."

If confirmed on 5090, run the quick E2E regression test:

```bash
cd example
bash test_quick_e2e.sh
```

This compares:
- **sage3_standalone** (triton kernel) — the test subject
- **sage3** (CUTE kernel) — the reference

Metrics:
- **PSNR** threshold: 20 dB (env: `PSNR_THRESHOLD`)
- **Cosine similarity** threshold: 0.95 (env: `COS_THRESHOLD`)
- Generates 1 frame by default (env: `NUM_FRAMES`)

If the user provides `$ARGUMENTS`, parse them for optional overrides:
- `--psnr <value>` → set `PSNR_THRESHOLD`
- `--cos <value>` → set `COS_THRESHOLD`
- `--frames <value>` → set `NUM_FRAMES`

After the test completes, summarize:
1. PSNR value and whether it passed
2. Cosine similarity value and whether it passed
3. Overall verdict (PASS only if both pass)

If the test fails, suggest checking recent changes to `standalone/sageattention3_standalone.py` or quantization parameters.
