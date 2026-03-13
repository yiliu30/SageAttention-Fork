Task Desc

Goal:
Reproduce the SageAttention v3 accuracy results on video generation.

Steps:
1. Read the paper and summarize the accuracy-related tests.
2. Read the paper and codebase to check if there are any existing accuracy test scripts we can reuse.
3. Define the test scope and summarize the steps.

Source:
- paper:
    - /mnt/disk1/yiliu7/SageAttention-Fork/papers/sageattention/SageAttention3-2505.11594.pdf
    - /mnt/disk1/yiliu7/SageAttention-Fork/papers/sageattention/sage_v3
    - /mnt/disk1/yiliu7/SageAttention-Fork/papers/sageattention/paper_summaries.md
- codebase:
    - All Sage (v1, v2, v3): /mnt/disk1/yiliu7/SageAttention-Fork
    - Sage v3: /mnt/disk1/yiliu7/SageAttention-Fork/sageattention3_blackwell

Notes:
1. No coding required for now.

---

## Step 1: Paper Accuracy Tests Summary (Video Generation Only)

### Models & Configurations
- **Models:** CogVideoX (2B), HunyuanVideo, Mochi
- **Attention configs compared:**
  - Full-Precision (16-bit) — baseline
  - SageAttention2 (8-bit)
  - SageAttention3 (4-bit) — our reproduction target

### Dataset
- **Prompts:** open-sora prompt sets (paper Appendix C.3)
- NOT the 3-prompt `testing_prompts.txt` in `example/videos/`

### 5 Video Quality Metrics
| Metric | Direction | Measures | Likely Tool |
|--------|-----------|----------|-------------|
| CLIPSIM | ↑ | Text-video semantic alignment | EvalCrafter / CLIP |
| CLIP-T | ↑ | Temporal text-video consistency | EvalCrafter / CLIP |
| VQA-a | ↑ | Aesthetic quality | DOVER / Q-Align |
| VQA-t | ↑ | Technical quality | DOVER / Q-Align |
| FScore | ↑ | Temporal/flow consistency | RAFT optical flow |

### Paper Reference Numbers (Video Models)
| Model | Attention | CLIPSIM | CLIP-T | VQA-a | VQA-t | FScore |
|-------|-----------|---------|--------|-------|-------|--------|
| CogVideoX | FP16 | 0.1865 | 0.9968 | 70.476 | 69.875 | 4.780 |
| CogVideoX | Sage2-8b | 0.1880 | 0.9969 | 69.414 | 70.750 | 4.534 |
| CogVideoX | **Sage3-4b** | 0.1881 | 0.9969 | 69.860 | 70.364 | 4.035 |
| HunyuanVideo | FP16 | 0.1838 | 0.9993 | 68.998 | 78.891 | 1.479 |
| HunyuanVideo | Sage2-8b | 0.1836 | 0.9993 | 69.497 | 77.019 | 1.474 |
| HunyuanVideo | **Sage3-4b** | 0.1866 | 0.9993 | 70.552 | 75.440 | 1.232 |
| Mochi | FP16 | 0.1828 | 0.9990 | 61.984 | 61.000 | 1.804 |
| Mochi | Sage2-8b | 0.1819 | 0.9990 | 61.009 | 60.373 | 1.754 |
| Mochi | **Sage3-4b** | 0.1800 | 0.9993 | 61.863 | 59.429 | 1.649 |

---

## Step 2: Existing Scripts Assessment

### Inference Scripts
| Script | Status | Sage3? | Reusable? | Notes |
|--------|--------|--------|-----------|-------|
| `example/cogvideox_infer.py` | Working | **Yes** (`--attention_type sage3`) | **Yes** | Local model at `models/zai-org/CogVideoX-2b`. 49 frames, 50 steps. |
| `example/mochi_infer.py` | Working | **No** | Needs small fix | Add `sage3` to choices + import. Uses custom `modify_mochi` patching. |
| `example/hunyuan_infer.py` | **DISABLED** | No | **No** | `attention_mask` incompatible with sageattn. Paper used official HunyuanVideo repo, not diffusers. |
| `sageattention3_blackwell/examples/sageattn3_demo.py` | Working | Yes | No | Kernel-level smoke test only (random tensors), not end-to-end. |

### What's Missing
1. **Evaluation scripts** — no code to compute the 5 metrics
2. **Open-sora prompt set** — `testing_prompts.txt` has only 3 prompts; paper uses open-sora (hundreds)
3. **Metric tooling** — EvalCrafter, VBench, DOVER/Q-Align need to be installed and integrated
4. **Model weights** — only CogVideoX-2b is downloaded locally; Mochi and HunyuanVideo need to be fetched


## GitHub Issues/PRs Analysis (thu-ml/SageAttention)

Searched all open/closed issues and PRs on the official repo.
Key finding: Nobody has publicly reproduced the end-to-end accuracy metrics

---



### Open Questions
- [ ] What exact open-sora prompt set was used? How many prompts? (Open-Sora repo has `assets/texts/sora.csv` — need to check size)
- [ ] VQA-a/VQA-t tool confirmed: DOVER (`\cite{wu2023exploring}` = DOVER paper). DOVER gives aesthetic + technical scores natively.
- [ ] Do we use EvalCrafter's bundled scripts or implement metrics independently? (Recommend: independent for Phase 2-3, EvalCrafter for Phase 4)
- [ ] FScore largest gap in paper (4.78 → 4.04). Is this acceptable or a concern?
- [ ] Is `SageAttention2 (8bit)` the same as `--attention_type sage` in the scripts? (Likely yes — sage = sageattention v2 INT8)

---

## Step 4: Metric Evaluation Time Analysis

### Scope: CogVideoX-2b, BF16 SDPA (baseline) vs Sage3 (4-bit)

### Per-Metric Time Estimates (per video, single GPU)

| # | Metric | Tool/Repo | Model Size | Est. Time/Video | Setup Complexity | Notes |
|---|--------|-----------|------------|-----------------|------------------|-------|
| 1 | CLIPSIM | `openai/CLIP` (ViT-B/32) | 150M | **~2s** | Low | Extract frames → CLIP embed → cosine sim with prompt. Pure forward pass. |
| 2 | CLIP-T | `openai/CLIP` (ViT-B/32) | 150M | **~2s** | Low | CLIP embed consecutive frames → mean pairwise cosine sim. Same CLIP model as CLIPSIM. |
| 3 | VQA-a | `DOVER` (aesthetic branch) | ~56M (Swin-T backbone) | **~4s** | Medium | DOVER splits quality into aesthetic + technical. 3.6s/video per their README. |
| 4 | VQA-t | `DOVER` (technical branch) | ~56M (Swin-T backbone) | **~4s** | Medium | Same DOVER model, different head. Runs together with VQA-a. |
| 5 | FScore | `RAFT` optical flow | ~5M | **~10-15s** | Medium-High | 48 frame-pairs for 49-frame video. ~0.2-0.3s per pair on GPU. Needs RAFT weights + CUDA extension. |

**Combined VQA-a + VQA-t note:** DOVER evaluates both in a single pass (~4s total, not 4s each).

### Inference Time Estimate (CogVideoX-2b, per video)

| Config | Est. Time/Video | Notes |
|--------|-----------------|-------|
| BF16 SDPA | ~3-5 min | 49 frames, 50 steps, model CPU offload |
| Sage3 (4-bit) | ~2-3 min | Expected ~1.5-2x speedup on attention |

### Total Time Summary (N = number of prompts)

| Task | Per-Video | 3 prompts (smoke) | ~100 prompts (full) |
|------|-----------|---------|----------|
| Video Gen (×2 configs) | ~4 min avg | **~24 min** | **~13 hrs** |
| CLIPSIM (×2 configs) | ~2s | **<1 min** | **~7 min** |
| CLIP-T (×2 configs) | ~2s | **<1 min** | **~7 min** |
| VQA-a+t (×2 configs) | ~4s | **<1 min** | **~13 min** |
| FScore (×2 configs) | ~12s | **~1 min** | **~40 min** |
| **Total** | | **~27 min** | **~15 hrs** |

**Bottleneck:** Video generation dominates (>95% of total time). All 5 metrics combined < 1 hour even for 100 prompts.

---

## Step 5: Test Plan (Ordered by Time, Short → Long)

### Config: CogVideoX-2b, 2 attention types: `sdpa` (BF16 baseline), `sage3` (4-bit)

---

### Phase 0: Environment & Smoke Test (~5 min)
- [ ] 0.1 Verify sage3 kernel: `python sageattn3_demo.py` (random tensor smoke test)
- [ ] 0.2 Verify CogVideoX inference: `python cogvideox_infer.py --model cogvideox-2b --attention_type sdpa --smoke`
- [ ] 0.3 Verify CogVideoX + sage3: `python cogvideox_infer.py --model cogvideox-2b --attention_type sage3 --smoke`
- **Output:** 2 smoke videos (1 per config), confirms pipeline works

### Phase 1: Mini Eval — 3 Prompts (~30 min)
- [ ] 1.1 Generate videos: 3 prompts × 2 configs = 6 videos (~24 min)
- [ ] 1.2 CLIPSIM eval on 6 videos (<1 min)
- [ ] 1.3 CLIP-T eval on 6 videos (<1 min)
- [ ] 1.4 DOVER (VQA-a + VQA-t) eval on 6 videos (<1 min)
- [ ] 1.5 FScore (RAFT) eval on 6 videos (~1 min)
- [ ] 1.6 Compare sdpa vs sage3 scores (sanity check — are they close?)
- **Output:** Table with 5 metrics × 2 configs. Verifies full pipeline E2E.

### Phase 2: Full Eval — Open-Sora Prompt Set (~15 hrs)
- [ ] 2.1 Source and prepare open-sora prompts (or use a suitable subset)
- [ ] 2.2 Generate videos: N prompts × 2 configs
- [ ] 2.3 Run all 5 metrics on all videos
- [ ] 2.4 Compare against paper Table numbers
- **Output:** Final reproduction numbers

---

### Eval Metric Task Order (fastest first, within each phase)
1. **CLIPSIM** — fastest, just CLIP forward + cosine sim
2. **CLIP-T** — same CLIP model, frame-pair cosine sim
3. **VQA-a + VQA-t** — single DOVER pass, ~4s/video
4. **FScore** — RAFT optical flow, ~12s/video (most complex setup)

### Prerequisites Checklist
- [ ] Python env with PyTorch ≥2.8, CUDA ≥12.8
- [ ] SageAttention3 compiled (`.so` files in `sageattention3_blackwell/`)
- [ ] `pip install clip-model` (or `openai-clip`)
- [ ] DOVER installed + weights downloaded (`DOVER.pth`)
- [ ] RAFT installed + weights downloaded (`raft-things.pth`)
- [ ] CogVideoX-2b weights at `models/zai-org/CogVideoX-2b` (already present)