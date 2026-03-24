---
name: eval
description: Run 5-metric EvalCrafter video quality evaluation (CLIP scores, flow score, DOVER VQA). Use after generating video outputs to measure quality against baselines.
disable-model-invocation: true
---

Run the EvalCrafter 5-metric evaluation pipeline on generated videos.

## Environment

- **EvalCrafter repo**: `/home/yiliu7/workspace/EvalCrafter`
- **Python venv**: `/home/yiliu7/workspace/envs/crafter/bin/python3`
- **Checkpoints**: Already at `EvalCrafter/checkpoints/`
- Runtime: ~2.5 minutes for 48 videos on B200

## Metrics (5 total, higher is better)

| Metric | Category | What It Measures |
|--------|----------|-----------------|
| `clip_score` | Text-Video Alignment | How well video matches text prompt |
| `clip_temp_score` | Temporal Consistency | Frame-to-frame visual coherence |
| `flow_score` | Motion Quality | Amount/smoothness of optical flow |
| `VQA_A` | Visual Quality | Aesthetic quality (composition, appeal) |
| `VQA_T` | Visual Quality | Technical quality (noise, blur, artifacts) |

## Usage

Parse `$ARGUMENTS` for:
- First positional arg or `--dir_videos <path>` — path to the video directory (required)
- `--gpu <id>` — CUDA device ID (default: 1)

### Single run
```bash
cd /home/yiliu7/workspace/EvalCrafter
bash eval_3metrics.sh <dir_videos> [gpu_id]
```

### Multiple runs (must be sequential — shared result files prevent parallel execution)
```bash
cd /home/yiliu7/workspace/EvalCrafter
bash eval_3metrics.sh /path/to/run1/videos 3
bash eval_3metrics.sh /path/to/run2/videos 3
bash eval_3metrics.sh /path/to/run3/videos 3
```

### Compare runs after all evaluations
```bash
cd /home/yiliu7/workspace/EvalCrafter
python3 compare_runs.py
# Produces: results/comparison_table.md and results/comparison_table.csv
```

## Video Directory Requirements

- Files must be named `<N>.mp4` (0-indexed integers, no zero-padding): `0.mp4`, `1.mp4`, ...
- A `open_sora_prompts.txt` file should be discoverable by walking up from the video directory (one prompt per line, line i = video i.mp4)

## Output

Results go to `EvalCrafter/results/`:
- `<run_name>_per_video.csv` — Primary artifact: all 5 metrics per video + AVERAGE row
- `comparison_table.md` — Cross-run comparison after `compare_runs.py`

## After Completion

Present results as a table and compare against reference baselines (CogVideoX-2B, 48 Open-Sora Prompts):

| Metric | sdpa (baseline) | nvfp4 cute | nvfp4 triton | mxfp4_s1 | mxfp8_s1 |
|--------|:-:|:-:|:-:|:-:|:-:|
| clip_score | **0.2007** | 0.1992 | 0.1984 | 0.1979 | 0.2003 |
| clip_temp | **0.9970** | 0.9968 | 0.9966 | 0.9969 | 0.9970 |
| flow_score | 5.64 | 4.51 | **6.52** | 3.92 | 5.42 |
| VQA_A | **26.89** | 23.53 | 18.46 | 22.90 | 26.82 |
| VQA_T | 59.80 | 58.12 | 54.31 | 58.98 | **60.64** |

Highlight any metric where the new run beats or degrades vs the baselines.
