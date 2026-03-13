#!/usr/bin/env python3
"""
Evaluate CLIPSIM and CLIP-T metrics for comparing attention backends.

CLIPSIM: mean cosine similarity between CLIP image embeddings (8-frame sample) and CLIP text embedding.
CLIP-T:  mean cosine similarity between consecutive-frame CLIP image embeddings (all frames).

Reuses VBench's clip_transform and read_frames_decord_by_fps utilities.
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
import clip
from tqdm import tqdm

# Add VBench to path
VBENCH_ROOT = os.environ.get("VBENCH_ROOT", "/mnt/disk1/yiliu7/VBench")
sys.path.insert(0, VBENCH_ROOT)
from vbench.utils import clip_transform, read_frames_decord_by_fps


def compute_clipsim(clip_model, video_paths, prompts, device, num_frames=8):
    """CLIPSIM: avg cosine similarity between sampled-frame CLIP features and text CLIP feature."""
    transform = clip_transform(224)
    scores = []
    per_video = []

    for video_path, prompt in tqdm(zip(video_paths, prompts), total=len(video_paths), desc="CLIPSIM"):
        text_tok = clip.tokenize([prompt], truncate=True).to(device)
        with torch.no_grad():
            text_feat = F.normalize(clip_model.encode_text(text_tok), dim=-1)
            frames = read_frames_decord_by_fps(video_path, num_frames=num_frames, sample="middle")
            frames = transform(frames).to(device)
            img_feat = F.normalize(clip_model.encode_image(frames), dim=-1)
            sim = (img_feat @ text_feat.T).mean().item()

        scores.append(sim)
        per_video.append({"video": os.path.basename(video_path), "clipsim": round(sim, 6)})

    return float(np.mean(scores)), per_video


def compute_clipt(clip_model, video_paths, device):
    """CLIP-T: avg cosine similarity between consecutive-frame CLIP features (all frames)."""
    transform = clip_transform(224)
    scores = []
    per_video = []

    for video_path in tqdm(video_paths, desc="CLIP-T "):
        # Read ALL frames
        import decord
        decord.bridge.set_bridge("torch")
        vr = decord.VideoReader(video_path, num_threads=1)
        n = len(vr)
        frames = vr.get_batch(list(range(n)))          # (N, H, W, C) uint8
        frames = frames.permute(0, 3, 1, 2)            # (N, C, H, W)
        frames = transform(frames).to(device)

        with torch.no_grad():
            feats = F.normalize(clip_model.encode_image(frames), dim=-1)
            if feats.shape[0] < 2:
                continue
            sim = F.cosine_similarity(feats[:-1], feats[1:], dim=-1)
            mean_sim = sim.mean().item()

        scores.append(mean_sim)
        per_video.append({"video": os.path.basename(video_path), "clipt": round(mean_sim, 6)})

    return float(np.mean(scores)), per_video


def load_video_prompt_pairs(video_dir, prompt_file):
    """Load sorted mp4 paths + prompts, paired by index."""
    videos = sorted(
        [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(".mp4")]
    )
    with open(prompt_file, "r") as f:
        prompts = [line.strip() for line in f if line.strip()]
    assert len(videos) == len(prompts), (
        f"Mismatch: {len(videos)} videos vs {len(prompts)} prompts"
    )
    return videos, prompts


def main():
    parser = argparse.ArgumentParser(description="Evaluate CLIPSIM & CLIP-T")
    parser.add_argument(
        "--video_dirs", nargs="+", required=True,
        help="One or more directories containing mp4 videos (e.g. sdpa/ sage3/)"
    )
    parser.add_argument(
        "--prompt_file", type=str, required=True,
        help="Text file with one prompt per line, same order as sorted mp4 files"
    )
    parser.add_argument(
        "--output_path", type=str, default=None,
        help="Optional JSON file to save detailed results"
    )
    parser.add_argument(
        "--clipsim_frames", type=int, default=8,
        help="Number of frames to sample for CLIPSIM (default: 8)"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model, _ = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    print(f"Loaded CLIP ViT-B/32 on {device}\n")

    all_results = {}

    for video_dir in args.video_dirs:
        label = os.path.basename(os.path.normpath(video_dir))
        videos, prompts = load_video_prompt_pairs(video_dir, args.prompt_file)
        print(f"=== {label} ({len(videos)} videos) ===")

        clipsim_avg, clipsim_detail = compute_clipsim(
            clip_model, videos, prompts, device, num_frames=args.clipsim_frames
        )
        clipt_avg, clipt_detail = compute_clipt(clip_model, videos, device)

        print(f"  CLIPSIM : {clipsim_avg:.6f}")
        print(f"  CLIP-T  : {clipt_avg:.6f}\n")

        all_results[label] = {
            "clipsim": round(clipsim_avg, 6),
            "clipt": round(clipt_avg, 6),
            "clipsim_per_video": clipsim_detail,
            "clipt_per_video": clipt_detail,
        }

    # Summary table
    print("=" * 50)
    print(f"{'Config':<12} {'CLIPSIM':>10} {'CLIP-T':>10}")
    print("-" * 50)
    for label, res in all_results.items():
        print(f"{label:<12} {res['clipsim']:>10.6f} {res['clipt']:>10.6f}")
    print("=" * 50)

    if args.output_path:
        with open(args.output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nDetailed results saved to {args.output_path}")


if __name__ == "__main__":
    main()
